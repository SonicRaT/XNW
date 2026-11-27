//! Which title memory the title wrote since a host copy took it. The runtime
//! reserves title GPU memory with write watching (`xnw-hle` xmem), so the
//! kernel keeps a dirty bit per 4 KiB page; asking for a range's written
//! pages also clears them. Host copies of the same memory can overlap
//! (nested buffers, textures in buffers), so what one asks is kept here for
//! the others: a journal of each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! The paired reservation (xmem, Series S: CPU view `0x1_0000_0000`, GPU
//! view `0x5_0000_0000`, one memory seen twice): the title writes through
//! the CPU view and places GPU resources in the GPU view. Protection is per
//! view, so guards and the journal work on the CPU-view address
//! ([`use_pair`]); answers are given back at the address asked.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).
//!
//! Ported from VIctory-base (branch `xnw/base` @
//! 5d0516e = VIctory d2a4bae + an upstream change) `crates/victory-gpu/src/watch.rs`
//! 1-379, GPL-3.0-or-later. Changed: `windows-sys` -> `xnw_core::win`; the
//! diagnostics counters dropped; the pair mapping added; the tests' temp
//! file is a plain one (no `tempfile`).
//!
//! [`Sweep`] (one asking per page per submission) is ported from VIctory
//! an upstream branch (schmiticle/VIctory `gpu/write-watch-once-per-submission`
//! @ c326bfa, watch.rs `Sweep`, `gaps`, `sweep_in` and two tests,
//! GPL-3.0-or-later). Changed: what a sweep asked is the sweeping thread's
//! own (a thread local, not a thread id in the journal), so a sweep on one
//! thread never ends or clears another's (parallel tests).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use xnw_core::win;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// The paired reservation: CPU view, GPU view, length.
static PAIR: Mutex<Option<(usize, usize, usize)>> = Mutex::new(None);

/// The runtime found the paired reservation: `[gpu, gpu + len)` shows the
/// pages of `[cpu, cpu + len)` (the GPU view lies above the CPU one).
pub fn use_pair(cpu: usize, gpu: usize, len: usize) {
    if gpu > cpu {
        *PAIR.lock().unwrap_or_else(|e| e.into_inner()) = Some((cpu, gpu, len));
    }
}

/// How far `[start, end)` lies above its CPU-view address: the GPU view's
/// distance from the CPU view if it lies in the GPU view, else 0.
fn alias(start: usize, end: usize) -> usize {
    match *PAIR.lock().unwrap_or_else(|e| e.into_inner()) {
        Some((cpu, gpu, len)) if start >= gpu && end <= gpu + len => gpu - cpu,
        _ => 0,
    }
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal { asked: 0, written: BTreeMap::new(), found: Vec::new() });

thread_local! {
    /// While this thread is in a [`Sweep`]: the page ranges it asked the
    /// kernel about in it (sorted, disjoint, end exclusive).
    static SWEPT: std::cell::RefCell<Option<Vec<(usize, usize)>>> = const { std::cell::RefCell::new(None) };
    /// While this thread is in a [`Sweep`]: the hot blocks already taken as
    /// written in it (once per submission, not once per range asked).
    static HOT_SEEN: std::cell::RefCell<Option<BTreeSet<usize>>> = const { std::cell::RefCell::new(None) };
    /// The asking under way wants exact answers: no block is taken as hot.
    static EXACT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`writes`] with guards, exact: blocks written every frame are guarded
/// again rather than taken as written at every asking (the descriptor
/// heap's scan, which examines every slot of a block reported written:
/// 3.6 million slots a frame with hot heap blocks, a test run).
pub(crate) fn writes_exact(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    EXACT.with(|e| e.set(true));
    let r = writes(start, end, true);
    EXACT.with(|e| e.set(false));
    r
}

/// While alive, the calling thread prepares one submission: the title's
/// writes it must see were all made before the submission began (but for
/// steps the preparation runs itself, [`Sweep::title_written`]), so a page
/// the kernel was asked about once in the sweep need not be asked about
/// again. Nested buffers in the world (several views into one 174 MB block)
/// otherwise asked about the same pages ~20 times a frame, 120-180 ms of a
/// frame on an RX 9070 XT (VIctory).
pub(crate) struct Sweep(());

impl Sweep {
    pub(crate) fn begin() -> Self {
        SWEPT.with_borrow_mut(|s| *s = Some(Vec::new()));
        HOT_SEEN.with_borrow_mut(|s| *s = Some(BTreeSet::new()));
        Sweep(())
    }

    /// The thread in a sweep is about to write title memory itself (a step
    /// run on the CPU between lists): what it asked about before must be
    /// asked about again.
    pub(crate) fn title_written() {
        SWEPT.with_borrow_mut(|s| {
            if let Some(s) = s {
                s.clear();
            }
        });
        HOT_SEEN.with_borrow_mut(|s| {
            if let Some(s) = s {
                s.clear();
            }
        });
    }

    /// As [`Sweep::title_written`], for a write of `[at, at + len)` only:
    /// just its pages are asked about again.
    pub(crate) fn title_written_range(at: usize, len: usize) {
        let end = at.saturating_add(len);
        let d = alias(at, end);
        let (first, last) = ((at - d) / PAGE, (end - d).div_ceil(PAGE));
        SWEPT.with_borrow_mut(|s| {
            if let Some(s) = s {
                let mut out = Vec::with_capacity(s.len() + 1);
                for &(a, e) in s.iter() {
                    if e <= first || a >= last {
                        out.push((a, e));
                        continue;
                    }
                    if a < first {
                        out.push((a, first));
                    }
                    if e > last {
                        out.push((last, e));
                    }
                }
                *s = out;
            }
        });
        HOT_SEEN.with_borrow_mut(|s| {
            if let Some(s) = s {
                s.retain(|&b| b < first || b >= last);
            }
        });
    }
}

impl Drop for Sweep {
    fn drop(&mut self) {
        SWEPT.with_borrow_mut(|s| *s = None);
        HOT_SEEN.with_borrow_mut(|s| *s = None);
    }
}

/// The parts of `[first, last)` not in `swept` (sorted, disjoint).
fn gaps(swept: &[(usize, usize)], first: usize, last: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut at = first;
    for &(a, e) in swept {
        if e <= at {
            continue;
        }
        if a >= last {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(e);
        if at >= last {
            break;
        }
    }
    if at < last {
        out.push((at, last));
    }
    out
}

/// Adds `[first, last)` to `swept`, keeping it sorted and disjoint.
fn sweep_in(swept: &mut Vec<(usize, usize)>, first: usize, last: usize) {
    let i = swept.partition_point(|&(_, e)| e < first);
    let mut j = i;
    let (mut a, mut e) = (first, last);
    while j < swept.len() && swept[j].0 <= e {
        a = a.min(swept[j].0);
        e = e.max(swept[j].1);
        j += 1;
    }
    swept.splice(i..j, [(a, e)]);
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded). An address in the
/// GPU view of the paired reservation is asked at its CPU view.
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let d = alias(start, end);
    let (now, pages) = writes_at(start - d, end - d, guarded)?;
    Some((now, pages.into_iter().map(|(p, w)| (p + d / PAGE, w)).collect()))
}

/// The number of the last asking (0: none yet). An asker that notes it
/// before bringing a range up to date can later tell, with [`unchanged`],
/// whether the range was written since.
pub(crate) fn asked() -> u64 {
    JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).asked
}

/// For each `(start, end, since)`: true if no page of `[start, end)` can
/// have been written after asking `since` (non-zero), known without asking
/// the kernel. Guarded view memory: every block still clean and none
/// journalled after `since` (a write opens a block, and only a later
/// asking guards it again, journalling it). Write-watched memory: the
/// calling thread's sweep has asked about all of it and none is journalled
/// after `since`. False otherwise (ask with [`writes`]). One look at the
/// journal and the guards for all ranges (2,544 bindless reads per
/// submission each asked the kernel twice, 50,888 a frame).
pub(crate) fn unchanged(ranges: &[(usize, usize, u64)]) -> Vec<Unchanged> {
    let pair = *PAIR.lock().unwrap_or_else(|e| e.into_inner());
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let g = guards();
    // A range of one block (most bounces): its clean state and last write
    // looked up once per call however many ranges lie in it (155,000
    // bounce checks a frame, a test run).
    let mut one: std::collections::HashMap<usize, (bool, u64)> = std::collections::HashMap::new();
    SWEPT.with_borrow(|swept| {
        ranges
            .iter()
            .map(|&(start, end, since)| {
                if since == 0 || end <= start {
                    return Unchanged::Never;
                }
                let d = match pair {
                    Some((cpu, gpu, len)) if start >= gpu && end <= gpu + len => gpu - cpu,
                    _ => 0,
                };
                let (first, last) = ((start - d) / PAGE, (end - d).div_ceil(PAGE));
                if last == first + 1 {
                    let &mut (clean, written) = one.entry(first).or_insert_with(|| (g.clean.contains(&first), j.written.get(&first).copied().unwrap_or(0)));
                    let known = clean || swept.as_ref().is_some_and(|s| gaps(s, first, last).is_empty());
                    return if !known {
                        Unchanged::Open
                    } else if written <= since {
                        Unchanged::Yes
                    } else {
                        Unchanged::Newer
                    };
                }
                let known = (first..last).all(|b| g.clean.contains(&b))
                    || swept.as_ref().is_some_and(|s| gaps(s, first, last).is_empty());
                if !known {
                    Unchanged::Open
                } else if j.written.range(first..last).all(|(_, &w)| w <= since) {
                    Unchanged::Yes
                } else {
                    Unchanged::Newer
                }
            })
            .collect()
    })
}

/// [`unchanged`]'s answer for one range.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Unchanged {
    /// Unwritten since: nothing to do.
    Yes,
    /// Never brought up to date (no asking noted).
    Never,
    /// A block of it is open (written since it was guarded, or never
    /// guarded) and not asked about in this sweep.
    Open,
    /// Clean now, but a block was found written after the asking noted.
    Newer,
}

/// View regions (start, end) found by [`query`], until the runtime remaps
/// memory there: views are never write-watched, so neither the kernel's
/// write watch nor `VirtualQuery` is asked about them again (both were
/// asked at every refresh of view memory, two system calls each).
static VIEWS: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

/// Whether `at` lies in a view region found before.
fn known_view(at: usize) -> bool {
    VIEWS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .range(..=at)
        .next_back()
        .is_some_and(|(_, &end)| at < end)
}

/// Forgets view regions overlapping `[at, end)`.
fn forget_views(at: usize, end: usize) {
    let mut v = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let gone: Vec<usize> = v.range(..end).filter(|&(_, &e)| e > at).map(|(&s, _)| s).collect();
    for s in gone {
        v.remove(&s);
    }
}

/// Whether `address` lies in the paired reservation (either view).
pub(crate) fn paired(address: usize) -> bool {
    in_pair(address, address.saturating_add(1))
}

/// The GPU-view address of `address` in the CPU view of the paired
/// reservation (none elsewhere).
pub(crate) fn gpu_alias(address: usize) -> Option<usize> {
    match *PAIR.lock().unwrap_or_else(|e| e.into_inner()) {
        Some((cpu, gpu, len)) if (cpu..cpu + len).contains(&address) => Some(address + (gpu - cpu)),
        _ => None,
    }
}

/// How far `[start, end)` lies above its CPU-view address (0 outside the
/// GPU view of the paired reservation).
pub(crate) fn alias_of(start: usize, end: usize) -> usize {
    alias(start, end)
}
/// Whether `[start, end)` lies in the paired reservation (either view):
/// section views, never write-watched, so the kernel is not asked.
fn in_pair(start: usize, end: usize) -> bool {
    PAIR.lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some_and(|(cpu, gpu, len)| start >= cpu && end <= cpu + len || start >= gpu && end <= gpu + len)
}

fn writes_at(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // A range starting in a view: the kernel answers for whole allocations
    // or not at all, so it would fail.
    let pair = in_pair(start, end) || known_view(start);
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // In a sweep, only what this thread has not asked about yet in it.
    // The paired reservation is never write-watched: its asking failed
    // every time (a system call per range) before it was guarded.
    let asks = SWEPT.with_borrow(|s| match s {
        _ if pair => Vec::new(),
        Some(s) => gaps(s, first, last),
        None => vec![(first, last)],
    });
    let mut found: Vec<usize> = Vec::new();
    let mut failed = pair;
    for &(a, e) in &asks {
        let mut count = (e - a) * PAGE / KERNEL_PAGE;
        j.found.resize(count.max(j.found.len()), 0);
        let mut granularity = 0;
        // SAFETY: the answer buffer holds `count` entries.
        let asking = std::time::Instant::now();
        failed = unsafe {
            win::GetWriteWatch(win::WRITE_WATCH_FLAG_RESET, (a * PAGE) as *const _, (e - a) * PAGE, j.found.as_mut_ptr().cast(), &mut count, &mut granularity)
        } != 0;
        xnw_core::syscalls::count(xnw_core::syscalls::Call::WriteWatch, asking);
        if failed {
            break;
        }
        found.extend(j.found[..count].iter().map(|&at| at / PAGE));
    }
    let guarded = if failed {
        // Not write-watched memory (the kernel answers for a whole
        // allocation or not at all), so none of the range goes in the sweep.
        // Gaps asked before the failing one were reset by the kernel: what
        // they found is journalled, not dropped as upstream does.
        if !found.is_empty() {
            j.asked += 1;
            let now = j.asked;
            for page in found.drain(..) {
                j.written.insert(page, now);
            }
        }
        if !guarded {
            return None;
        }
        guard(first * PAGE, last * PAGE)?
    } else {
        SWEPT.with_borrow_mut(|s| {
            if let Some(s) = s {
                for &(a, e) in &asks {
                    sweep_in(s, a, e);
                }
            }
        });
        Vec::new()
    };
    j.asked += 1;
    let now = j.asked;
    for page in found.into_iter().chain(guarded) {
        j.written.insert(page, now);
    }
    Some((now, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks trapped against all access, with their tag.
    trapped: BTreeMap<usize, u32>,
    /// Of the blocks whose trap went, those a callback is filling now, with
    /// the filling thread.
    filling: BTreeMap<usize, u32>,
    /// Blocks the title writes every frame, left unguarded since a frame
    /// ([`hot`]); and per block, the frames in a row it was found written
    /// and the last such frame.
    hot: BTreeMap<usize, u64>,
    streak: BTreeMap<usize, (u32, u64)>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards { protected: BTreeSet::new(), clean: BTreeSet::new(), trapped: BTreeMap::new(), filling: BTreeMap::new(), hot: BTreeMap::new(), streak: BTreeMap::new() });

// ---- access traps ------------------------------------------------------
//
// Besides the write guard (read-only, opened at the first write), a block can
// be trapped: no access at all, so the first read or write of it faults into
// our handler, which notes who (tag, read or write, thread, instruction)
// and gives the access back (read-only again if the block is also guarded).
// Traps and guards are one state per block under one lock. Everything of
// ours that reads or writes title memory through the kernel (which fails on
// a no-access page instead of faulting) lifts traps first: [`writing`] for
// writes (DirectStorage, files, copy-back), [`reading`] for reads (host
// copies, captures), both through `host::copy_mapped`/`read_into`/
// `write_mapped`. A lift by the runtime is not counted as the title's.

/// The protection a block should have now: none if trapped, read-only if
/// guarded, else read-write.
fn protection(g: &Guards, block: usize) -> u32 {
    if g.trapped.contains_key(&block) {
        win::PAGE_NOACCESS
    } else if g.protected.contains(&block) {
        win::PAGE_READONLY
    } else {
        win::PAGE_READWRITE
    }
}

fn set_protection(block: usize, protect: u32) -> bool {
    let mut old = 0;
    // SAFETY: title memory; only its protection changes.
    let setting = std::time::Instant::now();
    let ok = unsafe { win::VirtualProtect((block * BLOCK) as *const _, BLOCK, protect, &mut old) != 0 };
    xnw_core::syscalls::count(xnw_core::syscalls::Call::ProtectTrap, setting);
    // A block of the paired reservation's CPU view is the same memory
    // at the GPU view: both get the protection, so a title access through
    // either is seen (a trap on one view only let reads of the other
    // through, unsettled). The handler takes a GPU-view fault at its CPU
    // block.
    if let Some((cpu, gpu, len)) = *PAIR.lock().unwrap_or_else(|e| e.into_inner())
        && (cpu..cpu + len).contains(&(block * BLOCK))
    {
        // SAFETY: as above, at the other view.
        unsafe { win::VirtualProtect((block * BLOCK + (gpu - cpu)) as *const _, BLOCK, protect, &mut old) };
    }
    ok
}

/// What the traps of one tag caught: reads, writes, and up to 8 distinct
/// faulting instructions (address, or title rva once [`set_title_image`]
/// is known) with the thread of the first fault there.
#[derive(Default, Clone)]
pub struct TrapStats {
    pub reads: u64,
    pub writes: u64,
    pub sites: Vec<(usize, u32)>,
}

static TRAP_STATS: Mutex<BTreeMap<u32, TrapStats>> = Mutex::new(BTreeMap::new());

/// The title image, for faulting instructions as rvas: base, bytes.
static TITLE_IMAGE: Mutex<(usize, usize)> = Mutex::new((0, 0));

/// The loader tells where the title's image lies (for trap sites as rvas).
pub fn set_title_image(base: usize, len: usize) {
    *TITLE_IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = (base, len);
}

/// Called when a trapped block of a tag in its range faults (or is about to
/// be read by us), on the faulting thread, with the guards' lock released
/// and the block writable, before the faulting access goes on: `(tag,
/// block start, bytes)`. The texture bypass fills bypassed texture pages from disk
/// (0x1000-0x1FFF); the buffer roles put GPU results back (0x10000 on).
pub type TrapCallback = fn(tag: u32, address: usize, len: usize);
static CALLBACKS: Mutex<Vec<(std::ops::Range<u32>, TrapCallback)>> = Mutex::new(Vec::new());

/// Registers `callback` for the traps whose tag lies in `tags`.
pub fn on_trap(tags: std::ops::Range<u32>, callback: TrapCallback) {
    CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).push((tags, callback));
}

fn callback_for(tag: u32) -> Option<TrapCallback> {
    CALLBACKS.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(r, _)| r.contains(&tag)).map(|&(_, c)| c)
}

/// Traps the blocks (64 KiB) of `[start, start + len)` against reads and
/// writes, noting faults under `tag` (the texture bypass uses 0x1000-0x1FFF; the buffer
/// roles 0x10000 on). Blocks that cannot be protected (unmapped) are left.
/// An address in the GPU view of the paired reservation is trapped at its
/// CPU view.
pub fn trap_reads(start: usize, len: usize, tag: u32) {
    if len == 0 {
        return;
    }
    let end = start.saturating_add(len);
    let d = alias(start, end);
    install();
    let mut g = guards();
    for block in (start - d) / BLOCK..(end - d).div_ceil(BLOCK) {
        if g.trapped.contains_key(&block) {
            g.trapped.insert(block, tag);
            continue;
        }
        if set_protection(block, win::PAGE_NOACCESS) {
            g.trapped.insert(block, tag);
        }
    }
}

/// Lifts the traps of `[start, start + len)` (a guard stays).
pub fn untrap(start: usize, len: usize) {
    let end = start.saturating_add(len);
    let d = alias(start, end);
    lift(&mut guards(), start - d, end - d);
}

/// Lifts the traps of `[start, end)` (CPU-view addresses); true if any was.
fn lift(g: &mut Guards, start: usize, end: usize) -> bool {
    let blocks: Vec<usize> = g.trapped.range(start / BLOCK..end.div_ceil(BLOCK)).map(|(&b, _)| b).collect();
    for &block in &blocks {
        g.trapped.remove(&block);
        set_protection(block, protection(g, block));
    }
    !blocks.is_empty()
}

/// Before our own read of title memory `[at, at + len)` (through the kernel
/// or directly): its traps are lifted, without counting it as the title's.
/// With a trap hook (buffer roles on), a block holding GPU results not yet
/// copied back gets them first.
pub fn reading(at: usize, len: usize) {
    let end = at.saturating_add(len);
    let d = alias(at, end);
    let tags: Vec<(usize, u32)> = {
        let g = guards();
        if g.trapped.is_empty() {
            return;
        }
        g.trapped.range((at - d) / BLOCK..(end - d).div_ceil(BLOCK)).map(|(&b, &t)| (b, t)).collect()
    };
    for (block, tag) in tags {
        settle(block, tag, false, false);
    }
}

/// As [`reading`] for many small ranges `(at, len)`: one look at the pair
/// and the guards for all (the texture table's descriptor records,
/// about 1,240 a frame, took both locks per record).
pub fn reading_all(ranges: &[(usize, usize)]) {
    if ranges.is_empty() {
        return;
    }
    let pair = *PAIR.lock().unwrap_or_else(|e| e.into_inner());
    let mut tags: Vec<(usize, u32)> = {
        let g = guards();
        if g.trapped.is_empty() {
            return;
        }
        let mut tags = Vec::new();
        for &(at, len) in ranges {
            let end = at.saturating_add(len);
            let d = match pair {
                Some((cpu, gpu, n)) if at >= gpu && end <= gpu + n => gpu - cpu,
                _ => 0,
            };
            tags.extend(g.trapped.range((at - d) / BLOCK..(end - d).div_ceil(BLOCK)).map(|(&b, &t)| (b, t)));
        }
        tags
    };
    tags.sort_unstable();
    tags.dedup();
    for (block, tag) in tags {
        settle(block, tag, false, false);
    }
}

/// Reads and writes the traps of `tag` caught, and their sites (title rvas,
/// or addresses outside the title).
pub fn read_trap_stats(tag: u32) -> (u64, u64, Vec<u64>) {
    let s = TRAP_STATS.lock().unwrap_or_else(|e| e.into_inner()).get(&tag).cloned().unwrap_or_default();
    (s.reads, s.writes, s.sites.into_iter().map(|(a, _)| a as u64).collect())
}

/// Takes the stats of every tag at or above `from` (and below `to`) and
/// clears them: per frame statistics.
pub fn take_trap_stats(from: u32, to: u32) -> Vec<(u32, TrapStats)> {
    let mut all = TRAP_STATS.lock().unwrap_or_else(|e| e.into_inner());
    let keys: Vec<u32> = all.range(from..to).map(|(&k, _)| k).collect();
    keys.into_iter().filter_map(|k| all.remove(&k).map(|s| (k, s))).collect()
}

/// A trapped block faulted or is about to be read by us: the hook runs
/// (copy-back on demand), then access comes back. `counted`: the title's
/// own access (not ours); `write`: a write fault.
fn settle(block: usize, tag: u32, counted: bool, _write: bool) {
    let callback = callback_for(tag);
    {
        let mut g = guards();
        if g.trapped.get(&block) != Some(&tag) {
            // Settled by another thread meanwhile, or being filled by one.
            drop(g);
            wait_filled(block);
            return;
        }
        g.trapped.remove(&block);
        if callback.is_some() {
            // Writable before the lock goes: a thread touching the block
            // from now on does not fault on a block no one will open.
            // SAFETY: a plain query.
            g.filling.insert(block, unsafe { win::GetCurrentThreadId() });
            set_protection(block, win::PAGE_READWRITE);
        } else {
            // Access back under the lock: a thread faulting meanwhile then
            // finds the block readable.
            set_protection(block, protection(&g, block));
            if !counted {
                RUNTIME_LIFTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            return;
        }
    }
    if let Some(callback) = callback {
        // The callback writes the block (a page from disk, GPU results):
        // access for it, the lock released (it may read or write title
        // memory through us). Other threads that faulted on the block wait
        // in `wait_filled`.
        // shortcut: a thread that touches the block for the first time
        // while it is filled does not fault and may see it half written.
        IN_CALLBACK.with(|d| d.set(d.get() + 1));
        callback(tag, block * BLOCK, BLOCK);
        IN_CALLBACK.with(|d| d.set(d.get() - 1));
        guards().filling.remove(&block);
    }
    let g = guards();
    set_protection(block, protection(&g, block));
    if !counted {
        RUNTIME_LIFTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

thread_local! {
    /// How deep this thread is in trap callbacks.
    static IN_CALLBACK: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Waits while another thread's callback fills `block`. Never from inside
/// a callback, nor for a block this thread fills: then the block is made
/// writable and the access goes on.
///
/// A test run's stall: two threads each filling one block, each callback writing the other's
/// block (the owed ranges of overlapping host copies), each waiting here for
/// the other. A callback's write must not wait: the bytes it writes are
/// what the waiting thread would have written.
fn wait_filled(block: usize) {
    // SAFETY: a plain query.
    let me = unsafe { win::GetCurrentThreadId() };
    let inside = IN_CALLBACK.with(|d| d.get() > 0);
    loop {
        let g = guards();
        match g.filling.get(&block) {
            None => return,
            Some(&owner) if inside || owner == me => {
                set_protection(block, win::PAGE_READWRITE);
                return;
            }
            Some(_) => {}
        }
        drop(g);
        std::thread::yield_now();
    }
}

/// Traps lifted by our own reads and writes (not the title's), for the
/// statistics.
pub static RUNTIME_LIFTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Notes a fault of `tag` at instruction `pc` on this thread.
fn note_fault(tag: u32, write: bool, pc: usize) {
    let (mut base, mut len) = *TITLE_IMAGE.lock().unwrap_or_else(|e| e.into_inner());
    if base == 0 {
        // The device's title base (set at start); the image's end unknown:
        // within 1 GiB of it counts as the title's.
        (base, len) = (super::object::title_base(), 1 << 30);
    }
    let site = if base != 0 && (base..base + len).contains(&pc) { pc - base } else { pc };
    // SAFETY: a plain query of the calling thread.
    let thread = unsafe { win::GetCurrentThreadId() };
    let mut all = TRAP_STATS.lock().unwrap_or_else(|e| e.into_inner());
    let s = all.entry(tag).or_default();
    if write {
        s.writes += 1;
    } else {
        s.reads += 1;
    }
    if s.sites.len() < 8 && !s.sites.iter().any(|&(a, _)| a == site) {
        s.sites.push((site, thread));
    }
}

fn guards() -> GuardsHeld {
    let g = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a plain query.
    GUARDS_OWNER.store(unsafe { win::GetCurrentThreadId() }, std::sync::atomic::Ordering::Relaxed);
    GuardsHeld(g)
}

/// The thread holding the guards' lock (0: none), for stall dumps.
pub(crate) static GUARDS_OWNER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The guards' lock, noting its holder.
struct GuardsHeld(std::sync::MutexGuard<'static, Guards>);

impl Drop for GuardsHeld {
    fn drop(&mut self) {
        GUARDS_OWNER.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl std::ops::Deref for GuardsHeld {
    type Target = Guards;
    fn deref(&self) -> &Guards {
        &self.0
    }
}

impl std::ops::DerefMut for GuardsHeld {
    fn deref_mut(&mut self) -> &mut Guards {
        &mut self.0
    }
}

/// `VirtualQuery` of our own process.
fn query(at: usize) -> Option<win::MemoryBasicInformation> {
    // SAFETY: plain data filled in by the query.
    let mut info: win::MemoryBasicInformation = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { win::VirtualQueryEx(win::GetCurrentProcess(), at as *const _, &mut info, size_of_val(&info)) };
    (n != 0).then_some(info)
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    if !known_view(start) {
        let info = xnw_core::syscalls::timed(xnw_core::syscalls::Call::QueryView, || query(start))?;
        if info.kind != win::MEM_MAPPED {
            return None;
        }
        VIEWS.lock().unwrap_or_else(|e| e.into_inner()).insert(info.base, info.base + info.region_size);
    }
    install();
    let frame = super::queue::presented();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    for block in first..=last {
        let mut again = false;
        if block < last && !g.clean.contains(&block) {
            let pages = block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE;
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                written.extend(pages);
                g.clean.insert(block);
            } else if !EXACT.with(std::cell::Cell::get) && hot(&mut g, block, frame) {
                // Taken as written once per submission (sweep), not at
                // every range asked about in it: overlapping ranges copied
                // the same hot pages again and again (888 MiB a
                // frame copied).
                if HOT_SEEN.with_borrow_mut(|s| s.as_mut().is_none_or(|s| s.insert(block))) {
                    written.extend(pages);
                }
            } else {
                written.extend(pages);
                g.hot.remove(&block);
                again = true;
            }
        }
        if again {
            run.get_or_insert(block);
            continue;
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Frames in a row a block must be found written to stay unguarded.
const HOT_AFTER: u32 = 3;
/// Frames a block stays unguarded before it is guarded again (to see
/// whether the title still writes it every frame).
const HOT_FOR: u64 = 32;

/// Whether `block`, found written at an asking in `frame`, is one the title
/// writes every frame ("hot"): then it is left unguarded and taken as
/// written at every asking (only its pages are copied), instead of guarded
/// again and opened by the title's next write. A test run spent 664 ms a
/// frame on 'bindless reads', ranges over blocks the title rewrites each
/// frame, each refresh guarding them again (a system call) and each title
/// write faulting them open (an exception). After [`HOT_FOR`] frames a hot
/// block is guarded again to look.
fn hot(g: &mut Guards, block: usize, frame: u64) -> bool {
    // `XNW_HOT=0`: no block is left unguarded (each is guarded again at
    // every asking and opened by the title's next write), to weigh faults
    // against the copies of hot pages at every submission ('bindless
    // reads' 486 ms, 'refresh flush' 169 ms a frame in a test run).
    static OFF: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_HOT").is_ok_and(|v| v.trim() == "0"));
    if *OFF {
        return false;
    }
    if let Some(&since) = g.hot.get(&block) {
        if frame < since + HOT_FOR {
            return true;
        }
        g.hot.remove(&block);
        g.streak.remove(&block);
        return false;
    }
    let s = g.streak.entry(block).or_insert((0, u64::MAX));
    if s.1 != frame {
        s.0 = if s.1.wrapping_add(1) == frame { s.0 + 1 } else { 1 };
        s.1 = frame;
    }
    if s.0 >= HOT_AFTER {
        g.hot.insert(block, frame);
        HOT_BLOCKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return true;
    }
    false
}

/// Blocks unguarded now (hot).
pub(crate) fn hot_now() -> usize {
    guards().hot.len()
}

/// Blocks found written every frame and left unguarded so far.
pub(crate) static HOT_BLOCKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail (unmapped).
fn protect(g: &mut Guards, from: usize, to: usize) {
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        xnw_core::syscalls::timed(xnw_core::syscalls::Call::ProtectGuard, || unsafe { win::VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, win::PAGE_READONLY, &mut old) != 0 })
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    // A trapped block stays without access (any write faults anyway).
    let trapped = g.trapped.range(from..to).next().is_some();
    if !trapped && set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if g.trapped.contains_key(&block) || set(block, block + 1) {
            note(g, block, block + 1);
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let blocks: Vec<usize> = g.protected.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        // A trapped block keeps no access until its trap is lifted.
        set_protection(block, protection(g, block));
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    let end = at.saturating_add(len);
    // Traps first: a kernel write into a no-access page fails (as the
    // 0x800703e6 reads did); GPU results not copied back yet go first.
    reading(at, len);
    let d = alias(at, end);
    open(&mut guards(), at - d, end - d);
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded, and any placed
/// textures backed by that range are evicted.
pub fn title_memory_remapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    {
        let d = alias(at, end);
        forget_views(at - d, end - d);
        forget_views(at, end);
        let mut g = guards();
        // New memory: what was learnt about its blocks goes.
        let span = (at - d) / BLOCK..(end - d).div_ceil(BLOCK);
        let hot: Vec<usize> = g.hot.range(span.clone()).map(|(&b, _)| b).collect();
        for b in hot {
            g.hot.remove(&b);
        }
        let streak: Vec<usize> = g.streak.range(span).map(|(&b, _)| b).collect();
        for b in streak {
            g.streak.remove(&b);
        }
        // New memory: traps over it go (what they held is gone).
        let trapped: Vec<usize> = g.trapped.range((at - d) / BLOCK..(end - d).div_ceil(BLOCK)).map(|(&b, _)| b).collect();
        for block in trapped {
            g.trapped.remove(&block);
            set_protection(block, protection(&g, block));
        }
        let blocks: Vec<usize> = g.protected.range((at - d) / BLOCK..(end - d).div_ceil(BLOCK)).copied().collect();
        for block in blocks {
            g.protected.remove(&block);
            g.clean.remove(&block);
            // A copy may have guarded the block between the runtime's map
            // and this call (the map is made outside our lock): then the new
            // view is read-only and nothing would open it again. Seen as an
            // unhandled write fault in the first world frames (run 0023,
            // 0x274c80000). Fails harmlessly on a view about to be unmapped.
            let mut old = 0;
            // SAFETY: title view memory; only its protection changes.
            xnw_core::syscalls::timed(xnw_core::syscalls::Call::ProtectOpen, || unsafe { win::VirtualProtect((block * BLOCK) as *const _, BLOCK, win::PAGE_READWRITE, &mut old) });
        }
    }
    // Both views of the paired reservation (lookups use either).
    crate::shared::remapped(at, end - at);
    let d = alias(at, end);
    if d != 0 {
        crate::shared::remapped(at - d, end - at);
    } else if let Some(gpu) = gpu_alias(at) {
        crate::shared::remapped(gpu, end - at);
    }
    super::resource::evict_placed_textures(at as u64, end as u64);
    super::shadow::evict_loose(at, len);
    super::descriptor::host::bindless_remapped(at as u64, (end - at) as u64);
}

/// Entries of the write handler by what they were: an access trap,
/// a block being filled, a read outside our guards, a write that opened a
/// guarded block, a write outside our guards, another access violation, an
/// exception of another code.
static VEH: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
const VEH_TRAPPED: usize = 0;
const VEH_FILLING: usize = 1;
const VEH_READ_OTHER: usize = 2;
const VEH_OPENED: usize = 3;
const VEH_WRITE_OTHER: usize = 4;
const VEH_AV_OTHER: usize = 5;
const VEH_OTHER_CODE: usize = 6;

/// The handler's entries per frame over `frames`, taken.
pub(crate) fn veh_text(frames: u64) -> String {
    let names = ["trapped", "filling", "read outside guards", "write opened a guard", "write outside guards", "other access violation", "other code"];
    let v: Vec<String> = VEH.iter().zip(names).map(|(c, n)| format!("{n} {}", c.swap(0, Ordering::Relaxed) / frames.max(1))).collect();
    format!("exceptions per frame: {}", v.join(", "))
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    unsafe extern "system" fn on_write(info: *mut win::ExceptionPointers) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).record };
        const READ: usize = 0;
        const WRITE: usize = 1;
        if r.code != win::EXCEPTION_ACCESS_VIOLATION || r.param_count < 2 || !matches!(r.params[0], READ | WRITE) {
            VEH[if r.code == win::EXCEPTION_ACCESS_VIOLATION { VEH_AV_OTHER } else { VEH_OTHER_CODE }].fetch_add(1, Ordering::Relaxed);
            return win::EXCEPTION_CONTINUE_SEARCH;
        }
        // At its CPU view: traps and guards are kept there (a fault at
        // the GPU view of the paired reservation is the same block).
        // (A fault at a non-canonical address is reported as address !0: no overflow.)
        let at = r.params[1] - alias(r.params[1], r.params[1].saturating_add(1));
        let write = r.params[0] == WRITE;
        // A trapped block: noted, settled (copy-back on demand), access
        // back; a write then faults again on a guard if there is one.
        //
        // One hold of the lock from this look to the verdict below. The
        // look used to be its own hold: a fault on a trap another thread
        // had just lifted found the block neither trapped nor filling, and
        // a trap laid again before the verdict asked the kernel made the
        // page no-access once more, so the fault was passed on as someone
        // else's and ended the process (0xc0000005; test
        // `a_trap_laid_again_while_a_fault_is_judged_is_settled`, 20 of 20
        // runs). Under the lock no trap is laid or lifted in between.
        let mut g = guards();
        if let Some(tag) = g.trapped.get(&(at / BLOCK)).copied() {
            drop(g);
            VEH[VEH_TRAPPED].fetch_add(1, Ordering::Relaxed);
            note_fault(tag, write, r.address);
            settle(at / BLOCK, tag, true, write);
            return win::EXCEPTION_CONTINUE_EXECUTION;
        }
        if g.filling.contains_key(&(at / BLOCK)) {
            drop(g);
            VEH[VEH_FILLING].fetch_add(1, Ordering::Relaxed);
            // Another thread's callback fills the block: wait, then retry.
            wait_filled(at / BLOCK);
            return win::EXCEPTION_CONTINUE_EXECUTION;
        }
        if !write {
            VEH[VEH_READ_OTHER].fetch_add(1, Ordering::Relaxed);
            // Another thread may have settled the trap since this one
            // faulted: then the read goes through now.
            return if xnw_core::syscalls::timed(xnw_core::syscalls::Call::QueryFault, || query(r.params[1])).is_some_and(|i| i.state == win::MEM_COMMIT && i.protect & (win::PAGE_READONLY | win::PAGE_READWRITE) != 0) {
                win::EXCEPTION_CONTINUE_EXECUTION
            } else {
                win::EXCEPTION_CONTINUE_SEARCH
            };
        }
        // Still held: another thread may have opened the block since this
        // one faulted, and a copy may have guarded it again since (then it
        // is ours again, not read-only for some other reason).
        if open(&mut g, at, at.saturating_add(1)) {
            VEH[VEH_OPENED].fetch_add(1, Ordering::Relaxed);
            return win::EXCEPTION_CONTINUE_EXECUTION;
        }
        VEH[VEH_WRITE_OTHER].fetch_add(1, Ordering::Relaxed);
        if writable(r.params[1]) { win::EXCEPTION_CONTINUE_EXECUTION } else { win::EXCEPTION_CONTINUE_SEARCH }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        win::AddVectoredExceptionHandler(1, on_write);
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    xnw_core::syscalls::timed(xnw_core::syscalls::Call::QueryFault, || query(at)).is_some_and(|info| info.state == win::MEM_COMMIT && info.protect == win::PAGE_READWRITE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guards, traps, journal and fault handler are the process's and
    /// cargo runs tests on parallel threads: one test at a time.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static ONE: Mutex<()> = Mutex::new(());
        ONE.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        let _serial = serial();
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe { win::VirtualAlloc(std::ptr::null(), 4 * PAGE, win::MEM_RESERVE | win::MEM_COMMIT | win::MEM_WRITE_WATCH, win::PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        forget_journal(p, 4 * PAGE);
        let (start, end) = (p, p + 4 * PAGE);
        // The journal is the process's: pages of views other tests freed
        // here may be listed from earlier askings; only newer ones count.
        let (first, before) = writes(start, end, false).expect("watched");
        let fresh = |v: Vec<(usize, u64)>| v.into_iter().filter(|&(_, w)| w >= first).collect::<Vec<_>>();
        assert!(fresh(before).is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(fresh(after), [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(fresh(again), [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(writes(plain.as_mut_ptr() as usize, plain.as_ptr() as usize + PAGE, true).is_none());
        // Not freed: the journal is the process's and keeps what it noted for
        // these pages; a later test's allocation at the same address would
        // list them (a_sweep_shares_the_kernels_answer_with_nested_ranges
        // failed 3 in 20 loops that way).
    }

    #[test]
    fn gaps_and_sweeps_keep_ranges_disjoint() {
        let _serial = serial();
        let mut s = Vec::new();
        assert_eq!(gaps(&s, 0, 10), [(0, 10)]);
        sweep_in(&mut s, 4, 6);
        sweep_in(&mut s, 10, 12);
        assert_eq!(gaps(&s, 0, 20), [(0, 4), (6, 10), (12, 20)]);
        assert_eq!(gaps(&s, 4, 6), []);
        assert_eq!(gaps(&s, 5, 11), [(6, 10)]);
        sweep_in(&mut s, 6, 10); // touching both: one range
        assert_eq!(s, [(4, 12)]);
        sweep_in(&mut s, 0, 2);
        sweep_in(&mut s, 1, 5);
        assert_eq!(s, [(0, 12)]);
    }

    /// A fresh write-watched allocation of `pages` pages. Never freed: the
    /// journal keeps its pages, and a later test given the same addresses
    /// would find them written.
    fn watched(pages: usize) -> usize {
        // SAFETY: a fresh allocation, kept for the rest of the run.
        let p = unsafe { win::VirtualAlloc(std::ptr::null(), pages * PAGE, win::MEM_RESERVE | win::MEM_COMMIT | win::MEM_WRITE_WATCH, win::PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        forget_journal(p, pages * PAGE);
        p
    }

    /// The journal is the process's: pages of views other tests unmapped may
    /// sit at the addresses this allocation got (a_sweep_* and
    /// written_pages_* failed 3 in 30 loops listing them). Fresh memory has
    /// no history.
    fn forget_journal(at: usize, len: usize) {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        let pages: Vec<usize> = j.written.range(at / PAGE..(at + len).div_ceil(PAGE)).map(|(&k, _)| k).collect();
        for k in pages {
            j.written.remove(&k);
        }
    }

    /// In a sweep a nested range is not asked about again, yet sees what
    /// the outer asking found; after the sweep the kernel is asked as before.
    #[test]
    fn a_sweep_shares_the_kernels_answer_with_nested_ranges() {
        let _serial = serial();
        let p = watched(4);
        // SAFETY: committed pages of ours.
        unsafe { ((p + 2 * PAGE) as *mut u8).write(1) };
        let sweep = Sweep::begin();
        let (outer, all) = writes(p, p + 4 * PAGE, false).expect("watched");
        assert_eq!(all, [(p / PAGE + 2, outer)]);
        // Written after the outer asking, inside the sweep: seen from the
        // next submission's sweep on.
        // SAFETY: as above.
        unsafe { ((p + 3 * PAGE) as *mut u8).write(1) };
        let (inner, nested) = writes(p + 2 * PAGE, p + 4 * PAGE, false).expect("watched");
        assert!(inner > outer);
        assert_eq!(nested, [(p / PAGE + 2, outer)]);
        drop(sweep);
        let (later, after) = writes(p + 2 * PAGE, p + 4 * PAGE, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 2, outer), (p / PAGE + 3, later)]);
    }

    /// A write the sweeping thread makes itself (after `title_written`) is
    /// seen by the next asking in the same sweep.
    #[test]
    fn a_sweep_asks_again_after_its_own_write() {
        let _serial = serial();
        let p = watched(2);
        let sweep = Sweep::begin();
        let (_, before) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert!(before.is_empty());
        Sweep::title_written();
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE) as *mut u8).write(1) };
        let (now, after) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        drop(sweep);
    }

    /// A pagefile section of `blocks` blocks, mapped read-write.
    fn view(blocks: usize) -> usize {
        // SAFETY: plain section creation and mapping.
        let view = unsafe {
            let section = win::CreateFileMappingW(win::INVALID_HANDLE_VALUE, std::ptr::null(), win::PAGE_READWRITE, 0, (blocks * BLOCK) as u32, std::ptr::null());
            assert!(!section.is_null());
            let at = win::MapViewOfFile(section, win::FILE_MAP_WRITE, 0, 0, blocks * BLOCK) as usize;
            win::CloseHandle(section);
            at
        };
        assert_ne!(view, 0);
        // New memory has no history: what an earlier test's view at this address
        // left in the guard, trap and journal tables goes, as at a title remap
        // (a stale trap entry made trap_reads skip protecting the new view:
        // the read did not trap, 4 in 40 loops).
        title_memory_remapped(view, blocks * BLOCK);
        forget_journal(view, blocks * BLOCK);
        view
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        let _serial = serial();
        use std::io::{Read, Seek, Write};
        let view = view(2);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let path = std::env::temp_dir().join(format!("xnw-watch-{}.bin", std::process::id()));
        let mut file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        drop(file);
        let _ = std::fs::remove_file(&path);
        // SAFETY: our view.
        unsafe { win::UnmapViewOfFile(view as *const _) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// One section seen at two addresses, as the paired reservation: a copy
    /// taken at the GPU address guards the CPU one, a CPU write shows as a
    /// write at the GPU address, and pages come back at the GPU address.
    #[test]
    fn the_gpu_view_of_the_pair_is_watched_at_the_cpu_view() {
        let _serial = serial();
        // Two views of one section standing in for the reservation (only
        // this test sets the pair; it clears it at the end).
        // SAFETY: plain section creation and two mappings of it.
        let (cpu, gpu) = unsafe {
            let section = win::CreateFileMappingW(win::INVALID_HANDLE_VALUE, std::ptr::null(), win::PAGE_READWRITE, 0, (2 * BLOCK) as u32, std::ptr::null());
            assert!(!section.is_null());
            let a = win::MapViewOfFile(section, win::FILE_MAP_WRITE, 0, 0, 2 * BLOCK) as usize;
            let b = win::MapViewOfFile(section, win::FILE_MAP_WRITE, 0, 0, 2 * BLOCK) as usize;
            win::CloseHandle(section);
            (a, b)
        };
        assert!(cpu != 0 && gpu != 0);
        let (low, high) = (cpu.min(gpu), cpu.max(gpu));
        // The pair's rule needs the GPU view above the CPU one.
        let (cpu, gpu) = (low, high);
        use_pair(cpu, gpu, 2 * BLOCK);
        let (first, all) = writes(gpu, gpu + 2 * BLOCK, true).expect("a view");
        assert_eq!(all.iter().map(|&(p, _)| p).collect::<Vec<_>>(), [gpu / PAGE, gpu / PAGE + 1], "first asking: everything");
        // The CPU view is now read-only, the GPU one is not.
        assert_eq!(query(cpu).unwrap().protect, win::PAGE_READONLY);
        assert_eq!(query(gpu).unwrap().protect, win::PAGE_READWRITE);
        // SAFETY: a write through the CPU view: the handler opens the block.
        unsafe { ((cpu + BLOCK + 5) as *mut u8).write(7) };
        // SAFETY: the same byte through the GPU view.
        assert_eq!(unsafe { ((gpu + BLOCK + 5) as *const u8).read() }, 7);
        let (now, after) = writes(gpu, gpu + 2 * BLOCK, true).expect("a view");
        assert!(now > first);
        assert_eq!(after, [(gpu / PAGE, first), (gpu / PAGE + 1, now)], "only the written block is new");
        writing(gpu, 2 * BLOCK);
        assert_eq!(query(cpu).unwrap().protect, win::PAGE_READWRITE, "our write at the GPU address opens the CPU view");
        // SAFETY: our views.
        unsafe {
            win::UnmapViewOfFile(cpu as *const _);
            win::UnmapViewOfFile(gpu as *const _);
        }
        title_memory_remapped(gpu, 2 * BLOCK);
        *PAIR.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// A trapped block faults on the first read and on the first write,
    /// each noted under its tag with the faulting site, and access comes
    /// back; our own reads lift a trap without counting; a guard under a
    /// trap stays (read-only again after the read, opened by the write).
    #[test]
    fn traps_note_reads_and_writes_and_compose_with_guards() {
        let _serial = serial();
        let tag = 0x4000 + std::process::id() % 0x1000;
        let view = view(2);
        // SAFETY: our view.
        unsafe { (view as *mut u32).write(5) };
        trap_reads(view, BLOCK, tag);
        assert_eq!(query(view).unwrap().protect, win::PAGE_NOACCESS);
        // SAFETY: a read of our trapped view: the handler gives access back.
        assert_eq!(unsafe { std::ptr::read_volatile(view as *const u32) }, 5);
        let (reads, writes, sites) = read_trap_stats(tag);
        assert_eq!((reads, writes, sites.len()), (1, 0, 1));
        assert_eq!(query(view).unwrap().protect, win::PAGE_READWRITE, "untrapped after the fault");
        trap_reads(view, BLOCK, tag);
        // SAFETY: a write: noted as one.
        unsafe { std::ptr::write_volatile((view + 8) as *mut u32, 6) };
        assert_eq!(read_trap_stats(tag).1, 1);
        // Our own read lifts without counting.
        trap_reads(view, BLOCK, tag);
        reading(view, 16);
        assert_eq!(read_trap_stats(tag).0, 1, "not the title's read");
        assert_eq!(query(view).unwrap().protect, win::PAGE_READWRITE);
        // Guarded, then trapped: the read leaves the guard; the write opens it.
        assert!(guard(view, view + BLOCK).is_some());
        trap_reads(view, BLOCK, tag);
        // SAFETY: as above.
        assert_eq!(unsafe { std::ptr::read_volatile((view + 8) as *const u32) }, 6);
        assert_eq!(query(view).unwrap().protect, win::PAGE_READONLY, "guard kept");
        // SAFETY: as above; the guard's handler opens the block.
        unsafe { std::ptr::write_volatile(view as *mut u32, 7) };
        assert_eq!(query(view).unwrap().protect, win::PAGE_READWRITE);
        trap_reads(view, 2 * BLOCK, tag);
        untrap(view, 2 * BLOCK);
        assert_eq!(query(view + BLOCK).unwrap().protect, win::PAGE_READWRITE);
        assert_eq!(read_trap_stats(tag).0, 2);
        // SAFETY: our view.
        unsafe { win::UnmapViewOfFile(view as *const _) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A test run's stall: two threads each settle one trapped block and
    /// each callback writes the other's block while both are being filled
    /// (and, here, not writable). Neither may wait for the other. One
    /// thread does it holding a lock, as the submission lock is held by our
    /// own reads at submission.
    ///
    /// The test took the other block's access away with nothing holding
    /// the two callbacks together after that: where one fill had ended
    /// (its entry gone, its protection given back) before the other
    /// callback took the access away, that write faulted on a page no
    /// state of ours stood for, and the handler rightly passed it on:
    /// 0xc0000005, 21 of 30 runs, every stack `cross` <- `settle`, block
    /// "trapped None filling None", protection no-access. The barriers
    /// keep both fills open until both writes have landed, so each write
    /// faults on a block that is being filled.
    #[test]
    fn crossed_fills_do_not_wait_on_each_other() {
        let _serial = serial();
        // Below the roles' tags (0x10000 on) and the texture bypass's (0x1000-0x1FFF).
        const TAGS: std::ops::Range<u32> = 0x2000..0x2002;
        static BLOCKS: std::sync::OnceLock<[usize; 2]> = std::sync::OnceLock::new();
        static BOTH: std::sync::Barrier = std::sync::Barrier::new(2);
        static CLOSED: std::sync::Barrier = std::sync::Barrier::new(2);
        static WRITTEN: std::sync::Barrier = std::sync::Barrier::new(2);
        // Writes made while the block written had no access (so they faulted).
        static FAULTED: AtomicU64 = AtomicU64::new(0);
        fn cross(tag: u32, _address: usize, _len: usize) {
            let blocks = BLOCKS.get().unwrap();
            let other = blocks[1 - (tag - TAGS.start) as usize];
            // Both blocks are being filled now.
            BOTH.wait();
            // As if the other's protection had not been opened yet.
            set_protection(other / BLOCK, win::PAGE_NOACCESS);
            CLOSED.wait();
            if query(other).is_some_and(|i| i.protect == win::PAGE_NOACCESS) {
                FAULTED.fetch_add(1, Ordering::Relaxed);
            }
            // SAFETY: a write into the other test view: the handler lets it
            // through without waiting for the other thread.
            unsafe { std::ptr::write_volatile(other as *mut u8, 0xCD) };
            // Neither fill ends before both writes have gone through.
            WRITTEN.wait();
        }
        on_trap(TAGS, cross);
        let (a, b) = (view(1), view(1));
        BLOCKS.set([a, b]).unwrap();
        trap_reads(a, BLOCK, TAGS.start);
        trap_reads(b, BLOCK, TAGS.start + 1);
        let (done, finished) = std::sync::mpsc::channel();
        static SUBMISSION: Mutex<()> = Mutex::new(());
        let d1 = done.clone();
        std::thread::spawn(move || {
            let _held = SUBMISSION.lock().unwrap();
            reading(a, 4);
            d1.send(()).unwrap();
        });
        std::thread::spawn(move || {
            // SAFETY: a read of our trapped view.
            unsafe { std::ptr::read_volatile(b as *const u8) };
            done.send(()).unwrap();
        });
        for _ in 0..2 {
            finished.recv_timeout(std::time::Duration::from_secs(10)).expect("a fill waited on the other");
        }
        // Both writes met a block without access that was being filled.
        assert_eq!(FAULTED.load(Ordering::Relaxed), 2);
        // SAFETY: both views are ours.
        unsafe { assert_eq!((*(a as *const u8), *(b as *const u8)), (0xCD, 0xCD)) };
        // SAFETY: our views.
        unsafe {
            win::UnmapViewOfFile(a as *const _);
            win::UnmapViewOfFile(b as *const _);
        }
        title_memory_remapped(a, BLOCK);
        title_memory_remapped(b, BLOCK);
    }

    /// A fault is judged on the block's state when it is judged, not when
    /// the handler first looked: one thread reads and writes a block while
    /// another lays and lifts a trap on it, until the handler has met
    /// `XNW_WATCH_STRESS` (default 5,000) faults whose trap was lifted
    /// before it looked, or 3 s have passed. Such a fault found the block
    /// neither trapped nor filling, and where the trap was laid again
    /// before the handler asked the kernel whether the access may go on,
    /// it took the no-access page for someone else's and passed the
    /// exception on (0xc0000005, the process ends: 20 of 20 runs before
    /// the fix on each PC, at "unhandled write, block trapped
    /// Some(0x2800)" or at the read's query).
    #[test]
    fn a_trap_laid_again_while_a_fault_is_judged_is_settled() {
        let _serial = serial();
        // No callback for this tag.
        const TAG: u32 = 0x2800;
        static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let env = |name: &str, default: u64| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
        // Spins the trap is held for, so the reader meets it.
        let (wanted, hold) = (env("XNW_WATCH_STRESS", 5000), env("XNW_WATCH_HOLD", 16));
        let lifted = || VEH[VEH_READ_OTHER].load(Ordering::Relaxed) + VEH[VEH_WRITE_OTHER].load(Ordering::Relaxed);
        let view = view(1);
        let before = lifted();
        let trapper = std::thread::spawn(move || {
            let (start, mut laid) = (std::time::Instant::now(), 0u64);
            while lifted() - before < wanted && (laid % 256 != 0 || start.elapsed().as_secs() < 3) {
                trap_reads(view, BLOCK, TAG);
                for _ in 0..hold {
                    std::hint::spin_loop();
                }
                untrap(view, BLOCK);
                laid += 1;
            }
            DONE.store(true, Ordering::Relaxed);
            laid
        });
        let (mut sum, mut rounds) = (0u32, 0u64);
        while !DONE.load(Ordering::Relaxed) {
            // SAFETY: our view; the handler gives the access back.
            unsafe {
                sum += u32::from(std::ptr::read_volatile((view + 8) as *const u8));
                std::ptr::write_volatile((view + 16) as *mut u8, rounds as u8);
            }
            rounds += 1;
        }
        let laid = trapper.join().unwrap();
        let (reads, writes, _) = read_trap_stats(TAG);
        println!("{laid} traps laid, {rounds} rounds: {reads} reads and {writes} writes trapped, {} faults whose trap was lifted before the look", lifted() - before);
        assert_eq!(sum, 0);
        assert!(reads + writes > 0, "the reader met no trap");
        // SAFETY: our view.
        unsafe { win::UnmapViewOfFile(view as *const _) };
        title_memory_remapped(view, BLOCK);
    }

    /// A callback registered for a tag range runs on the faulting
    /// thread before the read goes on, and can write the block (a fill);
    /// our own read runs it too.
    #[test]
    fn trap_callbacks_fill_before_the_access() {
        let _serial = serial();
        // Below the roles' tags (0x10000 on) and the texture bypass's (0x1000-0x1FFF).
        const TAGS: std::ops::Range<u32> = 0x3000..0x3100;
        fn fill(_tag: u32, address: usize, len: usize) {
            // SAFETY: the block the trap gave the callback, writable now.
            unsafe { std::slice::from_raw_parts_mut(address as *mut u8, len).fill(0xAB) };
        }
        on_trap(TAGS, fill);
        let view = view(1);
        trap_reads(view, BLOCK, TAGS.start);
        // SAFETY: a read of our trapped view: filled first.
        assert_eq!(unsafe { std::ptr::read_volatile((view + 100) as *const u8) }, 0xAB);
        assert_eq!(read_trap_stats(TAGS.start).0, 1);
        // SAFETY: our view, writable again.
        unsafe { (view as *mut u8).write_bytes(0, BLOCK) };
        trap_reads(view, BLOCK, TAGS.start + 1);
        reading(view, 4);
        // SAFETY: our view.
        assert_eq!(unsafe { *(view as *const u8) }, 0xAB, "our read fills too");
        // SAFETY: our view.
        unsafe { win::UnmapViewOfFile(view as *const _) };
        title_memory_remapped(view, BLOCK);
    }

    /// A block the title writes every frame is left unguarded after
    /// [`HOT_AFTER`] frames (no system call, no fault) and still reported
    /// written at every asking.
    #[test]
    fn blocks_written_every_frame_stay_unguarded() {
        let _serial = serial();
        let view = view(1);
        for _ in 0..HOT_AFTER + 1 {
            super::super::queue::pass_frame();
            // SAFETY: our view; the handler opens a guarded block.
            unsafe { std::ptr::write_volatile(view as *mut u8, 1) };
            assert!(guard(view, view + BLOCK).is_some_and(|w| w.contains(&(view / PAGE))));
        }
        assert_eq!(query(view).unwrap().protect, win::PAGE_READWRITE, "hot: not guarded again");
        let again = guard(view, view + BLOCK).unwrap();
        assert!(again.contains(&(view / PAGE)), "a hot block is taken as written");
        // In a submission (sweep), once: overlapping ranges do not copy it
        // again; an exact asking is not answered from the hot state.
        let sweep = Sweep::begin();
        assert!(guard(view, view + BLOCK).unwrap().contains(&(view / PAGE)));
        assert!(!guard(view, view + BLOCK).unwrap().contains(&(view / PAGE)), "once per sweep");
        drop(sweep);
        // SAFETY: our view.
        unsafe { win::UnmapViewOfFile(view as *const _) };
        title_memory_remapped(view, BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let _serial = serial();
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}
