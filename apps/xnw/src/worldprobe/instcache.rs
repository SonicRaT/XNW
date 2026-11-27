//! The instance-batch cache (`XNW_HOOK_CACHE=on`, default off; HOOK1):
//! UE 5.9's cached mesh draw commands mapped onto the draw hook
//!. There the scene builds a
//! static mesh's command once at `FPrimitiveSceneInfo::AddToScene`
//! (`CacheMeshDrawCommands`, Renderer/Private/PrimitiveSceneInfo.cpp:587),
//! drops it at removal (`RemoveCachedMeshDrawCommands`, :795) and only
//! references it per frame. Here no add event exists: a batch is decoded on
//! its first sight on a thread, dropped when worldahead's unload ring names
//! the drawable it was first drawn with, or after 600 frames unused.
//!
//! Per instanced draw the uncached hook (`capture_instances`) reads the
//! IcRec (0xA4), its desc (0x90), the list header L (0xA0), the handler's
//! two words, the LOD record (140 B, copied into a new buffer) and, per frame
//! and thread, takes the static-records lock. Here, per draw: the IcDraw,
//! rec+0..0x28 (desc, handler, LOD record T, near bitset: one line, checked
//! against the batch's first sight) and rec+0xA0 (the LOD mask), the hide
//! bitset ctx+0xA10 and both bitsets' words (copied once per frame as
//! before), and kind C's CPU records (copied once per frame: they may change).
//! Once per thread and batch: desc, L and its vtable, the handler's words,
//! the LOD record, the static records (the global `STATIC_INST`, whose lock
//! is now taken on first sight only: this map is its per-thread L1, the
//! worldtex pattern, worldtex.rs:248-250) and a refused batch's fallback.
//!
//! Assumed [I]: a batch's desc, L, handler words and LOD record bytes do not
//! change while rec+0, +8, +0x10 keep their values and its drawable stays
//! loaded. The LOD records stay per thread rather than in the global table:
//! a global table keyed by address would hand a freed record's bytes to a
//! later batch at the same address, a per-thread entry is read again with
//! its batch. Gate: `cached_instances_equal_the_uncached_hook` (thin.rs).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use super::{Capture, InstInfo, LODREC_BYTES, MAX_BATCH, NO_LOD, NO_PAL, bitset_copy, dcopy, dread, fallback_mode, fallback_records, hide_mask, is_grass_list, le16, le32, le64, log_refusal, static_records};

/// `XNW_HOOK_CACHE=on`.
pub(super) fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNW_HOOK_CACHE").is_ok_and(|v| v.trim() == "on"))
}

/// Entries unused this many frames go (the static records' age).
const AGE: u64 = 600;
/// A thread's map is swept early past this many entries.
const MAX_ENTRIES: usize = 1 << 16;

/// A batch's records, decided at first sight.
enum Block {
    /// None to draw: a GPU batch with count 0, or no CPU list.
    None,
    /// Kind S, accepted: the static map records at L+0x90 (per-frame key
    /// (records, n)); None when their copy failed (the batch is not kept).
    List { recs: u64, b: Option<Arc<Vec<u8>>> },
    /// Kind S, refused for `why` (inst_calls 2-4): the fallback read at
    /// first sight with the switches then (`mode`), asked again per call.
    Refused { why: usize, l: u64, grass: bool, mode: u8, fb: Option<(u64, Arc<Vec<u8>>)> },
    /// Kind C: the CPU records desc+0x78, copied once per frame.
    Cpu,
}

/// One batch's statics: what `capture_instances` reads that does not
/// change while the batch is the same (rec+0, +8, +0x10 its key words).
struct Batch {
    desc: u64,
    h8: u64,
    tp: u64,
    /// The drawable of the draw it was first seen with (unloads drop it).
    drawable: u64,
    count: u32,
    n: u32,
    gpu: u64,
    cpu: u64,
    handler: u64,
    h228: u32,
    h33c: u16,
    lod: Option<[u8; LODREC_BYTES]>,
    block: Block,
    used: u64,
}

/// A pointer hash: one folded multiply per word (aligned pointers spread
/// over the buckets); also the capture's key maps (worldprobe.rs `KeyMap`).
#[derive(Default, Clone, Copy)]
pub(super) struct PtrHash(u64);

impl std::hash::Hasher for PtrHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, b: &[u8]) {
        for x in b {
            self.write_u64(u64::from(*x));
        }
    }
    fn write_u64(&mut self, v: u64) {
        let m = u128::from(self.0 ^ v) * 0x9E37_79B9_7F4A_7C15;
        self.0 = (m >> 64) as u64 ^ m as u64;
    }
    fn write_u32(&mut self, v: u32) {
        self.write_u64(u64::from(v));
    }
    fn write_u8(&mut self, v: u8) {
        self.write_u64(u64::from(v));
    }
}

type ByPtr<V> = HashMap<u64, V, std::hash::BuildHasherDefault<PtrHash>>;

/// One recording thread's batches by IcRec pointer.
struct Cache {
    map: ByPtr<Batch>,
    /// The batches first drawn with each drawable (stale rows are skipped: their batch names another).
    by_drawable: ByPtr<Vec<u64>>,
    frame: u64,
    swept: u64,
    /// worldahead's unload ring, read up to here.
    tail: u64,
    /// This thread's counts since the last frame: [hits, decoded, stale, unloaded, aged].
    n: [u64; 5],
    /// The entries last reported to `ENTRIES`.
    reported: u64,
}

thread_local! {
    static CACHE: RefCell<Option<Cache>> = const { RefCell::new(None) };
}

/// Since the last [`line`]: [hits, decoded, stale, unloaded, aged]; and the threads' entries now.
static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static ENTRIES: AtomicU64 = AtomicU64::new(0);

impl Cache {
    fn new(frame: u64) -> Cache {
        Cache { map: ByPtr::default(), by_drawable: ByPtr::default(), frame, swept: frame, tail: crate::worldahead::unload_head(), n: [0; 5], reported: 0 }
    }

    /// The first instanced draw of a frame on this thread: the counts out,
    /// the unloads since the last frame applied, the age sweep when due.
    fn new_frame(&mut self, frame: u64) {
        self.frame = frame;
        let (map, by, n) = (&mut self.map, &mut self.by_drawable, &mut self.n);
        self.tail = crate::worldahead::unloads_since(self.tail, &mut |o| {
            for r in by.remove(&o).unwrap_or_default() {
                if map.get(&r).is_some_and(|b| b.drawable == o) {
                    map.remove(&r);
                    n[3] += 1;
                }
            }
        });
        if frame.saturating_sub(self.swept) >= AGE || map.len() > MAX_ENTRIES {
            let before = map.len();
            map.retain(|_, b| frame.saturating_sub(b.used) <= AGE);
            n[4] += (before - map.len()) as u64;
            by.clear();
            for (r, b) in map.iter() {
                if b.drawable != 0 {
                    by.entry(b.drawable).or_default().push(*r);
                }
            }
            self.swept = frame;
        }
        for (k, v) in self.n.iter_mut().enumerate() {
            STATS[k].fetch_add(std::mem::take(v), Ordering::Relaxed);
        }
        let now = self.map.len() as u64;
        ENTRIES.fetch_add(now.wrapping_sub(self.reported), Ordering::Relaxed);
        self.reported = now;
    }
}

/// A batch's statics as `capture_instances` reads them from rec's first
/// 0x28 bytes, and whether it may be kept (every read that the uncached
/// hook would try again on its next call succeeded). None: desc unreadable.
fn decode(head: &[u8; 0x28], frame: u64, drawable: u64) -> Option<(Batch, bool)> {
    let desc = le64(head, 0);
    // SAFETY: the batch desc the call reads (fault-fixed).
    let h = unsafe { dread::<[u8; 0x90]>(desc) }?;
    let n = le32(&h, 8).min(MAX_BATCH);
    let (gpu, cpu) = (le64(&h, 0x68), le64(&h, 0x78));
    let tp = le64(head, 0x10);
    // SAFETY: the LOD record the call hands to the culling shader (fault-fixed).
    let lod = unsafe { dread::<[u8; LODREC_BYTES]>(tp) };
    let handler = le64(head, 8).wrapping_sub(0x2C4);
    // SAFETY: the batch's handler (fault-fixed reads).
    let (h228, h33c) = unsafe { (dread::<u32>(handler.wrapping_add(0x228)).unwrap_or(u32::MAX), dread::<u16>(handler.wrapping_add(0x33C)).unwrap_or(u16::MAX)) };
    let mut keep = lod.is_some();
    let block = if gpu != 0 && n != 0 {
        let l = le64(&h, 0);
        // SAFETY: the instance list the title's culling reads for this call.
        let lh = unsafe { dread::<[u8; 0xA0]>(l) };
        let (recs, count) = lh.map_or((0, 0), |lh| (le64(&lh, 0x90), u32::from(le16(&lh, 0x98))));
        let why = if lh.is_none() { 4 } else if recs == 0 { 2 } else if count != le32(&h, 8) { 3 } else { 0 };
        if why == 0 {
            let b = static_records((l, recs, n), frame);
            // A failed copy: this call gets no block, the next one asks again.
            keep &= b.is_some();
            Block::List { recs, b }
        } else {
            log_refusal(why, l, lh, recs, count, h, gpu);
            let grass = is_grass_list(l);
            let mode = fallback_mode(grass);
            Block::Refused { why, l, grass, mode, fb: fallback_records(mode, l, gpu, n, frame) }
        }
    } else if gpu != 0 || cpu == 0 || n == 0 {
        Block::None
    } else {
        Block::Cpu
    };
    let b = Batch { desc, h8: le64(head, 8), tp, drawable, count: le32(&h, 8), n, gpu, cpu, handler, h228, h33c, lod, block, used: frame };
    Some((b, keep))
}

/// A block into the frame's list (once per key per frame).
fn put(c: &mut Capture, key: (u64, u32), b: &Arc<Vec<u8>>, src: u8) -> u32 {
    if let Some(i) = c.inst_keys.get(&key) {
        return *i;
    }
    let i = c.inst_list.len() as u32;
    c.inst_list.push(b.clone());
    c.inst_src.push(src);
    c.inst_keys.insert(key, i);
    i
}

/// The draw's `InstInfo` and block from a batch, with the per-call words
/// (rec+0x20 near bitset, rec+0xA0 LOD mask) and the live bitsets, in the
/// order and with the counts of `capture_instances`.
fn apply(c: &mut Capture, b: &Batch, draw: u64, ctx: u64, near_p: u64, a0: u32) {
    let lmask = a0 >> 15 & 0xF;
    let first_lod = if lmask == 0 { NO_LOD } else { lmask.trailing_zeros() as u8 };
    let lodrec = match c.lod_keys.get(&b.tp) {
        Some(i) => *i,
        None => match &b.lod {
            Some(bytes) => {
                let i = c.lod_list.len() as u32;
                c.lod_list.push(*bytes);
                c.lod_keys.insert(b.tp, i);
                i
            }
            None => NO_PAL,
        },
    };
    // SAFETY: the IcDraw.
    let lod = unsafe { dread::<u32>(draw + 8) }.unwrap_or(0);
    let mask = hide_mask(c, ctx);
    let near = if near_p == 0 { NO_PAL } else { bitset_copy(c, near_p) };
    if let Some(d) = c.draws.last_mut() {
        d.inst = Some(InstInfo { desc: b.desc, count: b.count, cpu: b.cpu, gpu: b.gpu, lod, block: NO_PAL, first_lod, lodrec, mask, near, handler: b.handler, h228: b.h228, h33c: b.h33c, rec_a0: a0 });
    }
    let n = b.n;
    let block = match &b.block {
        Block::None => return,
        Block::List { recs, b: rb } => {
            c.inst_calls[1] += 1;
            match (c.inst_keys.get(&(*recs, n)), rb) {
                (Some(i), _) => *i,
                (None, Some(rb)) => put(c, (*recs, n), rb, 1),
                (None, None) => return,
            }
        }
        Block::Refused { why, l, grass, mode, fb } => {
            c.inst_calls[*why] += 1;
            let now = fallback_mode(*grass);
            let read = if now == *mode && (fb.is_some() || now == 0) { fb.clone() } else { fallback_records(now, *l, b.gpu, n, c.frame) };
            let Some((addr, rb)) = read else { return };
            c.inst_calls[5] += 1;
            put(c, (addr, n), &rb, 2)
        }
        Block::Cpu => {
            let key = (b.cpu, n);
            let block = match c.inst_keys.get(&key) {
                Some(i) => *i,
                None => {
                    if !(0x1_0000..0x8000_0000_0000).contains(&b.cpu) {
                        return;
                    }
                    // The CPU records this call draws (desc+0x78 with no GPU set).
                    let Some(bytes) = dcopy(b.cpu, 64 * n as usize) else { return };
                    c.copied += bytes.len() as u64;
                    c.inst_calls[0] += 1;
                    c.inst_calls[6] += u64::from(n);
                    let i = c.inst_list.len() as u32;
                    c.inst_list.push(Arc::new(bytes));
                    c.inst_src.push(0);
                    c.inst_keys.insert(key, i);
                    i
                }
            };
            if let Some(info) = c.draws.last_mut().and_then(|d| d.inst.as_mut()) {
                info.block = block;
            }
            return;
        }
    };
    c.inst_calls[6] += u64::from(n);
    if let Some(info) = c.draws.last_mut().and_then(|d| d.inst.as_mut()) {
        info.block = block;
    }
}

/// `capture_instances` through this thread's batch cache.
pub(super) fn capture(c: &mut Capture, draw: u64, ctx: u64) {
    if c.draws.is_empty() {
        return;
    }
    // SAFETY: the call's own arguments and what it reads (as capture_instances).
    let rec = unsafe { dread::<u64>(draw) }.unwrap_or(0);
    // IcRec +0 desc, +8 handler + 0x2C4, +0x10 the LOD record T, +0x20 the near bitset; +0xA0 bits 15-18 the LOD mask.
    // SAFETY: as above; with the two reads every page of rec's 0xA4 bytes is read.
    let Some(head) = (unsafe { dread::<[u8; 0x28]>(rec) }) else { return };
    // SAFETY: as above.
    let Some(a0) = (unsafe { dread::<u32>(rec.wrapping_add(0xA0)) }) else { return };
    let near_p = le64(&head, 0x20);
    let drawable = c.draws.last().map_or(0, |d| d.drawable);
    let frame = c.frame;
    CACHE.with(|cell| {
        let mut g = cell.borrow_mut();
        let k = g.get_or_insert_with(|| Cache::new(frame));
        if k.frame != frame {
            k.new_frame(frame);
        }
        if let Some(b) = k.map.get_mut(&rec).filter(|b| b.desc == le64(&head, 0) && b.h8 == le64(&head, 8) && b.tp == le64(&head, 0x10)) {
            b.used = frame;
            k.n[0] += 1;
            apply(c, b, draw, ctx, near_p, a0);
            return;
        }
        let stale = k.map.contains_key(&rec);
        k.n[2] += u64::from(stale);
        let Some((b, keep)) = decode(&head, frame, drawable) else {
            if stale {
                k.map.remove(&rec);
            }
            return;
        };
        k.n[1] += 1;
        apply(c, &b, draw, ctx, near_p, a0);
        if !keep {
            if stale {
                k.map.remove(&rec);
            }
            return;
        }
        let first = drawable != 0 && k.map.get(&rec).is_none_or(|p| p.drawable != drawable);
        k.map.insert(rec, b);
        if first {
            k.by_drawable.entry(drawable).or_default().push(rec);
        }
    });
}

/// Tests: [hits, decoded, stale, unloaded, aged] so far (counted at each thread's next frame).
#[cfg(test)]
pub(super) fn stats() -> [u64; 5] {
    STATS.each_ref().map(|a| a.load(Ordering::Relaxed))
}

/// "; instance cache: H hits, D decoded (S stale), U unloaded, A aged,
/// E entries" since the last call, and resets; empty when off or idle.
pub(super) fn line() -> String {
    if !wanted() {
        return String::new();
    }
    let [h, d, s, u, a] = STATS.each_ref().map(|x| x.swap(0, Ordering::Relaxed));
    if h + d == 0 {
        return String::new();
    }
    format!("; instance cache: {h} hits, {d} decoded ({s} stale), {u} unloaded, {a} aged, {} entries", ENTRIES.load(Ordering::Relaxed))
}
