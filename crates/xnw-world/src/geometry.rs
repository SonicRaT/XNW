//! The world pass's geometry budget (the world went missing).
//!
//! Cause (5a83473 code): the pass kept every geometry it ever
//! uploaded (no eviction), each with a heap slot of 16,384 and two committed
//! upload-heap buffers; once the slots ran out, `upload` returned nothing
//! and the draw went `without geometry`, while the recorder (its own cache)
//! believed the geometry delivered and never sent it again.
//!
//! Now: the pass is the authority on what it holds. [`Known`] is shared
//! with the recorder: a key is `Resident` (on the device), `Pending`
//! (bytes queued here, waiting for the budget) or `Sent` (copied by the
//! recorder at a frame, not arrived yet); the recorder copies a geometry
//! only when [`Known::wants`] says so (absent, or sent and lost for
//! [`RESEND_FRAMES`]). [`GeomPolicy`] (no device: the soak test drives it)
//! decides per frame which pending geometries upload (draws of this frame
//! first, nearest first; then the oldest), under a per-frame byte and
//! count budget, a device byte budget and the heap's slots, evicting only
//! geometries no draw used for [`GeomBudget::keep`] frames (LRU), and caps
//! the CPU bytes it queues (the dropped ones are re-sent later).

use std::collections::HashMap;
use std::sync::Mutex;

use super::{GeomKey, HostGeometry};

/// Frames after which a key sent by the recorder but never seen here is
/// wanted again (a list replaced and lost, a geometry dropped).
pub const RESEND_FRAMES: u64 = 30;

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    Sent(u64),
    Pending,
    Resident,
    /// Queued or on the device from a residency offer, its
    /// bytes copied at placement time. A draw that uses it gets the bytes
    /// copied again at draw time (the placement-time copy is not trusted:
    /// a test run lost the street with these copies drawn).
    Ahead,
}

/// What the pass holds, shared with the recorder.
#[derive(Default)]
pub struct Known {
    map: Mutex<HashMap<GeomKey, State>>,
}

impl Known {
    /// The recorder asks before copying a geometry at `frame`; `true` marks
    /// it sent (the caller copies and publishes it).
    pub fn wants(&self, key: &GeomKey, frame: u64) -> bool {
        let Ok(mut m) = self.map.lock() else { return false };
        match m.get(key) {
            Some(State::Resident | State::Pending) => false,
            // Drawn for the first time: the draw-time copy replaces it.
            Some(State::Ahead) => {
                m.insert(*key, State::Sent(frame));
                true
            }
            Some(State::Sent(f)) if frame.saturating_sub(*f) < RESEND_FRAMES && *f <= frame => false,
            _ => {
                m.insert(*key, State::Sent(frame));
                true
            }
        }
    }

    /// The recorder could not copy it (unreadable): forget the mark.
    pub fn not_sent(&self, key: &GeomKey) {
        if let Ok(mut m) = self.map.lock()
            && matches!(m.get(key), Some(State::Sent(_)))
        {
            m.remove(key);
        }
    }

    fn set(&self, key: GeomKey, s: Option<State>) {
        if let Ok(mut m) = self.map.lock() {
            match s {
                Some(s) => {
                    m.insert(key, s);
                }
                None => {
                    m.remove(&key);
                }
            }
        }
    }

    /// Keys known (any state): the recorder-side memory of this protocol.
    pub fn len(&self) -> usize {
        self.map.lock().map_or(0, |m| m.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops `Sent` marks older than `frames` (bounded map).
    pub fn sweep(&self, frame: u64, frames: u64) {
        if let Ok(mut m) = self.map.lock() {
            m.retain(|_, s| !matches!(s, State::Sent(f) if frame.saturating_sub(*f) > frames));
        }
    }
}

/// The process's [`Known`] (recorder thread and screen pass).
/// Keys whose title object was unloaded (the residency unload hook):
/// the pass drops them from its cache at its next frame.
static EVICT: Mutex<Vec<GeomKey>> = Mutex::new(Vec::new());

pub fn request_evict(keys: Vec<GeomKey>) {
    if let Ok(mut e) = EVICT.lock() {
        if e.len() < 1 << 20 {
            e.extend(keys);
        }
    }
}

pub fn take_evict() -> Vec<GeomKey> {
    EVICT.lock().map(|mut e| std::mem::take(&mut *e)).unwrap_or_default()
}

pub fn known() -> &'static Known {
    static K: std::sync::OnceLock<Known> = std::sync::OnceLock::new();
    K.get_or_init(Known::default)
}

/// Budgets (bytes are allocation sizes: 64 KiB granules on the device).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeomBudget {
    /// Device memory for geometry (default 1,024 MiB of the 8 GB card: the
    /// title's host textures take up to 4,430 MiB, `XNW_TEX_BUDGET_MB` in a test run,
    /// plus its targets and buffers).
    pub device_bytes: u64,
    /// CPU bytes queued waiting for an upload (default 256 MiB).
    pub cpu_bytes: u64,
    /// Upload bytes and geometries per frame (the first always goes).
    pub frame_bytes: u64,
    pub frame_count: u32,
    /// Heap slots for raw vertex buffers.
    pub slots: u32,
    /// Frames a geometry must go unused before it may be evicted (LRU, so
    /// the longest unused go first; 8: a camera cut can replace the set).
    pub keep: u64,
    /// The ahead pool (geometry offered from residency, not drawn yet):
    /// its device bytes at most, and per frame at most these bytes and
    /// uploads (the pass sets the count from 2 ms of measured upload cost).
    pub ahead_pool: u64,
    pub ahead_frame_bytes: u64,
    pub ahead_frame_count: u32,
}

/// Only 5% of placed geometry is ever drawn (a test run): the ahead pool's cap
/// and its per-frame upload bytes; 2 ms of uploads a frame (the pass).
pub const AHEAD_POOL: u64 = 128 << 20;
pub const AHEAD_FRAME_BYTES: u64 = 8 << 20;
pub const AHEAD_FRAME_MS: f64 = 2.0;

impl GeomBudget {
    pub const SCREEN: GeomBudget = GeomBudget { device_bytes: 1024 << 20, cpu_bytes: 256 << 20, frame_bytes: super::SCREEN_UPLOAD_BUDGET, frame_count: 2048, slots: super::GEOM_SLOTS, keep: 8, ahead_pool: AHEAD_POOL, ahead_frame_bytes: AHEAD_FRAME_BYTES, ahead_frame_count: 256 };
    pub const UNBOUNDED: GeomBudget = GeomBudget { device_bytes: u64::MAX / 4, cpu_bytes: u64::MAX / 4, frame_bytes: u64::MAX / 4, frame_count: u32::MAX, slots: super::GEOM_SLOTS, keep: 8, ahead_pool: AHEAD_POOL, ahead_frame_bytes: AHEAD_FRAME_BYTES, ahead_frame_count: 256 };
}

/// The device allocation a geometry takes: VB, IB at a 256-byte offset, in
/// 64 KiB granules (what one committed buffer costs).
/// A vertex buffer read in place (ZC2) takes no device bytes; with
/// `XNW_ZEROCOPY_BUDGET=full` it is counted as if copied, so the budgets
/// upload the same geometries as the copy path (the same-picture proof).
pub fn device_bytes_of(g: &HostGeometry) -> u64 {
    static FULL: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_ZEROCOPY_BUDGET").is_ok_and(|v| v.trim() == "full"));
    let vb = match &g.raw_vb {
        Some(_) if *FULL && g.vb_place > 0 => g.vb_place.min(super::MAX_VB_BYTES as u64),
        Some(v) => v.len().min(super::MAX_VB_BYTES) as u64,
        None => (g.vertices.len() * 4) as u64,
    };
    let ib = (g.indices.len().min(super::MAX_INDICES) * 4) as u64;
    (vb.next_multiple_of(256) + ib.max(4)).next_multiple_of(64 << 10)
}

fn cpu_bytes_of(g: &HostGeometry) -> u64 {
    (g.raw_vb.as_ref().map_or(0, Vec::len) + g.vertices.len() * 4 + g.indices.len() * 4) as u64
}

struct Res {
    bytes: u64,
    slot: bool,
    used: u64,
    /// Uploaded while no draw of the frame wanted it (ahead), not drawn since.
    ahead: bool,
    /// In the ahead pool (offered from residency, not drawn yet).
    pool: bool,
}

struct Pend {
    g: HostGeometry,
    bytes: u64,
    cpu: u64,
    since: u64,
    /// Offered from residency (no draw has asked for it).
    ahead: bool,
}

/// Why a draw had no geometry (per draw, this frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Missing {
    /// Queued here; the frame's upload budget is spent.
    Waiting,
    /// Queued here; device bytes or slots full of geometry used in the last
    /// `keep` frames.
    Full,
    /// Not here: the recorder has not delivered it (in flight, its copy
    /// failed, or it was dropped from the queue and is re-sent).
    NotSent,
}

/// Counters (cumulative, and peaks).
#[derive(Clone, Copy, Debug, Default)]
pub struct GeomStats {
    pub uploaded: u64,
    pub uploaded_bytes: u64,
    pub evicted: u64,
    pub dropped: u64,
    pub failed: u64,
    pub peak_device: u64,
    pub peak_cpu: u64,
    /// Uploaded ahead of any draw, and of those, drawn later.
    pub ahead: u64,
    pub ahead_drawn: u64,
    /// Uploaded ahead and evicted before any draw used it.
    pub ahead_evicted: u64,
    /// Dropped because the title unloaded their object.
    pub unloaded: u64,
    /// Uploads into the ahead pool (residency offers).
    pub pool_uploads: u64,
}

/// One frame's plan: what to free on the device, what to upload (owned
/// bytes: dropped after the upload, no CPU copy kept).
#[derive(Default)]
pub struct Plan {
    pub evict: Vec<GeomKey>,
    pub upload: Vec<(GeomKey, HostGeometry)>,
    pub upload_bytes: u64,
}

pub struct GeomPolicy {
    pub budget: GeomBudget,
    resident: HashMap<GeomKey, Res>,
    pending: HashMap<GeomKey, Pend>,
    pub device: u64,
    pub cpu: u64,
    slots: u32,
    /// Keys left waiting this frame for the device budget (vs the frame's).
    full: std::collections::HashSet<GeomKey>,
    /// Device bytes of the ahead pool.
    pool: u64,
    /// Resident keys delivered again: freed at the next plan.
    replaced: Vec<GeomKey>,
    pub stats: GeomStats,
}

impl GeomPolicy {
    pub fn new(budget: GeomBudget) -> GeomPolicy {
        GeomPolicy { budget, resident: HashMap::new(), pending: HashMap::new(), device: 0, cpu: 0, slots: 0, full: Default::default(), pool: 0, replaced: Vec::new(), stats: GeomStats::default() }
    }

    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn is_resident(&self, key: &GeomKey) -> bool {
        self.resident.contains_key(key)
    }

    pub fn is_pending(&self, key: &GeomKey) -> bool {
        self.pending.contains_key(key)
    }

    /// The distinct queued geometries among `keys` and their device bytes
    /// (what a fast cut must upload for the main view).
    pub fn pending_for<'a>(&self, keys: impl Iterator<Item = &'a GeomKey>) -> (usize, u64) {
        let mut seen = std::collections::HashSet::new();
        keys.filter(|k| seen.insert(**k)).filter_map(|k| self.pending.get(k)).fold((0, 0), |(n, b), p| (n + 1, b + p.bytes))
    }

    /// A queued geometry's index count (0: not queued here).
    pub fn pending_indices(&self, key: &GeomKey) -> u32 {
        self.pending.get(key).map_or(0, |p| p.g.indices.len() as u32)
    }

    /// A geometry delivered at `frame` (replaces a queued or resident one:
    /// the recorder copied it again).
    pub fn offer(&mut self, k: &Known, key: GeomKey, g: HostGeometry, frame: u64) {
        let (bytes, cpu) = (device_bytes_of(&g), cpu_bytes_of(&g));
        // Sent again while on the device (its contents changed): the old one goes.
        if let Some(r) = self.resident.remove(&key) {
            self.pool = self.pool.saturating_sub(if r.pool { r.bytes } else { 0 });
            self.device = self.device.saturating_sub(r.bytes);
            self.slots = self.slots.saturating_sub(u32::from(r.slot));
            self.replaced.push(key);
        }
        if let Some(old) = self.pending.insert(key, Pend { g, bytes, cpu, since: frame, ahead: false }) {
            self.cpu = self.cpu.saturating_sub(old.cpu);
        }
        self.cpu += cpu;
        self.stats.peak_cpu = self.stats.peak_cpu.max(self.cpu);
        k.set(key, Some(State::Pending));
    }

    /// Its title object was unloaded: off the device and out of the
    /// queue; true when it was resident (the pass frees its buffer).
    pub fn drop_key(&mut self, k: &Known, key: &GeomKey) -> bool {
        let mut resident = false;
        if let Some(r) = self.resident.remove(key) {
            self.device = self.device.saturating_sub(r.bytes);
            self.slots = self.slots.saturating_sub(u32::from(r.slot));
            self.pool = self.pool.saturating_sub(if r.pool { r.bytes } else { 0 });
            self.stats.unloaded += 1;
            resident = true;
        }
        if let Some(p) = self.pending.remove(key) {
            self.cpu = self.cpu.saturating_sub(p.cpu);
        }
        self.full.remove(key);
        k.set(*key, None);
        resident
    }

    /// A geometry offered from residency (no draw asked for it): queued
    /// behind every drawn and hook-seen geometry, uploaded into the ahead
    /// pool under its caps. Already here: nothing changes.
    pub fn offer_ahead(&mut self, k: &Known, key: GeomKey, g: HostGeometry, frame: u64) {
        if self.resident.contains_key(&key) || self.pending.contains_key(&key) {
            return;
        }
        let (bytes, cpu) = (device_bytes_of(&g), cpu_bytes_of(&g));
        self.pending.insert(key, Pend { g, bytes, cpu, since: frame, ahead: true });
        self.cpu += cpu;
        self.stats.peak_cpu = self.stats.peak_cpu.max(self.cpu);
        k.set(key, Some(State::Ahead));
    }

    /// Device bytes in the ahead pool now.
    pub fn ahead_pool_bytes(&self) -> u64 {
        self.pool
    }

    /// The pass could not make it on the device: not resident, re-sent later.
    pub fn failed(&mut self, k: &Known, key: &GeomKey) {
        if let Some(r) = self.resident.remove(key) {
            self.pool = self.pool.saturating_sub(if r.pool { r.bytes } else { 0 });
            self.device = self.device.saturating_sub(r.bytes);
            self.slots = self.slots.saturating_sub(u32::from(r.slot));
        }
        self.stats.failed += 1;
        k.set(*key, None);
    }

    /// Why `key` is not drawable now (None: it is resident).
    pub fn missing(&self, key: &GeomKey) -> Option<Missing> {
        if self.resident.contains_key(key) {
            None
        } else if self.pending.contains_key(key) {
            Some(if self.full.contains(key) { Missing::Full } else { Missing::Waiting })
        } else {
            Some(Missing::NotSent)
        }
    }

    /// The frame's plan. `refs`: this frame's drawn keys with a distance
    /// (camera-relative, cheap); they are marked used first, so nothing a
    /// draw of the last `keep` frames used is evicted.
    pub fn plan(&mut self, k: &Known, frame: u64, refs: &[(GeomKey, f32)]) -> Plan {
        let mut plan = Plan { evict: std::mem::take(&mut self.replaced), ..Default::default() };
        let mut want: HashMap<GeomKey, f32> = HashMap::new();
        for (key, d) in refs {
            if let Some(r) = self.resident.get_mut(key) {
                r.used = frame;
                if r.pool {
                    r.pool = false;
                    self.pool = self.pool.saturating_sub(r.bytes);
                }
                if r.ahead {
                    r.ahead = false;
                    self.stats.ahead_drawn += 1;
                }
            } else if self.pending.contains_key(key) {
                let e = want.entry(*key).or_insert(f32::MAX);
                *e = e.min(*d);
            }
        }
        // Order: drawn this frame, biggest on screen first (triangles over
        // squared distance, a cheap screen-coverage proxy: after a cut the big
        // things arrive first), then the rest by age.
        // Then geometry a draw hook saw (other stages) by age, then the
        // residency offers, newest first.
        let mut order: Vec<(u8, f32, u64, GeomKey)> = self.pending.iter().map(|(key, p)| match want.get(key) {
            Some(d) => (0, -(p.g.indices.len() as f32 / 3.0) / d.max(1.0), p.since, *key),
            None if p.ahead => (2, 0.0, u64::MAX - p.since, *key),
            None => (1, 0.0, p.since, *key),
        }).collect();
        order.sort_by(|a, b| (a.0, a.1, a.2).partial_cmp(&(b.0, b.1, b.2)).unwrap_or(std::cmp::Ordering::Equal));
        // Idle residents, least recently used first (made once, when needed).
        let mut idle: Option<Vec<(u64, GeomKey)>> = None;
        self.full.clear();
        let (mut n, mut bytes) = (0u32, 0u64);
        let (mut an, mut abytes) = (0u32, 0u64);
        let b = self.budget;
        for (class, _, _, key) in order {
            let Some(p) = self.pending.get(&key) else { continue };
            let slot = p.g.raw_vb.is_some();
            if n > 0 && (n >= b.frame_count || bytes + p.bytes > b.frame_bytes) {
                break; // the rest wait for the next frame
            }
            let to_pool = class == 2;
            if to_pool {
                if an >= b.ahead_frame_count || abytes + p.bytes > b.ahead_frame_bytes {
                    break; // ahead offers come last: the rest wait
                }
                // The pool's cap: its oldest undrawn entries make room.
                if self.pool + p.bytes > b.ahead_pool {
                    let mut olds: Vec<(u64, GeomKey)> = self.resident.iter().filter(|(_, r)| r.pool).map(|(k, r)| (r.used, *k)).collect();
                    olds.sort_unstable_by(|x, y| y.0.cmp(&x.0));
                    while self.pool + p.bytes > b.ahead_pool {
                        let Some((_, victim)) = olds.pop() else { break };
                        if let Some(r) = self.resident.remove(&victim) {
                            self.device = self.device.saturating_sub(r.bytes);
                            self.slots = self.slots.saturating_sub(u32::from(r.slot));
                            self.pool = self.pool.saturating_sub(r.bytes);
                            self.stats.evicted += 1;
                            self.stats.ahead_evicted += 1;
                            k.set(victim, None);
                            plan.evict.push(victim);
                        }
                    }
                    if self.pool + p.bytes > b.ahead_pool {
                        continue;
                    }
                }
            }
            // Room on the device: evict idle geometry, oldest first.
            let fits = |s: &GeomPolicy| s.device + p.bytes <= b.device_bytes && (!slot || s.slots < b.slots);
            if !fits(self) {
                let list = idle.get_or_insert_with(|| {
                    let mut v: Vec<(u64, GeomKey)> = self.resident.iter().filter(|(_, r)| frame.saturating_sub(r.used) > b.keep).map(|(k, r)| (r.used, *k)).collect();
                    v.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // pop() takes the oldest
                    v
                });
                while !fits(self) {
                    let Some((_, victim)) = list.pop() else { break };
                    if let Some(r) = self.resident.remove(&victim) {
                        self.device = self.device.saturating_sub(r.bytes);
                        self.slots = self.slots.saturating_sub(u32::from(r.slot));
                        self.pool = self.pool.saturating_sub(if r.pool { r.bytes } else { 0 });
                        self.stats.evicted += 1;
                        self.stats.ahead_evicted += u64::from(r.ahead);
                        k.set(victim, None);
                        plan.evict.push(victim);
                    }
                }
                if !fits(self) {
                    self.full.insert(key);
                    continue;
                }
            }
            let Some(p) = self.pending.remove(&key) else { continue };
            self.cpu = self.cpu.saturating_sub(p.cpu);
            self.device += p.bytes;
            self.slots += u32::from(slot);
            n += 1;
            bytes += p.bytes;
            let ahead = !want.contains_key(&key);
            self.stats.ahead += u64::from(ahead);
            if to_pool {
                (an, abytes) = (an + 1, abytes + p.bytes);
                self.pool += p.bytes;
                self.stats.pool_uploads += 1;
            }
            self.resident.insert(key, Res { bytes: p.bytes, slot, used: frame, ahead, pool: to_pool });
            k.set(key, Some(if to_pool { State::Ahead } else { State::Resident }));
            plan.upload.push((key, p.g));
        }
        plan.upload_bytes = bytes;
        self.stats.uploaded += u64::from(n);
        self.stats.uploaded_bytes += bytes;
        self.stats.peak_device = self.stats.peak_device.max(self.device);
        // The CPU queue stays under its cap: geometries no draw of this
        // frame wants go first (oldest), then the farthest; they are re-sent.
        if self.cpu > b.cpu_bytes {
            // (a test run: 10,791 dropped while 548 MiB of residency offers
            // queued): residency offers go first, then the rest by age.
            let mut drop: Vec<(u8, f32, u64, GeomKey)> = self.pending.iter().map(|(key, p)| match want.get(key) {
                Some(d) => (2, -*d, p.since, *key),
                None if p.ahead => (0, 0.0, p.since, *key),
                None => (1, 0.0, p.since, *key),
            }).collect();
            drop.sort_by(|a, b| (a.0, a.1, a.2).partial_cmp(&(b.0, b.1, b.2)).unwrap_or(std::cmp::Ordering::Equal));
            for (_, _, _, key) in drop {
                if self.cpu <= b.cpu_bytes {
                    break;
                }
                if let Some(p) = self.pending.remove(&key) {
                    self.cpu = self.cpu.saturating_sub(p.cpu);
                    self.stats.dropped += 1;
                    self.full.remove(&key);
                    k.set(key, None);
                }
            }
        }
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(bytes: usize) -> HostGeometry {
        HostGeometry { vertices: Vec::new(), indices: vec![0, 1, 2], raw_vb: Some(vec![0u8; bytes]), vb_place: 0 }
    }

    fn key(i: u64) -> GeomKey {
        GeomKey { mesh_at: 0x1000 + i, hash: i, ..Default::default() }
    }

    /// Residency offers go last, newest first, under the per-frame
    /// caps; the pool's cap evicts its oldest undrawn entries; a drawn one
    /// leaves the pool; a draw-hook offer is never displaced by one.
    #[test]
    fn ahead_pool_policy() {
        let k = Known::default();
        let b = GeomBudget { ahead_pool: 4 << 16, ahead_frame_bytes: 2 << 16, ahead_frame_count: 100, ..GeomBudget::SCREEN };
        let mut p = GeomPolicy::new(b);
        for i in 0..6 {
            p.offer_ahead(&k, key(100 + i), geom(1000), 10 + i);
        }
        p.offer(&k, key(1), geom(1000), 10); // seen by a draw hook (another stage)
        let up: Vec<GeomKey> = p.plan(&k, 20, &[]).upload.into_iter().map(|(k, _)| k).collect();
        assert_eq!(up, vec![key(1), key(105), key(104)], "hook-seen first, then the newest two (2 granules a frame)");
        assert_eq!(p.ahead_pool_bytes(), 2 << 16);
        p.plan(&k, 21, &[]);
        assert_eq!(p.ahead_pool_bytes(), 4 << 16, "at the cap");
        let plan = p.plan(&k, 22, &[]);
        assert_eq!(plan.evict.len(), 2, "the two oldest pool entries make room");
        assert!(p.ahead_pool_bytes() <= 4 << 16);
        // A drawn pool entry leaves the pool (it is ordinary geometry now).
        let before = p.ahead_pool_bytes();
        p.plan(&k, 23, &[(key(105), 1.0)]);
        assert!(p.ahead_pool_bytes() < before.max(1) || !p.is_resident(&key(105)));
        assert!(p.is_resident(&key(1)), "the hook-seen geometry was never displaced");
        // An offer of something already here changes nothing.
        p.offer_ahead(&k, key(1), geom(1000), 30);
        assert!(p.is_resident(&key(1)) && !p.is_pending(&key(1)));
        // An ahead copy is not trusted: the first draw that uses it
        // gets its bytes copied again, and that copy replaces it.
        p.offer_ahead(&k, key(200), geom(1000), 31);
        assert!(k.wants(&key(200), 32), "queued ahead: the draw hook copies it again");
        assert!(!k.wants(&key(200), 33), "then in flight");
        p.offer(&k, key(200), geom(2000), 33);
        let plan = p.plan(&k, 33, &[(key(200), 1.0)]);
        assert_eq!(plan.upload.len(), 1);
        assert_eq!(plan.upload[0].1.raw_vb.as_ref().map(Vec::len), Some(2000), "the draw-time copy");
        assert!(!k.wants(&key(200), 40), "resident from the draw hook");
    }

    #[test]
    fn an_unloaded_key_leaves_device_and_queue() {
        let k = Known::default();
        let mut p = GeomPolicy::new(GeomBudget::SCREEN);
        p.offer(&k, key(1), geom(100), 1);
        p.offer(&k, key(2), geom(100), 1);
        p.plan(&k, 1, &[(key(1), 1.0)]);
        assert!(p.is_resident(&key(1)));
        p.offer(&k, key(3), geom(100), 2);
        let dev = p.device;
        assert!(p.drop_key(&k, &key(1)), "resident: the pass frees it");
        assert!(p.device < dev && !p.is_resident(&key(1)));
        assert!(!p.drop_key(&k, &key(3)) && !p.is_pending(&key(3)), "queued: gone from the queue");
        assert!(k.wants(&key(1), 50), "asked for again if a draw needs it");
        assert_eq!(p.stats.unloaded, 1);
    }

    #[test]
    fn recorder_sends_once_until_lost() {
        let k = Known::default();
        let mut p = GeomPolicy::new(GeomBudget::SCREEN);
        assert!(k.wants(&key(1), 10));
        assert!(!k.wants(&key(1), 11), "in flight");
        assert!(k.wants(&key(1), 10 + RESEND_FRAMES), "lost: sent again");
        p.offer(&k, key(1), geom(100), 41);
        assert!(!k.wants(&key(1), 500), "queued here");
        let plan = p.plan(&k, 41, &[(key(1), 1.0)]);
        assert_eq!(plan.upload.len(), 1);
        assert!(!k.wants(&key(1), 900) && p.missing(&key(1)).is_none());
        p.failed(&k, &key(1));
        assert!(k.wants(&key(1), 901), "a failed upload is asked for again");
        assert_eq!(p.missing(&key(1)), Some(Missing::NotSent));
    }

    #[test]
    fn frame_budget_carries_over_nearest_first() {
        let k = Known::default();
        let b = GeomBudget { frame_bytes: 2 << 16, frame_count: 100, ..GeomBudget::SCREEN };
        let mut p = GeomPolicy::new(b);
        for i in 0..5 {
            p.offer(&k, key(i), geom(1000), 1);
        }
        let refs: Vec<(GeomKey, f32)> = (0..5).map(|i| (key(i), 50.0 - i as f32)).collect();
        let up: Vec<GeomKey> = p.plan(&k, 1, &refs).upload.into_iter().map(|(k, _)| k).collect();
        assert_eq!(up, vec![key(4), key(3)], "two 64 KiB granules a frame, nearest first");
        assert_eq!(p.missing(&key(0)), Some(Missing::Waiting));
        assert_eq!(p.plan(&k, 2, &refs).upload.len(), 2);
        assert_eq!(p.plan(&k, 3, &refs).upload.len(), 1);
        assert_eq!(p.pending_count(), 0);
    }

    #[test]
    fn eviction_spares_recent_use() {
        let k = Known::default();
        let b = GeomBudget { device_bytes: 3 << 16, keep: 10, ..GeomBudget::SCREEN };
        let mut p = GeomPolicy::new(b);
        for i in 0..3 {
            p.offer(&k, key(i), geom(10), 1);
        }
        p.plan(&k, 1, &[]);
        assert_eq!(p.resident_count(), 3);
        p.offer(&k, key(9), geom(10), 5);
        let plan = p.plan(&k, 5, &[(key(0), 1.0), (key(9), 1.0)]);
        assert!(plan.upload.is_empty() && plan.evict.is_empty(), "all used within keep");
        assert_eq!(p.missing(&key(9)), Some(Missing::Full));
        let plan = p.plan(&k, 20, &[(key(0), 1.0), (key(9), 1.0)]);
        assert_eq!(plan.upload.len(), 1);
        assert_eq!(plan.evict.len(), 1);
        assert!(p.is_resident(&key(0)), "used at frame 20: kept");
    }

    /// The soak a test run did by accident, in seconds: 20,000 distinct
    /// geometries (more than the 16,380 heap slots) over 30,000 frames,
    /// 2,000 live at a time (a sliding window that moves 1 a frame, plus a
    /// camera cut every 3,000 frames that replaces the whole window), sizes
    /// from a test run (most 1-64 KiB, some 0.25-4 MiB, one in 500 at 24 MiB).
    /// The recorder follows `Known::wants`; arrivals go through `offer`.
    /// Asserts: no live geometry misses for more than K frames except after
    /// a cut; device and CPU bytes under their budgets; slots recycled; no
    /// growth after warm-up.
    #[test]
    fn churn_soak_30000_frames() {
        let k = Known::default();
        let b = GeomBudget::SCREEN;
        let mut p = GeomPolicy::new(b);
        let size = |i: u64| -> usize {
            let h = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
            match h % 500 {
                0 => 24 << 20,
                1..=25 => (256 << 10) + (h as usize % (4 << 20)),
                _ => 1024 + (h as usize % (64 << 10)),
            }
        };
        let (live, frames, distinct) = (2000u64, 30_000u64, 20_000u64);
        let mut missing_since: HashMap<u64, u64> = HashMap::new();
        let (mut worst, mut worst_after_cut, mut misses) = (0u64, 0u64, 0u64);
        let mut sizes_after = Vec::new();
        let mut last_cut = 1;
        let t0 = std::time::Instant::now();
        for f in 1..=frames {
            // Window start: slides; a cut every 3,000 frames jumps it.
            let cut = f / 3000;
            if f % 3000 == 0 {
                last_cut = f;
            }
            let start = (cut * 7919 + f / 2) % (distinct - live);
            let ids = start..start + live;
            // The recorder: copies what the pass wants.
            for i in ids.clone() {
                if k.wants(&key(i), f) {
                    p.offer(&k, key(i), geom(size(i)), f);
                }
            }
            let refs: Vec<(GeomKey, f32)> = ids.clone().map(|i| (key(i), (i % 97) as f32)).collect();
            let plan = p.plan(&k, f, &refs);
            drop(plan);
            for i in ids {
                if p.is_resident(&key(i)) {
                    if let Some(s) = missing_since.remove(&i) {
                        let n = f - s;
                        if s >= last_cut + 200 { worst = worst.max(n) } else { worst_after_cut = worst_after_cut.max(n) }
                    }
                } else {
                    misses += 1;
                    missing_since.entry(i).or_insert(f);
                }
            }
            assert!(p.device <= b.device_bytes && p.cpu <= b.cpu_bytes.max(64 << 20), "frame {f}: device {} cpu {}", p.device, p.cpu);
            assert!(p.slots <= b.slots);
            k.sweep(f, 600);
            if f % 5000 == 0 {
                sizes_after.push((p.resident_count(), p.pending_count(), k.len()));
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        eprintln!("churn soak: {frames} frames in {secs:.1} s; uploaded {} ({} MiB), evicted {}, dropped {}, peak device {} MiB, peak cpu {} MiB; worst wait {worst} frames (sliding), {worst_after_cut} after a cut; {misses} draw-misses; (resident, pending, known) every 5,000: {sizes_after:?}",
            p.stats.uploaded, p.stats.uploaded_bytes >> 20, p.stats.evicted, p.stats.dropped, p.stats.peak_device >> 20, p.stats.peak_cpu >> 20);
        assert!(p.stats.uploaded > 16_384, "slots recycled past the heap size: {}", p.stats.uploaded);
        assert!(worst <= 8, "a sliding-window geometry waited {worst} frames");
        assert!(worst_after_cut <= 60, "after a cut a geometry waited {worst_after_cut} frames");
        let first = sizes_after[1];
        assert!(sizes_after.iter().skip(1).all(|s| s.0 <= first.0 + 2000 && s.2 <= first.2 + 4000), "no growth after warm-up: {sizes_after:?}");
        assert!(secs < 60.0);
    }
}
