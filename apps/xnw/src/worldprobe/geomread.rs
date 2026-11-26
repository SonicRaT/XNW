//! The list's geometry reads (`geometry reads` in the slow-frame line): per
//! geometry the pass asks for, its vertex buffer and its indices copied from
//! title memory (meshlets expanded to vertex-buffer indices, or a classic
//! index buffer widened to u32), timed in parts.
//!
//! Switches (`XNW_WORLD_RENDER`, both default off):
//! - `geomcopy=direct`: copies by the fault-fixed `rep movsb` (`dcopy`, as
//!   the hook reads) instead of ReadProcessMemory, which is a syscall per
//!   buffer that maps the pages into the kernel in pieces.
//! - `geomasync=on`: the reads run on a reader thread. The list sends it the
//!   keys the pass asks for and takes the geometries read so far; until one
//!   arrives its draws keep their rows and the pass draws nothing for them
//!   (`without geometry ... not delivered`), as for a geometry sent and not
//!   yet uploaded; a cut's coverage counts them missing and holds. A dumped
//!   frame reads in the frame (its blobs go with its list).

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock, mpsc};
use std::time::Instant;

use xnw_world::{GeomKey, HostGeometry};

use super::{Draw, NOT_READY_RETRIES, SKIP_INDICES, SKIP_NO_BUFFERS, SKIP_RETRY_CAP, SKIP_UNREADABLE, SKIP_VB_NOT_READY, bones_used, key_of};

/// Where the reads' time goes (ns), and what they read.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Parts {
    /// The pass's asks (`Known::wants`), one per draw.
    pub ask_ns: u64,
    /// Copies of title memory (vertex buffer, meshlets or index buffer).
    pub copy_ns: u64,
    /// Meshlet expansion or index widening.
    pub decode_ns: u64,
    /// Geometries whose vertex buffer was read, and the bytes copied.
    pub reads: u32,
    pub bytes: u64,
    /// `XNW_ZEROCOPY` (ZC2): vertex buffers left in title memory (only
    /// their header copied), and their bytes.
    pub in_place: u32,
    pub in_place_bytes: u64,
    /// `skinrules` bit 5 (VEH4): one-bone vehicle parts given index and
    /// weight streams (`one_bone_streams`), and one-bone parts left as they
    /// were because their vertex buffer is drawn in place.
    pub one_bone: u32,
    pub one_bone_in_place: u32,
}

impl Parts {
    fn add(&mut self, o: &Parts) {
        self.ask_ns += o.ask_ns;
        self.copy_ns += o.copy_ns;
        self.decode_ns += o.decode_ns;
        self.reads += o.reads;
        self.bytes += o.bytes;
        self.in_place += o.in_place;
        self.in_place_bytes += o.in_place_bytes;
        self.one_bone += o.one_bone;
        self.one_bone_in_place += o.one_bone_in_place;
    }

    /// `asks A ms, copies C ms (N reads, K KiB), decode D ms`.
    pub fn line(&self) -> String {
        let ms = |ns: u64| ns as f64 / 1e6;
        let zc = if self.in_place > 0 { format!(" (vertex buffers in place {}, {} KiB not copied)", self.in_place, self.in_place_bytes >> 10) } else { String::new() };
        let ob = if self.one_bone + self.one_bone_in_place > 0 { format!(", one-bone parts {} (in place, not skinned {})", self.one_bone, self.one_bone_in_place) } else { String::new() };
        format!("asks {:.1} ms, copies {:.1} ms ({} reads, {} KiB{zc}), decode {:.1} ms{ob}", ms(self.ask_ns), ms(self.copy_ns), self.reads, self.bytes >> 10, ms(self.decode_ns))
    }
}

fn since(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

/// `geomcopy=direct` (any value but `off`) and `geomasync=on` (read once).
pub(super) fn direct() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| super::render_switch("geomcopy", false))
}

pub(super) fn asynchronous() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| super::render_switch("geomasync", false))
}

/// `len` bytes of title memory at `at` (None: unreadable).
fn copy(at: u64, len: usize, direct: bool) -> Option<Vec<u8>> {
    if direct { super::dcopy(at, len) } else { super::bytes(at, len) }
}

/// One geometry: its vertex buffer, then (unless `capped`: the frame's
/// not-ready reads used up) its indices. Err is the SKIP_* reason.
pub(super) fn read(key: &GeomKey, direct: bool, capped: bool, p: &mut Parts) -> Result<HostGeometry, u8> {
    let t = Instant::now();
    let len = key.vb_len.min(64 << 20);
    // `XNW_ZEROCOPY` (ZC2): a vertex buffer whose page window is open is
    // drawn in place; only its header is copied (`vb_ready`, the decoder's
    // stream words). An address lookup, no copy of the rest.
    let place = (xnw_gpu::shared::world_part(xnw_gpu::shared::WORLD_GEOM) && len > VB_HEADER && xnw_gpu::shared::world_ready(key.vb_at, len & !3, 16)).then_some(len & !3);
    let vb = copy(key.vb_at, if place.is_some() { VB_HEADER as usize } else { len as usize }, direct);
    if let Some(n) = place {
        p.in_place += 1;
        p.in_place_bytes += n;
    }
    p.copy_ns += since(t);
    p.reads += 1;
    p.bytes += vb.as_ref().map_or(0, Vec::len) as u64;
    // A vertex buffer the title has not filled yet [V] (no stream table) is
    // not taken: asked for again on a later frame.
    if vb.as_deref().is_some_and(|b| !super::vb_ready(b)) {
        return Err(SKIP_VB_NOT_READY);
    }
    if capped {
        return Err(SKIP_RETRY_CAP);
    }
    let mut vb = vb.ok_or(SKIP_UNREADABLE)?;
    let t = Instant::now();
    let (blob, elem) = if key.mesh_len != 0 {
        (copy(key.mesh_at, key.mesh_len.min(16 << 20) as usize, direct), 0)
    } else {
        let (n, elem) = ((key.ib >> 8).min(3 << 20) as usize, (key.ib & 0xFF) as usize);
        (copy(key.mesh_at, n * elem, direct), elem)
    };
    p.copy_ns += since(t);
    let blob = blob.ok_or(SKIP_INDICES)?;
    p.bytes += blob.len() as u64;
    let t = Instant::now();
    let indices = match elem {
        0 => super::expand_meshlets(&blob).ok(),
        2 => Some(blob.chunks_exact(2).map(|c| u32::from(u16::from_le_bytes([c[0], c[1]]))).collect()),
        4 => Some(blob.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()),
        _ => None,
    };
    let indices = indices.ok_or(SKIP_INDICES)?;
    // `skinrules` bit 5: a one-bone vehicle part (MSR programs: one bone per
    // vertex or per mesh, no weights) gets ordinary index and weight streams,
    // so the pass poses it with its bone (an open door) instead of drawing
    // the bind pose (the closed door). The program is not known here: the
    // data rule picks exactly those parts in the captures (VEH4).
    if elem == 0 && super::skinhook::rules() & super::skinhook::VEH_ONE_BONE != 0 {
        one_bone(&mut vb, &blob, &indices, place.is_some(), p);
    }
    p.decode_ns += since(t);
    Ok(HostGeometry { vertices: Vec::new(), indices, raw_vb: Some(vb), vb_place: place.unwrap_or(0) })
}

/// `skinrules` bit 5 on one meshlet geometry: index and weight streams
/// appended to its vertex buffer copy when the data rule says it is a
/// one-bone part. `in_place`: only the header was copied (the title's bytes
/// are bound): nothing appended, counted (whole-mesh form only: the array
/// form's bounds need the buffer).
fn one_bone(vb: &mut Vec<u8>, blob: &[u8], indices: &[u32], in_place: bool, p: &mut Parts) {
    let Some(w0) = blob.get(0..4).map(|b| u32::from_le_bytes(b.try_into().unwrap())) else { return };
    let verts = indices.iter().max().map_or(0, |m| m + 1);
    if in_place {
        let n = u32::from(w0 >> 29 & 3 == 3 && xnw_frame::audit::one_bone_data_rule(vb, w0, 0));
        p.one_bone_in_place += n;
        super::skinhook::ONE_BONE[1].fetch_add(u64::from(n), std::sync::atomic::Ordering::Relaxed);
    } else if xnw_frame::audit::one_bone_data_rule(vb, w0, verts) && xnw_frame::audit::one_bone_streams(vb, w0, verts) {
        p.one_bone += 1;
        super::skinhook::ONE_BONE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The vertex header the reader copies of a vertex buffer read in place.
const VB_HEADER: u64 = 0x80;

/// What one frame's geometry step did.
#[derive(Default)]
pub(super) struct Taken {
    pub parts: Parts,
    /// Bytes handed to the pass (vertex bytes + 4 per index).
    pub copied: u64,
    pub failed: u32,
    pub not_ready: u32,
    /// Asynchronous: keys sent to the reader, geometries taken from it.
    pub sent: u32,
    pub arrived: u32,
    /// Frames from a key's send to its arrival, at most.
    pub waited: u64,
}

fn note_bones(skinlog: bool, key: &GeomKey, g: &HostGeometry) {
    if skinlog
        && g.vb_place == 0
        && let Some(vb) = &g.raw_vb
        && let Ok(mut m) = super::BONES.lock()
    {
        m.get_or_insert_with(HashMap::new).insert(*key, bones_used(vb, &g.indices));
    }
}

/// The frame's geometry step: the pass is asked per draw whether it lacks the
/// geometry (it holds, has queued or was just sent a geometry it does not
/// want again); a wanted one is read (in the frame, or by the reader
/// thread when `asynch`) and goes with the list; one that cannot be read is
/// asked for again later. `geom_why` gets the reason a key is not cached.
pub(super) fn take(
    draws: &[Draw],
    frame: u64,
    cache: &mut xnw_world::GeometryCache<()>,
    new_geoms: &mut Vec<(GeomKey, HostGeometry)>,
    geom_why: &mut HashMap<GeomKey, u8>,
    skinlog: bool,
    asynch: bool,
    direct: bool,
) -> Taken {
    let known = xnw_world::geometry::known();
    let mut t = Taken::default();
    let mut deliver = |key: GeomKey, g: HostGeometry, cache: &mut xnw_world::GeometryCache<()>, t: &mut Taken| {
        t.copied += (g.raw_vb.as_ref().map_or(0, Vec::len) + 4 * g.indices.len()) as u64;
        cache.get_or_make(key, frame, || Some(()));
        new_geoms.push((key, g));
    };
    if !asynch {
        for d in draws {
            let key = key_of(d);
            if key.vb_at == 0 || key.mesh_at == 0 {
                geom_why.insert(key, SKIP_NO_BUFFERS);
                continue;
            }
            let ta = Instant::now();
            let wants = known.wants(&key, frame);
            t.parts.ask_ns += since(ta);
            if !wants {
                cache.get_or_make(key, frame, || Some(()));
                continue;
            }
            match read(&key, direct, t.not_ready >= NOT_READY_RETRIES, &mut t.parts) {
                Ok(g) => {
                    note_bones(skinlog, &key, &g);
                    deliver(key, g, cache, &mut t);
                }
                Err(why) => {
                    match why {
                        SKIP_VB_NOT_READY => t.not_ready += 1,
                        SKIP_RETRY_CAP => {}
                        _ => t.failed += 1,
                    }
                    geom_why.insert(key, why);
                    known.not_sent(&key);
                }
            }
        }
        return t;
    }
    let r = reader();
    let Ok(mut flight) = r.in_flight.lock() else { return t };
    // The reader's results so far (in the order read).
    if let Ok(done) = r.done.lock() {
        while let Ok(d) = done.try_recv() {
            flight.remove(&d.key);
            t.waited = t.waited.max(frame.saturating_sub(d.frame));
            match d.got {
                Ok(g) => {
                    t.arrived += 1;
                    deliver(d.key, g, cache, &mut t);
                }
                Err(SKIP_VB_NOT_READY) => t.not_ready += 1,
                Err(SKIP_RETRY_CAP) => {}
                Err(_) => t.failed += 1,
            }
        }
    }
    let mut batch = Vec::new();
    for d in draws {
        let key = key_of(d);
        if key.vb_at == 0 || key.mesh_at == 0 {
            geom_why.insert(key, SKIP_NO_BUFFERS);
            continue;
        }
        // A key being read keeps its rows: the pass counts their geometry
        // missing (not delivered), so a cut's coverage and hold see them, as
        // for a geometry sent in the frame and not yet arrived.
        if flight.contains(&key) {
            cache.get_or_make(key, frame, || Some(()));
            continue;
        }
        let ta = Instant::now();
        let wants = known.wants(&key, frame);
        t.parts.ask_ns += since(ta);
        if !wants {
            cache.get_or_make(key, frame, || Some(()));
            continue;
        }
        flight.insert(key);
        cache.get_or_make(key, frame, || Some(()));
        batch.push(key);
    }
    t.sent = batch.len() as u32;
    if !batch.is_empty() && r.jobs.lock().map(|tx| tx.send(Job { frame, direct, skinlog, keys: batch })).is_err() {
        xnw_core::warn!("world render: geometry reader gone");
    }
    t
}

struct Job {
    frame: u64,
    direct: bool,
    skinlog: bool,
    keys: Vec<GeomKey>,
}

struct Done {
    key: GeomKey,
    frame: u64,
    got: Result<HostGeometry, u8>,
}

struct Reader {
    jobs: Mutex<mpsc::Sender<Job>>,
    done: Mutex<mpsc::Receiver<Done>>,
    /// Keys sent and not yet taken back (the list thread's own).
    in_flight: Mutex<HashSet<GeomKey>>,
}

/// The reader's parts since the last [`stats`].
static STATS: Mutex<Parts> = Mutex::new(Parts { ask_ns: 0, copy_ns: 0, decode_ns: 0, reads: 0, bytes: 0, in_place: 0, in_place_bytes: 0, one_bone: 0, one_bone_in_place: 0 });

/// The reader's parts since the last call, and the keys in flight.
pub(super) fn stats() -> (Parts, usize) {
    let p = STATS.lock().map(|mut s| std::mem::take(&mut *s)).unwrap_or_default();
    (p, READER.get().and_then(|r| r.in_flight.lock().ok().map(|f| f.len())).unwrap_or(0))
}

static READER: OnceLock<Reader> = OnceLock::new();

/// The process's reader thread (started at the first use).
fn reader() -> &'static Reader {
    READER.get_or_init(|| {
        let (jobs, rx) = mpsc::channel::<Job>();
        let (tx, done) = mpsc::channel::<Done>();
        let spawned = std::thread::Builder::new().name("xnw-geom-reader".into()).spawn(move || {
            while let Ok(job) = rx.recv() {
                let mut p = Parts::default();
                let mut not_ready = 0u32;
                for key in job.keys {
                    let got = read(&key, job.direct, not_ready >= NOT_READY_RETRIES, &mut p);
                    match &got {
                        Ok(g) => note_bones(job.skinlog, &key, g),
                        Err(why) => {
                            not_ready += u32::from(*why == SKIP_VB_NOT_READY);
                            xnw_world::geometry::known().not_sent(&key);
                        }
                    }
                    if tx.send(Done { key, frame: job.frame, got }).is_err() {
                        return;
                    }
                }
                if let Ok(mut s) = STATS.lock() {
                    s.add(&p);
                }
            }
        });
        if let Err(e) = spawned {
            xnw_core::warn!("world render: geometry reader not started: {e}");
        }
        Reader { jobs: Mutex::new(jobs), done: Mutex::new(done), in_flight: Mutex::new(HashSet::new()) }
    })
}

/// Meshlets (the layout `expand_meshlets` reads) holding the triangles
/// `idx` (vertex-buffer indices below 65,536): at most 128 vertices and
/// 128 triangles a meshlet. Tests and `xnw burst` (captures keep expanded
/// indices, not the title's meshlets) make title-like meshlets with it.
pub(super) fn meshlets(idx: &[u32]) -> Vec<u8> {
    let (mut heads, mut prims, mut verts) = (Vec::<[u32; 4]>::new(), Vec::<u32>::new(), Vec::<u16>::new());
    let (mut local, mut np) = (Vec::<u32>::new(), 0u32);
    let flush = |local: &mut Vec<u32>, np: &mut u32, heads: &mut Vec<[u32; 4]>, verts: &mut Vec<u16>, prims_len: usize| {
        if *np == 0 {
            return;
        }
        let (fv, fp) = (verts.len() as u32, (prims_len as u32) - *np);
        let h0 = (local.len() as u32 - 1) | (*np - 1) << 7 | (fv >> 16 & 1) << 14 | (fp >> 16 & 1) << 15;
        heads.push([h0, fv & 0xFFFF | (fp & 0xFFFF) << 16, 0, 0]);
        verts.extend(local.iter().map(|v| *v as u16));
        local.clear();
        *np = 0;
    };
    for t in idx.chunks_exact(3) {
        let fresh = t.iter().filter(|v| !local.contains(v)).count();
        if local.len() + fresh > 128 || np == 128 {
            flush(&mut local, &mut np, &mut heads, &mut verts, prims.len());
        }
        let mut w = 0u32;
        for (k, v) in t.iter().enumerate() {
            let i = local.iter().position(|x| x == v).unwrap_or_else(|| {
                local.push(*v);
                local.len() - 1
            });
            w |= (i as u32) << (7 * k);
        }
        prims.push(w);
        np += 1;
    }
    flush(&mut local, &mut np, &mut heads, &mut verts, prims.len());
    let mut b = Vec::new();
    for w in [heads.len() as u32, prims.len() as u32, 0] {
        b.extend_from_slice(&w.to_le_bytes());
    }
    heads.iter().flatten().chain(&prims).for_each(|w| b.extend_from_slice(&w.to_le_bytes()));
    verts.iter().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_bone_parts() {
        // Whole-mesh form (bone 170): streams appended, counted once.
        let mut vb = vec![0u8; 0x80];
        let (blob, idx) = (0xe154_0001u32.to_le_bytes(), [0u32, 1, 2]);
        let mut p = Parts::default();
        one_bone(&mut vb, &blob, &idx, false, &mut p);
        assert_eq!((p.one_bone, p.one_bone_in_place, vb.len()), (1, 0, 0x80 + 3 * 8));
        assert_eq!(u32::from_le_bytes(vb[0x80..0x84].try_into().unwrap()), 170);
        // Again (streams there now): untouched.
        let before = vb.clone();
        one_bone(&mut vb, &blob, &idx, false, &mut p);
        assert_eq!((p.one_bone, vb), (1, before));
        // In place: only the header, nothing appended, counted.
        let mut head = vec![0u8; 0x80];
        one_bone(&mut head, &blob, &idx, true, &mut p);
        assert_eq!((p.one_bone_in_place, head.len()), (1, 0x80));
        // Not a one-bone word (array form at offset 0): untouched.
        let mut vb = vec![0u8; 0x80];
        one_bone(&mut vb, &5u32.to_le_bytes(), &idx, false, &mut p);
        assert_eq!((p.one_bone, vb.len()), (1, 0x80));
        assert!(p.line().ends_with("one-bone parts 1 (in place, not skinned 1)"));
    }

    #[test]
    fn meshlet_encoder_round_trips() {
        let idx: Vec<u32> = (0..900u32).map(|i| (i * 7919) % 400).collect();
        assert_eq!(super::super::expand_meshlets(&meshlets(&idx)).unwrap(), idx);
    }

    /// Geometries laid out in one arena like title memory (64 KiB aligned):
    /// each key's vertex buffer, and its meshlets (or a u32 index buffer).
    struct Arena {
        mem: Vec<u8>,
        /// (vertex buffer offset and length, mesh offset and length, classic, hash, expected indices)
        geoms: Vec<(usize, usize, usize, usize, bool, u64, Vec<u32>)>,
        /// Draw rows: index into geoms.
        rows: Vec<usize>,
    }

    impl Arena {
        fn draws(&self, salt: u64) -> Vec<Draw> {
            let base = self.mem.as_ptr() as u64;
            self.rows
                .iter()
                .map(|&g| {
                    let (vo, vl, mo, ml, classic, h, ref idx) = self.geoms[g];
                    let mut d = Draw { ghash: h ^ salt, vb_at: base + vo as u64, vb_len: vl as u64, mesh_at: base + mo as u64, ..Default::default() };
                    if classic {
                        d.ib_len = (idx.len() as u64) << 8 | 4;
                    } else {
                        d.mesh_len = ml as u64;
                    }
                    d
                })
                .collect()
        }
    }

    /// A captured frame's list (list.tsv rows, blobs k-<mesh_at>-<hash>.vb/.idx).
    fn arena(fdir: &std::path::Path) -> Option<Arena> {
        let text = std::fs::read_to_string(fdir.join("list.tsv")).ok()?;
        let shared = fdir.parent()?.join("blobs");
        let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0);
        let mut at: HashMap<(u64, u64), usize> = HashMap::new();
        let (mut mem, mut geoms, mut rows) = (Vec::new(), Vec::new(), Vec::new());
        let align = |m: &mut Vec<u8>| m.resize(m.len().div_ceil(1 << 16) << 16, 0);
        for l in text.lines().skip(1) {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() < 6 {
                continue;
            }
            let (mesh_at, hash) = (hex(f[0]), hex(f[5]));
            if let Some(&g) = at.get(&(mesh_at, hash)) {
                rows.push(g);
                continue;
            }
            let stem = shared.join(format!("k-{mesh_at:x}-{hash:x}"));
            let (Ok(vb), Ok(idx)) = (std::fs::read(stem.with_extension("vb")), std::fs::read(stem.with_extension("idx"))) else { continue };
            let idx: Vec<u32> = idx.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let classic = idx.iter().any(|&i| i > 0xFFFF);
            let mesh = if classic { idx.iter().flat_map(|i| i.to_le_bytes()).collect() } else { meshlets(&idx) };
            align(&mut mem);
            let vo = mem.len();
            mem.extend_from_slice(&vb);
            align(&mut mem);
            let mo = mem.len();
            mem.extend_from_slice(&mesh);
            at.insert((mesh_at, hash), geoms.len());
            rows.push(geoms.len());
            geoms.push((vo, vb.len(), mo, mesh.len(), classic, hash, idx));
        }
        (!geoms.is_empty()).then_some(Arena { mem, geoms, rows })
    }

    fn capture() -> Option<Arena> {
        // This PC's capture, or another machine's copy (a capture of frame 819).
        let caps = std::env::var("RX_CAPS").map(|c| format!(r"{c}\w-cap\w\frame-819"));
        let dir = std::env::var("XNW_GEOM_FRAME").or(caps).unwrap_or_else(|_| r"out\w-cap\frame-819".into());
        arena(std::path::Path::new(&dir))
    }

    /// One frame's step, timed (ms), in a mode; then (asynchronous) the
    /// frames until every geometry has arrived, each timed.
    fn run(a: &Arena, salt: u64, asynch: bool, direct: bool) -> (f64, Vec<f64>, Taken, Vec<(GeomKey, HostGeometry)>) {
        let draws = a.draws(salt);
        let mut cache = xnw_world::GeometryCache::<()>::default();
        let (mut news, mut why) = (Vec::new(), HashMap::new());
        let frame = 1_000_000 + salt;
        let t = Instant::now();
        let mut taken = take(&draws, frame, &mut cache, &mut news, &mut why, false, asynch, direct);
        let first = t.elapsed().as_secs_f64() * 1e3;
        // Every draw keeps its row while its geometry is read (the pass counts it missing).
        assert!(draws.iter().all(|d| cache.get_or_make(key_of(d), frame, || None).is_some()), "rows kept");
        let mut later = Vec::new();
        let mut f = frame;
        while asynch && news.len() < taken.sent as usize && later.len() < 2000 {
            std::thread::sleep(std::time::Duration::from_millis(5));
            f += 1;
            let t = Instant::now();
            let k = take(&draws, f, &mut cache, &mut news, &mut why, false, true, direct);
            later.push(t.elapsed().as_secs_f64() * 1e3);
            taken.arrived += k.arrived;
            taken.copied += k.copied;
            taken.waited = taken.waited.max(k.waited);
        }
        // A frame after: nothing new (every key held or sent).
        let t = Instant::now();
        let k = take(&draws, f + 1, &mut cache, &mut news, &mut why, false, asynch, direct);
        later.push(t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(k.sent + k.parts.reads + k.arrived, 0, "nothing read again");
        (first, later, taken, news)
    }

    /// The split of `geometry reads` on a captured cut frame (capture frame
    /// 819), each mode on fresh keys: ReadProcessMemory against the direct
    /// copy, in the frame against the reader thread; the geometries the
    /// reader hands over equal those read in the frame, and the meshlets
    /// expand to the capture's own indices.
    #[test]
    fn geometry_reads_split_and_reader() {
        let Some(a) = capture() else {
            eprintln!("no capture (XNW_GEOM_FRAME): skipped");
            return;
        };
        let mib = a.geoms.iter().map(|g| g.1 + g.3).sum::<usize>() >> 20;
        eprintln!("capture: {} draws, {} geometries, {mib} MiB (vertex + mesh)", a.rows.len(), a.geoms.len());
        // The decode alone, on the arena's meshlets (heap memory, warm).
        let ml: Vec<&[u8]> = a.geoms.iter().filter(|g| !g.4).map(|g| &a.mem[g.2..g.2 + g.3]).collect();
        let n: usize = a.geoms.iter().map(|g| g.6.len()).sum();
        for _ in 0..2 {
            let t = Instant::now();
            let x: usize = ml.iter().map(|b| super::super::expand_meshlets(b).unwrap().len()).sum();
            let fast = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let y: usize = ml.iter().map(|b| xnw_frame::expand_meshlets_checked(b).unwrap().len()).sum();
            let checked = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(x, y);
            eprintln!("decode alone: {} meshlet geometries ({x} indices; {n} indices in all {} geometries): slices {fast:.1} ms, read by read {checked:.1} ms", ml.len(), a.geoms.len());
        }
        let mut salt = 0x5eed_0000u64;
        let mut reference: HashMap<u64, (Vec<u32>, Vec<u8>)> = HashMap::new();
        for round in 0..3 {
            for (asynch, direct, name) in [(false, false, "frame, ReadProcessMemory"), (false, true, "frame, direct"), (true, true, "reader, direct")] {
                salt += 1;
                let (first, mut later, t, news) = run(&a, salt, asynch, direct);
                let steady = later.pop().unwrap_or(0.0);
                let worst = later.iter().copied().fold(0.0, f64::max);
                eprintln!(
                    "round {round} {name}: step {first:.1} ms ({}; {} KiB to the pass; failed {}, not ready {}){}; next frame (nothing new) {steady:.2} ms",
                    t.parts.line(),
                    t.copied >> 10,
                    t.failed,
                    t.not_ready,
                    if asynch { format!("; reader: sent {}, arrived {} over {} frames, those steps worst {worst:.2} ms", t.sent, t.arrived, later.len()) } else { String::new() }
                );
                assert_eq!(news.len(), a.geoms.len(), "{name}: every geometry read");
                for (k, g) in &news {
                    let got = (g.indices.clone(), g.raw_vb.clone().unwrap());
                    match reference.get(&(k.hash ^ salt)) {
                        Some(r) => assert!(*r == got, "{name}: geometry {:#x} differs", k.hash ^ salt),
                        None => {
                            let e = a.geoms.iter().find(|e| e.5 == k.hash ^ salt).unwrap();
                            assert_eq!(got.0, e.6, "indices as captured");
                            reference.insert(k.hash ^ salt, got);
                        }
                    }
                }
            }
        }
    }
}
