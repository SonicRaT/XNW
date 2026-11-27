//! The thin draw hook (`XNW_HOOK_THIN=on`, default off): on the title's
//! thread the hook copies only what the title may change before the frame
//! ends; threads of ours work out the rest of each draw before the list is
//! built.
//!
//! Structure ported from VIctory-7900XTX `crates/victory-gpu/src/list/thin.rs`
//! (commit 3a99ee6, GPL-3.0-or-later; upstream: "a call appends its arguments
//! (and copies of what they point at) to a stream ... threads of ours run the
//! stream"). There the title's list calls write their arguments down and 8
//! threads translate them; here the geometry-draw hook writes its draw into
//! the thread's own capture (the per-thread ring of draws the hook already
//! fills) and the decode runs on a small pool at the frame's cut. Trimmed to
//! what our hook defers; no code of the list translator applies.
//!
//! What stays at the hook (the title reuses it after the draw): the TLS
//! words, the render context's words, the transform it binds (the CPU block
//! or the cache entry), the per-draw record (skin constants), the bones and
//! mask words, the viewport's camera on first sight, the draw-list entry
//! and entity, the draw's own shader object and its texVars values
//! (`worldtex::head`), the geometry's 0x48-B head. What moves here (objects
//! that live while the frame draws [I: the console's card reads them when
//! the list runs, after the cut]): the geometry's buffer objects and the
//! per-thread key cache, the views the texVars values name with their
//! residency records and descriptor slots (`worldtex::tail`), the census
//! count. The model header (skinned, a rigid part's bone index) is read once
//! per model and thread (`model`) instead of per draw.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};
use std::time::Instant;

use super::{Capture, Draw, geom_apply, geom_key};

/// [`Pending::flags`]: the geometry key, the albedo tail, the census count.
pub(super) const GEOM: u8 = 1;
pub(super) const TEX: u8 = 2;
pub(super) const CENSUS: u8 = 4;

/// What the hook left for [`decode`] (flags 0: nothing).
#[derive(Clone, Copy, Default)]
pub(super) struct Pending {
    pub(super) flags: u8,
    /// The geometry head's buffer objects (+8, +0x10, +0x40).
    pub(super) g: [u64; 3],
    /// `census=1`: the context's program set (ctx+0x12328) at the draw.
    pub(super) progs: u64,
    pub(super) tex: crate::worldtex::Head,
}

/// `XNW_HOOK_THIN=on`.
pub(super) fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNW_HOOK_THIN").is_ok_and(|v| v.trim() == "on"))
}

/// Decode threads (`XNW_HOOK_THIN_THREADS`, 1-16, default 4); the writer
/// thread takes one share itself.
pub(super) fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("XNW_HOOK_THIN_THREADS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4).clamp(1, 16))
}

/// A model's header (model+0x14 u16, +0x33 bone, +0x35 bit 0), as the hook
/// tests it per draw: skinned when +0x14 != 0 or +0x35 bit 0; else a rigid
/// part composed with bones[+0x33] (0x1401D77D0).
#[derive(Clone, Copy, Default)]
pub(super) struct Model {
    at: u64,
    pub(super) skinned: bool,
    pub(super) rigid: bool,
    pub(super) bone: u8,
}

const MODELS_N: usize = 1024;

thread_local! {
    /// Per recording thread, direct-mapped by model pointer.
    static MODELS: std::cell::RefCell<Option<Box<[Model; MODELS_N]>>> = const { std::cell::RefCell::new(None) };
}

/// The model's header, read once per model on this thread (a model's
/// header does not change while it is loaded [I: the hook's skinned cache
/// has assumed it since a test run]).
pub(super) fn model(at: u64) -> Model {
    let i = (at >> 4) as usize & (MODELS_N - 1);
    if let Some(m) = MODELS.with(|c| c.borrow().as_ref().map(|t| t[i])).filter(|m| m.at == at) {
        return m;
    }
    // SAFETY: the draw's model (TLS+0x1208), fault-fixed.
    let m = match unsafe { super::dread::<[u8; 0x38]>(at) } {
        Some(h) => {
            let skinned = super::le16(&h, 0x14) != 0 || h[0x35] & 1 != 0;
            Model { at, skinned, rigid: !skinned, bone: h[0x33] }
        }
        None => Model { at, ..Default::default() },
    };
    MODELS.with(|c| c.borrow_mut().get_or_insert_with(|| Box::new([Model::default(); MODELS_N]))[i] = m);
    m
}

/// One share's results, joined into the capture.
#[derive(Default)]
struct Part {
    geom_misses: u32,
    census: HashMap<(u64, u64, u32, u8), u32>,
    ns: u64,
}

/// The rest of one draw.
fn decode_one(d: &mut Draw, tex: &crate::worldtex::TexGlobals, p: &mut Part) {
    let f = d.thin.flags;
    if f == 0 {
        return;
    }
    if f & GEOM != 0 {
        geom_apply(d, geom_key(d.geom, d.ghash, d.thin.g, &mut p.geom_misses));
    }
    let mut why = 4;
    if f & TEX != 0 {
        (d.tex_desc, d.alpha_clip) = crate::worldtex::tail(tex, &d.thin.tex, true);
        d.layer_desc = if d.thin.tex.fx == 0 { [0; 4] } else { crate::worldtex::last_layers() };
        d.extra = crate::worldtex::last_extra();
        why = crate::worldtex::last_why();
    }
    if f & CENSUS != 0 {
        *p.census.entry((d.fx, d.thin.progs, d.stage_tls, why)).or_default() += 1;
    }
    d.thin.flags = 0;
}

fn decode_all(draws: &mut [Draw], tex: &crate::worldtex::TexGlobals) -> Part {
    let t0 = Instant::now();
    let mut p = Part::default();
    for d in draws {
        decode_one(d, tex, &mut p);
    }
    p.ns = t0.elapsed().as_nanos() as u64;
    p
}

/// A share of a capture's draws, decoded in place by a pool thread; the
/// result goes back on `done`.
struct Job {
    at: usize,
    n: usize,
    tex: crate::worldtex::TexGlobals,
    done: mpsc::Sender<Part>,
}

/// The pool: long-lived threads (their geometry and albedo caches stay warm).
fn pool() -> &'static [Mutex<mpsc::Sender<Job>>] {
    static POOL: OnceLock<Vec<Mutex<mpsc::Sender<Job>>>> = OnceLock::new();
    POOL.get_or_init(|| {
        (0..threads().saturating_sub(1))
            .filter_map(|k| {
                let (tx, rx) = mpsc::channel::<Job>();
                std::thread::Builder::new()
                    .name(format!("xnw-thin-{k}"))
                    .spawn(move || {
                        xnw_core::win::helper_thread();
                        for job in rx {
                            // SAFETY: `decode` hands out disjoint shares of its
                            // capture's draws and waits for every share's answer
                            // (or its sender's drop) before it uses them again.
                            let draws = unsafe { std::slice::from_raw_parts_mut(job.at as *mut Draw, job.n) };
                            let p = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode_all(draws, &job.tex))).unwrap_or_default();
                            let _ = job.done.send(p);
                        }
                    })
                    .ok()
                    .map(|_| Mutex::new(tx))
            })
            .collect()
    })
}

/// Since the last [`line`]: frames, draws decoded, wall and summed thread ns.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The draws of a frame the hook left thin, decoded on the pool and this
/// thread, in place, before the list build.
pub(super) fn decode(c: &mut Capture) {
    let n = c.draws.len();
    if n == 0 {
        return;
    }
    let t0 = Instant::now();
    let tex = crate::worldtex::globals();
    let pool = pool();
    // Shares of at least 256 draws (a share costs a channel round trip).
    let shares = (pool.len() + 1).min(n.div_ceil(256)).max(1);
    let per = n.div_ceil(shares);
    let (tx, rx) = mpsc::channel::<Part>();
    let mut sent = 0;
    let mut parts = Vec::with_capacity(shares);
    let base = c.draws.as_mut_ptr();
    for (k, w) in pool.iter().enumerate().take(shares - 1) {
        let from = (k + 1) * per;
        if from >= n {
            break;
        }
        let len = per.min(n - from);
        // SAFETY: a share inside the vector, disjoint from the others and from ours.
        let job = Job { at: unsafe { base.add(from) } as usize, n: len, tex, done: tx.clone() };
        if w.lock().is_ok_and(|w| w.send(job).is_ok()) {
            sent += 1;
        } else {
            // That thread is gone: its share here.
            // SAFETY: as above; nobody else holds it.
            let share = unsafe { std::slice::from_raw_parts_mut(base.add(from), len) };
            parts.push(decode_all(share, &tex));
        }
    }
    drop(tx);
    // SAFETY: the first share, disjoint from those sent.
    let mine = unsafe { std::slice::from_raw_parts_mut(base, per.min(n)) };
    parts.push(decode_all(mine, &tex));
    // Every share answers (or its thread dropped the sender) before the draws are used again.
    while sent > 0 {
        match rx.recv() {
            Ok(p) => parts.push(p),
            Err(_) => break,
        }
        sent -= 1;
    }
    let mut cpu = 0;
    for p in parts {
        cpu += p.ns;
        join(c, p);
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(n as u64, Ordering::Relaxed);
    STATS[2].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    STATS[3].fetch_add(cpu, Ordering::Relaxed);
}

fn join(c: &mut Capture, p: Part) {
    c.geom_misses += p.geom_misses;
    for (k, v) in p.census {
        *c.census.entry(k).or_default() += v;
    }
}

/// "; thin decode (N frames): D draws, wall X ms a frame, threads Y ms a
/// frame (Z us a draw)", and resets; empty when nothing was decoded.
pub(super) fn line() -> String {
    let [f, n, wall, cpu] = STATS.each_ref().map(|a| a.swap(0, Ordering::Relaxed));
    if f == 0 {
        return String::new();
    }
    format!(
        "; thin decode ({f} frames, {} threads): {n} draws, {:.2} ms a frame wall, {:.2} ms a frame on the threads ({:.2} us a draw)",
        threads(),
        wall as f64 / 1e6 / f as f64,
        cpu as f64 / 1e6 / f as f64,
        cpu as f64 / 1e3 / n.max(1) as f64
    )
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    /// Made-up title memory for the hook: per thread a TLS block, a render
    /// context and a viewport; per draw its own shader object and texVars,
    /// CPU matrix block and per-draw record; shared effects, views (plain
    /// and through a handle record), models (skinned and rigid), meshlet and
    /// classic geometries with their buffer objects. Each object sits on its
    /// own page and line of one arena (`stride`), so with an arena past the
    /// cache every first touch in a frame misses as in the title.
    struct Fake {
        arena: Vec<u64>,
        next: usize,
        stride: usize,
        threads: Vec<[u64; 3]>,
        shaders: Vec<u64>,
        models: Vec<u64>,
        geoms: Vec<u64>,
        blocks: Vec<u64>,
        palette: u64,
        icdraw: u64,
        g: Globals,
        view_vtbl: u64,
        /// `add_batches`: many instanced calls (in place of the one `icdraw`)
        /// and the hide bitsets the contexts bind (ctx+0xA10).
        batches: Vec<FakeBatch>,
        hides: Vec<u64>,
    }

    /// One instanced call of `add_batches`: IcDraw -> IcRec -> desc, its
    /// records (kind S through a list L, else kind C), and a second desc.
    #[derive(Clone, Copy)]
    struct FakeBatch {
        icdraw: u64,
        rec: u64,
        desc: u64,
        alt: u64,
        records: u64,
        n: u32,
        kind_s: bool,
    }

    /// worldtex's globals, set once for the process: the handle table is a
    /// qword of ours; the view vtable value follows from it.
    fn tex_globals() -> crate::worldtex::TexGlobals {
        static SET: std::sync::Once = std::sync::Once::new();
        SET.call_once(|| {
            let handles: &'static mut u64 = Box::leak(Box::new(0));
            crate::worldtex::set_globals((handles as *mut u64 as usize).wrapping_sub(0x1_49C3_E108 - 0x1_4000_0000));
        });
        crate::worldtex::globals()
    }

    impl Fake {
        fn alloc(&mut self, bytes: usize) -> u64 {
            let at = self.next;
            self.next += bytes.div_ceil(self.stride) * self.stride;
            assert!(self.next <= self.arena.len() * 8, "arena full");
            self.arena.as_ptr() as u64 + at as u64
        }
        fn put<T: Copy>(&mut self, at: u64, v: T) {
            let base = self.arena.as_ptr() as u64;
            assert!(at >= base && at + std::mem::size_of::<T>() as u64 <= base + 8 * self.arena.len() as u64);
            // SAFETY: inside the arena (checked).
            unsafe { (at as *mut T).write_unaligned(v) };
        }

        /// `threads` recording threads, `shaders` per-draw shader objects
        /// (as many CPU matrix blocks), `geoms` geometries.
        fn new(mb: usize, stride: usize, threads: usize, shaders: usize, geoms: usize, salt: u64) -> Fake {
            let tex = tex_globals();
            let mut f = Fake { arena: vec![0u64; mb << 17], next: 0, stride, threads: Vec::new(), shaders: Vec::new(), models: Vec::new(), geoms: Vec::new(), blocks: Vec::new(), palette: 0, icdraw: 0, g: Globals::default(), view_vtbl: tex.view_vtbl, batches: Vec::new(), hides: Vec::new() };
            // Title globals: slot word, transform cache, per-draw records, palette.
            let slot_word = f.alloc(8);
            f.put(slot_word, 5u16 << 2);
            let cache = f.alloc(4096 * 128);
            for i in 0..4096 * 32 {
                f.put(cache + 4 * i, if i % 5 == 0 { 1.0f32 } else { 0.25 });
            }
            let skin_cb = f.alloc(4096 * 128 + 0x80);
            let palette = f.alloc(48 * 4096);
            for i in 0..4096 * 12 {
                f.put(palette + 4 * i, (i % 7) as f32 * 0.5);
            }
            let pal_base = f.alloc(16);
            f.put(pal_base, palette);
            f.palette = palette;
            f.g = Globals { matrix_slot: slot_word, matrix_cache: cache, skin_cb, pal_base, pal_size: 0, arena: 0 };
            for i in 0..4096u64 {
                // {curIndex, prevIndex, curCount, prevCount}, then +0x74/+0x78 remap bases (-1).
                let r = skin_cb + 128 * i + 0x30;
                f.put(r, (i % 64) as u32 * 32);
                f.put(r + 8, if i % 3 == 0 { 24u32 } else { 0 });
                f.put(r + 0x44, u32::MAX);
                f.put(r + 0x48, u32::MAX);
            }
            // Handle records: table at the handles qword.
            let table = f.alloc(16 * 64);
            // SAFETY: tex.handles is our leaked qword (tex_globals).
            unsafe { (tex.handles as *mut u64).write(table) };
            // Views (plain), and objects behind handle records.
            let views: Vec<u64> = (0..512u64).map(|k| {
                let v = f.alloc(32);
                f.put(v, if k % 17 == 5 { 0x1234 } else { f.view_vtbl });
                f.put(v + 0x18, 0x5_0000_0000 + 0x20 * k);
                v
            }).collect();
            let handles: Vec<u64> = (0..64u64).map(|k| {
                let obj = f.alloc(0x30);
                f.put(obj + 0x28, views[k as usize * 7 % 512]);
                f.put(table + 16 * k, obj);
                f.put(table + 16 * k + 12, if k % 5 == 0 { 0u8 } else { 2 });
                f.put(table + 16 * k + 14, 1u8);
                let h = f.alloc(8);
                f.put(h + 6, k as u16);
                h
            }).collect();
            // Effects: names (cut-out, layered, plain), templates with variables.
            let names = ["normal_spec", "ped_standard_cutout", "standard_uber_4lyr_spx", "trees_lod", "default"];
            let effects: Vec<u64> = (0..40usize).map(|k| {
                let name = f.alloc(48);
                for (i, b) in names[k % names.len()].bytes().enumerate() {
                    f.put(name + i as u64, b);
                }
                // Variables: a constant, the albedo (or a layer) at texVars[2], a second layer at [3].
                let vars = f.alloc(48);
                f.put(vars, 0x1111_1111u32);
                f.put(vars + 12, 3u8);
                let hash = match k % names.len() { 2 => 0x3463_94EAu32, 3 => 0xD7C3_8241, 4 => 0x17D3_B5C2, _ => 0xC8E8_C282 };
                f.put(vars + 20, hash);
                f.put(vars + 24, 2u16);
                f.put(vars + 32, 0x7D4F_12E2u32);
                f.put(vars + 40, 3u16);
                let template = f.alloc(0x58);
                f.put(template + 4, 3u8);
                f.put(template + 0x50, vars);
                let fx = f.alloc(0xD10);
                f.put(fx, template);
                f.put(fx + 0xD08, name);
                fx
            }).collect();
            for k in 0..shaders {
                let tv = f.alloc(32);
                let pick = |i: usize| if i % 4 == 3 { 1 << 63 | handles[i % 64] } else { views[i % 512] };
                f.put(tv + 16, pick(k * 13 + 1));
                f.put(tv + 24, pick(k * 29 + 2));
                let sh = f.alloc(24);
                f.put(sh, effects[k * 7 % effects.len()]);
                f.put(sh + 16, tv);
                f.shaders.push(sh);
                let m = f.alloc(128);
                for i in 0..32 {
                    f.put(m + 4 * i, (k + i as usize) as f32 * 0.125);
                }
                f.blocks.push(m);
            }
            // Models: skinned (+0x14), skinned by bit, rigid with a bone.
            for k in 0..256u64 {
                let m = f.alloc(0x38);
                match k % 3 {
                    0 => f.put(m + 0x14, 3u16),
                    1 => f.put(m + 0x35, 1u8),
                    _ => f.put(m + 0x33, (k % 40) as u8),
                }
                f.models.push(m);
            }
            // Geometries: meshlets (buffer object behind a pointer), or a classic index buffer.
            for k in 0..geoms as u64 {
                let grc = |f: &mut Fake, n: u32, elem: u16| {
                    let o = f.alloc(0x20);
                    f.put(o + 8, n);
                    f.put(o + 0xC, elem);
                    let data = 0x7_0000_0000 + 0x1000 * k;
                    f.put(o + 0x18, data);
                    o
                };
                let g = f.alloc(0x48);
                let vb = grc(&mut f, 64 + k as u32, 16);
                f.put(g + 8, vb);
                f.put(g + 0x30, salt ^ (k + 1));
                if k % 4 == 0 {
                    let ib = grc(&mut f, 300, 2);
                    f.put(g + 0x10, ib);
                } else {
                    let ml = grc(&mut f, 12, 16);
                    let p = f.alloc(8);
                    f.put(p, ml);
                    f.put(g + 0x40, p);
                }
                f.geoms.push(g);
            }
            // One instanced call (IcDraw -> IcRec -> desc, 64 CPU records, a LOD record).
            let records = f.alloc(64 * 64);
            let lodrec = f.alloc(140);
            let desc = f.alloc(0x90);
            f.put(desc + 8, 64u32);
            f.put(desc + 0x78, records);
            let rec = f.alloc(0xA4);
            f.put(rec, desc);
            f.put(rec + 0x10, lodrec);
            f.put(rec + 0xA0, 1u32 << 15);
            f.icdraw = f.alloc(16);
            f.put(f.icdraw, rec);
            for _ in 0..threads {
                let (tls, ctx, vp) = (f.alloc(0x1400), f.alloc(0x1_3000), f.alloc(0x800));
                f.put(tls + TLS_CTX, ctx);
                f.put(tls + TLS_VIEWPORT, vp);
                f.put(vp + 0x40, 1.0f32);
                f.put(vp + 0x708, [0.5f32, -0.5]);
                f.threads.push([tls, ctx, vp]);
            }
            f
        }

        /// Writes draw `i`'s state into thread `t`'s TLS block and context
        /// (as the title does before the call) and returns its geometry.
        fn set(&mut self, t: usize, i: usize, frame: u64) -> u64 {
            let [tls, ctx, _] = self.threads[t];
            let k = (i + 977 * frame as usize) % self.shaders.len();
            let m = self.models[i % self.models.len()];
            self.put(tls + TLS_SHADER, if i % 23 == 0 { 0 } else { self.shaders[k] });
            self.put(tls + TLS_MODEL, if i % 11 == 0 { 0 } else { m });
            self.put(tls + TLS_PALETTE, self.palette + 48 * (i as u64 % 64) * 32);
            self.put(tls + TLS_PALETTE_N, 64u32);
            self.put(tls + TLS_DRAWABLE, 0x1234_0000 + i as u64);
            self.put(tls + TLS_STAGE, 60 + (i % 20) as u32);
            self.put(tls + TLS_CB_A, (i % 4000) as u32);
            self.put(tls + TLS_CB_B, (i % 3) as u8);
            // The transform: a CPU block (lazy 0x80) for most draws, else the cache by index.
            self.put(ctx + CTX_SLOTS + 8 * 5, self.blocks[k]);
            self.put(ctx + CTX_SRV_LAZY + 10, if i % 5 == 0 { 0u16 } else { 0x80 });
            self.put(ctx + CTX_MATRIX_INDEX, (i % 4096) as i32);
            self.put(ctx + CTX_PROGRAMS, 0xABC0 + (i % 9) as u64);
            self.put(ctx + 0x1_2360, u64::MAX - (i % 2) as u64);
            self.put(ctx + 0x1_2368, (i % 7) as u16);
            self.put(ctx + 0x1_2378, (i % 13) as u16);
            if !self.hides.is_empty() {
                self.put(ctx + 0xA10, if i % 5 == 0 { 0 } else { self.hides[i % self.hides.len()] });
            }
            self.geoms[(i * 7 + frame as usize) % self.geoms.len()]
        }

        /// Draw `g`'s instanced call (every third draw has one).
        fn icdraw_of(&self, g: usize) -> u64 {
            if self.batches.is_empty() { self.icdraw } else { self.batches[g / 3 % self.batches.len()].icdraw }
        }

        /// A bitset object {bits, words +8, bits +0xC} of `words` words.
        fn bitset(&mut self, words: u32, seed: u32) -> u64 {
            let bits = self.alloc(4 * words as usize);
            for w in 0..words {
                self.put(bits + 4 * u64::from(w), (seed.wrapping_mul(0x9E37_79B9) ^ w.wrapping_mul(0x85EB_CA6B)) & 0x0F0F_F0F0);
            }
            let p = self.alloc(16);
            self.put(p, bits);
            self.put(p + 8, words);
            self.put(p + 0xC, 32 * words);
            p
        }

        /// `k` instanced calls of 1..=`max` instances: kind C every
        /// `cpu_every`-th (b % cpu_every == 1), else kind S (accepted, or
        /// refused for a count that differs), with their LOD records,
        /// handlers, near bitsets on every third; 8 hide bitsets.
        fn add_batches(&mut self, k: usize, max: u32, cpu_every: usize) {
            for b in 0..k {
                let n = 1 + (b as u32 * 7) % max;
                let records = self.alloc(64 * n as usize);
                for i in 0..16 * u64::from(n) {
                    self.put(records + 4 * i, ((b as u64 * 31 + i) % 97) as f32 * 0.5);
                }
                let lodrec = self.alloc(140);
                for i in 0..35u64 {
                    self.put(lodrec + 4 * i, (b as u64 + i) as f32);
                }
                let desc = self.alloc(0x90);
                self.put(desc + 8, n);
                let kind_s = b % cpu_every != 1;
                if kind_s {
                    let l = self.alloc(0xA0);
                    self.put(l + 0x90, records);
                    self.put(l + 0x98, if b % 10 == 4 { n as u16 + 1 } else { n as u16 });
                    self.put(desc, l);
                    self.put(desc + 0x68, 0x5555_0000u64 + b as u64);
                } else {
                    self.put(desc + 0x78, records);
                }
                // A second desc over the same records (kind C, one fewer).
                let alt = self.alloc(0x90);
                self.put(alt + 8, n.max(2) - 1);
                self.put(alt + 0x78, records);
                let handler = self.alloc(0x340);
                self.put(handler + 0x228, b as u32);
                self.put(handler + 0x33C, (b % 7) as u16);
                let rec = self.alloc(0xA4);
                self.put(rec, desc);
                self.put(rec + 8, handler + 0x2C4);
                self.put(rec + 0x10, lodrec);
                self.put(rec + 0xA0, 1u32 << (15 + b % 4) | 7);
                if b % 3 == 0 {
                    let p = self.bitset(n.div_ceil(32), b as u32);
                    self.put(rec + 0x20, p);
                }
                let icdraw = self.alloc(16);
                self.put(icdraw, rec);
                self.put(icdraw + 8, (b % 4) as u32);
                self.batches.push(FakeBatch { icdraw, rec, desc, alt, records, n, kind_s });
            }
            for h in 0..8u32 {
                let p = self.bitset(2, 1000 + h);
                self.hides.push(p);
            }
        }
    }

    fn state(f: &Fake, thin: bool, census: bool) -> State {
        state_with(f, thin, census, false)
    }

    fn state_with(f: &Fake, thin: bool, census: bool, cache: bool) -> State {
        let (tx, _rx) = mpsc::channel::<Capture>();
        let cfg = Config { dir: std::env::temp_dir(), frames: 3, max: 100_000, at: Vec::new(), raw: 0, render: true, seq: false, skinlog: 0, census, auto: None, auto_veh: 0, cut_present: false, pack: false, scan_exit: false, hookin: false, clock: Vec::new(), clock_frames: 1 };
        State { cfg, title: 0, tls_index: 0, g: f.g, base: AtomicU64::new(0), wait_world: false, done: AtomicBool::new(false), pending: AtomicU64::new(u64::MAX), slots: Mutex::new(Vec::new()), sent: AtomicU64::new(u64::MAX), late: [AtomicU64::new(0), AtomicU64::new(0)], tx: Mutex::new(tx), thin, cache }
    }

    /// The title's recording threads: long-lived (their caches stay warm
    /// from frame to frame, as the title's do), one per TLS block of `f`.
    struct Title<'a> {
        go: Vec<mpsc::Sender<u64>>,
        done: mpsc::Receiver<()>,
        s: &'a State,
    }

    impl<'a> Title<'a> {
        fn start<'s>(sc: &'s std::thread::Scope<'s, 'a>, f: &'a Mutex<Fake>, s: &'a State, threads: usize, draws: usize) -> Title<'a> {
            Title::start_with(sc, f, s, threads, draws, true)
        }

        /// `inst` false: the same draws without their instanced calls (a baseline).
        fn start_with<'s>(sc: &'s std::thread::Scope<'s, 'a>, f: &'a Mutex<Fake>, s: &'a State, threads: usize, draws: usize, inst: bool) -> Title<'a> {
            let (dtx, done) = mpsc::channel();
            let go = (0..threads)
                .map(|t| {
                    let (tx, rx) = mpsc::channel::<u64>();
                    let dtx = dtx.clone();
                    sc.spawn(move || {
                        for now in rx {
                            for i in 0..draws {
                                let (geom, tls, ic) = {
                                    let mut f = f.lock().unwrap();
                                    (f.set(t, t * draws + i, now), f.threads[t][0], f.icdraw_of(t * draws + i))
                                };
                                hook_with(s, now, geom, 0x1_4000_1000 + i as u64, (inst && i % 3 == 0).then_some(ic), tls);
                            }
                            let _ = dtx.send(());
                        }
                    });
                    tx
                })
                .collect();
            Title { go, done, s }
        }

        /// One frame; each thread's capture back.
        fn frame(&self, now: u64) -> Vec<Capture> {
            for g in &self.go {
                g.send(now).unwrap();
            }
            for _ in &self.go {
                self.done.recv().unwrap();
            }
            self.s.slots.lock().unwrap().iter().filter_map(|slot| slot.lock().unwrap().take()).collect()
        }
    }

    /// Every field the list build and the dumps take from a light draw.
    fn fields(d: &Draw) -> String {
        format!(
            "{:#x} {:#x} {:#x} {:#x} {:?} {:#x} {:#x} {:#x} {:#x} {:#x} {} {} {:?} {:#x} {:?} {} {:?} {:?} {:#x} {:#x} {:#x} {:?} {:?} {:?} {:?} {:#x} {} {:#x} {:?} {:?} {:?} {:?} {:?} {:?} {} {:?}",
            d.geom, d.ghash, d.mesh_at, d.mesh_len, (d.vb_at, d.vb_len, d.ib_len), d.drawable, d.model, d.shader, d.viewport, d.tex_desc, d.alpha_clip, d.skinned,
            d.rigid.map(|m| m.map(f32::to_bits)), d.fx, d.layer_desc, d.stage_tls, d.palette, d.skin_cb, d.bound, d.cache_at, d.world_at,
            d.world.map(|m| m.map(f32::to_bits)), d.world_src, d.state, d.inst, d.stipple, d.srv_lazy, d.ret, d.remap, d.range, d.hook_bones, d.veh, d.pass, d.ent, d.seq_no, d.ped_skin
        )
    }

    /// The thin hook and its decode give every draw of the full hook,
    /// field by field, and the same census, cameras and instance blocks.
    #[test]
    fn thin_draws_equal_the_full_hook() {
        let _heavy = super::super::tests::heavy();
        let salt = Instant::now().elapsed().as_nanos() as u64 ^ 0x5EED_0000_0000;
        let f = Mutex::new(Fake::new(64, 192, 3, 900, 400, salt << 20));
        let (full, thin) = {
            let g = f.lock().unwrap();
            (state(&g, false, true), state(&g, true, true))
        };
        std::thread::scope(|sc| {
            let (ta, tb) = (Title::start(sc, &f, &full, 3, 700), Title::start(sc, &f, &thin, 3, 700));
            for now in [7u64, 8] {
                let mut a = ta.frame(now);
                let mut b = tb.frame(now);
                assert_eq!((a.len(), b.len()), (3, 3));
                let pending = b.iter().flat_map(|c| &c.draws).filter(|d| d.thin.flags != 0).count();
                assert!(pending > 1800, "the thin hook leaves its draws pending ({pending})");
                for c in &mut b {
                    decode(c);
                    assert!(c.draws.iter().all(|d| d.thin.flags == 0));
                }
                a.sort_by_key(|c| c.draws.first().map(|d| d.drawable));
                b.sort_by_key(|c| c.draws.first().map(|d| d.drawable));
                let (mut tex, mut layers, mut rigid, mut skinned, mut inst) = (0, 0, 0, 0, 0);
                for (ca, cb) in a.iter().zip(&b) {
                    assert_eq!(ca.draws.len(), cb.draws.len());
                    for (da, db) in ca.draws.iter().zip(&cb.draws) {
                        assert_eq!(fields(da), fields(db), "draw {} of frame {now}", da.seq_no);
                        tex += usize::from(da.tex_desc != 0);
                        layers += usize::from(da.layer_desc[0] != 0);
                        rigid += usize::from(da.rigid.is_some());
                        skinned += usize::from(da.skinned);
                        inst += usize::from(da.inst.is_some());
                    }
                    let mut census_a: Vec<_> = ca.census.iter().collect();
                    let mut census_b: Vec<_> = cb.census.iter().collect();
                    census_a.sort();
                    census_b.sort();
                    assert_eq!(census_a, census_b);
                    assert_eq!(ca.cameras.len(), cb.cameras.len());
                    assert_eq!((ca.inst_list.len(), ca.lod_list.len(), ca.unread), (cb.inst_list.len(), cb.lod_list.len(), cb.unread));
                }
                eprintln!("frame {now}: {} draws equal ({tex} with an albedo slot, {layers} layered, {rigid} rigid, {skinned} skinned, {inst} instanced)", a.iter().map(|c| c.draws.len()).sum::<usize>());
                assert!(tex > 100 && layers > 50 && rigid > 50 && skinned > 100 && inst > 100, "the fixture covers every path");
                for c in a.into_iter().chain(b) {
                    c.recycle();
                }
            }
            drop((ta, tb));
        });
    }

    /// The hook's time on the title's threads per draw, full against thin,
    /// over memory that is cold at the frame's start (a 64 MiB sweep between
    /// frames; 6,000 per-draw shader objects and blocks, 3,000 geometries on
    /// their own pages): 5 threads x 1,200 draws, best of 5 frames each, the
    /// two hooks frame by frame in turn. Then the thin decode of a frame.
    #[test]
    fn thin_hook_cost_cold() {
        let _heavy = super::super::tests::heavy();
        const THREADS: usize = 5;
        const DRAWS: usize = 1_200;
        let salt = Instant::now().elapsed().as_nanos() as u64 ^ 0xC01D_0000_0000;
        let f = Mutex::new(Fake::new(256, 4160, THREADS, 6_000, 3_000, salt << 20));
        let (full, thin) = {
            let g = f.lock().unwrap();
            (state(&g, false, false), state(&g, true, false))
        };
        let mut sweep = vec![0u64; 8 << 20];
        let mut cold = |k: u64| {
            for (i, w) in sweep.iter_mut().enumerate().step_by(8) {
                *w = w.wrapping_add(i as u64 ^ k);
            }
            std::hint::black_box(&sweep);
        };
        let per_draw = |cs: &[Capture]| cs.iter().map(|c| c.hook_ns).sum::<u64>() as f64 / 1e3 / cs.iter().map(|c| c.draws.len()).sum::<usize>().max(1) as f64;
        let parts = |cs: &[Capture]| {
            let mut p = [0u64; PARTS];
            for c in cs {
                for (a, b) in p.iter_mut().zip(c.parts) {
                    *a += b;
                }
            }
            let n = cs.iter().map(|c| c.draws.len()).sum::<usize>().max(1) as f64;
            let known: u64 = p[..6].iter().sum();
            p[6] = p[6].saturating_sub(known);
            (0..PARTS).map(|k| format!("{} {:.2}", PART_NAMES[k], p[k] as f64 / 1e3 / n)).collect::<Vec<_>>().join(", ")
        };
        std::thread::scope(|sc| {
            let (ta, tb) = (Title::start(sc, &f, &full, THREADS, DRAWS), Title::start(sc, &f, &thin, THREADS, DRAWS));
            // Warm the caches of ours (geometry keys, albedo table, model headers) once.
            for c in ta.frame(1).into_iter().chain(tb.frame(1)) {
                c.recycle();
            }
            let (mut best_full, mut best_thin, mut best_decode) = ((f64::MAX, String::new()), (f64::MAX, String::new()), (f64::MAX, 0.0));
            for round in 0..5u64 {
                cold(round);
                let a = ta.frame(2 + 2 * round);
                cold(round + 100);
                let mut b = tb.frame(3 + 2 * round);
                let (pa, pb) = (per_draw(&a), per_draw(&b));
                if pa < best_full.0 {
                    best_full = (pa, parts(&a));
                }
                if pb < best_thin.0 {
                    best_thin = (pb, parts(&b));
                }
                // The decode as the writer runs it: the frame's captures merged, then decoded.
                let mut merged = b.drain(..).reduce(Capture::merge).unwrap();
                let n = merged.draws.len() as f64;
                let t0 = Instant::now();
                decode(&mut merged);
                let wall = t0.elapsed().as_secs_f64() * 1e3;
                if wall < best_decode.0 {
                    best_decode = (wall, wall * 1e3 / n);
                }
                for c in a {
                    c.recycle();
                }
                merged.recycle();
            }
            drop((ta, tb));
            let _ = line();
            eprintln!("hook on the title's threads, cold frame, 5 x {DRAWS} draws: full {:.3} us a draw [{}]; thin {:.3} us a draw [{}]; thin decode {:.2} ms wall a frame ({:.3} us a draw, {} threads)", best_full.0, best_full.1, best_thin.0, best_thin.1, best_decode.0, best_decode.1, threads());
            assert!(best_thin.0 < best_full.0 || !std::env::var("XNW_TEST_TIMING").is_ok_and(|v| v == "1"));
        });
    }

    /// What the title changes between the frames of the cache test, before frame `now`.
    fn mutate(f: &mut Fake, now: u64, threads: usize, draws: usize) {
        let k = f.batches.len();
        // SAFETY: a qword inside the fake's arena.
        let rd = |at: u64| unsafe { (at as *const u64).read_unaligned() };
        for (b, fb) in f.batches.clone().into_iter().enumerate() {
            match now {
                // Descs swapped (rec+0), then back.
                9 | 11 if b % 5 == 1 => f.put(fb.rec, if now == 9 { fb.alt } else { fb.desc }),
                // LOD masks (rec+0xA0).
                10 if b % 4 == 2 => f.put(fb.rec + 0xA0, 1u32 << (15 + (b + 1) % 4) | 7),
                // A desc changed in place (kind C, one fewer), its drawables unloaded.
                11 if b % 6 == 3 => {
                    f.put(fb.desc + 8, fb.n.max(2) - 1);
                    // Every drawable an instanced draw of the batch has (Title::start: draw i of thread t is g = t * draws + i).
                    for g in (0..threads).flat_map(|t| (0..draws).step_by(3).map(move |i| t * draws + i)).filter(|g| g / 3 % k == b) {
                        crate::worldahead::test_unload(0x1234_0000 + g as u64);
                    }
                }
                // LOD records moved (rec+0x10: the next batch's).
                12 if b % 7 == 5 => {
                    let other = rd(f.batches[(b + 1) % k].rec + 0x10);
                    f.put(fb.rec + 0x10, other);
                }
                _ => {}
            }
            // CPU records rewritten (kind C: copied every frame).
            if now == 10 && !fb.kind_s {
                f.put(fb.records, now as f32);
            }
        }
        if now == 10 {
            for h in f.hides.clone() {
                let bits = rd(h);
                f.put(bits, 0xFFFF_0000u32);
            }
        }
    }

    /// The instance-batch cache (`XNW_HOOK_CACHE=on`, instcache.rs) gives
    /// every draw, instance block, LOD record and bitset of the uncached hook,
    /// and its counts, over 7 frames in which the title swaps descs (rec+0),
    /// LOD masks (rec+0xA0) and LOD records (rec+0x10), rewrites hide bits and
    /// CPU records, and changes a desc in place with its drawables unloaded.
    #[test]
    fn cached_instances_equal_the_uncached_hook() {
        let _heavy = super::super::tests::heavy();
        const THREADS: usize = 3;
        const DRAWS: usize = 700;
        let salt = Instant::now().elapsed().as_nanos() as u64 ^ 0xCAC4_0000_0000;
        let mut fk = Fake::new(96, 192, THREADS, 900, 400, salt << 20);
        fk.add_batches(240, 40, 2);
        let f = Mutex::new(fk);
        let (plain, cached) = {
            let g = f.lock().unwrap();
            (state_with(&g, false, true, false), state_with(&g, false, true, true))
        };
        let s0 = super::super::instcache::stats();
        std::thread::scope(|sc| {
            let (ta, tb) = (Title::start(sc, &f, &plain, THREADS, DRAWS), Title::start(sc, &f, &cached, THREADS, DRAWS));
            for now in 7u64..=13 {
                mutate(&mut f.lock().unwrap(), now, THREADS, DRAWS);
                let mut a = ta.frame(now);
                let mut b = tb.frame(now);
                assert_eq!((a.len(), b.len()), (THREADS, THREADS));
                a.sort_by_key(|c| c.draws.first().map(|d| d.drawable));
                b.sort_by_key(|c| c.draws.first().map(|d| d.drawable));
                let (mut inst, mut blocks, mut masked, mut near) = (0, 0, 0, 0);
                let mut calls = [0u64; 7];
                for (ca, cb) in a.iter().zip(&b) {
                    assert_eq!(ca.draws.len(), cb.draws.len());
                    for (da, db) in ca.draws.iter().zip(&cb.draws) {
                        assert_eq!(fields(da), fields(db), "draw {} of frame {now}", da.seq_no);
                        if let Some(i) = da.inst {
                            inst += 1;
                            blocks += usize::from(i.block != NO_PAL);
                            masked += usize::from(i.mask != NO_PAL);
                            near += usize::from(i.near != NO_PAL);
                        }
                    }
                    let bytes = |c: &Capture| c.inst_list.iter().map(|x| x.as_slice().to_vec()).collect::<Vec<_>>();
                    assert_eq!(bytes(ca), bytes(cb), "instance blocks of frame {now}");
                    assert_eq!(ca.inst_src, cb.inst_src);
                    let lods = |c: &Capture| c.lod_list.clone();
                    assert_eq!(lods(ca), lods(cb), "LOD records of frame {now}");
                    assert_eq!(ca.mask_list, cb.mask_list, "bitsets of frame {now}");
                    assert_eq!((ca.inst_calls, ca.copied), (cb.inst_calls, cb.copied), "counts of frame {now}");
                    for (x, y) in calls.iter_mut().zip(ca.inst_calls) {
                        *x += y;
                    }
                }
                eprintln!("frame {now}: {inst} instanced draws equal ({blocks} with a block, {masked} hide-masked, {near} near-masked; calls {calls:?})");
                assert!(inst > 500 && blocks > 300 && masked > 300 && near > 100 && calls[0] > 50 && calls[1] > 100 && calls[3] > 10, "the fixture covers every path");
                for c in a.into_iter().chain(b) {
                    c.recycle();
                }
            }
            drop((ta, tb));
        });
        let s1 = super::super::instcache::stats();
        let d: [u64; 5] = std::array::from_fn(|k| s1[k] - s0[k]);
        eprintln!("instance cache: {} hits, {} decoded ({} stale), {} unloaded", d[0], d[1], d[2], d[3]);
        assert!(d[0] > 2000 && d[2] > 50 && d[3] > 10, "hits, stale entries and unloads all seen: {d:?}");
    }

    /// The hook's instances part on the title's threads, uncached against
    /// cached, over memory cold at each frame's start (a 64 MiB sweep): 5
    /// threads x 1,200 draws, every third an instanced call over 600 batches
    /// of 1-64 instances; best of 5 frames each, the two in turn.
    #[test]
    fn instance_cache_cost_cold() {
        let _heavy = super::super::tests::heavy();
        const THREADS: usize = 5;
        const DRAWS: usize = 1_200;
        let salt = Instant::now().elapsed().as_nanos() as u64 ^ 0xCAC5_0000_0000;
        let mut fk = Fake::new(320, 4160, THREADS, 6_000, 3_000, salt << 20);
        fk.add_batches(600, 64, 6);
        let f = Mutex::new(fk);
        let (plain, cached) = {
            let g = f.lock().unwrap();
            (state_with(&g, false, false, false), state_with(&g, false, false, true))
        };
        let mut sweep = vec![0u64; 8 << 20];
        let mut cold = |k: u64| {
            for (i, w) in sweep.iter_mut().enumerate().step_by(8) {
                *w = w.wrapping_add(i as u64 ^ k);
            }
            std::hint::black_box(&sweep);
        };
        // (instanced calls ms, every one timed; hook ms; instanced draws) of a frame.
        let cost = |cs: &[Capture]| {
            let inst = cs.iter().map(|c| c.inst_ns).sum::<u64>() as f64 / 1e6;
            let hook = cs.iter().map(|c| c.hook_ns).sum::<u64>() as f64 / 1e6;
            (inst, hook, cs.iter().flat_map(|c| &c.draws).filter(|d| d.inst.is_some()).count())
        };
        std::thread::scope(|sc| {
            let (ta, tb) = (Title::start(sc, &f, &plain, THREADS, DRAWS), Title::start(sc, &f, &cached, THREADS, DRAWS));
            for c in ta.frame(1).into_iter().chain(tb.frame(1)) {
                c.recycle();
            }
            // Best instanced-calls ms of [uncached, cached] x [cold, warm] frames.
            let mut best = [[f64::MAX; 2]; 2];
            let mut calls = 0;
            for round in 0..40u64 {
                for (k, t) in [&ta, &tb].into_iter().enumerate() {
                    // Even rounds cold; odd rounds as warm as the frame before left them.
                    let w = (round % 2) as usize;
                    if w == 0 {
                        cold(2 * round + k as u64);
                    }
                    let cs = t.frame(2 + 2 * round + k as u64);
                    let (ms, _, n) = cost(&cs);
                    best[k][w] = best[k][w].min(ms);
                    calls = n;
                    for c in cs {
                        c.recycle();
                    }
                }
            }
            drop((ta, tb));
            let us = |ms: f64| ms * 1e3 / calls.max(1) as f64;
            let cut = |w: usize| 100.0 * (1.0 - best[1][w] / best[0][w]);
            eprintln!(
                "instanced calls on the title's threads, {THREADS} x {DRAWS} draws, {calls} calls a frame: cold frame {:.3} ms ({:.3} us a call) uncached, {:.3} ms ({:.3} us) cached: -{:.0} %; warm frame {:.3} ms ({:.3} us) uncached, {:.3} ms ({:.3} us) cached: -{:.0} %",
                best[0][0], us(best[0][0]), best[1][0], us(best[1][0]), cut(0), best[0][1], us(best[0][1]), best[1][1], us(best[1][1]), cut(1)
            );
            assert!(best[1][0] < best[0][0] || !std::env::var("XNW_TEST_TIMING").is_ok_and(|v| v == "1"));
        });
    }
}
