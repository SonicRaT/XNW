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
use std::sync::{Arc, Mutex};

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
                bridged::cache().ask(key, msc, psc, &host.device, &host.caps, false);
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
    let Some(key) = pipe_key(d.pipeline) else { return };
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

fn read(addr: u64, len: u64) -> Option<Vec<u8>> {
    if addr == 0 || len == 0 || !(0x1_0000..0x8000_0000_0000).contains(&addr) || addr.checked_add(len).is_none_or(|e| e > 0x8000_0000_0000) {
        return None;
    }
    let _site = super::fault_site_or_kept(super::SITE_BRIDGE);
    super::dcopy(addr, len as usize)
}

/// Bytes read from title memory this frame (the frame line), and the per
/// frame view cache: (frame, (address, length) -> bytes).
static READ_BYTES: AtomicU64 = AtomicU64::new(0);
type Views = (u64, HashMap<(u64, u64), Arc<Vec<u8>>>);
static VIEWS: Mutex<Option<Views>> = Mutex::new(None);
/// Root views kept across frames: (address, length) -> (hash of the first
/// 256 bytes, bytes, frame last used).
type Kept = HashMap<(u64, u64), (u128, Arc<Vec<u8>>, u64)>;
static KEPT: Mutex<Option<Kept>> = Mutex::new(None);
/// Frames a kept root view stays unused before it goes.
const KEEP_FRAMES: u64 = 600;
type Samplers = (u64, HashMap<usize, Option<Arc<Vec<(u32, SamplerDesc)>>>>);
static SAMPLERS: Mutex<Option<Samplers>> = Mutex::new(None);

/// Bytes read since the last call.
pub fn take_read_bytes() -> u64 {
    READ_BYTES.swap(0, Ordering::Relaxed)
}

fn frame_read(frame: u64, addr: u64, len: u64) -> Option<Arc<Vec<u8>>> {
    {
        let mut g = lock(&VIEWS);
        let v = g.get_or_insert_with(|| (frame, HashMap::new()));
        if v.0 != frame {
            *v = (frame, HashMap::new());
        }
        if let Some(b) = v.1.get(&(addr, len)) {
            return Some(b.clone());
        }
    }
    // Read outside the lock: other title threads' sinks do not wait on it.
    let b = Arc::new(read(addr, len)?);
    READ_BYTES.fetch_add(len, Ordering::Relaxed);
    let mut g = lock(&VIEWS);
    if let Some(v) = g.as_mut().filter(|v| v.0 == frame) {
        v.1.insert((addr, len), b.clone());
    }
    Some(b)
}

fn kept_read(frame: u64, addr: u64, len: u64) -> Option<Arc<Vec<u8>>> {
    let head = read(addr, len.min(256))?;
    let h = xnw_core::hash128(&[&head]);
    {
        let mut g = lock(&KEPT);
        let m = g.get_or_insert_with(HashMap::new);
        if let Some(e) = m.get_mut(&(addr, len)).filter(|e| e.0 == h) {
            e.2 = frame;
            return Some(e.1.clone());
        }
    }
    let b = Arc::new(read(addr, len)?);
    READ_BYTES.fetch_add(len, Ordering::Relaxed);
    let mut g = lock(&KEPT);
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() > 4096 || frame.is_multiple_of(60) {
        m.retain(|_, e| frame.saturating_sub(e.2) <= KEEP_FRAMES);
    }
    m.insert((addr, len), (h, b.clone(), frame));
    Some(b)
}

fn src_view(v: &TapView) -> SrcView {
    SrcView { kind: v.kind, addr: v.addr, bytes: v.bytes, format: v.format, width: v.width }
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

    fn bytes(&self, _: u16, _: u32, v: &SrcView) -> Option<Arc<Vec<u8>>> {
        frame_read(self.frame, v.addr, v.bytes.min(VIEW_MAX))
    }

    fn root_view(&self, _: u16, addr: u64) -> Option<Arc<Vec<u8>>> {
        let (s_at, s_len) = self.stream;
        let stream = |x: u64| (s_at != 0 && x >= s_at && x < s_at + s_len).then(|| s_at + s_len - x);
        let (at, full, _, how) = super::pack::root_view_at(addr, tap::buffer_left, stream);
        if how == super::pack::ROOT_NONE {
            return None;
        }
        kept_read(self.frame, at, full.min(VIEW_MAX))
    }

    fn words(&self) -> Option<[u32; WORDS]> {
        let v = self.table(0, WORDS_SLOT).filter(|v| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER | view::CONSTANT))?;
        let off = match self.d.roots.get(WORDS_ROOT) {
            Some(TapRoot::Constants(c)) => u64::from(*c.get(WORDS_DWORD)?),
            _ => return None,
        };
        if off + 4 * WORDS as u64 > v.bytes {
            return None;
        }
        let b = read(v.addr + off, 4 * WORDS as u64)?;
        Some(std::array::from_fn(|k| u32::from_le_bytes([b[4 * k], b[4 * k + 1], b[4 * k + 2], b[4 * k + 3]])))
    }

    fn heap(&self, index: u32) -> Option<(u64, SrcView)> {
        let (_, h) = tap::heap_slot(self.d.view_heap, index)?;
        // The slot's record first (fault-safe): `tap::view` reads it in place.
        read(h, tap::SLOT)?;
        Some((h, src_view(&tap::view(h))))
    }

    fn samplers(&self) -> Option<Arc<Vec<(u32, SamplerDesc)>>> {
        let heap = self.d.sampler_heap;
        {
            let mut g = lock(&SAMPLERS);
            let v = g.get_or_insert_with(|| (self.frame, HashMap::new()));
            if v.0 != self.frame {
                *v = (self.frame, HashMap::new());
            }
            if let Some(s) = v.1.get(&heap) {
                return s.clone();
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

    #[test]
    fn frame_reads_are_shared_within_a_frame() {
        let buf = vec![7u8; 4096];
        let a = buf.as_ptr() as u64;
        let x = frame_read(1_000_001, a, 4096).expect("read");
        let y = frame_read(1_000_001, a, 4096).expect("read again");
        assert!(Arc::ptr_eq(&x, &y));
        // A new frame reads afresh.
        let z = frame_read(1_000_002, a, 4096).expect("next frame");
        assert!(!Arc::ptr_eq(&x, &z));
    }
}
