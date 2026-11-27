//! Albedo textures for our world pass.
//!
//! A draw carries the descriptor slot of its albedo view (`view+0x18`, read
//! at the hook: HD swaps and residency rewrite that slot, never `view+8`).
//! Once a frame, per distinct slot, [`TexTable::prepare`] reads the slot's
//! record (guarded copy), checks its generation (`descriptor::entry`), turns
//! the view into the host texture our layer already made for it
//! (`View::Placed` -> `placed_texture`, `View::Shader` -> the resource's
//! host copy), and, when its contents are due, takes the upload of
//! `texture::for_pass` (pack, cached pages, else title memory: the title's
//! own refresh) and records it on our pass's list. Nothing here reads the
//! pack by key or detiles by hand.
//!
//! Bounds: the shader indexes only our own table of [`TEX_SLOTS`] SRVs at
//! the top of the pass's heap, with an index our CPU code picked (checked
//! against the table's use; the shader clamps it again). Index 0 is "flat":
//! the draw keeps its flat colour (no texture, not resident, not 2D, table
//! full, upload waiting for its turn). Uploads are capped per frame (count
//! and bytes); the rest wait. `Checker` mode (offline dumps carry no texels)
//! gives every distinct slot its own checker texture.

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use xnw_gpu::d3d12x::descriptor::{self, View};
use xnw_gpu::d3d12x::{host, resource, texture};

/// SRVs in our table (the shader's `albedoTex` array and its root range);
/// `tex=<n>` uses fewer. The most distinct slots one frame asks for:
/// 821 in the card runs without bridged rows, 1,501 with
/// them, 1,427 in the burst (pk7a3 f961,
/// Entries stay with their slot across frames, so the table
/// holds what several frames ask: 4096 is 2.7 x the largest frame. The
/// views cost 4096 descriptors of the pass heap; textures are bounded by
/// the layer's budget, not by entries.
pub const TEX_SLOTS: u32 = 4096;
/// Entries used unless `tex=<n>` says otherwise.
pub const TEX_DEFAULT: u32 = TEX_SLOTS;
/// Entries below this are the layers' (the shader's 8-bit field, k[50]);
/// albedo and bridged textures take entries from here up first.
pub const TEX_LOW: u32 = 256;

/// `XNW_TEX_STABLE` (default on; `off`: every frame assigns its entries
/// afresh, the rule before): a slot keeps its entry while it stays
/// resident, its view is written only when it changes, the least
/// recently used entries go when the table is full.
pub fn stable_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("XNW_TEX_STABLE").as_deref().map(str::trim), Ok("off" | "0")))
}

/// A layer's entry as the shader's 8-bit field takes it: entries past 255
/// draw the layer as the base colour (0), never another texture. Layers
/// take the first entries ([`super::Pass`] asks for them first).
pub fn layer_entry(i: u32) -> u32 {
    if i < 256 { i } else { 0 }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TexMode {
    /// Flat colours only (`tex=off`).
    Off,
    /// The title's textures through our layer's host copies.
    Live,
    /// One checker texture per distinct slot (offline dumps: no texels).
    Checker,
    /// Offline, texels a provider gives per slot (a frame pack v2);
    /// a slot it has none for is flat.
    Provided,
}

/// One slot's texels for [`TexMode::Provided`]: the finest mip,
/// linear rows (block rows for compressed formats), packed.
pub struct Provided {
    pub width: u32,
    pub height: u32,
    /// DXGI format number of the host texture and its view.
    pub format: u32,
    pub rows: Vec<u8>,
    /// Bytes of one row and the number of rows in `rows`.
    pub row_bytes: u32,
    pub row_count: u32,
    /// Its shape: [`TEX_2D`], [`TEX_1D`], [`TEX_3D`], [`TEX_2D_ARRAY`] or
    /// [`TEX_CUBE`] (`depth` slices, `row_count` = height x depth). Our
    /// pass samples 2D only; the title's own programs get the others
    /// through [`TexTable::host_view`] and the draw bridge.
    pub dim: u8,
    pub depth: u32,
    /// Mips in `rows` (0 or 1: the one mip the fields above describe).
    /// More: `rows` holds every subresource in D3D12 order (slice after
    /// slice, each its mips from `width` x `height` down; a 3D texture mip
    /// after mip, each with its depth slices), every row at its own mip's
    /// length, nothing between them ([`provided_fill`]).
    pub mips: u32,
}

/// [`Provided::dim`] (D3D12 resource dimensions).
pub const TEX_1D: u8 = 2;
pub const TEX_2D: u8 = 3;
pub const TEX_3D: u8 = 4;
/// A 2D array (LIT5): `depth` slices, `row_count` = height x depth, the
/// slices' rows in order (D3D12 has no array dimension; past the three).
pub const TEX_2D_ARRAY: u8 = 5;
/// A cube (BUG3): `depth` = 6 faces in the console's and D3D12's slice order
/// (+x, -x, +y, -y, +z, -z), viewed as a cube; a cube array gives its first cube.
pub const TEX_CUBE: u8 = 6;

/// What gives a slot's texels in [`TexMode::Provided`].
pub type Provider = Box<dyn FnMut(u64) -> Option<Provided> + Send>;
/// A slot's content key in [`TexMode::Provided`] (cheap: no texels read):
/// equal keys give equal texels, so a texture made for one frame's pack
/// serves the next frames'. None: no texels for the slot.
pub type Keys = Box<dyn FnMut(u64) -> Option<u64> + Send>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexOptions {
    pub mode: TexMode,
    /// Table entries used (1..=TEX_SLOTS; entry 0 is flat).
    pub slots: u32,
    /// Uploads per frame at most, and their bytes (the first always goes).
    pub uploads: u32,
    pub upload_bytes: u64,
    /// Reads (pack, pages, title memory; a provider's texels) on a reader
    /// thread, never in the frame: a slot whose texels are not read yet is
    /// flat (`waiting`) and takes its texture on the frame after the read
    /// lands; no time cap. The screen pass's default; `texasync=off`: in the
    /// frame, under `frame_ms`.
    pub async_reads: bool,
    /// New reads in a frame stop after this many ms of its table (the
    /// first always goes; 0: no time cap). `texms=<ms>`; a fast camera cut
    /// sets its own (cuts::CUT_TEX_MS).
    pub frame_ms: f64,
}

impl Default for TexOptions {
    fn default() -> Self {
        TexOptions { mode: TexMode::Off, slots: TEX_DEFAULT, uploads: 32, upload_bytes: 32 << 20, async_reads: false, frame_ms: 0.0 }
    }
}

/// A title resource hold moved to the reader thread with its read.
struct Hold {
    _placed: Option<resource::HeldTexture>,
    _live: Option<resource::LiveHold>,
}
// SAFETY: a reference count on a title object: taken here, given back on
// whatever thread drops it (the object's release is thread-agnostic, as the
// title's own lists release on its render threads).
unsafe impl Send for Hold {}

/// What a descriptor slot holds now (placed or shader).
#[derive(Debug, PartialEq)]
pub(crate) enum Slot {
    /// Outside the address space, or no record tag there (zeros).
    NoRecord,
    /// The record's generation is not its entry's any more (released, or
    /// rewritten since: re-ask).
    Stale,
    /// A shader view of memory (non-buffer resource description).
    Placed((u32, u32), View),
    /// A shader view of a resource object.
    Shader((u32, u32), usize, Option<[u32; 10]>),
    /// Anything else (buffer, UAV, constant, sampler).
    Other,
}

/// The view at descriptor slot `slot`: the record read with a guarded copy
/// (a title-supplied address is never dereferenced), then the view entry
/// with a generation check.
pub(crate) fn classify(slot: u64) -> Slot {
    if !(0x1_0000..0x8000_0000_0000).contains(&slot) {
        return Slot::NoRecord;
    }
    let mut rec = [0u8; 32];
    host::copy_mapped(rec.as_mut_ptr(), slot as *const u8, rec.len(), false);
    // SAFETY: our own 32-byte copy of the record.
    let Some((i, generation)) = (unsafe { descriptor::record_id(rec.as_ptr() as u64) }) else { return Slot::NoRecord };
    let Some((view, current, _)) = descriptor::entry(i) else { return Slot::Stale };
    if current != generation {
        return Slot::Stale;
    }
    match &view {
        View::Placed(false, r, _) if r[0] != resource::BUFFER => Slot::Placed((i, generation), view.clone()),
        View::Shader(r, d) if d.is_none_or(|d| d[1] != 1) => Slot::Shader((i, generation), *r, *d),
        _ => Slot::Other,
    }
}

/// The per-frame upload budget: `n` uploads and `bytes`; the first passes.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) struct Budget {
    pub(crate) count: u32,
    pub(crate) bytes: u64,
    /// No new read after this instant (the frame's time cap).
    pub(crate) until: Option<std::time::Instant>,
}

impl Budget {
    /// A budget whose reads stop `ms` after now (0: no time cap).
    pub(crate) fn timed(ms: f64) -> Budget {
        Budget { until: (ms > 0.0).then(|| std::time::Instant::now() + std::time::Duration::from_secs_f64(ms / 1e3)), ..Default::default() }
    }

    /// Past the frame's time cap.
    pub(crate) fn late(&self) -> bool {
        self.until.is_some_and(|u| std::time::Instant::now() >= u)
    }

    pub(crate) fn room(&mut self, o: &TexOptions, bytes: u64) -> bool {
        let first = self.count == 0;
        if !first && (self.count >= o.uploads || self.bytes + bytes > o.upload_bytes || self.late()) {
            return false;
        }
        self.count += 1;
        self.bytes += bytes;
        true
    }
}

/// sRGB view formats (sampled raw values are re-encoded in the shader: our
/// target is UNORM and the flat path writes display values).
fn is_srgb(f: u32) -> bool {
    matches!(f, 29 | 72 | 75 | 78 | 91 | 93 | 99)
}

/// A slot known from an earlier frame.
struct Known {
    id: (u32, u32),
    copy: Option<Arc<resource::HostTexture>>,
    srv: D3D12_SHADER_RESOURCE_VIEW_DESC,
    srgb: bool,
    ready: bool,
    ours: bool,
    used: u64,
    /// `srv` is a view of another dimension ([`wide_srv`]): for the bridged rows only.
    wide: bool,
    /// A slot of this generation that is flat for good (not a plain
    /// 2D texture, or the title's GPU output): no read, no layer lookup again.
    flat: Option<Flat>,
    /// The frame a partly streamed copy (pending, texels shown, no
    /// upload) was last looked at: looked at again [`recheck_frames`] later.
    partial_at: Option<u64>,
    /// Its view is a placed one (else a shader view): the resolve's kind
    /// counters when [`batch_on`] skips [`classify`].
    placed: bool,
    /// A read of its copy is with the reader thread (`async_reads`): looked
    /// up again every frame until its upload has been handed over.
    flying: bool,
}

/// Whether a known slot is answered from what the table knows, with no
/// lookup in the layer (pure; the hand-over's rule). Not while a read of
/// its copy is `flying`: the copy's own state changes when the reader has
/// READ, before the pass has the texels. `pending`: the copy still waits for
/// pages; `partial_fresh`: the slot shows a partly streamed copy
/// (`partial_at`) looked at too recently to look again. A partial view whose
/// copy became complete meanwhile (another slot's or resource's read) is
/// looked up once more: its view starts at the mip that was whole then.
pub(crate) fn known_stands(ready: bool, flying: bool, pending: bool, partial: bool, partial_fresh: bool) -> bool {
    ready && !flying && if pending { partial && partial_fresh } else { !partial }
}

/// Frames between looks at a partly streamed copy that showed no new
/// page (`XNW_TEX_RECHECK=<n>`, default 4; 1: every frame, the rule before).
/// About 800 such asks a frame went through the layer's locks and
/// the staging's page check with bridgeps on.
fn recheck_frames() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XNW_TEX_RECHECK").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4).max(1))
}

fn switch_on(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| matches!(v.trim(), "on" | "1"))
}

/// `XNW_TEX_BATCH=on` (default off): the live table reads every
/// distinct slot's record and checks its generation once a frame for all
/// slots (one look at the traps, guarded copies in place, one look at the
/// views: [`host::copy_records`], [`descriptor::current`]); a known slot
/// whose record is unchanged skips [`classify`], which took the pair, guard
/// and views locks and a `ReadProcessMemory` per slot (6.6 us a `flat
/// known` resolve in run-br12e).
pub fn batch_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| switch_on("XNW_TEX_BATCH"))
}

/// `XNW_TEX_ARRIVALS=on` (default off): a partly streamed copy is
/// looked at again when pages of its texture arrived
/// ([`xnw_formats::origin::arrived_since`], once a frame), else by a slow
/// sweep: each slot once every 4 x [`recheck_frames`] frames, slots spread
/// over the frames by address. Off: every `XNW_TEX_RECHECK` frames (about
/// 4,000 looks a second in the street, nearly all `partial no new pages`).
/// A copy without a texture origin (title-memory uploads) and the title's
/// own writes to texture memory wait for the sweep.
pub fn arrivals_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| switch_on("XNW_TEX_ARRIVALS"))
}

/// The arrivals a live table saw this frame ([`arrivals_on`]).
#[derive(Default)]
struct Arrivals {
    /// The registry's arrival count last read (None: never).
    seen: Option<u64>,
    /// This frame: textures with new pages, and "look at every one" (the
    /// first frame, or arrivals cut before they were read).
    textures: std::collections::HashSet<(u32, u32)>,
    all: bool,
    buf: Vec<(u32, u32)>,
}

impl Arrivals {
    /// Reads the arrivals since the last frame's.
    fn take(&mut self) {
        let (upto, lost) = xnw_formats::origin::arrived_since(self.seen, &mut self.buf);
        self.seen = Some(upto);
        self.all = lost;
        self.textures.clear();
        self.textures.extend(self.buf.iter().copied());
    }
}

/// Whether a partly streamed copy last looked at in frame `at` is looked
/// at again in `frame`: `XNW_TEX_RECHECK` frames later, or with arrivals
/// ([`arrivals_on`]) when pages of its texture arrived or the slot's turn
/// of the sweep came.
fn partial_due(slot: u64, at: u64, frame: u64, origin: Option<(u32, u32)>, arrivals: Option<(&std::collections::HashSet<(u32, u32)>, bool)>) -> bool {
    match arrivals {
        None => frame >= at + recheck_frames(),
        Some((arrived, all)) => all || origin.is_some_and(|o| arrived.contains(&o)) || (slot >> 5).wrapping_add(frame).is_multiple_of(4 * recheck_frames()),
    }
}

/// Why a drawn draw is flat (the `TEX-REASONS:` names, in order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flat {
    /// The draw names no albedo descriptor (`albedo_desc` 0).
    None = 0,
    /// The table had no free entry when the slot came up.
    Full = 1,
    /// The slot's record is gone or rewritten (stale), or there is none.
    Stale = 2,
    /// The view or its copy is not a plain 2D texture (or a buffer, UAV...).
    Not2d = 3,
    /// Contents not there yet: upload waiting for its turn or its read.
    Waiting = 4,
    /// Anything else (no host copy, textures off, a checker failed).
    Other = 5,
    /// The resource is the title's own render, depth or compute target: its
    /// GPU work is nulled, so nothing ever writes it; not refreshed, no budget.
    GpuOut = 6,
}

/// The reason names, indexed by [`Flat`].
pub const FLAT_REASONS: [&str; 7] = ["none", "full", "stale", "not2d", "waiting", "other", "gpuout"];

/// Counters, per log period.
#[derive(Default, Debug, Clone, Copy)]
pub struct TexStats {
    pub bound_draws: u64,
    pub flat_draws: u64,
    /// `flat_draws` split by [`Flat`] (sums to `flat_draws`).
    pub flat_by: [u64; 7],
    pub uploads: u64,
    pub upload_bytes: u64,
    /// Distinct slots waiting for an upload in the last frame.
    pub waiting: u64,
    pub distinct: u64,
    /// Distinct non-zero slots the last frame asked for (albedo and
    /// layers; what a table needs to hold them all).
    pub demand: u64,
    /// Slot kinds per resolve.
    pub placed: u64,
    pub shader: u64,
    pub other: u64,
    pub stale: u64,
    pub no_record: u64,
    /// Upload sources (pack, pages, memory, hashed memory,
    /// unknown) and re-uploads of a texture we uploaded (new contents: a
    /// streamed upgrade, or the title writing it).
    pub sources: [u64; 5],
    pub again: u64,
    pub not_2d: u64,
    pub no_copy: u64,
    pub full: u64,
    /// Entries: resident after the last frame; views written, kept as
    /// they were, written null; entries evicted for room.
    pub resident: u64,
    pub written: u64,
    pub kept: u64,
    pub nulls: u64,
    pub evicted: u64,
    /// Provided textures taken again by their key (no read, no upload).
    pub provided_kept: u64,
    /// Live resolves by path (known, flat known, partial kept, looked
    /// up, waited past the read time) and their milliseconds.
    pub paths: [u64; 5],
    pub path_ms: [f64; 5],
}

/// The table's entries across frames (pure: no device). An entry stays
/// with its slot while the slot resolves; a new slot takes a free entry,
/// else the least recently used eighth of its pool not used this frame
/// goes at once (hysteresis: one scan frees room for many asks). Entries
/// below `low` are the layers' pool; the others take entries from `low`
/// up, then from the layers' pool.
pub(crate) struct Entries {
    slots: u32,
    low: u32,
    /// Per entry: its slot (0: free), the view it holds (identity), the
    /// frame it was last used.
    owner: Vec<(u64, u64, u64)>,
    at: HashMap<u64, u32>,
    /// Free entries of the layers' pool and of the rest; popped lowest first.
    free: [Vec<u32>; 2],
    frame: u64,
    /// Entries let go since the frame began (written null if still free).
    pub(crate) nulls: Vec<u32>,
    /// Counters: views written, views kept, entries evicted (for room).
    pub(crate) written: u64,
    pub(crate) kept: u64,
    pub(crate) evicted: u64,
}

impl Entries {
    pub(crate) fn new(slots: u32) -> Entries {
        let low = TEX_LOW.min(slots);
        Entries { slots, low, owner: vec![(0, 0, 0); slots as usize], at: HashMap::new(), free: [(1..low).rev().collect(), (low..slots).rev().collect()], frame: 0, nulls: Vec::new(), written: 0, kept: 0, evicted: 0 }
    }

    /// A new frame; `fresh` lets every entry go first (the old rule).
    pub(crate) fn begin(&mut self, frame: u64, fresh: bool) {
        self.frame = frame;
        if fresh {
            let held: Vec<u64> = self.at.keys().copied().collect();
            for slot in held {
                self.release(slot);
            }
            for f in &mut self.free {
                f.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
    }

    /// The entry `slot` has (none: not resident).
    pub(crate) fn entry(&self, slot: u64) -> Option<u32> {
        self.at.get(&slot).copied()
    }

    /// Lets `slot`'s entry go (a view it holds is written null at the
    /// frame's end).
    pub(crate) fn release(&mut self, slot: u64) {
        if let Some(e) = self.at.remove(&slot) {
            if self.owner[e as usize].1 != 0 {
                self.nulls.push(e);
            }
            self.owner[e as usize] = (0, 0, 0);
            self.free[usize::from(e >= self.low)].push(e);
        }
    }

    /// Frees the least recently used eighth (at least one) of pool `p`'s
    /// entries not used this frame; false when every one is in use.
    fn evict(&mut self, p: usize) -> bool {
        let range = if p == 0 { 1..self.low } else { self.low..self.slots };
        let mut old: Vec<(u64, u32)> = range.clone().filter(|&e| self.owner[e as usize].0 != 0 && self.owner[e as usize].2 < self.frame).map(|e| (self.owner[e as usize].2, e)).collect();
        if old.is_empty() {
            return false;
        }
        let n = (range.len() / 8).max(1).min(old.len());
        old.select_nth_unstable(n - 1);
        for &(_, e) in &old[..n] {
            let slot = self.owner[e as usize].0;
            self.release(slot);
            self.evicted += 1;
        }
        true
    }

    /// A new entry: a layer's from its pool; albedo's from its own pool,
    /// then the layers'. Each pool: a free entry, else (with `evict`) the
    /// least recently used go. Without `evict`, from albedo's pool only.
    fn alloc(&mut self, low: bool, evict: bool) -> Option<u32> {
        let pools: &[usize] = match (low, evict) {
            (true, _) => &[0],
            (false, true) => &[1, 0],
            (false, false) => &[1],
        };
        for &p in pools {
            if let Some(e) = self.free[p].pop() {
                return Some(e);
            }
            if evict && self.evict(p) {
                return self.free[p].pop();
            }
        }
        None
    }

    /// The entry `slot` uses this frame (a layer's below `low`): its own,
    /// else a new one; None when the table is full of this frame's.
    pub(crate) fn reserve(&mut self, slot: u64, low: bool) -> Option<u32> {
        self.reserve_with(slot, low, true)
    }

    /// As [`Self::reserve`], a new entry only from the free ones.
    pub(crate) fn reserve_free(&mut self, slot: u64, low: bool) -> Option<u32> {
        self.reserve_with(slot, low, false)
    }

    /// Whether `slot` holds an entry it may use as asked.
    pub(crate) fn holds(&self, slot: u64, low: bool) -> bool {
        self.entry(slot).is_some_and(|e| !low || e < self.low)
    }

    fn reserve_with(&mut self, slot: u64, low: bool, evict: bool) -> Option<u32> {
        if let Some(e) = self.entry(slot) {
            if !low || e < self.low {
                self.owner[e as usize].2 = self.frame;
                return Some(e);
            }
            // A layer's slot held above the layers' pool moves down.
            let n = self.alloc(true, evict)?;
            self.release(slot);
            return Some(self.take(slot, n));
        }
        let n = self.alloc(low, evict)?;
        Some(self.take(slot, n))
    }

    fn take(&mut self, slot: u64, e: u32) -> u32 {
        self.owner[e as usize] = (slot, 0, self.frame);
        self.at.insert(slot, e);
        e
    }

    /// Entry `e` holds `view` now: true when its descriptor must be written.
    pub(crate) fn hold(&mut self, e: u32, view: u64) -> bool {
        let o = &mut self.owner[e as usize];
        if o.1 == view && view != 0 {
            self.kept += 1;
            return false;
        }
        o.1 = view;
        self.written += 1;
        true
    }

    /// Lets go of every entry for which `keep(slot, last used)` is false.
    pub(crate) fn sweep(&mut self, keep: &dyn Fn(u64, u64) -> bool) {
        let gone: Vec<u64> = self.owner.iter().filter(|o| o.0 != 0 && !keep(o.0, o.2)).map(|o| o.0).collect();
        for slot in gone {
            self.release(slot);
        }
    }

    /// Entries let go since the last call and not taken again (to be
    /// written null).
    pub(crate) fn take_nulls(&mut self) -> Vec<u32> {
        let mut v: Vec<u32> = std::mem::take(&mut self.nulls).into_iter().filter(|&e| self.owner[e as usize].0 == 0).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    pub(crate) fn resident(&self) -> usize {
        self.at.len()
    }
}

/// An identity of a view: the resource and the view's description.
fn view_id(r: &ID3D12Resource, srv: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (windows::core::Interface::as_raw(r) as usize, srv.Format.0, srv.ViewDimension.0, srv.Shader4ComponentMapping).hash(&mut h);
    // SAFETY: every view we make is a 2D texture view.
    let t = unsafe { srv.Anonymous.Texture2D };
    (t.MostDetailedMip, t.MipLevels, t.PlaneSlice, t.ResourceMinLODClamp.to_bits()).hash(&mut h);
    h.finish() | 1
}

/// Our table of albedo SRVs and what it knows about slots.
pub struct TexTable {
    pub opts: TexOptions,
    /// First heap slot of the table (the top of the pass's heap).
    base: u32,
    /// This frame: slot -> (table index, sRGB).
    frame: HashMap<u64, (u32, bool)>,
    /// This frame: slot -> why it is flat (slots with an entry are absent).
    why: HashMap<u64, Flat>,
    /// Entries across frames, and the resource each one's view names
    /// (held while the view stands).
    entries: Entries,
    res: Vec<Option<ID3D12Resource>>,
    known: HashMap<u64, Known>,
    /// This frame's title textures that are not plain 2D (arrays,
    /// cubes, 3D) with a view of their own dimension: for the bridged rows
    /// only (our draws read the table as 2D and stay flat on them).
    wide: HashMap<u64, (ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)>,
    /// Copies sampled this frame and uploads recorded (kept to the fence).
    held: Vec<Arc<resource::HostTexture>>,
    uploads: Vec<(Arc<resource::HostTexture>, texture::PassUpload)>,
    checkers: HashMap<u64, (ID3D12Resource, Option<ID3D12Resource>)>,
    /// The provider, its content keys, and its textures by key (texture,
    /// upload until copied, format, frame last used); this provider's
    /// slot -> key.
    provider: Option<SharedProvider>,
    keys: Option<Keys>,
    provided: HashMap<u64, ProvidedTex>,
    slot_key: HashMap<u64, Option<u64>>,
    /// Keys whose texels did not decode (they never will: same inputs).
    provided_none: std::collections::HashSet<u64>,
    frame_no: u64,
    pub stats: TexStats,
    last_log: std::time::Instant,
    /// This frame's entries: their texture and view (the title's own
    /// programs in our pass make their own views of them, `bridged`).
    views: HashMap<u32, (ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)>,
    /// [`batch_on`]: this frame's slots whose record names a current view
    /// entry, and that entry and generation.
    current: HashMap<u64, (u32, u32)>,
    /// [`arrivals_on`]: this frame's texture arrivals.
    arrivals: Arrivals,
    /// `async_reads` in [`TexMode::Provided`]: the provider's reads on the
    /// table's own thread.
    reads: ProvidedReads,
    /// The provided slots asked this frame are read before it is drawn
    /// (the reader is waited for; in the frame without `async_reads`, with
    /// no time cap): a replay's last frame, so that the two paths' pictures
    /// of it can be compared with every read landed.
    pub settle: bool,
}

/// A provided texture made off the frame: the texture, its filled upload
/// buffer, the footprints to copy, its view, "not plain 2D", its bytes.
type Made = (ID3D12Resource, ID3D12Resource, Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>, D3D12_SHADER_RESOURCE_VIEW_DESC, bool, u64);

/// The provider shared with the table's reader thread.
type SharedProvider = Arc<std::sync::Mutex<Provider>>;

/// A read for the reader thread: provider epoch, key, slot, the provider.
struct ReadAsk {
    epoch: u64,
    key: u64,
    slot: u64,
    provider: SharedProvider,
    device: ID3D12Device,
}
// SAFETY: D3D12 objects are free-threaded; the provider is `Send` behind its lock.
unsafe impl Send for ReadAsk {}

/// A read the thread finished: epoch, key, the texture (None: no texels).
struct ReadLanded(u64, u64, Option<Made>);
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for ReadLanded {}

/// The provided reads off the frame (`async_reads`): one thread, made
/// on the first ask, gone with the table (its channel closes).
#[derive(Default)]
struct ProvidedReads {
    to: Option<std::sync::mpsc::Sender<ReadAsk>>,
    from: Option<std::sync::mpsc::Receiver<ReadLanded>>,
    /// Keys asked and not landed (of this epoch).
    flying: std::collections::HashSet<u64>,
    /// Bumped when the provider's textures start over (results of an
    /// older epoch are dropped).
    epoch: u64,
}

impl ProvidedReads {
    /// Queues the read of `key` (once); false: no thread.
    fn ask(&mut self, key: u64, slot: u64, provider: &SharedProvider, device: &ID3D12Device) -> bool {
        if self.flying.contains(&key) {
            return true;
        }
        if self.to.is_none() {
            let (to, asks) = std::sync::mpsc::channel::<ReadAsk>();
            let (landed, from) = std::sync::mpsc::channel::<ReadLanded>();
            let spawned = std::thread::Builder::new().name("xnw provided texture reads".into()).spawn(move || {
                xnw_core::win::helper_thread();
                for a in asks {
                    let p = (a.provider.lock().unwrap_or_else(|e| e.into_inner()))(a.slot);
                    let made = p.and_then(|p| make_provided(&a.device, &p).map(|(t, up, f, srv)| (t, up, f, srv, p.dim != TEX_2D, p.rows.len() as u64)));
                    if landed.send(ReadLanded(a.epoch, a.key, made)).is_err() {
                        break;
                    }
                }
            });
            if spawned.is_err() {
                return false;
            }
            (self.to, self.from) = (Some(to), Some(from));
        }
        let sent = self.to.as_ref().is_some_and(|t| t.send(ReadAsk { epoch: self.epoch, key, slot, provider: provider.clone(), device: device.clone() }).is_ok());
        if sent {
            self.flying.insert(key);
        }
        sent
    }

    /// The reads landed since the last call (`wait`: blocks for one when
    /// any is flying); those of an older epoch are dropped.
    fn take(&mut self, wait: bool) -> Vec<(u64, Option<Made>)> {
        let mut out = Vec::new();
        let Some(from) = self.from.as_ref() else { return out };
        let mut wait = wait && !self.flying.is_empty();
        loop {
            let r = if wait { from.recv().ok() } else { from.try_recv().ok() };
            let Some(ReadLanded(epoch, key, made)) = r else { break };
            if epoch == self.epoch {
                self.flying.remove(&key);
                out.push((key, made));
                wait = false;
            }
        }
        out
    }

    /// The provider's textures start over: what is flying is not waited for.
    fn restart(&mut self) {
        self.epoch += 1;
        self.flying.clear();
    }
}

/// A provided texture by its key: the upload is kept until its copy ran.
struct ProvidedTex {
    texture: ID3D12Resource,
    upload: Option<ID3D12Resource>,
    srv: D3D12_SHADER_RESOURCE_VIEW_DESC,
    /// Not a plain 2D texture (a cube, an array, 3D): among [`TexTable::wide`].
    shaped: bool,
    used: u64,
}

// SAFETY: D3D12 objects are free-threaded; the table is used by its pass's owner.
unsafe impl Send for TexTable {}

const PS_STATE: u32 = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 as u32 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 as u32;
const COPY_DEST: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;

impl TexTable {
    pub fn new(opts: TexOptions, base: u32) -> TexTable {
        let slots = opts.slots.clamp(1, TEX_SLOTS);
        TexTable {
            opts: TexOptions { slots, ..opts },
            base,
            frame: HashMap::new(),
            why: HashMap::new(),
            // Entries past `slots` may lie outside the heap (terrain's own table): never written.
            entries: Entries::new(slots),
            res: vec![None; slots as usize],
            known: HashMap::new(),
            wide: HashMap::new(),
            held: Vec::new(),
            uploads: Vec::new(),
            checkers: HashMap::new(),
            provider: None,
            keys: None,
            provided: HashMap::new(),
            slot_key: HashMap::new(),
            provided_none: std::collections::HashSet::new(),
            frame_no: 0,
            stats: TexStats::default(),
            last_log: std::time::Instant::now(),
            views: HashMap::new(),
            current: HashMap::new(),
            arrivals: Arrivals::default(),
            reads: ProvidedReads::default(),
            settle: false,
        }
    }

    /// [`batch_on`]: reads the records of `slots` (repeats allowed) and
    /// checks their generations, one look at the traps and one at the views
    /// for all, into [`Self::current`].
    fn read_records(&mut self, slots: &mut dyn Iterator<Item = u64>) {
        self.current.clear();
        let mut list: Vec<u64> = slots.filter(|s| (0x1_0000..0x8000_0000_0000).contains(s)).collect();
        list.sort_unstable();
        list.dedup();
        self.current.reserve(list.len());
        let mut records = Vec::with_capacity(list.len());
        host::copy_records(&list, &mut records);
        let (mut at, mut ids) = (Vec::with_capacity(list.len()), Vec::with_capacity(list.len()));
        for (&slot, (rec, ok)) in list.iter().zip(&records) {
            // SAFETY: our own 32-byte copy of the record.
            if let Some(id) = ok.then(|| unsafe { descriptor::record_id(rec.as_ptr() as u64) }).flatten() {
                at.push(slot);
                ids.push(id);
            }
        }
        let mut now = Vec::with_capacity(ids.len());
        descriptor::current(&ids, &mut now);
        self.current.extend(at.into_iter().zip(ids).zip(now).filter(|(_, c)| *c).map(|(s, _)| s));
    }

    /// The slot's view as [`classify`] gives it, counted; `Err`: flat, why
    /// (its known state goes).
    #[allow(clippy::type_complexity)]
    fn classified(&mut self, slot: u64) -> Result<((u32, u32), View, Option<[u32; 10]>), Flat> {
        match classify(slot) {
            Slot::Placed(id, view) => {
                self.stats.placed += 1;
                let words = match &view {
                    View::Placed(_, _, d) => d[..10].try_into().ok(),
                    _ => None,
                };
                Ok((id, view, words))
            }
            Slot::Shader(id, r, d) => {
                self.stats.shader += 1;
                Ok((id, View::Shader(r, d), d))
            }
            k => {
                let why = match k {
                    Slot::Stale => {
                        self.stats.stale += 1;
                        Flat::Stale
                    }
                    Slot::NoRecord => {
                        self.stats.no_record += 1;
                        Flat::Stale
                    }
                    _ => {
                        self.stats.other += 1;
                        Flat::Not2d
                    }
                };
                self.known.remove(&slot);
                Err(why)
            }
        }
    }

    /// The table's first heap slot (bound as the pixel shader's t3 table).
    pub fn base(&self) -> u32 {
        self.base
    }

    /// The table index and sRGB flag of `slot` this frame (0: flat).
    pub fn lookup(&self, slot: u64) -> (u32, bool) {
        self.frame.get(&slot).copied().unwrap_or((0, false))
    }

    /// The texture and view of `slot` this frame (None: flat).
    pub fn host_view(&self, slot: u64) -> Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)> {
        self.views.get(&self.lookup(slot).0).cloned().or_else(|| self.wide.get(&slot).cloned())
    }

    /// `slot`'s cube, array or 3D view for this frame's [`Self::host_view`].
    /// `XNW_TEX_SHAPED=off`: none is kept, and the title's programs get
    /// the stand-in of the dimension there, as before BUG3;
    /// `cube`, `array` or `3d`: only those are kept (A/B switches).
    fn keep_wide(&mut self, slot: u64, texture: ID3D12Resource, srv: D3D12_SHADER_RESOURCE_VIEW_DESC) {
        static ONLY: std::sync::OnceLock<Option<D3D12_SRV_DIMENSION>> = std::sync::OnceLock::new();
        let only = *ONLY.get_or_init(|| match std::env::var("XNW_TEX_SHAPED").as_deref().map(str::trim) {
            Ok("off" | "0") => Some(D3D12_SRV_DIMENSION_UNKNOWN),
            Ok("cube") => Some(D3D12_SRV_DIMENSION_TEXTURECUBE),
            Ok("array") => Some(D3D12_SRV_DIMENSION_TEXTURE2DARRAY),
            Ok("3d") => Some(D3D12_SRV_DIMENSION_TEXTURE3D),
            _ => None,
        });
        if only.is_none_or(|d| d == srv.ViewDimension) {
            self.wide.insert(slot, (texture, srv));
        }
    }

    /// Why `slot` is flat this frame (asked for draws `lookup` gave 0).
    pub fn flat_reason(&self, slot: u64) -> Flat {
        if slot == 0 {
            return Flat::None;
        }
        self.why.get(&slot).copied().unwrap_or(Flat::Other)
    }

    /// Counts one drawn draw with albedo slot `slot`: bound or flat by reason.
    pub fn count_draw(&mut self, slot: u64) {
        if self.lookup(slot).0 != 0 {
            self.stats.bound_draws += 1;
        } else {
            self.stats.flat_draws += 1;
            self.stats.flat_by[self.flat_reason(slot) as usize] += 1;
        }
    }

    /// The flat split as `none N, full N, ...` (the log line and the replay).
    pub fn flat_split(s: &TexStats) -> String {
        FLAT_REASONS.iter().zip(s.flat_by).map(|(n, v)| format!("{n} {v}")).collect::<Vec<_>>().join(", ")
    }

    /// After the frame's fence: lets go of what the GPU used.
    pub fn finished(&mut self) {
        self.held.clear();
        self.uploads.clear();
    }

    /// Builds this frame's table for `layers` (entries below [`TEX_LOW`])
    /// and `slots` (each draw's albedo slot, 0 for none) on the open
    /// `list`: SRVs into the heap at `cpu(base + i)` for entries whose view
    /// changed, uploads recorded, then every sampled copy moved to a
    /// shader-resource state. Called after the last frame's fence: an
    /// entry rewritten here is read by no submission in flight.
    pub(crate) fn prepare(&mut self, device: &ID3D12Device, list: &ID3D12GraphicsCommandList, layers: &mut dyn Iterator<Item = u64>, slots: &mut dyn Iterator<Item = u64>, cpu: &dyn Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE) {
        self.frame.clear();
        self.wide.clear();
        self.why.clear();
        self.held.clear();
        self.uploads.clear();
        self.frame_no += 1;
        // A new entry count (`tex=<n>`, the replay's `--tex`): every entry starts over.
        let want = self.opts.slots.clamp(1, self.res.len() as u32);
        if want != self.entries.slots {
            let mut fresh = Entries::new(want);
            fresh.nulls = (1..self.res.len() as u32).filter(|&e| self.res[e as usize].is_some()).collect();
            self.entries = fresh;
        }
        self.entries.begin(self.frame_no, !stable_on() || self.opts.mode == TexMode::Off);
        // The frame's records in one batch, and the arrivals since the last frame.
        let live = self.opts.mode == TexMode::Live;
        let (lv, sv): (Vec<u64>, Vec<u64>) = if live && batch_on() { (layers.collect(), slots.collect()) } else { (Vec::new(), Vec::new()) };
        let (mut li, mut si) = (lv.iter().copied(), sv.iter().copied());
        let (layers, slots): (&mut dyn Iterator<Item = u64>, &mut dyn Iterator<Item = u64>) = if live && batch_on() {
            self.read_records(&mut lv.iter().chain(&sv).copied());
            (&mut li, &mut si)
        } else {
            self.current.clear();
            (layers, slots)
        };
        if live && arrivals_on() {
            self.arrivals.take();
        }
        // Reads off the frame cost the frame no time: no time cap.
        let mut budget = Budget::timed(if self.opts.async_reads || self.settle { 0.0 } else { self.opts.frame_ms });
        let mut waiting = 0u64;
        let mut demand = 0u64;
        let mut placed = 0u64;
        let mut copies: Vec<Arc<resource::HostTexture>> = Vec::new();
        let mut checker_copies: Vec<(ID3D12Resource, ID3D12Resource)> = Vec::new();
        let mut provided_copies: Vec<(ID3D12Resource, ID3D12Resource, Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>)> = Vec::new();
        let mut made = HashMap::new();
        // The hand-over (`async_reads`): every read the reader finished is
        // recorded on this frame's list, asked for this frame or not; the
        // slots that waited for them resolve below as ready.
        if live && self.opts.async_reads {
            for (copy, up, source) in texture::pass_finished() {
                self.stats.uploads += 1;
                self.stats.sources[source.min(4)] += 1;
                self.uploads.push((copy, up));
            }
        }
        if self.opts.mode == TexMode::Provided && self.opts.async_reads {
            let landed = self.reads.take(false);
            self.land(landed, &mut provided_copies);
        }
        if self.opts.mode != TexMode::Off {
            let base = self.base;
            let mut get = |t: &mut Self, slot: u64| {
                // A new provided texture past the time cap waits (the first goes).
                let provided_new = t.opts.mode == TexMode::Provided && !t.provided_has(slot);
                match t.opts.mode {
                    // A new checker texture past the time cap waits (the first goes).
                    TexMode::Checker if !t.checkers.contains_key(&slot) && !checker_copies.is_empty() && budget.late() => Err(Flat::Waiting),
                    TexMode::Checker => t.checker(device, slot, &mut checker_copies).map(|(r, v, s)| ((r, v), s)).ok_or(Flat::Other),
                    TexMode::Provided if provided_new && !provided_copies.is_empty() && budget.late() => Err(Flat::Waiting),
                    TexMode::Provided => {
                        let r = t.provide(device, slot, &mut provided_copies).map(|(r, v, s)| ((r, v), s));
                        waiting += u64::from(r.as_ref().is_err_and(|w| *w == Flat::Waiting));
                        r
                    }
                    _ => {
                        let (t0, n) = (std::time::Instant::now(), t.stats.paths);
                        let r = t.resolve(slot, &mut budget, &mut waiting, &mut copies).map(|(r, v, s)| ((r, v), s));
                        let k = (0..5).find(|&i| t.stats.paths[i] != n[i]).unwrap_or(3);
                        t.stats.path_ms[k] += t0.elapsed().as_secs_f64() * 1e3;
                        r
                    }
                }
            };
            let mut place = |t: &mut Self, (resource, srv): (ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC), i: u32| {
                if t.entries.hold(i, view_id(&resource, &srv)) {
                    // SAFETY: a live resource and a slot of our heap's table.
                    unsafe { device.CreateShaderResourceView(&resource, Some(&srv), cpu(base + i)) };
                    t.res[i as usize] = Some(resource.clone());
                }
                made.insert(i, (resource, srv));
            };
            demand = self.fill(layers, true, &mut get, &mut place);
            demand += self.fill(slots, false, &mut get, &mut place);
            placed = made.len() as u64;
        }
        self.views = made;
        // Entries let go and not taken again: null views (the table is
        // bound whole), and their resources go.
        let null = null_view();
        let nulls = self.entries.take_nulls();
        // SAFETY: slots of our heap's table; a null resource makes a null view.
        unsafe {
            device.CreateShaderResourceView(None, Some(&null), cpu(self.base));
            for &i in &nulls {
                device.CreateShaderResourceView(None, Some(&null), cpu(self.base + i));
            }
        }
        for &i in &nulls {
            self.res[i as usize] = None;
        }
        self.stats.nulls += nulls.len() as u64;
        self.stats.waiting = waiting;
        self.stats.distinct = placed;
        self.stats.demand = demand;
        self.stats.resident = self.entries.resident() as u64;
        // Barriers: uploaded copies to COPY_DEST, the copies, then every
        // sampled or uploaded copy to a shader-resource state (the copy's
        // state word is the truth; the title's lists read it too).
        let ups = std::mem::take(&mut self.uploads);
        let mut before = Vec::new();
        for (copy, _) in &ups {
            let real = copy.state.swap(COPY_DEST, std::sync::atomic::Ordering::Relaxed);
            if real != COPY_DEST {
                before.push(barrier(&copy.texture, real, COPY_DEST));
            }
        }
        for (t, _) in &checker_copies {
            before.push(barrier(t, D3D12_RESOURCE_STATE_COMMON.0 as u32, COPY_DEST));
        }
        for (t, _, _) in &provided_copies {
            before.push(barrier(t, D3D12_RESOURCE_STATE_COMMON.0 as u32, COPY_DEST));
        }
        // SAFETY: an open list; the barriers' references are released after.
        unsafe { submit_barriers(list, before) };
        for (copy, up) in &ups {
            self.stats.upload_bytes += up.record(list, &copy.texture);
        }
        for (t, up) in &checker_copies {
            copy_checker(list, t, up);
        }
        for (t, up, f) in &provided_copies {
            copy_footprints(list, t, up, f);
        }
        let mut after = Vec::new();
        for copy in copies.iter().chain(ups.iter().map(|(c, _)| c)) {
            let real = copy.state.load(std::sync::atomic::Ordering::Relaxed);
            if real & D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 as u32 == 0 || real == COPY_DEST {
                copy.state.store(PS_STATE, std::sync::atomic::Ordering::Relaxed);
                after.push(barrier(&copy.texture, real, PS_STATE));
            }
        }
        for (t, _) in &checker_copies {
            after.push(barrier(t, COPY_DEST, PS_STATE));
        }
        for (t, _, _) in &provided_copies {
            after.push(barrier(t, COPY_DEST, PS_STATE));
        }
        // SAFETY: as above.
        unsafe { submit_barriers(list, after) };
        resource::resident(copies.iter().chain(ups.iter().map(|(c, _)| c)));
        self.held.extend(copies);
        self.uploads = ups;
        // Slots not seen for 120 frames go (their copies and entries with
        // them; a live entry never outlasts its known copy). Provided
        // textures no frame used for 8 frames go (an entry holds its own).
        if self.frame_no.is_multiple_of(60) {
            let now = self.frame_no;
            self.known.retain(|_, k| now - k.used <= 120);
            let (live, known) = (self.opts.mode == TexMode::Live, &self.known);
            self.entries.sweep(&|slot, used| now - used <= 120 && (!live || known.contains_key(&slot)));
        }
        let now = self.frame_no;
        self.provided.retain(|_, p| now - p.used <= 8);
    }

    /// This frame's entries: each distinct non-zero slot in order keeps its
    /// entry or takes one (`place`d) when `get` resolves it and the table
    /// has room, else its flat reason; `low`: the layers' (entries below
    /// [`TEX_LOW`]). A slot that does not resolve lets its entry go.
    /// Returns the distinct slots asked for.
    fn fill<R>(&mut self, slots: &mut dyn Iterator<Item = u64>, low: bool, get: &mut dyn FnMut(&mut Self, u64) -> Result<(R, bool), Flat>, place: &mut dyn FnMut(&mut Self, R, u32)) -> u64 {
        let mut demand = 0;
        // Pools found full this frame: [layers', albedo's].
        let mut full = [false; 2];
        for slot in slots {
            if slot == 0 {
                continue;
            }
            // Asked again: a layer's slot placed above the layers' pool this frame moves down.
            if let Some(&(i, _)) = self.frame.get(&slot) {
                if !(low && i >= self.entries.low) {
                    continue;
                }
                demand -= 1;
            }
            demand += 1;
            // A slot without an entry takes a free one before its read, else
            // one made by eviction once it resolves (a slot that does not
            // resolve evicts nothing); past a full pool, none is read.
            let have = self.entries.holds(slot, low);
            if !have && full[usize::from(!low)] {
                self.stats.full += 1;
                self.frame.insert(slot, (0, false));
                self.why.insert(slot, Flat::Full);
                continue;
            }
            let e = if have { self.entries.reserve(slot, low) } else { self.entries.reserve_free(slot, low) };
            match get(self, slot) {
                Ok((r, srgb)) => {
                    let Some(e) = e.or_else(|| self.entries.reserve(slot, low)) else {
                        // A layer found its pool full, albedo both.
                        full[0] = true;
                        full[1] |= !low;
                        self.stats.full += 1;
                        self.frame.insert(slot, (0, false));
                        self.why.insert(slot, Flat::Full);
                        continue;
                    };
                    place(self, r, e);
                    self.frame.insert(slot, (e, srgb));
                    self.why.remove(&slot);
                }
                Err(why) => {
                    self.entries.release(slot);
                    self.frame.insert(slot, (0, false));
                    self.why.insert(slot, why);
                }
            }
        }
        demand
    }

    /// The live path for one slot: its host copy (asking for an upload when
    /// new, re-generated or pending), the SRV and sRGB flag; `Err` = flat, why.
    fn resolve(&mut self, slot: u64, budget: &mut Budget, waiting: &mut u64, copies: &mut Vec<Arc<resource::HostTexture>>) -> Result<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC, bool), Flat> {
        let frame = self.frame_no;
        // `XNW_TEX_BATCH`: a known slot whose record names its entry still
        // (read for all slots at the frame's start) skips `classify`.
        let fast = self.current.get(&slot).copied().filter(|id| self.known.get(&slot).is_some_and(|k| k.id == *id));
        let mut got = None;
        let id = match fast {
            Some(id) => id,
            None => {
                let g = self.classified(slot)?;
                let id = g.0;
                got = Some(g);
                id
            }
        };
        // The kind counters `classified` keeps, for a fast resolve that ends here.
        let kind = |stats: &mut TexStats, placed: bool| {
            if fast.is_some() {
                if placed {
                    stats.placed += 1;
                } else {
                    stats.shader += 1;
                }
            }
        };
        let arrivals = arrivals_on().then_some((&self.arrivals.textures, self.arrivals.all));
        // Known, same generation, contents complete: no refresh (no hashing
        // of title memory every frame).: flat for good, or partly
        // streamed and looked at less than `recheck_frames` ago: as it was.
        if let Some(k) = self.known.get_mut(&slot)
            && k.id == id
        {
            if let Some(why) = k.flat {
                k.used = frame;
                kind(&mut self.stats, k.placed);
                self.stats.paths[1] += 1;
                if why == Flat::Not2d {
                    self.stats.not_2d += 1;
                }
                return Err(why);
            }
            if let Some(copy) = &k.copy
                && known_stands(k.ready, k.flying, copy.pending.load(std::sync::atomic::Ordering::Relaxed), k.partial_at.is_some(), k.partial_at.is_some_and(|at| !partial_due(slot, at, frame, copy.origin.get().map(|o| o.file_texture()), arrivals)))
            {
                k.used = frame;
                kind(&mut self.stats, k.placed);
                self.stats.paths[if k.partial_at.is_some() { 2 } else { 0 }] += 1;
                let (copy, srv, srgb, wide) = (copy.clone(), k.srv, k.srgb, k.wide);
                let t = copy.texture.clone();
                copies.push(copy);
                if wide {
                    self.stats.not_2d += 1;
                    self.keep_wide(slot, t, srv);
                    return Err(Flat::Not2d);
                }
                return Ok((t, srv, srgb));
            }
        }
        // A fast slot that is due: its view as `classify` gives it now.
        let (id, view, words) = match got {
            Some(g) => g,
            None => self.classified(slot)?,
        };
        // The frame's read time spent, a slot with no texels yet waits
        // without the layer's lookups (each would be refused by the budget).
        if budget.late() && !self.known.get(&slot).is_some_and(|k| k.id == id && k.ready) {
            self.stats.paths[4] += 1;
            *waiting += 1;
            return Err(Flat::Waiting);
        }
        self.stats.paths[3] += 1;
        let placed = matches!(view, View::Placed(..));
        let flat = |t: &mut Self, why: Flat| {
            t.known.insert(slot, Known { id, copy: None, srv: D3D12_SHADER_RESOURCE_VIEW_DESC::default(), srgb: false, ready: false, ours: false, used: frame, wide: false, flat: Some(why), partial_at: None, placed, flying: false });
            Err(why)
        };
        // Ask our layer: the resource (held for the call) and its copy.
        let (this, _held_placed, _held_live) = match &view {
            View::Placed(..) => {
                let (v, held) = descriptor::host::placed_texture(&view).ok_or(Flat::Other)?;
                let View::Shader(this, _) = v else { return Err(Flat::Other) };
                (this, Some(held), None)
            }
            View::Shader(r, _) => (*r, None, Some(resource::hold_live(*r as xnw_gpu::d3d12x::object::This).ok_or(Flat::Other)?)),
            _ => return Err(Flat::Other),
        };
        if texture::gpu_output(this as xnw_gpu::d3d12x::object::This) {
            return flat(self, Flat::GpuOut);
        }
        let o = self.opts;
        // With `async_reads` the read goes to the pass's reader thread (the hold with it).
        let pt = if o.async_reads {
            texture::for_pass_async(this as xnw_gpu::d3d12x::object::This, Box::new(Hold { _placed: _held_placed, _live: _held_live }), &mut |b| budget.room(&o, b))
        } else {
            texture::for_pass(this as xnw_gpu::d3d12x::object::This, true, &mut |b| budget.room(&o, b))
        };
        let Some(pt) = pt else {
            self.stats.no_copy += 1;
            return Err(Flat::Other);
        };
        let ours_before = self.known.get(&slot).is_some_and(|k| k.ours && k.id == id);
        // SAFETY: a live resource.
        let d = unsafe { pt.copy.texture.GetDesc() };
        let srv = srv_desc(&d, words, pt.first_mip, pt.copy.dropped);
        // `landed`: its upload was taken at the frame's start ([`texture::pass_finished`]).
        let uploading = pt.upload.is_some() || pt.landed;
        self.stats.again += u64::from(pt.landed && ours_before);
        if let Some(up) = pt.upload {
            // Recorded even when we cannot sample it: the copy now counts
            // as uploaded for everyone.
            self.stats.uploads += 1;
            self.stats.sources[pt.source.min(4)] += 1;
            self.stats.again += u64::from(ours_before);
            self.uploads.push((pt.copy.clone(), up));
        }
        if pt.waiting {
            *waiting += 1;
        }
        let (srv, srgb, wide) = match srv {
            Some((srv, srgb)) => (srv, srgb, false),
            None => {
                self.stats.not_2d += 1;
                // Not plain 2D: a view of its own dimension for the bridged
                // rows (their programs sample 2D arrays: a 2D view there read
                // zeros, black roads and window frames in a test run); else
                // flat for good (its upload, if any, was recorded above).
                let Some(w) = wide_srv(&d, words, pt.first_mip, pt.copy.dropped) else {
                    return flat(self, Flat::Not2d);
                };
                (w, false, true)
            }
        };
        // `pt.ready` alone: an upload none of whose mips is whole is recorded, not sampled.
        let ready = pt.ready;
        let partial_at = (ready && !uploading && pt.copy.pending.load(std::sync::atomic::Ordering::Relaxed)).then_some(frame);
        self.known.insert(slot, Known { id, copy: Some(pt.copy.clone()), srv, srgb, ready, ours: uploading || ours_before, used: frame, wide, flat: None, partial_at, placed, flying: pt.flying });
        if wide {
            if ready {
                self.keep_wide(slot, pt.copy.texture.clone(), srv);
                copies.push(pt.copy);
            }
            return Err(Flat::Not2d);
        }
        if !ready {
            return Err(Flat::Waiting);
        }
        let t = pt.copy.texture.clone();
        copies.push(pt.copy);
        Ok((t, srv, srgb))
    }

    /// Checker mode: an 8x8 checker of two colours from the slot's hash.
    fn checker(&mut self, device: &ID3D12Device, slot: u64, uploads: &mut Vec<(ID3D12Resource, ID3D12Resource)>) -> Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC, bool)> {
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
        };
        if let Some((t, up)) = self.checkers.get_mut(&slot) {
            // The upload buffer is kept one frame past its copy.
            *up = None;
            return Some((t.clone(), srv, false));
        }
        let (t, up) = make_checker(device, slot)?;
        uploads.push((t.clone(), up.clone()));
        self.checkers.insert(slot, (t.clone(), Some(up)));
        self.stats.uploads += 1;
        Some((t, srv, false))
    }

    /// A new provider (a new frame's pack): its textures are made afresh.
    pub fn set_provider(&mut self, p: Option<Provider>) {
        self.set_provider_keyed(p, None);
    }

    /// A new provider with its content keys: a slot whose key a texture
    /// already holds takes it again (no read, no upload); without keys
    /// every texture is made afresh.
    /// `XNW_TEX_KEYS=off`: keys are dropped (every frame's pack makes its
    /// textures afresh, the rule before).
    pub fn set_provider_keyed(&mut self, p: Option<Provider>, keys: Option<Keys>) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| !matches!(std::env::var("XNW_TEX_KEYS").as_deref().map(str::trim), Ok("off" | "0")));
        let keys = keys.filter(|_| on);
        if keys.is_none() || self.keys.is_none() {
            self.provided.clear();
            self.provided_none.clear();
            self.reads.restart();
        }
        self.provider = p.map(|p| Arc::new(std::sync::Mutex::new(p)));
        self.keys = keys;
        self.slot_key.clear();
    }

    /// `slot`'s key with this provider (the slot itself without keys).
    fn key_of(&mut self, slot: u64) -> Option<u64> {
        let Some(keys) = self.keys.as_mut() else { return Some(slot) };
        *self.slot_key.entry(slot).or_insert_with(|| keys(slot))
    }

    /// Whether a texture for `slot` is made already.
    fn provided_has(&mut self, slot: u64) -> bool {
        self.key_of(slot).is_some_and(|k| self.provided.contains_key(&k))
    }

    /// Reads the reader thread finished: each texture is kept by its key
    /// and its copy queued on this frame's list (`uploads`); a key without
    /// texels is remembered as such.
    fn land(&mut self, landed: Vec<(u64, Option<Made>)>, uploads: &mut Vec<(ID3D12Resource, ID3D12Resource, Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>)>) {
        for (key, made) in landed {
            let Some((t, up, f, srv, shaped, bytes)) = made else {
                self.provided_none.insert(key);
                continue;
            };
            uploads.push((t.clone(), up.clone(), f));
            self.provided.insert(key, ProvidedTex { texture: t, upload: Some(up), srv, shaped, used: self.frame_no });
            self.stats.uploads += 1;
            self.stats.upload_bytes += bytes;
        }
    }

    /// A provided slot's texture, made and its upload queued on first use
    /// of its key (`async_reads`: made by the reader thread, the slot
    /// `Waiting` until it lands). A cube, an array or a 3D texture goes to
    /// [`Self::wide`] and is `Not2d` here.
    fn provide(&mut self, device: &ID3D12Device, slot: u64, uploads: &mut Vec<(ID3D12Resource, ID3D12Resource, Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>)>) -> Result<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC, bool), Flat> {
        let key = self.key_of(slot).ok_or(Flat::Other)?;
        let frame = self.frame_no;
        let mut fresh = false;
        if self.opts.async_reads && !self.provided.contains_key(&key) && !self.provided_none.contains(&key) {
            let provider = self.provider.clone().ok_or(Flat::Other)?;
            if self.reads.ask(key, slot, &provider, device) {
                if !self.settle {
                    return Err(Flat::Waiting);
                }
                while self.reads.flying.contains(&key) {
                    let landed = self.reads.take(true);
                    if landed.is_empty() {
                        break; // the thread is gone
                    }
                    self.land(landed, uploads);
                }
                fresh = true;
            }
        }
        let (texture, srv, shaped) = if let Some(p) = self.provided.get_mut(&key) {
            // The upload buffer is kept one frame past its copy (it ran: the last frame's fence).
            if p.used != frame {
                p.upload = None;
            }
            p.used = frame;
            self.stats.provided_kept += u64::from(!fresh);
            (p.texture.clone(), p.srv, p.shaped)
        } else {
            if (self.keys.is_some() || self.opts.async_reads) && self.provided_none.contains(&key) {
                return Err(Flat::Other);
            }
            let provider = self.provider.clone().ok_or(Flat::Other)?;
            let Some(p) = (provider.lock().unwrap_or_else(|e| e.into_inner()))(slot) else {
                self.provided_none.insert(key);
                return Err(Flat::Other);
            };
            let (t, up, f, srv) = make_provided(device, &p).ok_or(Flat::Other)?;
            let shaped = p.dim != TEX_2D;
            uploads.push((t.clone(), up.clone(), f));
            self.provided.insert(key, ProvidedTex { texture: t.clone(), upload: Some(up), srv, shaped, used: frame });
            self.stats.uploads += 1;
            self.stats.upload_bytes += p.rows.len() as u64;
            (t, srv, shaped)
        };
        if shaped {
            self.keep_wide(slot, texture, srv);
            return Err(Flat::Not2d);
        }
        Ok((texture, srv, false))
    }

    /// The per-second lines (and the counters, the waiting exits),
    /// then a new period.
    pub fn log_line(&mut self) -> Option<[String; 3]> {
        if self.last_log.elapsed().as_secs_f64() < 1.0 {
            return None;
        }
        self.last_log = std::time::Instant::now();
        let mut s = std::mem::take(&mut self.stats);
        let e = &mut self.entries;
        (s.written, s.kept, s.evicted) = (std::mem::take(&mut e.written), std::mem::take(&mut e.kept), std::mem::take(&mut e.evicted));
        Some([
            {
                let (p, r, n, ms) = texture::pass_reader_counts();
                format!("world render: textures: bound {}, flat {} ({}), uploaded {} ({:.1} MiB), waiting {}; reads off the frame: pending {p}, ready {r}, done {n}, read {ms:.0} ms in total; copies refused (bounds) {}", s.bound_draws, s.flat_draws, Self::flat_split(&s), s.uploads, s.upload_bytes as f64 / (1 << 20) as f64, s.waiting, texture::PASS_COPY_REFUSED.load(std::sync::atomic::Ordering::Relaxed))
            },
            format!(
                "world render: textures: {} distinct in the last frame of {} asked (table {}); slots placed {}, shader {}, other {}, stale {}, no record {}; uploads from pack {}, pages {}, memory {}, hashed memory {}, unknown {}, again {}; not 2D {}, no copy {}, table full {}; entries ({}): resident {}, views written {}, kept {}, nulled {}, evicted {}; provided kept {}; resolves (count/ms): known {}/{:.1}, flat known {}/{:.1}, partial kept {}/{:.1}, looked up {}/{:.1}, waited past the read time {}/{:.1}",
                s.distinct, s.demand, self.opts.slots, s.placed, s.shader, s.other, s.stale, s.no_record, s.sources[0], s.sources[1], s.sources[2], s.sources[3], s.sources[4], s.again, s.not_2d, s.no_copy, s.full, if stable_on() { "stable" } else { "per frame" }, s.resident, s.written, s.kept, s.nulls, s.evicted, s.provided_kept,
                s.paths[0], s.path_ms[0], s.paths[1], s.path_ms[1], s.paths[2], s.path_ms[2], s.paths[3], s.path_ms[3], s.paths[4], s.path_ms[4]
            ),
            // Why asks gave no new texels (the exits).
            texture::exits_text(),
        ])
    }
}

/// The null view of a table entry: a 2D texture that reads zeros.
fn null_view() -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
    }
}

/// A null view at every one of the `declared` entries of a table whose
/// first heap slot is `base`, once, when its heap is made. The table is
/// bound whole as the root signature declares it, and [`TexTable::prepare`]
/// writes only the entries draws take or let go: an entry no draw had taken
/// yet was never written. Under the debug layer that was id 646 "has not
/// been initialized" for the pass's `albedoTex` range at every draw (1024
/// stored messages on pk6 f1205) since the table went from 1024 to 4096
/// entries; the layer's text: on resource binding tier 2 hardware a
/// declared table must be populated and initialised.
pub(crate) fn null_table(device: &ID3D12Device, cpu: &dyn Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE, base: u32, declared: u32) {
    let null = null_view();
    // SAFETY: `declared` slots of the caller's heap from `base`; a null resource makes a null view.
    unsafe {
        for i in 0..declared {
            device.CreateShaderResourceView(None, Some(&null), cpu(base + i));
        }
    }
}

/// POL4: the title view's component mapping (its words[2], the D3D12 view
/// description's `Shader4ComponentMapping`) when it is a valid one (the
/// always-set bit 12, nothing above it), else the default. The title's
/// programs read a texture through its view's swizzle (a one- or
/// two-channel mask read as r and b, a greyscale read as rrr); our layer's
/// own view of it keeps it (xnw-gpu descriptor/host.rs `srv_desc` takes the
/// title's ten words whole). `XNW_TEX_MAPPING=off`: the default mapping,
/// the rule before.
fn view_mapping(words: Option<[u32; 10]>) -> u32 {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| !matches!(std::env::var("XNW_TEX_MAPPING").as_deref().map(str::trim), Ok("off" | "0")));
    mapping_of(words, on)
}

fn mapping_of(words: Option<[u32; 10]>, on: bool) -> u32 {
    match words.map(|w| w[2]) {
        Some(m) if on && m & 0x1000 != 0 && m >> 13 == 0 => m,
        _ => D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
    }
}

/// Our SRV of a host copy: 2D, one slice, one sample; the view's format
/// when the copy is typeless (else the copy's own), mips from the finest
/// with contents (and the title's MostDetailedMip, moved by the mips the
/// copy dropped), the view's component mapping ([`view_mapping`]). `None`
/// for anything but a plain 2D texture.
fn srv_desc(d: &D3D12_RESOURCE_DESC, words: Option<[u32; 10]>, first_mip: u32, dropped: u32) -> Option<(D3D12_SHADER_RESOURCE_VIEW_DESC, bool)> {
    const TEXTURE2D_VIEW: u32 = 4;
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.DepthOrArraySize != 1 || d.SampleDesc.Count > 1 || d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) {
        return None;
    }
    if words.is_some_and(|w| w[1] != TEXTURE2D_VIEW) {
        return None;
    }
    let host = d.Format.0 as u32;
    let format = if resource::is_typeless(host) {
        let v = words.map_or(0, |w| resource::pc_format(w[0]));
        if v != 0 && resource::family(v) == host { v } else { resource::first_concrete(host) }
    } else {
        host
    };
    let mips = u32::from(d.MipLevels).max(1);
    let title_most = words.map_or(0, |w| w[4]).saturating_sub(dropped);
    let most = first_mip.max(title_most).min(mips - 1);
    Some((
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT(format as i32),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: view_mapping(words),
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: most, MipLevels: mips - most, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
        },
        is_srgb(format),
    ))
}

/// A view of a host copy that is not plain 2D, of the dimension the
/// title's view has (its words: 5 2D array, 8 3D, 9 cube, 10 cube array;
/// a shader view without words: the resource's own shape), all slices,
/// mips as [`srv_desc`]. `None`: a multisampled or depth copy, or a kind
/// none of these.
fn wide_srv(d: &D3D12_RESOURCE_DESC, words: Option<[u32; 10]>, first_mip: u32, dropped: u32) -> Option<D3D12_SHADER_RESOURCE_VIEW_DESC> {
    const ARRAY: u32 = 5;
    const VOLUME: u32 = 8;
    const CUBE: u32 = 9;
    const CUBE_ARRAY: u32 = 10;
    if d.SampleDesc.Count > 1 || d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) {
        return None;
    }
    let kind = match (words.map(|w| w[1]), d.Dimension) {
        (Some(k), _) => k,
        (None, D3D12_RESOURCE_DIMENSION_TEXTURE3D) => VOLUME,
        (None, D3D12_RESOURCE_DIMENSION_TEXTURE2D) if d.DepthOrArraySize > 1 => ARRAY,
        _ => return None,
    };
    let host = d.Format.0 as u32;
    let format = if resource::is_typeless(host) {
        let v = words.map_or(0, |w| resource::pc_format(w[0]));
        if v != 0 && resource::family(v) == host { v } else { resource::first_concrete(host) }
    } else {
        host
    };
    let mips = u32::from(d.MipLevels).max(1);
    let most = first_mip.max(words.map_or(0, |w| w[4]).saturating_sub(dropped)).min(mips - 1);
    let slices = u32::from(d.DepthOrArraySize).max(1);
    let mut v = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT(format as i32), Shader4ComponentMapping: view_mapping(words), ..Default::default() };
    match (kind, d.Dimension) {
        (ARRAY, D3D12_RESOURCE_DIMENSION_TEXTURE2D) => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            v.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MostDetailedMip: most, MipLevels: mips - most, FirstArraySlice: 0, ArraySize: slices, PlaneSlice: 0, ResourceMinLODClamp: 0.0 };
        }
        (CUBE, D3D12_RESOURCE_DIMENSION_TEXTURE2D) if slices >= 6 => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            v.Anonymous.TextureCube = D3D12_TEXCUBE_SRV { MostDetailedMip: most, MipLevels: mips - most, ResourceMinLODClamp: 0.0 };
        }
        (CUBE_ARRAY, D3D12_RESOURCE_DIMENSION_TEXTURE2D) if slices >= 6 => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
            v.Anonymous.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV { MostDetailedMip: most, MipLevels: mips - most, First2DArrayFace: 0, NumCubes: slices / 6, ResourceMinLODClamp: 0.0 };
        }
        (VOLUME, D3D12_RESOURCE_DIMENSION_TEXTURE3D) => {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            v.Anonymous.Texture3D = D3D12_TEX3D_SRV { MostDetailedMip: most, MipLevels: mips - most, ResourceMinLODClamp: 0.0 };
        }
        _ => return None,
    }
    Some(v)
}

fn barrier(r: &ID3D12Resource, from: u32, to: u32) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(r.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATES(from as i32),
                StateAfter: D3D12_RESOURCE_STATES(to as i32),
            }),
        },
    }
}

/// # Safety
/// `list` is open.
unsafe fn submit_barriers(list: &ID3D12GraphicsCommandList, b: Vec<D3D12_RESOURCE_BARRIER>) {
    if b.is_empty() {
        return;
    }
    // SAFETY: the caller's; each barrier's reference is released after.
    unsafe { list.ResourceBarrier(&b) };
    for b in b {
        // SAFETY: a transition barrier built by `barrier`.
        drop(ManuallyDrop::into_inner(unsafe { ManuallyDrop::into_inner(b.Anonymous.Transition) }.pResource));
    }
}

/// Two colours from a slot's hash (8 x 8 texels, 2 x 2 squares).
pub fn checker_texels(slot: u64) -> Vec<u8> {
    let h = slot.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let c = |s: u32| [64 + ((h >> s) & 0x7F) as u8, 64 + ((h >> (s + 8)) & 0x7F) as u8, 64 + ((h >> (s + 16)) & 0x7F) as u8, 255];
    let (a, b) = (c(0), c(24).map(|v| v.saturating_add(60)));
    (0..64).flat_map(|i| if ((i % 8) / 2 + (i / 8) / 2) % 2 == 0 { a } else { b }).collect()
}

fn make_checker(device: &ID3D12Device, slot: u64) -> Option<(ID3D12Resource, ID3D12Resource)> {
    let d = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 8, Height: 8, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
    let mut t: Option<ID3D12Resource> = None;
    let mut up: Option<ID3D12Resource> = None;
    let texels = checker_texels(slot);
    let b = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 8 * 256, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
    // SAFETY: descriptions outlive the calls; the mapping covers 8 rows of 256 bytes.
    unsafe {
        device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, None, &mut t).ok()?;
        device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &b, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up).ok()?;
        let up = up?;
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        up.Map(0, None, Some(&mut p)).ok()?;
        for row in 0..8 {
            std::ptr::copy_nonoverlapping(texels[row * 32..row * 32 + 32].as_ptr(), p.cast::<u8>().add(row * 256), 32);
        }
        up.Unmap(0, None);
        Some((t?, up))
    }
}

fn copy_checker(list: &ID3D12GraphicsCommandList, t: &ID3D12Resource, up: &ID3D12Resource) {
    let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: 8, Height: 8, Depth: 1, RowPitch: 256 } };
    let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(t.clone())), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
    let src = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(up.clone())), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint } };
    // SAFETY: an open list; both locations name live resources.
    unsafe { list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
    drop(ManuallyDrop::into_inner(dst.pResource));
    drop(ManuallyDrop::into_inner(src.pResource));
}

/// The host texture of a provided texture: its description, its
/// subresources and the view of its shape. A cube is a 2D texture of six
/// slices with a cube view (a cube array's first cube); an array one
/// subresource a slice and mip; a 3D texture one a mip.
pub(crate) fn provided_shape(p: &Provided) -> (D3D12_RESOURCE_DESC, u32, D3D12_SHADER_RESOURCE_VIEW_DESC) {
    let format = DXGI_FORMAT(p.format as i32);
    let mips = p.mips.clamp(1, 15);
    let slices = p.depth.clamp(1, u32::from(u16::MAX));
    let (dimension, height, depth, subs) = match p.dim {
        TEX_1D => (D3D12_RESOURCE_DIMENSION_TEXTURE1D, 1, 1, mips),
        TEX_3D => (D3D12_RESOURCE_DIMENSION_TEXTURE3D, p.height, slices, mips),
        TEX_2D_ARRAY | TEX_CUBE => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, p.height, slices, slices * mips),
        _ => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, p.height, 1, mips),
    };
    let desc = D3D12_RESOURCE_DESC { Dimension: dimension, Width: u64::from(p.width), Height: height, DepthOrArraySize: depth as u16, MipLevels: mips as u16, Format: format, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
    let (view, a) = match p.dim {
        TEX_1D => (D3D12_SRV_DIMENSION_TEXTURE1D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture1D: D3D12_TEX1D_SRV { MipLevels: mips, ..Default::default() } }),
        TEX_3D => (D3D12_SRV_DIMENSION_TEXTURE3D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_SRV { MipLevels: mips, ..Default::default() } }),
        TEX_2D_ARRAY => (D3D12_SRV_DIMENSION_TEXTURE2DARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: mips, ArraySize: slices, ..Default::default() } }),
        TEX_CUBE if slices >= 6 => (D3D12_SRV_DIMENSION_TEXTURECUBE, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { TextureCube: D3D12_TEXCUBE_SRV { MipLevels: mips, ..Default::default() } }),
        _ => (D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: mips, ..Default::default() } }),
    };
    (desc, subs, D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: view, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: a })
}

/// `GetCopyableFootprints` of the first `subs` subresources of `desc`:
/// the footprints, each one's rows and row bytes, and the bytes in all.
pub(crate) fn footprints(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, subs: u32) -> (Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>, Vec<u32>, Vec<u64>, u64) {
    let n = subs as usize;
    let (mut fps, mut rows, mut row_size, mut total) = (vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n], vec![0u32; n], vec![0u64; n], 0u64);
    // SAFETY: outputs for `subs` subresources.
    unsafe { device.GetCopyableFootprints(desc, 0, subs, 0, Some(fps.as_mut_ptr()), Some(rows.as_mut_ptr()), Some(row_size.as_mut_ptr()), Some(&mut total)) };
    (fps, rows, row_size, total)
}

/// Copies `p`'s rows to the mapped upload buffer `to` at [`footprints`]'
/// places, subresource after subresource. One mip: the source rows are
/// `row_bytes` apart, `row_count` of them (slices and depth slices follow
/// one another). More mips: packed, each row its own mip's length. Rows
/// the source does not hold stay as the buffer has them.
///
/// # Safety
/// `to` is writable for the footprints' total bytes.
pub(crate) unsafe fn provided_fill(p: &Provided, fps: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT], rows: &[u32], row_size: &[u64], to: *mut u8) {
    let packed = p.mips > 1;
    let end = if packed { p.rows.len() } else { p.rows.len().min(p.row_bytes as usize * p.row_count as usize) };
    let mut from = 0usize;
    for (i, fp) in fps.iter().enumerate() {
        let stride = if packed { row_size[i] as usize } else { p.row_bytes as usize };
        let n = (row_size[i] as usize).min(stride);
        let count = rows[i] as usize * fp.Footprint.Depth as usize;
        for y in 0..count {
            let at = from + y * stride;
            if at + n > end {
                break;
            }
            // SAFETY: row `y` of subresource `i` lies inside the footprints' bytes (the caller's `to`).
            unsafe { std::ptr::copy_nonoverlapping(p.rows[at..].as_ptr(), to.add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize), n) };
        }
        from += count * stride;
    }
}

/// A texture of `p`'s shape and format and an upload buffer holding its
/// subresources at their footprints, with the view of its shape.
fn make_provided(device: &ID3D12Device, p: &Provided) -> Option<(ID3D12Resource, ID3D12Resource, Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>, D3D12_SHADER_RESOURCE_VIEW_DESC)> {
    let (d, subs, srv) = provided_shape(p);
    if p.mips <= 1 && p.rows.len() < (p.row_bytes * p.row_count) as usize {
        return None;
    }
    let (fps, rows, row_size, total) = footprints(device, &d, subs);
    if total == 0 || total == u64::MAX {
        return None;
    }
    let b = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
    let mut t: Option<ID3D12Resource> = None;
    let mut up: Option<ID3D12Resource> = None;
    // SAFETY: descriptions outlive the calls; the mapping covers the footprints' `total` bytes.
    unsafe {
        device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, None, &mut t).ok()?;
        device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &b, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up).ok()?;
        let up = up?;
        let mut m: *mut std::ffi::c_void = std::ptr::null_mut();
        up.Map(0, None, Some(&mut m)).ok()?;
        provided_fill(p, &fps, &rows, &row_size, m.cast());
        up.Unmap(0, None);
        Some((t?, up, fps, srv))
    }
}

fn copy_footprints(list: &ID3D12GraphicsCommandList, t: &ID3D12Resource, up: &ID3D12Resource, footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT]) {
    for (i, f) in footprints.iter().enumerate() {
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(t.clone())), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(up.clone())), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *f } };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        drop(ManuallyDrop::into_inner(dst.pResource));
        drop(ManuallyDrop::into_inner(src.pResource));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_gpu::d3d12x::descriptor::{create_placed_srv, create_srv, forget};

    /// The slot resolve on synthetic records: a shader view of a resource,
    /// a placed view of texture memory, a placed buffer view, a released
    /// resource (stale), zeros and an address outside the space.
    #[test]
    fn slots_classify_by_their_records() {
        let null = std::ptr::null();
        let slots = [0u64; 4 * 6];
        let at = |i: usize| slots.as_ptr() as u64 + 32 * i as u64;
        let r = 0x7a11_0000usize as xnw_gpu::d3d12x::object::This;
        create_srv(null, r, std::ptr::null(), at(0));
        match classify(at(0)) {
            Slot::Shader(_, res, None) => assert_eq!(res, 0x7a11_0000),
            s => panic!("shader view: {s:?}"),
        }
        // A placed 2D texture view of memory (resource dimension 3).
        let mut resource = [0u32; 14];
        resource[0] = 3;
        let mut desc = [0u32; 16];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (desc[10], desc[11]) = (0x5000_0000, 0);
        create_placed_srv(null, &resource, &desc, at(1));
        assert!(matches!(classify(at(1)), Slot::Placed(_, View::Placed(false, _, _))), "{:?}", classify(at(1)));
        // A placed buffer view: not a texture.
        resource[0] = 1;
        desc[1] = 1;
        create_placed_srv(null, &resource, &desc, at(2));
        assert_eq!(classify(at(2)), Slot::Other);
        // Released: the record's generation is gone.
        forget(0x7a11_0000);
        assert_eq!(classify(at(0)), Slot::Stale);
        assert_eq!(classify(at(5)), Slot::NoRecord, "zeros");
        assert_eq!(classify(0x10), Slot::NoRecord, "below the space");
        assert_eq!(classify(u64::MAX - 64), Slot::NoRecord, "past the space");
    }

    /// PERF2, the pending -> ready hand-over of a slot whose read is off
    /// the frame: looked up every frame while the read is with the reader
    /// (whatever the copy's own state says: the reader sets it when it has
    /// read, before the pass has the texels), answered from the table once
    /// it is not; a partial view whose copy completed is looked up once more.
    #[test]
    fn a_flying_slot_is_looked_up_until_its_read_is_handed_over() {
        // (ready, flying, pending, partial, partial fresh)
        assert!(!known_stands(false, true, true, false, false), "queued, no texels: flat, asked again");
        assert!(!known_stands(false, true, false, false, false), "read (copy says complete), not handed over: asked again");
        assert!(!known_stands(true, true, false, false, false), "old contents shown while read again: asked again");
        assert!(known_stands(true, false, false, false, false), "handed over, complete: the table's answer");
        assert!(!known_stands(false, false, false, false, false), "never ready: asked");
        assert!(known_stands(true, false, true, true, true), "partly streamed, looked at lately: kept");
        assert!(!known_stands(true, false, true, true, false), "partly streamed, due");
        assert!(!known_stands(true, false, true, false, false), "pending without texels shown");
        assert!(!known_stands(true, false, false, true, true), "a partial view of a copy complete since: once more");
    }

    /// PERF2: the provided reads' bookkeeping without a device: a key is
    /// asked once, results of an older provider epoch are dropped.
    #[test]
    fn provided_reads_drop_what_an_older_provider_asked() {
        let mut r = ProvidedReads::default();
        let (landed, from) = std::sync::mpsc::channel::<ReadLanded>();
        r.from = Some(from);
        r.flying.insert(7);
        r.restart();
        assert!(r.flying.is_empty());
        r.flying.insert(9);
        landed.send(ReadLanded(0, 7, None)).unwrap();
        landed.send(ReadLanded(1, 9, None)).unwrap();
        let got = r.take(true);
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), [9]);
        assert!(r.flying.is_empty() && r.take(true).is_empty(), "nothing flying: no wait");
    }

    /// A slot on a page with nothing mapped reads as no record (guarded copy).
    #[test]
    fn unmapped_slot_reads_nothing() {
        // A reserved, never committed page.
        // SAFETY: a plain reservation, released at the end.
        let p = unsafe { xnw_core::win::VirtualAlloc(std::ptr::null(), 1 << 16, xnw_core::win::MEM_RESERVE, xnw_core::win::PAGE_READWRITE) };
        assert!(!p.is_null());
        assert_eq!(classify(p as u64 + 64), Slot::NoRecord);
        // SAFETY: the reservation made above.
        unsafe { xnw_core::win::VirtualFree(p, 0, xnw_core::win::MEM_RELEASE) };
    }

    /// batch: the frame's records read for all slots name the same
    /// entries `classify` finds, and only current ones (released, zeros,
    /// unmapped and out-of-space slots are not there: they take the full path).
    #[test]
    fn batched_records_match_classify() {
        let null = std::ptr::null();
        let slots = [0u64; 4 * 6];
        let at = |i: usize| slots.as_ptr() as u64 + 32 * i as u64;
        let r = 0x7a12_0000usize as xnw_gpu::d3d12x::object::This;
        create_srv(null, r, std::ptr::null(), at(0));
        let mut resource = [0u32; 14];
        resource[0] = 3;
        let mut desc = [0u32; 16];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (desc[10], desc[11]) = (0x5000_0000, 0);
        create_placed_srv(null, &resource, &desc, at(1));
        // SAFETY: a plain reservation, released at the end.
        let p = unsafe { xnw_core::win::VirtualAlloc(std::ptr::null(), 1 << 16, xnw_core::win::MEM_RESERVE, xnw_core::win::PAGE_READWRITE) };
        let mut t = TexTable::new(TexOptions::default(), 0);
        let ask = [at(0), at(1), at(0), at(5), p as u64 + 64, 0x10, 0];
        t.read_records(&mut ask.iter().copied());
        let id = |s| match classify(s) {
            Slot::Shader(id, ..) | Slot::Placed(id, _) => id,
            k => panic!("{k:?}"),
        };
        assert_eq!(t.current.len(), 2, "{:?}", t.current);
        assert_eq!(t.current.get(&at(0)), Some(&id(at(0))));
        assert_eq!(t.current.get(&at(1)), Some(&id(at(1))));
        forget(0x7a12_0000);
        t.read_records(&mut ask.iter().copied());
        assert_eq!(t.current.keys().copied().collect::<Vec<_>>(), vec![at(1)], "released: stale");
        // SAFETY: the reservation made above.
        unsafe { xnw_core::win::VirtualFree(p, 0, xnw_core::win::MEM_RELEASE) };
    }

    /// arrivals: a partly streamed copy is due when its texture's
    /// pages arrived, when told to look at all, or on its turn of the sweep
    /// (once every 4 x `XNW_TEX_RECHECK` frames, spread by address); off,
    /// `XNW_TEX_RECHECK` frames after its last look.
    #[test]
    fn partial_copies_due_by_arrival_or_sweep() {
        let n = recheck_frames();
        assert!(!partial_due(0x1000, 10, 10 + n - 1, None, None));
        assert!(partial_due(0x1000, 10, 10 + n, None, None));
        let mut arrived = std::collections::HashSet::new();
        arrived.insert((2, 7));
        let on = Some((&arrived, false));
        assert!(partial_due(0x1000, 10, 11, Some((2, 7)), on), "its pages arrived");
        assert!(partial_due(0x1000, 10, 11, None, Some((&arrived, true))), "look at all");
        // Over one sweep period a slot is due once; on every frame some address is.
        let p = 4 * n;
        let due = |slot: u64| (0..p).filter(|&f| partial_due(slot, 0, 100 + f, Some((2, 8)), on)).count();
        assert_eq!(due(0x1000), 1);
        assert_eq!(due(0x1020), 1);
        assert!((0..p).all(|f| (0..p).any(|s| partial_due(s << 5, 0, f, None, on))), "spread");
    }

    /// bench (`cargo test -p xnw-world --lib -- --ignored resolve_cost --nocapture`):
    /// 1,240 slots, `classify` each against the batch (records and
    /// generations for all, then a map look per slot), alone and with a
    /// thread writing descriptors (the views' lock) and one reading title
    /// memory (the guards' lock) meanwhile. Prints us per slot.
    #[test]
    #[ignore]
    fn resolve_cost() {
        const N: usize = 1240;
        let null = std::ptr::null();
        let slots = vec![0u64; 4 * N];
        let at = |i: usize| slots.as_ptr() as u64 + 32 * i as u64;
        for i in 0..N {
            create_srv(null, (0x7b00_0000 + 0x100 * i) as xnw_gpu::d3d12x::object::This, std::ptr::null(), at(i));
        }
        let ask: Vec<u64> = (0..N).map(at).collect();
        let mut t = TexTable::new(TexOptions::default(), 0);
        let run = |t: &mut TexTable, label: &str| {
            let frames = 200;
            let t0 = std::time::Instant::now();
            for _ in 0..frames {
                for &s in &ask {
                    std::hint::black_box(classify(s));
                }
            }
            let per = t0.elapsed().as_secs_f64() * 1e6 / (frames * N) as f64;
            let t0 = std::time::Instant::now();
            for _ in 0..frames {
                t.read_records(&mut ask.iter().copied());
                for &s in &ask {
                    std::hint::black_box(t.current.get(&s));
                }
            }
            let batch = t0.elapsed().as_secs_f64() * 1e6 / (frames * N) as f64;
            // Its parts: the records' copies, the generations' look.
            let mut recs = Vec::new();
            let t0 = std::time::Instant::now();
            for _ in 0..frames {
                host::copy_records(&ask, &mut recs);
            }
            let copies = t0.elapsed().as_secs_f64() * 1e6 / (frames * N) as f64;
            // SAFETY: our own copies of the records.
            let ids: Vec<(u32, u32)> = recs.iter().filter_map(|(r, _)| unsafe { descriptor::record_id(r.as_ptr() as u64) }).collect();
            let mut now = Vec::new();
            let t0 = std::time::Instant::now();
            for _ in 0..frames {
                descriptor::current(&ids, &mut now);
            }
            let gens = t0.elapsed().as_secs_f64() * 1e6 / (frames * N) as f64;
            println!(
                "resolve_cost {label}: classify {per:.3} us a slot ({:.2} ms a frame), batch {batch:.3} us a slot ({:.2} ms a frame; copies {copies:.3}, generations {gens:.3})",
                per * N as f64 / 1e3,
                batch * N as f64 / 1e3
            );
            assert_eq!(t.current.len(), N);
        };
        run(&mut t, "alone");
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let other = vec![0u64; 4 * 64];
        let other_at = other.as_ptr() as u64;
        let mut busy = Vec::new();
        {
            let stop = stop.clone();
            busy.push(std::thread::spawn(move || {
                let mut i = 0usize;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    create_srv(std::ptr::null(), (0x7c00_0000 + 0x100 * (i % 64)) as xnw_gpu::d3d12x::object::This, std::ptr::null(), other_at + 32 * (i % 64) as u64);
                    i += 1;
                }
            }));
        }
        {
            let stop = stop.clone();
            busy.push(std::thread::spawn(move || {
                let mut b = [0u8; 64];
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    host::copy_mapped(b.as_mut_ptr(), other_at as *const u8, 64, false);
                }
            }));
        }
        run(&mut t, "with a descriptor writer and a title-memory reader");
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for b in busy {
            b.join().unwrap();
        }
        drop(other);
    }

    #[test]
    fn upload_budget_counts_and_bytes() {
        let o = TexOptions { uploads: 2, upload_bytes: 100, ..Default::default() };
        let mut b = Budget::default();
        assert!(b.room(&o, 500), "the first of a frame always goes");
        assert!(!b.room(&o, 1), "bytes spent");
        let mut b = Budget::default();
        assert!(b.room(&o, 10) && b.room(&o, 10));
        assert!(!b.room(&o, 1), "count spent");
        assert_eq!(b, Budget { count: 2, bytes: 20, until: None });
        // The time cap (CUT1): past it only the frame's first read goes.
        let o = TexOptions { frame_ms: 1.0, ..Default::default() };
        let mut b = Budget::timed(o.frame_ms);
        assert!(b.room(&o, 1) && b.room(&o, 1), "in time");
        std::thread::sleep(std::time::Duration::from_millis(3));
        assert!(b.late() && !b.room(&o, 1), "late");
        let mut b = Budget::timed(o.frame_ms);
        std::thread::sleep(std::time::Duration::from_millis(3));
        assert!(b.room(&o, 1), "the first goes even late");
        assert!(!Budget::timed(0.0).late(), "0: no cap");
    }

    #[test]
    fn our_views_are_2d_from_the_finest_mip_with_contents() {
        let d = |format: i32, mips: u16, array: u16| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 64, Height: 64, DepthOrArraySize: array, MipLevels: mips, Format: DXGI_FORMAT(format), SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
        let mip = |v: &D3D12_SHADER_RESOURCE_VIEW_DESC| {
            // SAFETY: a Texture2D view.
            unsafe { (v.Anonymous.Texture2D.MostDetailedMip, v.Anonymous.Texture2D.MipLevels) }
        };
        // BC1 typeless copy, title view BC1_UNORM_SRGB, MostDetailedMip 2, one mip dropped, contents from mip 3.
        let mut w = [0u32; 10];
        w[..6].copy_from_slice(&[72, 4, 0x1688, 0, 2, 7]);
        let (v, srgb) = srv_desc(&d(70, 6, 1), Some(w), 3, 1).unwrap();
        assert_eq!((v.Format.0, srgb, mip(&v)), (72, true, (3, 3)));
        let (v, srgb) = srv_desc(&d(70, 6, 1), Some(w), 0, 1).unwrap();
        assert_eq!((v.Format.0, srgb, mip(&v)), (72, true, (1, 5)), "title's MostDetailedMip less the dropped one");
        // A typed copy keeps its own format; no words: the resource's own view.
        let (v, srgb) = srv_desc(&d(28, 1, 1), None, 9, 0).unwrap();
        assert_eq!((v.Format.0, srgb, mip(&v)), (28, false, (0, 1)), "clamped to the last mip");
        assert!(srv_desc(&d(28, 1, 6), None, 0, 0).is_none(), "arrays/cubes: flat");
        w[1] = 9; // a cube view
        assert!(srv_desc(&d(70, 6, 1), Some(w), 0, 0).is_none());
        // POL4: the title view's component mapping (words[2]) is ours: a BC5 mask read as (r, 0, g, 1)
        // = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(0, 4, 1, 5); invalid words (no bit 12, bits past it) or
        // no words: the default; off: the default.
        let rgb = 0 | 4 << 3 | 1 << 6 | 5 << 9 | 0x1000;
        w[1] = 4;
        w[2] = rgb;
        assert_eq!(srv_desc(&d(83, 6, 1), Some(w), 0, 0).unwrap().0.Shader4ComponentMapping, rgb);
        assert_eq!(wide_srv(&d(83, 6, 6), Some([w[0], 5, rgb, 0, 0, 6, 0, 6, 0, 0]), 0, 0).unwrap().Shader4ComponentMapping, rgb);
        for bad in [rgb & !0x1000, rgb | 0x2000, 0] {
            assert_eq!(mapping_of(Some([0, 4, bad, 0, 0, 0, 0, 0, 0, 0]), true), D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
        }
        assert_eq!((mapping_of(None, true), mapping_of(Some(w), false)), (D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING));
        assert_eq!(D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 0x1688, "the default is (0, 1, 2, 3) with bit 12");
    }

    /// A provided cube and a provided array on the device (WARP): made,
    /// uploaded and viewed as the table does it, then sampled by a compute
    /// program. Each face's texels answer the direction of that face in
    /// D3D12's order (+x, -x, +y, -y, +z, -z = slices 0-5, the console's
    /// order too), each mip its own level, each array slice its index.
    #[test]
    fn provided_cube_and_array_texels_reach_a_sampler() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let (_gpu, host) = crate::tests::warp_host();
        let dev = &host.device;
        // 4x4 then 2x2 RGBA8 a slice, packed: red = 40 x slice + 10, green = 100 x mip + 50.
        let rows = |slices: u32| (0..slices).flat_map(|s| [(0u8, 16), (1, 4)].into_iter().flat_map(move |(m, n)| std::iter::repeat_n([40 * s as u8 + 10, 100 * m + 50, 0, 255], n).flatten())).collect::<Vec<u8>>();
        let cube = Provided { width: 4, height: 4, format: 28, rows: rows(6), row_bytes: 16, row_count: 24, dim: TEX_CUBE, depth: 6, mips: 2 };
        let array = Provided { width: 4, height: 4, format: 28, rows: rows(3), row_bytes: 16, row_count: 12, dim: TEX_2D_ARRAY, depth: 3, mips: 2 };
        let src = "#define RS \"DescriptorTable(SRV(t0, numDescriptors = 2)), UAV(u0), StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_POINT)\"
TextureCube<float4> c : register(t0); Texture2DArray<float4> a : register(t1); SamplerState s : register(s0); RWByteAddressBuffer o : register(u0);
static const float3 D[6] = { float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1) };
[RootSignature(RS)][numthreads(1, 1, 1)] void cs() { for (uint i = 0; i < 12; i++) { float4 v = c.SampleLevel(s, D[i % 6], i / 6); o.Store(4 * i, uint(v.x * 255 + 0.5) | uint(v.y * 255 + 0.5) << 8); } for (uint k = 0; k < 6; k++) { float4 v = a.SampleLevel(s, float3(0.5, 0.5, k % 3), k / 3); o.Store(48 + 4 * k, uint(v.x * 255 + 0.5) | uint(v.y * 255 + 0.5) << 8); } }
";
        let cs = crate::compile_source(src, c"cs", c"cs_5_1").expect("compile");
        let e = |e: windows::core::Error| e.to_string();
        // SAFETY: our device, list and live resources; the readback is mapped after the fence.
        let got: Vec<u32> = unsafe {
            let bytes = std::slice::from_raw_parts(cs.GetBufferPointer().cast::<u8>(), cs.GetBufferSize());
            let root: ID3D12RootSignature = dev.CreateRootSignature(0, bytes).map_err(e).expect("root");
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: std::mem::transmute_copy(&root), CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.GetBufferPointer(), BytecodeLength: cs.GetBufferSize() }, ..Default::default() };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).map_err(e).expect("pso");
            let out = crate::bridge::buffer(dev, 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON).expect("out");
            let rb = crate::bridge::buffer(dev, 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST).expect("rb");
            let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 2, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).map_err(e).expect("heap");
            let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(e).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e).expect("fence");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e).expect("list");
            let mut keep = Vec::new();
            for (i, p) in [&cube, &array].into_iter().enumerate() {
                let (t, up, fps, srv) = make_provided(dev, p).expect("texture");
                assert_eq!(fps.len(), p.depth as usize * 2);
                submit_barriers(&list, vec![barrier(&t, D3D12_RESOURCE_STATE_COMMON.0 as u32, COPY_DEST)]);
                copy_footprints(&list, &t, &up, &fps);
                submit_barriers(&list, vec![barrier(&t, COPY_DEST, PS_STATE)]);
                dev.CreateShaderResourceView(&t, Some(&srv), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: heap.GetCPUDescriptorHandleForHeapStart().ptr + i * step });
                keep.push((t, up));
            }
            submit_barriers(&list, vec![barrier(&out, D3D12_RESOURCE_STATE_COMMON.0 as u32, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32)]);
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
            list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            submit_barriers(&list, vec![barrier(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32)]);
            list.CopyBufferRegion(&rb, 0, &out, 0, 72);
            list.Close().map_err(e).expect("close");
            queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(e).expect("cast"))]);
            queue.Signal(&fence, 1).map_err(e).expect("signal");
            while fence.GetCompletedValue() < 1 {
                dev.GetDeviceRemovedReason().map_err(e).expect("device kept");
                std::thread::yield_now();
            }
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).map_err(e).expect("map");
            let v = std::slice::from_raw_parts(p.cast::<u32>(), 18).to_vec();
            rb.Unmap(0, None);
            v
        };
        let want = |slice: u32, mip: u32| (40 * slice + 10) | (100 * mip + 50) << 8;
        assert_eq!(got[..12], std::array::from_fn::<u32, 12, _>(|i| want(i as u32 % 6, i as u32 / 6)), "cube: face by direction, then mip 1");
        assert_eq!(got[12..], std::array::from_fn::<u32, 6, _>(|k| want(k as u32 % 3, k as u32 / 3)), "array: slice, then mip 1");
    }

    /// A provided cube is six slices with their mips and a cube view; its
    /// packed rows land subresource by subresource, each row at its own
    /// mip's length. One mip keeps the rule the draw bridge had.
    #[test]
    fn provided_shapes_and_their_rows() {
        let p = |dim: u8, depth: u32, mips: u32, rows: Vec<u8>| Provided { width: 4, height: 4, format: 28, rows, row_bytes: 16, row_count: 4 * depth, dim, depth, mips };
        let (d, subs, v) = provided_shape(&p(TEX_CUBE, 6, 3, Vec::new()));
        assert_eq!((d.Dimension, d.DepthOrArraySize, d.MipLevels, subs, v.ViewDimension), (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 6, 3, 18, D3D12_SRV_DIMENSION_TEXTURECUBE));
        let (d, subs, v) = provided_shape(&p(TEX_3D, 4, 2, Vec::new()));
        assert_eq!((d.Dimension, d.DepthOrArraySize, subs, v.ViewDimension), (D3D12_RESOURCE_DIMENSION_TEXTURE3D, 4, 2, D3D12_SRV_DIMENSION_TEXTURE3D));
        let (d, subs, v) = provided_shape(&p(TEX_2D_ARRAY, 5, 1, Vec::new()));
        assert_eq!((d.DepthOrArraySize, subs, v.ViewDimension), (5, 5, D3D12_SRV_DIMENSION_TEXTURE2DARRAY));
        let (d, subs, v) = provided_shape(&p(TEX_2D, 1, 1, Vec::new()));
        assert_eq!((d.DepthOrArraySize, d.MipLevels, subs, v.ViewDimension), (1, 1, 1, D3D12_SRV_DIMENSION_TEXTURE2D));
        // Two slices of two mips (4x4 then 2x2 RGBA8), packed: 64 + 16 bytes a slice; footprints at a 256-byte pitch.
        let fp = |offset: u64, w: u32| D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: offset, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT(28), Width: w, Height: w, Depth: 1, RowPitch: 256 } };
        let fps = [fp(0, 4), fp(1024, 2), fp(2048, 4), fp(3072, 2)];
        let src: Vec<u8> = (0..160u32).map(|i| i as u8).collect();
        let mut to = vec![0xEEu8; 4096];
        // SAFETY: `to` holds the footprints' bytes.
        unsafe { provided_fill(&p(TEX_2D_ARRAY, 2, 2, src.clone()), &fps, &[4, 2, 4, 2], &[16, 8, 16, 8], to.as_mut_ptr()) };
        assert_eq!((&to[..16], &to[256..272], to[16]), (&src[..16], &src[16..32], 0xEE));
        assert_eq!((&to[1024..1032], &to[1024 + 256..1024 + 264]), (&src[64..72], &src[72..80]));
        assert_eq!((&to[2048..2064], &to[3072 + 256..3072 + 264]), (&src[80..96], &src[152..160]));
        // One mip, two slices: rows `row_bytes` apart, slice after slice; a short source leaves the rest.
        let mut to = vec![0xEEu8; 2048];
        // SAFETY: as above.
        unsafe { provided_fill(&p(TEX_2D_ARRAY, 2, 1, src[..112].to_vec()), &[fp(0, 4), fp(1024, 4)], &[4, 4], &[16, 16], to.as_mut_ptr()) };
        assert_eq!((&to[768..784], &to[1024..1040], &to[1024 + 512..1024 + 528], to[1024 + 768]), (&src[48..64], &src[64..80], &src[96..112], 0xEE));
    }

    /// Every flat draw is counted under one reason; the table's entries
    /// go in slot order, the rest are `full`; textured draws are not split.
    /// The next frame keeps each slot's entry and writes no view again.
    #[test]
    fn flat_draws_split_by_reason() {
        let mut t = TexTable::new(TexOptions { mode: TexMode::Checker, slots: 3, ..Default::default() }, 0);
        // Two entries (entry 0 is flat): 10 and 18 take them; 12 stale, 13 not 2D, 14 waiting, 15 other; 16, 17 find the table full.
        let why = |s: u64| match s {
            12 => Err(Flat::Stale),
            13 => Err(Flat::Not2d),
            14 => Err(Flat::Waiting),
            15 => Err(Flat::Other),
            19 => Err(Flat::GpuOut),
            _ => Ok((s, s == 11)),
        };
        let asked = [10u64, 0, 12, 10, 13, 14, 15, 19, 18, 11, 16, 17, 16];
        let frame = |t: &mut TexTable, n: u64| {
            t.frame.clear();
            t.why.clear();
            t.entries.begin(n, false);
            let mut placed = Vec::new();
            let demand = t.fill(&mut asked.iter().copied(), false, &mut |_, s| why(s), &mut |t, v, i| {
                if t.entries.hold(i, v) {
                    placed.push(i);
                }
            });
            (demand, placed)
        };
        assert_eq!(frame(&mut t, 1), (10, vec![1, 2]), "11 comes after the table's last entry");
        assert_eq!(t.entries.resident(), 2);
        assert_eq!((t.lookup(10), t.lookup(18)), ((1, false), (2, false)));
        assert_eq!(t.lookup(11), (0, false));
        assert_eq!(t.flat_reason(11), Flat::Full);
        assert_eq!(t.stats.full, 3, "11, 16, 17");
        // One draw per asked slot (17 twice), plus a draw with no slot.
        for s in [10, 10, 0, 12, 13, 14, 15, 18, 11, 16, 17, 17, 99, 19, 19] {
            t.count_draw(s);
        }
        let s = t.stats;
        assert_eq!((s.bound_draws, s.flat_draws), (3, 12));
        assert_eq!(s.flat_by, [1, 4, 1, 1, 1, 2, 2], "none, full (11, 16, 17 x2), stale, not2d, waiting, other (15 and the never-asked 99), gpuout (19 x2)");
        assert_eq!(s.flat_by.iter().sum::<u64>(), s.flat_draws);
        assert_eq!(TexTable::flat_split(&s), "none 1, full 4, stale 1, not2d 1, waiting 1, other 2, gpuout 2");
        // The next frame: 10 and 18 keep entries 1 and 2, no view is written again.
        assert_eq!(frame(&mut t, 2), (10, vec![]));
        assert_eq!((t.lookup(10), t.lookup(18), t.entries.kept), ((1, false), (2, false), 2));
        // A table with room: nothing is full.
        t.entries = Entries::new(64);
        assert_eq!(frame(&mut t, 3), (10, vec![1, 2, 3, 4, 5]));
        assert_eq!(t.flat_reason(19), Flat::GpuOut);
        assert_eq!((t.lookup(11), t.flat_reason(14)), ((3, true), Flat::Waiting));
        assert!(!t.why.values().any(|w| *w == Flat::Full));
    }

    /// Entries stay with their slots across frames; a full pool lets go of
    /// its least recently used eighth at once (never an entry this frame
    /// uses) and the next new slots take the rest of that room; a slot
    /// that stops resolving lets its entry go, written null once.
    #[test]
    fn entries_stay_and_the_least_recently_used_go() {
        let mut e = Entries::new(TEX_LOW + 16);
        e.begin(1, false);
        let first: Vec<u32> = (1..=16u64).map(|s| e.reserve(s, false).unwrap()).collect();
        assert_eq!(first, (TEX_LOW..TEX_LOW + 16).collect::<Vec<_>>(), "albedo entries from TEX_LOW up, lowest first");
        for (i, &x) in first.iter().enumerate() {
            assert!(e.hold(x, 100 + i as u64));
        }
        assert!(e.take_nulls().is_empty());
        // Frame 2 uses 1..=8 and a new slot: the high pool is full, two of 9..=16 go.
        e.begin(2, false);
        for s in 1..=8u64 {
            assert_eq!(e.reserve(s, false), Some(first[s as usize - 1]));
            assert!(!e.hold(first[s as usize - 1], 100 + s - 1), "same view: not written");
        }
        let n = e.reserve(100, false).unwrap();
        e.hold(n, 200);
        assert!(n >= TEX_LOW && first[8..].contains(&n));
        assert_eq!(e.evicted, 2, "one eighth of 16");
        let m = e.reserve(101, false).unwrap();
        e.hold(m, 201);
        assert_eq!(e.evicted, 2, "the second takes the room the first made");
        assert!(first[8..].contains(&m) && m != n);
        for s in 1..=8u64 {
            assert_eq!(e.entry(s), Some(first[s as usize - 1]), "this frame's entries stay");
        }
        assert!(e.take_nulls().is_empty(), "both freed entries were taken again");
        // Frame 3: slot 1 stops resolving; its entry is written null.
        e.begin(3, false);
        e.reserve(1, false);
        e.release(1);
        assert_eq!(e.take_nulls(), vec![first[0]]);
        assert_eq!(e.entry(1), None);
        // The per-frame rule (`XNW_TEX_STABLE=off`): every entry goes, new ones from the lowest.
        e.begin(4, true);
        assert_eq!(e.resident(), 0);
        assert_eq!(e.reserve(7, false), Some(TEX_LOW));
        assert_eq!(e.take_nulls().len(), 15, "the 15 resident before; 7 takes an entry freed in frame 3");
    }

    /// Layers take entries below TEX_LOW (the shader's 8 bits): a slot held
    /// above moves down; albedo past its pool falls back below; a pool
    /// full of this frame's entries is full.
    #[test]
    fn layers_stay_below_their_bound() {
        let mut e = Entries::new(TEX_LOW + 1);
        e.begin(1, false);
        assert_eq!(e.reserve(5, false), Some(TEX_LOW));
        e.hold(TEX_LOW, 9);
        assert_eq!(e.reserve(6, false), Some(1), "albedo past its pool");
        e.begin(2, false);
        assert_eq!(e.reserve(5, true), Some(2), "a layer's slot moves below TEX_LOW");
        assert_eq!(e.take_nulls(), vec![TEX_LOW]);
        let mut small = Entries::new(3);
        small.begin(1, false);
        assert_eq!((small.reserve(1, true), small.reserve(2, false), small.reserve(3, true)), (Some(1), Some(2), None));
    }

    /// A table smaller than `TEX_SLOTS` (terrain's) never writes past its own entries.
    #[test]
    fn table_writes_stay_within_its_entries() {
        let mut t = TexTable::new(TexOptions { slots: 40, ..Default::default() }, 0);
        assert_eq!((t.entries.slots, t.res.len()), (40, 40));
        t.entries.begin(1, false);
        let got: Vec<u32> = (1..100u64).filter_map(|s| t.entries.reserve(s, s % 2 == 0)).collect();
        assert_eq!(got.len(), 39);
        assert!(got.iter().all(|&e| (1..40).contains(&e)));
        assert_eq!(TexTable::new(TexOptions { slots: u32::MAX, ..Default::default() }, 0).opts.slots, TEX_SLOTS);
    }

    /// The shader's table, its root range and its clamp are `TEX_SLOTS`.
    #[test]
    fn albedo_table_matches_tex_slots() {
        let src = super::super::RAW_SHADER;
        assert!(src.contains(&format!("Texture2D albedoTex[{TEX_SLOTS}] : register(t3);")));
        assert!(src.contains(&format!("SRV(t3, numDescriptors={TEX_SLOTS})")));
        assert!(src.contains(&format!("albedoTex[min(texIndex, {})]", TEX_SLOTS - 1)));
        assert_eq!((layer_entry(0), layer_entry(255), layer_entry(TEX_LOW), layer_entry(TEX_SLOTS - 1)), (0, 255, 0, 0));
    }

    #[test]
    fn checkers_differ_per_slot() {
        let (a, b) = (checker_texels(0x1000), checker_texels(0x2000));
        assert_eq!(a.len(), 256);
        assert_ne!(a, b);
        assert_ne!(a[0..4], a[8..12], "two colours");
    }
}