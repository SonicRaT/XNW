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
/// `tex=<n>` uses fewer. The most distinct slots (albedo and layers)
/// one frame asked for over every frame of two captures is 756
/// (the replay's `asked`), all list rows at most 821:
/// 1024 leaves 35 % over the first, 25 % over the second.
pub const TEX_SLOTS: u32 = 1024;
/// Entries used unless `tex=<n>` says otherwise.
pub const TEX_DEFAULT: u32 = TEX_SLOTS;

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
    /// The draw bridge's view of it: [`TEX_2D`], [`TEX_1D`] or
    /// [`TEX_3D`] or [`TEX_2D_ARRAY`] (`depth` slices, `row_count` = height x depth). The
    /// pass takes 2D only.
    pub dim: u8,
    pub depth: u32,
}

/// [`Provided::dim`] (D3D12 resource dimensions).
pub const TEX_1D: u8 = 2;
pub const TEX_2D: u8 = 3;
pub const TEX_3D: u8 = 4;
/// A 2D array (LIT5): `depth` slices, `row_count` = height x depth, the
/// slices' rows in order (D3D12 has no array dimension; past the three).
pub const TEX_2D_ARRAY: u8 = 5;

/// What gives a slot's texels in [`TexMode::Provided`].
pub type Provider = Box<dyn FnMut(u64) -> Option<Provided> + Send>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexOptions {
    pub mode: TexMode,
    /// Table entries used (1..=TEX_SLOTS; entry 0 is flat).
    pub slots: u32,
    /// Uploads per frame at most, and their bytes (the first always goes).
    pub uploads: u32,
    pub upload_bytes: u64,
    /// Reads (pack, pages, title memory) on the pass's reader thread,
    /// never in the frame (screen mode; `texasync=off`). Off: in the frame.
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
    next: u32,
    /// Table entries written last frame (rewritten null when unused).
    written: u32,
    known: HashMap<u64, Known>,
    /// Copies sampled this frame and uploads recorded (kept to the fence).
    held: Vec<Arc<resource::HostTexture>>,
    uploads: Vec<(Arc<resource::HostTexture>, texture::PassUpload)>,
    checkers: HashMap<u64, (ID3D12Resource, Option<ID3D12Resource>)>,
    /// The provider and its textures (texture, upload until copied, format).
    provider: Option<Provider>,
    provided: HashMap<u64, (ID3D12Resource, Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)>, u32)>,
    frame_no: u64,
    pub stats: TexStats,
    last_log: std::time::Instant,
    /// This frame's entries: their texture and view (the title's own
    /// programs in our pass make their own views of them, `bridged`).
    views: HashMap<u32, (ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)>,
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
            next: 1,
            // Entries past `slots` may lie outside the heap (terrain's own table).
            written: slots,
            known: HashMap::new(),
            held: Vec::new(),
            uploads: Vec::new(),
            checkers: HashMap::new(),
            provider: None,
            provided: HashMap::new(),
            frame_no: 0,
            stats: TexStats::default(),
            last_log: std::time::Instant::now(),
            views: HashMap::new(),
        }
    }

    /// The table's first heap slot (bound as the pixel shader's t3 table).
    pub fn base(&self) -> u32 {
        self.base
    }

    /// The table index and sRGB flag of `slot` this frame (0: flat).
    pub fn lookup(&self, slot: u64) -> (u32, bool) {
        let (i, srgb) = self.frame.get(&slot).copied().unwrap_or((0, false));
        if i < self.next { (i, srgb) } else { (0, false) }
    }

    /// The texture and view of `slot` this frame (None: flat).
    pub fn host_view(&self, slot: u64) -> Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)> {
        self.views.get(&self.lookup(slot).0).cloned()
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

    /// Builds this frame's table for `slots` (each draw's albedo slot, 0 for
    /// none) on the open `list`: SRVs into the heap at `cpu(base + i)`,
    /// uploads recorded, then every sampled copy moved to a shader-resource
    /// state.
    pub(crate) fn prepare(&mut self, device: &ID3D12Device, list: &ID3D12GraphicsCommandList, slots: &mut dyn Iterator<Item = u64>, cpu: &dyn Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE) {
        self.frame.clear();
        self.why.clear();
        self.next = 1;
        self.held.clear();
        self.uploads.clear();
        self.frame_no += 1;
        let mut budget = Budget::timed(self.opts.frame_ms);
        let mut waiting = 0u64;
        let mut demand = 0u64;
        let mut copies: Vec<Arc<resource::HostTexture>> = Vec::new();
        let mut checker_copies: Vec<(ID3D12Resource, ID3D12Resource)> = Vec::new();
        let mut provided_copies: Vec<(ID3D12Resource, ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> = Vec::new();
        let mut made = HashMap::new();
        if self.opts.mode != TexMode::Off {
            let base = self.base;
            demand = self.fill(
                slots,
                &mut |t, slot| match t.opts.mode {
                    // A new checker or provided texture past the time cap waits (the first goes).
                    TexMode::Checker if !t.checkers.contains_key(&slot) && !checker_copies.is_empty() && budget.late() => Err(Flat::Waiting),
                    TexMode::Checker => t.checker(device, slot, &mut checker_copies).map(|(r, v, s)| ((r, v), s)).ok_or(Flat::Other),
                    TexMode::Provided if !t.provided.contains_key(&slot) && !provided_copies.is_empty() && budget.late() => Err(Flat::Waiting),
                    TexMode::Provided => t.provide(device, slot, &mut provided_copies).map(|(r, v, s)| ((r, v), s)).ok_or(Flat::Other),
                    _ => t.resolve(slot, &mut budget, &mut waiting, &mut copies).map(|(r, v, s)| ((r, v), s)),
                },
                &mut |(resource, srv), i| {
                    // SAFETY: a live resource and a slot of our heap's table.
                    unsafe { device.CreateShaderResourceView(&resource, Some(&srv), cpu(base + i)) };
                    made.insert(i, (resource, srv));
                },
            );
        }
        self.views = made;
        // Entries used last frame and not now: null views (the table is
        // bound whole).
        let null = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
        };
        // SAFETY: slots of our heap's table; a null resource makes a null view.
        unsafe {
            device.CreateShaderResourceView(None, Some(&null), cpu(self.base));
            for i in self.next..self.written.max(self.next) {
                device.CreateShaderResourceView(None, Some(&null), cpu(self.base + i));
            }
        }
        self.written = self.next;
        self.stats.waiting = waiting;
        self.stats.distinct = u64::from(self.next - 1);
        self.stats.demand = demand;
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
            copy_footprint(list, t, up, *f);
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
        // Slots not seen for 120 frames go (their copies with them).
        if self.frame_no.is_multiple_of(60) {
            let now = self.frame_no;
            self.known.retain(|_, k| now - k.used <= 120);
        }
    }

    /// This frame's entries: each distinct non-zero slot in order gets the
    /// next entry (`place`d) when `get` resolves it and the table has room,
    /// else its flat reason. Returns the distinct slots asked for.
    fn fill<R>(&mut self, slots: &mut dyn Iterator<Item = u64>, get: &mut dyn FnMut(&mut Self, u64) -> Result<(R, bool), Flat>, place: &mut dyn FnMut(R, u32)) -> u64 {
        let mut demand = 0;
        for slot in slots {
            if slot == 0 || self.frame.contains_key(&slot) {
                continue;
            }
            demand += 1;
            if self.next >= self.opts.slots {
                self.stats.full += 1;
                self.frame.insert(slot, (0, false));
                self.why.insert(slot, Flat::Full);
                continue;
            }
            match get(self, slot) {
                Ok((r, srgb)) => {
                    place(r, self.next);
                    self.frame.insert(slot, (self.next, srgb));
                    self.next += 1;
                }
                Err(why) => {
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
        let (id, view, words) = match classify(slot) {
            Slot::Placed(id, view) => {
                self.stats.placed += 1;
                let words = match &view {
                    View::Placed(_, _, d) => d[..10].try_into().ok(),
                    _ => None,
                };
                (id, view, words)
            }
            Slot::Shader(id, r, d) => {
                self.stats.shader += 1;
                (id, View::Shader(r, d), d)
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
                return Err(why);
            }
        };
        // Known, same generation, contents complete: no refresh (no hashing
        // of title memory every frame).
        if let Some(k) = self.known.get_mut(&slot)
            && k.id == id
            && let Some(copy) = &k.copy
            && k.ready
            && !copy.pending.load(std::sync::atomic::Ordering::Relaxed)
        {
            k.used = frame;
            let (copy, srv, srgb) = (copy.clone(), k.srv, k.srgb);
            let t = copy.texture.clone();
            copies.push(copy);
            return Ok((t, srv, srgb));
        }
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
            self.known.remove(&slot);
            return Err(Flat::GpuOut);
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
        let uploading = pt.upload.is_some();
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
        let Some((srv, srgb)) = srv else {
            self.stats.not_2d += 1;
            return Err(Flat::Not2d);
        };
        // `pt.ready` alone: an upload none of whose mips is whole is recorded, not sampled.
        let ready = pt.ready;
        self.known.insert(slot, Known { id, copy: Some(pt.copy.clone()), srv, srgb, ready, ours: uploading || ours_before, used: frame });
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
        self.provider = p;
        self.provided.clear();
    }

    /// A provided slot's texture, made and its upload queued on first use.
    fn provide(&mut self, device: &ID3D12Device, slot: u64, uploads: &mut Vec<(ID3D12Resource, ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)>) -> Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC, bool)> {
        let srv = |format: u32| D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT(format as i32),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
        };
        if let Some((t, up, format)) = self.provided.get_mut(&slot) {
            *up = None;
            return Some((t.clone(), srv(*format), false));
        }
        let p = self.provider.as_mut()?(slot)?;
        let (t, up, f) = make_provided(device, &p)?;
        uploads.push((t.clone(), up.clone(), f));
        self.provided.insert(slot, (t.clone(), Some((up, f)), p.format));
        self.stats.uploads += 1;
        self.stats.upload_bytes += p.rows.len() as u64;
        Some((t, srv(p.format), false))
    }

    /// The per-second lines (and the counters, the waiting exits),
    /// then a new period.
    pub fn log_line(&mut self) -> Option<[String; 3]> {
        if self.last_log.elapsed().as_secs_f64() < 1.0 {
            return None;
        }
        self.last_log = std::time::Instant::now();
        let s = std::mem::take(&mut self.stats);
        Some([
            {
                let (p, r, n, ms) = texture::pass_reader_counts();
                format!("world render: textures: bound {}, flat {} ({}), uploaded {} ({:.1} MiB), waiting {}; reads off the frame: pending {p}, ready {r}, done {n}, read {ms:.0} ms in total; copies refused (bounds) {}", s.bound_draws, s.flat_draws, Self::flat_split(&s), s.uploads, s.upload_bytes as f64 / (1 << 20) as f64, s.waiting, texture::PASS_COPY_REFUSED.load(std::sync::atomic::Ordering::Relaxed))
            },
            format!(
                "world render: textures: {} distinct in the last frame of {} asked (table {}); slots placed {}, shader {}, other {}, stale {}, no record {}; uploads from pack {}, pages {}, memory {}, hashed memory {}, unknown {}, again {}; not 2D {}, no copy {}, table full {}",
                s.distinct, s.demand, self.opts.slots, s.placed, s.shader, s.other, s.stale, s.no_record, s.sources[0], s.sources[1], s.sources[2], s.sources[3], s.sources[4], s.again, s.not_2d, s.no_copy, s.full
            ),
            // Why asks gave no new texels (the exits).
            texture::exits_text(),
        ])
    }
}

/// Our SRV of a host copy: 2D, one slice, one sample; the view's format
/// when the copy is typeless (else the copy's own), mips from the finest
/// with contents (and the title's MostDetailedMip, moved by the mips the
/// copy dropped). `None` for anything but a plain 2D texture.
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
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: most, MipLevels: mips - most, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
        },
        is_srgb(format),
    ))
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

/// A 2D texture of `p`'s size and format and an upload buffer holding
/// its rows at 256-byte aligned pitch.
fn make_provided(device: &ID3D12Device, p: &Provided) -> Option<(ID3D12Resource, ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let format = DXGI_FORMAT(p.format as i32);
    let d = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: u64::from(p.width), Height: p.height, DepthOrArraySize: 1, MipLevels: 1, Format: format, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
    let pitch = p.row_bytes.div_ceil(256) * 256;
    if p.rows.len() < (p.row_bytes * p.row_count) as usize {
        return None;
    }
    let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: p.width, Height: p.height, Depth: 1, RowPitch: pitch } };
    let b = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: u64::from(pitch) * u64::from(p.row_count), Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
    let mut t: Option<ID3D12Resource> = None;
    let mut up: Option<ID3D12Resource> = None;
    // SAFETY: descriptions outlive the calls; the mapping covers `row_count` rows of `pitch` bytes.
    unsafe {
        device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, None, &mut t).ok()?;
        device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &b, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up).ok()?;
        let up = up?;
        let mut m: *mut std::ffi::c_void = std::ptr::null_mut();
        up.Map(0, None, Some(&mut m)).ok()?;
        let rb = p.row_bytes as usize;
        for row in 0..p.row_count as usize {
            std::ptr::copy_nonoverlapping(p.rows[row * rb..row * rb + rb].as_ptr(), m.cast::<u8>().add(row * pitch as usize), rb);
        }
        up.Unmap(0, None);
        Some((t?, up, footprint))
    }
}

fn copy_footprint(list: &ID3D12GraphicsCommandList, t: &ID3D12Resource, up: &ID3D12Resource, footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
    let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(t.clone())), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
    let src = D3D12_TEXTURE_COPY_LOCATION { pResource: ManuallyDrop::new(Some(up.clone())), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint } };
    // SAFETY: an open list; both locations name live resources.
    unsafe { list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
    drop(ManuallyDrop::into_inner(dst.pResource));
    drop(ManuallyDrop::into_inner(src.pResource));
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
    }

    /// Every flat draw is counted under one reason; the table's entries
    /// go in slot order, the rest are `full`; textured draws are not split.
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
            _ => Ok(((), s == 11)),
        };
        let mut placed = Vec::new();
        let asked = [10u64, 0, 12, 10, 13, 14, 15, 19, 18, 11, 16, 17, 16];
        let demand = t.fill(&mut asked.iter().copied(), &mut |_, s| why(s), &mut |(), i| placed.push(i));
        assert_eq!((demand, placed, t.next), (10, vec![1, 2], 3), "11 comes after the table's last entry");
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
        // A table with room: nothing is full, the second frame starts over.
        t.opts.slots = 64;
        t.frame.clear();
        t.why.clear();
        t.next = 1;
        let mut placed = Vec::new();
        t.fill(&mut asked.iter().copied(), &mut |_, s| why(s), &mut |(), i| placed.push(i));
        assert_eq!(placed, vec![1, 2, 3, 4, 5]);
        assert_eq!(t.flat_reason(19), Flat::GpuOut);
        assert_eq!((t.lookup(11), t.flat_reason(14)), ((3, true), Flat::Waiting));
        assert!(!t.why.values().any(|w| *w == Flat::Full));
    }

    /// A table smaller than `TEX_SLOTS` (terrain's) never nulls past its own entries.
    #[test]
    fn table_writes_stay_within_its_entries() {
        let t = TexTable::new(TexOptions { slots: 40, ..Default::default() }, 0);
        assert_eq!(t.written, 40);
        assert_eq!(TexTable::new(TexOptions { slots: u32::MAX, ..Default::default() }, 0).opts.slots, TEX_SLOTS);
    }

    /// The shader's table, its root range and its clamp are `TEX_SLOTS`.
    #[test]
    fn albedo_table_matches_tex_slots() {
        let src = super::super::RAW_SHADER;
        assert!(src.contains(&format!("Texture2D albedoTex[{TEX_SLOTS}] : register(t3);")));
        assert!(src.contains(&format!("SRV(t3, numDescriptors={TEX_SLOTS})")));
        assert!(src.contains(&format!("albedoTex[min(texIndex, {})]", TEX_SLOTS - 1)));
        assert_eq!((layer_entry(0), layer_entry(255), layer_entry(256), layer_entry(TEX_SLOTS - 1)), (0, 255, 0, 0));
    }

    #[test]
    fn checkers_differ_per_slot() {
        let (a, b) = (checker_texels(0x1000), checker_texels(0x2000));
        assert_eq!(a.len(), 256);
        assert_ne!(a, b);
        assert_ne!(a[0..4], a[8..12], "two colours");
    }
}