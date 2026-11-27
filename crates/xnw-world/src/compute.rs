//! One compute runner for captured title dispatches: any translated
//! compute program over buffers and textures, results kept on the device by
//! title address so the next dispatch or draw of a chain reads what an
//! earlier one wrote, as the title's GPU would (the sky and ambient chain,
//! the hair cut map, the head graph).
//!
//! - [`uses`] (pure, from the console container): how the program reads
//!   each table slot it names: as a buffer, or as an image of a dimension,
//!   stored or not. The capture says what the slot holds; the two must agree.
//! - [`Job`]: a dispatch as the runner takes it: root arguments by root
//!   parameter (constants, descriptor tables of [`Res`], root views), the
//!   heap-picked views, samplers and the x, y, z group counts.
//! - [`check`] (pure): the card-safety rules (hang1). A texture the program
//!   reads as a buffer, a buffer it reads as an image, an image of another
//!   dimension, a heap index past [`HEAP_MAX`], a texture UAV in a format
//!   without typed stores: the dispatch is dropped with a counted reason,
//!   never run.
//! - [`Runner`]: the device side. The layout is [`bridge::bind_at`]'s (the
//!   shared bridge binding: tables after the heap-picked views, raw twins,
//!   stride table, bounds, guards). A UAV with a title address is kept on
//!   the device under that address ([`Runner::held`], [`Runner::read`],
//!   [`Runner::provided`]); a later view of that address reads it.
//!   Software adapter only.

use std::collections::{BTreeMap, HashMap};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use xnw_shader::emit::{self, HostShader};
use xnw_shader::resource::{Role, Source};
use xnw_shader::{HwStage, NativeProgram, RootType};

use crate::bridge::{self, Draw, Root, SamplerDesc, Slot};
use crate::preskin::{self, Kind, View};
use crate::textures::Provided;

/// How a program reads one table slot (root parameter `root`, descriptor
/// `slot` of its table): `image` None is a buffer (or constant buffer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Use {
    pub root: u8,
    pub slot: u32,
    pub image: Option<ImageUse>,
    /// Written (a store or an atomic).
    pub store: bool,
    /// Declared an unordered view: written, or the program's resource
    /// range there is a UAV (emit's `used_kind`). The runner binds the slot
    /// by this, not by the capture's view (head1 #939: the title binds
    /// unordered views its programs only read; bound as UAVs, WARP ran
    /// them run-to-run differently).
    pub uav: bool,
}

/// An image read: the GFX10 MIMG dimension (0 1D, 1 2D, 2 3D, 3 cube, 4 1D
/// array, 5 2D array, 6/7 MSAA) and whether an atomic touches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageUse {
    pub dim: u8,
    pub atomic: bool,
}

/// The table slots `container`'s program (its first vertex, pixel, compute
/// or mesh program) reads, one entry per (root, slot, buffer or image).
/// Descriptors picked by a computed index are not listed.
pub fn uses(container: &[u8]) -> Result<Vec<Use>, String> {
    let c = xnw_shader::Container::parse(container).map_err(|e| e.to_string())?;
    let part = c.part(xnw_shader::FourCC::DXIL).ok_or("no DXIL part")?;
    let programs = xnw_shader::native_programs(part.data).map_err(|e| e.to_string())?;
    let native = programs.iter().find(|p| matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms)).ok_or("no program")?;
    uses_of(native)
}

/// [`uses`] of one native program.
pub fn uses_of(native: &NativeProgram) -> Result<Vec<Use>, String> {
    // As emit::translate: a compute program whose root arguments all sit past
    // the 16 user SGPRs reads them through the spill table.
    let spilled = native.stage == HwStage::Cs && !native.layout.is_empty() && native.layout.iter().all(|r| native.stage.user_sgpr(r.user_data) >= 16);
    let p = xnw_shader::ir::build_with(&native.patched_code(), native.wave64, native.vgprs, spilled).map_err(|e| e.to_string())?;
    let res = xnw_shader::resource::track_with(&p, &native.layout, native.stage, true).map_err(|e| e.to_string())?;
    let mut out: Vec<Use> = Vec::new();
    for d in res.descriptors.iter().filter(|d| d.role == Role::Resource) {
        let Some(inst) = p.blocks.get(d.site.0).and_then(|b| b.insts.get(d.site.1)) else { continue };
        let name = inst.name();
        let image = match &inst.fields {
            xnw_shader::gfx10::Fields::Image { dim, .. } => Some(ImageUse { dim: *dim, atomic: name.starts_with("image_atomic") }),
            _ => None,
        };
        let store = name.contains("_store") || name.contains("_atomic");
        for leaf in d.source.leaves() {
            let Source::Loaded { load, first } = leaf else { continue };
            let Some((_, a)) = res.scalar.iter().find(|(s, _)| s == load) else { continue };
            if a.dynamic.is_some() || a.rel.is_some() {
                continue;
            }
            let Some(rp) = native.layout.get(a.param) else { continue };
            if rp.root != RootType::Table {
                continue;
            }
            // As emit's `used_kind`: the dword from the table's start, the
            // program's range at or before it.
            let dword = a.offset / 4 + u32::from(*first);
            let declared = native.used.iter().filter(|u| u.root == RootType::Table && u.user_data == rp.user_data && u.table_offset_dwords <= dword).max_by_key(|u| u.table_offset_dwords);
            let uav = store || declared.is_some_and(|u| u.kind == xnw_shader::ResourceType::Uav);
            let slot = dword / 8;
            match out.iter_mut().find(|u| u.root == rp.slot && u.slot == slot && u.image.is_some() == image.is_some() && u.image.map(|i| i.dim) == image.map(|i| i.dim)) {
                Some(u) => {
                    u.store |= store;
                    u.uav |= uav;
                    if let (Some(i), Some(n)) = (&mut u.image, image) {
                        i.atomic |= n.atomic;
                    }
                }
                None => out.push(Use { root: rp.slot, slot, image, store, uav }),
            }
        }
    }
    out.sort_by_key(|u| (u.root, u.slot, u.image.map(|i| i.dim)));
    Ok(out)
}

/// A texture's shape (mip 0 only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Tex1D,
    Tex2D,
    Tex2DArray,
    Tex3D,
}

impl Shape {
    /// The GFX10 MIMG dimension a program reads this shape with.
    pub fn dim(self) -> u8 {
        match self {
            Shape::Tex1D => 0,
            Shape::Tex2D => 1,
            Shape::Tex2DArray => 5,
            Shape::Tex3D => 2,
        }
    }
}

/// A texture view: `key` the title address of its texels (0: none, not
/// kept), the resource's DXGI format, size (`depth`: array size or depth),
/// whether it is an unordered view, and its texels (mip 0, slices in order,
/// rows as `GetCopyableFootprints` sizes them, unpadded; empty: zeros).
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    pub key: u64,
    pub shape: Shape,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub uav: bool,
    pub texels: &'a [u8],
}

/// A buffer view: `key` the title address of its first byte (0: not kept),
/// its kind (raw, structured, typed or constant; never `RawUav`), whether
/// it is an unordered view, its length and the bytes it starts with.
#[derive(Clone, Copy, Debug)]
pub struct Buffer<'a> {
    pub key: u64,
    pub kind: Kind,
    pub uav: bool,
    pub len: u64,
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub enum Res<'a> {
    Buffer(Buffer<'a>),
    Image(Image<'a>),
}

impl Res<'_> {
    fn key(&self) -> u64 {
        match self {
            Res::Buffer(b) => b.key,
            Res::Image(i) => i.key,
        }
    }
    fn uav(&self) -> bool {
        match self {
            Res::Buffer(b) => b.uav,
            Res::Image(i) => i.uav,
        }
    }
}

/// One root argument by the title's root parameter.
#[derive(Clone, Debug, Default)]
pub enum Arg<'a> {
    #[default]
    None,
    Constants(Vec<u32>),
    /// A descriptor table: views at their offsets (slots of 32 B).
    Table(Vec<(u32, Res<'a>)>),
    /// A root view: the bytes from its address.
    View(&'a [u8]),
}

/// A captured dispatch as the runner takes it.
#[derive(Clone, Debug, Default)]
pub struct Job<'a> {
    pub roots: Vec<Arg<'a>>,
    /// Views the program picks from the heap by an index held in data.
    pub heap: Vec<(u32, Res<'a>)>,
    pub samplers: Vec<(u32, SamplerDesc)>,
    pub sampler_slots: u32,
    /// x, y, z groups (pack kind 60; without it the caller passes x, 1, 1).
    pub groups: [u32; 3],
}

/// Heap-picked view indices from this one on drop the dispatch (the
/// runner's heap holds every index up to the largest).
pub const HEAP_MAX: u32 = 1 << 16;

/// Why a dispatch was not run (counted by [`Runner::dropped`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dropped {
    pub reason: &'static str,
    pub detail: String,
}

pub mod why {
    pub const IMAGE_AS_BUFFER: &str = "a texture the program reads as a buffer";
    pub const BUFFER_AS_IMAGE: &str = "a buffer the program reads as an image";
    pub const DIMENSION: &str = "an image of another dimension than the program reads";
    pub const NO_VIEW: &str = "an image dimension the runner makes no view of (1D array, cube, MSAA)";
    pub const HEAP_INDEX: &str = "a heap index past the bound";
    pub const NO_TYPED_UAV: &str = "a texture UAV in a format without typed stores";
    pub const ATOMIC_FORMAT: &str = "an image atomic on a format other than 32-bit single channel";
    pub const KEPT_KIND: &str = "a kept resource viewed as the other kind";
    pub const KEPT_SHAPE: &str = "a kept texture viewed at another shape";
    /// The title views one memory in formats of one size (pk8a sink 853: an
    /// R32 texture read as R9G9B9E5); D3D12 casts only within a family.
    pub const KEPT_FORMAT: &str = "a kept texture viewed in a format its resource does not cast to";
    pub const READ_WRITE: &str = "one address read and written in one dispatch";
    pub const GROUPS: &str = "no groups, or a count past 65535";
    pub const BINDING: &str = "the binding: the capture lacks a root the program reads";
    pub const PROGRAM: &str = "a program with root UAVs, an address map or descriptor words";
    /// An upstream idea: a stand-in whose pipeline failed reached a
    /// dispatch with the previous root signature bound (layer error 953, a
    /// hang). Here a failed pipeline drops its dispatch before anything is
    /// recorded, and every dispatch records its own root signature and
    /// pipeline on a fresh list.
    pub const PIPELINE: &str = "the program's pipeline failed";
    pub const TRANSLATE: &str = "the program does not translate";
    /// A view whose format the device cannot make removes the device
    /// (WARP 1.0.22: an SRV of R32G8X24_TYPELESS, pk7d sink 2689).
    pub const VIEW_FORMAT: &str = "a view format the device cannot read or write";
}

fn dropped(reason: &'static str, detail: String) -> Dropped {
    Dropped { reason, detail }
}

/// The resources of `job` by (root, slot) for the table views and by heap
/// index for the heap-picked ones (root `u8::MAX`).
fn views<'j, 'a>(job: &'j Job<'a>) -> impl Iterator<Item = (u8, u32, &'j Res<'a>)> {
    let tables = job.roots.iter().enumerate().flat_map(|(r, a)| match a {
        Arg::Table(v) => v.iter().map(move |(k, x)| (r as u8, *k, x)).collect::<Vec<_>>(),
        _ => Vec::new(),
    });
    tables.chain(job.heap.iter().map(|(k, x)| (u8::MAX, *k, x)))
}

/// `job` with each table view the program names bound as the program
/// declares it (SRV or UAV, [`Use::uav`]), whatever the capture's view was.
pub fn by_program<'a>(job: &Job<'a>, uses: &[Use]) -> Job<'a> {
    let mut out = job.clone();
    for (r, a) in out.roots.iter_mut().enumerate() {
        let Arg::Table(t) = a else { continue };
        for (k, x) in t.iter_mut() {
            let Some(u) = uses.iter().find(|u| usize::from(u.root) == r && u.slot == *k) else { continue };
            match x {
                Res::Buffer(b) => b.uav = u.uav,
                Res::Image(i) => i.uav = u.uav,
            }
        }
    }
    out
}

/// The unsigned integer format with the layout of a 16-bit float or signed
/// integer buffer format (one, two or four channels), if `format` is one.
/// From a victory-fork open pull request (`descriptor::host::half_format`, head
/// 5a55384, GPL-3.0-or-later).
pub fn half_format(format: i32) -> Option<i32> {
    match format {
        // R16G16B16A16_FLOAT and _SINT: R16G16B16A16_UINT.
        10 | 14 => Some(12),
        // R16G16_FLOAT and _SINT: R16G16_UINT.
        34 | 38 => Some(36),
        // R16_FLOAT and _SINT: R16_UINT.
        54 | 59 => Some(57),
        _ => None,
    }
}

/// `job` as a program that moves 16-bit channels as unsigned integers
/// through the table views at `half` ([`HostShader::half_views`]: root
/// parameter, dword in the table) is shown it: a typed buffer view there
/// in a 16-bit float or signed format in the unsigned format of the same
/// layout, through which every 16-bit pattern goes as it is (victory-fork
/// 5a55384, the host's `half_view`). Other views as they are.
pub fn with_half_views<'a>(job: &Job<'a>, half: &[(u8, u32)]) -> Job<'a> {
    let mut out = job.clone();
    for &(root, dword) in half {
        let Some(Arg::Table(t)) = out.roots.get_mut(usize::from(root)) else { continue };
        for (k, x) in t.iter_mut() {
            if *k != dword / 8 {
                continue;
            }
            if let Res::Buffer(Buffer { kind: Kind::Typed { format, .. }, .. }) = x
                && let Some(f) = half_format(*format)
            {
                *format = f;
            }
        }
    }
    out
}

/// The card-safety rules (pure): `typed_uav(format)` says whether the
/// device stores typed to a texture view of that format.
pub fn check(job: &Job, uses: &[Use], typed_uav: &dyn Fn(u32) -> bool) -> Result<(), Dropped> {
    if job.groups.contains(&0) || job.groups.iter().any(|&g| g > 65_535) {
        return Err(dropped(why::GROUPS, format!("{:?}", job.groups)));
    }
    if let Some((k, _)) = job.heap.iter().find(|(k, _)| *k >= HEAP_MAX) {
        return Err(dropped(why::HEAP_INDEX, format!("heap index {k} (bound {HEAP_MAX})")));
    }
    for u in uses {
        let Some(Arg::Table(t)) = job.roots.get(usize::from(u.root)) else { continue };
        let Some((_, r)) = t.iter().find(|(k, _)| *k == u.slot) else { continue };
        let at = format!("root {} slot {}", u.root, u.slot);
        match (r, u.image) {
            (Res::Image(_), None) => return Err(dropped(why::IMAGE_AS_BUFFER, at)),
            (Res::Buffer(_), Some(_)) => return Err(dropped(why::BUFFER_AS_IMAGE, at)),
            (Res::Image(i), Some(m)) => {
                if !matches!(m.dim, 0 | 1 | 2 | 5) {
                    return Err(dropped(why::NO_VIEW, format!("{at}: dim {}", m.dim)));
                }
                if i.shape.dim() != m.dim {
                    return Err(dropped(why::DIMENSION, format!("{at}: {:?}, read as dim {}", i.shape, m.dim)));
                }
                if m.atomic && !matches!(i.format, 39..=43) {
                    return Err(dropped(why::ATOMIC_FORMAT, format!("{at}: format {}", i.format)));
                }
            }
            (Res::Buffer(_), None) => {}
        }
    }
    for (root, k, r) in views(job) {
        if let Res::Image(i) = r
            && i.uav
            && !typed_uav(typed(i.format))
        {
            return Err(dropped(why::NO_TYPED_UAV, format!("root {root} slot {k}: format {}", i.format)));
        }
    }
    let mut seen: HashMap<u64, bool> = HashMap::new();
    for (root, k, r) in views(job) {
        if r.key() == 0 {
            continue;
        }
        if let Some(&uav) = seen.get(&r.key())
            && uav != r.uav()
        {
            return Err(dropped(why::READ_WRITE, format!("root {root} slot {k}: {:#x}", r.key())));
        }
        seen.insert(r.key(), r.uav());
    }
    Ok(())
}

/// A typed format for views of a typeless one (the usual reading: float
/// where the family has one, else unorm) [I]; others as they are.
pub fn typed(format: u32) -> u32 {
    match format {
        1 => 2,
        5 => 6,
        9 => 10,
        15 => 16,
        23 => 24,
        27 => 28,
        33 => 34,
        39 => 41,
        48 => 49,
        53 => 54,
        60 => 61,
        90 => 87,
        // Depth-stencil families: the depth as the shader reads it.
        19 => 21,
        44 => 46,
        f => f,
    }
}

/// [`Runner`]'s formats never unordered on the hardware adapter (VIctory).
pub const CARD_NO_UAV: [u32; 3] = [67, 86, 115];

/// The format a kept texture's resource is made in: a 32-bit
/// single-channel one typeless, so its views read it as float or as uint
/// (an image atomic's R32_UINT view).
pub fn res_format(format: u32) -> u32 {
    if matches!(format, 40..=43) { 39 } else { format }
}

/// Whether a view of `view` may be made of a resource of `res` (equal, or
/// a typed member of a typeless family).
pub fn castable(res: u32, view: u32) -> bool {
    let family: &[u32] = match res {
        1 => &[2, 3, 4],
        5 => &[6, 7, 8],
        9 => &[10, 11, 12, 13, 14],
        15 => &[16, 17, 18],
        19 => &[20, 21, 22],
        23 => &[24, 25],
        27 => &[28, 29, 30, 31, 32],
        33 => &[34, 35, 36, 37, 38],
        39 => &[40, 41, 42, 43],
        44 => &[45, 46, 47],
        48 => &[49, 50, 51, 52],
        53 => &[54, 55, 56, 57, 58, 59],
        60 => &[61, 62, 63, 64],
        90 => &[87, 91],
        _ => &[],
    };
    res == view || family.contains(&view)
}

/// What the runner holds at a title address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Held {
    Buffer { len: u64 },
    Image { shape: Shape, format: u32, width: u32, height: u32, depth: u32 },
}

struct Kept {
    res: ID3D12Resource,
    /// The resource's own format (textures; 0 for buffers).
    format: u32,
    state: D3D12_RESOURCE_STATES,
    held: Held,
}

/// One dispatch run: GPU seconds, and the addresses it wrote.
#[derive(Clone, Debug, Default)]
pub struct Ran {
    pub seconds: f64,
    pub wrote: Vec<u64>,
    /// Addresses it read that an earlier dispatch had kept.
    pub read_kept: Vec<u64>,
}

#[derive(Clone, Debug)]
pub enum Outcome {
    Ran(Ran),
    Dropped(Dropped),
}

/// The device side: pipelines by program, resources kept by title address.
pub struct Runner<'h> {
    host: &'h xnw_gpu::host::Host,
    kept: HashMap<u64, Kept>,
    pipes: HashMap<u64, (ID3D12RootSignature, ID3D12PipelineState)>,
    /// Dropped dispatches by reason.
    pub dropped: BTreeMap<&'static str, u32>,
    pub ran: u32,
    pub seconds: f64,
    /// Seconds one submission may take before the runner gives up.
    pub limit: f64,
    /// HAIR7 (the live strand chain): an input buffer with a title address
    /// and no bytes is bound in place in title memory (`XNW_ZEROCOPY`,
    /// `xnw_gpu::shared::world_range`) where its page window is open; else
    /// zeros, as before. Views bound in place since the last [`Runner::forget`].
    pub in_place: bool,
    pub placed: u32,
    /// FACE1: dispatches recorded into one command list ([`Runner::batch`]),
    /// submitted once (`flush`) before anything reads or holds.
    batch: Option<Batch>,
    /// Submissions (each a fence wait) since the last [`Runner::forget`].
    submissions: std::cell::Cell<u32>,
    /// FACE1: a batch's descriptor heaps and upload buffers, reused once its
    /// list has run (~190 dispatches a head, each making two heaps
    /// and its uploads, cost 400-1000 ms a frame of driver calls on the card).
    pool: Pool,
}

/// Descriptor heaps (type, capacity) and upload buffers (capacity) free for
/// a batch, and those a batch took (back to free when its list has run).
#[derive(Default)]
struct Pool {
    heaps: std::cell::RefCell<Vec<(i32, u32, ID3D12DescriptorHeap)>>,
    uploads: std::cell::RefCell<Vec<(u64, ID3D12Resource)>>,
    taken_heaps: std::cell::RefCell<Vec<(i32, u32, ID3D12DescriptorHeap)>>,
    taken_uploads: std::cell::RefCell<Vec<(u64, ID3D12Resource)>>,
    /// Objects made (not taken from the free lists) since the last [`Runner::forget`].
    made: std::cell::Cell<u32>,
}

/// Free objects kept at most, per kind.
const POOL_MAX: usize = 4096;

impl Pool {
    fn heap(&self, dev: &ID3D12Device, ty: D3D12_DESCRIPTOR_HEAP_TYPE, n: u32) -> Result<ID3D12DescriptorHeap, String> {
        let n = n.max(1);
        let mut free = self.heaps.borrow_mut();
        let got = match free.iter().position(|h| h.0 == ty.0 && h.1 >= n) {
            Some(i) => free.swap_remove(i),
            None => {
                let cap = n.next_power_of_two().max(64);
                // SAFETY: a plain creation call.
                let h: ID3D12DescriptorHeap = unsafe { dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: cap, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }) }.map_err(|e| e.message().to_string())?;
                self.made.set(self.made.get() + 1);
                (ty.0, cap, h)
            }
        };
        let h = got.2.clone();
        self.taken_heaps.borrow_mut().push(got);
        Ok(h)
    }

    /// As [`upload`]: `bytes` at offset 0, zeros to `size` (at least 256).
    fn upload(&self, dev: &ID3D12Device, bytes: &[u8], size: u64) -> Result<ID3D12Resource, String> {
        let need = size.max(256).max(bytes.len() as u64);
        let mut free = self.uploads.borrow_mut();
        let got = match free.iter().position(|u| u.0 >= need) {
            Some(i) => free.swap_remove(i),
            None => {
                let cap = need.next_power_of_two();
                self.made.set(self.made.get() + 1);
                (cap, bridge::buffer(dev, cap, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?)
            }
        };
        let mut at = std::ptr::null_mut();
        // SAFETY: an upload buffer of `got.0` >= `need` bytes, mapped here; no list in flight uses it (taken lists come back after their fence).
        unsafe {
            got.1.Map(0, None, Some(&mut at)).map_err(|e| e.message().to_string())?;
            std::ptr::write_bytes(at.cast::<u8>(), 0, need as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), at.cast::<u8>(), bytes.len().min(need as usize));
            got.1.Unmap(0, None);
        }
        let r = got.1.clone();
        self.taken_uploads.borrow_mut().push(got);
        Ok(r)
    }

    /// The taken objects back to the free lists (their list has run).
    fn recycle(&self) {
        let mut h = self.heaps.borrow_mut();
        h.append(&mut self.taken_heaps.borrow_mut());
        h.truncate(POOL_MAX);
        let mut u = self.uploads.borrow_mut();
        u.append(&mut self.taken_uploads.borrow_mut());
        u.truncate(POOL_MAX);
    }
}

/// A command list the dispatches of a batch are recorded into, and what
/// they use until it is submitted.
struct Batch {
    _alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    keep: Vec<ID3D12Resource>,
    heaps: Vec<ID3D12DescriptorHeap>,
    recorded: u32,
}

fn content(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::hash::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}

/// A UAV barrier over every resource: the dispatches before it finish their
/// unordered writes before the ones after it start.
fn uav_all() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE, Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }) } }
}

fn barrier(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            // SAFETY: a borrowed pointer the barrier never releases.
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: unsafe { std::mem::transmute_copy(r) }, Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: before, StateAfter: after }),
        },
    }
}

fn texture_desc(i: &Image, uav: bool) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: match i.shape {
            Shape::Tex1D => D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            Shape::Tex3D => D3D12_RESOURCE_DIMENSION_TEXTURE3D,
            _ => D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        },
        Width: u64::from(i.width.max(1)),
        Height: if i.shape == Shape::Tex1D { 1 } else { i.height.max(1) },
        DepthOrArraySize: i.depth.clamp(1, u32::from(u16::MAX)) as u16,
        MipLevels: 1,
        // A 32-bit single-channel texture is made typeless: its views read
        // it as float or as uint (an image atomic's R32_UINT view).
        Format: DXGI_FORMAT(res_format(i.format) as i32),
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    }
}

fn held_of(r: &Res) -> Held {
    match r {
        Res::Buffer(b) => Held::Buffer { len: b.len.max(b.bytes.len() as u64) },
        Res::Image(i) => Held::Image { shape: i.shape, format: i.format, width: i.width, height: i.height, depth: i.depth },
    }
}

/// Subresource footprints of a texture: (footprint, rows, unpadded row bytes) each, and the total.
fn footprints(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> (Vec<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u32, u64)>, u64) {
    let n = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { u32::from(desc.DepthOrArraySize) };
    let mut fp = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    let mut rows = vec![0u32; n as usize];
    let mut sizes = vec![0u64; n as usize];
    let mut total = 0u64;
    // SAFETY: arrays of `n` entries each.
    unsafe { dev.GetCopyableFootprints(desc, 0, n, 0, Some(fp.as_mut_ptr()), Some(rows.as_mut_ptr()), Some(sizes.as_mut_ptr()), Some(&mut total)) };
    (fp.into_iter().zip(rows).zip(sizes).map(|((f, r), s)| (f, r, s)).collect(), total)
}

impl<'h> Runner<'h> {
    /// A runner on `host`: the title's translated programs run on the
    /// software adapter; on the hardware adapter only as a scheduled card
    /// sample under the GPU watchdog ([`bridge::title_programs_here`]),
    /// each submission then given 2 s before the runner stops waiting (the
    /// watchdog exits at its budget first).
    pub fn new(host: &'h xnw_gpu::host::Host) -> Result<Runner<'h>, String> {
        bridge::title_programs_here(host)?;
        let limit = if host.caps.software { 600.0 } else { 2.0 };
        Ok(Runner { host, kept: HashMap::new(), pipes: HashMap::new(), dropped: BTreeMap::new(), ran: 0, seconds: 0.0, limit, in_place: false, placed: 0, batch: None, submissions: std::cell::Cell::new(0), pool: Pool::default() })
    }

    /// FACE1: `on`: the dispatches from here on are recorded into one
    /// command list, ordered by UAV barriers, and submitted once (one fence
    /// wait) when something reads, holds or hands out a resource, or at
    /// `batch(false)`; their GPU seconds are the submission's wall time
    /// ([`Runner::seconds`]), not per dispatch. `off`: submits what is recorded.
    pub fn batch(&mut self, on: bool) -> Result<(), String> {
        if on {
            if self.batch.is_none() {
                let (alloc, list) = self.list()?;
                self.batch = Some(Batch { _alloc: alloc, list, keep: Vec::new(), heaps: Vec::new(), recorded: 0 });
            }
            return Ok(());
        }
        self.flush()?;
        self.batch = None;
        Ok(())
    }

    /// Submits the batch's recorded dispatches (if any) and starts a new list.
    fn flush(&mut self) -> Result<(), String> {
        let Some(b) = self.batch.as_ref().filter(|b| b.recorded > 0) else { return Ok(()) };
        let t = std::time::Instant::now();
        let list = b.list.clone();
        let sent = self.submit(&list);
        self.seconds += t.elapsed().as_secs_f64();
        // Its heaps and uploads are free again once it has run (not after a failure: the list may still hold them).
        if sent.is_ok() {
            self.pool.recycle();
        }
        let (alloc, list) = self.list()?;
        self.batch = Some(Batch { _alloc: alloc, list, keep: Vec::new(), heaps: Vec::new(), recorded: 0 });
        sent
    }

    /// Submissions (each with its fence wait) since the last [`Runner::forget`].
    pub fn submissions(&self) -> u32 {
        self.submissions.get()
    }

    /// Descriptor heaps and upload buffers a batch made since the last
    /// [`Runner::forget`] (the rest came from its pool).
    pub fn made(&self) -> u32 {
        self.pool.made.get()
    }

    /// [`Runner::forget`], keeping the resources `keep` names (FACE1: the
    /// head graph's large buffers, the same title object frame to frame,
    /// stay on the device as the console's do).
    pub fn forget_except(&mut self, keep: &dyn Fn(u64, Held) -> bool) {
        self.kept.retain(|k, v| keep(*k, v.held));
        self.dropped.clear();
        (self.ran, self.seconds, self.placed) = (0, 0.0, 0);
        self.submissions.set(0);
        self.pool.made.set(0);
    }

    /// Drops every kept resource and the counts; the pipelines stay (HAIR6:
    /// one runner for the live path's chains, a frame each).
    pub fn forget(&mut self) {
        self.kept.clear();
        self.dropped.clear();
        (self.ran, self.seconds, self.placed) = (0, 0.0, 0);
        self.submissions.set(0);
    }

    /// HAIR7: the resource kept at title address `key`, left in the common
    /// state (a submission of its barrier when it was in another): the strand
    /// draws read the chain's pool there, on the device, without a copy.
    pub fn resource(&mut self, key: u64) -> Result<Option<bridge::DeviceBuffer>, String> {
        self.flush()?;
        let Some(k) = self.kept.get(&key) else { return Ok(None) };
        let (res, state) = (k.res.clone(), k.state);
        if state != D3D12_RESOURCE_STATE_COMMON {
            let (_alloc, list) = self.list()?;
            // SAFETY: a barrier on a live resource the runner keeps.
            unsafe { list.ResourceBarrier(&[barrier(&res, state, D3D12_RESOURCE_STATE_COMMON)]) };
            self.submit(&list)?;
            if let Some(k) = self.kept.get_mut(&key) {
                k.state = D3D12_RESOURCE_STATE_COMMON;
            }
        }
        Ok(Some(bridge::DeviceBuffer(res)))
    }

    /// What the runner holds at title address `key`.
    pub fn held(&self, key: u64) -> Option<Held> {
        self.kept.get(&key).map(|k| k.held)
    }

    /// The addresses held, in order.
    pub fn keys(&self) -> Vec<u64> {
        let mut k: Vec<u64> = self.kept.keys().copied().collect();
        k.sort_unstable();
        k
    }

    /// Whether the device stores typed to a texture UAV of `format`.
    pub fn typed_uav(&self, format: u32) -> bool {
        if self.no_uav(format) {
            return false;
        }
        let init = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(format as i32), ..Default::default() };
        xnw_gpu::host::feature(&self.host.device, D3D12_FEATURE_FORMAT_SUPPORT, init).is_some_and(|f| f.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0 && f.Support2.0 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE.0 != 0)
    }

    /// Whether texture `i` lays out as many bytes per row and rows as `res`
    /// (a read in another format of the same size is a copy of its bits).
    fn same_bytes(&self, res: &ID3D12Resource, i: &Image) -> bool {
        // SAFETY: a plain query.
        let a = footprints(&self.host.device, &unsafe { res.GetDesc() }).0;
        let b = footprints(&self.host.device, &texture_desc(i, false)).0;
        a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.1 == y.1 && x.2 == y.2 && x.0.Footprint.Depth == y.0.Footprint.Depth)
    }

    /// Whether the device makes a view of `format` (a buffer's or a
    /// texture's; an unordered one with typed stores).
    pub fn views(&self, format: u32, buffer: bool, uav: bool) -> bool {
        let init = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: DXGI_FORMAT(format as i32), ..Default::default() };
        let Some(f) = xnw_gpu::host::feature(&self.host.device, D3D12_FEATURE_FORMAT_SUPPORT, init) else { return false };
        let s1 = f.Support1.0;
        let has = |b: D3D12_FORMAT_SUPPORT1| s1 & b.0 != 0;
        if uav {
            if self.no_uav(format) {
                return false;
            }
            return has(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) && f.Support2.0 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE.0 != 0;
        }
        has(D3D12_FORMAT_SUPPORT1_SHADER_LOAD) && (buffer && has(D3D12_FORMAT_SUPPORT1_BUFFER) || !buffer && (has(D3D12_FORMAT_SUPPORT1_TEXTURE2D) || has(D3D12_FORMAT_SUPPORT1_TEXTURE3D)))
    }

    /// Formats never made unordered on the hardware adapter, whatever it
    /// reports: R9G9B9E5_SHAREDEXP 67, B5G5R5A1_UNORM 86, B4G4R4A4_UNORM 115
    /// as render target or UAV were 14 of 20 card crashes upstream
    /// (VIctory, after #2719/#2725). A view not made is a
    /// dropped dispatch, named in the runner's drop reasons.
    fn no_uav(&self, format: u32) -> bool {
        !self.host.caps.software && CARD_NO_UAV.contains(&format)
    }

    fn submit(&self, list: &ID3D12GraphicsCommandList) -> Result<(), String> {
        self.submissions.set(self.submissions.get() + 1);
        let e = |e: windows::core::Error| e.message().to_string();
        // SAFETY: plain COM calls on live objects.
        unsafe {
            list.Close().map_err(e)?;
            self.host.queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(list).map_err(e)?)]);
            let sent = xnw_gpu::gpuwatch::begin("compute");
            let fence: ID3D12Fence = self.host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
            self.host.queue.Signal(&fence, 1).map_err(e)?;
            let t = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                self.host.device.GetDeviceRemovedReason().map_err(|x| format!("device removed: {}", e(x)))?;
                if t.elapsed().as_secs_f64() > self.limit {
                    xnw_gpu::gpuwatch::abandon("compute", self.limit, xnw_gpu::gpuwatch::GRACE);
                    return Err(format!("the submission took over {} s", self.limit));
                }
                std::thread::yield_now();
            }
            xnw_gpu::gpuwatch::end("compute", sent, None);
        }
        Ok(())
    }

    fn list(&self) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList), String> {
        let e = |e: windows::core::Error| e.message().to_string();
        // SAFETY: plain COM calls.
        unsafe {
            let alloc: ID3D12CommandAllocator = self.host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
            let list: ID3D12GraphicsCommandList = self.host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
            Ok((alloc, list))
        }
    }

    /// Puts a texture or buffer at title address `key` (texels or bytes as
    /// a [`Res`] gives them): an input the chain reads that no dispatch of
    /// it writes (a captured texture, a stand-in).
    pub fn hold(&mut self, r: &Res) -> Result<(), String> {
        self.flush()?;
        let (_alloc, list) = self.list()?;
        let mut keep = Vec::new();
        self.make(&list, r, &mut keep)?;
        self.submit(&list)
    }

    /// Makes the kept resource for `r` (key non-zero) and records its first
    /// bytes' copy on `list`; it ends in the common state.
    fn make(&mut self, list: &ID3D12GraphicsCommandList, r: &Res, keep: &mut Vec<ID3D12Resource>) -> Result<(), String> {
        let dev = &self.host.device;
        let common = D3D12_RESOURCE_STATE_COMMON;
        // SAFETY: D3D12 calls with descriptions that outlive them; the
        // upload buffers live in `keep` until the caller's submission ends.
        unsafe {
            let res = match r {
                Res::Buffer(b) => {
                    let size = b.len.max(b.bytes.len() as u64).next_multiple_of(256).max(256);
                    let res = bridge::buffer(dev, size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)?;
                    if !b.bytes.is_empty() {
                        let up = upload(dev, b.bytes, size)?;
                        list.CopyBufferRegion(&res, 0, &up, 0, b.bytes.len() as u64);
                        keep.push(up);
                    }
                    list.ResourceBarrier(&[barrier(&res, D3D12_RESOURCE_STATE_COPY_DEST, common)]);
                    res
                }
                Res::Image(i) => {
                    let desc = texture_desc(i, self.typed_uav(typed(i.format)));
                    let mut got: Option<ID3D12Resource> = None;
                    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut got).map_err(|e| format!("texture {}x{}x{} format {}: {}", i.width, i.height, i.depth, i.format, e.message()))?;
                    let res = got.ok_or("no texture")?;
                    if !i.texels.is_empty() {
                        let (fps, total) = footprints(dev, &desc);
                        let up = bridge::buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
                        let mut to = std::ptr::null_mut();
                        up.Map(0, None, Some(&mut to)).map_err(|e| e.message().to_string())?;
                        let mut from = 0usize;
                        for (fp, rows, row) in &fps {
                            let slices = fp.Footprint.Depth as usize;
                            for z in 0..slices {
                                for y in 0..*rows as usize {
                                    let n = (*row as usize).min(i.texels.len().saturating_sub(from));
                                    let at = fp.Offset as usize + (z * *rows as usize + y) * fp.Footprint.RowPitch as usize;
                                    std::ptr::copy_nonoverlapping(i.texels.as_ptr().add(from), to.cast::<u8>().add(at), n);
                                    from += *row as usize;
                                }
                            }
                        }
                        up.Unmap(0, None);
                        for (k, (fp, _, _)) in fps.iter().enumerate() {
                            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&res), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
                            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
                            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                        }
                        keep.push(up);
                    }
                    list.ResourceBarrier(&[barrier(&res, D3D12_RESOURCE_STATE_COPY_DEST, common)]);
                    res
                }
            };
            let format = if let Res::Image(i) = r { res_format(i.format) } else { 0 };
            self.kept.insert(r.key(), Kept { res, format, state: common, held: held_of(r) });
        }
        Ok(())
    }

    /// Runs `shader` (translated from a container [`uses`] read, `wave`
    /// lanes) over `job`. A dispatch [`check`] refuses, or whose views
    /// disagree with what the runner keeps, is dropped and counted; `Err`
    /// is a device or pipeline failure.
    pub fn dispatch(&mut self, shader: &HostShader, wave: u32, uses: &[Use], job: &Job) -> Result<Outcome, String> {
        let job = &by_program(job, uses);
        let twins;
        let job = if shader.half_views.is_empty() {
            job
        } else {
            twins = with_half_views(job, &shader.half_views);
            &twins
        };
        match self.prepare(shader, wave, uses, job) {
            Err(d) => {
                *self.dropped.entry(d.reason).or_default() += 1;
                Ok(Outcome::Dropped(d))
            }
            Ok(b) => {
                let pipe = match self.pipeline(shader) {
                    Ok(p) => p,
                    Err(x) => return Ok(Outcome::Dropped(self.refuse(why::PIPELINE, x))),
                };
                let before: Vec<u64> = self.kept.keys().copied().collect();
                let ran = self.run(shader, pipe, uses, job, &b);
                if ran.is_err() {
                    // Resources made for a list that never ran hold nothing: forget them
                    // (in a batch, alive until its list is submitted: its copies name them).
                    let gone: Vec<u64> = self.kept.keys().filter(|k| !before.contains(k)).copied().collect();
                    for k in gone {
                        if let (Some(v), Some(b)) = (self.kept.remove(&k), self.batch.as_mut()) {
                            b.keep.push(v.res);
                        }
                    }
                }
                // A failure on a removed device carries the validation layer's errors.
                // SAFETY: a plain query.
                let ran = ran.map_err(|e| if unsafe { self.host.device.GetDeviceRemovedReason() }.is_err() { format!("device removed: {e}{}", bridge::layer_errors(&self.host.device)) } else { e })?;
                self.ran += 1;
                self.seconds += ran.seconds;
                Ok(Outcome::Ran(ran))
            }
        }
    }

    /// Counts a dispatch the caller does not run (a program that does not
    /// translate, say) under `reason`.
    pub fn refuse(&mut self, reason: &'static str, detail: String) -> Dropped {
        *self.dropped.entry(reason).or_default() += 1;
        dropped(reason, detail)
    }

    fn prepare<'a>(&self, shader: &HostShader, wave: u32, uses: &[Use], job: &Job<'a>) -> Result<bridge::Binding<'a>, Dropped> {
        if let Some(gap) = preskin::adapter_gap(&self.host.caps, wave) {
            return Err(dropped("the adapter cannot run the program", gap));
        }
        check(job, uses, &|f| self.typed_uav(f))?;
        for (root, k, r) in views(job) {
            let Some(kept) = self.kept.get(&r.key()).filter(|_| r.key() != 0) else { continue };
            let at = format!("root {root} slot {k}: {:#x}", r.key());
            match (r, kept.held) {
                (Res::Buffer(_), Held::Image { .. }) | (Res::Image(_), Held::Buffer { .. }) => return Err(dropped(why::KEPT_KIND, at)),
                (Res::Image(i), Held::Image { shape, width, height, depth, .. }) if (shape, width, height, depth) != (i.shape, i.width, i.height, i.depth) => return Err(dropped(why::KEPT_SHAPE, format!("{at}: kept {shape:?} {width}x{height}x{depth}"))),
                (Res::Image(i), _) if !castable(kept.format, typed(i.format)) && !castable(kept.format, i.format) && (i.uav || !self.same_bytes(&kept.res, i)) => return Err(dropped(why::KEPT_FORMAT, format!("{at}: resource format {}, view format {}", kept.format, i.format))),
                _ => {}
            }
        }
        if !shader.root_uavs.is_empty() || shader.address_map || shader.descriptor_words {
            return Err(dropped(why::PROGRAM, String::new()));
        }
        for (root, k, r) in views(job) {
            let (format, buffer, uav) = match r {
                Res::Image(i) => (typed(i.format), false, i.uav),
                Res::Buffer(Buffer { kind: Kind::Typed { format, .. }, uav, .. }) => (*format as u32, true, *uav),
                Res::Buffer(_) => continue,
            };
            if !self.views(format, buffer, uav) {
                return Err(dropped(why::VIEW_FORMAT, format!("root {root} slot {k}: format {format}{}", if uav { " (UAV)" } else { "" })));
            }
        }
        // The layout: buffers as their views, textures as handles the runner fills.
        let slot = |r: &Res<'a>| match r {
            Res::Buffer(b) => Slot::Buffer(View { kind: if b.uav && b.kind == Kind::Raw { Kind::RawUav { len: b.len.min(u64::from(u32::MAX)) as u32 } } else { b.kind }, bytes: b.bytes }),
            Res::Image(_) => Slot::Handle(0),
        };
        let roots = job
            .roots
            .iter()
            .map(|a| match a {
                Arg::None => Root::None,
                Arg::Constants(c) => Root::Constants(c.clone()),
                Arg::Table(v) => Root::Table(v.iter().map(|(k, r)| (*k, slot(r))).collect()),
                Arg::View(b) => Root::View(b),
            })
            .collect();
        let draw = Draw { roots, heap: job.heap.iter().map(|(k, r)| (*k, slot(r))).collect(), samplers: job.samplers.clone(), sampler_slots: job.sampler_slots, groups: job.groups[0] };
        let none = HostShader { stage: HwStage::Ps, dxil: Vec::new(), user_data: Vec::new(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false };
        bridge::bind_at(shader, &none, &draw, 0).map_err(|e| dropped(why::BINDING, e))
    }

    fn pipeline(&mut self, shader: &HostShader) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
        let key = content(&shader.dxil);
        if let Some(p) = self.pipes.get(&key) {
            return Ok(p.clone());
        }
        let eh = emit::host::Host::from_device(self.host.device.clone());
        let root = eh.root_signature_for(&[(shader, D3D12_SHADER_VISIBILITY_ALL)])?;
        let pso = eh.compute_pipeline_with(&root, &shader.dxil).map_err(|x| format!("compute pipeline: {x}"))?;
        self.pipes.insert(key, (root.clone(), pso.clone()));
        Ok((root, pso))
    }

    fn run(&mut self, shader: &HostShader, pipe: (ID3D12RootSignature, ID3D12PipelineState), uses: &[Use], job: &Job, b: &bridge::Binding) -> Result<Ran, String> {
        let (root, pso) = pipe;
        let dev = self.host.device.clone();
        let e = |e: windows::core::Error| e.message().to_string();
        // Each view at its heap slot: a table's first slot is its user-data dword / 32.
        // (heap slot, view, an image atomic touches it)
        let mut items: Vec<(u32, Res, bool)> = Vec::new();
        for (r, a) in job.roots.iter().enumerate() {
            if let Arg::Table(v) = a
                && shader.user_data.iter().any(|u| usize::from(u.slot) == r && u.root == RootType::Table)
            {
                let first = b.views.constants.get(r).copied().unwrap_or(0) / 32;
                items.extend(v.iter().map(|(k, x)| (first + k, *x, uses.iter().any(|u| usize::from(u.root) == r && u.slot == *k && u.image.is_some_and(|m| m.atomic)))));
            }
        }
        items.extend(job.heap.iter().map(|(k, x)| (*k, *x, false)));
        // FACE1: in a batch, recorded into its list and submitted later.
        let (_alloc, list) = match &self.batch {
            Some(b) => (None, b.list.clone()),
            None => self.list().map(|(a, l)| (Some(a), l))?,
        };
        let batched = self.batch.is_some();
        let mut keep: Vec<ID3D12Resource> = Vec::new();
        let mut ran = Ran::default();
        // Kept resources first: made on first sight (a UAV, or any view
        // with an address read before anything wrote it stays a plain input).
        for (_, r, _) in &items {
            if r.key() != 0 && r.uav() && !self.kept.contains_key(&r.key()) {
                self.make(&list, r, &mut keep)?;
            }
        }
        // SAFETY: D3D12 calls with descriptions that outlive them; every
        // resource lives in `keep` or `self.kept` until the fence says the GPU is done.
        unsafe {
            // FACE1: a batch takes its heaps and uploads from the pool.
            let pool = &self.pool;
            let heap = |ty, n: u32| -> Result<ID3D12DescriptorHeap, String> {
                if batched {
                    return pool.heap(&dev, ty, n);
                }
                dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: n.max(1), Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).map_err(e)
            };
            let upload = |dev: &ID3D12Device, bytes: &[u8], size: u64| if batched { pool.upload(dev, bytes, size) } else { upload(dev, bytes, size) };
            let views = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, b.views.heap_slots)?;
            let samplers = heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, b.sampler_slots)?;
            let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let at = |slot: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: views.GetCPUDescriptorHandleForHeapStart().ptr + slot as usize * step };
            let srv_buf = |format, n: u32, stride: u32, flags| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_BUFFER, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: n, StructureByteStride: stride, Flags: flags } } };
            let uav_buf = |format, n: u32, stride: u32, flags| D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: format, ViewDimension: D3D12_UAV_DIMENSION_BUFFER, Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: n, StructureByteStride: stride, CounterOffsetInBytes: 0, Flags: flags } } };
            // Typed nulls everywhere (raw buffers), the guards' kinds (as bridge::upload).
            for s in 0..b.views.heap_slots {
                dev.CreateShaderResourceView(None, Some(&srv_buf(DXGI_FORMAT_R32_TYPELESS, 0, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s));
            }
            let g = &b.views.bounds;
            let tex2d = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } } };
            dev.CreateShaderResourceView(None, Some(&tex2d), at(g[1]));
            dev.CreateUnorderedAccessView(None, None, Some(&uav_buf(DXGI_FORMAT_R32_TYPELESS, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW)), at(g[3]));
            dev.CreateUnorderedAccessView(None, None, Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() }), at(g[4]));
            dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at(g[5]));
            // Kept resources' states for this dispatch.
            let mut wanted: HashMap<u64, D3D12_RESOURCE_STATES> = HashMap::new();
            for (s, r, atomic) in &items {
                let kept = (r.key() != 0).then(|| self.kept.get(&r.key())).flatten();
                if kept.is_some() {
                    wanted.insert(r.key(), if r.uav() { D3D12_RESOURCE_STATE_UNORDERED_ACCESS } else { D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE });
                    if !r.uav() {
                        ran.read_kept.push(r.key());
                    }
                }
                let twin = b.views.strides.get(*s as usize).map_or(0, |w| w[1] & emit::StrideEntry::RAW_SLOTS);
                match r {
                    Res::Buffer(x) => {
                        let len = match kept.map(|k| k.held) {
                            Some(Held::Buffer { len }) => len,
                            _ => x.len.max(x.bytes.len() as u64),
                        };
                        let kind = if x.uav && x.kind == Kind::Raw { Kind::RawUav { len: len.min(u64::from(u32::MAX)) as u32 } } else { x.kind };
                        let Some((size, n)) = kind.extent(len) else { continue };
                        // HAIR7: an input with an address and no bytes, in title memory in place.
                        let placed = if self.in_place && kept.is_none() && !x.uav && x.bytes.is_empty() && x.key != 0 && kind != Kind::Cbv {
                            xnw_gpu::shared::world_range(x.key, size).filter(|t| t.offset.is_multiple_of(bridge::unit(kind)))
                        } else {
                            None
                        };
                        if let Some(t) = placed {
                            let first = t.offset / bridge::unit(kind);
                            let srv_first = |format, n: u32, stride: u32, flags, first: u64| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_BUFFER, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: first, NumElements: n, StructureByteStride: stride, Flags: flags } } };
                            let raw_n = (size / 4).min(u64::from(u32::MAX)) as u32;
                            match kind {
                                Kind::Raw => dev.CreateShaderResourceView(&t.buffer, Some(&srv_first(DXGI_FORMAT_R32_TYPELESS, n, 0, D3D12_BUFFER_SRV_FLAG_RAW, t.offset / 4)), at(*s)),
                                Kind::Structured(stride) => dev.CreateShaderResourceView(&t.buffer, Some(&srv_first(DXGI_FORMAT_UNKNOWN, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE, first)), at(*s)),
                                Kind::Typed { format, .. } => dev.CreateShaderResourceView(&t.buffer, Some(&srv_first(DXGI_FORMAT(format), n, 0, D3D12_BUFFER_SRV_FLAG_NONE, first)), at(*s)),
                                Kind::Cbv | Kind::RawUav { .. } => {}
                            }
                            if twin != 0 {
                                dev.CreateShaderResourceView(&t.buffer, Some(&srv_first(DXGI_FORMAT_R32_TYPELESS, raw_n, 0, D3D12_BUFFER_SRV_FLAG_RAW, t.offset / 4)), at(twin - 1));
                            }
                            self.placed += 1;
                            keep.push(t.buffer);
                            continue;
                        }
                        let res = match kept {
                            Some(k) => k.res.clone(),
                            None if x.uav => {
                                let r = bridge::buffer(&dev, size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)?;
                                if !x.bytes.is_empty() {
                                    let up = upload(&dev, x.bytes, size)?;
                                    list.CopyBufferRegion(&r, 0, &up, 0, x.bytes.len() as u64);
                                    keep.push(up);
                                }
                                list.ResourceBarrier(&[barrier(&r, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
                                r
                            }
                            None => upload(&dev, x.bytes, size)?,
                        };
                        let raw = |n: u64| (n / 4).min(u64::from(u32::MAX)) as u32;
                        match (kind, x.uav) {
                            (Kind::Cbv, _) => dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: res.GetGPUVirtualAddress(), SizeInBytes: n }), at(*s)),
                            (Kind::Raw, false) => dev.CreateShaderResourceView(&res, Some(&srv_buf(DXGI_FORMAT_R32_TYPELESS, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(*s)),
                            (Kind::RawUav { .. } | Kind::Raw, true) | (Kind::RawUav { .. }, false) => dev.CreateUnorderedAccessView(&res, None, Some(&uav_buf(DXGI_FORMAT_R32_TYPELESS, raw(size), 0, D3D12_BUFFER_UAV_FLAG_RAW)), at(*s)),
                            (Kind::Structured(stride), false) => dev.CreateShaderResourceView(&res, Some(&srv_buf(DXGI_FORMAT_UNKNOWN, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(*s)),
                            (Kind::Structured(stride), true) => dev.CreateUnorderedAccessView(&res, None, Some(&uav_buf(DXGI_FORMAT_UNKNOWN, n, stride.max(4), D3D12_BUFFER_UAV_FLAG_NONE)), at(*s)),
                            (Kind::Typed { format, .. }, false) => dev.CreateShaderResourceView(&res, Some(&srv_buf(DXGI_FORMAT(format), n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(*s)),
                            (Kind::Typed { format, .. }, true) => dev.CreateUnorderedAccessView(&res, None, Some(&uav_buf(DXGI_FORMAT(format), n, 0, D3D12_BUFFER_UAV_FLAG_NONE)), at(*s)),
                        }
                        // The raw twin of a structured or typed view: the same bytes, raw.
                        if twin != 0 {
                            if x.uav {
                                dev.CreateUnorderedAccessView(&res, None, Some(&uav_buf(DXGI_FORMAT_R32_TYPELESS, raw(size), 0, D3D12_BUFFER_UAV_FLAG_RAW)), at(twin - 1));
                            } else {
                                dev.CreateShaderResourceView(&res, Some(&srv_buf(DXGI_FORMAT_R32_TYPELESS, raw(size), 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(twin - 1));
                            }
                        }
                        keep.push(res);
                    }
                    Res::Image(i) => {
                        let res = match kept {
                            // Read in a format its resource does not cast to (the same
                            // bytes, `prepare`): the bits copied into a texture of that format.
                            Some(k) if !castable(k.format, typed(i.format)) && !castable(k.format, i.format) => {
                                let desc = texture_desc(i, false);
                                let mut got: Option<ID3D12Resource> = None;
                                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut got).map_err(e)?;
                                let alias = got.ok_or("no texture")?;
                                let (kf, total) = footprints(&dev, &k.res.GetDesc());
                                let (af, _) = footprints(&dev, &desc);
                                let bits = bridge::buffer(&dev, total, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
                                let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
                                if k.state != copy {
                                    list.ResourceBarrier(&[barrier(&k.res, k.state, copy)]);
                                }
                                for (j, (fp, _, _)) in kf.iter().enumerate() {
                                    let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&bits), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
                                    let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&k.res), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: j as u32 } };
                                    list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                                }
                                list.ResourceBarrier(&[barrier(&bits, D3D12_RESOURCE_STATE_COPY_DEST, copy)]);
                                for (j, ((fp, _, _), (afp, _, _))) in kf.iter().zip(&af).enumerate() {
                                    let mut f = *fp;
                                    f.Footprint.Format = afp.Footprint.Format;
                                    let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&alias), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: j as u32 } };
                                    let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&bits), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f } };
                                    list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                                }
                                list.ResourceBarrier(&[barrier(&alias, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
                                if k.state != copy {
                                    list.ResourceBarrier(&[barrier(&k.res, copy, k.state)]);
                                }
                                keep.push(bits);
                                alias
                            }
                            Some(k) => k.res.clone(),
                            None => {
                                // An input texture with no kept writer: made for this dispatch.
                                let desc = texture_desc(i, false);
                                let mut got: Option<ID3D12Resource> = None;
                                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut got).map_err(|x| format!("texture {}x{}x{} format {}: {}", i.width, i.height, i.depth, i.format, e(x)))?;
                                let r = got.ok_or("no texture")?;
                                if !i.texels.is_empty() {
                                    let (fps, total) = footprints(&dev, &desc);
                                    let up = bridge::buffer(&dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
                                    let mut to = std::ptr::null_mut();
                                    up.Map(0, None, Some(&mut to)).map_err(e)?;
                                    let mut from = 0usize;
                                    for (fp, rows, row) in &fps {
                                        for z in 0..fp.Footprint.Depth as usize {
                                            for y in 0..*rows as usize {
                                                let n = (*row as usize).min(i.texels.len().saturating_sub(from));
                                                std::ptr::copy_nonoverlapping(i.texels.as_ptr().add(from), to.cast::<u8>().add(fp.Offset as usize + (z * *rows as usize + y) * fp.Footprint.RowPitch as usize), n);
                                                from += *row as usize;
                                            }
                                        }
                                    }
                                    up.Unmap(0, None);
                                    for (k, (fp, _, _)) in fps.iter().enumerate() {
                                        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: k as u32 } };
                                        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
                                        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                                    }
                                    keep.push(up);
                                }
                                list.ResourceBarrier(&[barrier(&r, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
                                r
                            }
                        };
                        let format = DXGI_FORMAT(if *atomic && i.uav && matches!(i.format, 39..=43) { 42 } else { typed(i.format) } as i32);
                        if i.uav {
                            let mut d = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: format, ..Default::default() };
                            match i.shape {
                                Shape::Tex1D => {
                                    d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
                                    d.Anonymous.Texture1D = D3D12_TEX1D_UAV::default();
                                }
                                Shape::Tex2D => {
                                    d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                                    d.Anonymous.Texture2D = D3D12_TEX2D_UAV::default();
                                }
                                Shape::Tex2DArray => {
                                    d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                                    d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV { ArraySize: i.depth.max(1), ..Default::default() };
                                }
                                Shape::Tex3D => {
                                    d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                                    d.Anonymous.Texture3D = D3D12_TEX3D_UAV { WSize: i.depth.max(1), ..Default::default() };
                                }
                            }
                            dev.CreateUnorderedAccessView(&res, None, Some(&d), at(*s));
                        } else {
                            let mut d = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, ..Default::default() };
                            match i.shape {
                                Shape::Tex1D => {
                                    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
                                    d.Anonymous.Texture1D = D3D12_TEX1D_SRV { MipLevels: 1, ..Default::default() };
                                }
                                Shape::Tex2D => {
                                    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                                    d.Anonymous.Texture2D = D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() };
                                }
                                Shape::Tex2DArray => {
                                    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                                    d.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: i.depth.max(1), ..Default::default() };
                                }
                                Shape::Tex3D => {
                                    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                                    d.Anonymous.Texture3D = D3D12_TEX3D_SRV { MipLevels: 1, ..Default::default() };
                                }
                            }
                            dev.CreateShaderResourceView(&res, Some(&d), at(*s));
                        }
                        keep.push(res);
                    }
                }
                if r.uav() && r.key() != 0 {
                    ran.wrote.push(r.key());
                }
            }
            for (key, state) in &wanted {
                if let Some(k) = self.kept.get_mut(key)
                    && k.state != *state
                {
                    list.ResourceBarrier(&[barrier(&k.res, k.state, *state)]);
                    k.state = *state;
                }
            }
            // Samplers: the default (linear, wrap) everywhere, the captured entries at their indices.
            let sstep = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
            let sat = |s: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: samplers.GetCPUDescriptorHandleForHeapStart().ptr + s * sstep };
            let wrap = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            for s in 0..b.sampler_slots as usize {
                dev.CreateSampler(&D3D12_SAMPLER_DESC { Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR, AddressU: wrap, AddressV: wrap, AddressW: wrap, MaxLOD: f32::MAX, ..Default::default() }, sat(s));
            }
            for (s, desc) in b.samplers.iter().filter(|s| s.0 < b.sampler_slots) {
                dev.CreateSampler(desc, sat(*s as usize));
            }
            // Root parameters (emit::host root_signature_records with no
            // records): user data, root views by (root, SRV), stride table, bounds.
            list.SetDescriptorHeaps(&[Some(views.clone()), Some(samplers.clone())]);
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRoot32BitConstants(0, b.mesh.constants.len() as u32, b.mesh.constants.as_ptr().cast(), 0);
            let mut param = 1u32;
            let mut rv: Vec<(u8, bool)> = shader.root_cbvs.iter().map(|&r| (r, false)).chain(shader.root_srvs.iter().map(|&r| (r, true))).collect();
            rv.sort_unstable();
            rv.dedup();
            for (r, srv) in rv {
                let bytes = b.mesh.views.iter().find(|v| v.0 == r && v.1 == srv).map_or(&[][..], |v| v.2);
                let buf = upload(&dev, bytes, (bytes.len() as u64).next_multiple_of(256).max(256))?;
                if srv {
                    list.SetComputeRootShaderResourceView(param, buf.GetGPUVirtualAddress());
                } else {
                    list.SetComputeRootConstantBufferView(param, buf.GetGPUVirtualAddress());
                }
                keep.push(buf);
                param += 1;
            }
            if shader.reads_stride_table() {
                let words: Vec<u8> = b.views.strides.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
                let table = upload(&dev, &words, words.len() as u64)?;
                list.SetComputeRootShaderResourceView(param, table.GetGPUVirtualAddress());
                keep.push(table);
                param += 1;
            }
            if shader.reads_heap_bounds() {
                list.SetComputeRoot32BitConstants(param, 8, b.views.bounds.as_ptr().cast(), 0);
            }
            if batched {
                // After the dispatches recorded before it (they may write what it reads).
                list.ResourceBarrier(&[uav_all()]);
                list.Dispatch(job.groups[0], job.groups[1], job.groups[2]);
                let b = self.batch.as_mut().expect("batch");
                b.keep.extend(keep);
                b.heaps.extend([views, samplers]);
                b.recorded += 1;
                ran.wrote.sort_unstable();
                ran.wrote.dedup();
                ran.read_kept.sort_unstable();
                ran.read_kept.dedup();
                return Ok(ran);
            }
            let stamps: ID3D12QueryHeap = {
                let mut h: Option<ID3D12QueryHeap> = None;
                dev.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut h).map_err(e)?;
                h.ok_or("no query heap")?
            };
            let times = bridge::buffer(&dev, 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            list.Dispatch(job.groups[0], job.groups[1], job.groups[2]);
            list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &times, 0);
            self.submit(&list)?;
            let mut p = std::ptr::null_mut();
            times.Map(0, None, Some(&mut p)).map_err(e)?;
            let t = std::slice::from_raw_parts(p.cast::<u64>(), 2);
            ran.seconds = t[1].saturating_sub(t[0]) as f64 / self.host.queue.GetTimestampFrequency().map_err(e)?.max(1) as f64;
            times.Unmap(0, None);
            drop((views, samplers, stamps, keep));
        }
        ran.wrote.sort_unstable();
        ran.wrote.dedup();
        ran.read_kept.sort_unstable();
        ran.read_kept.dedup();
        Ok(ran)
    }

    /// `len` bytes at title address `at` from the held buffer that covers
    /// them (a record at an offset of a buffer an earlier dispatch wrote:
    /// an indirect dispatch's group counts, LIT6); None when none does.
    pub fn read_at(&mut self, at: u64, len: u64) -> Option<Vec<u8>> {
        let base = covering(self.kept.iter().map(|(k, v)| (*k, v.held)), at, len)?;
        let b = self.read(base).ok()?;
        b.get((at - base) as usize..(at - base + len) as usize).map(<[u8]>::to_vec)
    }

    /// The bytes held at `key`: a buffer's whole length, a texture's mip 0
    /// as unpadded rows, slices in order (the layout [`Image::texels`] takes).
    pub fn read(&mut self, key: u64) -> Result<Vec<u8>, String> {
        self.flush()?;
        let (_alloc, list) = self.list()?;
        let dev = self.host.device.clone();
        let e = |e: windows::core::Error| e.message().to_string();
        let k = self.kept.get_mut(&key).ok_or_else(|| format!("nothing held at {key:#x}"))?;
        // SAFETY: D3D12 calls on live objects; the readback buffer is mapped after the fence.
        unsafe {
            let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
            if k.state != copy {
                list.ResourceBarrier(&[barrier(&k.res, k.state, copy)]);
                k.state = copy;
            }
            let (rb, layout) = match k.held {
                Held::Buffer { len } => {
                    let rb = bridge::buffer(&dev, len, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
                    list.CopyBufferRegion(&rb, 0, &k.res, 0, len.min(k.res.GetDesc().Width));
                    (rb, None)
                }
                Held::Image { .. } => {
                    let desc = k.res.GetDesc();
                    let (fps, total) = footprints(&dev, &desc);
                    let rb = bridge::buffer(&dev, total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
                    for (i, (fp, _, _)) in fps.iter().enumerate() {
                        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
                        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&k.res), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 } };
                        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                    }
                    (rb, Some(fps))
                }
            };
            let held = k.held;
            self.submit(&list)?;
            let mut p = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut p)).map_err(e)?;
            let base = p.cast::<u8>();
            let out = match (held, layout) {
                (Held::Buffer { len }, _) => std::slice::from_raw_parts(base, len as usize).to_vec(),
                (_, Some(fps)) => {
                    let mut out = Vec::new();
                    for (fp, rows, row) in fps {
                        for z in 0..fp.Footprint.Depth as usize {
                            for y in 0..rows as usize {
                                let at = fp.Offset as usize + (z * rows as usize + y) * fp.Footprint.RowPitch as usize;
                                out.extend_from_slice(std::slice::from_raw_parts(base.add(at), row as usize));
                            }
                        }
                    }
                    out
                }
                _ => Vec::new(),
            };
            rb.Unmap(0, None);
            Ok(out)
        }
    }

    /// A 2D texture held at `key` as a [`Provided`] (the draw bridge's
    /// texture input): the next consumer of a chain that is a draw.
    pub fn provided(&mut self, key: u64) -> Result<Provided, String> {
        self.provided_as(key, None, false)
    }

    /// [`Runner::provided`] read in `view` format (the consumer's view: an
    /// R32 texture read as R9G9B9E5 takes its bits, `same_bytes`); a 1D
    /// texture as a one-row 2D one when `one_row` (the consumer reads it
    /// 2D), else 1D; a 3D one as 3D ([`Provided::dim`]).
    pub fn provided_as(&mut self, key: u64, view: Option<u32>, one_row: bool) -> Result<Provided, String> {
        let Some(Held::Image { shape, format, width, height, depth }) = self.held(key) else {
            return Err(format!("no texture held at {key:#x}"));
        };
        let format = match view.filter(|&v| v != 0 && !castable(res_format(format), v) && !castable(res_format(format), typed(v))) {
            None => view.filter(|&v| v != 0).map_or(typed(format), typed),
            Some(v) => {
                let as_view = Image { key, shape, format: v, width, height, depth, uav: false, texels: &[] };
                let res = self.kept.get(&key).map(|k| k.res.clone()).ok_or("not held")?;
                if !self.same_bytes(&res, &as_view) {
                    return Err(format!("{key:#x}: format {format} held, read as {v}: other sizes"));
                }
                v
            }
        };
        let rows = self.read(key)?;
        let (dim, height, depth) = match shape {
            Shape::Tex1D if !one_row => (crate::textures::TEX_1D, 1, 1),
            Shape::Tex3D => (crate::textures::TEX_3D, height.max(1), depth.max(1)),
            Shape::Tex2DArray => (crate::textures::TEX_2D_ARRAY, height.max(1), depth.max(1)),
            _ => (crate::textures::TEX_2D, height.max(1), 1),
        };
        let row_count = height * depth;
        let row_bytes = (rows.len() / row_count as usize) as u32;
        Ok(Provided { width, height, format, rows, row_bytes, row_count, dim, depth, mips: 1 })
    }
}

/// The key of the held buffer (the nearest start at or below `at`) whose
/// length covers `len` bytes at `at`.
fn covering(held: impl Iterator<Item = (u64, Held)>, at: u64, len: u64) -> Option<u64> {
    held.filter(|(k, h)| matches!(h, Held::Buffer { len: l } if *k <= at && at + len <= k + l)).map(|(k, _)| k).max()
}

fn upload(dev: &ID3D12Device, bytes: &[u8], size: u64) -> Result<ID3D12Resource, String> {
    let r = bridge::buffer(dev, size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
    let mut at = std::ptr::null_mut();
    // SAFETY: an upload buffer of at least max(size, 256) bytes, mapped here.
    unsafe {
        r.Map(0, None, Some(&mut at)).map_err(|e| e.message().to_string())?;
        std::ptr::write_bytes(at.cast::<u8>(), 0, size.max(256) as usize);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), at.cast::<u8>(), bytes.len().min(size as usize));
        r.Unmap(0, None);
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_shader::native::{ComputeIo, ResourceType, RootParam, StageIo, UsedResource};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// SOP1 `s_mov_b32 sdst, src0` (xnw-shader image_atomic_tests).
    const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
    }

    /// SMEM `s_load_dwordx{4,8} s[dst:..], s[base:base+1], byte_off`.
    const fn s_load(op: u8, dst: u8, base: u8, byte_off: u32) -> [u32; 2] {
        [0x3D << 26 | (op as u32) << 18 | (dst as u32) << 6 | (base as u32 >> 1), 125 << 25 | byte_off]
    }

    /// VOP1 `v_mov_b32 vdst, src0` (0..105 SGPR, 128 = 0, 129 = 1, 256.. VGPR).
    const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
        0x7E00_0000 | (vdst as u32) << 17 | 0x001 << 9 | (src0 as u32 & 0x1FF)
    }

    /// MIMG `op vdata, vaddr[2], s[srsrc:srsrc+7] dmask dim:2D`.
    const fn mimg(op: u8, vdata: u8, vaddr: u8, srsrc: u8, dmask: u8) -> [u32; 2] {
        [0x3C << 26 | ((op as u32) & 0x7F) << 18 | (dmask as u32) << 8 | 1 << 3, (srsrc as u32 >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32]
    }

    const IMAGE_LOAD: u8 = 0x00;
    const IMAGE_STORE: u8 = 0x08;

    /// One thread: root constants s0..s3, the table (root 1) at s4. Loads
    /// texel (0, 0) of slot 0 (a 2D SRV), stores s0..s3 at texel (0, 0) of
    /// slot 1 (a 2D UAV) and the loaded texel at (1, 0).
    fn program() -> NativeProgram {
        let mut w = vec![s_mov_b32(5, 0x80)];
        w.extend(s_load(0x03, 8, 4, 0));
        w.extend(s_load(0x03, 16, 4, 32));
        w.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
        for i in 0..4u8 {
            w.push(v_mov_b32(i + 1, u16::from(i)));
        }
        w.push(v_mov_b32(5, 128));
        w.push(v_mov_b32(6, 128));
        w.extend(mimg(IMAGE_LOAD, 7, 5, 8, 0xF));
        w.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
        w.extend(mimg(IMAGE_STORE, 1, 5, 16, 0xF));
        w.push(v_mov_b32(5, 129));
        w.extend(mimg(IMAGE_STORE, 7, 5, 16, 0xF));
        w.push(S_ENDPGM);
        let used = |kind, table_offset_dwords| UsedResource { kind, root: RootType::Table, table_offset_dwords, lower_bound: table_offset_dwords / 8, stride: 0, dwords: 8, user_data: 4, slot: 1 };
        NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 12,
            layout: vec![
                RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
                RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 1 },
            ],
            used: vec![used(ResourceType::Srv, 0), used(ResourceType::Uav, 8)],
            io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [1, 1, 1] }),
            code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    fn translate(native: &NativeProgram) -> HostShader {
        let p = xnw_shader::ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
        let res = xnw_shader::resource::track(&p, &native.layout, native.stage).expect("track");
        let io = xnw_shader::io::raise(&p, native).expect("io");
        let st = xnw_shader::structure::structure(&p).expect("structure");
        // A container of no parts: a compute program has no signatures.
        let mut empty = b"DXBC".to_vec();
        empty.extend([0; 16]);
        empty.extend(1u16.to_le_bytes());
        empty.extend(0u16.to_le_bytes());
        empty.extend(32u32.to_le_bytes());
        empty.extend(0u32.to_le_bytes());
        let c = xnw_shader::Container::parse(&empty).expect("container");
        emit::emit(&c, native, &p, &st, &io, &res).expect("emit")
    }

    const RGBA32F: u32 = 2;

    fn image(key: u64, uav: bool, texels: &[u8]) -> Res<'_> {
        Res::Image(Image { key, shape: Shape::Tex2D, format: RGBA32F, width: 4, height: 1, depth: 1, uav, texels })
    }

    fn job<'a>(c: [f32; 4], src: Res<'a>, dst: Res<'a>) -> Job<'a> {
        Job { roots: vec![Arg::Constants(c.map(f32::to_bits).to_vec()), Arg::Table(vec![(0, src), (1, dst)])], heap: Vec::new(), samplers: Vec::new(), sampler_slots: 1, groups: [1, 1, 1] }
    }

    #[test]
    fn uses_read_the_program() {
        let u = uses_of(&program()).expect("uses");
        assert_eq!(u, vec![Use { root: 1, slot: 0, image: Some(ImageUse { dim: 1, atomic: false }), store: false, uav: false }, Use { root: 1, slot: 1, image: Some(ImageUse { dim: 1, atomic: false }), store: true, uav: true }]);
        // The capture's unordered view of a slot the program only reads binds as an SRV.
        let j = by_program(&job([0.0; 4], image(0x10, true, &[]), image(0x20, false, &[])), &u);
        let Arg::Table(t) = &j.roots[1] else { panic!("table") };
        assert_eq!(t.iter().map(|(_, r)| r.uav()).collect::<Vec<_>>(), vec![false, true]);
    }

    #[test]
    fn covering_finds_the_buffer_around_a_record() {
        let held = [(0x100, Held::Buffer { len: 64 }), (0x120, Held::Buffer { len: 16 }), (0x200, Held::Image { shape: Shape::Tex2D, format: 2, width: 4, height: 4, depth: 1 })];
        assert_eq!(covering(held.into_iter(), 0x120, 12), Some(0x120), "the nearest start wins");
        assert_eq!(covering(held.into_iter(), 0x134, 12), Some(0x100), "past the small one: the big one");
        assert_eq!(covering(held.into_iter(), 0x13c, 12), None, "runs past the end");
        assert_eq!(covering(held.into_iter(), 0x200, 4), None, "textures hold no records");
    }

    #[test]
    fn check_drops_wrong_kinds_and_heap_indices() {
        let u = uses_of(&program()).expect("uses");
        let all = |_: u32| true;
        let buf = Res::Buffer(Buffer { key: 0x10, kind: Kind::Raw, uav: false, len: 64, bytes: &[] });
        assert!(check(&job([0.0; 4], image(0x10, false, &[]), image(0x20, true, &[])), &u, &all).is_ok());
        assert_eq!(check(&job([0.0; 4], buf, image(0x20, true, &[])), &u, &all).map_err(|d| d.reason), Err(why::BUFFER_AS_IMAGE));
        let arr = Res::Image(Image { key: 0x10, shape: Shape::Tex3D, format: RGBA32F, width: 4, height: 4, depth: 4, uav: false, texels: &[] });
        assert_eq!(check(&job([0.0; 4], arr, image(0x20, true, &[])), &u, &all).map_err(|d| d.reason), Err(why::DIMENSION));
        assert_eq!(check(&job([0.0; 4], image(0x20, false, &[]), image(0x20, true, &[])), &u, &all).map_err(|d| d.reason), Err(why::READ_WRITE));
        assert_eq!(check(&job([0.0; 4], image(0x10, false, &[]), image(0x20, true, &[])), &u, &|_| false).map_err(|d| d.reason), Err(why::NO_TYPED_UAV));
        let mut j = job([0.0; 4], image(0x10, false, &[]), image(0x20, true, &[]));
        j.heap.push((HEAP_MAX, buf));
        assert_eq!(check(&j, &u, &all).map_err(|d| d.reason), Err(why::HEAP_INDEX));
        j.heap.clear();
        j.groups = [1, 0, 1];
        assert_eq!(check(&j, &u, &all).map_err(|d| d.reason), Err(why::GROUPS));
        // A buffer program over a texture: the reverse rule.
        let b = vec![Use { root: 1, slot: 0, image: None, store: false, uav: false }];
        assert_eq!(check(&job([0.0; 4], image(0x10, false, &[]), image(0x20, true, &[])), &b, &all).map_err(|d| d.reason), Err(why::IMAGE_AS_BUFFER));
        assert!(castable(res_format(41), 42) && castable(9, 10) && !castable(39, 67), "R32 casts to uint, not to R9G9B9E5");
    }

    /// A texture UAV round trip on WARP: dispatch 1 writes texture B (kept
    /// by its address), dispatch 2 reads B as an SRV and writes C; C read
    /// back holds dispatch 2's constants and dispatch 1's. Skips without
    /// `XNW_TEST_GPU=1` or on an adapter below shader model 6.6.
    #[test]
    fn texture_uav_round_trip() {
        if !xnw_gpu::host::device_allowed() {
            return;
        }
        let native = program();
        let shader = translate(&native);
        let u = uses_of(&native).expect("uses");
        let (_gpu, host) = crate::tests::warp_host();
        if let Some(gap) = preskin::adapter_gap(&host.caps, 32) {
            eprintln!("compute: not run: {gap}");
            return;
        }
        let mut runner = Runner::new(&host).expect("runner");
        let seed: Vec<u8> = [0.25f32, 0.5, 0.75, 1.0].iter().chain(&[0.0; 12]).flat_map(|f| f.to_le_bytes()).collect();
        let first = runner.dispatch(&shader, 32, &u, &job([1.0, 2.0, 3.0, 4.0], image(0, false, &seed), image(0xB000, true, &[]))).expect("dispatch 1");
        assert!(matches!(&first, Outcome::Ran(r) if r.wrote == vec![0xB000]), "{first:?}");
        let second = runner.dispatch(&shader, 32, &u, &job([5.0, 6.0, 7.0, 8.0], image(0xB000, false, &[]), image(0xC000, true, &[]))).expect("dispatch 2");
        assert!(matches!(&second, Outcome::Ran(r) if r.read_kept == vec![0xB000] && r.wrote == vec![0xC000]), "{second:?}");
        let f = |b: Vec<u8>| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>();
        let b = f(runner.read(0xB000).expect("read B"));
        assert_eq!(&b[..8], &[1.0, 2.0, 3.0, 4.0, 0.25, 0.5, 0.75, 1.0], "dispatch 1: constants at 0, the seed at 1");
        let c = f(runner.read(0xC000).expect("read C"));
        eprintln!("compute: B {:?}, C {:?} ({:.6} s on the GPU)", &b[..8], &c[..8], runner.seconds);
        assert_eq!(&c[..8], &[5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0], "dispatch 2 read what dispatch 1 wrote");
        assert_eq!(runner.provided(0xC000).map(|p| (p.width, p.height, p.row_bytes)), Ok((4, 1, 64)));
        // A kept texture read as a buffer: dropped, counted, nothing run.
        let buf = Res::Buffer(Buffer { key: 0xB000, kind: Kind::Raw, uav: false, len: 64, bytes: &[] });
        let b = vec![Use { root: 1, slot: 0, image: Some(ImageUse { dim: 1, atomic: false }), store: false, uav: false }];
        let third = runner.dispatch(&shader, 32, &b, &job([0.0; 4], buf, image(0xD000, true, &[]))).expect("dispatch 3");
        assert!(matches!(third, Outcome::Dropped(ref d) if d.reason == why::BUFFER_AS_IMAGE), "{third:?}");
        // A program whose pipeline fails: dropped before anything is recorded.
        let broken = HostShader { dxil: vec![0; 32], ..shader.clone() };
        let fourth = runner.dispatch(&broken, 32, &u, &job([0.0; 4], image(0, false, &seed), image(0xE000, true, &[]))).expect("dispatch 4");
        assert!(matches!(fourth, Outcome::Dropped(ref d) if d.reason == why::PIPELINE), "{fourth:?}");
        assert_eq!(runner.held(0xE000), None, "nothing made for a dropped dispatch");
        assert_eq!(runner.ran, 2);
        assert_eq!(runner.dropped.get(why::BUFFER_AS_IMAGE), Some(&1));
        assert_eq!(runner.dropped.get(why::PIPELINE), Some(&1));
    }
}
