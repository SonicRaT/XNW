//! The title's own pixel programs in our pass (`XNW_WORLD_RENDER=...,
//! bridgeps=on|<hash>+<hash>`, default off). The rows of the named
//! pixel programs are drawn a second time, after our own pass and inside
//! its command list, with the title's mesh program (compute form) and
//! pixel program into our colour target, depth tested GREATER_EQUAL
//! against our depth without writing it (`bridge::Into`): where our pass
//! shows the surface, the title's program gives its colour.
//!
//! One path for the live title and the replay:
//! - [`take`] (pure over a [`Source`]): one draw's binding inputs as owned
//!   bytes ([`Owned`]): root constants, root views, the table views the two
//!   programs name, its 16 draw words and the textures they name (by
//!   descriptor handle), the sampler heap. The live source is the list
//!   sink's tap (apps/xnw `worldprobe/bridgeps.rs`), the replay's a frame
//!   pack sink (xnw-frame `bridge_ps.rs`).
//! - [`Cache`]: the translated pair and its two pipelines, made once per
//!   (mesh, pixel) container pair, on a thread of its own (live: a row
//!   whose pipelines are not ready is drawn by our programs alone).
//! - [`Gpu::record`]: every row of a [`Frame`] on our pass's open list. A
//!   texture is the host copy our texture table holds for its handle this
//!   frame (a 1x1 grey stands in when it has none). Its device objects are
//!   kept across frames: one view heap for all rows (each row's slots
//!   at its own base, [`bridge::bind_at`]), sampler heaps by content, the
//!   inputs a source hands over again as the same bytes in one default
//!   buffer ([`bridge::Kept`]: the live tap's kept views, vertex streams),
//!   one mapped upload buffer for the rest (frame tables, constants).
//!
//! The draw words are rewritten: word k naming a texture becomes the
//! row's base + k (that texture at view heap slot base + k; rows of one
//! words offset share the first one's), every other word [`NO_VIEW`] (past
//! the heap bounds: the guard). A row's slots are the 16 word slots and
//! the tables, not the title's heap up to its largest index. The frame
//! table's slot 86 (the words' buffer, past 32 MiB in a captured frame) is one buffer
//! our pass keeps across frames: each row's rewritten words at the
//! row's own offset (root 4 dword 1, which the mesh programs read too:
//! not rewritten); the bytes between are never read.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use xnw_formats::framepack::view;
use xnw_shader::emit::{self, HostShader};
use xnw_shader::HwStage;

use super::bridge::{self, Arena, At, Draw, KEEP_MIN, KEPT_LEAST, Kept, Params, Programs, Root, SamplerDesc, Slot, unit};
use super::preskin::{Kind, View};

/// In [`Used`]: a slot read as a constant buffer; [`NAMED_IMAGE`] a slot
/// read through an image descriptor (8 dwords); other values: the stride a
/// buffer is read at.
pub const NAMED_CBV: u32 = u32::MAX;
pub const NAMED_IMAGE: u32 = u32::MAX - 1;
/// The table slots the two programs name, (root, slot) -> stride or [`NAMED_CBV`].
pub type Used = HashMap<(u16, u32), u32>;
/// The draw words: the frame table's (root 0) slot 86, at the byte offset
/// in root 4's constant dword 1 (`worldprobe/pack.rs` `draw_words`).
pub const WORDS_SLOT: u32 = 86;
pub const WORDS_ROOT: usize = 4;
pub const WORDS_DWORD: usize = 1;
pub const WORDS: usize = 16;
/// A rewritten draw word that names no texture: past every heap bound
/// (our view heap holds at most 1,000,000 slots; times 32 still fits 32
/// bits). Small enough that the stride table covers it: the translated
/// programs read the table at a slot before the slot is bounded
/// (`emit::lower::address` `stride_entry`), and a root view has no bounds
/// on the hardware adapter (WARP reads zeros past the buffer).
pub const NO_VIEW: u32 = 0x000F_FFFF;
/// One table view's bytes at most.
pub const VIEW_MAX: u64 = 16 << 20;
/// Bit of a mesh program's first root constant: set, group k draws the
/// meshlet the title's visible-meshlet list names at k (a GPU output our
/// pass does not have); cleared, meshlet k.
const VISIBLE_LIST: u32 = 1 << 28;
const GPU_ALIAS: u64 = 0x4_0000_0000;

/// The table slots a console container's programs name, with the widest
/// stride each is read at, into `used`; `wave` becomes 64 for a mesh
/// program of 64-lane waves.
pub fn named(container: &[u8], used: &mut Used, wave: &mut u32) -> Result<(), String> {
    let parsed = xnw_shader::Container::parse(container).map_err(|e| e.to_string())?;
    let part = parsed.part(xnw_shader::FourCC::DXIL).ok_or("no DXIL part")?;
    for p in xnw_shader::native_programs(part.data).map_err(|e| e.to_string())? {
        if p.stage == HwStage::Ms && p.wave64 {
            *wave = 64;
        }
        for u in p.used.iter().filter(|u| u.root == xnw_shader::RootType::Table) {
            let e = used.entry((u16::from(u.slot), u.table_offset_dwords / 8)).or_insert(0);
            *e = (*e).max(match u.kind {
                xnw_shader::ResourceType::Cbv => NAMED_CBV,
                _ if u.dwords == 8 => NAMED_IMAGE,
                _ => u.stride,
            });
        }
    }
    Ok(())
}

/// The translated pair of a mesh and a pixel container, and the table
/// slots they name.
pub fn translate(msc: &[u8], psc: &[u8]) -> Result<(Programs, Used), String> {
    // `XNW_HEAP_NONUNIFORM=<hash prefix>,..|all` reaches the bridged programs too.
    let opts = |c: &[u8]| emit::Options { nonuniform_heap: emit::nonuniform_heap_for(xnw_core::hash128(&[c])), ..bridge::options() };
    let (ms, layout) = emit::mesh::translate_fallback(msc, opts(msc)).map_err(|e| format!("mesh program: {e}"))?;
    let ps = emit::translate_with(psc, opts(psc)).map_err(|e| format!("pixel program: {e}"))?;
    let vs = emit::mesh::passthrough_vs(&layout, Some(&ps.dxil)).map_err(|e| format!("pass-through vertex shader: {e}"))?;
    let mut used = HashMap::new();
    let mut wave = 32;
    for c in [msc, psc] {
        named(c, &mut used, &mut wave)?;
    }
    Ok((Programs { ms, layout, vs, ps, wave, vertex: None }, used))
}

/// A descriptor slot as a source describes it (`xnw_formats::framepack::view` kinds).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SrcView {
    pub kind: u32,
    pub addr: u64,
    pub bytes: u64,
    pub format: u32,
    /// A buffer view's element size (0: none given).
    pub width: u64,
}

/// A root argument as a source has it.
#[derive(Clone, Debug, PartialEq)]
pub enum SrcRoot {
    None,
    Constants(Vec<u32>),
    /// A descriptor table: its first slot's handle.
    Table(u64),
    /// A root view's GPU address.
    View(u64),
}

/// Where a draw's binding inputs come from (the title at its list sink,
/// or a frame pack's sink).
pub trait Source {
    fn roots(&self) -> Vec<SrcRoot>;
    /// The view at table root `root`, slot `slot` (None: no view there).
    fn table(&self, root: u16, slot: u32) -> Option<SrcView>;
    /// A buffer or constant view's bytes (at most [`VIEW_MAX`]).
    fn bytes(&self, root: u16, slot: u32, v: &SrcView) -> Option<Arc<Vec<u8>>>;
    /// A root view's bytes from its address.
    fn root_view(&self, root: u16, addr: u64) -> Option<Arc<Vec<u8>>>;
    /// The draw's 16 words (None: not readable).
    fn words(&self) -> Option<[u32; WORDS]>;
    /// The view at heap index `index` of the bound view heap, and its handle.
    fn heap(&self, index: u32) -> Option<(u64, SrcView)>;
    /// The bound sampler heap's entries, as host descriptions.
    fn samplers(&self) -> Option<Arc<Vec<(u32, SamplerDesc)>>>;
}

/// A slot's input as owned bytes or a texture's handle.
#[derive(Clone, Debug)]
pub enum OSlot {
    Buffer(Kind, Arc<Vec<u8>>),
    Texture(u64),
}

#[derive(Clone, Debug, Default)]
pub enum ORoot {
    #[default]
    None,
    Constants(Vec<u32>),
    Table(Vec<(u32, OSlot)>),
    View(Arc<Vec<u8>>),
}

/// The (mesh, pixel) container hashes of a pipeline.
pub type Key = (u128, u128);

/// One draw's binding inputs ([`take`]).
#[derive(Clone, Debug, Default)]
pub struct Owned {
    pub key: Key,
    pub roots: Vec<ORoot>,
    /// Textures the draw words name: (word k, handle).
    pub heap: Vec<(u32, u64)>,
    /// The rewritten words and their byte offset in the title's slot 86.
    pub words: [u32; WORDS],
    pub words_at: u32,
    pub samplers: Arc<Vec<(u32, SamplerDesc)>>,
    /// Mesh groups (the meshlet buffer's count; 0: not known yet).
    pub groups: u32,
    /// Buffer views by title address and their first word (the meshlet
    /// buffer is the one at the row's `mesh_at`).
    pub firsts: Vec<(u64, u32)>,
    /// What stood in, by [`LACKS`] index.
    pub lacks: u32,
    /// Title address of each buffer input, by its bytes' address (the
    /// `Arc`'s data): `XNW_ZEROCOPY=on` binds it there in place (ZC1).
    pub addrs: Vec<(usize, u64)>,
}

/// What [`Owned::lacks`] counts, by bit.
pub const LACKS: [&str; 9] = ["no draw words", "a word below the heap bound naming no view", "a named constant buffer without bytes (zeros)", "a named view without bytes", "no samplers (defaults)", "a root view without bytes", "no group count", "a texture read as a buffer (dropped)", "a buffer read as an image (dropped)"];

impl Owned {
    /// The group count from the meshlet buffer at `mesh_at` (CPU or GPU address).
    pub fn set_groups(&mut self, mesh_at: u64) {
        if let Some(&(_, w)) = self.firsts.iter().find(|(a, _)| *a >= mesh_at && a.wrapping_sub(mesh_at) % GPU_ALIAS == 0) {
            self.groups = w & 0x1FFFF;
        }
        if self.groups == 0 {
            self.lacks |= 1 << 6;
        }
    }

    /// The textures this draw samples, by handle.
    pub fn handles(&self) -> impl Iterator<Item = u64> + '_ {
        self.heap.iter().map(|h| h.1).chain(self.roots.iter().flat_map(|r| match r {
            ORoot::Table(v) => v.iter().filter_map(|s| if let OSlot::Texture(h) = s.1 { Some(h) } else { None }).collect(),
            _ => Vec::new(),
        }))
    }
}

/// The binding inputs of one draw of pipeline `key` whose programs name
/// `used` (pure over `src`).
pub fn take(src: &dyn Source, key: Key, used: &Used) -> Owned {
    let mut out = Owned { key, ..Owned::default() };
    let roots = src.roots();
    out.roots = vec![ORoot::None; roots.len()];
    for (i, r) in roots.iter().enumerate() {
        let root = i as u16;
        out.roots[i] = match r {
            SrcRoot::None => ORoot::None,
            SrcRoot::Constants(c) => ORoot::Constants(c.clone()),
            SrcRoot::View(a) => match src.root_view(root, *a) {
                Some(b) => {
                    out.addrs.push((b[..].as_ptr() as usize, *a));
                    ORoot::View(b)
                }
                None => {
                    out.lacks |= 1 << 5;
                    ORoot::None
                }
            },
            SrcRoot::Table(handle) => {
                let mut slots = Vec::new();
                let mut named: Vec<(u32, u32)> = used.iter().filter(|(k, _)| k.0 == root).map(|(k, s)| (k.1, *s)).collect();
                named.sort_unstable();
                for (slot, stride) in named {
                    // Slot 86 of the frame table is the frame's sparse words buffer.
                    if root == 0 && slot == WORDS_SLOT {
                        continue;
                    }
                    let Some(v) = src.table(root, slot) else {
                        if stride == NAMED_CBV {
                            // A program loading a constant buffer where the heap holds no view removes the device.
                            out.lacks |= 1 << 2;
                            slots.push((slot, OSlot::Buffer(Kind::Cbv, Arc::new(vec![0; 256]))));
                        }
                        continue;
                    };
                    match v.kind {
                        view::CONSTANT | view::BUFFER | view::PLACED_BUFFER if stride == NAMED_IMAGE => out.lacks |= 1 << 8,
                        view::CONSTANT | view::BUFFER | view::PLACED_BUFFER => {
                            let Some(b) = src.bytes(root, slot, &v) else {
                                if stride == NAMED_CBV {
                                    out.lacks |= 1 << 2;
                                    slots.push((slot, OSlot::Buffer(Kind::Cbv, Arc::new(vec![0; 256]))));
                                } else {
                                    out.lacks |= 1 << 3;
                                }
                                continue;
                            };
                            if b.len() >= 4 {
                                out.firsts.push((v.addr, u32::from_le_bytes([b[0], b[1], b[2], b[3]])));
                            }
                            // R32_TYPELESS (39) is a raw view.
                            let kind = match (v.kind, v.format, stride) {
                                (view::CONSTANT, ..) => Kind::Cbv,
                                (view::BUFFER, f, _) if f != 0 && f != 39 => Kind::Typed { format: f as i32, bytes: v.width.max(1) as u32 },
                                (_, _, NAMED_CBV) => Kind::Cbv,
                                (_, _, s) if s > 4 => Kind::Structured(s),
                                _ => Kind::Raw,
                            };
                            out.addrs.push((b[..].as_ptr() as usize, v.addr));
                            slots.push((slot, OSlot::Buffer(kind, b)));
                        }
                        // A texture read through a buffer descriptor gives the
                        // hardware a wild address (victory-fork 3c81420): the row is dropped.
                        view::TEXTURE | view::PLACED if stride != NAMED_IMAGE && stride != NAMED_CBV => out.lacks |= 1 << 7,
                        view::TEXTURE | view::PLACED => slots.push((slot, OSlot::Texture(handle + 32 * u64::from(slot)))),
                        _ => {}
                    }
                }
                ORoot::Table(slots)
            }
        };
    }
    for r in &mut out.roots {
        if let ORoot::Constants(c) = r
            && let Some(w) = c.first_mut()
        {
            *w &= !VISIBLE_LIST;
        }
    }
    out.words_at = match roots.get(WORDS_ROOT) {
        Some(SrcRoot::Constants(c)) => c.get(WORDS_DWORD).copied().unwrap_or(0),
        _ => 0,
    };
    match src.words() {
        Some(words) => {
            for (k, w) in words.iter().enumerate() {
                out.words[k] = NO_VIEW;
                match src.heap(*w) {
                    Some((h, v)) if v.kind == view::TEXTURE || v.kind == view::PLACED => {
                        out.words[k] = k as u32;
                        out.heap.push((k as u32, h));
                    }
                    Some(_) => {}
                    // A value past every heap bound is the material's data, not
                    // a view (floats 1.0, -1.0, 0.6... in pk6 f1205: words 4-15,
                    // `--bridgeps-inputs`): kept as it is (the sheet is unchanged).
                    None if *w > NO_VIEW => out.words[k] = *w,
                    None => out.lacks |= 1 << 1,
                }
            }
        }
        None => {
            out.words = [NO_VIEW; WORDS];
            out.lacks |= 1;
        }
    }
    out.samplers = src.samplers().unwrap_or_else(|| {
        out.lacks |= 1 << 4;
        Arc::new(Vec::new())
    });
    out
}

/// A frame's bridged rows: (list row, inputs), and the end of slot 86's
/// words (the largest offset + 64).
#[derive(Debug, Default)]
pub struct Frame {
    pub rows: Vec<(u32, Owned)>,
    pub words_end: u64,
    /// Draws of the wanted programs whose pipelines were not ready, and
    /// draws not read (tap counters, live).
    pub not_ready: u32,
    pub unread: u32,
}

impl Frame {
    pub fn new(rows: Vec<(u32, Owned)>) -> Frame {
        let words_end = rows.iter().map(|r| u64::from(r.1.words_at) + 4 * WORDS as u64).max().unwrap_or(0);
        Frame { rows, words_end, not_ready: 0, unread: 0 }
    }

    /// Writes every row's words at its offset into `to` (`len` bytes; rows past it left out).
    pub fn write_words(&self, to: &mut [u8]) {
        for (_, o) in &self.rows {
            if let Some(d) = to.get_mut(o.words_at as usize..o.words_at as usize + 4 * WORDS) {
                for (k, w) in o.words.iter().enumerate() {
                    d[4 * k..4 * k + 4].copy_from_slice(&w.to_le_bytes());
                }
            }
        }
    }

    /// Row `i` as the binding takes it (textures by handle).
    pub fn draw(&self, i: usize, sampler_slots: u32) -> Draw<'_> {
        let o = &self.rows[i].1;
        fn slot(s: &OSlot) -> Slot<'_> {
            match s {
                OSlot::Buffer(kind, bytes) => Slot::Buffer(View { kind: *kind, bytes }),
                OSlot::Texture(h) => Slot::Handle(*h),
            }
        }
        let roots = o
            .roots
            .iter()
            .enumerate()
            .map(|(r, x)| match x {
                ORoot::None => Root::None,
                ORoot::Constants(c) => Root::Constants(c.clone()),
                ORoot::Table(v) => {
                    let mut t: Vec<(u32, Slot)> = v.iter().map(|(k, s)| (*k, slot(s))).collect();
                    // Slot 86 holds its place; [`record`] views our words buffer there.
                    if r == 0 {
                        t.push((WORDS_SLOT, Slot::Buffer(View { kind: Kind::Raw, bytes: &WORDS_PLACE })));
                        t.sort_by_key(|s| s.0);
                    }
                    Root::Table(t)
                }
                ORoot::View(b) => Root::View(b),
            })
            .collect();
        Draw { roots, heap: o.heap.iter().map(|&(k, h)| (k, Slot::Handle(h))).collect(), samplers: o.samplers.to_vec(), sampler_slots, groups: o.groups }
    }
}

static WORDS_PLACE: [u8; 4 * WORDS] = [0; 4 * WORDS];

/// A translated pair with its pipelines for drawing into our pass.
pub struct Built {
    pub programs: Programs,
    pub used: Used,
    croot: ID3D12RootSignature,
    cpso: ID3D12PipelineState,
    groot: ID3D12RootSignature,
    gpso: ID3D12PipelineState,
    cp: Params,
    gp: Params,
    /// The pair's table layout ([`Lay`]).
    pub lay: Lay,
    /// Milliseconds the translation and the pipelines took.
    pub ms: f64,
}

// SAFETY: D3D12 objects are free-threaded; the programs are plain data.
unsafe impl Send for Built {}
// SAFETY: as above.
unsafe impl Sync for Built {}

/// The pair's pipelines on `dev`: the mesh program's compute form, and the
/// pass-through vertex shader with the pixel program into our colour
/// target (RGBA8; its other targets dropped) and depth (GREATER_EQUAL, no write).
pub fn build(dev: &ID3D12Device, caps: &xnw_gpu::host::Caps, msc: &[u8], psc: &[u8]) -> Result<Built, String> {
    let t0 = std::time::Instant::now();
    let (p, used) = translate(msc, psc)?;
    if let Some(gap) = bridge::gap(caps, &p.needs()) {
        return Err(gap);
    }
    if bridge::targets(&p.ps.dxil).first().copied() != Some((0, DXGI_FORMAT_R32G32B32A32_FLOAT.0)) {
        return Err("the pixel program's target 0 is not a float target".into());
    }
    let eh = emit::host::Host::from_device(dev.clone());
    use emit::mesh::{INDEX_REGISTER, RECORD_REGISTER};
    let croot = eh.root_signature_records(&[(&p.ms, D3D12_SHADER_VISIBILITY_ALL)], &[(true, RECORD_REGISTER), (true, INDEX_REGISTER), (false, RECORD_REGISTER)])?;
    let cpso = eh.compute_pipeline_with(&croot, &p.ms.dxil).map_err(|x| format!("compute pipeline: {x}"))?;
    let vs = HostShader { stage: HwStage::Vs, dxil: p.vs.clone(), user_data: Vec::new(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), address_map: false, descriptor_words: false };
    let gstages = [(&vs, D3D12_SHADER_VISIBILITY_VERTEX), (&p.ps, D3D12_SHADER_VISIBILITY_PIXEL)];
    let groot = eh.root_signature_records(&gstages, &[(false, RECORD_REGISTER)])?;
    let (cp, gp) = (bridge::params(&[&p.ms], 3), bridge::params(&[&vs, &p.ps], 1));
    // SAFETY: a description whose pointers outlive the call.
    let gpso = unsafe {
        let mut g = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&groot),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: p.vs.as_ptr().cast(), BytecodeLength: p.vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: p.ps.dxil.as_ptr().cast(), BytecodeLength: p.ps.dxil.len() },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC { DepthEnable: true.into(), DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO, DepthFunc: D3D12_COMPARISON_FUNC_GREATER_EQUAL, ..Default::default() },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        g.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        g.RTVFormats[0] = bridge::INTO_COLOUR;
        dev.CreateGraphicsPipelineState::<ID3D12PipelineState>(&g).map_err(|x| format!("graphics pipeline: {}", x.message()))?
    };
    let lay = Lay::new(&p, &used);
    Ok(Built { programs: p, used, croot, cpso, groot, gpso, cp, gp, lay, ms: t0.elapsed().as_secs_f64() * 1e3 })
}

enum Entry {
    Building,
    Ready(Arc<Built>),
    Failed,
}

/// The pipelines of the bridged pairs, made once each.
pub struct Cache {
    map: Mutex<HashMap<Key, Entry>>,
}

struct Device(ID3D12Device, xnw_gpu::host::Caps);
// SAFETY: D3D12 devices are free-threaded.
unsafe impl Send for Device {}

/// The process's cache (live and replay).
pub fn cache() -> &'static Cache {
    static C: OnceLock<Cache> = OnceLock::new();
    C.get_or_init(|| Cache { map: Mutex::new(HashMap::new()) })
}

impl Cache {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Key, Entry>> {
        self.map.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The pair's pipelines when made (None: building, failed or never asked).
    pub fn get(&self, key: Key) -> Option<Arc<Built>> {
        match self.lock().get(&key) {
            Some(Entry::Ready(b)) => Some(b.clone()),
            _ => None,
        }
    }

    /// Whether `key` was asked for already (made, building or failed).
    pub fn asked(&self, key: Key) -> bool {
        self.lock().contains_key(&key)
    }

    /// Makes the pair's pipelines once: on a thread of its own (`wait`
    /// false: the live pass never stalls on it) or here.
    pub fn ask(&'static self, key: Key, msc: Vec<u8>, psc: Vec<u8>, dev: &ID3D12Device, caps: &xnw_gpu::host::Caps, wait: bool) {
        {
            let mut m = self.lock();
            if m.contains_key(&key) {
                return;
            }
            m.insert(key, Entry::Building);
        }
        let d = Device(dev.clone(), caps.clone());
        let work = move || {
            let d = d;
            let e = match build(&d.0, &d.1, &msc, &psc) {
                Ok(b) => {
                    xnw_core::info!("bridgeps: pipelines of pixel program {:08x} (mesh {:08x}) made in {:.0} ms", (key.1 >> 96) as u32, (key.0 >> 96) as u32, b.ms);
                    Entry::Ready(Arc::new(b))
                }
                Err(e) => {
                    xnw_core::warn!("bridgeps: pixel program {:08x} (mesh {:08x}): {e}: its rows stay ours", (key.1 >> 96) as u32, (key.0 >> 96) as u32);
                    Entry::Failed
                }
            };
            self.lock().insert(key, e);
        };
        if wait {
            work();
        } else if std::thread::Builder::new().name("xnw-bridgeps".into()).spawn(work).is_err() {
            self.lock().insert(key, Entry::Failed);
        }
    }

    /// `made M, building B, failed F`.
    pub fn counts(&self) -> [usize; 3] {
        let m = self.lock();
        let mut n = [0; 3];
        for e in m.values() {
            n[match e {
                Entry::Ready(_) => 0,
                Entry::Building => 1,
                Entry::Failed => 2,
            }] += 1;
        }
        n
    }
}

/// What one [`Gpu::record`] did.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// The first refusal's reason.
    pub why: String,
    pub drawn: u32,
    /// Rows whose pipelines are not made (yet, or failed).
    pub no_pipeline: u32,
    /// Rows the binding refused, or without a group count.
    pub refused: u32,
    /// Textures from our table, and grey stand-ins.
    pub textures: u32,
    pub grey: u32,
    /// The grey ones by our texture table's reason ([`super::textures::FLAT_REASONS`]).
    pub grey_why: [u32; 7],
    /// Buffers uploaded this frame and their bytes (each distinct buffer once).
    pub buffers: u32,
    pub bytes: u64,
    /// Inputs kept on the device: bytes in use, and bytes copied there this frame.
    pub kept: u64,
    pub kept_new: u64,
    /// `XNW_ZEROCOPY=on`: inputs bound in place in title memory (views,
    /// bytes) instead of copied (ZC1).
    pub in_place: u32,
    pub in_place_bytes: u64,
    /// View heap slots this frame.
    pub slots: u32,
    pub ms: f64,
    /// Milliseconds: bindings, uploads, descriptors, list calls.
    pub parts: [f64; 4],
    /// Descriptor writes by [`WRITES`] kind.
    pub writes: [u32; WRITES.len()],
    /// The descriptors part by [`VIEW_PARTS`], milliseconds.
    pub view_ms: [f64; VIEW_PARTS.len()],
    /// Table blocks ([`tables_on`]): the rows' blocks, distinct this frame,
    /// kept from an earlier frame, made this frame; their slots: all rows',
    /// made; flushes of the block cache.
    pub blocks: [u32; 4],
    pub block_slots: [u32; 2],
    pub flushes: u32,
    /// Blocks by root (-1: the draw words' textures): rows', distinct, kept from an earlier frame.
    pub by_root: Vec<(i32, [u32; 3])>,
    /// Table blocks taken from an earlier row of the frame without cutting.
    pub uncut: u32,
    /// [`small_on`]: small inputs placed, made this frame, bytes made.
    pub small: [u32; 2],
    pub small_bytes: u64,
}

/// [`Stats::writes`]: typed nulls copied (slots), guards, null images,
/// raw, structured, typed, constant buffer views, the words buffer view,
/// textures, grey stand-ins.
pub const WRITES: [&str; 10] = ["nulls", "guards", "null images", "raw", "structured", "typed", "cbv", "words", "textures", "grey"];
/// [`Stats::view_ms`]: typed nulls, guards and null images, buffer views,
/// words and textures, samplers, blocks looked up in the cache, rows cut
/// into blocks with their keys, our texture table's lookups.
pub const VIEW_PARTS: [&str; 8] = ["nulls", "guards", "buffers", "textures", "samplers", "lookups", "cut", "texture lookups"];

/// `XNW_BRIDGE_TABLES` (default on; `off`: every row writes its whole
/// view block every frame, as did): each table root of a row, and its
/// draw-word textures, is one block of the view heap keyed by the exact
/// descriptors it holds; a block already in the heap (this frame's or an
/// earlier frame's) is bound where it is, with no write (after
/// victory-fork `State::tables`).
pub fn tables_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("XNW_BRIDGE_TABLES").as_deref(), Ok("off" | "0")))
}

/// What a [`Gpu::record`] needs of our pass.
pub struct Target<'a> {
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub width: u32,
    pub height: u32,
    /// A texture by handle: its host copy and view this frame.
    pub texture: &'a dyn Fn(u64) -> Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)>,
    /// Why our table has no copy of a texture this frame (a [`super::textures::Flat`]).
    pub flat: &'a dyn Fn(u64) -> usize,
    /// A 1x1 texture in a shader-resource state (the grey stand-in).
    pub grey: &'a ID3D12Resource,
    /// The words buffer (upload memory), its bytes and its mapping: each
    /// drawn row's words are written there at the row's offset.
    pub words: (&'a ID3D12Resource, u64, *mut u8),
}

fn transition(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            // SAFETY: a borrowed pointer the barrier never releases.
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: unsafe { std::mem::transmute_copy(r) }, Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: before, StateAfter: after }),
        },
    }
}

/// Sampler heap slots of a row at least (the title's indices reach 105 in a captured frame).
const SAMPLER_SLOTS: u32 = 128;
/// View heap slots: least, and most (resource binding tier 2 and up).
const HEAP_LEAST: u32 = 1 << 16;
const HEAP_MOST: u32 = 1_000_000;
/// Stride table entries: every slot up to [`NO_VIEW`].
const STRIDE_ENTRIES: u64 = NO_VIEW as u64 + 1;
/// Sampler heaps kept by content at most.
const SAMPLER_HEAPS: usize = 64;
/// The state the kept buffer is read in.
const KEPT_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0);

/// FxHash's multiply-rotate step, eight bytes at a time: the keys below
/// are compared whole on a hit, so the hash need only spread them
/// (SipHash over them took 3-8 ms a frame).
#[derive(Default)]
struct Fx(u64);

impl std::hash::Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut c = bytes.chunks_exact(8);
        for w in &mut c {
            self.write_u64(u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]));
        }
        for &b in c.remainder() {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

/// A map by whole content key.
type ByContent<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<Fx>>;

/// A hasher that takes a key hashed once already.
#[derive(Default)]
struct Hashed(u64);

impl std::hash::Hasher for Hashed {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}


/// The table blocks in the view heap by key: heads by hash, entries
/// chained, key words in one list; no allocation a block.
#[derive(Default)]
struct BlockCache {
    head: HashMap<u64, u32, std::hash::BuildHasherDefault<Hashed>>,
    /// Key words, (first slot, slots, last frame), next entry of the hash (`u32::MAX`: none).
    ents: Vec<(std::ops::Range<usize>, (u32, u32, u64), u32)>,
    words: Vec<u64>,
}

impl BlockCache {
    fn find(&self, h: u64, k: &[u64]) -> Option<usize> {
        let mut i = *self.head.get(&h)?;
        while i != u32::MAX {
            let e = &self.ents[i as usize];
            if self.words[e.0.clone()] == *k {
                return Some(i as usize);
            }
            i = e.2;
        }
        None
    }

    fn get_mut(&mut self, h: u64, k: &[u64]) -> Option<&mut (u32, u32, u64)> {
        let i = self.find(h, k)?;
        Some(&mut self.ents[i].1)
    }

    fn insert(&mut self, h: u64, k: &[u64], v: (u32, u32, u64)) {
        let w = self.words.len();
        self.words.extend_from_slice(k);
        let next = self.head.insert(h, self.ents.len() as u32).unwrap_or(u32::MAX);
        self.ents.push((w..self.words.len(), v, next));
    }

    fn is_empty(&self) -> bool {
        self.ents.is_empty()
    }

    fn clear(&mut self) {
        self.head.clear();
        self.ents.clear();
        self.words.clear();
    }
}

/// A sampler entry list's content key.
fn sampler_key(s: &[(u32, SamplerDesc)], slots: u32) -> Vec<u32> {
    let mut w: Vec<u32> = vec![slots];
    for (k, d) in s {
        w.extend([*k, d.Filter.0 as u32, d.AddressU.0 as u32, d.AddressV.0 as u32, d.AddressW.0 as u32, d.MipLODBias.to_bits(), d.MaxAnisotropy, d.ComparisonFunc.0 as u32]);
        w.extend(d.BorderColor.map(f32::to_bits));
        w.extend([d.MinLOD.to_bits(), d.MaxLOD.to_bits()]);
    }
    w
}

/// `XNW_BRIDGE_SMALL` (default on with [`tables_on`]; `off`: in the
/// frame's upload memory by order, as): inputs under [`KEEP_MIN`] in
/// [`Small`].
pub fn small_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| tables_on() && !matches!(std::env::var("XNW_BRIDGE_SMALL").as_deref(), Ok("off" | "0")))
}

/// Inputs under [`KEEP_MIN`] (constant buffers and the like) by content in
/// one upload buffer kept across frames: equal bytes lie at one
/// offset on every frame that uses them, so the table blocks naming them
/// stay equal across frames (in the frame's upload memory their offsets
/// moved with the rows' order: pk8a f1030 -> f1031 kept 0 of 164 mesh-table
/// blocks, all 164 equal but for where their inputs lay). Bytes are written
/// once; an entry no row used for [`SMALL_FRAMES`] frames is freed at a
/// frame's start (the pass waited for the last frame's fence), its place
/// reused by size and alignment. Pure bookkeeping: the caller writes.
#[derive(Default)]
pub struct Small {
    pub size: u64,
    pub end: u64,
    /// Entries by content hash: bytes, size, unit; offset, capacity, alignment, last frame used.
    at: HashMap<u64, Vec<(Box<[u8]>, u64, u64, [u64; 4])>, std::hash::BuildHasherDefault<Hashed>>,
    free: HashMap<(u64, u64), Vec<u64>>,
    frame: u64,
    /// This frame: entries placed, made, bytes made.
    pub placed: u32,
    pub made: u32,
    pub made_bytes: u64,
}

impl Small {
    /// A new frame: entries unused for [`SMALL_FRAMES`] frames are freed.
    pub fn next_frame(&mut self) {
        self.frame += 1;
        let (f, free) = (self.frame, &mut self.free);
        self.at.retain(|_, v| {
            v.retain(|e| {
                let keep = e.3[3] + SMALL_FRAMES >= f;
                if !keep {
                    free.entry((e.3[1], e.3[2])).or_default().push(e.3[0]);
                }
                keep
            });
            !v.is_empty()
        });
        (self.placed, self.made, self.made_bytes) = (0, 0, 0);
    }

    /// Starts again empty with `size` bytes (the caller makes the new buffer).
    pub fn start(&mut self, size: u64) {
        (self.size, self.end) = (size, 0);
        self.at.clear();
        self.free.clear();
    }

    /// Where `bytes` (`size` padded, element size `unit`) lie, and whether
    /// the caller writes them there now (`true`: their first frame there);
    /// None when the buffer has no room (the frame's upload memory then).
    pub fn place(&mut self, bytes: &[u8], size: u64, unit: u64) -> Option<(u64, bool)> {
        let mut h = Fx::default();
        std::hash::Hasher::write(&mut h, bytes);
        std::hash::Hasher::write_u64(&mut h, size);
        std::hash::Hasher::write_u64(&mut h, unit);
        let h = std::hash::Hasher::finish(&h);
        self.placed += 1;
        if let Some(e) = self.at.get_mut(&h).and_then(|v| v.iter_mut().find(|e| e.1 == size && e.2 == unit && *e.0 == *bytes)) {
            e.3[3] = self.frame;
            return Some((e.3[0], false));
        }
        let (cap, al) = (size.max(16).next_power_of_two(), bridge::align(unit));
        let o = match self.free.get_mut(&(cap, al)).and_then(|v| v.pop()) {
            Some(o) => o,
            None => {
                let o = self.end.next_multiple_of(al);
                if o + cap > self.size {
                    self.placed -= 1;
                    return None;
                }
                self.end = o + cap;
                o
            }
        };
        self.at.entry(h).or_default().push((bytes.into(), size, unit, [o, cap, al, self.frame]));
        self.made += 1;
        self.made_bytes += cap;
        Some((o, true))
    }

    /// The capacity of the entry at `o` (its bytes past the input are zeros).
    fn cap(size: u64) -> u64 {
        size.max(16).next_power_of_two()
    }
}

/// Frames a small input stays placed unused (the title alternates some
/// inputs frame to frame: pk7a3 f962 found 393 of f960's mesh-table blocks).
const SMALL_FRAMES: u64 = 4;
/// The small inputs' buffer: least size.
const SMALL_LEAST: u64 = 8 << 20;

/// The bridged rows' device objects, kept by our pass across frames (it
/// waits for each frame's fence before the next is recorded): one
/// shader-visible view heap (each row at its own base:
/// [`bridge::bind_at`]), sampler heaps by content, the inputs handed over
/// again on a later frame in one default-heap buffer ([`Kept`]), one
/// mapped upload buffer for the rest, the mesh programs' records and indices.
#[derive(Default)]
pub struct Gpu {
    views: Option<(ID3D12DescriptorHeap, u32)>,
    /// A CPU heap of typed nulls: a row's slots are filled from it in one copy.
    nulls: Option<(ID3D12DescriptorHeap, u32)>,
    samplers: ByContent<Vec<u32>, (ID3D12DescriptorHeap, u64)>,
    up: Option<(ID3D12Resource, *mut u8, u64)>,
    /// The stride table of the whole view heap (mapped upload memory):
    /// [`STRIDE_ENTRIES`] entries, so that a read at any slot up to
    /// [`NO_VIEW`] stays inside it; the entries last written.
    strides: Option<(ID3D12Resource, *mut u8)>,
    strides_len: usize,
    out: Option<(ID3D12Resource, ID3D12Resource, u64, u64)>,
    /// The kept buffer and whether it is in its read state.
    kept_buf: Option<(ID3D12Resource, bool)>,
    pub kept: Kept,
    frame: u64,
    /// [`tables_on`]: the block heap and its slots, the blocks in it by
    /// key, the next free slot, the resources their descriptors name (held
    /// so that a pointer names one resource while its blocks live), and
    /// the buffers every block may name (upload, kept, words, grey: a
    /// change of any flushes the blocks).
    bheap: Option<(ID3D12DescriptorHeap, u32)>,
    blocks: BlockCache,
    bump: u32,
    /// The rows cut into blocks, lists kept across frames.
    cuts: Cuts,
    /// [`small_on`]: the small inputs and their mapped upload buffer.
    pub small: Small,
    small_buf: Option<(ID3D12Resource, *mut u8)>,
    held: HashMap<usize, windows::core::IUnknown>,
    bases: Vec<windows::core::IUnknown>,
}

/// The block heap's first slots: the guards every row's bounds name
/// (null 2D texture, raw buffer SRV, raw buffer UAV, 2D texture UAV, CBV).
const GUARDS: u32 = 5;

/// One descriptor of a table block: the typed null of the slot, a
/// buffer view, a texture by handle, our words buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Desc {
    Null,
    NullImage,
    Buf(Kind, At, u32),
    Tex(u64),
    Words,
}

/// One slot of a block that is not a typed null with a zero stride entry
/// Its slot in the block, its stride entry (a raw twin by its slot
/// in the block, plus one), its descriptor.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ent {
    j: u32,
    stride: [u32; 2],
    desc: Desc,
}

/// One block of a row: its root (-1: the draw words' textures), its slots,
/// its entries ([`Cuts::ents`]), its key's hash and words ([`Cuts::keys`];
/// None: not keyable, made afresh), its first slot.
struct Block {
    root: i32,
    len: u32,
    ents: std::ops::Range<usize>,
    key: Option<(u64, std::ops::Range<usize>)>,
    first: u32,
    /// The earlier block of the frame it was taken from uncut.
    from: Option<usize>,
}

/// A frame's rows cut into blocks, in flat lists kept across frames (no
/// allocation a row): the blocks, their entries and key words, each
/// row's blocks; the rest is scratch.
#[derive(Default)]
struct Cuts {
    blocks: Vec<Block>,
    ents: Vec<Ent>,
    keys: Vec<u64>,
    rows: Vec<std::ops::Range<usize>>,
    /// The row's slots, block after block (descriptor and stride entry).
    dense: Vec<([u32; 2], Desc)>,
    /// Each block's first slot in `dense`.
    at: Vec<u32>,
    /// Raw twins: (twin's local slot, block, slot in it), by twin.
    twins: Vec<(u32, usize, u32)>,
    /// Regions of the binding: (start, span, block).
    regions: Vec<(u32, u32, usize)>,
}

impl Cuts {
    fn clear(&mut self) {
        self.blocks.clear();
        self.ents.clear();
        self.keys.clear();
        self.rows.clear();
    }

    fn key(&self, blk: &Block) -> Option<(u64, &[u64])> {
        blk.key.as_ref().map(|(h, r)| (*h, &self.keys[r.clone()]))
    }
}

/// A pipeline's table layout, made once with its pipelines: the
/// table roots the two programs read, in order; per root one past the
/// highest slot the programs name, and the slots they read through an
/// image descriptor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lay {
    pub roots: Vec<u8>,
    pub named: Vec<u32>,
    pub images: Vec<Vec<u32>>,
}

impl Lay {
    pub fn new(p: &Programs, used: &Used) -> Lay {
        let roots = table_roots(&p.ms, &p.ps);
        let named = roots.iter().map(|&r| used.keys().filter(|u| u.0 == u16::from(r)).map(|u| u.1 + 1).max().unwrap_or(0)).collect();
        let images = roots
            .iter()
            .map(|&r| {
                let mut v: Vec<u32> = used.iter().filter(|u| *u.1 == NAMED_IMAGE && u.0.0 == u16::from(r)).map(|u| u.0.1).collect();
                v.sort_unstable();
                v
            })
            .collect();
        Lay { roots, named, images }
    }
}

/// A stopwatch of the descriptors part's pieces.
struct Lap(std::time::Instant);

impl Lap {
    fn to(&mut self, ms: &mut f64) {
        let now = std::time::Instant::now();
        *ms += (now - self.0).as_secs_f64() * 1e3;
        self.0 = now;
    }
}

/// A buffer view's [`WRITES`] index.
fn write_kind(kind: Kind) -> usize {
    match kind {
        Kind::Raw | Kind::RawUav { .. } => 3,
        Kind::Structured(_) => 4,
        Kind::Typed { .. } => 5,
        Kind::Cbv => 6,
    }
}

/// A texture by handle as our texture table has it this frame.
type Texs = ByContent<u64, Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)>>;

/// What a block's key resolves: textures as our table has them this frame,
/// the grey stand-in, the in-place ranges.
struct KeyCx<'a> {
    texs: &'a Texs,
    grey: u64,
    zc: &'a [xnw_gpu::shared::SharedRange],
}

/// The key words of one entry: its slot and stride entry, and the exact
/// descriptor (buffers by where they lie, textures by resource and view,
/// the grey stand-in); false when a texture's view is not one this key
/// spells out (only 2D views are), and the block is then made afresh.
fn key_words(e: &Ent, cx: &KeyCx, w: &mut Vec<u64>) -> bool {
    w.push(u64::from(e.j) | u64::from(e.stride[1]) << 32);
    let tag = |t: u64| u64::from(e.stride[0]) | t << 32;
    match e.desc {
        Desc::Null => w.push(tag(0)),
        Desc::NullImage => w.push(tag(1)),
        Desc::Words => w.push(tag(2)),
        Desc::Buf(kind, at, n) => {
            let k = match kind {
                Kind::Raw => 0,
                Kind::Structured(s) => 1 | u64::from(s) << 8,
                Kind::Typed { format, bytes } => 2 | u64::from(format as u32) << 8 | u64::from(bytes) << 40,
                Kind::Cbv => 3,
                Kind::RawUav { len } => 4 | u64::from(len) << 8,
            };
            match at {
                At::Frame(o) => w.extend([tag(3), k, u64::from(n), o]),
                At::Kept(o) => w.extend([tag(4), k, u64::from(n), o]),
                At::Small(o) => w.extend([tag(8), k, u64::from(n), o]),
                At::Shared(i) => {
                    let Some(r) = cx.zc.get(i as usize) else { return false };
                    w.extend([tag(5), k, u64::from(n), windows::core::Interface::as_raw(&r.buffer) as u64, r.offset]);
                }
            }
        }
        Desc::Tex(h) => match cx.texs.get(&h) {
            Some(Some((tr, desc))) => {
                if desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D {
                    return false;
                }
                // SAFETY: a 2D view's description holds its Texture2D member.
                let t2 = unsafe { desc.Anonymous.Texture2D };
                w.extend([tag(6), windows::core::Interface::as_raw(tr) as u64, u64::from(desc.Format.0 as u32) | u64::from(desc.Shader4ComponentMapping) << 32, u64::from(t2.MostDetailedMip) | u64::from(t2.MipLevels) << 32, u64::from(t2.PlaneSlice) | u64::from(t2.ResourceMinLODClamp.to_bits()) << 32]);
            }
            _ => w.extend([tag(7), cx.grey]),
        },
    }
    true
}

/// A table's inputs' identity hash: slots, kinds' places and bytes by pointer, textures by handle.
fn table_hash(t: &[(u32, OSlot)]) -> u64 {
    let mut h = Fx::default();
    for (k, s) in t {
        std::hash::Hasher::write_u64(&mut h, u64::from(*k));
        std::hash::Hasher::write_u64(&mut h, match s {
            OSlot::Buffer(_, b) => b[..].as_ptr() as u64,
            OSlot::Texture(x) => *x,
        });
    }
    std::hash::Hasher::finish(&h)
}

/// Two tables of the same inputs: equal slots, kinds and handles, the same bytes by pointer.
fn same_table(a: &[(u32, OSlot)], b: &[(u32, OSlot)]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.0 == y.0
                && match (&x.1, &y.1) {
                    (OSlot::Buffer(ka, ba), OSlot::Buffer(kb, bb)) => ka == kb && Arc::ptr_eq(ba, bb),
                    (OSlot::Texture(p), OSlot::Texture(q)) => p == q,
                    _ => false,
                }
        })
}

/// The table roots the two programs read, in order.
fn table_roots(ms: &HostShader, ps: &HostShader) -> Vec<u8> {
    let mut t: Vec<u8> = [ms, ps].iter().flat_map(|s| s.user_data.iter().filter(|u| u.root == xnw_shader::RootType::Table).map(|u| u.slot)).collect();
    t.sort_unstable();
    t.dedup();
    t
}

/// A row's binding (made at heap slot `base`) cut into blocks, appended to
/// `out` with their keys: the draw words' textures, then one per table
/// root of `lay` with its raw twins after its slots. A table block reaches
/// at least past every slot the programs name in it (a named slot without
/// a view holds its typed null in the block, never a neighbour's view).
/// Only the slots that are not a typed null with a zero stride entry are
/// listed; a table root with `reuse[k]` takes that earlier block's entries
/// and key without being cut.
fn cut(b: &bridge::Binding, lay: &Lay, views: &[(u32, Kind, At, u32)], base: u32, cx: &KeyCx, reuse: &[Option<usize>], out: &mut Cuts) {
    use xnw_shader::emit::StrideEntry;
    let strides = &b.views.strides;
    let pre = strides.len() as u32;
    let b0 = out.blocks.len();
    out.twins.clear();
    out.regions.clear();
    let twins = strides.iter().filter(|s| s[1] & StrideEntry::RAW_SLOTS != 0).count() as u32;
    let t0 = pre - twins;
    let first = |k: usize| (b.views.constants[usize::from(lay.roots[k])] / 32).saturating_sub(base);
    let wspan = if lay.roots.is_empty() { t0 } else { first(0) };
    if wspan > 0 {
        out.regions.push((0, wspan, out.blocks.len()));
        out.blocks.push(Block { root: -1, len: wspan, ents: 0..0, key: None, first: 0, from: None });
    }
    let tb = out.blocks.len();
    for k in 0..lay.roots.len() {
        let start = first(k);
        let span = if k + 1 < lay.roots.len() { first(k + 1) } else { t0 }.saturating_sub(start);
        out.regions.push((start, span, out.blocks.len()));
        out.blocks.push(Block { root: i32::from(lay.roots[k]), len: span.max(lay.named[k]).max(1), ents: 0..0, key: None, first: 0, from: None });
    }
    let find = |regions: &[(u32, u32, usize)], j: u32| regions.iter().find(|g| j >= g.0 && j < g.0 + g.1).map(|g| (g.2, j - g.0));
    // A block that equals an earlier row's (same pipeline, same table inputs) is that block.
    let reused = |bk: usize| if bk >= tb { reuse.get(bk - tb).copied().flatten() } else { None };
    // Twins go after their owner's slots, in twin order: each block's length
    // first (a block taken uncut has its earlier block's length).
    for &(start, span, bk) in &out.regions {
        if let Some(from) = reused(bk) {
            out.blocks[bk].len = out.blocks[from].len;
            continue;
        }
        for j in start..start + span {
            let raw = strides[j as usize][1] & StrideEntry::RAW_SLOTS;
            if raw != 0 {
                let blk = &mut out.blocks[bk];
                out.twins.push((raw - 1 - base, bk, blk.len));
                blk.len += 1;
            }
        }
    }
    out.twins.sort_unstable_by_key(|x| x.0);
    out.at.clear();
    let mut total = 0u32;
    for bk in b0..out.blocks.len() {
        out.at.push(total);
        if reused(bk).is_none() {
            total += out.blocks[bk].len;
        }
    }
    out.dense.clear();
    out.dense.resize(total as usize, ([0, 0], Desc::Null));
    let at = |bk: usize, j: u32| (out.at[bk - b0] + j) as usize;
    // The regions' stride entries, the owners' twins, named images, views, textures, words: a later one wins.
    for &(start, span, bk) in &out.regions {
        if reused(bk).is_some() {
            continue;
        }
        let a = at(bk, 0);
        for j in 0..span as usize {
            out.dense[a + j].0 = [strides[start as usize + j][0], 0];
        }
    }
    for &(start, span, bk) in &out.regions {
        if reused(bk).is_some() {
            continue;
        }
        for local in 0..span {
            let s = strides[(start + local) as usize];
            let raw = s[1] & StrideEntry::RAW_SLOTS;
            let Some(i) = (raw != 0).then(|| out.twins.binary_search_by_key(&(raw - 1 - base), |x| x.0).ok()).flatten() else { continue };
            let tw = out.twins[i].2;
            out.dense[at(bk, local)].0 = [s[0], tw + 1];
            out.dense[at(bk, tw)].0 = strides.get((raw - 1 - base) as usize).copied().unwrap_or([4, 0]);
        }
    }
    for (k, images) in lay.images.iter().enumerate() {
        if reused(tb + k).is_none() {
            for &j in images {
                out.dense[at(tb + k, j)].1 = Desc::NullImage;
            }
        }
    }
    let place = |out: &Cuts, s: u32| -> Option<usize> {
        let j = s.checked_sub(base)?;
        let (bk, l) = if j >= t0 && j < pre { out.twins.binary_search_by_key(&j, |x| x.0).ok().map(|i| (out.twins[i].1, out.twins[i].2))? } else { find(&out.regions, j)? };
        reused(bk).is_none().then(|| at(bk, l))
    };
    for &(s, kind, where_, n) in views {
        if let Some(i) = place(out, s) {
            out.dense[i].1 = Desc::Buf(kind, where_, n);
        }
    }
    for &(s, h) in &b.handles {
        if let Some(i) = place(out, s) {
            out.dense[i].1 = Desc::Tex(h);
        }
    }
    if let Some(k) = lay.roots.iter().position(|&r| r == 0)
        && reused(tb + k).is_none()
        && WORDS_SLOT < out.blocks[tb + k].len
        && first(k) + WORDS_SLOT < t0
    {
        out.dense[at(tb + k, WORDS_SLOT)].1 = Desc::Words;
    }
    // Each block's entries and key.
    for bk in b0..out.blocks.len() {
        if let Some(from) = reused(bk) {
            let (ents, key) = (out.blocks[from].ents.clone(), out.blocks[from].key.clone());
            let blk = &mut out.blocks[bk];
            (blk.ents, blk.key, blk.from) = (ents, key, Some(from));
            continue;
        }
        let e0 = out.ents.len();
        let (a, len) = (at(bk, 0), out.blocks[bk].len as usize);
        for (j, &(stride, desc)) in out.dense[a..a + len].iter().enumerate() {
            if desc != Desc::Null || stride != [0, 0] {
                out.ents.push(Ent { j: j as u32, stride, desc });
            }
        }
        let k0 = out.keys.len();
        out.keys.push(len as u64);
        let mut ok = true;
        for e in &out.ents[e0..] {
            if !key_words(e, cx, &mut out.keys) {
                ok = false;
                break;
            }
        }
        let key = if ok {
            let mut h = Fx::default();
            for &w in &out.keys[k0..] {
                std::hash::Hasher::write_u64(&mut h, w);
            }
            Some((std::hash::Hasher::finish(&h), k0..out.keys.len()))
        } else {
            out.keys.truncate(k0);
            None
        };
        let blk = &mut out.blocks[bk];
        blk.ents = e0..out.ents.len();
        blk.key = key;
    }
    out.rows.push(b0..out.blocks.len());
}

/// One row as [`Gpu::record`] binds it.
struct Row<'a> {
    i: usize,
    built: Arc<Built>,
    b: bridge::Binding<'a>,
    views: Vec<(u32, Kind, At, u32)>,
    mroots: Vec<(u32, bool, At)>,
    proots: Vec<(u32, bool, At)>,
    rec: u64,
    idx: u64,
    /// The view heap slot of word 0's texture.
    words_base: u32,
    /// The slot the binding was made at.
    base: u32,
}

impl Gpu {
    /// Records every row of `f` whose pipelines are made on the open list
    /// `l` (our pass's, after its own draws: colour and depth in their
    /// target states). Every mesh program first (their records and index
    /// triples in one buffer each), one barrier, then every pixel program.
    /// Objects replaced this frame and the textures go to `keep`, for the
    /// caller to hold until the list's fence. The caller binds its own
    /// heaps again after it if it draws more.
    pub fn record<'f>(&mut self, dev: &ID3D12Device, l: &ID3D12GraphicsCommandList, f: &'f Frame, t: &Target, keep: &mut Vec<windows::core::IUnknown>) -> Result<Stats, String> {
        let t0 = std::time::Instant::now();
        let mut st = Stats::default();
        let tables = tables_on();
        let e = |e: windows::core::Error| e.message().to_string();
        self.frame += 1;
        if let Some(size) = self.kept.restart {
            if let Some((old, _)) = self.kept_buf.take() {
                keep.push(windows::core::Interface::cast(&old).map_err(e)?);
            }
            self.kept.start(size);
        }
        if self.kept.size == 0 {
            self.kept.start(KEPT_LEAST);
        }
        self.kept.next_frame();
        let small = small_on();
        if small {
            // A new buffer when there is none or the last one is three quarters used.
            if self.small_buf.is_none() || self.small.end > self.small.size / 4 * 3 {
                if let Some((old, _)) = self.small_buf.take() {
                    keep.push(windows::core::Interface::cast(&old).map_err(e)?);
                }
                let size = (2 * self.small.size).max(SMALL_LEAST).min(1 << 30);
                // SAFETY: a description that outlives the call; the buffer is mapped once and lives in `self`.
                unsafe {
                    let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
                    let mut r: Option<ID3D12Resource> = None;
                    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut r).map_err(e)?;
                    let r = r.ok_or("no small inputs' buffer")?;
                    let mut to = std::ptr::null_mut();
                    r.Map(0, None, Some(&mut to)).map_err(e)?;
                    self.small_buf = Some((r, to.cast::<u8>()));
                }
                self.small.start(size);
            }
            self.small.next_frame();
        }
        let small_to = self.small_buf.as_ref().map_or(std::ptr::null_mut(), |b| b.1);
        // The frame's keepable inputs by their bytes' address.
        let mut arcs: HashMap<usize, &Arc<Vec<u8>>> = HashMap::new();
        for (_, o) in &f.rows {
            for r in &o.roots {
                match r {
                    ORoot::Table(v) => {
                        for (_, s) in v {
                            if let OSlot::Buffer(_, b) = s
                                && b.len() >= KEEP_MIN
                            {
                                arcs.insert(b[..].as_ptr() as usize, b);
                            }
                        }
                    }
                    ORoot::View(b) if b.len() >= KEEP_MIN => {
                        arcs.insert(b[..].as_ptr() as usize, b);
                    }
                    _ => {}
                }
            }
        }
        // The bindings, each row at its base in the view heap, and where every input and output goes.
        let mut rows: Vec<Row> = Vec::new();
        let mut arena = Arena::default();
        let mut copies: Vec<(usize, usize, u64, u64)> = Vec::new();
        let (mut rec_end, mut idx_end) = (0u64, 0u64);
        let (mut next, mut most) = (0u32, 0u32);
        let mut strides: Vec<[u32; 2]> = Vec::new();
        // Words offset -> (its rows' word base, the words, the textures they name).
        let mut words_at: HashMap<u32, (u32, [u32; WORDS], &[(u32, u64)])> = HashMap::new();
        // With blocks: inputs shorter than [`KEEP_MIN`] (constant buffers and
        // the like) by content, so that rows handing over equal bytes share
        // one place in the frame's upload memory, one view, one block.
        let mut canon: ByContent<&'f [u8], &'f [u8]> = ByContent::default();
        // `XNW_ZEROCOPY=on` (ZC1): inputs bound in place in title memory.
        // Constant buffers stay copies: every lane reads them, and over the
        // bus in-place reads that depend on each other ran ~30x slower
        // (an idea from upstream; a copy path stays for hot data).
        let zc_on = xnw_gpu::shared::world_part(xnw_gpu::shared::WORLD_BRIDGE);
        let mut zc: Vec<xnw_gpu::shared::SharedRange> = Vec::new();
        let mut zc_bytes = 0u64;
        for i in 0..f.rows.len() {
            let o = &f.rows[i].1;
            let Some(built) = cache().get(o.key) else {
                st.no_pipeline += 1;
                continue;
            };
            let refuse = |st: &mut Stats, why: String| {
                st.refused += 1;
                if st.why.is_empty() {
                    st.why = format!("row {}: {why}", f.rows[i].0);
                }
            };
            // A texture read as a buffer, or a buffer as an image: the RX 5700 XT
            // reads a wild address (victory-fork 3c81420). Not drawn.
            if o.lacks & (1 << 7 | 1 << 8) != 0 {
                refuse(&mut st, "a view read through a descriptor of the other kind".into());
                continue;
            }
            if u64::from(o.words_at) + 4 * WORDS as u64 > t.words.1 {
                refuse(&mut st, format!("its words at {:#x} lie past our words buffer ({} B)", o.words_at, t.words.1));
                continue;
            }
            // Rows of one words offset share its textures: the first row's base names them.
            let words_base = match words_at.get(&o.words_at) {
                Some(w) if w.1 == o.words && w.2 == &o.heap[..] => w.0,
                Some(_) => {
                    refuse(&mut st, format!("its words at {:#x} differ from an earlier row's", o.words_at));
                    continue;
                }
                None => next,
            };
            let p = &built.programs;
            let b = match bridge::bind_at(&p.ms, &p.ps, &f.draw(i, SAMPLER_SLOTS), next) {
                Ok(b) if b.groups > 0 => b,
                r => {
                    refuse(&mut st, r.err().unwrap_or_else(|| "no group count".into()));
                    continue;
                }
            };
            if next + b.views.heap_slots > HEAP_MOST {
                refuse(&mut st, format!("the view heap is full ({next} slots)"));
                continue;
            }
            let kept = &mut self.kept;
            let addrs = &o.addrs;
            // `align`: the offset an in-place view of these bytes needs.
            let canon = &mut canon;
            let small_in = &mut self.small;
            let mut place = |a: &mut Arena, c: &mut Vec<(usize, usize, u64, u64)>, bytes: &'f [u8], size: u64, unit: u64, align: u64, cbv: bool| -> At {
                if zc_on
                    && !cbv
                    && let Some(&(_, title)) = addrs.iter().find(|x| x.0 == bytes.as_ptr() as usize)
                    && let Some(r) = xnw_gpu::shared::world_range(title, size)
                    && r.offset.is_multiple_of(align)
                {
                    zc_bytes += size;
                    zc.push(r);
                    return At::Shared(zc.len() as u32 - 1);
                }
                if small
                    && bytes.len() < KEEP_MIN
                    && let Some((o, fresh)) = small_in.place(bytes, size, unit)
                {
                    if fresh {
                        let n = bytes.len().min(size as usize);
                        // SAFETY: `o` and its capacity lie inside the mapped buffer
                        // (`Small::place`); no list in flight reads it (the pass
                        // waited for the last frame's fence; the entry is new).
                        unsafe {
                            std::ptr::copy_nonoverlapping(bytes.as_ptr(), small_to.add(o as usize), n);
                            std::ptr::write_bytes(small_to.add(o as usize + n), 0, Small::cap(size) as usize - n);
                        }
                    }
                    return At::Small(o);
                }
                let bytes = if tables && bytes.len() < KEEP_MIN { *canon.entry(bytes).or_insert(bytes) } else { bytes };
                let (at, fresh) = kept.place(a, bytes, size, unit, arcs.get(&(bytes.as_ptr() as usize)).copied());
                if let (At::Kept(o), true) = (at, fresh) {
                    c.push((bytes.as_ptr() as usize, bytes.len().min(size as usize), size, o));
                }
                at
            };
            let mut views = Vec::new();
            for &(s, v) in &b.views.slots {
                if let Kind::RawUav { .. } = v.kind {
                    continue;
                }
                if let Some((size, n)) = v.kind.extent(v.bytes.len() as u64) {
                    views.push((s, v.kind, place(&mut arena, &mut copies, v.bytes, size, unit(v.kind), unit(v.kind), v.kind == Kind::Cbv), n));
                }
            }
            let mut roots = |vs: &[(u8, bool, &'f [u8])], params: &[(u8, bool, u32)]| -> Result<Vec<(u32, bool, At)>, String> {
                params
                    .iter()
                    .map(|&(r, is_srv, param)| {
                        let v = vs.iter().find(|v| v.0 == r && v.1 == is_srv).ok_or_else(|| format!("root view {r} not bound"))?;
                        Ok((param, is_srv, place(&mut arena, &mut copies, v.2, (v.2.len() as u64).next_multiple_of(256), 256, 4, !is_srv)))
                    })
                    .collect()
            };
            let (mroots, proots) = match (roots(&b.mesh.views, &built.cp.views), roots(&b.pixel.views, &built.gp.views)) {
                (Ok(m), Ok(p)) => (m, p),
                (Err(x), _) | (_, Err(x)) => {
                    refuse(&mut st, x);
                    continue;
                }
            };
            words_at.entry(o.words_at).or_insert((words_base, o.words, &o.heap[..]));
            // The row's stride entries at its slots in the frame's one table.
            let base = next as usize;
            if !tables {
                strides.resize(base + b.views.heap_slots as usize, [0, 0]);
                strides[base..base + b.views.strides.len()].copy_from_slice(&b.views.strides);
            }
            next += b.views.heap_slots;
            most = most.max(b.views.heap_slots);
            let lay = &p.layout;
            let groups = u64::from(b.groups);
            let (rec, idx) = (rec_end, idx_end);
            rec_end += (groups * u64::from(lay.record_bytes())).next_multiple_of(256);
            idx_end += (groups * u64::from(lay.index_bytes())).next_multiple_of(256);
            rows.push(Row { i, built: built.clone(), b, views, mroots, proots, rec, idx, words_base, base: base as u32 });
        }
        st.slots = next;
        (st.small, st.small_bytes) = ([self.small.placed, self.small.made], self.small.made_bytes);
        let base_at = arena.place(&[0; 16], 16, 16);
        let stride_bytes: Vec<u8> = strides.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        // The kept inputs' copies go through the upload memory too.
        let staged: Vec<u64> = copies
            .iter()
            .map(|&(p, n, size, _)| {
                // SAFETY: `p` and `n` name bytes of an input of `f` (alive here).
                let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, n) };
                arena.place(bytes, size, 16)
            })
            .collect();
        st.parts[0] = t0.elapsed().as_secs_f64() * 1e3;
        if rows.is_empty() {
            st.ms = st.parts[0];
            return Ok(st);
        }
        let t1 = std::time::Instant::now();
        // SAFETY: D3D12 calls with descriptions that outlive them; the
        // arena's parts point into `f` (the rows' bytes) and this
        // function's own tables, alive here; our objects live in `self`
        // (the caller waits for the frame's fence before the next record),
        // and those replaced go to `keep`.
        unsafe {
            let committed = |size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES| -> Result<ID3D12Resource, String> {
                let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size.max(256), Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: flags, ..Default::default() };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).map_err(e)?;
                r.ok_or_else(|| "no resource".to_string())
            };
            // The upload buffer: mapped once, written each frame.
            if self.up.as_ref().is_none_or(|u| u.2 < arena.end) {
                if let Some((old, _, _)) = self.up.take() {
                    old.Unmap(0, None);
                    keep.push(windows::core::Interface::cast(&old).map_err(e)?);
                }
                let size = arena.end.next_power_of_two().max(16 << 20);
                let r = committed(size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
                let mut to = std::ptr::null_mut();
                r.Map(0, None, Some(&mut to)).map_err(e)?;
                self.up = Some((r, to.cast::<u8>(), size));
            }
            let (up, to, _) = self.up.clone().ok_or("no upload buffer")?;
            for &(o, ptr, n, size) in &arena.parts {
                std::ptr::copy_nonoverlapping(ptr as *const u8, to.add(o as usize), n);
                std::ptr::write_bytes(to.add(o as usize + n), 0, size.max(16) as usize - n);
            }
            // The stride table: this frame's entries, zeros past them.
            if self.strides.is_none() {
                let r = committed(STRIDE_ENTRIES * 8, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
                let mut to = std::ptr::null_mut();
                r.Map(0, None, Some(&mut to)).map_err(e)?;
                std::ptr::write_bytes(to.cast::<u8>(), 0, (STRIDE_ENTRIES * 8) as usize);
                self.strides = Some((r, to.cast::<u8>()));
                self.strides_len = 0;
            }
            let (stable, sp) = self.strides.clone().ok_or("no stride table")?;
            // Blocks write their own entries when they are made (below).
            if !tables {
                std::ptr::copy_nonoverlapping(stride_bytes.as_ptr(), sp, stride_bytes.len());
                if self.strides_len > stride_bytes.len() {
                    std::ptr::write_bytes(sp.add(stride_bytes.len()), 0, self.strides_len - stride_bytes.len());
                }
                self.strides_len = stride_bytes.len();
            }
            let sva = stable.GetGPUVirtualAddress();
            st.buffers = arena.parts.len() as u32;
            st.bytes = arena.end;
            st.kept = self.kept.used;
            st.kept_new = copies.iter().map(|c| c.2).sum();
            st.in_place = zc.len() as u32;
            st.in_place_bytes = zc_bytes;
            let uva = up.GetGPUVirtualAddress();
            // The mesh programs' outputs.
            let have = self.out.as_ref().map_or((0, 0), |o| (o.2, o.3));
            if have.0 < rec_end || have.1 < idx_end {
                if let Some((r, i, _, _)) = self.out.take() {
                    keep.push(windows::core::Interface::cast(&r).map_err(e)?);
                    keep.push(windows::core::Interface::cast(&i).map_err(e)?);
                }
                let (rn, inn) = (rec_end.next_power_of_two().max(have.0).max(1 << 20), idx_end.next_power_of_two().max(have.1).max(1 << 20));
                // Buffers start in (and decay to) COMMON; the first dispatch
                // promotes them to UAV (the layer ignores another initial state, id 1328).
                let common = D3D12_RESOURCE_STATE_COMMON;
                self.out = Some((committed(rn, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, common)?, committed(inn, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, common)?, rn, inn));
            }
            let (records, indices, _, _) = self.out.clone().ok_or("no outputs")?;
            let (rva, iva) = (records.GetGPUVirtualAddress(), indices.GetGPUVirtualAddress());
            // The kept buffer, and this frame's copies into it.
            if self.kept_buf.is_none() {
                self.kept_buf = Some((committed(self.kept.size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?, false));
            }
            let (kept, read) = self.kept_buf.clone().ok_or("no kept buffer")?;
            let before = if read { KEPT_READ } else { D3D12_RESOURCE_STATE_COMMON };
            if copies.is_empty() {
                if !read {
                    l.ResourceBarrier(&[transition(&kept, before, KEPT_READ)]);
                }
            } else {
                l.ResourceBarrier(&[transition(&kept, before, D3D12_RESOURCE_STATE_COPY_DEST)]);
                for (c, from) in copies.iter().zip(&staged) {
                    l.CopyBufferRegion(&kept, c.3, &up, *from, c.2);
                }
                l.ResourceBarrier(&[transition(&kept, D3D12_RESOURCE_STATE_COPY_DEST, KEPT_READ)]);
            }
            self.kept_buf = Some((kept.clone(), true));
            let kva = kept.GetGPUVirtualAddress();
            let sva_small = self.small_buf.as_ref().map_or(0, |b| b.0.GetGPUVirtualAddress());
            let addr = |a: At| match a {
                At::Small(o) => sva_small + o,
                At::Frame(o) => uva + o,
                At::Kept(o) => kva + o,
                At::Shared(i) => zc[i as usize].buffer.GetGPUVirtualAddress() + zc[i as usize].offset,
            };
            // The drawn rows' words, each naming its textures at the row's base
            // (with blocks: once their places are known, below).
            let words = std::slice::from_raw_parts_mut(t.words.2, t.words.1 as usize);
            for r in rows.iter().filter(|_| !tables) {
                let o = &f.rows[r.i].1;
                if let Some(d) = words.get_mut(o.words_at as usize..o.words_at as usize + 4 * WORDS) {
                    for (k, w) in o.words.iter().enumerate() {
                        let w = if o.heap.iter().any(|h| h.0 == k as u32) { w + r.words_base } else { *w };
                        d[4 * k..4 * k + 4].copy_from_slice(&w.to_le_bytes());
                    }
                }
            }
            st.parts[1] = t1.elapsed().as_secs_f64() * 1e3;
            let t2 = std::time::Instant::now();
            let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let sstep = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
            let srv = |format, first: u64, n: u32, stride: u32, flags| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_BUFFER, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: first, NumElements: n, StructureByteStride: stride, Flags: flags } } };
            let tex2d = |format| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } } };
            let raw_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R32_TYPELESS, ViewDimension: D3D12_UAV_DIMENSION_BUFFER, Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } } };
            let tex_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() };
            let mut lap = Lap(std::time::Instant::now());
            // With blocks: textures by handle, once a frame (held to the
            // fence), then each row cut into its blocks with their keys
            //
            let mut texs = Texs::default();
            let mut cuts = std::mem::take(&mut self.cuts);
            cuts.clear();
            if tables {
                for r in &rows {
                    for &(_, h) in &r.b.handles {
                        texs.entry(h).or_insert_with(|| (t.texture)(h));
                    }
                }
                for (tr, _) in texs.values().flatten() {
                    keep.push(windows::core::Interface::cast(tr).map_err(e)?);
                }
                lap.to(&mut st.view_ms[7]);
                let cx = KeyCx { texs: &texs, grey: windows::core::Interface::as_raw(t.grey) as u64, zc: &zc };
                // A table whose inputs are an earlier row's (same pipeline and
                // root, the same bytes by pointer) takes that row's block uncut:
                // the frame table of most rows.
                let mut memo: ByContent<(usize, u8, u64), Vec<(&'f [(u32, OSlot)], usize)>> = ByContent::default();
                let mut reuse: Vec<Option<usize>> = Vec::new();
                let mut look: Vec<(usize, u64, &'f [(u32, OSlot)])> = Vec::new();
                for r in &rows {
                    let o = &f.rows[r.i].1;
                    let (lay, id) = (&r.built.lay, Arc::as_ptr(&r.built) as usize);
                    reuse.clear();
                    look.clear();
                    for (k, &root) in lay.roots.iter().enumerate() {
                        let mut from = None;
                        if let Some(ORoot::Table(t)) = o.roots.get(usize::from(root)) {
                            let h = table_hash(t);
                            from = memo.get(&(id, root, h)).and_then(|v| v.iter().find(|x| same_table(x.0, t)).map(|x| x.1));
                            look.push((k, h, &t[..]));
                        }
                        reuse.push(from);
                    }
                    cut(&r.b, lay, &r.views, r.base, &cx, &reuse, &mut cuts);
                    let tb = cuts.blocks.len() - lay.roots.len();
                    for &(k, h, t) in &look {
                        if reuse[k].is_none() {
                            memo.entry((id, lay.roots[k], h)).or_default().push((t, tb + k));
                        } else {
                            st.uncut += 1;
                        }
                    }
                }
                lap.to(&mut st.view_ms[6]);
            }
            let longest = cuts.blocks.iter().map(|b| b.len).max().unwrap_or(0);
            // The nulls' template: every row's (or block's) slots in one copy.
            let want_nulls = if tables { most.max(longest).max(GUARDS) } else { most };
            if self.nulls.as_ref().is_none_or(|n| n.1 < want_nulls) {
                let n = want_nulls.next_power_of_two().max(256);
                let h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: n, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, ..Default::default() }).map_err(e)?;
                let start = h.GetCPUDescriptorHandleForHeapStart().ptr;
                for s in 0..n as usize {
                    dev.CreateShaderResourceView(None, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, D3D12_BUFFER_SRV_FLAG_RAW)), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start + s * step });
                }
                self.nulls = Some((h, n));
            }
            let (nh, _) = self.nulls.clone().ok_or("no nulls")?;
            let nstart = nh.GetCPUDescriptorHandleForHeapStart();
            lap.to(&mut st.view_ms[0]);
            let vh: ID3D12DescriptorHeap = if tables {
                // The buffers blocks name: another one flushes them.
                let mut ids: Vec<windows::core::IUnknown> = vec![windows::core::Interface::cast(&up).map_err(e)?, windows::core::Interface::cast(&kept).map_err(e)?, windows::core::Interface::cast(t.words.0).map_err(e)?, windows::core::Interface::cast(t.grey).map_err(e)?];
                if let Some((b, _)) = &self.small_buf {
                    ids.push(windows::core::Interface::cast(b).map_err(e)?);
                }
                let mut flush = self.bases.len() != ids.len() || self.bases.iter().zip(&ids).any(|(a, b)| a != b);
                if flush {
                    self.bases = ids;
                }
                // Slots the frame makes: blocks neither in `have` nor made earlier this frame.
                // Distinct blocks by their key's hash (blocks taken uncut are
                // their earlier block): a slot count that two keys of one hash
                // would short, so a block is made only below the heap's end
                // (checked again where it is made).
                let need = |cuts: &Cuts, have: Option<&BlockCache>| -> u64 {
                    let mut seen: std::collections::HashSet<u64, std::hash::BuildHasherDefault<Hashed>> = Default::default();
                    let mut n = 0u64;
                    for blk in cuts.blocks.iter().filter(|b| b.from.is_none()) {
                        if let Some((h, k)) = cuts.key(blk) {
                            if have.is_some_and(|m| m.find(h, k).is_some()) || !seen.insert(h) {
                                continue;
                            }
                        }
                        n += u64::from(blk.len);
                    }
                    n
                };
                let cap = self.bheap.as_ref().map_or(0, |h| h.1);
                let mut fresh = need(&cuts, (!flush).then_some(&self.blocks));
                if flush || u64::from(self.bump) + fresh > u64::from(cap) {
                    flush = true;
                    fresh = need(&cuts, None);
                }
                lap.to(&mut st.view_ms[5]);
                // A new heap's time goes with the nulls (the earlier split).
                let mut whole = false;
                if u64::from(GUARDS) + fresh > u64::from(cap) {
                    if u64::from(GUARDS) + fresh > u64::from(HEAP_MOST) {
                        self.cuts = cuts;
                        return Err(format!("the view heap cannot hold the frame's {fresh} block slots"));
                    }
                    if let Some((old, _)) = self.bheap.take() {
                        keep.push(windows::core::Interface::cast(&old).map_err(e)?);
                    }
                    let n = (4 * (u64::from(GUARDS) + fresh)).next_power_of_two().clamp(u64::from(HEAP_LEAST), u64::from(HEAP_MOST)) as u32;
                    let h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: n, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).map_err(e)?;
                    self.bheap = Some((h, n));
                    (whole, flush) = (true, true);
                }
                let (vh, cap_now) = self.bheap.clone().ok_or("no view heap")?;
                let vstart = vh.GetCPUDescriptorHandleForHeapStart().ptr;
                let at = |slot: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: vstart + slot as usize * step };
                if flush {
                    // Only the guards: every slot below a row's bound lies in a
                    // block made (typed nulls first) since the last flush, so
                    // the slots past the next free one are never read; their
                    // stride entries go back to zeros.
                    if whole {
                        dev.CopyDescriptorsSimple(GUARDS, at(0), nstart, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                        dev.CreateShaderResourceView(None, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(0));
                        dev.CreateUnorderedAccessView(None, None, Some(&raw_uav), at(2));
                        dev.CreateUnorderedAccessView(None, None, Some(&tex_uav), at(3));
                        dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at(4));
                        st.writes[1] += 4;
                    }
                    let end = self.bump.max(self.strides_len as u32 / 8).max(GUARDS) as usize;
                    std::ptr::write_bytes(sp.add(GUARDS as usize * 8), 0, (end - GUARDS as usize) * 8);
                    self.strides_len = 0;
                    if !self.blocks.is_empty() {
                        st.flushes += 1;
                    }
                    self.blocks.clear();
                    // The resources the old blocks named go with this frame's list
                    // (released after its fence), never before (VIctory: a
                    // texture's release must not take a slot a recorded list reads).
                    keep.extend(self.held.drain().map(|x| x.1));
                    self.bump = GUARDS;
                }
                lap.to(&mut st.view_ms[0]);
                // Each block bound where it is, or made at the next free slots.
                let mut words_first: ByContent<u32, u32> = ByContent::default();
                // By root + 1 (0: the words' textures).
                let mut by_root = [[0u32; 3]; 257];
                for (r, rb) in rows.iter().zip(cuts.rows.clone()) {
                    let o = &f.rows[r.i].1;
                    for bi in rb {
                        let blk = &cuts.blocks[bi];
                        let len = blk.len;
                        st.blocks[0] += 1;
                        st.block_slots[0] += len;
                        let rc = &mut by_root[(blk.root + 1) as usize];
                        rc[0] += 1;
                        // Rows of one words offset share the first one's textures.
                        if blk.root < 0
                            && let Some(&w) = words_first.get(&o.words_at)
                        {
                            cuts.blocks[bi].first = w;
                            continue;
                        }
                        let key = cuts.key(blk);
                        let first = if let Some(from) = blk.from {
                            cuts.blocks[from].first
                        } else if let Some((h, k)) = key
                            && let Some(x) = self.blocks.get_mut(h, k)
                        {
                            // First use this frame: one distinct block.
                            if x.2 != self.frame {
                                st.blocks[1] += 1;
                                rc[1] += 1;
                                if x.2 < self.frame {
                                    st.blocks[2] += 1;
                                    rc[2] += 1;
                                }
                            }
                            x.2 = self.frame;
                            x.0
                        } else {
                            let first = self.bump;
                            if first + len > cap_now {
                                self.cuts = cuts;
                                return Err(format!("the view heap's {cap_now} slots cannot hold the frame's blocks"));
                            }
                            self.bump += len;
                            if let Some((h, k)) = key {
                                self.blocks.insert(h, k, (first, len, self.frame));
                            }
                            st.blocks[1] += 1;
                            st.blocks[3] += 1;
                            st.block_slots[1] += len;
                            rc[1] += 1;
                            lap.to(&mut st.view_ms[5]);
                            dev.CopyDescriptorsSimple(len, at(first), nstart, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                            std::ptr::write_bytes(sp.add(first as usize * 8), 0, len as usize * 8);
                            st.writes[0] += len;
                            lap.to(&mut st.view_ms[0]);
                            for en in &cuts.ents[blk.ents.clone()] {
                                let s = first + en.j;
                                match en.desc {
                                    Desc::Null => {}
                                    Desc::NullImage => {
                                        dev.CreateShaderResourceView(None, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(s));
                                        st.writes[2] += 1;
                                    }
                                    Desc::Buf(kind, where_, n) => {
                                        let (res, off) = match where_ {
                                            At::Frame(o) => (&up, o),
                                            At::Kept(o) => (&kept, o),
                                            At::Small(o) => (&self.small_buf.as_ref().ok_or("no small inputs' buffer")?.0, o),
                                            At::Shared(i) => {
                                                let b = &zc[i as usize].buffer;
                                                self.held.entry(windows::core::Interface::as_raw(b) as usize).or_insert(windows::core::Interface::cast(b).map_err(e)?);
                                                (b, zc[i as usize].offset)
                                            }
                                        };
                                        let first = off / unit(kind);
                                        match kind {
                                            Kind::Raw => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT_R32_TYPELESS, off / 4, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s)),
                                            Kind::Structured(stride) => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT_UNKNOWN, first, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                                            Kind::Typed { format, .. } => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT(format), first, n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                                            Kind::Cbv => dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: addr(where_), SizeInBytes: n }), at(s)),
                                            Kind::RawUav { .. } => {}
                                        }
                                        st.writes[write_kind(kind)] += 1;
                                    }
                                    Desc::Words => {
                                        dev.CreateShaderResourceView(t.words.0, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, (t.words.1 / 4) as u32, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s));
                                        st.writes[7] += 1;
                                    }
                                    Desc::Tex(h) => match texs.get(&h) {
                                        Some(Some((tr, desc))) => {
                                            dev.CreateShaderResourceView(tr, Some(desc), at(s));
                                            self.held.entry(windows::core::Interface::as_raw(tr) as usize).or_insert(windows::core::Interface::cast(tr).map_err(e)?);
                                            st.writes[8] += 1;
                                        }
                                        _ => {
                                            dev.CreateShaderResourceView(t.grey, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(s));
                                            st.writes[9] += 1;
                                        }
                                    },
                                }
                                // Its stride entry (a raw twin by its slot in the heap).
                                if en.stride != [0, 0] {
                                    let words = [en.stride[0], if en.stride[1] != 0 { first + en.stride[1] } else { 0 }];
                                    std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), sp.add(s as usize * 8), 8);
                                }
                            }
                            lap.to(&mut st.view_ms[2]);
                            first
                        };
                        cuts.blocks[bi].first = first;
                        if cuts.blocks[bi].root < 0 {
                            words_first.insert(o.words_at, first);
                        }
                    }
                }
                lap.to(&mut st.view_ms[5]);
                st.by_root = by_root.iter().enumerate().filter(|x| x.1[0] > 0).map(|(r, c)| (r as i32 - 1, *c)).collect();
                st.slots = self.bump;
                // The rows' table offsets, bounds and word base from their blocks.
                for (r, rb) in rows.iter_mut().zip(cuts.rows.clone()) {
                    let plan = &cuts.blocks[rb];
                    let p = &r.built.programs;
                    for (s, c) in [(&p.ms, &mut r.b.mesh.constants), (&p.ps, &mut r.b.pixel.constants)] {
                        for u in s.user_data.iter().filter(|u| u.root == xnw_shader::RootType::Table && u.dword == 0) {
                            if let Some(blk) = plan.iter().find(|b| b.root == i32::from(u.slot))
                                && let Some(w) = c.get_mut(u.index as usize)
                            {
                                *w = blk.first * 32;
                            }
                        }
                    }
                    let end = plan.iter().map(|b| b.first + b.len).max().unwrap_or(0).max(GUARDS);
                    r.b.views.bounds[..6].copy_from_slice(&[end, 0, 1, 2, 3, 4]);
                    r.words_base = plan.iter().find(|b| b.root < 0).map_or(0, |b| b.first);
                }
                let words = std::slice::from_raw_parts_mut(t.words.2, t.words.1 as usize);
                for r in &rows {
                    let o = &f.rows[r.i].1;
                    if let Some(d) = words.get_mut(o.words_at as usize..o.words_at as usize + 4 * WORDS) {
                        for (k, w) in o.words.iter().enumerate() {
                            let w = if o.heap.iter().any(|h| h.0 == k as u32) { w + r.words_base } else { *w };
                            d[4 * k..4 * k + 4].copy_from_slice(&w.to_le_bytes());
                        }
                    }
                }
                lap.to(&mut st.view_ms[3]);
                vh
            } else {
                // The view heap: every row's slots, rewritten each frame.
                if self.views.as_ref().is_none_or(|v| v.1 < next) {
                    if let Some((old, _)) = self.views.take() {
                        keep.push(windows::core::Interface::cast(&old).map_err(e)?);
                    }
                    let n = next.next_power_of_two().clamp(HEAP_LEAST, HEAP_MOST);
                    let h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: n, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).map_err(e)?;
                    self.views = Some((h, n));
                }
                self.views.clone().ok_or("no view heap")?.0
            };
            self.cuts = cuts;
            let vstart = vh.GetCPUDescriptorHandleForHeapStart().ptr;
            let at = |slot: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: vstart + slot as usize * step };
            let mut sheaps: Vec<ID3D12DescriptorHeap> = Vec::with_capacity(rows.len());
            let mut skeys: HashMap<(usize, u32), Vec<u32>> = HashMap::new();
            for r in &rows {
                let (o, b) = (&f.rows[r.i].1, &r.b);
                if tables {
                    for &(_, h) in &b.handles {
                        if let Some(Some(_)) = texs.get(&h) {
                            st.textures += 1;
                        } else {
                            st.grey += 1;
                            if let Some(n) = st.grey_why.get_mut((t.flat)(h)) {
                                *n += 1;
                            }
                        }
                    }
                } else {
                    // Typed nulls at every slot, then the guards, the buffers, the textures.
                    let g = &b.views.bounds;
                    dev.CopyDescriptorsSimple(b.views.heap_slots, at(g[0] - b.views.heap_slots), nstart, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                    st.writes[0] += b.views.heap_slots;
                    lap.to(&mut st.view_ms[0]);
                    dev.CreateShaderResourceView(None, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(g[1]));
                    dev.CreateUnorderedAccessView(None, None, Some(&raw_uav), at(g[3]));
                    dev.CreateUnorderedAccessView(None, None, Some(&tex_uav), at(g[4]));
                    dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at(g[5]));
                    st.writes[1] += 4;
                    // Slots the programs read through an image descriptor hold a null
                    // 2D texture, not the raw buffer null (a texture below overwrites it).
                    for (&(root, slot), _) in r.built.used.iter().filter(|u| *u.1 == NAMED_IMAGE) {
                        if matches!(o.roots.get(usize::from(root)), Some(ORoot::Table(_)))
                            && let Some(c) = b.views.constants.get(usize::from(root))
                            && c / 32 + slot < g[0]
                        {
                            dev.CreateShaderResourceView(None, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(c / 32 + slot));
                            st.writes[2] += 1;
                        }
                    }
                    lap.to(&mut st.view_ms[1]);
                    for &(s, kind, where_, n) in &r.views {
                        let (res, off) = match where_ {
                            At::Frame(o) => (&up, o),
                            At::Kept(o) => (&kept, o),
                            At::Small(o) => (&self.small_buf.as_ref().ok_or("no small inputs' buffer")?.0, o),
                            At::Shared(i) => (&zc[i as usize].buffer, zc[i as usize].offset),
                        };
                        let first = off / unit(kind);
                        match kind {
                            Kind::Raw => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT_R32_TYPELESS, off / 4, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s)),
                            Kind::Structured(stride) => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT_UNKNOWN, first, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                            Kind::Typed { format, .. } => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT(format), first, n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                            Kind::Cbv => dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: addr(where_), SizeInBytes: n }), at(s)),
                            Kind::RawUav { .. } => {}
                        }
                        st.writes[write_kind(kind)] += 1;
                    }
                    lap.to(&mut st.view_ms[2]);
                    // Slot 86 of the frame table: our words buffer, whole.
                    if let Some(s86) = b.views.constants.first().map(|c| c / 32 + WORDS_SLOT).filter(|s| b.views.slots.iter().any(|v| v.0 == *s)) {
                        dev.CreateShaderResourceView(t.words.0, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, (t.words.1 / 4) as u32, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s86));
                        st.writes[7] += 1;
                    }
                    for &(s, h) in &b.handles {
                        match (t.texture)(h) {
                            Some((tr, desc)) => {
                                dev.CreateShaderResourceView(&tr, Some(&desc), at(s));
                                keep.push(windows::core::Interface::cast(&tr).map_err(e)?);
                                st.textures += 1;
                                st.writes[8] += 1;
                            }
                            None => {
                                dev.CreateShaderResourceView(t.grey, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(s));
                                st.grey += 1;
                                st.writes[9] += 1;
                                if let Some(n) = st.grey_why.get_mut((t.flat)(h)) {
                                    *n += 1;
                                }
                            }
                        }
                    }
                }
                lap.to(&mut st.view_ms[3]);
                // Samplers: one heap per entry list, kept by content: one
                // default (linear, wrap) at every slot, the title's entries at their indices.
                let sk = skeys.entry((Arc::as_ptr(&o.samplers) as usize, b.sampler_slots)).or_insert_with(|| sampler_key(&b.samplers, b.sampler_slots));
                let sh = match self.samplers.get_mut(sk) {
                    Some(h) => {
                        h.1 = self.frame;
                        h.0.clone()
                    }
                    None => {
                        let h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, NumDescriptors: b.sampler_slots.max(1), Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).map_err(e)?;
                        let sat = |s: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.GetCPUDescriptorHandleForHeapStart().ptr + s * sstep };
                        let wrap = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                        for s in 0..b.sampler_slots as usize {
                            dev.CreateSampler(&D3D12_SAMPLER_DESC { Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR, AddressU: wrap, AddressV: wrap, AddressW: wrap, MaxLOD: f32::MAX, ..Default::default() }, sat(s));
                        }
                        for (s, desc) in &b.samplers {
                            dev.CreateSampler(desc, sat(*s as usize));
                        }
                        if self.samplers.len() >= SAMPLER_HEAPS
                            && let Some(old) = self.samplers.iter().filter(|x| x.1.1 != self.frame).min_by_key(|x| x.1.1).map(|x| x.0.clone())
                            && let Some((oh, _)) = self.samplers.remove(&old)
                        {
                            keep.push(windows::core::Interface::cast(&oh).map_err(e)?);
                        }
                        self.samplers.insert(sk.clone(), (h.clone(), self.frame));
                        h
                    }
                };
                sheaps.push(sh);
                lap.to(&mut st.view_ms[4]);
            }
            st.parts[2] = t2.elapsed().as_secs_f64() * 1e3;
            let t3 = std::time::Instant::now();
            // Heaps and pipelines are set only when they change. A root
            // signature that indexes the heaps directly is set again after
            // new heaps (the driver takes the heaps with it: on the RX 5700 XT
            // a frame with rows took over the 1 s watchdog without it, WARP did not mind).
            let mut bound: Option<*mut std::ffi::c_void> = None;
            let set_heaps = |sh: &ID3D12DescriptorHeap, bound: &mut Option<*mut std::ffi::c_void>| -> bool {
                let k = windows::core::Interface::as_raw(sh);
                if *bound != Some(k) {
                    l.SetDescriptorHeaps(&[Some(vh.clone()), Some(sh.clone())]);
                    *bound = Some(k);
                    return true;
                }
                false
            };
            // The mesh programs.
            let mut pipe: *mut std::ffi::c_void = std::ptr::null_mut();
            for (r, sh) in rows.iter().zip(&sheaps) {
                let (built, b) = (&r.built, &r.b);
                if set_heaps(sh, &mut bound) {
                    pipe = std::ptr::null_mut();
                }
                if windows::core::Interface::as_raw(&built.cpso) != pipe {
                    l.SetComputeRootSignature(&built.croot);
                    l.SetPipelineState(&built.cpso);
                    pipe = windows::core::Interface::as_raw(&built.cpso);
                }
                l.SetComputeRoot32BitConstants(0, b.mesh.constants.len() as u32, b.mesh.constants.as_ptr().cast(), 0);
                for &(param, is_srv, a) in &r.mroots {
                    if is_srv { l.SetComputeRootShaderResourceView(param, addr(a)) } else { l.SetComputeRootConstantBufferView(param, addr(a)) }
                }
                if let Some(param) = built.cp.stride {
                    l.SetComputeRootShaderResourceView(param, sva);
                }
                l.SetComputeRootUnorderedAccessView(built.cp.records, rva + r.rec);
                l.SetComputeRootUnorderedAccessView(built.cp.records + 1, iva + r.idx);
                l.SetComputeRootShaderResourceView(built.cp.records + 2, uva + base_at);
                if let Some(param) = built.cp.lengths {
                    let mut lengths = b.mesh.lengths.clone();
                    lengths.resize(lengths.len().next_multiple_of(4).max(4), 0);
                    l.SetComputeRoot32BitConstants(param, lengths.len() as u32, lengths.as_ptr().cast(), 0);
                }
                if let Some(param) = built.cp.bounds {
                    l.SetComputeRoot32BitConstants(param, 8, b.views.bounds.as_ptr().cast(), 0);
                }
                l.Dispatch(b.groups, 1, 1);
            }
            let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            l.ResourceBarrier(&[transition(&records, uav, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE), transition(&indices, uav, D3D12_RESOURCE_STATE_INDEX_BUFFER)]);
            // The pixel programs into our targets.
            l.OMSetRenderTargets(1, Some(&t.rtv), false, Some(&t.dsv));
            l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: t.width as f32, Height: t.height as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
            l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: t.width as i32, bottom: t.height as i32 }]);
            l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            pipe = std::ptr::null_mut();
            bound = None;
            for (r, sh) in rows.iter().zip(&sheaps) {
                let (built, b) = (&r.built, &r.b);
                if set_heaps(sh, &mut bound) {
                    pipe = std::ptr::null_mut();
                }
                let lay = &built.programs.layout;
                if windows::core::Interface::as_raw(&built.gpso) != pipe {
                    l.SetPipelineState(&built.gpso);
                    l.SetGraphicsRootSignature(&built.groot);
                    l.SetGraphicsRoot32BitConstants(0, 1, [0u32].as_ptr().cast(), 0);
                    pipe = windows::core::Interface::as_raw(&built.gpso);
                }
                l.SetGraphicsRoot32BitConstants(1, b.pixel.constants.len() as u32, b.pixel.constants.as_ptr().cast(), 0);
                for &(param, is_srv, a) in &r.proots {
                    if is_srv { l.SetGraphicsRootShaderResourceView(param, addr(a)) } else { l.SetGraphicsRootConstantBufferView(param, addr(a)) }
                }
                if let Some(param) = built.gp.stride {
                    l.SetGraphicsRootShaderResourceView(param, sva);
                }
                l.SetGraphicsRootShaderResourceView(built.gp.records, rva + r.rec);
                if let Some(param) = built.gp.bounds {
                    l.SetGraphicsRoot32BitConstants(param, 8, b.views.bounds.as_ptr().cast(), 0);
                }
                let n = u64::from(b.groups) * u64::from(lay.index_bytes());
                l.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW { BufferLocation: iva + r.idx, SizeInBytes: n as u32, Format: DXGI_FORMAT_R32_UINT }));
                l.DrawIndexedInstanced(b.groups * lay.max_prims * 3, 1, 0, 0, 0);
                st.drawn += 1;
            }
            l.ResourceBarrier(&[transition(&records, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, uav), transition(&indices, D3D12_RESOURCE_STATE_INDEX_BUFFER, uav)]);
            st.parts[3] = t3.elapsed().as_secs_f64() * 1e3;
        }
        st.ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(st)
    }
}

/// `bridgeps: frame F: rows R (drawn D, no pipelines N, refused X; not
/// ready T, unread U), textures from our table K (grey G), buffers B
/// (MiB), M ms; pipelines made/building/failed; lacking ...; record
/// parts ...; view heap S slots, kept on the device K MiB (N MiB new)`.
pub fn line(frame: u64, f: &Frame, st: &Stats) -> String {
    let c = cache().counts();
    let mut lacks = [0u32; LACKS.len()];
    for (_, o) in &f.rows {
        for (k, n) in lacks.iter_mut().enumerate() {
            *n += o.lacks >> k & 1;
        }
    }
    let lacks: Vec<String> = LACKS.iter().zip(lacks).filter(|(_, n)| *n > 0).map(|(w, n)| format!("{w} {n}")).collect();
    format!(
        "world render: frame {frame}: bridgeps: rows {} (drawn {}, no pipelines {}, refused {}; draws not ready {}, unread {}), textures from our table {} (grey {}), buffers {} ({:.1} MiB), record {:.1} ms; pipelines made {}, building {}, failed {}; lacking: {}{}; record parts: bindings {:.1} ms, uploads {:.1} ms, views {:.1} ms, list {:.1} ms; view heap {} slots, kept on the device {:.1} MiB ({:.1} MiB new); in place {} ({:.1} MiB); small inputs {} (made {}, {:.0} KiB); grey by reason: {}; descriptor writes: {}; views by part: {}; {}",
        f.rows.len(),
        st.drawn,
        st.no_pipeline,
        st.refused,
        f.not_ready,
        f.unread,
        st.textures,
        st.grey,
        st.buffers,
        st.bytes as f64 / f64::from(1 << 20),
        st.ms,
        c[0],
        c[1],
        c[2],
        if lacks.is_empty() { "nothing".into() } else { lacks.join(", ") },
        if st.why.is_empty() { String::new() } else { format!("; first refused: {}", st.why) },
        st.parts[0],
        st.parts[1],
        st.parts[2],
        st.parts[3],
        st.slots,
        st.kept as f64 / f64::from(1 << 20),
        st.kept_new as f64 / f64::from(1 << 20),
        st.in_place,
        st.in_place_bytes as f64 / f64::from(1 << 20),
        st.small[0],
        st.small[1],
        st.small_bytes as f64 / 1024.0,
        {
            let g: Vec<String> = super::textures::FLAT_REASONS.iter().zip(st.grey_why).filter(|x| x.1 > 0).map(|(w, n)| format!("{w} {n}")).collect();
            if g.is_empty() { "none".to_string() } else { g.join(", ") }
        },
        WRITES.iter().zip(st.writes).map(|(w, n)| format!("{w} {n}")).collect::<Vec<_>>().join(", "),
        VIEW_PARTS.iter().zip(st.view_ms).map(|(w, n)| format!("{w} {n:.2} ms")).collect::<Vec<_>>().join(", "),
        blocks_text(st),
    )
}

/// `table blocks: B (distinct D, kept from earlier frames K, made M), slots
/// S of T, flushes F, uncut U; by root r: B/D/K ...` ([`tables_on`]; `off` says so).
pub fn blocks_text(st: &Stats) -> String {
    if !tables_on() {
        return "table blocks off".into();
    }
    let roots: Vec<String> = st.by_root.iter().map(|(r, c)| format!("{} {}/{}/{}", if *r < 0 { "words".to_string() } else { format!("r{r}") }, c[0], c[1], c[2])).collect();
    format!("table blocks {} (distinct {}, kept from earlier frames {}, made {}), slots made {} of {}, flushes {}, uncut {}; by root (rows'/distinct/kept): {}", st.blocks[0], st.blocks[1], st.blocks[2], st.blocks[3], st.block_slots[1], st.block_slots[0], st.flushes, st.uncut, roots.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A made-up source: root 0 a table (slot 3 a constant buffer, slot 5
    /// a texture), root 4 constants (dword 1 the words' offset), heap
    /// index 7 a texture, 9 a buffer.
    struct Made;
    impl Source for Made {
        fn roots(&self) -> Vec<SrcRoot> {
            vec![SrcRoot::Table(0x1000), SrcRoot::None, SrcRoot::None, SrcRoot::None, SrcRoot::Constants(vec![VISIBLE_LIST | 5, 0x40])]
        }
        fn table(&self, root: u16, slot: u32) -> Option<SrcView> {
            match (root, slot) {
                (0, 3) => Some(SrcView { kind: view::CONSTANT, addr: 0x9000, bytes: 64, ..SrcView::default() }),
                (0, 4) => Some(SrcView { kind: view::BUFFER, addr: 0x5_0000_8000, bytes: 64, ..SrcView::default() }),
                (0, 5) => Some(SrcView { kind: view::TEXTURE, addr: 0xA000, bytes: 4096, ..SrcView::default() }),
                _ => None,
            }
        }
        fn bytes(&self, _: u16, slot: u32, v: &SrcView) -> Option<Arc<Vec<u8>>> {
            let mut b = vec![slot as u8; v.bytes as usize];
            b[..4].copy_from_slice(&0x2_0007u32.to_le_bytes());
            Some(Arc::new(b))
        }
        fn root_view(&self, _: u16, _: u64) -> Option<Arc<Vec<u8>>> {
            None
        }
        fn words(&self) -> Option<[u32; WORDS]> {
            let mut w = [0; WORDS];
            w[0] = 7;
            w[1] = 9;
            w[2] = 7;
            w[4] = 0x3f80_0000;
            w[5] = 0x2_0000;
            Some(w)
        }
        fn heap(&self, index: u32) -> Option<(u64, SrcView)> {
            let h = 0x10_0000 + 32 * u64::from(index);
            match index {
                _ if index >= 0x1_0000 => None,
                7 => Some((h, SrcView { kind: view::TEXTURE, ..SrcView::default() })),
                9 => Some((h, SrcView { kind: view::BUFFER, ..SrcView::default() })),
                _ => Some((h, SrcView::default())),
            }
        }
        fn samplers(&self) -> Option<Arc<Vec<(u32, SamplerDesc)>>> {
            None
        }
    }

    #[test]
    fn take_reads_the_named_slots_and_rewrites_the_words() {
        let used: Used = [((0, 3), NAMED_CBV), ((0, 4), 16), ((0, 5), NAMED_IMAGE), ((0, 6), NAMED_CBV), ((0, WORDS_SLOT), 4)].into_iter().collect();
        let mut o = take(&Made, (1, 2), &used);
        let ORoot::Table(t) = &o.roots[0] else { panic!("no table") };
        let kinds: Vec<(u32, String)> = t.iter().map(|(k, s)| (*k, format!("{s:?}").chars().take(12).collect())).collect();
        // Slot 3 a CBV, 4 structured, 5 a texture by handle, 6 zeros (named, no view), 86 left to the frame.
        assert_eq!(kinds.iter().map(|k| k.0).collect::<Vec<_>>(), vec![3, 4, 5, 6]);
        assert!(matches!(t[0].1, OSlot::Buffer(Kind::Cbv, _)) && matches!(t[1].1, OSlot::Buffer(Kind::Structured(16), _)));
        assert!(matches!(t[2].1, OSlot::Texture(0x10A0)));
        assert!(matches!(&t[3].1, OSlot::Buffer(Kind::Cbv, b) if b.iter().all(|x| *x == 0)));
        assert_eq!(o.lacks, 1 << 1 | 1 << 2 | 1 << 4);
        // Words: 7 a texture (word 0 and 2), 9 a buffer, 0 an empty slot.
        assert_eq!(&o.words[..4], &[0, NO_VIEW, 2, NO_VIEW]);
        // A word past every heap bound is data (1.0) and stays; one below it naming no view goes to the guard.
        assert_eq!(&o.words[4..6], &[0x3f80_0000, NO_VIEW]);
        assert_eq!(o.heap, vec![(0, 0x10_0000 + 32 * 7), (2, 0x10_0000 + 32 * 7)]);
        assert_eq!(o.words_at, 0x40);
        // The visible-meshlet bit is cleared.
        assert!(matches!(&o.roots[4], ORoot::Constants(c) if c[0] == 5));
        // The meshlet buffer at its GPU address names the group count.
        o.set_groups(0x1_0000_8000);
        assert_eq!(o.groups, 7);
        assert_eq!(o.handles().count(), 3);
    }

    #[test]
    fn frame_words_at_their_offsets() {
        let o = |at: u32, k: u32| Owned { words_at: at, words: [k; WORDS], ..Owned::default() };
        let f = Frame::new(vec![(0, o(0x40, 1)), (1, o(0, 2)), (2, o(0x400, 3))]);
        assert_eq!(f.words_end, 0x440);
        let mut b = vec![0u8; 0x80];
        f.write_words(&mut b);
        assert_eq!((&b[0..4], &b[0x40..0x44]), (&2u32.to_le_bytes()[..], &1u32.to_le_bytes()[..]));
        // The frame table of a row's draw holds slot 86's place; root 4 is kept.
        let row = Owned { roots: vec![ORoot::Table(Vec::new()), ORoot::None, ORoot::None, ORoot::None, ORoot::Constants(vec![9, 0x40, 7])], groups: 1, ..o(0x40, 2) };
        let f = Frame::new(vec![(0, row)]);
        let d = f.draw(0, 128);
        assert!(matches!(&d.roots[0], Root::Table(t) if t.len() == 1 && t[0].0 == WORDS_SLOT));
        assert!(matches!(&d.roots[4], Root::Constants(c) if c == &vec![9, 0x40, 7]));
    }
    /// A row's binding cut into blocks: the draw words' textures,
    /// each table with its raw twins after its slots, a named image slot
    /// past a table's views inside its block, slot 86 our words buffer.
    #[test]
    fn a_binding_cut_into_blocks() {
        use xnw_shader::RootType::{Constants, Table};
        let sh = |stage, user: &[(u32, u8, xnw_shader::RootType, u32)]| HostShader { stage, dxil: Vec::new(), user_data: user.iter().map(|&(index, slot, root, dword)| emit::UserData { index, slot, root, dword }).collect(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), address_map: false, descriptor_words: false };
        let ms = sh(HwStage::Ms, &[(2, 0, Table, 0), (3, 1, Table, 0), (5, 4, Constants, 0)]);
        let ps = sh(HwStage::Ps, &[(0, 0, Table, 0), (1, 2, Table, 0)]);
        let p = Programs { ms, layout: emit::mesh::Layout { max_vertices: 3, max_prims: 1, rows: 1, elements: Vec::new() }, vs: Vec::new(), ps, wave: 32, vertex: None };
        let used: Used = [((0, 1), NAMED_CBV), ((1, 2), 8), ((2, 0), NAMED_CBV), ((2, 3), NAMED_IMAGE), ((2, 5), NAMED_IMAGE)].into_iter().collect();
        let (cb, words, rec, st8, mat) = ([1u8; 256], [0u8; 64], [3u8; 128], [4u8; 64], [5u8; 256]);
        let v = |kind, bytes| Slot::Buffer(View { kind, bytes });
        let d = Draw {
            roots: vec![
                Root::Table(vec![(1, v(Kind::Cbv, &cb)), (86, v(Kind::Raw, &words))]),
                Root::Table(vec![(0, v(Kind::Raw, &rec)), (2, v(Kind::Structured(8), &st8))]),
                Root::Table(vec![(0, v(Kind::Cbv, &mat)), (3, Slot::Handle(0x5000))]),
                Root::None,
                Root::Constants(vec![7, 0x40]),
            ],
            heap: vec![(9, Slot::Handle(0x7000))],
            samplers: Vec::new(),
            sampler_slots: 1,
            groups: 1,
        };
        let b = bridge::bind_at(&p.ms, &p.ps, &d, 500).unwrap();
        let views: Vec<(u32, Kind, At, u32)> = b.views.slots.iter().map(|&(s, v)| (s, v.kind, At::Frame(u64::from(s)), 1)).collect();
        let lay = Lay::new(&p, &used);
        assert_eq!(lay, Lay { roots: vec![0, 1, 2], named: vec![2, 3, 6], images: vec![vec![], vec![], vec![3, 5]] });
        let texs = Texs::default();
        let cx = KeyCx { texs: &texs, grey: 1, zc: &[] };
        let mut c = Cuts::default();
        cut(&b, &lay, &views, 500, &cx, &[], &mut c);
        // A block's slots written out: descriptors and stride entries.
        let dense = |c: &Cuts, blk: &Block| -> (Vec<Desc>, Vec<[u32; 2]>) {
            let (mut d, mut s) = (vec![Desc::Null; blk.len as usize], vec![[0, 0]; blk.len as usize]);
            for e in &c.ents[blk.ents.clone()] {
                (d[e.j as usize], s[e.j as usize]) = (e.desc, e.stride);
            }
            (d, s)
        };
        let blocks: Vec<(Vec<Desc>, Vec<[u32; 2]>)> = c.blocks.iter().map(|x| dense(&c, x)).collect();
        assert_eq!(c.blocks.iter().map(|x| (x.root, x.len)).collect::<Vec<_>>(), vec![(-1, 10), (0, 87), (1, 4), (2, 6)]);
        assert_eq!(c.rows, vec![0..4]);
        assert_eq!(blocks[0].0[9], Desc::Tex(0x7000));
        assert!(blocks[0].0[..9].iter().all(|d| *d == Desc::Null));
        assert!(matches!(blocks[1].0[1], Desc::Buf(Kind::Cbv, ..)) && blocks[1].0[86] == Desc::Words);
        // Root 1: the structured view at 2 names its raw twin, the block's slot 3.
        assert!(matches!(blocks[2].0[2], Desc::Buf(Kind::Structured(8), ..)) && matches!(blocks[2].0[3], Desc::Buf(Kind::Raw, ..)));
        assert_eq!(blocks[2].1, vec![[4, 0], [0, 0], [8, 4], [4, 0]]);
        // Root 2: the texture at 3, the named image at 5 past its views a null 2D texture.
        assert_eq!((blocks[3].0[3], blocks[3].0[5], blocks[3].0[4]), (Desc::Tex(0x5000), Desc::NullImage, Desc::Null));
        // Only the slots that are not a plain null are listed.
        assert_eq!(c.ents[c.blocks[2].ents.clone()].len(), 3);
        // The same draw at another base cuts into the same blocks and keys, appended.
        let b2 = bridge::bind_at(&p.ms, &p.ps, &d, 20).unwrap();
        let views2: Vec<(u32, Kind, At, u32)> = b2.views.slots.iter().map(|&(s, v)| (s, v.kind, At::Frame(u64::from(s) + 480), 1)).collect();
        cut(&b2, &lay, &views2, 20, &cx, &[], &mut c);
        assert_eq!(c.rows, vec![0..4, 4..8]);
        for k in 0..4 {
            assert_eq!(blocks[k], dense(&c, &c.blocks[4 + k]));
            assert!(c.key(&c.blocks[k]).is_some());
            assert_eq!(c.key(&c.blocks[k]), c.key(&c.blocks[4 + k]));
        }
        assert_ne!(c.key(&c.blocks[2]), c.key(&c.blocks[3]));
        // A table taken from an earlier row's block: its entries and key, not cut again.
        cut(&b2, &lay, &views2, 20, &cx, &[Some(1), None, None], &mut c);
        assert_eq!((c.blocks[9].ents.clone(), c.blocks[9].len), (c.blocks[1].ents.clone(), 87));
        assert_eq!(c.key(&c.blocks[9]), c.key(&c.blocks[1]));
        assert_eq!(dense(&c, &c.blocks[10]), blocks[2]);
    }

    /// Small inputs by content: equal bytes at one offset on every
    /// frame that uses them, written once; freed after SMALL_FRAMES unused, their
    /// place taken by the next of that size; none past the buffer's end.
    #[test]
    fn small_inputs_stay_where_they_lie() {
        let mut m = Small::default();
        m.start(1024);
        m.next_frame();
        let (a, b, c) = ([1u8; 100], [2u8; 100], [3u8; 300]);
        assert_eq!(m.place(&a, 256, 256), Some((0, true)));
        assert_eq!(m.place(&b, 256, 256), Some((256, true)));
        assert_eq!(m.place(&a, 256, 256), Some((0, false)));
        // Another size or unit is another entry.
        assert_eq!(m.place(&a, 512, 256), Some((512, true)));
        for _ in 0..SMALL_FRAMES {
            m.next_frame();
            assert_eq!((m.place(&b, 256, 256), m.placed, m.made), (Some((256, false)), 1, 0));
        }
        m.next_frame();
        // Both `a` entries went unused for SMALL_FRAMES frames: freed; `c` takes the 512-byte place, a new 256-byte input the other.
        assert_eq!(m.place(&c, 512, 256), Some((512, true)));
        assert_eq!(m.place(&[9u8; 16], 256, 256), Some((0, true)));
        assert_eq!(m.place(&[8u8; 16], 256, 256), None);
        assert_eq!(m.place(&b, 256, 256), Some((256, false)));
    }

    /// Cut cost on a frame shaped like pk7c2 f1400 (722 rows; a frame
    /// table of 160 slots, 40 views; a mesh table of 12; a material table
    /// of 24 with 10 textures; 8 word textures): `cargo test -p xnw-world
    /// --lib cut_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn cut_cost() {
        use xnw_shader::RootType::Table;
        let sh = |stage, user: &[(u32, u8)]| HostShader { stage, dxil: Vec::new(), user_data: user.iter().map(|&(index, slot)| emit::UserData { index, slot, root: Table, dword: 0 }).collect(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), address_map: false, descriptor_words: false };
        let p = Programs { ms: sh(HwStage::Ms, &[(0, 0), (1, 1)]), layout: emit::mesh::Layout { max_vertices: 3, max_prims: 1, rows: 1, elements: Vec::new() }, vs: Vec::new(), ps: sh(HwStage::Ps, &[(0, 0), (1, 2)]), wave: 32, vertex: None };
        let used: Used = (0..160).map(|k| ((0u16, k), NAMED_CBV)).chain((0..12).map(|k| ((1, k), 4))).chain((0..24).map(|k| ((2, k), NAMED_IMAGE))).collect();
        let lay = Lay::new(&p, &used);
        let bytes = vec![0u8; 4096];
        let v = |kind| Slot::Buffer(View { kind, bytes: &bytes });
        let d = Draw {
            roots: vec![
                Root::Table((0..160).step_by(4).map(|k| (k, v(if k % 8 == 0 { Kind::Structured(16) } else { Kind::Cbv }))).collect()),
                Root::Table((0..12).map(|k| (k, v(if k % 3 == 0 { Kind::Typed { format: 2, bytes: 16 } } else { Kind::Raw }))).collect()),
                Root::Table((0..24).map(|k| if k < 10 { (k, Slot::Handle(0x5000 + u64::from(k))) } else { (k, v(Kind::Cbv)) }).collect()),
            ],
            heap: (0..8).map(|k| (k, Slot::Handle(0x9000 + u64::from(k)))).collect(),
            samplers: Vec::new(),
            sampler_slots: 1,
            groups: 1,
        };
        let rows: Vec<_> = (0..722u32).map(|i| bridge::bind_at(&p.ms, &p.ps, &d, i * 300).unwrap()).collect();
        let views: Vec<Vec<(u32, Kind, At, u32)>> = rows.iter().map(|b| b.views.slots.iter().map(|&(s, v)| (s, v.kind, At::Frame(u64::from(s)), 1)).collect()).collect();
        let texs = Texs::default();
        let cx = KeyCx { texs: &texs, grey: 1, zc: &[] };
        let mut c = Cuts::default();
        for round in 0..3 {
            for reuse in [&[][..], &[Some(1), None, None][..]] {
                c.clear();
                let t = std::time::Instant::now();
                for (i, b) in rows.iter().enumerate() {
                    cut(b, &lay, &views[i], i as u32 * 300, &cx, if i == 0 { &[] } else { reuse }, &mut c);
                }
                println!("round {round} reuse {}: {} rows {:.2} ms; blocks {}, ents {}, key words {}", !reuse.is_empty(), rows.len(), t.elapsed().as_secs_f64() * 1e3, c.blocks.len(), c.ents.len(), c.keys.len());
            }
        }
    }
}
