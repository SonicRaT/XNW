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
//! - [`record`]: every row of a [`Frame`] on our pass's open list. A
//!   texture is the host copy our texture table holds for its handle this
//!   frame (a 1x1 grey stands in when it has none).
//!
//! The draw words are rewritten: word k naming a texture becomes k (that
//! texture at view heap slot k), every other word [`NO_VIEW`] (past the
//! heap bounds: the guard). The view heap is then the 16 word slots and
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
use xnw_shader::emit::{self, HostShader, Options};
use xnw_shader::HwStage;

use super::bridge::{self, Draw, Params, Programs, Root, SamplerDesc, Slot};
use super::preskin::{Kind, View};

/// In [`Used`]: a slot read as a constant buffer (other values: the stride
/// a buffer is read at).
pub const NAMED_CBV: u32 = u32::MAX;
/// The table slots the two programs name, (root, slot) -> stride or [`NAMED_CBV`].
pub type Used = HashMap<(u16, u32), u32>;
/// The draw words: the frame table's (root 0) slot 86, at the byte offset
/// in root 4's constant dword 1 (`worldprobe/pack.rs` `draw_words`).
pub const WORDS_SLOT: u32 = 86;
pub const WORDS_ROOT: usize = 4;
pub const WORDS_DWORD: usize = 1;
pub const WORDS: usize = 16;
/// A rewritten draw word that names no texture: past every heap bound
/// (times 32 still fits 32 bits).
pub const NO_VIEW: u32 = 0x00FF_FFFF;
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
            *e = (*e).max(if u.kind == xnw_shader::ResourceType::Cbv { NAMED_CBV } else { u.stride });
        }
    }
    Ok(())
}

/// The translated pair of a mesh and a pixel container, and the table
/// slots they name.
pub fn translate(msc: &[u8], psc: &[u8]) -> Result<(Programs, Used), String> {
    let (ms, layout) = emit::mesh::translate_fallback(msc, Options::default()).map_err(|e| format!("mesh program: {e}"))?;
    let ps = emit::translate_with(psc, Options::default()).map_err(|e| format!("pixel program: {e}"))?;
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
}

/// What [`Owned::lacks`] counts, by bit.
pub const LACKS: [&str; 7] = ["no draw words", "a word texture outside the heap", "a named constant buffer without bytes (zeros)", "a named view without bytes", "no samplers (defaults)", "a root view without bytes", "no group count"];

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
                Some(b) => ORoot::View(b),
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
                            slots.push((slot, OSlot::Buffer(kind, b)));
                        }
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
    /// Milliseconds the translation and the pipelines took.
    pub ms: f64,
}

// SAFETY: D3D12 objects are free-threaded; the programs are plain data.
unsafe impl Send for Built {}
// SAFETY: as above.
unsafe impl Sync for Built {}

/// The pair's pipelines on `dev`: the mesh program's compute form, and the
/// pass-through vertex shader with the pixel program into our colour
/// target (RGBA8; its other targets dropped) and depth (D32, GREATER_EQUAL, no write).
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
    Ok(Built { programs: p, used, croot, cpso, groot, gpso, cp, gp, ms: t0.elapsed().as_secs_f64() * 1e3 })
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

/// What one [`record`] did.
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
    /// Buffers uploaded and their bytes (each distinct buffer once).
    pub buffers: u32,
    pub bytes: u64,
    pub ms: f64,
}

/// What a [`record`] needs of our pass.
pub struct Target<'a> {
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub width: u32,
    pub height: u32,
    /// A texture by handle: its host copy and view this frame.
    pub texture: &'a dyn Fn(u64) -> Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)>,
    /// A 1x1 texture in a shader-resource state (the grey stand-in).
    pub grey: &'a ID3D12Resource,
    /// The words buffer ([`Frame::write_words`] filled it) and its bytes.
    pub words: (&'a ID3D12Resource, u64),
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

/// One frame's upload memory: every distinct buffer once, each at an
/// offset its view can start at (`unit`: the element size; 256 for
/// constant buffers and root views).
#[derive(Default)]
struct Arena {
    at: HashMap<(usize, usize, u64, u64), u64>,
    parts: Vec<(u64, usize, usize, u64)>,
    end: u64,
}

impl Arena {
    fn place(&mut self, bytes: &[u8], size: u64, unit: u64) -> u64 {
        let k = (bytes.as_ptr() as usize, bytes.len(), size, unit);
        if let Some(&o) = self.at.get(&k) {
            return o;
        }
        let unit = unit.max(4);
        let align = 256 * unit / gcd(256, unit);
        let o = self.end.next_multiple_of(align);
        self.end = o + size.max(16);
        self.at.insert(k, o);
        self.parts.push((o, bytes.as_ptr() as usize, bytes.len().min(size as usize), size));
        o
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A view kind's element size in the arena (its start must be a multiple).
fn unit(kind: Kind) -> u64 {
    match kind {
        Kind::Raw | Kind::RawUav { .. } => 16,
        Kind::Structured(s) => u64::from(s.max(4)),
        Kind::Typed { bytes, .. } => u64::from(bytes.max(1)),
        Kind::Cbv => 256,
    }
}

/// Records every row of `f` whose pipelines are made on the open list
/// `l` (our pass's, after its own draws: colour and depth in their target
/// states). Every mesh program first (their records and index triples in
/// one buffer each), one barrier, then every pixel program; one upload
/// buffer holds every distinct input. Every object the GPU uses is pushed
/// to `keep`, for the caller to hold until the list's fence. The caller
/// binds its own heaps again after it if it draws more.
pub fn record(dev: &ID3D12Device, l: &ID3D12GraphicsCommandList, f: &Frame, t: &Target, keep: &mut Vec<windows::core::IUnknown>) -> Result<Stats, String> {
    let t0 = std::time::Instant::now();
    let mut st = Stats::default();
    let e = |e: windows::core::Error| e.message().to_string();
    // The bindings, and where every input and output goes.
    let mut rows = Vec::new();
    let mut arena = Arena::default();
    let (mut rec_end, mut idx_end) = (0u64, 0u64);
    for i in 0..f.rows.len() {
        let o = &f.rows[i].1;
        let Some(built) = cache().get(o.key) else {
            st.no_pipeline += 1;
            continue;
        };
        if u64::from(o.words_at) + 4 * WORDS as u64 > t.words.1 {
            st.refused += 1;
            if st.why.is_empty() {
                st.why = format!("row {}: its words at {:#x} lie past our words buffer ({} B)", f.rows[i].0, o.words_at, t.words.1);
            }
            continue;
        }
        let p = &built.programs;
        let b = match bridge::bind(&p.ms, &p.ps, &f.draw(i, SAMPLER_SLOTS)) {
            Ok(b) if b.groups > 0 => b,
            r => {
                st.refused += 1;
                if st.why.is_empty() {
                    st.why = format!("row {}: {}", f.rows[i].0, r.err().unwrap_or_else(|| "no group count".into()));
                }
                continue;
            }
        };
        let mut views = Vec::new();
        for &(s, v) in &b.views.slots {
            if let Kind::RawUav { .. } = v.kind {
                continue;
            }
            if let Some((size, n)) = v.kind.extent(v.bytes.len() as u64) {
                views.push((s, v.kind, arena.place(v.bytes, size, unit(v.kind)), n));
            }
        }
        let strides: Vec<u8> = b.views.strides.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        let root = |a: &mut Arena, vs: &[(u8, bool, &[u8])], params: &[(u8, bool, u32)]| -> Result<Vec<(u32, bool, u64)>, String> {
            params.iter().map(|&(r, is_srv, param)| {
                let v = vs.iter().find(|v| v.0 == r && v.1 == is_srv).ok_or_else(|| format!("root view {r} not bound"))?;
                Ok((param, is_srv, a.place(v.2, (v.2.len() as u64).next_multiple_of(256), 256)))
            }).collect()
        };
        let (mroots, proots) = match (root(&mut arena, &b.mesh.views, &built.cp.views), root(&mut arena, &b.pixel.views, &built.gp.views)) {
            (Ok(m), Ok(p)) => (m, p),
            (Err(x), _) | (_, Err(x)) => {
                st.refused += 1;
                if st.why.is_empty() {
                    st.why = format!("row {}: {x}", f.rows[i].0);
                }
                continue;
            }
        };
        let stride_at = arena.place(&strides, strides.len() as u64, 16);
        let lay = &p.layout;
        let groups = u64::from(b.groups);
        let (rec, idx) = (rec_end, idx_end);
        rec_end += (groups * u64::from(lay.record_bytes())).next_multiple_of(256);
        idx_end += (groups * u64::from(lay.index_bytes())).next_multiple_of(256);
        rows.push((i, built, b, views, mroots, proots, stride_at, rec, idx, strides));
    }
    let base_at = arena.place(&[0; 16], 16, 16);
    if rows.is_empty() {
        st.ms = t0.elapsed().as_secs_f64() * 1e3;
        return Ok(st);
    }
    // SAFETY: D3D12 calls with descriptions that outlive them; the arena's
    // parts point into `f` (the rows' bytes) and the rows' own stride
    // tables, both alive here; every object the list uses goes to `keep`.
    unsafe {
        let committed = |size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES| -> Result<ID3D12Resource, String> {
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size.max(256), Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: flags, ..Default::default() };
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).map_err(e)?;
            r.ok_or_else(|| "no resource".to_string())
        };
        let up = committed(arena.end, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        let mut to = std::ptr::null_mut();
        up.Map(0, None, Some(&mut to)).map_err(e)?;
        let to = to.cast::<u8>();
        for &(o, ptr, n, size) in &arena.parts {
            std::ptr::copy_nonoverlapping(ptr as *const u8, to.add(o as usize), n);
            std::ptr::write_bytes(to.add(o as usize + n), 0, size.max(16) as usize - n);
        }
        up.Unmap(0, None);
        st.buffers = arena.parts.len() as u32;
        st.bytes = arena.end;
        let uva = up.GetGPUVirtualAddress();
        let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let records = committed(rec_end, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, uav)?;
        let indices = committed(idx_end, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, uav)?;
        let (rva, iva) = (records.GetGPUVirtualAddress(), indices.GetGPUVirtualAddress());
        for r in [&up, &records, &indices] {
            keep.push(windows::core::Interface::cast(r).map_err(e)?);
        }
        let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let sstep = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
        let srv = |format, first: u64, n: u32, stride: u32, flags| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_BUFFER, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: first, NumElements: n, StructureByteStride: stride, Flags: flags } } };
        let tex2d = |format| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } } };
        let mut sampler_heaps: HashMap<(usize, u32), ID3D12DescriptorHeap> = HashMap::new();
        let mut heaps = Vec::with_capacity(rows.len());
        // The mesh programs.
        for (i, built, b, views, mroots, _, stride_at, rec, idx, _) in &rows {
            let o = &f.rows[*i].1;
            // The view heap: typed nulls, the guards, the buffers, the textures.
            let vh: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: b.views.heap_slots.max(1), Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).map_err(e)?;
            let at = |slot: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: vh.GetCPUDescriptorHandleForHeapStart().ptr + slot as usize * step };
            for s in 0..b.views.heap_slots {
                dev.CreateShaderResourceView(None, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s));
            }
            let g = &b.views.bounds;
            dev.CreateShaderResourceView(None, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(g[1]));
            let raw_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R32_TYPELESS, ViewDimension: D3D12_UAV_DIMENSION_BUFFER, Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } } };
            dev.CreateUnorderedAccessView(None, None, Some(&raw_uav), at(g[3]));
            dev.CreateUnorderedAccessView(None, None, Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() }), at(g[4]));
            dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at(g[5]));
            for &(s, kind, off, n) in views {
                let first = off / unit(kind);
                match kind {
                    Kind::Raw => dev.CreateShaderResourceView(&up, Some(&srv(DXGI_FORMAT_R32_TYPELESS, off / 4, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s)),
                    Kind::Structured(stride) => dev.CreateShaderResourceView(&up, Some(&srv(DXGI_FORMAT_UNKNOWN, first, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                    Kind::Typed { format, .. } => dev.CreateShaderResourceView(&up, Some(&srv(DXGI_FORMAT(format), first, n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                    Kind::Cbv => dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: uva + off, SizeInBytes: n }), at(s)),
                    Kind::RawUav { .. } => {}
                }
            }
            // Slot 86 of the frame table: our words buffer, whole.
            if let Some(s86) = b.views.constants.first().map(|c| c / 32 + WORDS_SLOT).filter(|s| b.views.slots.iter().any(|v| v.0 == *s)) {
                dev.CreateShaderResourceView(t.words.0, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, (t.words.1 / 4) as u32, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s86));
            }
            for &(s, h) in &b.handles {
                match (t.texture)(h) {
                    Some((r, desc)) => {
                        dev.CreateShaderResourceView(&r, Some(&desc), at(s));
                        keep.push(windows::core::Interface::cast(&r).map_err(e)?);
                        st.textures += 1;
                    }
                    None => {
                        dev.CreateShaderResourceView(t.grey, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(s));
                        st.grey += 1;
                    }
                }
            }
            // Samplers: one default (linear, wrap) at every slot, the
            // title's entries at their indices; one heap per entry list.
            let sk = (Arc::as_ptr(&o.samplers) as usize, b.sampler_slots);
            let sh = match sampler_heaps.get(&sk) {
                Some(h) => h.clone(),
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
                    keep.push(windows::core::Interface::cast(&h).map_err(e)?);
                    sampler_heaps.insert(sk, h.clone());
                    h
                }
            };
            keep.push(windows::core::Interface::cast(&vh).map_err(e)?);
            l.SetDescriptorHeaps(&[Some(vh.clone()), Some(sh.clone())]);
            l.SetComputeRootSignature(&built.croot);
            l.SetPipelineState(&built.cpso);
            l.SetComputeRoot32BitConstants(0, b.mesh.constants.len() as u32, b.mesh.constants.as_ptr().cast(), 0);
            for &(param, is_srv, off) in mroots {
                if is_srv { l.SetComputeRootShaderResourceView(param, uva + off) } else { l.SetComputeRootConstantBufferView(param, uva + off) }
            }
            if let Some(param) = built.cp.stride {
                l.SetComputeRootShaderResourceView(param, uva + stride_at);
            }
            l.SetComputeRootUnorderedAccessView(built.cp.records, rva + rec);
            l.SetComputeRootUnorderedAccessView(built.cp.records + 1, iva + idx);
            l.SetComputeRootShaderResourceView(built.cp.records + 2, uva + base_at);
            if let Some(param) = built.cp.lengths {
                let mut lengths = b.mesh.lengths.clone();
                lengths.resize(lengths.len().next_multiple_of(4).max(4), 0);
                l.SetComputeRoot32BitConstants(param, lengths.len() as u32, lengths.as_ptr().cast(), 0);
            }
            if let Some(param) = built.cp.bounds {
                l.SetComputeRoot32BitConstants(param, 8, g.as_ptr().cast(), 0);
            }
            l.Dispatch(b.groups, 1, 1);
            heaps.push((vh, sh));
        }
        l.ResourceBarrier(&[transition(&records, uav, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE), transition(&indices, uav, D3D12_RESOURCE_STATE_INDEX_BUFFER)]);
        // The pixel programs into our targets.
        l.OMSetRenderTargets(1, Some(&t.rtv), false, Some(&t.dsv));
        l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: t.width as f32, Height: t.height as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: t.width as i32, bottom: t.height as i32 }]);
        l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        for ((_, built, b, _, _, proots, stride_at, rec, idx, _), (vh, sh)) in rows.iter().zip(&heaps) {
            let lay = &built.programs.layout;
            l.SetDescriptorHeaps(&[Some(vh.clone()), Some(sh.clone())]);
            l.SetPipelineState(&built.gpso);
            l.SetGraphicsRootSignature(&built.groot);
            l.SetGraphicsRoot32BitConstants(0, 1, [0u32].as_ptr().cast(), 0);
            l.SetGraphicsRoot32BitConstants(1, b.pixel.constants.len() as u32, b.pixel.constants.as_ptr().cast(), 0);
            for &(param, is_srv, off) in proots {
                if is_srv { l.SetGraphicsRootShaderResourceView(param, uva + off) } else { l.SetGraphicsRootConstantBufferView(param, uva + off) }
            }
            if let Some(param) = built.gp.stride {
                l.SetGraphicsRootShaderResourceView(param, uva + stride_at);
            }
            l.SetGraphicsRootShaderResourceView(built.gp.records, rva + rec);
            if let Some(param) = built.gp.bounds {
                l.SetGraphicsRoot32BitConstants(param, 8, b.views.bounds.as_ptr().cast(), 0);
            }
            let n = u64::from(b.groups) * u64::from(lay.index_bytes());
            l.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW { BufferLocation: iva + idx, SizeInBytes: n as u32, Format: DXGI_FORMAT_R32_UINT }));
            l.DrawIndexedInstanced(b.groups * lay.max_prims * 3, 1, 0, 0, 0);
            st.drawn += 1;
        }
    }
    st.ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(st)
}

/// `bridgeps: frame F: rows R (drawn D, no pipelines N, refused X; not
/// ready T, unread U), textures from our table K (grey G), buffers B
/// (MiB), M ms; pipelines made/building/failed; lacking ...`.
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
        "world render: frame {frame}: bridgeps: rows {} (drawn {}, no pipelines {}, refused {}; draws not ready {}, unread {}), textures from our table {} (grey {}), buffers {} ({:.1} MiB), record {:.1} ms; pipelines made {}, building {}, failed {}; lacking: {}{}",
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
        if st.why.is_empty() { String::new() } else { format!("; first refused: {}", st.why) }
    )
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
            Some(w)
        }
        fn heap(&self, index: u32) -> Option<(u64, SrcView)> {
            let h = 0x10_0000 + 32 * u64::from(index);
            match index {
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
        let used: Used = [((0, 3), NAMED_CBV), ((0, 4), 16), ((0, 5), 0), ((0, 6), NAMED_CBV), ((0, WORDS_SLOT), 4)].into_iter().collect();
        let mut o = take(&Made, (1, 2), &used);
        let ORoot::Table(t) = &o.roots[0] else { panic!("no table") };
        let kinds: Vec<(u32, String)> = t.iter().map(|(k, s)| (*k, format!("{s:?}").chars().take(12).collect())).collect();
        // Slot 3 a CBV, 4 structured, 5 a texture by handle, 6 zeros (named, no view), 86 left to the frame.
        assert_eq!(kinds.iter().map(|k| k.0).collect::<Vec<_>>(), vec![3, 4, 5, 6]);
        assert!(matches!(t[0].1, OSlot::Buffer(Kind::Cbv, _)) && matches!(t[1].1, OSlot::Buffer(Kind::Structured(16), _)));
        assert!(matches!(t[2].1, OSlot::Texture(0x10A0)));
        assert!(matches!(&t[3].1, OSlot::Buffer(Kind::Cbv, b) if b.iter().all(|x| *x == 0)));
        assert_eq!(o.lacks, 1 << 2 | 1 << 4);
        // Words: 7 a texture (word 0 and 2), 9 a buffer, 0 an empty slot.
        assert_eq!(&o.words[..4], &[0, NO_VIEW, 2, NO_VIEW]);
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
    fn arena_places_each_buffer_once_at_its_unit() {
        let (x, y) = (vec![1u8; 40], vec![2u8; 8]);
        let mut a = Arena::default();
        let o1 = a.place(&x, 40, 256);
        let o2 = a.place(&y, 12, 12);
        assert_eq!((o1, o2), (0, 768));
        assert_eq!(a.place(&x, 40, 256), 0);
        assert_eq!(a.place(&y, 16, 16) % 256, 0);
        assert_eq!(a.parts.len(), 3);
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
}
