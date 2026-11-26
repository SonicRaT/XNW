//! One captured G-buffer draw through the title's own programs: its mesh
//! program in compute form (the mesh fallback: records and index triples,
//! then the pass-through vertex shader) and its pixel program, into 32-bit
//! float targets and a depth target of our own, with target 0 and the depth
//! read back. Without the pixel program (`Target::depth_only`) the records
//! are drawn into the depth target alone: the mesh program's geometry.
//!
//! - [`bind`] (pure): a captured draw's root arguments, tables and
//!   heap-picked views to the two stages' root constants and root views, one
//!   small view heap and the sampler heap. The view heap's layout is
//!   [`preskin::bind`]'s (heap-picked views at their captured indices, each
//!   table root past them, raw twins, guards, bounds); a texture keeps the
//!   slot the layout gave it.
//! - [`needs`] and [`gap`]: what an emitted program asks of an adapter, read
//!   from its container (shader model and stage from the DXIL part's program
//!   version, optional features from `SFI0`), and why an adapter cannot run it.
//! - [`run`]: the dispatch and the draw on a device. The title's translated
//!   programs run on the software adapter; the hardware adapter only with
//!   `XNW_BRIDGE_HW=1` and the per-submission GPU watchdog running
//!   ([`title_programs_here`]).

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use xnw_shader::emit::{self, HostShader};
use xnw_shader::{HwStage, RootType};

use super::preskin::{self, Kind, View};
use super::textures::Provided;
use std::collections::HashMap;

/// What a descriptor slot holds for the binding: a buffer view (as the
/// compute binding describes one) or a texture's texels.
#[derive(Clone, Copy)]
pub enum Slot<'a> {
    Buffer(View<'a>),
    Texture(&'a Provided),
    /// A texture by its descriptor handle in the title's heap: the caller
    /// makes its view (our pass's texture table, `bridged`).
    Handle(u64),
}

/// `D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE`.
pub const SAMPLER_HEAP_MAX: u32 = 2048;

/// A sampler heap entry as the host describes it.
pub type SamplerDesc = D3D12_SAMPLER_DESC;

/// A captured sampler heap entry's description (framepack `SamplerEntry`:
/// form 27 the console's description, as `CreateSamplerX` took it; form 13
/// a D3D12 description) as the host's; None for raw words.
pub fn sampler_desc(form: u32, desc: &[u32; 27]) -> Option<SamplerDesc> {
    let f = |i: usize| f32::from_bits(desc[i]);
    match form {
        27 => Some(xnw_gpu::d3d12x::sampler::console(desc)),
        13 => Some(D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER(desc[0] as i32),
            AddressU: D3D12_TEXTURE_ADDRESS_MODE(desc[1] as i32),
            AddressV: D3D12_TEXTURE_ADDRESS_MODE(desc[2] as i32),
            AddressW: D3D12_TEXTURE_ADDRESS_MODE(desc[3] as i32),
            MipLODBias: f(4),
            MaxAnisotropy: desc[5],
            ComparisonFunc: D3D12_COMPARISON_FUNC(desc[6] as i32),
            BorderColor: [f(7), f(8), f(9), f(10)],
            MinLOD: f(11),
            MaxLOD: f(12),
        }),
        _ => None,
    }
}

/// One root argument of the captured draw, by the title's root parameter.
#[derive(Clone, Default)]
pub enum Root<'a> {
    #[default]
    None,
    Constants(Vec<u32>),
    /// A descriptor table: the views at their offsets (slots of 32 B).
    Table(Vec<(u32, Slot<'a>)>),
    /// A root view: the bytes from its address.
    View(&'a [u8]),
}

/// A captured draw, as far as the binding needs it.
#[derive(Clone, Default)]
pub struct Draw<'a> {
    /// Root arguments by root parameter index.
    pub roots: Vec<Root<'a>>,
    /// Views the programs pick from the heap by an index held in data (the
    /// bindless words), at their captured heap indices.
    pub heap: Vec<(u32, Slot<'a>)>,
    /// Sampler heap entries at their captured indices, as host descriptions.
    pub samplers: Vec<(u32, SamplerDesc)>,
    /// Sampler heap slots to make (at least the captured ones).
    pub sampler_slots: u32,
    /// Mesh groups to dispatch.
    pub groups: u32,
}

/// One stage's root arguments beside the heaps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stage<'a> {
    /// The user-data dwords (`b0` space 0).
    pub constants: Vec<u32>,
    /// Root views: (root parameter, SRV or CBV, bytes).
    pub views: Vec<(u8, bool, &'a [u8])>,
    /// Bytes each root SRV may read, in the program's root SRV order.
    pub lengths: Vec<u32>,
}

/// What [`run`] binds for one draw.
#[derive(Clone, Default)]
pub struct Binding<'a> {
    /// The view heap: buffer views by slot, stride table, bounds. Its
    /// `constants` hold, per table root, the table's first slot times 32.
    pub views: preskin::Binding<'a>,
    /// Textures by view heap slot.
    pub textures: Vec<(u32, &'a Provided)>,
    /// Textures by descriptor handle, by view heap slot ([`Slot::Handle`]).
    pub handles: Vec<(u32, u64)>,
    pub samplers: Vec<(u32, SamplerDesc)>,
    pub sampler_slots: u32,
    pub mesh: Stage<'a>,
    pub pixel: Stage<'a>,
    pub groups: u32,
}

/// The binding of draw `d` for mesh program `ms` (either form) and pixel
/// program `ps` (pure). Both stages share one view heap: a table root is
/// placed once, and each stage's user-data dword of it holds its first
/// slot times 32. `Err` names the first input the record lacks.
pub fn bind<'a>(ms: &HostShader, ps: &HostShader, d: &Draw<'a>) -> Result<Binding<'a>, String> {
    bind_at(ms, ps, d, 0)
}

/// [`bind`] with the draw's view heap slots starting at slot `base` of a
/// heap shared by many draws (`bridged`): every slot, table offset, raw
/// twin, guard and the heap bound move by `base`; the draw's heap-picked
/// views (`d.heap`, by word) too. The stride table's entries are for
/// slots `base..`: the caller puts them there in one table of the whole heap.
pub fn bind_at<'a>(ms: &HostShader, ps: &HostShader, d: &Draw<'a>, base: u32) -> Result<Binding<'a>, String> {
    let stages = [("mesh", ms), ("pixel", ps)];
    for (name, s) in stages {
        if !s.root_uavs.is_empty() || s.address_map || s.descriptor_words {
            return Err(format!("the {name} program reads root UAVs, an address map or descriptor words"));
        }
    }
    let mut tables: Vec<u8> = stages.iter().flat_map(|(_, s)| s.user_data.iter().filter(|u| u.root == RootType::Table).map(|u| u.slot)).collect();
    tables.sort_unstable();
    tables.dedup();
    // A texture holds its slot in the layout as an empty raw view.
    let lay = |s: &Slot<'a>| match s {
        Slot::Buffer(v) => *v,
        Slot::Texture(_) | Slot::Handle(_) => View { kind: Kind::Raw, bytes: &[] },
    };
    let mut dispatch = preskin::Dispatch { groups: [d.groups, 1, 1], tables: Vec::new(), heap: d.heap.iter().map(|(k, s)| (*k, lay(s))).collect(), constants: Vec::new(), texels: Vec::new() };
    // victory-fork (5a55384): a typed buffer view either program moves
    // 16-bit halves through as unsigned integers (`HostShader::half_views`)
    // is shown in the unsigned format of its layout. shortcut: the table is
    // shared by both stages, so a stage that reads such a slot as floats
    // would see the integer view (none of the 77 programs that name half
    // views is drawn beside one that does, I).
    let half: Vec<(u8, u32)> = ms.half_views.iter().chain(&ps.half_views).copied().collect();
    for &r in &tables {
        let Some(Root::Table(views)) = d.roots.get(usize::from(r)) else {
            return Err(format!("the record has no table at root {r}"));
        };
        let twin = |k: u32, mut v: View<'a>| {
            if half.iter().any(|&(root, dword)| root == r && dword / 8 == k)
                && let Kind::Typed { format, .. } = &mut v.kind
                && let Some(f) = crate::compute::half_format(*format)
            {
                *format = f;
            }
            v
        };
        dispatch.tables.push((u32::from(r), views.iter().map(|(k, s)| (*k, twin(*k, lay(s)))).collect()));
    }
    let user_data = tables.iter().map(|&r| emit::UserData { index: u32::from(r), slot: r, root: RootType::Table, dword: 0 }).collect();
    let layout = HostShader { stage: HwStage::Cs, dxil: Vec::new(), user_data, samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false };
    let mut views = preskin::bind(&layout, &dispatch)?;
    if base > 0 {
        for &r in &tables {
            views.constants[usize::from(r)] += 32 * base;
        }
        for s in &mut views.slots {
            s.0 += base;
        }
        for w in &mut views.strides {
            if w[1] & emit::StrideEntry::RAW_SLOTS != 0 {
                w[1] += base;
            }
        }
        for g in &mut views.bounds[..6] {
            *g += base;
        }
    }
    let first: Vec<u32> = views.constants.iter().map(|c| c / 32).collect();
    let mut textures: Vec<(u32, &Provided)> = d.heap.iter().filter_map(|(k, s)| if let Slot::Texture(t) = s { Some((base + *k, *t)) } else { None }).collect();
    for &r in &tables {
        if let Some(Root::Table(v)) = d.roots.get(usize::from(r)) {
            textures.extend(v.iter().filter_map(|(k, s)| if let Slot::Texture(t) = s { Some((first[usize::from(r)] + k, *t)) } else { None }));
        }
    }
    textures.sort_by_key(|t| t.0);
    let mut handles: Vec<(u32, u64)> = d.heap.iter().filter_map(|(k, s)| if let Slot::Handle(h) = s { Some((base + *k, *h)) } else { None }).collect();
    for &r in &tables {
        if let Some(Root::Table(v)) = d.roots.get(usize::from(r)) {
            handles.extend(v.iter().filter_map(|(k, s)| if let Slot::Handle(h) = s { Some((first[usize::from(r)] + k, *h)) } else { None }));
        }
    }
    handles.sort_by_key(|t| t.0);
    views.slots.retain(|(s, _)| textures.binary_search_by_key(s, |t| t.0).is_err() && handles.binary_search_by_key(s, |t| t.0).is_err());
    let stage = |name: &str, s: &HostShader| -> Result<Stage<'a>, String> {
        let mut constants = vec![0u32; s.user_data_dwords().max(1) as usize];
        for u in &s.user_data {
            constants[u.index as usize] = match (u.root, d.roots.get(usize::from(u.slot))) {
                (RootType::Table, Some(Root::Table(_))) if u.dword == 0 => views.constants[usize::from(u.slot)],
                (RootType::Constants, Some(Root::Constants(w))) => *w.get(u.dword as usize).ok_or_else(|| format!("{name}: root {} has {} constants, user data {} wants dword {}", u.slot, w.len(), u.index, u.dword))?,
                (kind, _) => return Err(format!("{name}: user data {} wants root {} as {kind:?} (dword {}); the record has none", u.index, u.slot, u.dword)),
            };
        }
        let mut bound = Vec::new();
        for (roots, srv) in [(&s.root_cbvs, false), (&s.root_srvs, true)] {
            for &r in roots {
                let Some(Root::View(b)) = d.roots.get(usize::from(r)) else {
                    return Err(format!("{name}: the record has no bytes for the root view at root {r}"));
                };
                bound.push((r, srv, *b));
            }
        }
        let lengths = bound.iter().filter(|v| v.1).map(|v| v.2.len() as u32).collect();
        Ok(Stage { constants, views: bound, lengths })
    };
    let (mesh, pixel) = (stage("mesh", ms)?, stage("pixel", ps)?);
    // A shader-visible sampler heap holds at most 2048 entries (heap
    // creation fails past it; VIctory): entries past it are left
    // out, and the heap bound sends their indices to the bound's guard.
    let sampler_slots = d.samplers.iter().map(|s| s.0 + 1).max().unwrap_or(0).max(d.sampler_slots).clamp(1, SAMPLER_HEAP_MAX);
    views.bounds[6] = sampler_slots;
    Ok(Binding { views, textures, handles, samplers: d.samplers.iter().filter(|s| s.0 < sampler_slots).copied().collect(), sampler_slots, mesh, pixel, groups: d.groups })
}

/// Optional-feature bits of a DXIL container's `SFI0` part that an adapter
/// may lack.
pub mod feature {
    pub const DOUBLES: u64 = 1 << 0;
    pub const DOUBLE_EXTENSIONS: u64 = 1 << 5;
    pub const WAVE_OPS: u64 = 1 << 14;
    pub const INT64_OPS: u64 = 1 << 15;
    /// SM 6.1 barycentrics (computed barycentrics in pixel programs).
    pub const BARYCENTRICS: u64 = 1 << 17;
    pub const ATOMIC64_TYPED: u64 = 1 << 22;
    pub const VIEW_HEAP_INDEXING: u64 = 1 << 25;
    pub const SAMPLER_HEAP_INDEXING: u64 = 1 << 26;
    pub const ATOMIC64_HEAP: u64 = 1 << 28;
    pub const NAMES: [(u64, &str); 9] = [(DOUBLES, "doubles"), (DOUBLE_EXTENSIONS, "11.1 double extensions"), (WAVE_OPS, "wave ops"), (INT64_OPS, "64-bit integers"), (BARYCENTRICS, "barycentrics"), (ATOMIC64_TYPED, "64-bit atomics on typed resources"), (VIEW_HEAP_INDEXING, "view heap indexing"), (SAMPLER_HEAP_INDEXING, "sampler heap indexing"), (ATOMIC64_HEAP, "64-bit atomics on heap resources")];
}

/// What one emitted program asks of an adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Needs {
    /// `ps`, `vs`, `cs`, `ms`... from the program version's kind.
    pub stage: &'static str,
    pub model: (u32, u32),
    /// The `SFI0` flags (0 without the part).
    pub features: u64,
    /// The `WaveSize` the program declares (lanes), given by the caller:
    /// the translator declares the console's wave on a compute-form program.
    pub wave: Option<u32>,
}

impl Needs {
    pub fn text(&self) -> String {
        let named: Vec<&str> = feature::NAMES.iter().filter(|(bit, _)| self.features & bit != 0).map(|(_, n)| *n).collect();
        let other = self.features & !feature::NAMES.iter().fold(0, |a, (bit, _)| a | bit);
        format!("{}_{}_{}, {}, features {:#x} ({}{})", self.stage, self.model.0, self.model.1, self.wave.map_or("no WaveSize".into(), |w| format!("WaveSize({w})")), self.features, if named.is_empty() { "none named".into() } else { named.join(", ") }, if other != 0 { format!("; other bits {other:#x}") } else { String::new() })
    }
}

/// The parts of a DXBC-style container: (fourcc, data).
fn parts(c: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let word = |at: usize| c.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
    let mut out = Vec::new();
    if c.get(..4) != Some(b"DXBC") {
        return out;
    }
    for k in 0..word(28).unwrap_or(0) {
        let Some(at) = word(32 + 4 * k) else { break };
        let (Some(name), Some(len)) = (c.get(at..at + 4), word(at + 4)) else { break };
        if let Some(data) = c.get(at + 8..at + 8 + len) {
            out.push(([name[0], name[1], name[2], name[3]], data));
        }
    }
    out
}

/// What the emitted container `dxil` needs: stage and shader model from the
/// DXIL part's program version (kind << 16 | major << 4 | minor), features
/// from `SFI0`. None: not a DXIL container.
pub fn needs(dxil: &[u8], wave: Option<u32>) -> Option<Needs> {
    let parts = parts(dxil);
    let program = parts.iter().find(|p| &p.0 == b"DXIL")?.1;
    let version = u32::from_le_bytes(program.get(..4)?.try_into().ok()?);
    let stage = ["ps", "vs", "gs", "hs", "ds", "cs", "lib", "raygen", "intersection", "anyhit", "closesthit", "miss", "callable", "ms", "as"].get((version >> 16) as usize).copied().unwrap_or("?");
    let features = parts.iter().find(|p| &p.0 == b"SFI0").and_then(|p| p.1.get(..8)).map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap_or_default()));
    Some(Needs { stage, model: (version >> 4 & 0xf, version & 0xf), features, wave })
}

/// Why `caps` cannot run programs with these needs (None: it can).
pub fn gap(caps: &xnw_gpu::host::Caps, needs: &[Needs]) -> Option<String> {
    let mut gap = Vec::new();
    if let Some(n) = needs.iter().max_by_key(|n| n.model).filter(|n| caps.shader_model < n.model) {
        gap.push(format!("shader model {}.{} < {}.{} ({})", caps.shader_model.0, caps.shader_model.1, n.model.0, n.model.1, n.stage));
    }
    for w in needs.iter().filter_map(|n| n.wave) {
        let line = format!("wave lanes {}..{} lack {w}", caps.wave_lanes.0, caps.wave_lanes.1);
        if !(caps.wave_lanes.0..=caps.wave_lanes.1).contains(&w) && !gap.contains(&line) {
            gap.push(line);
        }
    }
    let all = needs.iter().fold(0, |a, n| a | n.features);
    for (bit, has, what) in [(feature::DOUBLES, caps.double_ops, "no double ops"), (feature::ATOMIC64_TYPED, caps.int64_typed, "no 64-bit atomics on typed resources"), (feature::ATOMIC64_HEAP, caps.int64_heap, "no 64-bit atomics on heap resources"), (feature::BARYCENTRICS, caps.barycentrics, "no barycentrics")] {
        if all & bit != 0 && !has {
            gap.push(what.into());
        }
    }
    if needs.iter().any(|n| n.stage == "ms") && !caps.mesh_shaders {
        gap.push("no mesh shaders".into());
    }
    (!gap.is_empty()).then(|| format!("{}: {}", caps.adapter, gap.join(", ")))
}

/// Root parameter indices of a signature
/// `emit::host::Host::root_signature_records` builds for `stages` with
/// `records` record views: user data of stage k is parameter k, then the
/// root views by (root, SRV), the stride table, the record views, the root
/// SRV lengths, the heap bounds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Params {
    pub views: Vec<(u8, bool, u32)>,
    pub stride: Option<u32>,
    pub records: u32,
    pub lengths: Option<u32>,
    pub bounds: Option<u32>,
}

pub fn params(stages: &[&HostShader], records: u32) -> Params {
    let mut next = stages.len() as u32;
    let mut take = || {
        next += 1;
        next - 1
    };
    let mut views: Vec<(u8, bool)> = stages.iter().flat_map(|s| s.root_cbvs.iter().map(|&r| (r, false)).chain(s.root_srvs.iter().map(|&r| (r, true)))).collect();
    views.sort_unstable();
    views.dedup();
    let views = views.into_iter().map(|(r, srv)| (r, srv, take())).collect();
    let stride = stages.iter().any(|s| s.reads_stride_table()).then(&mut take);
    let first = take();
    for _ in 1..records.max(1) {
        take();
    }
    let lengths = (records > 0 && stages.iter().any(|s| s.stage == HwStage::Ms && !s.root_srvs.is_empty())).then(&mut take);
    let bounds = stages.iter().any(|s| s.reads_heap_bounds()).then(&mut take);
    Params { views, stride, records: first, lengths, bounds }
}

/// The render targets a pixel program writes: (target index, format), from
/// its output signature (`OSG1`: 32-byte elements; `SV_Target` rows by
/// register; float, signed or unsigned 32-bit as the component type says).
pub fn targets(ps: &[u8]) -> Vec<(u32, i32)> {
    let Some(sig) = parts(ps).into_iter().find(|p| &p.0 == b"OSG1").map(|p| p.1) else { return Vec::new() };
    let word = |at: usize| sig.get(at..at + 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let mut out: Vec<(u32, i32)> = (0..word(0) as usize)
        .map(|k| 8 + 32 * k)
        .filter(|&e| {
            let name = word(e + 4) as usize;
            sig.get(name..name + 9).is_some_and(|n| n.eq_ignore_ascii_case(b"SV_Target")) && sig.get(name + 9) == Some(&0)
        })
        .map(|e| (word(e + 20), match word(e + 16) { 3 => DXGI_FORMAT_R32G32B32A32_FLOAT, 2 => DXGI_FORMAT_R32G32B32A32_SINT, _ => DXGI_FORMAT_R32G32B32A32_UINT }.0))
        .filter(|t| t.0 < 8)
        .collect();
    out.sort_unstable();
    out.dedup_by_key(|t| t.0);
    out
}

/// The translated pair of one draw.
#[derive(Clone)]
pub struct Programs {
    /// The mesh program in compute form and its record layout.
    pub ms: HostShader,
    pub layout: emit::mesh::Layout,
    /// The pass-through vertex shader for `ps`.
    pub vs: Vec<u8>,
    pub ps: HostShader,
    /// The compute form's `WaveSize`.
    pub wave: u32,
    /// A plain draw through the title's vertex program instead (the strand
    /// hair draws): `ms` is then that vertex program, drawn as (vertices,
    /// instances, triangle strip) with no compute step.
    pub vertex: Option<(u32, u32, bool)>,
}

impl Programs {
    /// What the three emitted programs need.
    pub fn needs(&self) -> Vec<Needs> {
        if self.vertex.is_some() {
            return [&self.ms.dxil, &self.ps.dxil].into_iter().filter_map(|c| needs(c, None)).collect();
        }
        [(&self.ms.dxil, Some(self.wave)), (&self.vs, None), (&self.ps.dxil, None)].into_iter().filter_map(|(c, w)| needs(c, w)).collect()
    }
}

/// The targets of one run: size, and the draw's depth state (test on,
/// write, D3D12 comparison function; reverse Z clears to 0).
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub width: u32,
    pub height: u32,
    pub depth: Option<(bool, i32)>,
    /// Seconds the submission may take before the run gives up.
    pub limit: f64,
    /// Draw the records into the depth target alone, without the pixel
    /// program (its binding and its targets are not used).
    pub depth_only: bool,
    /// Read the mesh program's records back too ([`Frame::records`]).
    pub records: bool,
}

/// One run read back: target 0 as float RGBA rows ([`CLEAR`] is the value
/// an untouched pixel holds; empty for a depth-only run) and the depth
/// target as rows (`far` where nothing was drawn).
pub struct Frame {
    pub rt0: Vec<[f32; 4]>,
    pub depth: Vec<f32>,
    pub far: f32,
    pub targets: usize,
    pub seconds: f64,
    /// The mesh program's records, `groups * layout.record_bytes()` bytes
    /// (empty unless [`Target::records`]).
    pub records: Vec<u8>,
}

/// Our pass's targets to draw into instead of the run's own (`bridgeps`):
/// its colour target (RGBA8, render-target state) and depth
/// target (depth-write state). Target 0 of the pixel program goes to
/// the colour target; its other targets are not bound (their outputs are
/// dropped). The depth test is GREATER_EQUAL against the pass's depth,
/// without writing it: the title's surface lies 5e-6 nearer than ours,
/// so only surfaces our pass shows take the title's colour.
pub struct Into {
    pub colour: ID3D12Resource,
    pub depth: ID3D12Resource,
}

/// The colour target format of [`Into`].
pub const INTO_COLOUR: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

/// Target 0's clear value: no program writes a negative albedo.
pub const CLEAR: [f32; 4] = [-1.0; 4];

pub(crate) fn barrier(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            // SAFETY: a borrowed pointer the barrier never releases.
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: unsafe { std::mem::transmute_copy(r) }, Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: before, StateAfter: after }),
        },
    }
}

/// The validation layer's stored errors, newest first (at most 4), as a
/// tail for a failure text; empty without the layer.
pub(crate) fn layer_errors(dev: &ID3D12Device) -> String {
    use windows::core::Interface;
    let Ok(queue) = dev.cast::<ID3D12InfoQueue>() else { return String::new() };
    let mut out = String::new();
    // SAFETY: plain queries into buffers of the size the queue asked for.
    unsafe {
        for i in (0..queue.GetNumStoredMessages()).rev() {
            let mut len = 0usize;
            if queue.GetMessage(i, None, &mut len).is_err() || len == 0 {
                continue;
            }
            let mut buffer = vec![0u64; len.div_ceil(8)];
            let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
            if queue.GetMessage(i, Some(message), &mut len).is_err() || (*message).Severity.0 >= D3D12_MESSAGE_SEVERITY_WARNING.0 {
                continue;
            }
            let text = std::slice::from_raw_parts((*message).pDescription, (*message).DescriptionByteLength.saturating_sub(1));
            out.push_str(&format!("; layer: {}", String::from_utf8_lossy(text)));
            if out.matches("; layer: ").count() == 4 {
                break;
            }
        }
    }
    out
}

/// Whether the title's translated programs may run on `host`: the software
/// adapter always; the hardware adapter only as a scheduled card sample
/// (`XNW_BRIDGE_HW=1`) with the per-submission GPU watchdog running
/// (`xnw_gpu::gpuwatch`: `--adapter hw` in the replay, the lab and
/// `xnw-lab head` starts it), so one submission past its budget ends the
/// process before the driver's 2 s timeout ("Card
/// safety"). Every offline submitter of title programs asks this: the
/// draw bridge, bridgeps, bridgems and the compute runner.
pub fn title_programs_here(host: &xnw_gpu::host::Host) -> Result<(), String> {
    if host.caps.software {
        return Ok(());
    }
    if std::env::var("XNW_BRIDGE_HW").ok().as_deref() != Some("1") {
        return Err(format!("the title's translated programs run on the software adapter; {} needs XNW_BRIDGE_HW=1 (a scheduled card sample)", host.caps.adapter));
    }
    if xnw_gpu::gpuwatch::budget().is_none() {
        return Err(format!("the title's translated programs on {} need the per-submission GPU watchdog running (--adapter hw starts it)", host.caps.adapter));
    }
    Ok(())
}

/// Pixel programs (first 32 bits of the container hash) the bridge does not
/// run on the hardware adapter, each with why and the evidence: their rows
/// stay our own pass's there. The software adapter runs them. An entry
/// leaves the list when its cause is fixed and a card sample the user
/// scheduled has shown it.
pub const DENY_HW: &[(u32, &str)] = &[(
    0x3811_0e02,
    "it hung the RX 5700 XT drawn alone in production form (the device hung 0x887a0006 after the pass wait; clean on WARP with GPU-based validation)",
)];

/// Why the bridge leaves the pixel program of container hash `ps` to our
/// own pass: it is in [`DENY_HW`] and the adapter is the hardware one
/// (`software` false). None: it may run. `XNW_BRIDGE_ALLOW=<hash>+<hash>`
/// (first 32 bits, hex) lets a listed program through for one card sample
/// the user has scheduled; nothing else sets it.
pub fn denied(ps: u128, software: bool) -> Option<String> {
    denied_with((ps >> 96) as u32, software, &std::env::var("XNW_BRIDGE_ALLOW").unwrap_or_default())
}

fn denied_with(ps: u32, software: bool, allow: &str) -> Option<String> {
    if software || allow.split('+').filter_map(|h| u32::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()).any(|h| h == ps) {
        return None;
    }
    let why = DENY_HW.iter().find(|d| d.0 == ps)?.1;
    Some(format!("pixel program {ps:08x} is not run on the hardware adapter: {why}"))
}

/// Dispatches the mesh program over `b.groups` groups and draws its records
/// with the pixel program into float targets and a depth target; returns
/// target 0. Refuses an adapter [`gap`] names, and the hardware adapter
/// unless [`title_programs_here`] allows it.
pub fn run(host: &xnw_gpu::host::Host, p: &Programs, b: &Binding, t: &Target) -> Result<Frame, String> {
    run_into(host, p, b, t, None)
}

/// [`run`], into our pass's targets when `into` is given (nothing is read
/// back then: the frame's pictures are empty).
pub fn run_into(host: &xnw_gpu::host::Host, p: &Programs, b: &Binding, t: &Target, into: Option<&Into>) -> Result<Frame, String> {
    // A failure on a removed device carries the validation layer's errors.
    // SAFETY: a plain query.
    submit(host, p, b, t, into).map_err(|e| if unsafe { host.device.GetDeviceRemovedReason() }.is_err() { format!("device removed: {e}{}", layer_errors(&host.device)) } else { e })
}

/// The pipelines of one program pair on a device: the compute form's root
/// signature and pipeline, the draw's (pass-through vertex shader, the
/// pixel program unless depth only). Made once per pair and target shape
/// (made per row they took 0.7 s a row on WARP).
pub struct Pipes {
    /// The compute step (none for a vertex-program draw).
    pub compute: Option<(ID3D12RootSignature, ID3D12PipelineState)>,
    pub cp: Params,
    pub groot: ID3D12RootSignature,
    pub gpso: ID3D12PipelineState,
    pub gp: Params,
    pub pixel: bool,
    pub vertex: Option<(u32, u32, bool)>,
}

impl Pipes {
    /// `formats`: the render targets (index, DXGI format) the draw binds;
    /// `depth`: the depth state (write, D3D12 comparison) or none.
    pub fn new(dev: &ID3D12Device, p: &Programs, pixel: bool, formats: &[(u32, i32)], depth: Option<(bool, i32)>) -> Result<Pipes, String> {
        let e = |e: windows::core::Error| e.message().to_string();
        let eh = emit::host::Host::from_device(dev.clone());
        use emit::mesh::{INDEX_REGISTER, RECORD_REGISTER};
        // A vertex-program draw has no compute step; its vertex stage is the title's program.
        let compute = match p.vertex {
            Some(_) => None,
            None => {
                let croot = eh.root_signature_records(&[(&p.ms, D3D12_SHADER_VISIBILITY_ALL)], &[(true, RECORD_REGISTER), (true, INDEX_REGISTER), (false, RECORD_REGISTER)])?;
                let cpso = eh.compute_pipeline_with(&croot, &p.ms.dxil).map_err(|x| format!("compute pipeline: {x}"))?;
                Some((croot, cpso))
            }
        };
        let passthrough = HostShader { stage: HwStage::Vs, dxil: p.vs.clone(), user_data: Vec::new(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false };
        let vs = if p.vertex.is_some() { &p.ms } else { &passthrough };
        let gstages: Vec<(&HostShader, D3D12_SHADER_VISIBILITY)> = if pixel { vec![(vs, D3D12_SHADER_VISIBILITY_VERTEX), (&p.ps, D3D12_SHADER_VISIBILITY_PIXEL)] } else { vec![(vs, D3D12_SHADER_VISIBILITY_VERTEX)] };
        let groot = eh.root_signature_records(&gstages, &[(false, RECORD_REGISTER)])?;
        let (cp, gp) = (params(&[&p.ms], 3), params(&gstages.iter().map(|s| s.0).collect::<Vec<_>>(), 1));
        let count = formats.iter().map(|f| f.0 as usize + 1).max().unwrap_or(0);
        // SAFETY: a description that outlives the call.
        let gpso = unsafe {
            let mut gdesc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&groot),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.dxil.as_ptr().cast(), BytecodeLength: vs.dxil.len() },
                PS: if pixel { D3D12_SHADER_BYTECODE { pShaderBytecode: p.ps.dxil.as_ptr().cast(), BytecodeLength: p.ps.dxil.len() } } else { D3D12_SHADER_BYTECODE::default() },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: count as u32,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            for rt in &mut gdesc.BlendState.RenderTarget {
                rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            }
            for &(n, f) in formats {
                gdesc.RTVFormats[n as usize] = DXGI_FORMAT(f);
            }
            if let Some((write, func)) = depth {
                gdesc.DepthStencilState = D3D12_DEPTH_STENCIL_DESC { DepthEnable: true.into(), DepthWriteMask: if write { D3D12_DEPTH_WRITE_MASK_ALL } else { D3D12_DEPTH_WRITE_MASK_ZERO }, DepthFunc: D3D12_COMPARISON_FUNC(func), ..Default::default() };
            }
            dev.CreateGraphicsPipelineState::<ID3D12PipelineState>(&gdesc).map_err(|x| format!("graphics pipeline: {}", e(x)))?
        };
        Ok(Pipes { compute, cp, groot, gpso, gp, pixel, vertex: p.vertex })
    }
}

impl Pipes {
    /// UI1: a vertex-program draw (`p.vertex` set: `p.ms` is the title's
    /// vertex program) with the title's input layout `inputs`, its pixel
    /// program, one colour target of `format` blended with `blend`, cull
    /// `cull` and no depth (the UI layer).
    pub fn new_ia(dev: &ID3D12Device, p: &Programs, format: DXGI_FORMAT, blend: D3D12_RENDER_TARGET_BLEND_DESC, cull: D3D12_CULL_MODE, inputs: &[D3D12_INPUT_ELEMENT_DESC]) -> Result<Pipes, String> {
        let e = |e: windows::core::Error| e.message().to_string();
        let eh = emit::host::Host::from_device(dev.clone());
        use emit::mesh::RECORD_REGISTER;
        let gstages: Vec<(&HostShader, D3D12_SHADER_VISIBILITY)> = vec![(&p.ms, D3D12_SHADER_VISIBILITY_VERTEX), (&p.ps, D3D12_SHADER_VISIBILITY_PIXEL)];
        let groot = eh.root_signature_records(&gstages, &[(false, RECORD_REGISTER)])?;
        let (cp, gp) = (params(&[&p.ms], 3), params(&gstages.iter().map(|s| s.0).collect::<Vec<_>>(), 1));
        // SAFETY: a description that outlives the call.
        let gpso = unsafe {
            let mut gdesc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&groot),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: p.ms.dxil.as_ptr().cast(), BytecodeLength: p.ms.dxil.len() },
                PS: D3D12_SHADER_BYTECODE { pShaderBytecode: p.ps.dxil.as_ptr().cast(), BytecodeLength: p.ps.dxil.len() },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: cull, DepthClipEnable: true.into(), ..Default::default() },
                InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: inputs.as_ptr(), NumElements: inputs.len() as u32 },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            gdesc.BlendState.RenderTarget[0] = blend;
            gdesc.RTVFormats[0] = format;
            dev.CreateGraphicsPipelineState::<ID3D12PipelineState>(&gdesc).map_err(|x| format!("graphics pipeline: {}", e(x)))?
        };
        Ok(Pipes { compute: None, cp, groot, gpso, gp, pixel: true, vertex: p.vertex })
    }
}

/// UI1: records an IA draw of the uploaded binding `u` with the pipes of
/// [`Pipes::new_ia`]: vertex buffers, an index buffer (DrawIndexed) or
/// none (Draw: `count` vertices from `first`), `instances`, `base` vertex.
/// Targets, viewport and scissor are the caller's.
#[allow(clippy::too_many_arguments)]
pub fn record_ia_draw(list: &ID3D12GraphicsCommandList, pipes: &Pipes, u: &Uploaded, records: u64, vbs: &[D3D12_VERTEX_BUFFER_VIEW], ibv: Option<&D3D12_INDEX_BUFFER_VIEW>, (count, instances, first, base): (u32, u32, u32, i32), strip: bool) {
    let (gp, g) = (&pipes.gp, &u.bounds);
    // SAFETY: as in record_dispatch.
    unsafe {
        list.SetDescriptorHeaps(&[Some(u.views.clone()), Some(u.samplers.clone())]);
        list.SetPipelineState(&pipes.gpso);
        list.SetGraphicsRootSignature(&pipes.groot);
        list.SetGraphicsRoot32BitConstants(0, u.mesh_constants.len() as u32, u.mesh_constants.as_ptr().cast(), 0);
        list.SetGraphicsRoot32BitConstants(1, u.pixel_constants.len() as u32, u.pixel_constants.as_ptr().cast(), 0);
        for &(param, is_srv, va) in &u.pixel_views {
            if is_srv { list.SetGraphicsRootShaderResourceView(u32::from(param), va) } else { list.SetGraphicsRootConstantBufferView(u32::from(param), va) }
        }
        if let Some(param) = gp.stride {
            list.SetGraphicsRootShaderResourceView(param, u.stride_table.GetGPUVirtualAddress());
        }
        list.SetGraphicsRootShaderResourceView(gp.records, records);
        if let Some(param) = gp.bounds {
            list.SetGraphicsRoot32BitConstants(param, 8, g.as_ptr().cast(), 0);
        }
        use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP};
        list.IASetPrimitiveTopology(if strip { D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP } else { D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST });
        if !vbs.is_empty() {
            list.IASetVertexBuffers(0, Some(vbs));
        }
        match ibv {
            Some(ibv) => {
                list.IASetIndexBuffer(Some(ibv));
                list.DrawIndexedInstanced(count, instances.max(1), 0, base, 0);
            }
            None => list.DrawInstanced(count, instances.max(1), first, 0),
        }
    }
}

/// One row's binding on the device ([`upload`]): its view and sampler
/// heaps, the buffers behind them, the stride table, the zero first-group
/// word, and the root views' addresses per stage.
pub struct Uploaded {
    pub views: ID3D12DescriptorHeap,
    pub samplers: ID3D12DescriptorHeap,
    stride_table: ID3D12Resource,
    base: ID3D12Resource,
    mesh_views: Vec<(u8, bool, u64)>,
    pixel_views: Vec<(u8, bool, u64)>,
    /// The binding's arguments the recording needs.
    mesh_constants: Vec<u32>,
    mesh_lengths: Vec<u32>,
    pixel_constants: Vec<u32>,
    bounds: [u32; 8],
    pub groups: u32,
    /// Bytes put on the device for this row.
    pub bytes: u64,
    _keep: Vec<ID3D12Resource>,
}

/// Buffers and textures the rows of one batch share by content: frame
/// tables and textures many rows name are put on the device once (a
/// batch of 1,000 rows that each copied them ran WARP out of memory).
/// A large buffer (a frame table's copy, megabytes) differs between rows
/// only where `load` wrote a row's own words (draw words, bone mask words
/// at the row's record): such rows share one buffer holding every row's
/// words ([`Shared::merge`]).
#[derive(Default)]
pub struct Shared {
    bufs: HashMap<(u64, usize), ID3D12Resource>,
    texs: HashMap<(u64, u32, u32, u32), ID3D12Resource>,
    large: HashMap<u64, Vec<Merged>>,
    /// Bytes not put on the device again; rows merged into a large buffer.
    pub saved: u64,
    pub merged: u64,
    /// New bytes by kind: large buffer views (not merged), small buffer
    /// views, textures, root views.
    pub new: [u64; 4],
    /// `XNW_ZEROCOPY` (ZC2): the title address of a view's bytes, by the
    /// bytes' address (the caller's, per row). A view with one is bound in
    /// place in title memory where its page window is open, else on one
    /// buffer per title address that every row's view of it shares (each
    /// row's own words written into it: the title's memory holds them all
    /// at once). Views of a row without an entry take the paths above.
    pub addrs: HashMap<usize, TitleView>,
    /// The frame's title spans (start, end) in layers and their buffers by
    /// (layer, start).
    spans: Vec<Vec<(u64, u64)>>,
    by_addr: HashMap<(usize, u64), AtAddr>,
    /// Views bound in place in title memory, on a buffer per title address
    /// (and its bytes), rows refused one for a byte two rows set
    /// differently, buffers made again larger.
    pub in_place: u64,
    pub by_addr_views: u64,
    pub by_addr_bytes: u64,
    pub by_addr_conflicts: u64,
    /// Large views (at least [`KEEP_MIN`], not constants) not on a span, by
    /// why (count, bytes): no title address, outside every span, a byte
    /// another row holds differently, the span offset not a multiple of the
    /// view's unit (ZC3).
    pub off_span: [(u64, u64); 4],
    /// Bytes of a row's copy another row's own words shadow on its span
    /// (its view reads theirs there), and bytes a row's own words wrote over
    /// other rows' copies.
    pub shadowed: u64,
    pub overrode: u64,
    /// `XNW_ZEROCOPY_SPANCHECK=1`: every view on a span (row `row` set by the
    /// caller, slot, span, offset, 4 KiB block hashes of its title bytes),
    /// compared with the spans' final bytes by [`Shared::span_check`].
    pub check: Option<Vec<SpanView>>,
    pub row: u32,
}

/// What of a view's bytes is title memory ([`Shared::addrs`], ZC3): the
/// title address of byte 0, the bytes copied from there (`captured`, from
/// byte 0), and the ranges `load` wrote over or past the copy (the row's own
/// bones, draw words, bone mask words: `overlays`). Every other byte is
/// padding: zero in the row's copy and no one's title memory, so it is not
/// written to a span and does not conflict (ZC2 wrote it: a row's padding
/// zeros patched over another row's bones, 8 rows of pk3 f1241 drew nothing).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TitleView {
    pub addr: u64,
    pub captured: usize,
    pub overlays: Vec<(usize, usize)>,
}

impl TitleView {
    /// The view's title bytes in `len`: (start, end, own words) in order,
    /// disjoint; own words win where they lie over the copy.
    pub fn parts(&self, len: usize) -> Vec<(usize, usize, bool)> {
        let mut own: Vec<(usize, usize)> = self.overlays.iter().map(|&(a, b)| (a.min(len), b.min(len))).filter(|r| r.0 < r.1).collect();
        own.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for r in own {
            match merged.last_mut() {
                Some(l) if r.0 <= l.1 => l.1 = l.1.max(r.1),
                _ => merged.push(r),
            }
        }
        let (mut out, mut at, cap) = (Vec::new(), 0, self.captured.min(len));
        for &(a, b) in &merged {
            if at < a.min(cap) {
                out.push((at, a.min(cap), false));
            }
            out.push((a, b, true));
            at = b;
        }
        if at < cap {
            out.push((at, cap, false));
        }
        out
    }
}

/// Title spans for `ranges` (address, bytes, the view's unit), in layers:
/// overlapping ranges share a span whose start puts every range at an
/// offset that is a multiple of its unit (and of 16 for raw and
/// structured views, which may have a raw twin); a range no start of the
/// span it overlaps can hold so goes to the next layer (ZC3: pk3 f1241's
/// raw views at addresses 4 mod 16 and 48-byte structured views).
pub fn plan_spans(mut ranges: Vec<(u64, u64, u64)>) -> Vec<Vec<(u64, u64)>> {
    // Congruence x = r (mod m) joined with x = r2 (mod m2), if any solves both.
    fn join(r: u64, m: u64, r2: u64, m2: u64) -> Option<(u64, u64)> {
        let l = m / gcd(m, m2) * m2;
        if l > 1 << 16 {
            return None;
        }
        (0..m2 / gcd(m, m2)).map(|k| r + k * m).find(|x| x % m2 == r2).map(|x| (x % l, l))
    }
    ranges.sort_unstable();
    let mut layers = Vec::new();
    while !ranges.is_empty() {
        // (first address, end, residue, modulus)
        let mut spans: Vec<(u64, u64, u64, u64)> = Vec::new();
        let mut next = Vec::new();
        for (a, n, unit) in ranges {
            let u = if unit.is_multiple_of(16) || unit < 4 { unit.max(1) } else { unit / gcd(unit, 16) * 16 };
            // Room for a view's extent rounding past its bytes.
            let e = a + n.next_multiple_of(unit.max(1)) + 256;
            match spans.last_mut() {
                Some(l) if a <= l.1 => match join(l.2, l.3, a % u, u) {
                    Some((r, m)) => (l.1, l.2, l.3) = (l.1.max(e), r, m),
                    None => next.push((a, n, unit)),
                },
                _ => spans.push((a, e, a % u, u)),
            }
        }
        layers.push(spans.iter().map(|&(lo, hi, r, m)| (lo - (lo + m - r) % m, hi)).collect());
        ranges = next;
    }
    layers
}

/// One view on a span buffer, for [`Shared::span_check`].
pub struct SpanView {
    pub row: u32,
    pub slot: u32,
    pub addr: u64,
    start: (usize, u64),
    off: usize,
    parts: Vec<(usize, usize, bool)>,
    blocks: Vec<u64>,
}

/// One title span's buffer ([`Shared::addrs`]): the bytes now, which bytes
/// a row wrote and which of them are a row's own words, the buffer (mapped).
struct AtAddr {
    now: Vec<u8>,
    written: Vec<u64>,
    own: Vec<u64>,
    res: ID3D12Resource,
    mapped: *mut u8,
}

/// 4 KiB block hashes of `parts` of `b` (a word sum: a diagnostic, not a key).
fn block_hashes(b: &[u8], parts: &[(usize, usize, bool)]) -> Vec<u64> {
    let mut out = Vec::new();
    for &(s, e, _) in parts {
        for c in b[s..e].chunks(4096) {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for w in c.chunks(8) {
                let mut x = [0u8; 8];
                x[..w.len()].copy_from_slice(w);
                h = (h ^ u64::from_le_bytes(x)).wrapping_mul(0x0100_0000_01b3).rotate_left(29);
            }
            out.push(h);
        }
    }
    out
}

struct Merged {
    first: Vec<u8>,
    now: Vec<u8>,
    res: ID3D12Resource,
    mapped: *mut u8,
}

/// Buffers from this size on are merged, smaller ones shared by content.
pub const MERGE_BYTES: usize = 1 << 20;

impl Shared {
    /// A device buffer of `size` holding `bytes` for this row: one already
    /// made whose bytes differ from these in under 1 % (and only where no
    /// other row changed them: those bytes are written into it), else a
    /// new one.
    fn merge(&mut self, dev: &ID3D12Device, bytes: &[u8], size: u64) -> Result<(ID3D12Resource, bool), String> {
        let list = self.large.entry(size).or_default();
        'next: for m in list.iter_mut() {
            let mut patch = Vec::new();
            let mut differ = 0usize;
            for (k, (a, b)) in m.now.chunks(64).zip(bytes.chunks(64)).enumerate() {
                if a == b {
                    continue;
                }
                let f = &m.first[64 * k..64 * k + a.len()];
                for (j, ((&now, &new), &first)) in a.iter().zip(b).zip(f).enumerate() {
                    if now == new || new == first {
                        continue;
                    }
                    // Changed by another row already: another value here is a conflict.
                    if now != first {
                        continue 'next;
                    }
                    differ += 1;
                    patch.push((64 * k + j, new));
                }
                if differ * 100 > bytes.len() {
                    continue 'next;
                }
            }
            for &(at, v) in &patch {
                m.now[at] = v;
                // SAFETY: inside the mapped upload buffer (`at` < its size).
                unsafe { *m.mapped.add(at) = v };
            }
            self.merged += 1;
            return Ok((m.res.clone(), true));
        }
        let res = upload_bytes(dev, bytes, size)?;
        let mut mapped = std::ptr::null_mut();
        // SAFETY: an upload buffer stays mapped for its life.
        unsafe { res.Map(0, None, Some(&mut mapped)) }.map_err(|e| e.message().to_string())?;
        list.push(Merged { first: bytes.to_vec(), now: bytes.to_vec(), res: res.clone(), mapped: mapped.cast() });
        Ok((res, false))
    }
}

impl Shared {
    /// Declares the title spans this frame's rows read, in layers (each
    /// sorted and disjoint; `bridge_ms` gathers them first): a view inside
    /// one is bound on that span's one buffer, at its title offset.
    pub fn use_spans(&mut self, layers: Vec<Vec<(u64, u64)>>) {
        self.spans = layers;
    }

    /// The span buffer holding title range `[addr, addr + size)` and the
    /// view's offset there (a multiple of `unit`: the first layer whose span
    /// holds the range at such an offset and no byte of it differently),
    /// with this row's title bytes ([`TitleView::parts`]) written where no
    /// row wrote yet (None: outside every span, or a byte another row holds
    /// differently, a conflict one buffer cannot hold: the row keeps a
    /// buffer of its own). A row's own words win over another row's copy
    /// there (`shadowed`, `overrode`); padding is never written, so every row
    /// reads its own title bytes. Whether the buffer was made now.
    fn at_addr(&mut self, dev: &ID3D12Device, tv: &TitleView, bytes: &[u8], size: u64, slot: u32, unit: u64) -> Result<Option<(ID3D12Resource, u64, bool)>, String> {
        let addr = tv.addr;
        let (mut misaligned, mut conflict) = (None, false);
        for layer in 0..self.spans.len() {
            let spans = &self.spans[layer];
            let i = spans.partition_point(|s| s.0 <= addr);
            let Some(&(start, end)) = i.checked_sub(1).and_then(|i| spans.get(i)) else { continue };
            if addr + size > end {
                continue;
            }
            if !(addr - start).is_multiple_of(unit) {
                misaligned.get_or_insert(addr - start);
                continue;
            }
            match self.on_span(dev, (layer, start), end, tv, bytes, slot)? {
                Some(r) => return Ok(Some(r)),
                None => conflict = true,
            }
        }
        let why = if conflict {
            self.by_addr_conflicts += 1;
            2
        } else if let Some(off) = misaligned {
            if self.check.is_some() {
                xnw_core::info!("bridgems span check: row {} heap slot {slot} at {addr:#x} ({} B): offset {off:#x} on the span, unit {unit}", self.row, bytes.len());
            }
            3
        } else {
            1
        };
        self.off_span[why].0 += 1;
        self.off_span[why].1 += size;
        Ok(None)
    }

    /// [`Shared::at_addr`] on span `key` (layer, start) ending at `end`.
    fn on_span(&mut self, dev: &ID3D12Device, key: (usize, u64), end: u64, tv: &TitleView, bytes: &[u8], slot: u32) -> Result<Option<(ID3D12Resource, u64, bool)>, String> {
        let (addr, start) = (tv.addr, key.1);
        let parts = tv.parts(bytes.len());
        let made = !self.by_addr.contains_key(&key);
        if made {
            let len = (end - start) as usize;
            let res = buffer(dev, len as u64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let mut mapped = std::ptr::null_mut();
            // SAFETY: an upload buffer stays mapped for its life.
            unsafe { res.Map(0, None, Some(&mut mapped)) }.map_err(|e| e.message().to_string())?;
            let bits = len.div_ceil(64);
            self.by_addr.insert(key, AtAddr { now: vec![0; len], written: vec![0; bits], own: vec![0; bits], res, mapped: mapped.cast() });
        }
        let m = self.by_addr.get_mut(&key).expect("made above");
        let off = (addr - start) as usize;
        let bit = |v: &[u64], x: usize| v[x / 64] >> (x % 64) & 1 == 1;
        // Pass 1: any byte another row holds differently where neither or
        // both are own words is a conflict; nothing is written then.
        let (mut shadowed, mut overrode) = (0u64, 0u64);
        for &(s, e, own) in &parts {
            for (k, chunk) in bytes[s..e].chunks(64).enumerate() {
                let at = off + s + 64 * k;
                if at % 64 == 0 && chunk.len() == 64 && m.written[at / 64] == u64::MAX && m.now[at..at + 64] == *chunk {
                    continue;
                }
                for (j, &new) in chunk.iter().enumerate() {
                    let x = at + j;
                    if !bit(&m.written, x) || m.now[x] == new {
                        continue;
                    }
                    match (bit(&m.own, x), own) {
                        (true, false) => shadowed += 1,
                        (false, true) => overrode += 1,
                        _ => {
                            if self.check.is_some() {
                                xnw_core::info!("bridgems span check: row {} heap slot {slot} at {addr:#x} ({} B): byte +{:#x} is {:#x} here, {:#x} on the span (layer {})", self.row, bytes.len(), x - off, new, m.now[x], key.0);
                            }
                            return Ok(None);
                        }
                    }
                }
            }
        }
        // Pass 2: write.
        for &(s, e, own) in &parts {
            for (k, chunk) in bytes[s..e].chunks(64).enumerate() {
                let at = off + s + 64 * k;
                if !own && at % 64 == 0 && chunk.len() == 64 && m.written[at / 64] == u64::MAX && m.now[at..at + 64] == *chunk {
                    continue;
                }
                for (j, &new) in chunk.iter().enumerate() {
                let x = at + j;
                let fresh = !bit(&m.written, x);
                if !fresh && (m.now[x] == new || (bit(&m.own, x) && !own)) {
                    if own {
                        m.own[x / 64] |= 1 << (x % 64);
                    }
                    continue;
                }
                m.written[x / 64] |= 1 << (x % 64);
                if own {
                    m.own[x / 64] |= 1 << (x % 64);
                }
                m.now[x] = new;
                // SAFETY: inside the mapped upload buffer (`x` < its size).
                unsafe { *m.mapped.add(x) = new };
                }
            }
        }
        self.shadowed += shadowed;
        self.overrode += overrode;
        if let Some(c) = self.check.as_mut() {
            c.push(SpanView { row: self.row, slot, addr, start: key, off, blocks: block_hashes(bytes, &parts), parts });
        }
        Ok(Some((m.res.clone(), off as u64, made)))
    }

    /// `XNW_ZEROCOPY_SPANCHECK=1`: the views whose title bytes on their span
    /// are not their own at the end of the frame: (row, slot, title address,
    /// blocks differing of all, first differing byte, whether in the row's
    /// own words).
    pub fn span_check(&self) -> Vec<(u32, u32, u64, usize, usize, usize, bool)> {
        let mut out = Vec::new();
        for v in self.check.iter().flatten() {
            let Some(m) = self.by_addr.get(&v.start) else { continue };
            let now = block_hashes(&m.now[v.off..], &v.parts);
            let differ: Vec<usize> = now.iter().zip(&v.blocks).enumerate().filter(|(_, (a, b))| a != b).map(|(i, _)| i).collect();
            if let Some(&first) = differ.first() {
                // Block `first` -> its byte in the view.
                let (mut i, mut at, mut own) = (first, 0, false);
                for &(s, e, o) in &v.parts {
                    let n = (e - s).div_ceil(4096);
                    if i < n {
                        (at, own) = (s + 4096 * i, o);
                        break;
                    }
                    i -= n;
                }
                out.push((v.row, v.slot, v.addr, differ.len(), v.blocks.len(), at, own));
            }
        }
        out
    }
}

fn content(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::hash::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}

pub(crate) fn committed(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES, clear: Option<*const D3D12_CLEAR_VALUE>) -> Result<ID3D12Resource, String> {
    let mut r: Option<ID3D12Resource> = None;
    // SAFETY: descriptions that outlive the call.
    unsafe { dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, desc, state, clear, &mut r) }.map_err(|e| e.message().to_string())?;
    r.ok_or_else(|| "no resource".to_string())
}

/// A buffer of `size` bytes (at least 256).
pub fn buffer(dev: &ID3D12Device, size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> Result<ID3D12Resource, String> {
    committed(dev, &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size.max(256), Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: flags, ..Default::default() }, heap, state, None)
}

pub(crate) fn upload_bytes(dev: &ID3D12Device, bytes: &[u8], size: u64) -> Result<ID3D12Resource, String> {
    let r = buffer(dev, size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
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

/// Puts binding `b` on the device for pipelines `pipes`: the view heap
/// (typed nulls, guards, buffer views), the sampler heap, the textures
/// (copies recorded on `list`), the root views. Nothing else is recorded.
pub fn upload(dev: &ID3D12Device, list: &ID3D12GraphicsCommandList, b: &Binding, pipes: &Pipes, shared: &mut Shared) -> Result<Uploaded, String> {
    let e = |e: windows::core::Error| e.message().to_string();
    let mut bytes = 0u64;
    // SAFETY: D3D12 calls with descriptions that outlive them; the
    // resources are kept in the result until the caller's GPU work is done.
    unsafe {
        let heap = |ty, n: u32, flags| -> Result<ID3D12DescriptorHeap, String> { dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: n.max(1), Flags: flags, ..Default::default() }).map_err(e) };
        let visible = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        let views = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, b.views.heap_slots, visible)?;
        let samplers = heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, b.sampler_slots, visible)?;
        let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let at = |slot: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: views.GetCPUDescriptorHandleForHeapStart().ptr + slot as usize * step };
        let srv_at = |first: u64, format, n: u32, stride: u32, flags| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_BUFFER, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: first, NumElements: n, StructureByteStride: stride, Flags: flags } } };
        let srv = |format, n: u32, stride: u32, flags| srv_at(0, format, n, stride, flags);
        let tex2d = |format| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } } };
        // Typed nulls everywhere (raw buffers), the guards' kinds, then the views.
        for s in 0..b.views.heap_slots {
            dev.CreateShaderResourceView(None, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s));
        }
        let g = &b.views.bounds;
        dev.CreateShaderResourceView(None, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(g[1]));
        let raw_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R32_TYPELESS, ViewDimension: D3D12_UAV_DIMENSION_BUFFER, Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } } };
        dev.CreateUnorderedAccessView(None, None, Some(&raw_uav), at(g[3]));
        dev.CreateUnorderedAccessView(None, None, Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() }), at(g[4]));
        dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at(g[5]));
        let mut keep = Vec::new();
        for &(s, v) in &b.views.slots {
            // An empty slot keeps its null view (`Kind::extent`); so does an
            // output buffer, which a draw's binding carries without bytes.
            if let Kind::RawUav { .. } = v.kind {
                continue;
            }
            let Some((size, n)) = v.kind.extent(v.bytes.len() as u64) else { continue };
            // `XNW_ZEROCOPY` (ZC2): a view of title memory, by its address.
            // Constant buffers stay copies (hot data: every lane reads them).
            let large = v.kind != Kind::Cbv && size >= KEEP_MIN as u64 && xnw_gpu::shared::world_part(xnw_gpu::shared::WORLD_BRIDGE);
            let tv = if large { shared.addrs.get(&(v.bytes.as_ptr() as usize)).cloned() } else { None };
            if large && tv.is_none() {
                shared.off_span[0].0 += 1;
                shared.off_span[0].1 += size;
                if shared.check.is_some() {
                    xnw_core::info!("bridgems span check: row {} heap slot {s}: {:?} {} B at {:#x} without a title address", shared.row, v.kind, v.bytes.len(), v.bytes.as_ptr() as usize);
                }
            }
            let addr = tv.as_ref().map(|t| t.addr);
            if let Some(a) = addr
                && let Some(t) = xnw_gpu::shared::world_range(a, size).filter(|t| t.offset.is_multiple_of(unit(v.kind)))
            {
                let first = t.offset / unit(v.kind);
                match v.kind {
                    Kind::Raw => dev.CreateShaderResourceView(&t.buffer, Some(&srv_at(t.offset / 4, DXGI_FORMAT_R32_TYPELESS, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s)),
                    Kind::Structured(stride) => dev.CreateShaderResourceView(&t.buffer, Some(&srv_at(t.offset / u64::from(stride.max(4)), DXGI_FORMAT_UNKNOWN, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                    Kind::Typed { format, .. } => dev.CreateShaderResourceView(&t.buffer, Some(&srv_at(first, DXGI_FORMAT(format), n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                    Kind::Cbv | Kind::RawUav { .. } => {}
                }
                shared.in_place += 1;
                keep.push(t.buffer);
                continue;
            }
            let by_addr = match &tv {
                Some(t) => shared.at_addr(dev, t, v.bytes, size, s, unit(v.kind))?,
                None => None,
            };
            if let Some((r, off, made)) = by_addr {
                if made {
                    let span = r.GetDesc().Width;
                    bytes += span;
                    shared.new[0] += span;
                    shared.by_addr_bytes += span;
                }
                shared.saved += size;
                shared.by_addr_views += 1;
                let first = off / unit(v.kind);
                match v.kind {
                    Kind::Raw => dev.CreateShaderResourceView(&r, Some(&srv_at(off / 4, DXGI_FORMAT_R32_TYPELESS, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s)),
                    Kind::Structured(stride) => dev.CreateShaderResourceView(&r, Some(&srv_at(off / u64::from(stride.max(4)), DXGI_FORMAT_UNKNOWN, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                    Kind::Typed { format, .. } => dev.CreateShaderResourceView(&r, Some(&srv_at(first, DXGI_FORMAT(format), n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                    Kind::Cbv | Kind::RawUav { .. } => {}
                }
                keep.push(r);
                continue;
            }
            let r = if v.bytes.len() >= MERGE_BYTES {
                let (r, old) = shared.merge(dev, v.bytes, size)?;
                if old {
                    shared.saved += size
                } else {
                    bytes += size;
                    shared.new[0] += size;
                }
                r
            } else {
                let key = (content(v.bytes), size as usize);
                match shared.bufs.get(&key) {
                    Some(r) => {
                        shared.saved += size;
                        r.clone()
                    }
                    None => {
                        let r = upload_bytes(dev, v.bytes, size)?;
                        bytes += size;
                        shared.new[1] += size;
                        shared.bufs.insert(key, r.clone());
                        r
                    }
                }
            };
            match v.kind {
                Kind::Raw => dev.CreateShaderResourceView(&r, Some(&srv(DXGI_FORMAT_R32_TYPELESS, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s)),
                Kind::Structured(stride) => dev.CreateShaderResourceView(&r, Some(&srv(DXGI_FORMAT_UNKNOWN, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                Kind::Typed { format, .. } => dev.CreateShaderResourceView(&r, Some(&srv(DXGI_FORMAT(format), n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                Kind::Cbv => dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: r.GetGPUVirtualAddress(), SizeInBytes: n }), at(s)),
                Kind::RawUav { .. } => {}
            }
            keep.push(r);
        }
        // Samplers: one default (linear, wrap) at every slot, the captured
        // entries (v2.4) at their indices.
        let sstep = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
        let wrap = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        let sat = |s: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: samplers.GetCPUDescriptorHandleForHeapStart().ptr + s * sstep };
        for s in 0..b.sampler_slots as usize {
            dev.CreateSampler(&D3D12_SAMPLER_DESC { Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR, AddressU: wrap, AddressV: wrap, AddressW: wrap, MaxLOD: f32::MAX, ..Default::default() }, sat(s));
        }
        for (s, desc) in b.samplers.iter().filter(|s| s.0 < b.sampler_slots) {
            dev.CreateSampler(desc, sat(*s as usize));
        }
        // Textures: one mip each, copied from an upload buffer.
        let shader_read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        // A texture takes the resource and the view of its shape (1D, 3D,
        // a 2D array, a cube; `textures::provided_shape`); the shared key
        // keeps the shapes apart.
        let view_of = |tx: &Provided| crate::textures::provided_shape(tx).2;
        for &(s, tx) in &b.textures {
            let key = (content(&tx.rows), tx.width, tx.height * tx.depth.max(1), tx.format | u32::from(tx.dim) << 24 | (tx.mips.clamp(1, 15) - 1) << 28);
            if let Some(r) = shared.texs.get(&key) {
                shared.saved += tx.rows.len() as u64;
                dev.CreateShaderResourceView(r, Some(&view_of(tx)), at(s));
                continue;
            }
            let (desc, subs, _) = crate::textures::provided_shape(tx);
            let r = committed(dev, &desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, None)?;
            // One subresource a slice and mip (a 3D texture: one a mip).
            let (fps, rows, row_size, total) = crate::textures::footprints(dev, &desc, subs);
            let up = buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            bytes += total;
            shared.new[2] += total;
            let mut to = std::ptr::null_mut();
            up.Map(0, None, Some(&mut to)).map_err(e)?;
            crate::textures::provided_fill(tx, &fps, &rows, &row_size, to.cast());
            up.Unmap(0, None);
            for (i, fp) in fps.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            list.ResourceBarrier(&[barrier(&r, D3D12_RESOURCE_STATE_COPY_DEST, shader_read)]);
            dev.CreateShaderResourceView(&r, Some(&view_of(tx)), at(s));
            shared.texs.insert(key, r.clone());
            keep.extend([r, up]);
        }
        let base = upload_bytes(dev, &[0; 16], 16)?;
        let strides: Vec<u8> = b.views.strides.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        let stride_table = upload_bytes(dev, &strides, strides.len() as u64)?;
        let mut root_views = |stage: &Stage, params: &[(u8, bool, u32)], name: &str| -> Result<Vec<(u8, bool, u64)>, String> {
            let mut out = Vec::new();
            for &(r, is_srv, param) in params {
                let v = stage.views.iter().find(|v| v.0 == r && v.1 == is_srv).ok_or_else(|| format!("{name}: root view {r} not bound"))?;
                let size = (v.2.len() as u64).next_multiple_of(256);
                let key = (content(v.2), size as usize);
                let buf = match shared.bufs.get(&key) {
                    Some(r) => {
                        shared.saved += size;
                        r.clone()
                    }
                    None => {
                        let r = upload_bytes(dev, v.2, size)?;
                        bytes += size;
                        shared.new[3] += size;
                        shared.bufs.insert(key, r.clone());
                        r
                    }
                };
                out.push((param as u8, is_srv, buf.GetGPUVirtualAddress()));
                keep.push(buf);
            }
            Ok(out)
        };
        let mesh_views = root_views(&b.mesh, &pipes.cp.views, "mesh")?;
        // A vertex-program draw's vertex stage is the record's mesh stage.
        let both = Stage { views: b.pixel.views.iter().chain(if pipes.vertex.is_some() { &b.mesh.views[..] } else { &[] }).copied().collect(), ..Stage::default() };
        let pixel_views = root_views(&both, &pipes.gp.views, "pixel")?;
        let mut mesh_lengths = b.mesh.lengths.clone();
        mesh_lengths.resize(mesh_lengths.len().next_multiple_of(4).max(4), 0);
        Ok(Uploaded { views, samplers, stride_table, base, mesh_views, pixel_views, mesh_constants: b.mesh.constants.clone(), mesh_lengths, pixel_constants: b.pixel.constants.clone(), bounds: b.views.bounds, groups: b.groups, bytes, _keep: keep })
    }
}

/// Records the mesh program's dispatch of the uploaded binding `u` (heaps,
/// root signature, pipeline, arguments) writing its records and index
/// triples at the two addresses. No barrier.
pub fn record_dispatch(list: &ID3D12GraphicsCommandList, pipes: &Pipes, u: &Uploaded, records: u64, indices: u64) {
    let (cp, g) = (&pipes.cp, &u.bounds);
    let Some((croot, cpso)) = &pipes.compute else { return };
    // SAFETY: recording with live objects; the caller keeps them until the GPU is done.
    unsafe {
        list.SetDescriptorHeaps(&[Some(u.views.clone()), Some(u.samplers.clone())]);
        list.SetComputeRootSignature(croot);
        list.SetPipelineState(cpso);
        list.SetComputeRoot32BitConstants(0, u.mesh_constants.len() as u32, u.mesh_constants.as_ptr().cast(), 0);
        for &(param, is_srv, va) in &u.mesh_views {
            if is_srv { list.SetComputeRootShaderResourceView(u32::from(param), va) } else { list.SetComputeRootConstantBufferView(u32::from(param), va) }
        }
        if let Some(param) = cp.stride {
            list.SetComputeRootShaderResourceView(param, u.stride_table.GetGPUVirtualAddress());
        }
        list.SetComputeRootUnorderedAccessView(cp.records, records);
        list.SetComputeRootUnorderedAccessView(cp.records + 1, indices);
        list.SetComputeRootShaderResourceView(cp.records + 2, u.base.GetGPUVirtualAddress());
        if let Some(param) = cp.lengths {
            list.SetComputeRoot32BitConstants(param, u.mesh_lengths.len() as u32, u.mesh_lengths.as_ptr().cast(), 0);
        }
        if let Some(param) = cp.bounds {
            list.SetComputeRoot32BitConstants(param, 8, g.as_ptr().cast(), 0);
        }
        list.Dispatch(u.groups, 1, 1);
    }
}

/// Records the draw of the uploaded binding's records (at `records`, index
/// triples in `ibv`) with the pass-through vertex shader and the pixel
/// program: heaps, root signature, pipeline, arguments. Targets, viewport
/// and topology are the caller's.
pub fn record_draw(list: &ID3D12GraphicsCommandList, pipes: &Pipes, u: &Uploaded, records: u64, ibv: &D3D12_INDEX_BUFFER_VIEW, max_prims: u32) {
    let (gp, g) = (&pipes.gp, &u.bounds);
    // SAFETY: as in record_dispatch.
    unsafe {
        list.SetDescriptorHeaps(&[Some(u.views.clone()), Some(u.samplers.clone())]);
        list.SetPipelineState(&pipes.gpso);
        list.SetGraphicsRootSignature(&pipes.groot);
        if pipes.vertex.is_some() {
            list.SetGraphicsRoot32BitConstants(0, u.mesh_constants.len() as u32, u.mesh_constants.as_ptr().cast(), 0);
        } else {
            list.SetGraphicsRoot32BitConstants(0, 1, [0u32].as_ptr().cast(), 0);
        }
        if pipes.pixel {
            list.SetGraphicsRoot32BitConstants(1, u.pixel_constants.len() as u32, u.pixel_constants.as_ptr().cast(), 0);
        }
        for &(param, is_srv, va) in &u.pixel_views {
            if is_srv { list.SetGraphicsRootShaderResourceView(u32::from(param), va) } else { list.SetGraphicsRootConstantBufferView(u32::from(param), va) }
        }
        if let Some(param) = gp.stride {
            list.SetGraphicsRootShaderResourceView(param, u.stride_table.GetGPUVirtualAddress());
        }
        list.SetGraphicsRootShaderResourceView(gp.records, records);
        if let Some(param) = gp.bounds {
            list.SetGraphicsRoot32BitConstants(param, 8, g.as_ptr().cast(), 0);
        }
        if let Some((vertices, instances, strip)) = pipes.vertex {
            use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP};
            list.IASetPrimitiveTopology(if strip { D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP } else { D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST });
            list.DrawInstanced(vertices, instances, 0, 0);
        } else {
            list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.IASetIndexBuffer(Some(ibv));
            list.DrawIndexedInstanced(u.groups * max_prims * 3, 1, 0, 0, 0);
        }
    }
}

fn submit(host: &xnw_gpu::host::Host, p: &Programs, b: &Binding, t: &Target, into: Option<&Into>) -> Result<Frame, String> {
    title_programs_here(host)?;
    if let Some(gap) = gap(&host.caps, &p.needs()) {
        return Err(gap);
    }
    let e = |e: windows::core::Error| e.message().to_string();
    let dev = &host.device;
    let pixel = !t.depth_only;
    let formats = if pixel { targets(&p.ps.dxil) } else { Vec::new() };
    let count = formats.last().map_or(0, |f| f.0 as usize + 1);
    if pixel && formats.first().map(|f| *f) != Some((0, DXGI_FORMAT_R32G32B32A32_FLOAT.0)) {
        return Err(format!("the pixel program's target 0 is not a float target ({formats:?})"));
    }
    let (formats, count, read) = if into.is_some() { (vec![(0, INTO_COLOUR.0)], 1, false) } else { (formats, count, true) };
    let (w, h) = (t.width, t.height);
    let depth_state = if into.is_some() { Some((false, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0)) } else { t.depth };
    let reverse = depth_state.is_some_and(|(_, func)| matches!(D3D12_COMPARISON_FUNC(func), D3D12_COMPARISON_FUNC_GREATER | D3D12_COMPARISON_FUNC_GREATER_EQUAL));
    let pipes = Pipes::new(dev, p, pixel, &formats, depth_state)?;
    // SAFETY: D3D12 calls with descriptions that outlive them; every
    // resource lives until the fence says the GPU is done.
    unsafe {
        let buffer = |size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES| buffer(dev, size, heap, flags, state);
        let texture = |w: u32, h: u32, format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES, clear: Option<*const D3D12_CLEAR_VALUE>| {
            committed(dev, &D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: u64::from(w), Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: format, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Flags: flags, ..Default::default() }, D3D12_HEAP_TYPE_DEFAULT, state, clear)
        };
        let heap = |ty, n: u32, flags| -> Result<ID3D12DescriptorHeap, String> { dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: n.max(1), Flags: flags, ..Default::default() }).map_err(e) };
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = match &pipes.compute {
            Some((_, cpso)) => dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, cpso),
            None => dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pipes.gpso),
        }
        .map_err(e)?;
        let up = upload(dev, &list, b, &pipes, &mut Shared::default())?;

        // The mesh program's records and index triples.
        let lay = &p.layout;
        let groups = u64::from(b.groups.max(1));
        let (rec_bytes, idx_bytes) = if pipes.compute.is_some() { (groups * u64::from(lay.record_bytes()), groups * u64::from(lay.index_bytes())) } else { (256, 256) };
        let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let records = buffer(rec_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, uav)?;
        let indices = buffer(idx_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, uav)?;
        record_dispatch(&list, &pipes, &up, records.GetGPUVirtualAddress(), indices.GetGPUVirtualAddress());
        list.ResourceBarrier(&[barrier(&records, uav, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE), barrier(&indices, uav, D3D12_RESOURCE_STATE_INDEX_BUFFER)]);

        // Targets: float colour targets cleared to CLEAR, D32.
        let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { Color: CLEAR } };
        let rtvs = heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, count as u32, D3D12_DESCRIPTOR_HEAP_FLAG_NONE)?;
        let rstep = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
        let rtv = |n: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtvs.GetCPUDescriptorHandleForHeapStart().ptr + n * rstep };
        let mut colour = Vec::new();
        if let Some(x) = into {
            dev.CreateRenderTargetView(&x.colour, None, rtv(0));
        }
        for n in (0..count).filter(|_| into.is_none()) {
            let format = DXGI_FORMAT(formats.iter().find(|f| f.0 as usize == n).map_or(DXGI_FORMAT_R32G32B32A32_FLOAT.0, |f| f.1));
            let float = format == DXGI_FORMAT_R32G32B32A32_FLOAT;
            let r = texture(w, h, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, rt, float.then_some(&clear as *const _))?;
            dev.CreateRenderTargetView(&r, None, rtv(n));
            if float {
                list.ClearRenderTargetView(rtv(n), &CLEAR, None);
            }
            colour.push(r);
        }
        let far = if reverse { 0.0 } else { 1.0 };
        let dclear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: far, Stencil: 0 } } };
        let depth = match into {
            Some(x) => x.depth.clone(),
            None => texture(w, h, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE, Some(&dclear as *const _))?,
        };
        let dsvs = heap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE)?;
        let dsv = dsvs.GetCPUDescriptorHandleForHeapStart();
        dev.CreateDepthStencilView(&depth, None, dsv);
        if read {
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, far, 0, None);
        }

        // The first handle outlives the call (a temporary in an `if` arm
        // is dropped at the arm's end: edition 2024; WARP 1.0.22 crashed).
        let first = rtv(0);
        list.OMSetRenderTargets(count as u32, (count > 0).then_some(&first as *const _), true, Some(&dsv));
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 }]);
        list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        let ibv = D3D12_INDEX_BUFFER_VIEW { BufferLocation: indices.GetGPUVirtualAddress(), SizeInBytes: idx_bytes as u32, Format: DXGI_FORMAT_R32_UINT };
        record_draw(&list, &pipes, &up, records.GetGPUVirtualAddress(), &ibv, lay.max_prims);

        // Target 0 back: 16 B per pixel, rows at the copy pitch. The depth
        // target after it, at the footprint the device gives its plane.
        let pitch = (w * 16).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        let back = buffer(u64::from(pitch) * u64::from(h), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
        if pixel && read {
            list.ResourceBarrier(&[barrier(&colour[0], rt, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&back), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Width: w, Height: h, Depth: 1, RowPitch: pitch } } } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&colour[0]), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        let ddesc = depth.GetDesc();
        let (mut dfp, mut dtotal) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), 0u64);
        dev.GetCopyableFootprints(&ddesc, 0, 1, 0, Some(&mut dfp), None, None, Some(&mut dtotal));
        let dback = buffer(dtotal, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
        if read {
            list.ResourceBarrier(&[barrier(&depth, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&dback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: dfp } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&depth), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        let rback = if t.records {
            let r = buffer(rec_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            list.ResourceBarrier(&[barrier(&records, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyBufferRegion(&r, 0, &records, 0, rec_bytes);
            Some(r)
        } else {
            None
        };
        list.Close().map_err(e)?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(e)?;
        let t0 = std::time::Instant::now();
        queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(e)?)]);
        let sent = xnw_gpu::gpuwatch::begin("bridge");
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        queue.Signal(&fence, 1).map_err(e)?;
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().map_err(|x| format!("during the submission: {}", e(x)))?;
            if t0.elapsed().as_secs_f64() > t.limit {
                xnw_gpu::gpuwatch::abandon("bridge", t.limit, xnw_gpu::gpuwatch::GRACE);
                return Err(format!("the submission took over {} s", t.limit));
            }
            std::thread::yield_now();
        }
        xnw_gpu::gpuwatch::end("bridge", sent, None);
        let seconds = t0.elapsed().as_secs_f64();
        let mut from = std::ptr::null_mut();
        back.Map(0, None, Some(&mut from)).map_err(e)?;
        let mut rt0 = Vec::with_capacity(if pixel && read { (w * h) as usize } else { 0 });
        for y in 0..if pixel && read { h as usize } else { 0 } {
            rt0.extend_from_slice(std::slice::from_raw_parts(from.cast::<u8>().add(y * pitch as usize).cast::<[f32; 4]>(), w as usize));
        }
        back.Unmap(0, None);
        dback.Map(0, None, Some(&mut from)).map_err(e)?;
        let mut zs = Vec::with_capacity(if read { (w * h) as usize } else { 0 });
        for y in 0..if read { h as usize } else { 0 } {
            zs.extend_from_slice(std::slice::from_raw_parts(from.cast::<u8>().add(dfp.Offset as usize + y * dfp.Footprint.RowPitch as usize).cast::<f32>(), w as usize));
        }
        dback.Unmap(0, None);
        let mut recs = Vec::new();
        if let Some(r) = &rback {
            r.Map(0, None, Some(&mut from)).map_err(e)?;
            recs.extend_from_slice(std::slice::from_raw_parts(from.cast::<u8>(), rec_bytes as usize));
            r.Unmap(0, None);
        }
        drop((up, colour, depth, records, indices));
        Ok(Frame { rt0, depth: zs, far, targets: count, seconds, records: recs })
    }
}

/// The pixel programs `bridgeps=on` names: the static world's most common
/// ones (first 32 bits of the container hash).
pub const BRIDGE_PS_DEFAULT: &[u32] = &[0x3ed6_8103, 0x3811_0e02, 0x46bf_b5fd, 0xd064_f214, 0xe01e_4d6f, 0x07b1_9da9];

/// `XNW_WORLD_RENDER=...,bridgeps=on|off|<hash>+<hash>...` (default off; the
/// last one wins): the pixel programs (first 32 bits of the container
/// hash, hex) whose draws our pass draws with the title's own mesh and
/// pixel program. Empty: off.
pub fn parse_switch(v: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for p in v.split(',').filter_map(|p| p.trim().strip_prefix("bridgeps=")) {
        out = match p {
            "on" => BRIDGE_PS_DEFAULT.to_vec(),
            "off" => Vec::new(),
            list => list.split('+').filter_map(|h| u32::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()).collect(),
        };
    }
    out
}

/// Our pass's colour target ([`Into::colour`], render-target state) read
/// back as RGBA8 rows.
pub fn read_colour(host: &xnw_gpu::host::Host, colour: &ID3D12Resource, w: u32, h: u32) -> Result<Vec<u8>, String> {
    let e = |e: windows::core::Error| e.message().to_string();
    let dev = &host.device;
    // SAFETY: D3D12 calls on resources that live until the fence says the GPU is done.
    unsafe {
        let pitch = (w * 4).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        let mut back: Option<ID3D12Resource> = None;
        let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: u64::from(pitch) * u64::from(h), Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back).map_err(e)?;
        let back = back.ok_or("no resource")?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
        list.ResourceBarrier(&[barrier(colour, rt, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&back), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: INTO_COLOUR, Width: w, Height: h, Depth: 1, RowPitch: pitch } } } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(colour), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.ResourceBarrier(&[barrier(colour, D3D12_RESOURCE_STATE_COPY_SOURCE, rt)]);
        list.Close().map_err(e)?;
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(e)?;
        queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(e)?)]);
        let sent = xnw_gpu::gpuwatch::begin("bridge readback");
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        queue.Signal(&fence, 1).map_err(e)?;
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().map_err(|x| format!("reading the colour target back: {}", e(x)))?;
            if xnw_gpu::gpuwatch::overdue("bridge readback", sent) {
                return Err("reading the colour target back: the submission was given up".into());
            }
            std::thread::yield_now();
        }
        xnw_gpu::gpuwatch::end("bridge readback", sent, None);
        let mut from = std::ptr::null_mut();
        back.Map(0, None, Some(&mut from)).map_err(e)?;
        let mut out = Vec::with_capacity((4 * w * h) as usize);
        for y in 0..h as usize {
            out.extend_from_slice(std::slice::from_raw_parts(from.cast::<u8>().add(y * pitch as usize), 4 * w as usize));
        }
        back.Unmap(0, None);
        Ok(out)
    }
}

/// Mean absolute difference of two RGB pictures over the pixels both cover:
/// (pixels compared, mean over the three channels). `ours` is RGBA8 rows;
/// `theirs` float rows where [`CLEAR`] marks an untouched pixel; `mask`
/// says which of our pixels the draw covered.
pub fn compare(ours: &[u8], mask: &[bool], theirs: &[[f32; 4]]) -> (usize, f64) {
    let (mut n, mut sum) = (0usize, 0f64);
    for (i, t) in theirs.iter().enumerate() {
        if *t == CLEAR || !mask.get(i).copied().unwrap_or(false) {
            continue;
        }
        let Some(o) = ours.get(4 * i..4 * i + 3) else { break };
        n += 1;
        sum += (0..3).map(|c| f64::from((t[c].clamp(0.0, 1.0) - f32::from(o[c]) / 255.0).abs())).sum::<f64>() / 3.0;
    }
    (n, if n == 0 { 0.0 } else { sum / n as f64 })
}

/// The translation options of the title's programs in the bridge: the
/// device's, as the live pipelines have them (`xnw_gpu::d3d12x::pipeline`
/// `options_for`;: computed barycentrics only on a device that has
/// them; none before a device is made).
pub fn options() -> emit::Options {
    // The heap-uniformity switches and adapter, as `runtime_options`.
    let (keep_uniform_heap, masked_lanes) = emit::heap_uniformity();
    emit::Options { barycentrics: xnw_gpu::host::host_barycentrics().unwrap_or(false), keep_uniform_heap, masked_lanes, ..emit::Options::default() }
}

/// One frame's upload memory: every distinct buffer once, each at an
/// offset its view can start at (`unit`: the element size; 256 for
/// constant buffers and root views). `parts`: (offset, bytes' address,
/// bytes, padded size) to copy, the rest of each part zeros.
#[derive(Default)]
pub struct Arena {
    at: HashMap<(usize, usize, u64, u64), u64>,
    pub parts: Vec<(u64, usize, usize, u64)>,
    pub end: u64,
}

impl Arena {
    pub fn place(&mut self, bytes: &[u8], size: u64, unit: u64) -> u64 {
        let k = (bytes.as_ptr() as usize, bytes.len(), size, unit);
        if let Some(&o) = self.at.get(&k) {
            return o;
        }
        let o = self.end.next_multiple_of(align(unit));
        self.end = o + size.max(16);
        self.at.insert(k, o);
        self.parts.push((o, bytes.as_ptr() as usize, bytes.len().min(size as usize), size));
        o
    }
}

/// The start a buffer of element size `unit` needs: 256 and the unit both divide it.
pub fn align(unit: u64) -> u64 {
    let unit = unit.max(4);
    256 * unit / gcd(256, unit)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A view kind's element size in upload memory (its start must be a multiple).
pub fn unit(kind: Kind) -> u64 {
    match kind {
        Kind::Raw | Kind::RawUav { .. } => 16,
        Kind::Structured(s) => u64::from(s.max(4)),
        Kind::Typed { bytes, .. } => u64::from(bytes.max(1)),
        Kind::Cbv => 256,
    }
}

/// Inputs this long or longer are kept on the device once seen on two frames.
pub const KEEP_MIN: usize = 4096;
/// Frames an input seen once is remembered, and a kept one stays unused, before it goes.
const SEEN_FRAMES: u64 = 8;
const KEPT_FRAMES: u64 = 600;
/// The kept buffer's least and largest size.
pub const KEPT_LEAST: u64 = 64 << 20;
const KEPT_MOST: u64 = 1 << 30;

/// Where an input lies: in the frame's upload memory, the kept buffer,
/// (`XNW_ZEROCOPY=on`, ZC1) in title memory: entry `n` of the frame's list
/// of in-place ranges, or (`bridged`) in the small inputs' upload
/// buffer, placed by content at an offset that stays across frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum At {
    Frame(u64),
    Kept(u64),
    Shared(u32),
    Small(u64),
}

/// Inputs kept on the device across frames (pure bookkeeping; `bridged`
/// for bridgeps, open to bridgems): an input the source hands over as the
/// same bytes (`Arc`) on a later frame is placed once in one default-heap
/// buffer of `size` bytes (filled up to `used`) and copied there once;
/// every other input goes with the frame ([`Arena`]). The bytes of every
/// placed or seen input are held, so their address names them alone. A
/// frame whose new inputs do not fit asks for a new buffer (`restart`:
/// twice the frame's kept bytes); until then they go with the frame.
#[derive(Default)]
pub struct Kept {
    pub size: u64,
    pub used: u64,
    at: HashMap<(usize, usize, u64), u64>,
    /// Address -> the bytes and the last frame they were placed or seen.
    held: HashMap<usize, (std::sync::Arc<Vec<u8>>, u64)>,
    frame: u64,
    frame_bytes: u64,
    pub restart: Option<u64>,
}

impl Kept {
    /// A new frame: inputs seen once long ago, and kept ones long unused, go.
    pub fn next_frame(&mut self) {
        self.frame += 1;
        self.frame_bytes = 0;
        let f = self.frame;
        // The sweep on every 8th frame (it walked every held input
        // each frame); an input goes up to 7 frames past its rule.
        if !f.is_multiple_of(8) {
            return;
        }
        let kept: std::collections::HashSet<usize> = self.at.keys().map(|k| k.0).collect();
        self.held.retain(|p, e| f.saturating_sub(e.1) <= if kept.contains(p) { KEPT_FRAMES } else { SEEN_FRAMES });
        let held = &self.held;
        self.at.retain(|k, _| held.contains_key(&k.0));
    }

    /// Starts again empty with `size` bytes (the caller makes the new buffer).
    pub fn start(&mut self, size: u64) {
        self.size = size;
        self.used = 0;
        self.at.clear();
        self.restart = None;
    }

    /// Where `bytes` (`size` padded, element size `unit`) go: `arc` (their
    /// holder; None: not kept) seen on an earlier frame is kept (`true`:
    /// copy it there this frame); anything else goes to `arena`.
    pub fn place(&mut self, arena: &mut Arena, bytes: &[u8], size: u64, unit: u64, arc: Option<&std::sync::Arc<Vec<u8>>>) -> (At, bool) {
        let p = bytes.as_ptr() as usize;
        let Some(arc) = arc.filter(|a| bytes.len() >= KEEP_MIN && a[..].as_ptr() as usize == p) else {
            return (At::Frame(arena.place(bytes, size, unit)), false);
        };
        let key = (p, bytes.len(), unit);
        if let Some(&o) = self.at.get(&key) {
            if let Some(e) = self.held.get_mut(&p) {
                if e.1 != self.frame {
                    self.frame_bytes += size;
                }
                e.1 = self.frame;
            }
            return (At::Kept(o), false);
        }
        let seen = self.held.get(&p).is_some_and(|e| e.1 < self.frame);
        if !seen {
            self.held.entry(p).or_insert_with(|| (arc.clone(), self.frame)).1 = self.frame;
            return (At::Frame(arena.place(bytes, size, unit)), false);
        }
        let o = self.used.next_multiple_of(align(unit));
        if o + size.max(16) > self.size {
            let want = (2 * (self.frame_bytes + size)).next_power_of_two().clamp(KEPT_LEAST, KEPT_MOST);
            self.restart = Some(want.max(self.size));
            return (At::Frame(arena.place(bytes, size, unit)), false);
        }
        self.used = o + size.max(16);
        self.frame_bytes += size;
        self.at.insert(key, o);
        self.held.insert(p, (arc.clone(), self.frame));
        (At::Kept(o), true)
    }
}

/// The validation layer's stored messages (`XNW_GPU_DEBUG=layer`), one
/// line per message id with its count and its first text; for errors also
/// the next three texts.
pub fn layer_messages(dev: &ID3D12Device) -> Vec<String> {
    use windows::core::Interface;
    let mut out = Vec::new();
    let Ok(queue) = dev.cast::<ID3D12InfoQueue>() else { return out };
    let mut by: std::collections::BTreeMap<(i32, i32), (u32, String)> = Default::default();
    // SAFETY: plain queries into buffers of the size the queue asked for.
    unsafe {
        let n = queue.GetNumStoredMessages();
        for i in 0..n {
            let mut len = 0usize;
            if queue.GetMessage(i, None, &mut len).is_err() || len == 0 {
                continue;
            }
            let mut buffer = vec![0u64; len.div_ceil(8)];
            let m = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
            if queue.GetMessage(i, Some(m), &mut len).is_err() {
                continue;
            }
            let text = std::slice::from_raw_parts((*m).pDescription, (*m).DescriptionByteLength.saturating_sub(1));
            let e = by.entry(((*m).Severity.0, (*m).ID.0)).or_insert_with(|| (0, String::from_utf8_lossy(text).chars().take(1600).collect()));
            e.0 += 1;
            // Errors (corruption, error): up to 4 texts each, whole (they name the draw, root parameter and shader).
            if (*m).Severity.0 <= 1 && (2..=4).contains(&e.0) {
                out.push(format!("layer severity {} id {} #{}: {}", (*m).Severity.0, (*m).ID.0, e.0, String::from_utf8_lossy(text).chars().take(1600).collect::<String>()));
            }
        }
        out.push(format!("validation layer: {n} messages stored"));
    }
    for ((sev, id), (n, text)) in by {
        out.push(format!("layer severity {sev} id {id} x{n}: {text}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader(stage: HwStage, user: &[(u32, u8, RootType, u32)], srvs: &[u8]) -> HostShader {
        HostShader { stage, dxil: Vec::new(), user_data: user.iter().map(|&(index, slot, root, dword)| emit::UserData { index, slot, root, dword }).collect(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: srvs.to_vec(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false }
    }

    /// Spans (ZC3): overlapping ranges share one whose start suits every
    /// unit; a raw view at 4 mod 16 next to one at 0 mod 16 goes to a layer of its own.
    #[test]
    fn spans_start_where_every_unit_fits() {
        let layers = plan_spans(vec![(0x1010, 0x100, 16), (0x1080, 0x40, 48), (0x5000, 0x10, 16), (0x1084, 0x20, 16)]);
        assert_eq!(layers.len(), 2);
        let (s, e) = layers[0][0];
        assert!(s <= 0x1010 && e >= 0x1110 && (0x1010 - s) % 16 == 0 && (0x1080 - s) % 48 == 0, "{s:#x}");
        assert_eq!(layers[0].len(), 2);
        assert!(layers[1][0].0 <= 0x1084 && (0x1084 - layers[1][0].0) % 16 == 0);
    }

    /// A view's title bytes (ZC3): the copy, the row's own words over and
    /// past it, no padding between them.
    #[test]
    fn title_view_parts_skip_padding() {
        let tv = TitleView { addr: 0x1000, captured: 100, overlays: vec![(150, 200), (90, 120), (190, 210)] };
        assert_eq!(tv.parts(300), vec![(0, 90, false), (90, 120, true), (150, 210, true)]);
        assert_eq!(tv.parts(160), vec![(0, 90, false), (90, 120, true), (150, 160, true)]);
        let stand_in = TitleView { addr: 0, captured: 0, overlays: vec![(64, 128)] };
        assert_eq!(stand_in.parts(128), vec![(64, 128, true)]);
        assert_eq!(TitleView { addr: 0, captured: 50, overlays: Vec::new() }.parts(50), vec![(0, 50, false)]);
    }

    /// A made-up draw in the shape of a captured one: root 0 a frame table
    /// (a constant buffer at slot 1, the bindless words at slot 86), root 1
    /// the mesh stage's table (record, meshlets), root 2 the pixel stage's
    /// (material constants, a texture), root 3 the vertex stream, root 4 two
    /// constants; one texture picked by heap index 9; one sampler at 5.
    #[test]
    fn binding_of_a_made_up_draw() {
        use RootType::{Constants, Table};
        let (cb, words, rec, lets, mat, vb) = ([1u8; 48], [2u8; 64], [3u8; 128], [4u8; 32], [5u8; 256], [6u8; 100]);
        let tex = Provided { width: 4, height: 4, format: 28, rows: vec![7; 64], row_bytes: 16, row_count: 4, dim: crate::textures::TEX_2D, depth: 1, mips: 1 };
        let v = |kind, bytes| Slot::Buffer(View { kind, bytes });
        let d = Draw {
            roots: vec![
                Root::Table(vec![(1, v(Kind::Cbv, &cb)), (86, v(Kind::Raw, &words))]),
                Root::Table(vec![(0, v(Kind::Raw, &rec)), (2, v(Kind::Raw, &lets))]),
                Root::Table(vec![(0, v(Kind::Cbv, &mat)), (3, Slot::Texture(&tex))]),
                Root::View(&vb),
                Root::Constants(vec![7, 0x40]),
            ],
            heap: vec![(9, Slot::Texture(&tex))],
            samplers: vec![(5, SamplerDesc { MaxAnisotropy: 4, ..Default::default() })],
            sampler_slots: 0,
            groups: 3,
        };
        let ms = shader(HwStage::Ms, &[(2, 0, Table, 0), (3, 1, Table, 0), (5, 4, Constants, 0)], &[3]);
        let ps = shader(HwStage::Ps, &[(0, 0, Table, 0), (1, 2, Table, 0), (6, 4, Constants, 1)], &[]);
        let b = bind(&ms, &ps, &d).unwrap();
        // Heap index 9 first; root 0 at 10 (87 slots), root 1 at 97 (3), root 2 at 100 (4).
        assert_eq!(b.mesh.constants, vec![0, 0, 10 * 32, 97 * 32, 0, 7, 0, 0]);
        assert_eq!(b.pixel.constants, vec![10 * 32, 100 * 32, 0, 0, 0, 0, 0x40, 0]);
        assert_eq!(b.views.slots.iter().map(|s| s.0).collect::<Vec<_>>(), vec![11, 96, 97, 99, 100]);
        assert_eq!(b.textures.iter().map(|t| t.0).collect::<Vec<_>>(), vec![9, 103]);
        assert_eq!((b.views.heap_slots, b.views.bounds), (109, [109, 104, 105, 106, 107, 108, 6, 0]));
        assert_eq!((b.mesh.views.clone(), b.mesh.lengths.clone()), (vec![(3u8, true, &vb[..])], vec![100]));
        assert!(b.pixel.views.is_empty() && b.samplers.iter().map(|s| (s.0, s.1.MaxAnisotropy)).eq([(5, 4)]) && b.groups == 3);
        // At slot 1000 of a shared heap: every slot, table, twin, guard and bound moves by 1000.
        let a = bind_at(&ms, &ps, &d, 1000).unwrap();
        assert_eq!(a.mesh.constants, vec![0, 0, 1010 * 32, 1097 * 32, 0, 7, 0, 0]);
        assert_eq!(a.pixel.constants, vec![1010 * 32, 1100 * 32, 0, 0, 0, 0, 0x40, 0]);
        assert_eq!(a.views.slots.iter().map(|s| s.0).collect::<Vec<_>>(), vec![1011, 1096, 1097, 1099, 1100]);
        assert_eq!(a.textures.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1009, 1103]);
        assert_eq!((a.views.heap_slots, a.views.bounds), (109, [1109, 1104, 1105, 1106, 1107, 1108, 6, 0]));
        let raw = |w: &[[u32; 2]]| w.iter().map(|e| e[1] & emit::StrideEntry::RAW_SLOTS).collect::<Vec<_>>();
        assert_eq!(raw(&a.views.strides), raw(&b.views.strides).iter().map(|&r| if r == 0 { 0 } else { r + 1000 }).collect::<Vec<_>>());
        // Sampler entries past the 2048 a shader-visible heap holds are left out.
        let mut wide = d.clone();
        wide.samplers.push((4095, SamplerDesc::default()));
        let w = bind(&ms, &ps, &wide).unwrap();
        assert_eq!((w.sampler_slots, w.views.bounds[6], w.samplers.len()), (SAMPLER_HEAP_MAX, SAMPLER_HEAP_MAX, 1));
        // What a record may lack, by name.
        let mut lacking = d.clone();
        lacking.roots[3] = Root::None;
        assert_eq!(bind(&ms, &ps, &lacking).err().unwrap(), "mesh: the record has no bytes for the root view at root 3");
        lacking.roots[2] = Root::None;
        assert_eq!(bind(&ms, &ps, &lacking).err().unwrap(), "the record has no table at root 2");
        let third = shader(HwStage::Ps, &[(0, 4, Constants, 2)], &[]);
        assert!(bind(&ms, &third, &d).err().unwrap().contains("root 4 has 2 constants"));
    }

    /// A container with a DXIL part (cs_6_6) and an SFI0 part (doubles, view
    /// heap indexing), and one with an OSG1 part of two targets and a depth.
    fn container(parts: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut c = b"DXBC".to_vec();
        c.resize(28, 0);
        c.extend((parts.len() as u32).to_le_bytes());
        let mut at = 32 + 4 * parts.len();
        for p in parts {
            c.extend((at as u32).to_le_bytes());
            at += 8 + p.1.len();
        }
        for p in parts {
            c.extend(p.0);
            c.extend((p.1.len() as u32).to_le_bytes());
            c.extend(&p.1);
        }
        c
    }

    #[test]
    fn needs_and_gap_from_a_container() {
        let c = container(&[(b"SFI0", (feature::DOUBLES | feature::VIEW_HEAP_INDEXING).to_le_bytes().to_vec()), (b"DXIL", (5u32 << 16 | 6 << 4 | 6).to_le_bytes().to_vec())]);
        let n = needs(&c, Some(32)).unwrap();
        assert_eq!(n, Needs { stage: "cs", model: (6, 6), features: 0x200_0001, wave: Some(32) });
        assert_eq!(n.text(), "cs_6_6, WaveSize(32), features 0x2000001 (doubles, view heap indexing)");
        assert!(needs(b"not a container", None).is_none());
        let caps = |model, lanes, doubles| xnw_gpu::host::Caps { adapter: "A".into(), vram_bytes: 0, shader_model: model, wave_lanes: lanes, mesh_shaders: false, ray_tracing: false, double_ops: doubles, agility: None, software: true, feature_level: 0, binding_tier: 0, tiled_tier: 0, int64_typed: false, int64_heap: false, enhanced_barriers: false, barycentrics: false, warp_module: None };
        assert_eq!(gap(&caps((6, 2), (4, 4), true), std::slice::from_ref(&n)).unwrap(), "A: shader model 6.2 < 6.6 (cs), wave lanes 4..4 lack 32");
        assert_eq!(gap(&caps((6, 6), (4, 64), false), std::slice::from_ref(&n)).unwrap(), "A: no double ops");
        assert_eq!(gap(&caps((6, 8), (32, 32), true), &[n]), None);
    }

    #[test]
    fn targets_and_parameter_order() {
        // Two elements: SV_Target register 1 (float), SV_Depth; names after them.
        let mut sig = vec![0u8; 8 + 64];
        sig[..4].copy_from_slice(&2u32.to_le_bytes());
        for (k, (name, comp, register)) in [(72u32, 3u32, 1u32), (82, 3, 0)].into_iter().enumerate() {
            let e = 8 + 32 * k;
            sig[e + 4..e + 8].copy_from_slice(&name.to_le_bytes());
            sig[e + 16..e + 20].copy_from_slice(&comp.to_le_bytes());
            sig[e + 20..e + 24].copy_from_slice(&register.to_le_bytes());
        }
        sig.extend(b"SV_Target\0SV_Depth\0");
        assert_eq!(targets(&container(&[(b"OSG1", sig)])), vec![(1, DXGI_FORMAT_R32G32B32A32_FLOAT.0)]);
        // Compute half: user data 0, root SRV 3 at 1, records 2..4, lengths 5.
        let ms = shader(HwStage::Ms, &[], &[3]);
        assert_eq!(params(&[&ms], 3), Params { views: vec![(3, true, 1)], stride: None, records: 2, lengths: Some(5), bounds: None });
        // Draw half: two user-data parameters, then the record view.
        let ps = shader(HwStage::Ps, &[], &[]);
        assert_eq!(params(&[&shader(HwStage::Vs, &[], &[]), &ps], 1), Params { views: Vec::new(), stride: None, records: 2, lengths: None, bounds: None });
    }

    #[test]
    fn bridgeps_switch() {
        assert!(parse_switch("screen,tex=live").is_empty());
        assert_eq!(parse_switch("screen,bridgeps=on"), BRIDGE_PS_DEFAULT.to_vec());
        assert_eq!(parse_switch("screen,bridgeps=3ed68103+0x38110e02"), vec![0x3ed6_8103, 0x3811_0e02]);
        assert!(parse_switch("bridgeps=on,bridgeps=off").is_empty());
    }

    /// The program that hung the card is left to our pass on the hardware
    /// adapter only, whether `bridgeps=on` or its own hash named it; the
    /// others run; the allow switch names single programs.
    #[test]
    fn the_program_that_hung_the_card_stays_ours_on_the_hardware_adapter() {
        assert!(BRIDGE_PS_DEFAULT.contains(&0x3811_0e02));
        let why = denied_with(0x3811_0e02, false, "").expect("denied on the card");
        assert!(why.contains("38110e02") && why.contains("0x887a0006"), "{why}");
        assert_eq!(denied_with(0x3811_0e02, true, ""), None, "the software adapter runs it");
        for &p in BRIDGE_PS_DEFAULT.iter().filter(|&&p| p != 0x3811_0e02) {
            assert_eq!(denied_with(p, false, ""), None, "{p:08x}");
        }
        assert_eq!(denied_with(0x3811_0e02, false, "07b19da9+0x38110e02"), None);
        assert!(denied_with(0x3811_0e02, false, "07b19da9").is_some());
        assert!(denied(0x3811_0e02_u128 << 96 | 5, true).is_none());
    }

    #[test]
    fn difference_over_covered_pixels() {
        let ours = [255u8, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128, 255];
        let theirs = [[1.0, 0.0, 0.3, 1.0], CLEAR, [0.5, 0.5, 0.5, 1.0]];
        let (n, d) = compare(&ours, &[true, true, false], &theirs);
        assert_eq!(n, 1);
        assert!((d - 0.1).abs() < 1e-6, "{d}");
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
    fn kept_inputs_go_to_the_device_on_their_second_frame_and_once() {
        let big = std::sync::Arc::new(vec![5u8; 8192]);
        let small = std::sync::Arc::new(vec![6u8; 64]);
        let other = std::sync::Arc::new(vec![7u8; 8192]);
        let mut k = Kept::default();
        k.start(KEPT_LEAST);
        let frame = |k: &mut Kept| {
            k.next_frame();
            let mut a = Arena::default();
            let x = k.place(&mut a, &big, 8192, 16, Some(&big));
            let y = k.place(&mut a, &small, 64, 16, Some(&small));
            // Bytes not handed over as their holder are never kept.
            let z = k.place(&mut a, &other, 8192, 16, None);
            (x, y, z, a.parts.len())
        };
        // Seen first: with the frame; then kept and copied once; then kept as it is.
        assert!(matches!(frame(&mut k), ((At::Frame(0), false), (At::Frame(_), false), (At::Frame(_), false), 3)), "first frame");
        assert!(matches!(frame(&mut k), ((At::Kept(0), true), (At::Frame(_), false), (At::Frame(_), false), 2)), "second frame");
        assert!(matches!(frame(&mut k), ((At::Kept(0), false), _, _, 2)), "third frame");
        assert_eq!(k.used, 8192);
        // A kept buffer too small asks to start again, twice the frame's kept bytes.
        let mut t = Kept::default();
        t.start(4096);
        let mut a = Arena::default();
        t.next_frame();
        t.place(&mut a, &big, 8192, 16, Some(&big));
        t.next_frame();
        assert!(matches!(t.place(&mut a, &big, 8192, 16, Some(&big)), (At::Frame(_), false)));
        assert_eq!(t.restart, Some(KEPT_LEAST));
    }
}
