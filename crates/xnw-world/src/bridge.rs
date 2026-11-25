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
//!   `XNW_BRIDGE_HW=1`.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use xnw_shader::emit::{self, HostShader};
use xnw_shader::{HwStage, RootType};

use super::preskin::{self, Kind, View};
use super::textures::Provided;

/// What a descriptor slot holds for the binding: a buffer view (as the
/// compute binding describes one) or a texture's texels.
#[derive(Clone, Copy)]
pub enum Slot<'a> {
    Buffer(View<'a>),
    Texture(&'a Provided),
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
        Slot::Texture(_) => View { kind: Kind::Raw, bytes: &[] },
    };
    let mut dispatch = preskin::Dispatch { groups: [d.groups, 1, 1], tables: Vec::new(), heap: d.heap.iter().map(|(k, s)| (*k, lay(s))).collect() };
    for &r in &tables {
        let Some(Root::Table(views)) = d.roots.get(usize::from(r)) else {
            return Err(format!("the record has no table at root {r}"));
        };
        dispatch.tables.push((u32::from(r), views.iter().map(|(k, s)| (*k, lay(s))).collect()));
    }
    let user_data = tables.iter().map(|&r| emit::UserData { index: u32::from(r), slot: r, root: RootType::Table, dword: 0 }).collect();
    let layout = HostShader { stage: HwStage::Cs, dxil: Vec::new(), user_data, samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), address_map: false, descriptor_words: false };
    let mut views = preskin::bind(&layout, &dispatch)?;
    let first: Vec<u32> = views.constants.iter().map(|c| c / 32).collect();
    let mut textures: Vec<(u32, &Provided)> = d.heap.iter().filter_map(|(k, s)| if let Slot::Texture(t) = s { Some((*k, *t)) } else { None }).collect();
    for &r in &tables {
        if let Some(Root::Table(v)) = d.roots.get(usize::from(r)) {
            textures.extend(v.iter().filter_map(|(k, s)| if let Slot::Texture(t) = s { Some((first[usize::from(r)] + k, *t)) } else { None }));
        }
    }
    textures.sort_by_key(|t| t.0);
    views.slots.retain(|(s, _)| textures.binary_search_by_key(s, |t| t.0).is_err());
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
    // creation fails past it; VIctory#2451): entries past it are left
    // out, and the heap bound sends their indices to the bound's guard.
    let sampler_slots = d.samplers.iter().map(|s| s.0 + 1).max().unwrap_or(0).max(d.sampler_slots).clamp(1, SAMPLER_HEAP_MAX);
    views.bounds[6] = sampler_slots;
    Ok(Binding { views, textures, samplers: d.samplers.iter().filter(|s| s.0 < sampler_slots).copied().collect(), sampler_slots, mesh, pixel, groups: d.groups })
}

/// Optional-feature bits of a DXIL container's `SFI0` part that an adapter
/// may lack.
pub mod feature {
    pub const DOUBLES: u64 = 1 << 0;
    pub const DOUBLE_EXTENSIONS: u64 = 1 << 5;
    pub const WAVE_OPS: u64 = 1 << 14;
    pub const INT64_OPS: u64 = 1 << 15;
    pub const ATOMIC64_TYPED: u64 = 1 << 22;
    pub const VIEW_HEAP_INDEXING: u64 = 1 << 25;
    pub const SAMPLER_HEAP_INDEXING: u64 = 1 << 26;
    pub const ATOMIC64_HEAP: u64 = 1 << 28;
    pub const NAMES: [(u64, &str); 8] = [(DOUBLES, "doubles"), (DOUBLE_EXTENSIONS, "11.1 double extensions"), (WAVE_OPS, "wave ops"), (INT64_OPS, "64-bit integers"), (ATOMIC64_TYPED, "64-bit atomics on typed resources"), (VIEW_HEAP_INDEXING, "view heap indexing"), (SAMPLER_HEAP_INDEXING, "sampler heap indexing"), (ATOMIC64_HEAP, "64-bit atomics on heap resources")];
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
    for (bit, has, what) in [(feature::DOUBLES, caps.double_ops, "no double ops"), (feature::ATOMIC64_TYPED, caps.int64_typed, "no 64-bit atomics on typed resources"), (feature::ATOMIC64_HEAP, caps.int64_heap, "no 64-bit atomics on heap resources")] {
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
pub struct Programs {
    /// The mesh program in compute form and its record layout.
    pub ms: HostShader,
    pub layout: emit::mesh::Layout,
    /// The pass-through vertex shader for `ps`.
    pub vs: Vec<u8>,
    pub ps: HostShader,
    /// The compute form's `WaveSize`.
    pub wave: u32,
    /// A plain draw through the title's vertex program instead (HAIR1:
    /// the strand draws): `ms` is then that vertex program, drawn as
    /// (vertices, instances, triangle strip) with no compute step.
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
/// target (D32, depth-write state). Target 0 of the pixel program goes to
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

/// The validation layer's stored errors, newest first (at most 4), as a
/// tail for a failure text; empty without the layer.
fn layer_errors(dev: &ID3D12Device) -> String {
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

/// Dispatches the mesh program over `b.groups` groups and draws its records
/// with the pixel program into float targets and a depth target; returns
/// target 0. Refuses an adapter [`gap`] names, and the hardware adapter
/// unless `XNW_BRIDGE_HW=1`.
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

fn submit(host: &xnw_gpu::host::Host, p: &Programs, b: &Binding, t: &Target, into: Option<&Into>) -> Result<Frame, String> {
    if !host.caps.software && std::env::var("XNW_BRIDGE_HW").ok().as_deref() != Some("1") {
        return Err("the title's translated programs run on the software adapter; the hardware adapter needs XNW_BRIDGE_HW=1 (a scheduled run)".into());
    }
    if let Some(gap) = gap(&host.caps, &p.needs()) {
        return Err(gap);
    }
    let e = |e: windows::core::Error| e.message().to_string();
    let dev = &host.device;
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
    let passthrough = HostShader { stage: HwStage::Vs, dxil: p.vs.clone(), user_data: Vec::new(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), address_map: false, descriptor_words: false };
    let vs = if p.vertex.is_some() { &p.ms } else { &passthrough };
    let pixel = !t.depth_only;
    let gstages: Vec<(&HostShader, D3D12_SHADER_VISIBILITY)> = if pixel { vec![(vs, D3D12_SHADER_VISIBILITY_VERTEX), (&p.ps, D3D12_SHADER_VISIBILITY_PIXEL)] } else { vec![(vs, D3D12_SHADER_VISIBILITY_VERTEX)] };
    let groot = eh.root_signature_records(&gstages, &[(false, RECORD_REGISTER)])?;
    let (cp, gp) = (params(&[&p.ms], 3), params(&gstages.iter().map(|s| s.0).collect::<Vec<_>>(), 1));
    let formats = if pixel { targets(&p.ps.dxil) } else { Vec::new() };
    let count = formats.last().map_or(0, |f| f.0 as usize + 1);
    if pixel && formats.first().map(|f| *f) != Some((0, DXGI_FORMAT_R32G32B32A32_FLOAT.0)) {
        return Err(format!("the pixel program's target 0 is not a float target ({formats:?})"));
    }
    let (formats, count, read) = if into.is_some() { (vec![(0, INTO_COLOUR.0)], 1, false) } else { (formats, count, true) };
    let (w, h) = (t.width, t.height);
    // SAFETY: D3D12 calls with descriptions that outlive them; every
    // resource lives until the fence says the GPU is done.
    unsafe {
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
        for &(n, f) in &formats {
            gdesc.RTVFormats[n as usize] = DXGI_FORMAT(f);
        }
        let depth_state = if into.is_some() { Some((false, D3D12_COMPARISON_FUNC_GREATER_EQUAL.0)) } else { t.depth };
        let reverse = depth_state.is_some_and(|(_, func)| matches!(D3D12_COMPARISON_FUNC(func), D3D12_COMPARISON_FUNC_GREATER | D3D12_COMPARISON_FUNC_GREATER_EQUAL));
        if let Some((write, func)) = depth_state {
            gdesc.DepthStencilState = D3D12_DEPTH_STENCIL_DESC { DepthEnable: true.into(), DepthWriteMask: if write { D3D12_DEPTH_WRITE_MASK_ALL } else { D3D12_DEPTH_WRITE_MASK_ZERO }, DepthFunc: D3D12_COMPARISON_FUNC(func), ..Default::default() };
        }
        let gpso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&gdesc).map_err(|x| format!("graphics pipeline: {}", e(x)))?;

        let committed = |desc: &D3D12_RESOURCE_DESC, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES, clear: Option<*const D3D12_CLEAR_VALUE>| -> Result<ID3D12Resource, String> {
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, desc, state, clear, &mut r).map_err(e)?;
            r.ok_or_else(|| "no resource".to_string())
        };
        let buffer = |size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES| {
            committed(&D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size.max(256), Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: flags, ..Default::default() }, heap, state, None)
        };
        let texture = |w: u32, h: u32, format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES, clear: Option<*const D3D12_CLEAR_VALUE>| {
            committed(&D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: u64::from(w), Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: format, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Flags: flags, ..Default::default() }, D3D12_HEAP_TYPE_DEFAULT, state, clear)
        };
        let upload = |bytes: &[u8], size: u64| -> Result<ID3D12Resource, String> {
            let r = buffer(size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let mut at = std::ptr::null_mut();
            r.Map(0, None, Some(&mut at)).map_err(e)?;
            std::ptr::write_bytes(at.cast::<u8>(), 0, size.max(256) as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), at.cast::<u8>(), bytes.len().min(size as usize));
            r.Unmap(0, None);
            Ok(r)
        };
        let heap = |ty, n: u32, flags| -> Result<ID3D12DescriptorHeap, String> { dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: n.max(1), Flags: flags, ..Default::default() }).map_err(e) };
        let visible = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        let views = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, b.views.heap_slots, visible)?;
        let samplers = heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, b.sampler_slots, visible)?;
        let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let at = |slot: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: views.GetCPUDescriptorHandleForHeapStart().ptr + slot as usize * step };
        let srv = |format, n: u32, stride: u32, flags| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_BUFFER, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: n, StructureByteStride: stride, Flags: flags } } };
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
            match v.kind {
                Kind::Raw => {
                    let r = upload(v.bytes, size)?;
                    dev.CreateShaderResourceView(&r, Some(&srv(DXGI_FORMAT_R32_TYPELESS, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s));
                    keep.push(r);
                }
                Kind::Structured(stride) => {
                    let r = upload(v.bytes, size)?;
                    dev.CreateShaderResourceView(&r, Some(&srv(DXGI_FORMAT_UNKNOWN, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(s));
                    keep.push(r);
                }
                Kind::Typed { format, .. } => {
                    let r = upload(v.bytes, size)?;
                    dev.CreateShaderResourceView(&r, Some(&srv(DXGI_FORMAT(format), n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(s));
                    keep.push(r);
                }
                Kind::Cbv => {
                    let r = upload(v.bytes, size)?;
                    dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: r.GetGPUVirtualAddress(), SizeInBytes: n }), at(s));
                    keep.push(r);
                }
                Kind::RawUav { .. } => {}
            }
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

        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = match &compute {
            Some((_, cpso)) => dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, cpso),
            None => dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &gpso),
        }
        .map_err(e)?;
        // Textures: one mip each, copied from an upload buffer.
        let shader_read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        for &(s, tx) in &b.textures {
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: u64::from(tx.width), Height: tx.height, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT(tx.format as i32), SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let r = committed(&desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, None)?;
            let (mut fp, mut rows, mut row_size, mut total) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), 0u32, 0u64, 0u64);
            dev.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_size), Some(&mut total));
            let up = buffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let mut to = std::ptr::null_mut();
            up.Map(0, None, Some(&mut to)).map_err(e)?;
            let n = (row_size as usize).min(tx.row_bytes as usize);
            for y in 0..rows.min(tx.row_count) as usize {
                if let Some(row) = tx.rows.get(y * tx.row_bytes as usize..y * tx.row_bytes as usize + n) {
                    std::ptr::copy_nonoverlapping(row.as_ptr(), to.cast::<u8>().add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize), n);
                }
            }
            up.Unmap(0, None);
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&up), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.ResourceBarrier(&[barrier(&r, D3D12_RESOURCE_STATE_COPY_DEST, shader_read)]);
            dev.CreateShaderResourceView(&r, Some(&tex2d(DXGI_FORMAT(tx.format as i32))), at(s));
            keep.extend([r, up]);
        }

        // The mesh program's records and index triples, and the piece's
        // first group (0: a direct dispatch of every group).
        let lay = &p.layout;
        let groups = u64::from(b.groups.max(1));
        let (rec_bytes, idx_bytes) = if compute.is_some() { (groups * u64::from(lay.record_bytes()), groups * u64::from(lay.index_bytes())) } else { (256, 256) };
        let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let records = buffer(rec_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, uav)?;
        let indices = buffer(idx_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, uav)?;
        let base = upload(&[0; 16], 16)?;
        let strides: Vec<u8> = b.views.strides.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        let stride_table = upload(&strides, strides.len() as u64)?;
        let mut root_view = |v: &(u8, bool, &[u8])| -> Result<u64, String> {
            let r = upload(v.2, (v.2.len() as u64).next_multiple_of(256))?;
            let va = r.GetGPUVirtualAddress();
            keep.push(r);
            Ok(va)
        };
        list.SetDescriptorHeaps(&[Some(views.clone()), Some(samplers.clone())]);
        if let Some((croot, _)) = &compute {
        list.SetComputeRootSignature(croot);
        list.SetComputeRoot32BitConstants(0, b.mesh.constants.len() as u32, b.mesh.constants.as_ptr().cast(), 0);
        for &(r, is_srv, param) in &cp.views {
            let v = b.mesh.views.iter().find(|v| v.0 == r && v.1 == is_srv).ok_or_else(|| format!("mesh: root view {r} not bound"))?;
            let va = root_view(v)?;
            if is_srv { list.SetComputeRootShaderResourceView(param, va) } else { list.SetComputeRootConstantBufferView(param, va) }
        }
        if let Some(param) = cp.stride {
            list.SetComputeRootShaderResourceView(param, stride_table.GetGPUVirtualAddress());
        }
        list.SetComputeRootUnorderedAccessView(cp.records, records.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(cp.records + 1, indices.GetGPUVirtualAddress());
        list.SetComputeRootShaderResourceView(cp.records + 2, base.GetGPUVirtualAddress());
        if let Some(param) = cp.lengths {
            let mut lengths = b.mesh.lengths.clone();
            lengths.resize(lengths.len().next_multiple_of(4).max(4), 0);
            list.SetComputeRoot32BitConstants(param, lengths.len() as u32, lengths.as_ptr().cast(), 0);
        }
        if let Some(param) = cp.bounds {
            list.SetComputeRoot32BitConstants(param, 8, g.as_ptr().cast(), 0);
        }
        list.Dispatch(b.groups, 1, 1);
        }
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

        list.SetPipelineState(&gpso);
        list.SetGraphicsRootSignature(&groot);
        if p.vertex.is_some() {
            list.SetGraphicsRoot32BitConstants(0, b.mesh.constants.len() as u32, b.mesh.constants.as_ptr().cast(), 0);
        } else {
            list.SetGraphicsRoot32BitConstants(0, 1, [0u32].as_ptr().cast(), 0);
        }
        if pixel {
            list.SetGraphicsRoot32BitConstants(1, b.pixel.constants.len() as u32, b.pixel.constants.as_ptr().cast(), 0);
        }
        for &(r, is_srv, param) in &gp.views {
            let v = b.pixel.views.iter().chain(if p.vertex.is_some() { &b.mesh.views[..] } else { &[] }).find(|v| v.0 == r && v.1 == is_srv).ok_or_else(|| format!("pixel: root view {r} not bound"))?;
            let va = root_view(v)?;
            if is_srv { list.SetGraphicsRootShaderResourceView(param, va) } else { list.SetGraphicsRootConstantBufferView(param, va) }
        }
        if let Some(param) = gp.stride {
            list.SetGraphicsRootShaderResourceView(param, stride_table.GetGPUVirtualAddress());
        }
        list.SetGraphicsRootShaderResourceView(gp.records, records.GetGPUVirtualAddress());
        if let Some(param) = gp.bounds {
            list.SetGraphicsRoot32BitConstants(param, 8, g.as_ptr().cast(), 0);
        }
        // The first handle outlives the call (a temporary in an `if` arm
        // is dropped at the arm's end: edition 2024; WARP 1.0.22 crashed).
        let first = rtv(0);
        list.OMSetRenderTargets(count as u32, (count > 0).then_some(&first as *const _), true, Some(&dsv));
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 }]);
        if let Some((vertices, instances, strip)) = p.vertex {
            use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP};
            list.IASetPrimitiveTopology(if strip { D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP } else { D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST });
            list.DrawInstanced(vertices, instances, 0, 0);
        } else {
            list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW { BufferLocation: indices.GetGPUVirtualAddress(), SizeInBytes: idx_bytes as u32, Format: DXGI_FORMAT_R32_UINT }));
            list.DrawIndexedInstanced(b.groups * lay.max_prims * 3, 1, 0, 0, 0);
        }

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
        let rback = if t.records && compute.is_some() {
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
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        queue.Signal(&fence, 1).map_err(e)?;
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().map_err(|x| format!("during the submission: {}", e(x)))?;
            if t0.elapsed().as_secs_f64() > t.limit {
                return Err(format!("the submission took over {} s", t.limit));
            }
            std::thread::yield_now();
        }
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
        drop((keep, colour, depth, records, indices, base, stride_table));
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
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        queue.Signal(&fence, 1).map_err(e)?;
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().map_err(|x| format!("reading the colour target back: {}", e(x)))?;
            std::thread::yield_now();
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn shader(stage: HwStage, user: &[(u32, u8, RootType, u32)], srvs: &[u8]) -> HostShader {
        HostShader { stage, dxil: Vec::new(), user_data: user.iter().map(|&(index, slot, root, dword)| emit::UserData { index, slot, root, dword }).collect(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: srvs.to_vec(), root_uavs: Vec::new(), address_map: false, descriptor_words: false }
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
        let tex = Provided { width: 4, height: 4, format: 28, rows: vec![7; 64], row_bytes: 16, row_count: 4 };
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
        let caps = |model, lanes, doubles| xnw_gpu::host::Caps { adapter: "A".into(), vram_bytes: 0, shader_model: model, wave_lanes: lanes, mesh_shaders: false, ray_tracing: false, double_ops: doubles, agility: None, software: true, feature_level: 0, binding_tier: 0, tiled_tier: 0, int64_typed: false, int64_heap: false, enhanced_barriers: false, warp_module: None };
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

    #[test]
    fn difference_over_covered_pixels() {
        let ours = [255u8, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128, 255];
        let theirs = [[1.0, 0.0, 0.3, 1.0], CLEAR, [0.5, 0.5, 0.5, 1.0]];
        let (n, d) = compare(&ours, &[true, true, false], &theirs);
        assert_eq!(n, 1);
        assert!((d - 0.1).abs() < 1e-6, "{d}");
    }
}
