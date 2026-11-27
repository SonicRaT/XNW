//! UE2 rank 2: the world pass's draw loop in two steps. A serial
//! resolve turns every draw into a flat [`Packet`] (pipeline, tables,
//! constants with the texture, layer and flag words, index view, arguments,
//! cull slot): all hashing, texture-table lookups, state and pipeline choice
//! happen there, and the texture table is only read after it. The walk then
//! records the packets in title order, with the loop's list events (chunk,
//! water and our lighting, bridged rows with the title's pixel program)
//! between them. By default on the pass's own list; with
//! `XNW_WORLD_RENDER=...,precord=on` (or `precord=N`, 2-8 lists) a run of at
//! least 2 x [`MIN_RANGE`] packets is cut into contiguous ranges recorded by
//! scoped threads on lists of their own and sent in order, so the picture is
//! the same (UE 5.9: `FParallelMeshDrawCommandPass::DispatchDraw`,
//! `r.RHICmdMinDrawsPerParallelCmdList` 64; ideas only, no UE code).
//!
//! Recorder threads write no descriptor (VIctory-7900XTX#42: shared
//! RTV/DSV rings raced under threaded recording): every view and table entry
//! is written in the serial part; they bind GPU handles and the pass's two
//! fixed RTV/DSV heap starts only.
//!
//! PERF3 `bridgepar=on|off` (default on): the bridged rows (the title's own
//! programs, [`bridged::Gpu::record_with`]) are prepared on a thread of
//! their own from the frame's start, once the last frame's fence was
//! reached (their upload, small-input and view memory is free again):
//! bindings, places, uploads, views and pipelines while the pass records
//! its own draws. Their texture views follow once the frame's table is made
//! ([`Pass::bridge_prep_textures`]); at the rows' place the pass hands its
//! open list over and waits ([`Pass::bridge_prep_finish`]): the list's
//! commands are the serial record's, in its order. That thread alone writes
//! the bridged heaps and buffers; the pass thread does not touch the list
//! while it records.

use super::*;

/// Fewest draws in one parallel range (UE's 64).
pub const MIN_RANGE: usize = 64;
/// Lists for `precord=on`.
pub const DEFAULT_LISTS: u32 = 4;
const ROOT_RAW: u8 = 1;
const ROOT_PLAIN: u8 = 2;
const NO_ARG: u32 = u32::MAX;

/// `precord=on|N|off` of `XNW_WORLD_RENDER` (default off): the lists of a
/// parallel run (0 or 1: recorded on the pass's list). The last one wins.
pub fn parse_switch(v: &str) -> u32 {
    v.split(',')
        .filter_map(|p| p.trim().strip_prefix("precord="))
        .map(|s| match s {
            "on" => DEFAULT_LISTS,
            "off" => 0,
            n => n.parse::<u32>().map_or(0, |n| n.min(8)),
        })
        .last()
        .unwrap_or(0)
}

/// `bridgepar=on|off` of `XNW_WORLD_RENDER` (default on). The last one wins.
pub fn parse_bridgepar(v: &str) -> bool {
    v.split(',').filter_map(|p| p.trim().strip_prefix("bridgepar=")).map(|s| !matches!(s, "off" | "0")).last().unwrap_or(true)
}

/// The rows' textures by handle as the frame's table has them: the host
/// copy and view ([`textures::TexTable::host_view`]) and the flat reason.
type TexSnap = bridged::FxMap<u64, (Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)>, usize)>;

/// What the preparing thread gives back: the bridged objects, what the
/// frame replaced (held to its fence), the record's result, handles the
/// texture snapshot lacked (0 expected), ms from the list's arrival to the end.
struct PrepOut {
    gpu: bridged::Gpu,
    keep: Vec<windows::core::IUnknown>,
    st: std::result::Result<bridged::Stats, String>,
    misses: u32,
    list_ms: f64,
}

/// The frame's bridged rows in preparation ([`Pass::bridge_prep_start`]).
pub(super) struct BridgePrep {
    frame: std::sync::Arc<bridged::Frame>,
    tex: Option<std::sync::mpsc::Sender<Shared<TexSnap>>>,
    list: std::sync::mpsc::Sender<Shared<ID3D12GraphicsCommandList>>,
    done: std::thread::JoinHandle<Shared<PrepOut>>,
}

/// One draw as the recorder issues it.
#[derive(Clone, Copy)]
pub(super) struct Packet {
    root: u8,
    /// Index into [`Resolved::psos`].
    pso: u8,
    /// Render targets bound (`gbuf=on`: 3 or 1).
    rts: u8,
    /// Table 1 (the vertex view) and table 3 (instances or the cull's output), raw root only.
    slot: u64,
    t3: u64,
    ibv: D3D12_INDEX_BUFFER_VIEW,
    vbv: D3D12_VERTEX_BUFFER_VIEW,
    /// Root constants: 58 on the raw root, the first 32 on ours.
    k: [u32; 58],
    indices: u32,
    instances: u32,
    /// The cull's argument slot ([`NO_ARG`]: drawn directly).
    arg: u32,
}

/// What the loop records between packets.
pub(super) enum Event {
    /// `chunk`: the list so far sent and waited for.
    Chunk,
    /// Our lighting first (when `light`), at `rts` targets, then the water.
    Water { light: bool, rts: u32 },
    /// A bridged row with the title's pixel program (list draw index).
    Title(u32),
}

pub(super) struct Resolved {
    packets: Vec<Packet>,
    /// (packets before it, event), in order.
    events: Vec<(usize, Event)>,
    psos: Vec<ID3D12PipelineState>,
    /// The targets bound when the pass began.
    start_rts: u32,
    /// The water was recorded (or is not this pass's: wire).
    pub water_done: bool,
}

/// The frame's inputs to the resolve.
pub(super) struct Frame<'a> {
    pub list: &'a FrameList,
    pub cam: &'a Camera,
    pub culled: Option<&'a gpucull::Culled>,
    pub pre: &'a std::collections::HashMap<u32, (u32, u32)>,
    pub device_rows: &'a std::collections::HashSet<u32>,
    pub first: &'a [u32],
    pub pal_bytes: u32,
    pub inst_bytes: u32,
    pub inst_total: u32,
    pub turn: usize,
    pub tex_table: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub light_va: u64,
    /// `ourlight=metal`: the lighting pass's view rays.
    pub clip_to_world: &'a Mat,
    pub ms_rows: Option<&'a bridge_ms::Batch>,
}

/// The loop's state across its passes and events.
pub(super) struct Walk {
    pub rts: u32,
    pub gbuf_done: bool,
    pub lit_done: bool,
    /// `ourlight=metal`: the bridged rows were drawn before the lighting pass.
    pub bridged_done: bool,
    pub chunk_from: usize,
    pub chunk_wait: f64,
    pub sub_k: u32,
}

/// What a recorder has bound on its list (root 0: nothing yet).
struct Bind {
    root: u8,
    pso: u8,
    t3: u64,
    rts: u32,
}

impl Bind {
    fn fresh(rts: u32) -> Bind {
        Bind { root: 0, pso: u8::MAX, t3: 0, rts }
    }
}

/// The frame's fixed bindings for every recorder.
struct Ctx<'a> {
    raw_root: &'a ID3D12RootSignature,
    root: &'a ID3D12RootSignature,
    psos: &'a [ID3D12PipelineState],
    t2: D3D12_GPU_DESCRIPTOR_HANDLE,
    t4: D3D12_GPU_DESCRIPTOR_HANDLE,
    light_va: u64,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    indirect: Option<(&'a ID3D12CommandSignature, &'a ID3D12Resource)>,
}

/// References handed to the recorder threads. D3D12 lists, root signatures,
/// pipelines and resources are free-threaded; each list is recorded by one thread.
struct Shared<T>(T);
// SAFETY: see above; the packets and pipeline table are only read.
unsafe impl<T> Send for Shared<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    fn get(&self) -> &T {
        &self.0
    }
}

/// The snapshot of `frame`'s textures in `tex` (after its `prepare`).
fn tex_snap(tex: &textures::TexTable, frame: &bridged::Frame) -> TexSnap {
    let mut snap = TexSnap::default();
    for (_, o) in &frame.rows {
        for h in o.handles() {
            snap.entry(h).or_insert_with(|| (tex.host_view(h), tex.flat_reason(h) as usize));
        }
    }
    snap
}

fn pso_at(psos: &mut Vec<ID3D12PipelineState>, p: &ID3D12PipelineState) -> u8 {
    let raw = windows::core::Interface::as_raw(p);
    match psos.iter().position(|q| windows::core::Interface::as_raw(q) == raw) {
        Some(i) => i as u8,
        None => {
            psos.push(p.clone());
            (psos.len() - 1) as u8
        }
    }
}

/// Records `packets` on the open list `l` from the bindings `b`.
/// SAFETY (caller): an open list; every handle and view in the packets lives until the list ran.
unsafe fn record(l: &ID3D12GraphicsCommandList, c: &Ctx, packets: &[Packet], b: &mut Bind) {
    // SAFETY: as above.
    unsafe {
        for q in packets {
            if q.root != b.root {
                if q.root == ROOT_RAW {
                    l.SetGraphicsRootSignature(c.raw_root);
                    l.SetGraphicsRootDescriptorTable(2, c.t2);
                    l.SetGraphicsRootDescriptorTable(3, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: q.t3 });
                    l.SetGraphicsRootDescriptorTable(4, c.t4);
                    l.SetGraphicsRootConstantBufferView(5, c.light_va);
                    b.t3 = q.t3;
                } else {
                    l.SetGraphicsRootSignature(c.root);
                }
                (b.root, b.pso) = (q.root, u8::MAX);
            }
            if q.pso != b.pso {
                l.SetPipelineState(&c.psos[q.pso as usize]);
                b.pso = q.pso;
            }
            if u32::from(q.rts) != b.rts {
                l.OMSetRenderTargets(u32::from(q.rts), Some(&c.rtv), true, Some(&c.dsv));
                b.rts = u32::from(q.rts);
            }
            l.IASetIndexBuffer(Some(&q.ibv));
            if q.root == ROOT_RAW {
                if q.t3 != b.t3 {
                    l.SetGraphicsRootDescriptorTable(3, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: q.t3 });
                    b.t3 = q.t3;
                }
                l.SetGraphicsRoot32BitConstants(0, 58, q.k.as_ptr().cast(), 0);
                l.SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: q.slot });
            } else {
                l.SetGraphicsRoot32BitConstants(0, 32, q.k.as_ptr().cast(), 0);
                l.IASetVertexBuffers(0, Some(&[q.vbv]));
            }
            match c.indirect.filter(|_| q.arg != NO_ARG) {
                Some((sig, args)) => l.ExecuteIndirect(sig, 1, args, u64::from(q.arg) * u64::from(gpucull::ARG_BYTES), None, 0),
                None => l.DrawIndexedInstanced(q.indices, q.instances, 0, 0, 0),
            }
        }
    }
}

impl Pass {
    /// `ourlight=on`: its pass at `w.rts` targets (the walk's lit from here);
    /// `metal`: the frame's bridged rows first (their target 2 feeds it).
    pub(super) fn lit_after_bridged(&mut self, l: &ID3D12GraphicsCommandList, list: &FrameList, frame: u64, w: &mut Walk, st: &mut PassStats, light_va: u64, clip_to_world: &Mat) -> std::result::Result<(), String> {
        if self.ourlight.as_ref().is_some_and(|x| x.metal_rtv().is_some()) && !w.bridged_done {
            w.bridged_done = true;
            self.record_bridgeps(l, list, frame, st, &mut w.sub_k, &mut w.chunk_wait)?;
            // SAFETY: our open list.
            unsafe {
                l.SetDescriptorHeaps(&[Some(self.heap.clone())]);
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            }
        }
        self.record_ourlight(l, light_va, w.rts, clip_to_world);
        w.lit_done = true;
        Ok(())
    }

    /// The frame's targets, viewport, topology and heap on a fresh list (as the frame's start set them).
    /// SAFETY (caller): an open list of ours.
    unsafe fn preamble(&self, l: &ID3D12GraphicsCommandList, rts: u32) {
        // SAFETY: as above.
        unsafe {
            let (rtv, dsv) = (self.rtv.GetCPUDescriptorHandleForHeapStart(), self.dsv.GetCPUDescriptorHandleForHeapStart());
            l.OMSetRenderTargets(rts, Some(&rtv), true, Some(&dsv));
            l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: self.width as f32, Height: self.height as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
            l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: self.width as i32, bottom: self.height as i32 }]);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.SetDescriptorHeaps(&[Some(self.heap.clone())]);
        }
    }

    /// Pool entry `i` (made on first use), its allocator reset once its
    /// last submission ended (at most 10 s, the trunk's fence rule).
    fn pool_entry(&mut self, i: usize, frame: u64) -> std::result::Result<(), String> {
        let e = |what: &'static str| move |x: windows::core::Error| format!("frame {frame}: precord {what}: {x}");
        // SAFETY: our device, fence and allocators.
        unsafe {
            while self.prec_pool.len() <= i {
                let a: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e("allocator"))?;
                let l: ID3D12GraphicsCommandList = self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, &self.pso).map_err(e("list"))?;
                l.Close().map_err(e("first Close"))?;
                self.prec_pool.push((a, l, 0));
            }
            let fence = self.prec_pool[i].2;
            if fence > self.fence.GetCompletedValue() {
                let event: HANDLE = CreateEventW(None, false, false, None).map_err(e("CreateEvent"))?;
                self.fence.SetEventOnCompletion(fence, event).map_err(e("SetEventOnCompletion"))?;
                let waited = WaitForSingleObject(event, 10_000);
                let _ = CloseHandle(event);
                if waited == WAIT_TIMEOUT {
                    return Err(format!("frame {frame}: precord: list {i}'s last submission never ended"));
                }
            }
            self.prec_pool[i].0.Reset().map_err(e("allocator Reset"))?;
        }
        Ok(())
    }

    /// The resolve: the loop's draws of one pass (`wire` or solid) as
    /// packets and events; every stat, texture count and pipeline choice of
    /// the loop is made here, in the loop's order.
    pub(super) fn resolve(&mut self, f: &Frame, draws: &[&DrawRecord], wire: bool, w: &mut Walk, st: &mut PassStats, new_psos: &mut usize) -> Resolved {
        let list = f.list;
        let mut r = Resolved { packets: Vec::with_capacity(draws.len()), events: Vec::new(), psos: Vec::new(), start_rts: w.rts, water_done: wire };
        // Heap handles by slot, without holding `self` (as `gpu_slot`).
        let (g0, gs) = (self.gpu_slot(0).ptr, self.gpu_slot(1).ptr - self.gpu_slot(0).ptr);
        let gpu = move |s: u32| g0 + u64::from(s) * gs;
        let t3_frame = gpu(2 + f.turn as u32);
        for d in draws {
            if self.chunk > 0 && st.draws + st.bridged >= w.chunk_from + self.chunk as usize {
                w.chunk_from = st.draws + st.bridged;
                r.events.push((r.packets.len(), Event::Chunk));
            }
            // Water after opaque and decals, before transparent (14); our lighting before it.
            if !r.water_done && TRANSPARENT_STAGES.contains(&d.stage) {
                r.water_done = true;
                r.events.push((r.packets.len(), Event::Water { light: !w.lit_done, rts: w.rts }));
                w.lit_done = true;
                // `gbuf=on`: transparent rows come after the title's G-buffer.
                w.gbuf_done = true;
            }
            let idx = ((*d as *const DrawRecord as usize).wrapping_sub(list.draws.as_ptr() as usize) / std::mem::size_of::<DrawRecord>()) as u32;
            // A bridged row draws from its records, not our geometry.
            if let Some(b) = f.ms_rows.filter(|_| !wire && (idx as usize) < list.draws.len())
                && let Some(it) = b.item(idx)
            {
                let pr = b.prog(it.prog);
                if pr.title_ps {
                    r.events.push((r.packets.len(), Event::Title(idx)));
                    st.bridged_title_ps += 1;
                } else if let (Some(pso), Some(slot), Some(j), Some((ibv, n)), Some((_, ring_bytes))) = (&self.rec_pso, self.rec_slot, pr.join, b.indices(it), b.records()) {
                    let c = draw_constants(d, f.cam);
                    let mut k = [0u32; 58];
                    for (i, v) in c.iter().enumerate() {
                        k[i] = v.to_bits();
                    }
                    let (ti, srgb) = self.tex.lookup(d.material.albedo_desc);
                    k[48] = ti.min(TEX_SLOTS - 1);
                    k[49] = u32::from(d.material.alpha_clip) | u32::from(srgb) << 1;
                    let jc = j.constants(&pr.p.layout);
                    (k[32], k[33], k[36], k[37], k[39], k[56]) = (ring_bytes.min(u64::from(u32::MAX)) as u32, b.record_at(it) as u32, pr.p.layout.record_bytes(), jc[0], jc[1], jc[2]);
                    k[35] = u32::from(d.ped) << 1 | u32::from(self.id_mode) << 2 | if w.lit_done { 0 } else { ourlight::DEFERRED };
                    k[57] = idx + 1;
                    if let Some(s) = list.sun {
                        for (j, v) in s.iter().enumerate() {
                            k[40 + j] = v.to_bits();
                        }
                    }
                    if let Some(s) = list.sky.filter(|_| self.sky) {
                        for (j, v) in s.fog.iter().enumerate() {
                            k[52 + j] = v.to_bits();
                        }
                    }
                    let p = pso_at(&mut r.psos, pso);
                    r.packets.push(Packet { root: ROOT_RAW, pso: p, rts: w.rts as u8, slot: gpu(slot), t3: t3_frame, ibv, vbv: D3D12_VERTEX_BUFFER_VIEW::default(), k, indices: n, instances: 1, arg: NO_ARG });
                    self.tex.count_draw(d.material.albedo_desc);
                } else {
                    continue;
                }
                st.bridged += 1;
                continue;
            }
            let Some((ibv, vbv, indices, raw)) = self.cache.get_or_make(d.geom, list.frame, || None).map(|g| (g.ibv, g.vbv, g.indices, g.raw)) else {
                if !wire {
                    st.skipped += 1;
                    // Why, per draw.
                    st.missing[match self.geom.missing(&d.geom) {
                        Some(geometry::Missing::Waiting) => 0,
                        Some(geometry::Missing::Full) => 1,
                        _ => 2,
                    }] += 1;
                }
                continue;
            };
            if indices == 0 || (wire && raw.is_none()) {
                continue;
            }
            let c = draw_constants(d, f.cam);
            // The title's state for this draw (raw path, solid pass); without one, the decal pipeline for stage 74.
            let want = if wire {
                Bound::Wire
            } else if self.id_mode {
                Bound::Raw
            } else if let Some(s) = effective_state(self.face_cull, list, d, f.device_rows.contains(&idx)).filter(|_| self.states && raw.is_some()) {
                st.culled_faces += usize::from(s.cull != 0);
                // New pipelines within the frame's budget; the rest draw with our defaults until a later frame makes them.
                if !self.pso_async && !self.state_psos.contains_key(&s) && self.state_psos.len() < MAX_STATE_PSOS && *new_psos < self.pso_budget {
                    let tp = std::time::Instant::now();
                    *new_psos += 1;
                    match pipeline_state(&self.device, &self.raw_root, &self.raw_shaders.0, &self.raw_shaders.1, self.raw_plain_ps.as_ref(), &s) {
                        Ok(p) => {
                            self.state_psos.insert(s, p);
                        }
                        Err(err) => xnw_core::warn!("world render: state pipeline {s:?} failed: {err}"),
                    }
                    st.pso_ms += tp.elapsed().as_secs_f64() * 1e3;
                }
                if self.state_psos.contains_key(&s) {
                    st.stated += 1;
                    Bound::State(s)
                } else if DECAL_STAGES.contains(&d.stage) {
                    Bound::Decal
                } else {
                    Bound::Raw
                }
            } else if raw.is_some() && DECAL_STAGES.contains(&d.stage) {
                Bound::Decal
            } else {
                Bound::Raw
            };
            // A dual-source state blends into one target only (bound with three, WARP drew it wrong).
            if self.gbuf.is_some() {
                w.rts = if w.gbuf_done || matches!(want, Bound::State(s) if gbuf::dual_source(&s.blend)) { 1 } else { 3 };
            }
            let mut k = [0u32; 58];
            let packet = match raw {
                Some((slot, vb_bytes)) => {
                    let pso = match want {
                        Bound::Raw => &self.raw_pso,
                        Bound::Wire => &self.wire_pso,
                        Bound::Decal => &self.decal_pso,
                        Bound::State(s) => &self.state_psos[&s],
                    };
                    let pso = pso_at(&mut r.psos, pso);
                    let (pf, pb) = match list.palettes.get(d.palette as usize) {
                        Some(p) if d.palette != NO_PALETTE && self.skin => match d.bone_count {
                            0 => (f.first[d.palette as usize], (p.len() / 12) as u32),
                            // Bone k of the draw is palette entry curIndex + k (bounded by palBytes in the shader).
                            n => (f.first[d.palette as usize] + 3 * d.bone_first, n),
                        },
                        _ => (0, 0),
                    };
                    st.skinned += usize::from(pb > 0 && !wire);
                    for (i, v) in c.iter().enumerate() {
                        k[i] = v.to_bits();
                    }
                    // The albedo entry of our table (0: flat) and the material flags.
                    let (ti, srgb) = self.tex.lookup(d.material.albedo_desc);
                    k[48] = ti.min(TEX_SLOTS - 1);
                    k[49] = u32::from(d.material.alpha_clip) | u32::from(srgb) << 1;
                    // DEC2: a decal state's pipeline blends the albedo target by the row's alpha.
                    if matches!(want, Bound::State(s) if gbuf::decal_rule(&s.blend)) {
                        k[49] |= gbuf::DECAL_ALPHA;
                    }
                    // CAR1: a vehicle paint-class row's clearcoat (`carshade=on`).
                    if d.material.coat && carshade::on() {
                        k[49] |= carshade::COAT;
                    }
                    // Layer entries (each checked < 256 by the table) and flags.
                    let has_layers = d.material.layers.iter().any(|s| *s != 0);
                    let debug = self.layers_debug && !has_layers && LAYER_DEBUG_STAGES.contains(&d.stage);
                    if self.layers != 0 && (has_layers || debug) {
                        k[50] = d.material.layers.iter().enumerate().map(|(j, s)| textures::layer_entry(self.tex.lookup(*s).0) << (8 * j)).fold(0, |a, b| a | b);
                        k[51] = 1 | (self.layers & 3) << 1 | u32::from(debug) << 3;
                        st.layered += usize::from(!wire);
                    }
                    // The sample mask: lanes kept in bits 4-7, bit 8 on.
                    if self.stipple && d.material.stipple != 0 {
                        k[51] |= 0x100 | u32::from(d.material.stipple & 0xF) << 4;
                        st.stippled += usize::from(!wire);
                    }
                    // INT1: albedo.w (k[31], `draw_constants`' textured flag, unread
                    // by RAW_SHADER) is the draw's material table in t2 at dword
                    // k[31] - 1 (0: none); such a draw takes no colour A.
                    // WHL1: not on a device-culled row (t2 is the cull's output there).
                    k[31] = 0;
                    let culled = f.culled.and_then(|c| c.slot_of(idx)).is_some();
                    let vehmat = self.vehmat && !wire && !culled && d.material.vehmat.is_some_and(|at| (at as usize + 256) * 4 <= f.inst_bytes as usize);
                    if vehmat && let Some(at) = d.material.vehmat {
                        k[31] = at + 1;
                        st.vehmat += 1;
                        // WHL1: the table's colour is the albedo; a vehicle 2lyr row's layers
                        // (paint and dirt, as the body's, which drop them at the hook) would
                        // blend the mask textures over it (`_wheel_inst` rims stayed yellow).
                        if k[51] & 1 != 0 {
                            (k[50], k[51]) = (0, k[51] & !0xF);
                            st.layered -= 1;
                        }
                        // NPC1 (`slodtint=on`): a far ped's block; the vertex program picks k.
                        if d.material.slod != 0 {
                            k[51] |= crate::vehmat::SLOD_FLAG | if d.material.slod == 2 { crate::vehmat::SLOD_RGB } else { 0 };
                        }
                        // POL2 (`vehlivery=on`): the livery section and its two masks
                        // (entries < 256, as layers: the table asks for them first).
                        if d.material.livery_at != 0 && d.material.livery_at < 0x1_0000 {
                            let e = |s: u64| textures::layer_entry(self.tex.lookup(s).0);
                            k[50] = e(d.material.livery[0]) | e(d.material.livery[1]) << 8 | d.material.livery_at << 16;
                            k[51] |= crate::vehmat::LIVERY_FLAG | crate::vehmat::livery_view();
                        }
                    }
                    // Ped tint (palette entry, rows) and vehicle paint (colour A in albedo.rgb).
                    if !wire && (d.material.tint_palette != 0 || d.material.paint_on) {
                        let entry = if d.material.tint_palette != 0 { self.tex.lookup(d.material.tint_palette).0 } else { 0 };
                        let (t, p) = crate::tint_paint_constants(&d.material, entry, k[51] & 1 != 0, self.tint, self.paint && !vehmat);
                        if let Some((l, f)) = t {
                            (k[50], k[51]) = (l, k[51] | f);
                            k[49] |= 8;
                            st.tinted += 1;
                        }
                        if let Some(a) = p {
                            for (j, v) in a.iter().enumerate() {
                                k[28 + j] = v.to_bits();
                            }
                            k[49] |= 4;
                            st.painted += 1;
                        }
                    }
                    if !wire {
                        self.tex.count_draw(d.material.albedo_desc);
                    }
                    k[32] = vb_bytes;
                    k[33] = pf;
                    k[34] = pb;
                    k[35] = u32::from(wire) | u32::from(d.ped) << 1 | u32::from(self.id_mode) << 2 | (self.skin_rules & 0xFF) << 8 | if w.lit_done || wire { 0 } else { ourlight::DEFERRED };
                    // The draw's index in the list + 1 (ID mode).
                    k[57] = idx + 1;
                    // The remap's dword index in t1 (relative to the draw's palette floats).
                    k[56] = match list.palettes.get(d.palette as usize) {
                        Some(_) if d.remap != NO_REMAP && pb > 0 => 4 * f.first[d.palette as usize] + d.remap,
                        _ => u32::MAX,
                    };
                    st.remapped += usize::from(k[56] != u32::MAX && !wire);
                    k[36] = f.pal_bytes;
                    // Instances: clamped to what this frame's buffer holds.
                    let first = d.inst_first.min(f.inst_total);
                    let count = d.inst_count.min(f.inst_total - first).min(MAX_INSTANCES);
                    k[37] = first;
                    k[38] = count;
                    k[39] = f.inst_bytes;
                    // A device-culled row reads the cull's output: its reserved run, the survivors drawn indirectly.
                    let cull_slot = f.culled.and_then(|c| c.slot_of(idx));
                    let mut t3 = t3_frame;
                    if let (Some(c), Some(_)) = (f.culled, cull_slot) {
                        k[37] = d.inst_first;
                        k[38] = d.inst_count.min(MAX_INSTANCES);
                        k[39] = c.out_bytes.min(u64::from(u32::MAX)) as u32;
                        t3 = gpu(c.view_slot);
                    }
                    if let Some(s) = list.sun {
                        for (j, v) in s.iter().enumerate() {
                            k[40 + j] = v.to_bits();
                        }
                    }
                    st.instanced += usize::from((count > 0 || cull_slot.is_some()) && !wire);
                    st.instances += u64::from(count) * u64::from(!wire && cull_slot.is_none());
                    if let Some(s) = list.sky.filter(|_| self.sky) {
                        for (j, v) in s.fog.iter().enumerate() {
                            k[52 + j] = v.to_bits();
                        }
                    }
                    // A pre-skinned draw reads its copy: positions already skinned, no palette.
                    let slot = match f.pre.get(&idx) {
                        Some(&(s, bytes)) => {
                            (k[32], k[34], k[56]) = (bytes, 0, u32::MAX);
                            s
                        }
                        None => slot,
                    };
                    let arg = match (cull_slot, f.culled) {
                        (Some(s), Some(_)) => s,
                        _ => NO_ARG,
                    };
                    Packet { root: ROOT_RAW, pso, rts: w.rts as u8, slot: gpu(slot), t3, ibv, vbv, k, indices, instances: count.max(1), arg }
                }
                None => {
                    for (i, v) in c.iter().enumerate() {
                        k[i] = v.to_bits();
                    }
                    let pso = pso_at(&mut r.psos, &self.pso);
                    Packet { root: ROOT_PLAIN, pso, rts: w.rts as u8, slot: 0, t3: 0, ibv, vbv, k, indices, instances: 1, arg: NO_ARG }
                }
            };
            r.packets.push(packet);
            if !wire {
                st.draws += 1;
                st.triangles += u64::from(indices / 3);
            }
        }
        r
    }

    /// Records the resolved pass on the open list `l` (the pass's own,
    /// `self.list`), its events between the packets; a long run on
    /// `self.precord` lists. `l` is open again at the end.
    /// SAFETY (caller): `l` is `self.list`, open; the frame's resources live until its fence.
    pub(super) unsafe fn record_resolved(&mut self, l: &ID3D12GraphicsCommandList, r: &Resolved, f: &Frame, w: &mut Walk, st: &mut PassStats, frame: u64) -> std::result::Result<(), String> {
        let at = |what: String| move |e: windows::core::Error| format!("frame {frame}: {what}: {e}");
        let (raw_root, root) = (self.raw_root.clone(), self.root.clone());
        // SAFETY: our heaps.
        let (rtv, dsv) = unsafe { (self.rtv.GetCPUDescriptorHandleForHeapStart(), self.dsv.GetCPUDescriptorHandleForHeapStart()) };
        let ctx = Ctx {
            raw_root: &raw_root,
            root: &root,
            psos: &r.psos,
            t2: self.gpu_slot(f.turn as u32),
            t4: f.tex_table,
            light_va: f.light_va,
            rtv,
            dsv,
            indirect: f.culled.map(|c| (&c.signature, &c.args)),
        };
        let mut b = Bind::fresh(r.start_rts);
        let mut from = 0usize;
        let mut events = r.events.iter();
        loop {
            let next = events.next();
            let end = next.map_or(r.packets.len(), |e| e.0);
            let run = &r.packets[from..end];
            from = end;
            let lists = (self.precord as usize).min(run.len() / MIN_RANGE);
            if lists >= 2 && self.chunk == 0 {
                let tp = std::time::Instant::now();
                // The list so far to the card (not waited for), then the run's lists in order.
                self.stamp_close(l, w.sub_k);
                // SAFETY: our open list.
                unsafe { l.Close() }.map_err(at("list Close before a parallel run".into()))?;
                self.send(l, frame)?;
                (st.submits, w.sub_k) = (st.submits + 1, w.sub_k + 1);
                // This run's entries: its lists, then our list's next allocator (distinct per run of the frame).
                let base = self.prec_next;
                self.prec_next += lists + 1;
                for i in base..=base + lists {
                    self.pool_entry(i, frame)?;
                }
                let pool: Vec<ID3D12GraphicsCommandList> = (0..lists).map(|i| self.prec_pool[base + i].1.clone()).collect();
                let per = run.len().div_ceil(lists);
                let rts0: Vec<u32> = (0..lists).map(|i| if i == 0 { b.rts } else { u32::from(run[i * per - 1].rts) }).collect();
                for (i, pl) in pool.iter().enumerate() {
                    // SAFETY: a reset allocator of the pool; our pipeline.
                    unsafe { pl.Reset(&self.prec_pool[base + i].0, &self.pso) }.map_err(at(format!("precord list {i} Reset")))?;
                    // SAFETY: an open list of ours.
                    unsafe { self.preamble(pl, rts0[i]) };
                }
                let job = Shared((&ctx, &pool, run, &rts0));
                std::thread::scope(|s| {
                    let job = &job;
                    let mut hands = Vec::new();
                    for i in 1..lists {
                        hands.push(s.spawn(move || {
                            let (c, pool, run, rts0) = *job.get();
                            let mut bi = Bind::fresh(rts0[i]);
                            // SAFETY: this thread's own open list; the frame's resources live to its fence.
                            unsafe { record(&pool[i], c, &run[i * per..((i + 1) * per).min(run.len())], &mut bi) };
                        }));
                    }
                    let (c, pool, run, rts0) = *job.get();
                    let mut b0 = Bind::fresh(rts0[0]);
                    // SAFETY: as above, on this thread.
                    unsafe { record(&pool[0], c, &run[..per.min(run.len())], &mut b0) };
                    for h in hands {
                        let _ = h.join();
                    }
                });
                let mut sent: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(lists);
                for (i, pl) in pool.iter().enumerate() {
                    // SAFETY: recorded above.
                    unsafe { pl.Close() }.map_err(at(format!("precord list {i} Close ({} draws)", run.len())))?;
                    sent.push(Some(windows::core::Interface::cast(pl).map_err(at("precord list cast".into()))?));
                }
                // SAFETY: our queue and fence; closed lists of ours.
                unsafe {
                    self.queue.ExecuteCommandLists(&sent);
                    self.fence_value += 1;
                    self.queue.Signal(&self.fence, self.fence_value).map_err(at("precord queue Signal".into()))?;
                }
                self.split_sent = self.fence_value;
                for e in &mut self.prec_pool[base..base + lists] {
                    e.2 = self.fence_value;
                }
                st.submits += 1;
                // Our list again on the next allocator; its fence is set when the frame's last submission is sent.
                let ai = base + lists;
                self.prec_cont.push(ai);
                // SAFETY: a reset allocator of the pool; our list was sent above.
                unsafe { self.list.Reset(&self.prec_pool[ai].0, &self.pso) }.map_err(at("list Reset after a parallel run".into()))?;
                self.stamp_open(l, w.sub_k);
                b = Bind::fresh(u32::from(run[run.len() - 1].rts));
                // SAFETY: our open list.
                unsafe { self.preamble(l, b.rts) };
                st.precord_lists += lists;
                st.precord_ms += tp.elapsed().as_secs_f64() * 1e3;
            } else {
                // SAFETY: our open list.
                unsafe { record(l, &ctx, run, &mut b) };
            }
            let Some((_, ev)) = next else { break };
            match ev {
                Event::Chunk => {
                    // Every N draws the list so far is sent and waited for; the new list gets the frame's targets, viewport and heap again.
                    self.stamp_close(l, w.sub_k);
                    // SAFETY: our open list.
                    unsafe { l.Close() }.map_err(at(format!("chunk list Close after {} draws", w.chunk_from)))?;
                    let (wt, g) = self.send_wait(l, w.sub_k, "pass chunk", frame)?;
                    (w.chunk_wait, st.submits, w.sub_k) = (w.chunk_wait + wt, st.submits + 1, w.sub_k + 1);
                    (st.submit_wait_ms, st.submit_gpu_ms) = (st.submit_wait_ms.max(wt), st.submit_gpu_ms.max(g.unwrap_or(0.0)));
                    // SAFETY: the chunk was waited for.
                    unsafe {
                        self.alloc.Reset().map_err(at("chunk allocator Reset".into()))?;
                        self.list.Reset(&self.alloc, &self.pso).map_err(at("chunk list Reset".into()))?;
                    }
                    self.stamp_open(l, w.sub_k);
                    // SAFETY: our open list.
                    unsafe { self.preamble(l, b.rts) };
                }
                Event::Water { light, rts } => {
                    if *light {
                        // At the targets bound there (the walk's own `rts` is already the pass's last).
                        let last = std::mem::replace(&mut w.rts, *rts);
                        let r = self.lit_after_bridged(l, f.list, frame, w, st, f.light_va, f.clip_to_world);
                        w.rts = last;
                        r?;
                    }
                    st.water = self.record_water(f.list, f.cam).map_err(at("water grid upload".into()))?;
                }
                Event::Title(idx) => {
                    if let Some(bm) = f.ms_rows
                        && let Some(it) = bm.item(*idx)
                    {
                        bm.record_title(l, it);
                    }
                    // SAFETY: our open list.
                    unsafe { l.SetDescriptorHeaps(&[Some(self.heap.clone())]) };
                }
            }
            // Our root, tables and pipeline are bound again at the next packet.
            b = Bind::fresh(b.rts);
        }
        Ok(())
    }
}

impl Pass {
    /// PERF3 `bridgepar`: the frame's bridged rows prepared from here on a
    /// thread of their own (none with `chunk`, before the grey stand-ins
    /// exist, or without rows: the serial record). A preparation left from
    /// an earlier frame (its pass ended early) is let go first.
    /// Called once the last frame's fence was reached.
    pub(super) fn bridge_prep_start(&mut self, list: &FrameList) -> std::result::Result<(), String> {
        let t = std::time::Instant::now();
        let r = self.bridge_prep_spawn(list);
        recprof::add(recprof::B_START, t.elapsed().as_secs_f64() * 1e3);
        r
    }

    fn bridge_prep_spawn(&mut self, list: &FrameList) -> std::result::Result<(), String> {
        if let Some(p) = self.bridge_prep.take() {
            self.bridge_prep_drop(p);
        }
        if !self.bridgepar || !self.bridge_on || self.chunk > 0 {
            return Ok(());
        }
        let Some(f) = list.bridged.as_ref().filter(|f| !f.rows.is_empty()) else { return Ok(()) };
        // Rows that take our pre-skin range are bound after it is recorded, on the pass's thread.
        if self.preskin.bind && f.rows.iter().any(|r| super::preskin_gpu::reads_range(&r.1)) {
            return Ok(());
        }
        let Some(grey) = self.bridge_grey.clone() else { return Ok(()) };
        let shapes = self.bridge_grey_shapes.clone();
        if shapes.len() != 5 {
            return Ok(());
        }
        let (fixed, shift, skip) = self.bridge_aim(list, f);
        let frame = fixed.map_or_else(|| f.clone(), std::sync::Arc::new);
        let (words, wp, wlen) = self.bridge_words_for(frame.words_end)?;
        // SAFETY: our heaps.
        let (rtv, dsv) = unsafe { (self.rtv.GetCPUDescriptorHandleForHeapStart(), self.dsv.GetCPUDescriptorHandleForHeapStart()) };
        let (metal, normal, emissive) = (self.ourlight.as_ref().and_then(|x| x.metal_rtv()), self.ourlight.as_ref().and_then(|x| x.normal_rtv()), self.ourlight.as_ref().and_then(|x| x.emissive_rtv()));
        let (width, height, bias) = (self.width, self.height, bridge_bias());
        let (ttx, trx) = std::sync::mpsc::channel::<Shared<TexSnap>>();
        let (ltx, lrx) = std::sync::mpsc::channel::<Shared<ID3D12GraphicsCommandList>>();
        let gpu = std::mem::take(&mut self.bridge_gpu);
        let job = Shared((gpu, self.device.clone(), frame.clone(), grey, shapes, words, wp as usize, trx, lrx));
        let done = std::thread::Builder::new()
            .name("xnw bridged prep".into())
            .spawn(move || {
                xnw_core::win::helper_thread();
                let job = job;
                let (mut gpu, dev, frame, grey, shapes, words, wp, trx, lrx) = job.0;
                let snap: std::sync::OnceLock<Option<TexSnap>> = std::sync::OnceLock::new();
                let get = |h: u64| snap.get_or_init(|| trx.recv().ok().map(|s| s.0)).as_ref().and_then(|s| s.get(&h));
                let misses = std::cell::Cell::new(0u32);
                let texture = |h: u64| match get(h) {
                    Some(e) => e.0.clone(),
                    None => {
                        misses.set(misses.get() + u32::from(h != 0));
                        None
                    }
                };
                let flat = |h: u64| get(h).map_or(usize::MAX, |e| e.1);
                let t = bridged::Target { rtv, dsv, width, height, shift, bias, skip: &skip, texture: &texture, flat: &flat, grey: &grey, grey_shapes: [&shapes[0], &shapes[1], &shapes[2], &shapes[3], &shapes[4]], words: (&words, wlen, wp as *mut u8), metal, normal, emissive, preskin: None };
                let mut keep = Vec::new();
                let mut got: Option<std::time::Instant> = None;
                let mut take = || {
                    let l = lrx.recv().map_err(|_| "bridgepar: the pass gave the rows no list".to_string())?;
                    got = Some(std::time::Instant::now());
                    Ok(l.0)
                };
                let st = gpu.record_with(&dev, &mut take, &frame, &t, &mut keep);
                let list_ms = got.map_or(0.0, |g| g.elapsed().as_secs_f64() * 1e3);
                Shared(PrepOut { gpu, keep, st, misses: misses.get(), list_ms })
            })
            .map_err(|e| format!("bridgepar: thread: {e}"))?;
        self.bridge_prep = Some(BridgePrep { frame, tex: Some(ttx), list: ltx, done });
        Ok(())
    }

    /// PERF3: the rows' textures to their preparing thread, as the frame's table has them now.
    pub(super) fn bridge_prep_textures(&mut self) {
        if let Some(p) = self.bridge_prep.as_mut()
            && let Some(tx) = p.tex.take()
        {
            let t = std::time::Instant::now();
            let _ = tx.send(Shared(tex_snap(&self.tex, &p.frame)));
            recprof::add(recprof::B_START, t.elapsed().as_secs_f64() * 1e3);
        }
    }

    /// PERF3: the prepared rows recorded on the open list `l` by their
    /// thread (this one waits; it records nothing meanwhile); their stats.
    pub(super) fn bridge_prep_finish(&mut self, mut p: BridgePrep, l: &ID3D12GraphicsCommandList) -> std::result::Result<bridged::Stats, String> {
        let t = std::time::Instant::now();
        if let Some(tx) = p.tex.take() {
            let _ = tx.send(Shared(tex_snap(&self.tex, &p.frame)));
        }
        // The thread asks for it only when it has rows to record.
        let _ = p.list.send(Shared(l.clone()));
        let out = self.bridge_prep_join(p)?;
        recprof::add(recprof::B_WAIT, (t.elapsed().as_secs_f64() * 1e3 - out.list_ms).max(0.0));
        out.st
    }

    /// PERF3: a preparation whose rows are not drawn (the pass ended early,
    /// or the frame took another road): its thread ends without a list, and
    /// the kept inputs it placed start again next frame (their copies were
    /// never recorded).
    pub(super) fn bridge_prep_drop(&mut self, p: BridgePrep) {
        if self.bridge_prep_join(p).is_ok() {
            self.bridge_gpu.kept.restart = Some(self.bridge_gpu.kept.size);
            // The frame's in-place changes went to the stand-in it held meanwhile: every range is made again.
            self.bridge_gpu.zc_changed(&xnw_gpu::shared::WorldChanges { overflow: true, ..Default::default() });
        }
    }

    fn bridge_prep_join(&mut self, p: BridgePrep) -> std::result::Result<PrepOut, String> {
        let BridgePrep { tex, list, done, .. } = p;
        drop((tex, list));
        match done.join() {
            Ok(Shared(mut out)) => {
                self.bridge_gpu = std::mem::take(&mut out.gpu);
                self.bridge_keep.append(&mut out.keep);
                if out.misses > 0 {
                    xnw_core::warn!("bridgepar: {} texture lookups of the bridged rows were not in the frame's snapshot (drawn grey)", out.misses);
                }
                Ok(out)
            }
            Err(_) => {
                self.bridge_gpu = bridged::Gpu::default();
                Err("bridgepar: the preparing thread panicked".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridgepar_is_on_unless_switched_off() {
        assert!(parse_bridgepar(""));
        assert!(parse_bridgepar("screen,precord=on"));
        assert!(!parse_bridgepar("screen,bridgepar=off"));
        assert!(!parse_bridgepar("bridgepar=0"));
        assert!(parse_bridgepar("bridgepar=off, bridgepar=on"));
    }
}
