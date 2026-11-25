//! The title's own mesh programs in our pass (`bridgems`). For the
//! rows a [`Batch`] holds, each mesh program runs in compute form (the mesh fallback)
//! and writes its records and index triples into one
//! ring per frame. Every row's dispatch is recorded first, then one barrier
//! (about 4 us fixed per mesh draw when each pays its own), and then
//! each row draws in its place in the pass from its ring segment:
//! - with our pixel stage: `vs_rec` of [`super::RAW_SHADER`] reads the
//!   record's position, texture coordinate and normal ([`Join`]) and our
//!   pixel shader shades it as it would our own draw of the row;
//! - or, for a row whose pixel program is wanted too (`bridgeps`), with the
//!   title's pixel program behind the pass-through vertex shader.
//!
//! Pipelines are made once per program pair and kept across frames
//! ([`Batch::program`]). Bindings are the bridge audit's ([`bridge::bind`]);
//! each row keeps its own view and sampler heap (the translated programs
//! index the heap by the title's own indices, so rows cannot share one).

use std::collections::HashMap;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use xnw_shader::emit::mesh::Layout;

use super::bridge::{self, Binding, Pipes, Programs, Uploaded};

/// Where our pixel stage's inputs sit in a mesh program's record: rows
/// (16 B each) and first channels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Join {
    pub position: u32,
    pub uv: Option<(u32, u8)>,
    pub normal: Option<(u32, u8)>,
}

impl Join {
    /// By semantic: `SV_Position`; the texture coordinate is `TEXCOORD0`
    /// (its first two channels), the normal `NORMAL0`. None without a
    /// position.
    pub fn of(layout: &Layout) -> Option<Join> {
        let find = |name: &str, index: u32| layout.elements.iter().find(|e| e.kind == 0 && e.name.eq_ignore_ascii_case(name) && e.index == index).map(|e| (e.row, e.col));
        Some(Join { position: layout.position_row()?, uv: find("TEXCOORD", 0), normal: find("NORMAL", 0) })
    }

    /// The record constants of `vs_rec` (RAW_SHADER dwords 37, 39, 56):
    /// max vertices | vertex stride << 16; position row | uv row << 8 |
    /// uv channel << 16 | uv on << 24; normal row | channel << 8 | on << 16.
    pub fn constants(&self, layout: &Layout) -> [u32; 3] {
        let pair = |p: Option<(u32, u8)>, shift: u32| p.map_or(0, |(r, c)| (r & 0xFF) | u32::from(c) << 8 | 1 << shift);
        let uv = pair(self.uv, 16);
        [layout.max_vertices & 0xFFFF | layout.vertex_stride() << 16, (self.position & 0xFF) | (uv & 0xFFFF) << 8 | (uv >> 16) << 24, pair(self.normal, 16)]
    }
}

/// One program pair on the device, kept across frames.
pub struct Prog {
    pub p: Programs,
    pub pipes: Pipes,
    pub join: Option<Join>,
    /// Drawn with the title's pixel program (else with ours).
    pub title_ps: bool,
}

/// One row of this frame: its binding on the device and its ring segment.
pub struct Item {
    pub row: u32,
    pub prog: usize,
    pub up: Uploaded,
    rec_at: u64,
    idx_at: u64,
}

/// What a frame's batch did.
#[derive(Clone, Debug, Default)]
pub struct BatchStats {
    pub rows: usize,
    pub groups: u64,
    pub programs: usize,
    pub pipelines_made: usize,
    pub pipeline_ms: f64,
    pub upload_bytes: u64,
    pub upload_ms: f64,
    pub ring_bytes: u64,
    /// Bytes rows shared instead of putting on the device again; the largest view heap.
    pub shared_bytes: u64,
    pub heap_slots: u32,
    /// Rows whose large buffer merged into another row's.
    pub merged: u64,
    /// New bytes: large views, small views, textures, root views.
    pub new: [u64; 4],
}

/// The bridged rows of one frame and the pipelines of every frame.
#[derive(Default)]
pub struct Batch {
    progs: Vec<Prog>,
    keys: HashMap<([u128; 2], bool), usize>,
    items: Vec<Item>,
    by_row: HashMap<u32, usize>,
    /// Records and index rings (device, COMMON between frames) and their capacities.
    ring: Option<(ID3D12Resource, ID3D12Resource, u64, u64)>,
    rec_used: u64,
    idx_used: u64,
    copies: Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    /// This frame's buffers and textures by content.
    shared: bridge::Shared,
    pub stats: BatchStats,
}

/// Ring segments start at a root view's alignment.
const SEGMENT_ALIGN: u64 = 256;

impl Batch {
    /// The program pair `key` (mesh and pixel container hashes; whether the
    /// title's pixel program draws), made by `make` and its pipelines the
    /// first time only: its index.
    pub fn program(&mut self, dev: &ID3D12Device, key: ([u128; 2], bool), make: impl FnOnce() -> Result<Programs, String>) -> Result<usize, String> {
        if let Some(&k) = self.keys.get(&key) {
            return Ok(k);
        }
        let t = std::time::Instant::now();
        let p = make()?;
        let title_ps = key.1;
        let (formats, depth) = if title_ps { (vec![(0, DXGI_FORMAT_R8G8B8A8_UNORM.0)], Some((true, D3D12_COMPARISON_FUNC_GREATER.0))) } else { (Vec::new(), None) };
        let pipes = Pipes::new(dev, &p, title_ps, &formats, depth)?;
        let join = Join::of(&p.layout);
        self.stats.pipelines_made += 1;
        self.stats.pipeline_ms += t.elapsed().as_secs_f64() * 1e3;
        self.progs.push(Prog { p, pipes, join, title_ps });
        self.keys.insert(key, self.progs.len() - 1);
        Ok(self.progs.len() - 1)
    }

    pub fn prog(&self, k: usize) -> &Prog {
        &self.progs[k]
    }

    /// Starts a frame: no rows, the copy list open.
    pub fn begin(&mut self, dev: &ID3D12Device) -> Result<(), String> {
        let e = |e: windows::core::Error| e.message().to_string();
        self.items.clear();
        self.by_row.clear();
        self.shared = bridge::Shared::default();
        (self.rec_used, self.idx_used) = (0, 0);
        let pipelines = (self.stats.pipelines_made, self.stats.pipeline_ms);
        self.stats = BatchStats { pipelines_made: pipelines.0, pipeline_ms: pipelines.1, ..Default::default() };
        // SAFETY: plain object creation and a reset of our own list.
        unsafe {
            match &self.copies {
                Some((a, l)) => {
                    a.Reset().map_err(e)?;
                    l.Reset(a, None).map_err(e)?;
                }
                None => {
                    let a: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
                    let l: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None).map_err(e)?;
                    self.copies = Some((a, l));
                }
            }
        }
        Ok(())
    }

    /// Row `row` of the list (its index) drawn by program `prog` with
    /// binding `b`: put on the device, given its ring segment.
    pub fn add(&mut self, dev: &ID3D12Device, row: u32, prog: usize, b: &Binding) -> Result<(), String> {
        let t = std::time::Instant::now();
        let (_, list) = self.copies.as_ref().ok_or("Batch::add before Batch::begin")?;
        let pr = &self.progs[prog];
        let up = bridge::upload(dev, list, b, &pr.pipes, &mut self.shared)?;
        self.stats.heap_slots = self.stats.heap_slots.max(b.views.heap_slots);
        let groups = u64::from(b.groups.max(1));
        let (rec_at, idx_at) = (self.rec_used, self.idx_used);
        self.rec_used += (groups * u64::from(pr.p.layout.record_bytes())).next_multiple_of(SEGMENT_ALIGN);
        self.idx_used += (groups * u64::from(pr.p.layout.index_bytes())).next_multiple_of(SEGMENT_ALIGN);
        self.stats.rows += 1;
        self.stats.groups += u64::from(b.groups);
        self.stats.upload_bytes += up.bytes;
        self.stats.upload_ms += t.elapsed().as_secs_f64() * 1e3;
        self.by_row.insert(row, self.items.len());
        self.items.push(Item { row, prog, up, rec_at, idx_at });
        Ok(())
    }

    /// Ends the frame's uploads: the texture copies run on `queue` (waited
    /// for), and the rings grow to what the rows need.
    pub fn finish(&mut self, dev: &ID3D12Device, queue: &ID3D12CommandQueue) -> Result<(), String> {
        let e = |e: windows::core::Error| e.message().to_string();
        let t = std::time::Instant::now();
        if let Some((_, list)) = &self.copies {
            // SAFETY: our list, closed and run once; the fence wait keeps
            // every uploaded resource alive past the GPU's use.
            unsafe {
                list.Close().map_err(e)?;
                queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(list).map_err(e)?)]);
                let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
                queue.Signal(&fence, 1).map_err(e)?;
                while fence.GetCompletedValue() < 1 {
                    dev.GetDeviceRemovedReason().map_err(|x| format!("bridgems uploads: {}", e(x)))?;
                    std::thread::yield_now();
                }
            }
        }
        let (rec, idx) = (self.rec_used.max(SEGMENT_ALIGN), self.idx_used.max(SEGMENT_ALIGN));
        if self.ring.as_ref().is_none_or(|r| r.2 < rec || r.3 < idx) {
            let (rec, idx) = (rec.next_power_of_two(), idx.next_power_of_two());
            let make = |n| bridge::buffer(dev, n, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
            self.ring = Some((make(rec)?, make(idx)?, rec, idx));
        }
        self.stats.ring_bytes = self.rec_used + self.idx_used;
        (self.stats.shared_bytes, self.stats.merged, self.stats.new) = (self.shared.saved, self.shared.merged, self.shared.new);
        self.stats.programs = self.items.iter().map(|i| i.prog).collect::<std::collections::HashSet<_>>().len();
        self.stats.upload_ms += t.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// The row's item, if the batch draws it.
    pub fn item(&self, row: u32) -> Option<&Item> {
        self.by_row.get(&row).map(|&k| &self.items[k])
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The record ring and its bytes (the join's raw view).
    pub fn records(&self) -> Option<(&ID3D12Resource, u64)> {
        self.ring.as_ref().map(|r| (&r.0, r.2))
    }

    /// Records every row's dispatch, then the one barrier: records to
    /// shader reads, index triples to the index buffer. The caller sets
    /// its own descriptor heaps again afterwards.
    pub fn record_compute(&self, l: &ID3D12GraphicsCommandList) {
        let Some((rec, idx, _, _)) = &self.ring else { return };
        if self.items.is_empty() {
            return;
        }
        // SAFETY: live rings; the caller's list.
        unsafe {
            let (rva, iva) = (rec.GetGPUVirtualAddress(), idx.GetGPUVirtualAddress());
            for it in &self.items {
                bridge::record_dispatch(l, &self.progs[it.prog].pipes, &it.up, rva + it.rec_at, iva + it.idx_at);
            }
            // The rings were promoted from COMMON to UAV by the dispatches (buffers).
            l.ResourceBarrier(&[transition(rec, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE), transition(idx, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDEX_BUFFER)]);
        }
    }

    /// The index buffer view of an item's segment and its index count.
    pub fn indices(&self, it: &Item) -> Option<(D3D12_INDEX_BUFFER_VIEW, u32)> {
        let (_, idx, _, _) = self.ring.as_ref()?;
        let lay = &self.progs[it.prog].p.layout;
        let groups = it.up.groups.max(1);
        // SAFETY: a live ring.
        let va = unsafe { idx.GetGPUVirtualAddress() } + it.idx_at;
        Some((D3D12_INDEX_BUFFER_VIEW { BufferLocation: va, SizeInBytes: groups * lay.index_bytes(), Format: DXGI_FORMAT_R32_UINT }, it.up.groups * lay.max_prims * 3))
    }

    /// An item's first record byte in the ring.
    pub fn record_at(&self, it: &Item) -> u64 {
        it.rec_at
    }

    /// Records an item's draw with the title's pixel program (its own heaps
    /// and root signature: the caller binds its own again afterwards).
    pub fn record_title(&self, l: &ID3D12GraphicsCommandList, it: &Item) {
        let (Some((rec, _, _, _)), Some((ibv, _))) = (&self.ring, self.indices(it)) else { return };
        let pr = &self.progs[it.prog];
        // SAFETY: a live ring.
        let va = unsafe { rec.GetGPUVirtualAddress() } + it.rec_at;
        bridge::record_draw(l, &pr.pipes, &it.up, va, &ibv, pr.p.layout.max_prims);
    }
}

/// Our pass's depth target ([`bridge::Into::depth`], depth-write state)
/// read back as float rows.
pub fn read_depth(host: &xnw_gpu::host::Host, depth: &ID3D12Resource, w: u32, h: u32) -> Result<Vec<f32>, String> {
    let e = |e: windows::core::Error| e.message().to_string();
    let dev = &host.device;
    // SAFETY: D3D12 calls on resources that live until the fence says the GPU is done.
    unsafe {
        let desc = depth.GetDesc();
        let (mut fp, mut total) = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), 0u64);
        dev.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        let back = bridge::buffer(dev, total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        let dw = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        list.ResourceBarrier(&[transition(depth, dw, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&back), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(depth), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.ResourceBarrier(&[transition(depth, D3D12_RESOURCE_STATE_COPY_SOURCE, dw)]);
        list.Close().map_err(e)?;
        host.queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(e)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        host.queue.Signal(&fence, 1).map_err(e)?;
        while fence.GetCompletedValue() < 1 {
            dev.GetDeviceRemovedReason().map_err(|x| format!("reading the depth target back: {}", e(x)))?;
            std::thread::yield_now();
        }
        let mut from = std::ptr::null_mut();
        back.Map(0, None, Some(&mut from)).map_err(e)?;
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h as usize {
            out.extend_from_slice(std::slice::from_raw_parts(from.cast::<u8>().add(fp.Offset as usize + y * fp.Footprint.RowPitch as usize).cast::<f32>(), w as usize));
        }
        back.Unmap(0, None);
        Ok(out)
    }
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

/// The mesh programs `bridgems=on` names: the static families the bridge
/// audit proved against our pass (pk3 f1241: every program with
/// agreeing rows and no `theirs none`; MS_VS_Transform and
/// MS_VS_TransformD bodies, rigid palettes among them) and MSNC_VS_Cable
/// 5f0302c7, which our pass does not draw (first 32 bits of the container
/// hash).
pub const BRIDGE_MS_DEFAULT: &[u32] = &[
    0xb31c_5ed5, 0x07d0_a626, 0x6de1_e27e, 0x1597_f32d, 0x3ce4_627b, 0x7d25_a7f8, 0x6c12_37e5, 0xc1cb_6985, 0x69d5_1347, 0x3e67_78c2, 0x5ff5_f10a, 0x8050_a8f9, 0xcb28_f685, 0x1a59_160b, 0x4064_96ff, 0x180a_ffd1, 0xf050_a767, 0x798d_6c66, 0xf5ef_22c5, 0xde58_a47f, 0x9d46_a4ea, 0x31d7_cb78, 0xdb9f_0449, 0x1254_a27e, 0x00c9_ed1b, 0x0bcf_8cee, 0x248b_38ff, 0xa9bb_0fca, 0x97b4_9886, 0x8d6f_07d1, 0x5378_29db,
    // MSNC_VS_Cable_da99eb95.
    0x5f03_02c7,
];

/// `XNW_WORLD_RENDER=...,bridgems=on|off|<hash>+<hash>...` (default off; the
/// last one wins): the mesh programs (first 32 bits of the container hash,
/// hex) whose rows our pass draws through the title's mesh program.
pub fn parse_switch(v: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for p in v.split(',').filter_map(|p| p.trim().strip_prefix("bridgems=")) {
        out = match p {
            "on" => BRIDGE_MS_DEFAULT.to_vec(),
            "off" => Vec::new(),
            list => list.split('+').filter_map(|h| u32::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()).collect(),
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_shader::emit::mesh::Element;

    #[test]
    fn switch_and_join() {
        assert_eq!(parse_switch("screen,bridgems=on").len(), BRIDGE_MS_DEFAULT.len());
        assert_eq!(parse_switch("bridgems=5f0302c7+b31c5ed5,bridgems=off"), Vec::<u32>::new());
        assert_eq!(parse_switch("bridgems=5f0302c7+0xb31c5ed5"), vec![0x5f03_02c7, 0xb31c_5ed5]);
        assert!(parse_switch("bridgeps=on").is_empty());
        let e = |name: &str, index, kind, row, col| Element { name: name.into(), index, kind, comp: 9, row, col, cols: 4 };
        let lay = Layout { max_vertices: 64, max_prims: 126, rows: 4, elements: vec![e("SV_Position", 0, 3, 0, 0), e("TEXCOORD", 0, 0, 2, 2), e("NORMAL", 0, 0, 3, 0), e("TEXCOORD", 1, 0, 1, 0)] };
        let j = Join::of(&lay).unwrap();
        assert_eq!(j, Join { position: 0, uv: Some((2, 2)), normal: Some((3, 0)) });
        let k = j.constants(&lay);
        assert_eq!(k, [64 | 64 << 16, 0 | 2 << 8 | 2 << 16 | 1 << 24, 3 | 1 << 16]);
    }
}
