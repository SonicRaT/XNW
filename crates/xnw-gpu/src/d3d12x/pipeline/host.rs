//! Host pipelines for title pipelines: the title's graphics state with the
//! translated shaders, under a root signature of our own that the
//! translated shaders' bindings define (`xnw_shader::emit`): per stage one
//! block of root constants for the user data, root views for the title's
//! root views, static samplers for the immediate samplers, and directly
//! indexed heaps.
//!
//! Ported from VIctory `crates/victory-gpu/src/pipeline/host.rs` at
//! xnw/base 5d0516e (VIctory d2a4bae + the heap-indexing flags and
//! Windows-10 retry changes, as merged there), GPL-3.0-or-later:
//! `GraphicsState` 22-231, `HostPipeline::new` 294-589, `root_signature`
//! 653-720. Changed: `tracing` -> xnw-core log; mesh pipelines become
//! the compute fallback pair (no mesh shaders on this card:
//! [`MeshFallback`]); the heap-indexing flags are always set
//! (the shaders' feature flags decide in the base; measured both
//! serialise here) with the base's retry without them. XNW additions:
//! the state of the title's pipeline cache (slots 57/55), decoded to the
//! minimum the first frames need ([`GraphicsState::from_cache`],
//! [`GraphicsState::apply_overrides`]), from five-star-recomp
//! "Fixed-function state" (GPL-3.0) and AMD's
//! RDNA2 register reference (field layout). From DexNom-VIctory
//! (ddb7332, GPL-3.0-or-later): blending
//! dropped on integer render targets
//! ([`GraphicsState::apply_integer_blend_bypass`]).

use std::ffi::{CStr, CString};
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;
use xnw_shader::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};

use super::Shader;
use super::library;

/// The console's depth formats the host lacks (VIctory base resource.rs
/// `pc_format`); others are the same numbers.
pub(crate) fn pc_format(format: u32) -> u32 {
    match format {
        118 => 45,
        119 => 46,
        120 => 47,
        other => other,
    }
}

/// A pipeline's graphics state, owned (the title's stream lives only for
/// the creation call). Subobjects a stream leaves out keep D3D12's defaults.
#[derive(Clone)]
pub(crate) struct GraphicsState {
    pub(crate) blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    pub(crate) rasterizer: D3D12_RASTERIZER_DESC,
    pub(crate) depth: D3D12_DEPTH_STENCIL_DESC,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    pub(crate) depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

// SAFETY: the element names point into `names`, owned alongside.
unsafe impl Send for GraphicsState {}
// SAFETY: as above; read only after it is built.
unsafe impl Sync for GraphicsState {}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC { RenderTarget: [target; 8], ..Default::default() },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_BACK, DepthClipEnable: true.into(), ..Default::default() },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        }
    }
}

fn rd<T: Copy>(at: *const u8, offset: usize) -> T {
    // SAFETY: callers read inside data the title passed.
    unsafe { at.add(offset).cast::<T>().read_unaligned() }
}

impl GraphicsState {
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned(),
                21 => {
                    // shortcut (base): the depth bounds test is dropped.
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(layout.pInputElementDescs, layout.NumElements as usize);
                    let list = elements
                        .iter()
                        .map(|e| (CStr::from_ptr(e.SemanticName.0.cast()).to_owned(), e.SemanticIndex, e.Format.0 as u32, e.InputSlot, e.AlignedByteOffset, e.InputSlotClass.0, e.InstanceDataStepRate))
                        .collect();
                    self.set_inputs(list);
                }
                13 => self.strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned()),
                14 => self.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned()),
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = pc_format(f.0 as u32) as i32;
                    }
                }
                16 => self.depth_format = DXGI_FORMAT(pc_format(data.cast::<u32>().read_unaligned()) as i32),
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Input elements: (name, index, format, slot, offset, class, step rate).
    fn set_inputs(&mut self, list: Vec<(CString, u32, u32, u32, u32, i32, u32)>) {
        self.names = list.iter().map(|e| e.0.clone()).collect();
        self.inputs = list
            .iter()
            .zip(&self.names)
            .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: e.1,
                Format: DXGI_FORMAT(pc_format(e.2) as i32),
                InputSlot: e.3,
                AlignedByteOffset: e.4,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION(e.5),
                InstanceDataStepRate: e.6,
            })
            .collect();
    }

    /// The state of a pipeline of the title's pipeline cache (slot 57): the
    /// outputs-and-inputs entry (formats, topology, input elements) and,
    /// from the context registers of the state entry, depth, stencil
    /// enable, cull mode and blending (enough for the first frames' blits
    /// and UI; the rest is the full decoder's).
    pub(crate) fn from_cache(state: &[u8], outputs: &[u8]) -> GraphicsState {
        let mut g = GraphicsState { topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, ..GraphicsState::default() };
        let outputs_size = outputs.len();
        let state_size = state.len();
        if outputs_size >= 0x50 {
            let o = outputs.as_ptr();
            let count = usize::from(rd::<u8>(o, 0x25)).min(8);
            g.targets.NumRenderTargets = count as u32;
            for i in 0..count {
                g.targets.RTFormats[i] = DXGI_FORMAT(pc_format(u32::from(rd::<u8>(o, 0x10 + i))) as i32);
            }
            g.depth_format = DXGI_FORMAT(pc_format(u32::from(rd::<u8>(o, 0x24))) as i32);
            g.topology = match rd::<u8>(o, 0x26) & 15 {
                t @ 1..=4 => D3D12_PRIMITIVE_TOPOLOGY_TYPE(i32::from(t)),
                _ => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            };
            let elements = rd::<u32>(o, 0x4c) as usize;
            let mut at = 0x50;
            let mut list = Vec::new();
            for _ in 0..elements.min(32) {
                if at + 4 > outputs_size {
                    break;
                }
                let n = rd::<u32>(o, at) as usize;
                if n == 0 || at + 4 + n + 24 > outputs_size {
                    break;
                }
                let name = &outputs[at + 4..at + 4 + n];
                let name = CString::new(name.split(|&b| b == 0).next().unwrap_or_default()).unwrap_or_default();
                let f = at + 4 + n;
                list.push((name, rd(o, f), rd(o, f + 4), rd(o, f + 8), rd(o, f + 12), rd(o, f + 16), rd(o, f + 20)));
                at = f + 24;
            }
            g.set_inputs(list);
        }
        if state_size >= 0x124 {
            let s = state.as_ptr();
            // Layout kind 6 (mesh) holds 0x54 more bytes before the pixel block.
            let shift = |off: usize| if rd::<u32>(s, 4) == 6 && off >= 0x58 { off + 0x54 } else { off };
            let reg = |off: usize| -> u32 { if shift(off) + 4 <= state_size { rd(s, shift(off)) } else { 0 } };
            let mode = reg(0xa0);
            g.rasterizer.CullMode = match mode & 3 {
                1 => D3D12_CULL_MODE_FRONT,
                2 => D3D12_CULL_MODE_BACK,
                _ => D3D12_CULL_MODE_NONE,
            };
            g.rasterizer.FrontCounterClockwise = (mode & 4 == 0).into();
            let dc = reg(0xb8);
            g.depth.DepthEnable = (dc & 2 != 0).into();
            g.depth.DepthWriteMask = if dc & 4 != 0 { D3D12_DEPTH_WRITE_MASK_ALL } else { D3D12_DEPTH_WRITE_MASK_ZERO };
            g.depth.DepthFunc = D3D12_COMPARISON_FUNC(((dc >> 4) & 7) as i32 + 1);
            g.depth.StencilEnable = false.into(); // stencil ops: the full decoder
            let mask = reg(0xcc);
            for i in 0..8 {
                let c = reg(0xd0 + 4 * i);
                let t = &mut g.blend.RenderTarget[i];
                t.RenderTargetWriteMask = ((mask >> (4 * i)) & 15) as u8;
                t.BlendEnable = (c & (1 << 30) != 0).into();
                if c & (1 << 30) != 0 {
                    let separate = c & (1 << 29) != 0;
                    t.SrcBlend = blend(c & 31);
                    t.BlendOp = blend_op((c >> 5) & 7);
                    t.DestBlend = blend((c >> 8) & 31);
                    t.SrcBlendAlpha = if separate { blend((c >> 16) & 31) } else { t.SrcBlend };
                    t.BlendOpAlpha = if separate { blend_op((c >> 21) & 7) } else { t.BlendOp };
                    t.DestBlendAlpha = if separate { blend((c >> 24) & 31) } else { t.DestBlend };
                }
            }
            g.blend.IndependentBlendEnable = true.into();
        }
        g
    }

    /// DexNom-VIctory (ddb7332, `pipeline/host.rs` 95-130, GPL-3.0-or-later):
    /// the guest color buffer bypasses its blender for integer surfaces,
    /// even when the draw asks for blending. D3D12 validates blend enable
    /// against every active target format when the pipeline is created.
    pub(crate) fn apply_integer_blend_bypass(&mut self) {
        let integer = |f: &DXGI_FORMAT| {
            matches!(
                *f,
                DXGI_FORMAT_R32G32B32A32_UINT
                    | DXGI_FORMAT_R32G32B32A32_SINT
                    | DXGI_FORMAT_R32G32B32_UINT
                    | DXGI_FORMAT_R32G32B32_SINT
                    | DXGI_FORMAT_R16G16B16A16_UINT
                    | DXGI_FORMAT_R16G16B16A16_SINT
                    | DXGI_FORMAT_R32G32_UINT
                    | DXGI_FORMAT_R32G32_SINT
                    | DXGI_FORMAT_R10G10B10A2_UINT
                    | DXGI_FORMAT_R8G8B8A8_UINT
                    | DXGI_FORMAT_R8G8B8A8_SINT
                    | DXGI_FORMAT_R16G16_UINT
                    | DXGI_FORMAT_R16G16_SINT
                    | DXGI_FORMAT_R32_UINT
                    | DXGI_FORMAT_R32_SINT
                    | DXGI_FORMAT_R8G8_UINT
                    | DXGI_FORMAT_R8G8_SINT
                    | DXGI_FORMAT_R16_UINT
                    | DXGI_FORMAT_R16_SINT
                    | DXGI_FORMAT_R8_UINT
                    | DXGI_FORMAT_R8_SINT
            )
        };
        let formats = &self.targets.RTFormats[..(self.targets.NumRenderTargets as usize).min(8)];
        if !self.blend.IndependentBlendEnable.as_bool() {
            let shared = self.blend.RenderTarget[0];
            if !shared.BlendEnable.as_bool() || !formats.iter().any(integer) {
                return;
            }
            // With shared blending D3D12 reads slot zero for every target.
            // Expand it before overriding individual targets, including
            // their shared write mask.
            self.blend.RenderTarget[..formats.len()].fill(shared);
            self.blend.IndependentBlendEnable = (formats.len() > 1).into();
        }
        for (format, target) in formats.iter().zip(&mut self.blend.RenderTarget) {
            if integer(format) {
                target.BlendEnable = false.into();
            }
        }
    }

    /// Slot 55's overrides: (kind, description) with kind 0 depth-stencil,
    /// 1 rasterizer, 2 blend (desktop layouts; five-star `d3d12x-pipelines.md`
    /// "Derive"); kind 5 and others are ignored.
    pub(crate) fn apply_overrides(&mut self, overrides: &[Override]) {
        for o in overrides {
            match o {
                Override::Depth(d) => self.depth = *d,
                Override::Rasterizer(r) => self.rasterizer = *r,
                Override::Blend(b) => self.blend = *b,
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| format!("{name:?} {} {} {} {} {} {}", e.SemanticIndex, e.Format.0, e.InputSlot, e.AlignedByteOffset, e.InputSlotClass.0, e.InstanceDataStepRate))
            .collect();
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}",
            self.blend, self.sample_mask, self.rasterizer, self.depth, self.strip_cut.0, self.topology.0, self.targets, self.depth_format.0, self.samples
        )
    }

    /// The state as bytes, for the pipeline list (field by field, little
    /// endian words; input names length-prefixed). [`GraphicsState::decode`]
    /// gives it back.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut w: Vec<u32> = Vec::with_capacity(160);
        let b = &self.blend;
        w.extend([b.AlphaToCoverageEnable.0 as u32, b.IndependentBlendEnable.0 as u32]);
        for t in &b.RenderTarget {
            w.extend([t.BlendEnable.0 as u32, t.LogicOpEnable.0 as u32, t.SrcBlend.0 as u32, t.DestBlend.0 as u32, t.BlendOp.0 as u32, t.SrcBlendAlpha.0 as u32, t.DestBlendAlpha.0 as u32, t.BlendOpAlpha.0 as u32, t.LogicOp.0 as u32, u32::from(t.RenderTargetWriteMask)]);
        }
        w.push(self.sample_mask);
        let r = &self.rasterizer;
        w.extend([r.FillMode.0 as u32, r.CullMode.0 as u32, r.FrontCounterClockwise.0 as u32, r.DepthBias as u32, r.DepthBiasClamp.to_bits(), r.SlopeScaledDepthBias.to_bits(), r.DepthClipEnable.0 as u32, r.MultisampleEnable.0 as u32, r.AntialiasedLineEnable.0 as u32, r.ForcedSampleCount, r.ConservativeRaster.0 as u32]);
        let d = &self.depth;
        w.extend([d.DepthEnable.0 as u32, d.DepthWriteMask.0 as u32, d.DepthFunc.0 as u32, d.StencilEnable.0 as u32, u32::from(d.StencilReadMask), u32::from(d.StencilWriteMask)]);
        for f in [&d.FrontFace, &d.BackFace] {
            w.extend([f.StencilFailOp.0 as u32, f.StencilDepthFailOp.0 as u32, f.StencilPassOp.0 as u32, f.StencilFunc.0 as u32]);
        }
        w.extend([self.strip_cut.0 as u32, self.topology.0 as u32, self.targets.NumRenderTargets]);
        w.extend(self.targets.RTFormats.iter().map(|f| f.0 as u32));
        w.extend([self.depth_format.0 as u32, self.samples.Count, self.samples.Quality, self.inputs.len() as u32]);
        let mut out: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        for (e, name) in self.inputs.iter().zip(&self.names) {
            let name = name.as_bytes();
            out.extend((name.len() as u32).to_le_bytes());
            out.extend(name);
            for v in [e.SemanticIndex, e.Format.0 as u32, e.InputSlot, e.AlignedByteOffset, e.InputSlotClass.0 as u32, e.InstanceDataStepRate] {
                out.extend(v.to_le_bytes());
            }
        }
        out
    }

    /// The state [`GraphicsState::encode`] wrote, or `None` if damaged.
    pub(crate) fn decode(bytes: &[u8]) -> Option<GraphicsState> {
        struct Reader<'a>(&'a [u8]);
        impl<'a> Reader<'a> {
            fn take(&mut self, n: usize) -> Option<&'a [u8]> {
                let (head, rest) = self.0.split_at_checked(n)?;
                self.0 = rest;
                Some(head)
            }
            fn word(&mut self) -> Option<u32> {
                Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
            }
        }
        let mut input = Reader(bytes);
        let mut word = || input.word();
        let mut g = GraphicsState::default();
        let b = &mut g.blend;
        b.AlphaToCoverageEnable.0 = word()? as i32;
        b.IndependentBlendEnable.0 = word()? as i32;
        for t in &mut b.RenderTarget {
            t.BlendEnable.0 = word()? as i32;
            t.LogicOpEnable.0 = word()? as i32;
            t.SrcBlend.0 = word()? as i32;
            t.DestBlend.0 = word()? as i32;
            t.BlendOp.0 = word()? as i32;
            t.SrcBlendAlpha.0 = word()? as i32;
            t.DestBlendAlpha.0 = word()? as i32;
            t.BlendOpAlpha.0 = word()? as i32;
            t.LogicOp.0 = word()? as i32;
            t.RenderTargetWriteMask = word()? as u8;
        }
        g.sample_mask = word()?;
        let r = &mut g.rasterizer;
        r.FillMode.0 = word()? as i32;
        r.CullMode.0 = word()? as i32;
        r.FrontCounterClockwise.0 = word()? as i32;
        r.DepthBias = word()? as i32;
        r.DepthBiasClamp = f32::from_bits(word()?);
        r.SlopeScaledDepthBias = f32::from_bits(word()?);
        r.DepthClipEnable.0 = word()? as i32;
        r.MultisampleEnable.0 = word()? as i32;
        r.AntialiasedLineEnable.0 = word()? as i32;
        r.ForcedSampleCount = word()?;
        r.ConservativeRaster.0 = word()? as i32;
        let d = &mut g.depth;
        d.DepthEnable.0 = word()? as i32;
        d.DepthWriteMask.0 = word()? as i32;
        d.DepthFunc.0 = word()? as i32;
        d.StencilEnable.0 = word()? as i32;
        d.StencilReadMask = word()? as u8;
        d.StencilWriteMask = word()? as u8;
        for f in [&mut d.FrontFace, &mut d.BackFace] {
            f.StencilFailOp.0 = word()? as i32;
            f.StencilDepthFailOp.0 = word()? as i32;
            f.StencilPassOp.0 = word()? as i32;
            f.StencilFunc.0 = word()? as i32;
        }
        g.strip_cut.0 = word()? as i32;
        g.topology.0 = word()? as i32;
        g.targets.NumRenderTargets = word()?;
        for f in &mut g.targets.RTFormats {
            f.0 = word()? as i32;
        }
        g.depth_format.0 = word()? as i32;
        g.samples.Count = word()?;
        g.samples.Quality = word()?;
        let n = word()? as usize;
        let mut list = Vec::with_capacity(n.min(32));
        for _ in 0..n {
            let len = input.word()? as usize;
            let name = CString::new(input.take(len)?).ok()?;
            let (index, format, slot, offset, class, step) = (input.word()?, input.word()?, input.word()?, input.word()?, input.word()? as i32, input.word()?);
            list.push((name, index, format, slot, offset, class, step));
        }
        if !input.0.is_empty() {
            return None;
        }
        // `set_inputs` maps console formats; these are host formats already.
        g.names = list.iter().map(|e| e.0.clone()).collect();
        g.inputs = list
            .iter()
            .zip(&g.names)
            .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: e.1,
                Format: DXGI_FORMAT(e.2 as i32),
                InputSlot: e.3,
                AlignedByteOffset: e.4,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION(e.5),
                InstanceDataStepRate: e.6,
            })
            .collect();
        Some(g)
    }

    /// UI1: the input layout as (semantic, index, format, slot, byte
    /// offset worked out, per instance, step rate).
    pub(crate) fn input_list(&self) -> Vec<(String, u32, u32, u32, u32, bool, u32)> {
        self.inputs
            .iter()
            .zip(&self.names)
            .zip(self.elements())
            .map(|((e, name), at)| (name.to_string_lossy().into_owned(), e.SemanticIndex, e.Format.0 as u32, at.slot, at.offset, at.per_instance, e.InstanceDataStepRate))
            .collect()
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs.iter().fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT { next[slot] } else { e.AlignedByteOffset };
                next[slot] = offset + element_bytes(format);
                Input { slot: slot as u32, offset, format, per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA }
            })
            .collect()
    }
}

/// Bytes per element of a vertex format (0 if not a vertex format).
fn element_bytes(format: u32) -> u32 {
    match format {
        1..=4 => 16,
        5..=8 => 12,
        9..=22 => 8,
        23..=47 | 67 | 87..=93 => 4,
        48..=59 | 85 | 86 | 115 => 2,
        60..=65 => 1,
        _ => 0,
    }
}

/// One override of slot 55, copied at creation.
#[derive(Clone, Copy)]
pub(crate) enum Override {
    Depth(D3D12_DEPTH_STENCIL_DESC),
    Rasterizer(D3D12_RASTERIZER_DESC),
    Blend(D3D12_BLEND_DESC),
}

impl Override {
    /// The override at `at` (a kind, then the description).
    pub(crate) fn read(at: *const u8) -> Option<Override> {
        Some(match rd::<u32>(at, 0) {
            0 => Override::Depth(rd(at, 4)),
            1 => Override::Rasterizer(rd(at, 4)),
            2 => Override::Blend(rd(at, 4)),
            _ => return None,
        })
    }
}

/// AMD CB_BLEND factor -> D3D12_BLEND.
fn blend(f: u32) -> D3D12_BLEND {
    D3D12_BLEND(match f {
        0..=10 => f as i32 + 1,
        13 => 14,
        14 => 15,
        15..=18 => f as i32 + 1,
        19 => 20,
        20 => 21,
        _ => 2,
    })
}

/// AMD COMB_FCN -> D3D12_BLEND_OP.
fn blend_op(f: u32) -> D3D12_BLEND_OP {
    D3D12_BLEND_OP(match f {
        1 => 2,
        2 => 4,
        3 => 5,
        4 => 3,
        _ => 1,
    })
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// Whether the title's root SRVs are bound as one-view tables (a raw
/// view from the address to the end of what backs it), whose reads past
/// the end return zeros, rather than as root descriptors, which have no
/// size (VIctory: mesh shaders reading megabytes past a root SRV's copy
/// page-fault). `XNW_ROOT_SRV_DIRECT=1` goes back to root
/// descriptors. Ported from victory-fork (fxceless/victory-fork @ 4e17478)
/// `crates/victory-gpu/src/pipeline/host.rs` 255-265, GPL-3.0-or-later.
pub(crate) fn bounded_root_srvs() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_ROOT_SRV_DIRECT").map_or(true, |v| v.trim() != "1"));
    *ON
}
/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    /// Root UAVs (XNW, the translator's root UAV class), likewise.
    pub(crate) root_uavs: Vec<(u8, u32)>,
    /// The root parameter of the runtime stride table (`t0` in
    /// [`xnw_shader::emit::STRIDE_SPACE`]), when the stage reads it.
    pub(crate) stride_table: Option<u32>,
    /// The root parameters of the address map and of the descriptor words,
    /// when it reads them (VIctory fe078c6, GPL-3.0-or-later).
    pub(crate) address_map: Option<u32>,
    pub(crate) descriptor_words: Option<u32>,
    /// The root parameter of the page table, when it reads it.
    pub(crate) page_table: Option<u32>,
    /// The program's host-copy bits (`PRIV`), read once
    /// here; `None`: translated without the page table.
    pub(crate) host_copies: Option<u32>,
    /// The root parameter of the heap bounds (8 root
    /// constants), when it reads them.
    pub(crate) heap_bounds: Option<u32>,
}

impl Stage {
    fn new(shader: &Arc<Shader>) -> Stage {
        let host_copies = shader.translated().and_then(|t| t.host.host_copies());
        Stage { shader: shader.clone(), user_data: None, root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), stride_table: None, address_map: None, descriptor_words: None, page_table: None, host_copies, heap_bounds: None }
    }
}

// ---- The gate ----
//
// A translated program may read title memory beyond its tables: through
// root UAVs, through the address map (addresses it computes), or through
// the console's descriptor words. Each needs a runtime piece bound with it;
// a program whose piece is not bound gets no host pipeline (logged once
// per reason, its dispatches and draws dropped and counted), so nothing of
// this family runs half-bound. `XNW_GATE=uav,map,words` holds pieces back
// (to bisect); the descriptor words are held back with `XNW_DESC=gpu`,
// whose records do not keep the console words the shader reads.

/// The pieces, by gate number.
pub(crate) const GATES: [&str; 4] = ["needs root UAV", "needs address map", "needs descriptor words", "needs page table"];
const GATE_KEYS: [&str; 4] = ["uav", "map", "words", "pagetable"];
/// Programs (host pipelines asked for) gated, and their draws and
/// dispatches dropped, by gate.
pub(crate) static GATED: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
pub(crate) static GATED_DROPS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Stages made with the page table bound.
pub(crate) static PAGE_TABLE_PROGRAMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the runtime binds piece `g` now.
fn piece_bound(g: usize) -> bool {
    static HELD: std::sync::LazyLock<Vec<String>> = std::sync::LazyLock::new(|| std::env::var("XNW_GATE").unwrap_or_default().split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect());
    piece_bound_with(g, &HELD, super::super::descriptor::host::gpu_mode()) && (g != 3 || crate::shared::page_table_on())
}

/// [`piece_bound`] with the pieces `held` back and GPU-side descriptors
/// on or not. The page table is bound only with GPU-side
/// descriptors (its entries name slots of the one global view heap).
pub(crate) fn piece_bound_with(g: usize, held: &[String], gpu: bool) -> bool {
    !held.iter().any(|h| h == GATE_KEYS[g] || h == "all") && !(g == 2 && gpu) && !(g == 3 && !gpu)
}

/// The first piece the translated shader `t` needs that is not bound.
pub(crate) fn gate(t: &xnw_shader::emit::HostShader) -> Option<usize> {
    [!t.root_uavs.is_empty(), t.address_map, t.descriptor_words, t.reads_page_table()].iter().enumerate().position(|(g, &wanted)| wanted && !piece_bound(g))
}

/// Refuses a pipeline whose stages need a piece not bound.
fn gated(stages: &[&Arc<Shader>], label: &str) -> bool {
    for s in stages {
        if let Some(g) = s.translated().and_then(|t| gate(&t.host)) {
            GATED[g].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            super::note_failure(GATES[g], label);
            return true;
        }
    }
    false
}

/// The frame statistics part (totals since start).
pub(crate) fn gate_text() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let parts: Vec<String> = GATES.iter().enumerate().map(|(g, name)| format!("{name} {} gated ({} draws/dispatches dropped)", GATED[g].load(Relaxed), GATED_DROPS[g].load(Relaxed))).collect();
    parts.join(", ")
}

/// The root parameters of `t`'s root UAVs, address map and descriptor
/// words, after `params`.
fn address_params(t: &xnw_shader::emit::HostShader, visibility: D3D12_SHADER_VISIBILITY, stage: &mut Stage, params: &mut Vec<D3D12_ROOT_PARAMETER1>) {
    let view = |kind, register, space| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 { Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: register, RegisterSpace: space, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE } },
        ShaderVisibility: visibility,
    };
    for &slot in &t.root_uavs {
        stage.root_uavs.push((slot, params.len() as u32));
        params.push(view(D3D12_ROOT_PARAMETER_TYPE_UAV, u32::from(slot), ROOT_CBV_SPACE));
    }
    for (wanted, space, out) in [(t.address_map, xnw_shader::emit::ADDRESS_MAP_SPACE, &mut stage.address_map), (t.descriptor_words, xnw_shader::emit::DESCRIPTOR_WORDS_SPACE, &mut stage.descriptor_words)] {
        if wanted {
            *out = Some(params.len() as u32);
            params.push(view(D3D12_ROOT_PARAMETER_TYPE_SRV, 0, space));
        }
    }
    // The page table, a raw root SRV.
    if t.reads_page_table() {
        PAGE_TABLE_PROGRAMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        stage.page_table = Some(params.len() as u32);
        params.push(view(D3D12_ROOT_PARAMETER_TYPE_SRV, 0, xnw_shader::emit::PAGE_TABLE_SPACE));
    }
    // The heap bounds, 8 root constants at `b0`.
    if t.reads_heap_bounds() {
        stage.heap_bounds = Some(params.len() as u32);
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: xnw_shader::emit::HEAP_BOUNDS_SPACE, Num32BitValues: 8 } },
            ShaderVisibility: visibility,
        });
    }
}

impl Stage {
    pub(crate) fn translated(&self) -> &super::Translated {
        self.shader.translated().expect("host pipelines are made from translated shaders")
    }
}

/// What a draw needs from a title pipeline (the draw's interface).
pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (base pipeline/host.rs 264): drawn through
    /// [`HostPipeline::fallback`].
    pub(crate) mesh: bool,
    /// A mesh pipeline's compute half and record layout; `pso`/`root`
    /// are then its draw (pass-through vertex shader and the title's pixel
    /// shader), `stages` its pixel stage.
    pub(crate) fallback: Option<Box<MeshFallback>>,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// The title's topology type (console 19 and 32: rectangles and quads,
    /// drawn as triangles).
    #[allow(dead_code)] // kept; draws take the topology from the list, as the base does
    pub(crate) topology: i32,
    /// Its shaders, for logs.
    pub(crate) name: String,
    /// Drawn through the rectangle geometry shader: the title's
    /// rectangle lists run as triangle lists, arguments unchanged.
    pub(crate) rect: bool,
    /// For a graphics pipeline not made for rectangles: what makes its
    /// rectangle variant, and the variant once a rectangle draw asked
    /// (only the list's topology tells, the pipeline type may not).
    pub(crate) rect_recipe: Option<Box<RectRecipe>>,
}

/// A graphics pipeline's description, kept to make its rectangle variant.
pub(crate) struct RectRecipe {
    g: GraphicsState,
    root_blob: Vec<u8>,
    variant: std::sync::OnceLock<Option<ID3D12PipelineState>>,
    /// Its making was handed to the background maker.
    queued: std::sync::atomic::AtomicBool,
}

impl GraphicsState {
    /// A copy whose input elements point at its own names.
    fn owned(&self) -> GraphicsState {
        let mut g = self.clone();
        for (e, name) in g.inputs.iter_mut().zip(&g.names) {
            e.SemanticName = PCSTR(name.as_ptr().cast());
        }
        g
    }
}

impl HostPipeline {
    /// The rectangle variant (vertex, rectangle geometry, pixel shader)
    /// of a graphics pipeline, made on first use (off the drawing
    /// thread, [`super::rect_variant`], unless `XNW_PIPELINE_WAIT=1`).
    pub(crate) fn rect_pso(&self) -> Option<&ID3D12PipelineState> {
        if !rects_on() {
            return None;
        }
        let r = self.rect_recipe.as_deref()?;
        if !super::make::blocking() {
            return r.variant.get()?.as_ref();
        }
        self.make_rect_pso()
    }

    /// Whether the rectangle variant is still to be made.
    pub(crate) fn rect_pending(&self) -> bool {
        rects_on() && self.rect_recipe.as_deref().is_some_and(|r| r.variant.get().is_none())
    }

    /// The first ask of a variant not made: 	rue once (queue it).
    pub(crate) fn rect_first_ask(&self) -> bool {
        self.rect_recipe.as_deref().is_some_and(|r| !r.queued.swap(true, std::sync::atomic::Ordering::Relaxed))
    }

    /// The rectangle variant, made here if not yet (a compile: seconds).
    pub(crate) fn make_rect_pso(&self) -> Option<&ID3D12PipelineState> {
        if !rects_on() {
            return None;
        }
        let r = self.rect_recipe.as_deref()?;
        r.variant
            .get_or_init(|| {
                let vs = &self.stages.first()?.translated().host.dxil;
                let ps = self.stages.get(1).map(|s| &s.translated().host.dxil[..]);
                let gs = match xnw_shader::emit::rect::rect_gs(vs) {
                    Ok(gs) => gs,
                    Err(e) => {
                        super::note_failure(&format!("rectangle geometry shader: {e}"), &self.name);
                        return None;
                    }
                };
                let host = super::super::host::host_device()?;
                let made = graphics_pso(host, &self.root, &r.root_blob, vs, Some(&gs), ps, &r.g);
                if let Err(e) = &made {
                    super::note_failure(&format!("rectangle variant refused: {:#010x}", e.code().0 as u32), &self.name);
                } else {
                    xnw_core::info!("d3d12x: rectangle variant made for {}", self.name);
                }
                made.ok()
            })
            .as_ref()
    }
}

/// `XNW_RECT=off`: no rectangle geometry shader or variants, the
/// Era behaviour (CPU-expanded direct rectangles, the rest dropped).
pub(crate) fn rects_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_RECT").map_or(true, |v| v != "off"));
    *ON
}

/// A graphics pipeline over `vs`, `gs`, `ps` with the state `g`.
fn graphics_pso(host: &crate::host::Host, root: &ID3D12RootSignature, root_blob: &[u8], vs: &[u8], gs: Option<&[u8]>, ps: Option<&[u8]>, g: &GraphicsState) -> windows::core::Result<ID3D12PipelineState> {
    let code = |c: Option<&[u8]>| c.map_or(D3D12_SHADER_BYTECODE::default(), |c| D3D12_SHADER_BYTECODE { pShaderBytecode: c.as_ptr().cast(), BytecodeLength: c.len() });
    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // SAFETY: a borrowed pointer the description never releases.
        pRootSignature: unsafe { std::mem::transmute_copy(root) },
        VS: code(Some(vs)),
        PS: code(ps),
        GS: code(gs),
        BlendState: g.blend,
        SampleMask: g.sample_mask,
        RasterizerState: g.rasterizer,
        DepthStencilState: g.depth,
        InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: g.inputs.as_ptr(), NumElements: g.inputs.len() as u32 },
        IBStripCutValue: g.strip_cut,
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: g.targets.NumRenderTargets,
        RTVFormats: g.targets.RTFormats,
        DSVFormat: g.depth_format,
        SampleDesc: g.samples,
        ..Default::default()
    };
    let name = library::name("graphics", root_blob, &[vs, ps.unwrap_or(&[]), gs.unwrap_or(&[])], &g.describe());
    library::pipeline(
        &name,
        // SAFETY: the description points at live data.
        |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
    )
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for HostPipeline {}
// SAFETY: as above.
unsafe impl Sync for HostPipeline {}

/// Why a pipeline gets no host pipeline: counted per reason, the first of
/// each logged with its shaders.
pub(crate) fn refuse(reason: String, name: &str) -> Option<HostPipeline> {
    super::note_failure(&reason, name);
    None
}

fn names(shaders: &[Arc<Shader>]) -> String {
    shaders.iter().map(|s| format!("{} {:032x}", s.stage, s.hash)).collect::<Vec<_>>().join(" + ")
}

impl HostPipeline {
    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        let label = names(shaders);
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if let Some(ms) = find("mesh") {
            super::MESH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return mesh_pipeline(ms, find("pixel"), g, label);
        }
        if compute.is_none() && shaders.iter().any(|s| !matches!(s.stage, "vertex" | "pixel")) {
            return refuse("only vertex, pixel and compute stages are translated".into(), &label);
        }
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let Some(vs) = find("vertex") else {
                return refuse("no vertex shader".into(), &label);
            };
            let mut stages = vec![(vs, D3D12_SHADER_VISIBILITY_VERTEX)];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages
        };
        for (s, _) in &stages {
            if let Err(e) = s.translation() {
                return refuse(format!("{} does not translate: {e}", s.stage), &label);
            }
        }
        if gated(&stages.iter().map(|s| s.0).collect::<Vec<_>>(), &label) {
            return None;
        }
        if let Some(why) = stages.iter().find_map(|s| library::denied_program(s.0)) {
            return refuse(why, &label);
        }
        let host = super::super::host::host_device()?;

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        let mut flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
        if compute.is_none() {
            flags |= D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        }
        for &(shader, visibility) in &stages {
            let t = &shader.translated()?.host;
            let mut stage = Stage::new(shader);
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: USER_DATA_SPACE, Num32BitValues: dwords } },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [(&t.root_cbvs, D3D12_ROOT_PARAMETER_TYPE_CBV, &mut stage.root_cbvs), (&t.root_srvs, D3D12_ROOT_PARAMETER_TYPE_SRV, &mut stage.root_srvs)] {
                for &slot in slots {
                    out.push((slot, params.len() as u32));
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: u32::from(slot), RegisterSpace: ROOT_CBV_SPACE, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE },
                        },
                        ShaderVisibility: visibility,
                    });
                }
            }
            address_params(t, visibility, &mut stage, &mut params);
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        // XNW: the runtime stride table, one root SRV for every stage
        // that reads it (VIctory pipeline/host.rs).
        if bound.iter().any(|s| s.translated().host.reads_stride_table()) {
            let param = params.len() as u32;
            params.push(D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: 0, RegisterSpace: xnw_shader::emit::STRIDE_SPACE, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
            for s in bound.iter_mut().filter(|s| s.translated().host.reads_stride_table()) {
                s.stride_table = Some(param);
            }
        }
        let (root, root_blob) = match root_signature(&host.device, &params, &samplers, flags) {
            Ok(root) => root,
            Err(e) => return refuse(format!("host root signature refused: {}", e.message()), &label),
        };
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(D3D12_SHADER_BYTECODE::default(), |t| D3D12_SHADER_BYTECODE {
                pShaderBytecode: t.host.dxil.as_ptr().cast(),
                BytecodeLength: t.host.dxil.len(),
            })
        };
        let bytes = |i: usize| stages.get(i).and_then(|(s, _)| s.translated()).map_or(&[][..], |t| &t.host.dxil[..]);
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = library::name("compute", &root_blob, &[bytes(0)], "");
            let made = library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => return refuse(format!("host compute pipeline refused: {:#010x}", e.code().0 as u32), &label),
            };
            return Some(HostPipeline { compute: true, mesh: false, fallback: None, pso, root, stages: bound, slots: 0, inputs: Vec::new(), topology: 0, name: label, rect: false, rect_recipe: None });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)".into(), &label);
        };
        let state = g.describe();
        // A rectangle pipeline (console type 19) gets the rectangle
        // geometry shader made from its vertex shader's outputs.
        let gs = if g.topology.0 == RECTANGLES && rects_on() {
            match xnw_shader::emit::rect::rect_gs(bytes(0)) {
                Ok(gs) => Some(gs),
                Err(e) => return refuse(format!("rectangle geometry shader: {e}"), &label),
            }
        } else {
            None
        };
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            PS: code(1),
            GS: gs.as_deref().map_or(D3D12_SHADER_BYTECODE::default(), |c| D3D12_SHADER_BYTECODE { pShaderBytecode: c.as_ptr().cast(), BytecodeLength: c.len() }),
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: g.inputs.as_ptr(), NumElements: g.inputs.len() as u32 },
            IBStripCutValue: g.strip_cut,
            // The console's rectangle and quad types draw as triangles.
            PrimitiveTopologyType: match g.topology.0 {
                RECTANGLES | QUADS | 0 => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                _ => g.topology,
            },
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let name = library::name("graphics", &root_blob, &[bytes(0), bytes(1), gs.as_deref().unwrap_or(&[])], &state);
        let made = library::pipeline(
            &name,
            // SAFETY: the description points at live data.
            |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
        );
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                return refuse(
                    format!(
                        "host graphics pipeline refused: {:#010x} (targets {:?}, depth {}, {} inputs, topology {})",
                        e.code().0 as u32,
                        &g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize].iter().map(|f| f.0).collect::<Vec<_>>(),
                        g.depth_format.0,
                        g.inputs.len(),
                        g.topology.0
                    ),
                    &label,
                );
            }
        };
        let rect = gs.is_some();
        let rect_recipe = (!rect).then(|| Box::new(RectRecipe { g: g.owned(), root_blob: root_blob.clone(), variant: std::sync::OnceLock::new(), queued: std::sync::atomic::AtomicBool::new(false) }));
        Some(HostPipeline { compute: false, mesh: false, fallback: None, pso, root, stages: bound, slots: g.slots(), inputs: g.elements(), topology: g.topology.0, name: label, rect, rect_recipe })
    }
}

// ---- Mesh pipelines on a host without mesh shaders ----
//
// Replaces the base's mesh branch (B:pipeline/host.rs 294-592 Ms stage,
// `mesh_pipeline` 593-644, a pipeline-state stream with the mesh shader):
// the program runs as compute writing records (`xnw_shader::emit::mesh`),
// drawn by a generated pass-through vertex shader and the title's pixel
// shader with the title's state. Two root signatures: the compute one is
// the mesh stage's bindings (visibility all) plus `u0`/`u1`, the draw's is
// the pixel stage's plus `t0`.

/// What a mesh draw needs besides the draw pipeline.
pub(crate) struct MeshFallback {
    /// The compute half: a compute host pipeline over the mesh stage, bound
    /// from the title's graphics root arguments.
    pub(crate) cs: HostPipeline,
    pub(crate) layout: xnw_shader::emit::mesh::Layout,
    /// Root parameters: compute `u0` (records), `u1` (indices), `t0`
    /// (first group); draw `t0`.
    pub(crate) records: u32,
    pub(crate) indices: u32,
    /// Compute `t0`: the dword holding the first console group of the
    /// dispatch (direct dispatches past one segment run in pieces).
    pub(crate) base: u32,
    /// Compute `b0` in ROOT_LEN_SPACE: the root parameter and its
    /// dwords, one per root SRV of the program (in its order), if it reads any.
    pub(crate) lengths: Option<(u32, u32)>,
    pub(crate) vertex_records: u32,
}

/// One stage's root parameters and static samplers, as [`HostPipeline::new`]
/// lays them out.
fn stage_params(shader: &Arc<Shader>, visibility: D3D12_SHADER_VISIBILITY, params: &mut Vec<D3D12_ROOT_PARAMETER1>, samplers: &mut Vec<D3D12_STATIC_SAMPLER_DESC>) -> Option<Stage> {
    let t = &shader.translated()?.host;
    let mut stage = Stage::new(shader);
    let dwords = t.user_data_dwords();
    if dwords > 0 {
        stage.user_data = Some((params.len() as u32, dwords));
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: USER_DATA_SPACE, Num32BitValues: dwords } },
            ShaderVisibility: visibility,
        });
    }
    for (slots, kind, out) in [(&t.root_cbvs, D3D12_ROOT_PARAMETER_TYPE_CBV, &mut stage.root_cbvs), (&t.root_srvs, D3D12_ROOT_PARAMETER_TYPE_SRV, &mut stage.root_srvs)] {
        for &slot in slots {
            out.push((slot, params.len() as u32));
            if kind == D3D12_ROOT_PARAMETER_TYPE_SRV && bounded_root_srvs() {
                // A one-view table: the view's size bounds the reads,
                // which a root descriptor leaves unbounded (victory-fork
                // pipeline/host.rs 255-265, 426-460). The translated code
                // binds `t<slot>` in ROOT_CBV_SPACE either way.
                let range: &'static D3D12_DESCRIPTOR_RANGE1 = Box::leak(Box::new(D3D12_DESCRIPTOR_RANGE1 {
                    RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    NumDescriptors: 1,
                    BaseShaderRegister: u32::from(slot),
                    RegisterSpace: ROOT_CBV_SPACE,
                    Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                    OffsetInDescriptorsFromTableStart: 0,
                }));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 { NumDescriptorRanges: 1, pDescriptorRanges: range } },
                    ShaderVisibility: visibility,
                });
                continue;
            }
            params.push(D3D12_ROOT_PARAMETER1 {
                ParameterType: kind,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: u32::from(slot), RegisterSpace: ROOT_CBV_SPACE, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE },
                },
                ShaderVisibility: visibility,
            });
        }
    }
    address_params(t, visibility, &mut stage, params);
    for (n, s) in t.static_samplers.iter().enumerate() {
        samplers.push(D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER(s.filter as i32),
            AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
            AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
            AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
            MipLODBias: s.mip_lod_bias,
            MaxAnisotropy: s.max_anisotropy,
            ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
            BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
            MinLOD: s.min_lod,
            MaxLOD: s.max_lod,
            ShaderRegister: n as u32,
            RegisterSpace: SAMPLER_SPACE,
            ShaderVisibility: visibility,
        });
    }
    Some(stage)
}

/// A raw-buffer root view in the record space.
fn record_view(kind: D3D12_ROOT_PARAMETER_TYPE, register: u32, visibility: D3D12_SHADER_VISIBILITY) -> D3D12_ROOT_PARAMETER1 {
    D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: register, RegisterSpace: xnw_shader::emit::mesh::RECORD_SPACE, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE },
        },
        ShaderVisibility: visibility,
    }
}

/// The fallback pair for a title mesh pipeline: compute over the mesh
/// program, draw with the pass-through vertex shader and `ps`.
fn mesh_pipeline(ms: &Arc<Shader>, ps: Option<&Arc<Shader>>, g: Option<&GraphicsState>, label: String) -> Option<HostPipeline> {
    use xnw_shader::emit::mesh::{INDEX_REGISTER, RECORD_REGISTER, passthrough_vs};
    for s in std::iter::once(ms).chain(ps) {
        if let Err(e) = s.translation() {
            return refuse(format!("{} does not translate: {e}", s.stage), &label);
        }
    }
    if gated(&std::iter::once(ms).chain(ps).collect::<Vec<_>>(), &label) {
        return None;
    }
    if let Some(why) = std::iter::once(ms).chain(ps).find_map(|s| library::denied_program(s)) {
        return refuse(why, &label);
    }
    let Some(layout) = ms.translated()?.mesh.clone() else {
        return refuse("mesh program without a record layout".into(), &label);
    };
    let Some(g) = g else {
        return refuse("graphics state not kept (plain description)".into(), &label);
    };
    let host = super::super::host::host_device()?;
    let heap_flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;

    // Compute half.
    let (mut params, mut samplers) = (Vec::new(), Vec::new());
    let stage = stage_params(ms, D3D12_SHADER_VISIBILITY_ALL, &mut params, &mut samplers)?;
    let records = params.len() as u32;
    params.push(record_view(D3D12_ROOT_PARAMETER_TYPE_UAV, RECORD_REGISTER, D3D12_SHADER_VISIBILITY_ALL));
    params.push(record_view(D3D12_ROOT_PARAMETER_TYPE_UAV, INDEX_REGISTER, D3D12_SHADER_VISIBILITY_ALL));
    // The piece's first group (`t0`, dword 0; see `console_group`).
    params.push(record_view(D3D12_ROOT_PARAMETER_TYPE_SRV, RECORD_REGISTER, D3D12_SHADER_VISIBILITY_ALL));
    // The bytes each root SRV may read (`b0` in ROOT_LEN_SPACE): the
    // program reads offset 0 past them.
    let root_srvs = ms.translated()?.host.root_srvs.len() as u32;
    let lengths = (root_srvs > 0).then(|| {
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: xnw_shader::emit::ROOT_LEN_SPACE, Num32BitValues: root_srvs.next_multiple_of(4) } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
        (params.len() as u32 - 1, root_srvs.next_multiple_of(4))
    });
    let (root, root_blob) = match root_signature(&host.device, &params, &samplers, heap_flags) {
        Ok(r) => r,
        Err(e) => return refuse(format!("mesh compute root signature refused: {}", e.message()), &label),
    };
    let cs_code = &ms.translated()?.host.dxil;
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        // SAFETY: a borrowed pointer the description never releases.
        pRootSignature: unsafe { std::mem::transmute_copy(&root) },
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs_code.as_ptr().cast(), BytecodeLength: cs_code.len() },
        ..Default::default()
    };
    let name = library::name("compute", &root_blob, &[cs_code], "mesh fallback");
    let made = library::pipeline(
        &name,
        // SAFETY: the description points at live data.
        |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { host.device.CreateComputePipelineState(&desc) },
    );
    let pso = match made {
        Ok(pso) => pso,
        Err(e) => return refuse(format!("mesh compute pipeline refused: {:#010x}", e.code().0 as u32), &label),
    };
    let cs = HostPipeline { compute: true, mesh: false, fallback: None, pso, root, stages: vec![stage], slots: 0, inputs: Vec::new(), topology: 0, name: format!("{label} (compute)"), rect: false, rect_recipe: None };

    // Draw half.
    let ps_code = ps.and_then(|p| p.translated()).map(|t| &t.host.dxil[..]);
    // How the pixel shader's inputs meet the record: inputs with no
    // record element of their name read by row or as 0.
    if let Some(code) = ps_code
        && let Ok([n, row, named, both]) = xnw_shader::emit::mesh::link_stats(&layout, code)
        && named < n
    {
        xnw_core::info!("d3d12x: mesh link: {n} pixel inputs, {named} by name ({both} at the same row), {} by row only, {} as 0 ({label})", row.saturating_sub(both).min(n - named), (n - named).saturating_sub(row.saturating_sub(both)));
    }
    let vs = match passthrough_vs(&layout, ps_code) {
        Ok(vs) => vs,
        Err(e) => return refuse(format!("mesh pass-through vertex shader: {e}"), &label),
    };
    let (mut params, mut samplers) = (Vec::new(), Vec::new());
    let mut stages = Vec::new();
    if let Some(ps) = ps {
        stages.push(stage_params(ps, D3D12_SHADER_VISIBILITY_PIXEL, &mut params, &mut samplers)?);
    }
    let vertex_records = params.len() as u32;
    params.push(record_view(D3D12_ROOT_PARAMETER_TYPE_SRV, RECORD_REGISTER, D3D12_SHADER_VISIBILITY_VERTEX));
    let (groot, groot_blob) = match root_signature(&host.device, &params, &samplers, heap_flags) {
        Ok(r) => r,
        Err(e) => return refuse(format!("mesh draw root signature refused: {}", e.message()), &label),
    };
    let state = g.describe();
    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // SAFETY: a borrowed pointer the description never releases.
        pRootSignature: unsafe { std::mem::transmute_copy(&groot) },
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
        PS: ps_code.map_or(D3D12_SHADER_BYTECODE::default(), |c| D3D12_SHADER_BYTECODE { pShaderBytecode: c.as_ptr().cast(), BytecodeLength: c.len() }),
        BlendState: g.blend,
        SampleMask: g.sample_mask,
        RasterizerState: g.rasterizer,
        DepthStencilState: g.depth,
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: g.targets.NumRenderTargets,
        RTVFormats: g.targets.RTFormats,
        DSVFormat: g.depth_format,
        SampleDesc: g.samples,
        ..Default::default()
    };
    let name = library::name("graphics", &groot_blob, &[&vs, ps_code.unwrap_or(&[])], &state);
    let made = library::pipeline(
        &name,
        // SAFETY: the description points at live data.
        |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
    );
    let pso = match made {
        Ok(pso) => pso,
        Err(e) => {
            return refuse(
                format!(
                    "mesh draw pipeline refused: {:#010x} (targets {:?}, depth {})",
                    e.code().0 as u32,
                    &g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize].iter().map(|f| f.0).collect::<Vec<_>>(),
                    g.depth_format.0
                ),
                &label,
            );
        }
    };
    let fallback = MeshFallback { cs, layout, records, indices: records + 1, base: records + 2, lengths, vertex_records };
    Some(HostPipeline { compute: false, mesh: true, fallback: Some(Box::new(fallback)), pso, root: groot, stages, slots: 0, inputs: Vec::new(), topology: 4, name: label, rect: false, rect_recipe: None })
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    flags: D3D12_ROOT_SIGNATURE_FLAGS,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let heap_indexing = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    let desc_with = |flags| D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: flags,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if D3D12SerializeVersionedRootSignature(&desc_with(flags), &mut blob, Some(&mut error)).is_err() {
            if let Some(error) = &error {
                let text = std::slice::from_raw_parts(error.GetBufferPointer().cast::<u8>(), error.GetBufferSize());
                xnw_core::warn!("d3d12x: host root signature: {}", String::from_utf8_lossy(text));
            }
            // Windows 10's in-box d3d12core refuses the heap-indexing flags
            // (as the base found): retry without them.
            blob = None;
            D3D12SerializeVersionedRootSignature(&desc_with(flags & !heap_indexing), &mut blob, None)?;
            xnw_core::warn!("d3d12x: host root signature serialised only without the heap-indexing flags");
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DexNom-VIctory (ddb7332, `pipeline/host.rs` test, GPL-3.0-or-later), verbatim.
    #[test]
    fn integer_target_bypasses_blending_and_preserves_mixed_target_masks() {
        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 3;
        g.targets.RTFormats[..3].copy_from_slice(&[DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_B4G4R4A4_UNORM, DXGI_FORMAT_R8_UINT]);
        g.blend.IndependentBlendEnable = true.into();
        for (i, target) in g.blend.RenderTarget[..3].iter_mut().enumerate() {
            target.BlendEnable = true.into();
            target.BlendOp = D3D12_BLEND_OP_MAX;
            target.RenderTargetWriteMask = [15, 15, 1][i];
        }
        let before = g.blend;
        g.apply_integer_blend_bypass();
        assert_eq!(g.blend.RenderTarget[0..2], before.RenderTarget[0..2]);
        let mut expected = before.RenderTarget[2];
        expected.BlendEnable = false.into();
        assert_eq!(g.blend.RenderTarget[2], expected);

        g.blend = before;
        g.blend.IndependentBlendEnable = false.into();
        g.blend.RenderTarget[0].RenderTargetWriteMask = 3;
        g.targets.RTFormats.swap(0, 2);
        g.apply_integer_blend_bypass();
        assert!(g.blend.IndependentBlendEnable.as_bool());
        assert!(!g.blend.RenderTarget[0].BlendEnable.as_bool());
        for target in &g.blend.RenderTarget[1..3] {
            assert!(target.BlendEnable.as_bool());
            assert_eq!(target.BlendOp, D3D12_BLEND_OP_MAX);
            assert_eq!(target.RenderTargetWriteMask, 3);
        }
    }

    /// The pipeline list's state encoding gives back the same state
    /// (`describe`, part of the pipeline key), input elements included.
    #[test]
    fn state_encoding_round_trips() {
        let mut g = GraphicsState { topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, ..GraphicsState::default() };
        g.set_inputs(vec![(CString::new("POSITION").unwrap(), 0, 6, 0, 0, 0, 0), (CString::new("TEXCOORD").unwrap(), 3, 16, 1, 12, 1, 1)]);
        g.blend.RenderTarget[2].BlendEnable = true.into();
        g.depth.StencilReadMask = 0x3c;
        g.rasterizer.DepthBias = -4;
        g.samples.Count = 4;
        let bytes = g.encode();
        let back = GraphicsState::decode(&bytes).expect("decodes");
        assert_eq!(back.describe(), g.describe());
        assert_eq!(back.encode(), bytes);
        assert!(GraphicsState::decode(&bytes[..bytes.len() - 1]).is_none());
        assert!(GraphicsState::decode(&[bytes.as_slice(), &[0]].concat()).is_none());
    }

    #[test]
    fn cache_state_decodes_depth_cull_and_blend() {
        let mut state = vec![0u8; 0x17c];
        let put = |s: &mut Vec<u8>, at: usize, v: u32| s[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut state, 0, 0x4005_d800);
        put(&mut state, 4, 4);
        put(&mut state, 0xa0, 2); // cull back
        put(&mut state, 0xb8, 2 | (6 << 4)); // depth on, no write, GEQUAL
        put(&mut state, 0xcc, 0xf);
        // src alpha, add, inv src alpha, enabled.
        put(&mut state, 0xd0, 4 | (5 << 8) | (1 << 30));
        let mut outputs = vec![0u8; 0x50];
        outputs[0x10] = 28;
        outputs[0x24] = 20;
        outputs[0x25] = 1;
        outputs[0x26] = 3;
        let g = GraphicsState::from_cache(&state, &outputs);
        assert_eq!(g.rasterizer.CullMode, D3D12_CULL_MODE_BACK);
        assert!(g.depth.DepthEnable.as_bool());
        assert_eq!(g.depth.DepthWriteMask, D3D12_DEPTH_WRITE_MASK_ZERO);
        assert_eq!(g.depth.DepthFunc, D3D12_COMPARISON_FUNC_GREATER_EQUAL);
        let t = g.blend.RenderTarget[0];
        assert!(t.BlendEnable.as_bool());
        assert_eq!((t.SrcBlend, t.DestBlend, t.BlendOp), (D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD));
        assert_eq!(g.blend.RenderTarget[1].RenderTargetWriteMask, 0);
        assert_eq!((g.targets.NumRenderTargets, g.targets.RTFormats[0].0, g.depth_format.0), (1, 28, 20));
        assert_eq!(g.topology, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    }
}
