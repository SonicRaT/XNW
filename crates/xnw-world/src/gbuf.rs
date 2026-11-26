//! `XNW_WORLD_RENDER=...,gbuf=on` (LIT2, default off): our pass also writes
//! the title's G-buffer layout beside its picture, for the title's own
//! lighting programs to read (`world-render --lighting`):
//! - target 1, RT0 of the title: albedo, R8G8B8A8_UNORM_SRGB, written
//!   linear (the sampled value, as the title's `PS_DeferredTextured`
//!   writes it: [V aa739e09]); alpha 0 (the flag byte is 0 on opaque rows).
//! - target 2, RT1 of the title: (0, b0, b1, b2) / 255 of the octahedral
//!   normal packed 12 + 12 bits ([`pack_normal`], the title's
//!   `PS_DeferredTextured` ff2c46ab 0x1d0..0x2ec [V disasm]).
//! - depth: the pass's own D32 (reverse Z), read back as R32_FLOAT.
//!
//! Only the raw path's pixel entry writes targets 1 and 2 (`#define GBUF`):
//! every other pipeline binds them with a zero write mask ([`targets`]
//! reads which targets a pixel program writes from its signature).

use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The extra targets' formats (title RT0, RT1).
pub const ALBEDO: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
pub const NORMAL: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

/// Process-wide: set by `Pass::new` before it makes its pipelines (one
/// pass per process; the pipeline makers read it).
static ON: AtomicBool = AtomicBool::new(false);

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

pub(crate) fn set(v: bool) {
    ON.store(v, Ordering::Relaxed);
}

/// `gbuf=on|off` in `XNW_WORLD_RENDER` (the last one wins; default off).
pub fn parse_switch(v: &str) -> bool {
    v.split(',').filter_map(|p| p.trim().strip_prefix("gbuf=")).next_back() == Some("on")
}

/// The raw shader source as the pass compiles it: with `#define GBUF 1`
/// first when the G-buffer is on (and `OURLIGHT` with `ourlight=on`).
pub fn source(src: &str) -> String {
    let src = plain_source(src);
    if on() && !src.contains("#define GBUF") { format!("#define GBUF 1\n{src}") } else { src }
}

/// The raw source of the entry without the G-buffer outputs (dual-source
/// states): `#define OURLIGHT 1` first with `ourlight=on` (its rows before
/// the lighting pass write their albedo), else as it is.
pub fn plain_source(src: &str) -> String {
    if crate::ourlight::on() && !src.contains("#define OURLIGHT") { format!("#define OURLIGHT 1\n{src}") } else { src.to_string() }
}

/// The title's normal packing (PS_DeferredTextured ff2c46ab): octahedral
/// (|x|+|y|+|z| = 1, the lower half folded), each of x', y' as
/// trunc(2047.5 v + 2047.5) in 12 bits, word = y' << 12 | x', stored as
/// RT1 = (0, byte 0, byte 1, byte 2) / 255. The HLSL in `RAW_SHADER` is the same.
pub fn pack_normal(n: [f32; 3]) -> [u8; 4] {
    let s = n[0].abs() + n[1].abs() + n[2].abs();
    let s = if s > 0.0 { s } else { 1.0 };
    let f = (-n[2] / s).clamp(0.0, 1.0);
    let fold = |v: f32| v + if v >= 0.0 { f } else { -f };
    let q = |v: f32| ((v * 2047.5 + 2047.5) as u32).min(4095);
    let w = q(fold(n[1] / s)) << 12 | q(fold(n[0] / s));
    [0, (w & 255) as u8, (w >> 8 & 255) as u8, (w >> 16 & 255) as u8]
}

/// The inverse of [`pack_normal`] (unit length), for tests and the sheets.
pub fn unpack_normal(b: [u8; 4]) -> [f32; 3] {
    let w = u32::from(b[1]) | u32::from(b[2]) << 8 | u32::from(b[3]) << 16;
    let d = |q: u32| (q as f32 + 0.5 - 2047.5) / 2047.5;
    let (x, y) = (d(w & 4095), d(w >> 12 & 4095));
    let z = 1.0 - x.abs() - y.abs();
    let (x, y) = if z < 0.0 { ((1.0 - y.abs()) * x.signum(), (1.0 - x.abs()) * y.signum()) } else { (x, y) };
    let l = (x * x + y * y + z * z).sqrt();
    [x / l, y / l, z / l]
}

/// The render targets a compiled pixel program (DXBC) writes, as a mask
/// of `SV_Target` registers (bit k: target k), from its output signature
/// (`OSGN` 24-byte elements, `OSG5` 28, `OSG1` 32: the stream word first).
pub fn targets(dxbc: &[u8]) -> u32 {
    let word = |b: &[u8], at: usize| b.get(at..at + 4).map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
    if dxbc.get(..4) != Some(b"DXBC") {
        return 1;
    }
    let mut mask = 0;
    for k in 0..word(dxbc, 28) as usize {
        let at = word(dxbc, 32 + 4 * k) as usize;
        let (Some(tag), data) = (dxbc.get(at..at + 4), at + 8) else { continue };
        let (size, lead) = match tag {
            b"OSGN" => (24, 0),
            b"OSG5" => (28, 4),
            b"OSG1" => (32, 4),
            _ => continue,
        };
        let chunk = &dxbc[data.min(dxbc.len())..];
        for e in 0..word(chunk, 0) as usize {
            let el = 8 + size * e + lead;
            let name = word(chunk, el) as usize;
            let is_target = chunk.get(name..name + 10).is_some_and(|n| n.eq_ignore_ascii_case(b"SV_Target\0"));
            let reg = word(chunk, el + 16);
            if is_target && reg < 8 {
                mask |= 1 << reg;
            }
        }
    }
    mask
}

/// Whether a title blend state (`RenderState::blend`: enable, src, dst, op,
/// srcA, dstA, opA, mask) blends with source 1 (dual source: one target only).
pub fn dual_source(b: &[u8; 8]) -> bool {
    b[0] != 0 && [b[1], b[2], b[4], b[5]].iter().any(|v| (16..=19).contains(v))
}

/// The pixel program for a title blend state: a state that blends with
/// source 1 gets `plain` (the entry compiled without `GBUF`) when there is
/// one. Dual-source blending reads the second output as source 1, and the
/// `GBUF` entry writes the albedo there: the glass of pk7a f960 (draw 5049,
/// state 272) let the interior through with the switch on (LIT2: 43,649 px
/// at 1280x720; LIT3: 0 with this).
pub(crate) fn pixel_for<'a, T>(ps: &'a T, plain: Option<&'a T>, blend: &[u8; 8]) -> &'a T {
    match plain {
        Some(p) if dual_source(blend) => p,
        _ => ps,
    }
}

/// Sets a pipeline description's targets: the pass's colour target, and
/// with the G-buffer on, targets 1 and 2 with a write mask only where the
/// pixel program `ps` writes them (independent blend: only target 0
/// blends). A state blending with source 1 (dual source: one target only)
/// keeps one target; the G-buffer keeps what the rows under it wrote.
pub(crate) fn layout(desc: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC, ps: &[u8]) {
    let rt = &desc.BlendState.RenderTarget[0];
    let src1 = |b: D3D12_BLEND| matches!(b, D3D12_BLEND_SRC1_COLOR | D3D12_BLEND_INV_SRC1_COLOR | D3D12_BLEND_SRC1_ALPHA | D3D12_BLEND_INV_SRC1_ALPHA);
    if !on() || (rt.BlendEnable.as_bool() && [rt.SrcBlend, rt.DestBlend, rt.SrcBlendAlpha, rt.DestBlendAlpha].into_iter().any(src1)) {
        return;
    }
    let writes = targets(ps);
    desc.BlendState.IndependentBlendEnable = true.into();
    desc.NumRenderTargets = 3;
    (desc.RTVFormats[1], desc.RTVFormats[2]) = (ALBEDO, NORMAL);
    for k in 1..3 {
        desc.BlendState.RenderTarget[k].RenderTargetWriteMask = if writes & 1 << k != 0 { D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8 } else { 0 };
    }
}

/// The G-buffer as read back: albedo and normal rows (RGBA8, `width * 4`
/// bytes a row, albedo sRGB-encoded) and the depth (reverse Z, 0 = far).
pub struct Read {
    pub width: u32,
    pub height: u32,
    pub albedo: Vec<u8>,
    pub normal: Vec<u8>,
    pub depth: Vec<f32>,
}

/// The two extra targets of a pass.
pub struct Targets {
    pub albedo: ID3D12Resource,
    pub normal: ID3D12Resource,
}

/// Reads the two targets and the depth back on a submission of its own
/// (they stay in the render-target / depth-write states between frames).
pub fn read(device: &ID3D12Device, queue: &ID3D12CommandQueue, t: &Targets, depth: &ID3D12Resource, width: u32, height: u32) -> Result<Read, String> {
    let e = |e: windows::core::Error| e.message().to_string();
    let pitch = (width * 4).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    let size = u64::from(pitch) * u64::from(height);
    // SAFETY: D3D12 calls on objects that live through the wait below.
    unsafe {
        let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(e)?;
        let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(e)?;
        let mut backs = Vec::new();
        for (r, format, state) in [(&t.albedo, ALBEDO, D3D12_RESOURCE_STATE_RENDER_TARGET), (&t.normal, NORMAL, D3D12_RESOURCE_STATE_RENDER_TARGET), (depth, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_STATE_DEPTH_WRITE)] {
            let back = crate::bridge::buffer(device, size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            crate::transition(&list, r, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
            let fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: if format == DXGI_FORMAT_R32_TYPELESS { DXGI_FORMAT_R32_FLOAT } else { format }, Width: width, Height: height, Depth: 1, RowPitch: pitch } };
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::ManuallyDrop::new(Some(back.clone())), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::ManuallyDrop::new(Some(r.clone())), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            drop(std::mem::ManuallyDrop::into_inner(dst.pResource));
            drop(std::mem::ManuallyDrop::into_inner(src.pResource));
            crate::transition(&list, r, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
            backs.push(back);
        }
        list.Close().map_err(e)?;
        queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).map_err(e)?)]);
        let sent = xnw_gpu::gpuwatch::begin("gbuf read");
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e)?;
        queue.Signal(&fence, 1).map_err(e)?;
        let t0 = std::time::Instant::now();
        while fence.GetCompletedValue() < 1 {
            device.GetDeviceRemovedReason().map_err(|x| format!("G-buffer read: {}", e(x)))?;
            if t0.elapsed().as_secs() > 60 {
                xnw_gpu::gpuwatch::abandon("gbuf read", 60.0, xnw_gpu::gpuwatch::GRACE);
                return Err("G-buffer read: over 60 s".into());
            }
            std::thread::yield_now();
        }
        xnw_gpu::gpuwatch::end("gbuf read", sent, None);
        let mut rows = Vec::new();
        for b in &backs {
            let mut p = std::ptr::null_mut();
            b.Map(0, None, Some(&mut p)).map_err(e)?;
            let mut v = Vec::with_capacity((width * 4 * height) as usize);
            for y in 0..height as usize {
                v.extend_from_slice(std::slice::from_raw_parts(p.cast::<u8>().add(y * pitch as usize), (width * 4) as usize));
            }
            b.Unmap(0, None);
            rows.push(v);
        }
        let depth = rows.pop().unwrap_or_default().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let normal = rows.pop().unwrap_or_default();
        let albedo = rows.pop().unwrap_or_default();
        Ok(Read { width, height, albedo, normal, depth })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_source_states_get_the_plain_entry() {
        // enable, src ONE (2), dst SRC1_COLOR (16), add: source 1 read.
        let dual = [1, 2, 16, 1, 2, 16, 1, 15];
        let plain_blend = [1, 5, 6, 1, 5, 6, 1, 15];
        let (g, p) = (1u32, 2u32);
        assert_eq!(*pixel_for(&g, Some(&p), &dual), 2);
        assert_eq!(*pixel_for(&g, Some(&p), &plain_blend), 1);
        assert_eq!(*pixel_for(&g, None, &dual), 1);
    }

    #[test]
    fn normal_packing_round_trips() {
        for n in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.6, -0.48, -0.64], [-0.3, 0.4, 0.866]] {
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2] as f32).sqrt();
            let n = n.map(|x: f32| x / l);
            let m = unpack_normal(pack_normal(n));
            let dot = n[0] * m[0] + n[1] * m[1] + n[2] * m[2];
            assert!(dot > 0.9999, "{n:?} -> {m:?}");
        }
        // Up: x' = y' = 0 -> 2047 each: word 0x7FF7FF, bytes FF F7 7F.
        assert_eq!(pack_normal([0.0, 0.0, 1.0]), [0, 0xFF, 0xF7, 0x7F]);
    }

    #[test]
    fn switch_last_wins() {
        assert!(!parse_switch("screen"));
        assert!(parse_switch("screen,gbuf=on"));
        assert!(!parse_switch("gbuf=on,gbuf=off"));
    }

    #[test]
    fn pixel_targets_from_the_signature() {
        let one = crate::compile_source(crate::RAW_SHADER, c"ps", c"ps_5_1").unwrap();
        let three = crate::compile_source(&format!("#define GBUF 1\n{}", crate::RAW_SHADER), c"ps", c"ps_5_1").unwrap();
        assert_eq!(targets(crate::blob_bytes(&one)), 1);
        assert_eq!(targets(crate::blob_bytes(&three)), 7);
    }
}
