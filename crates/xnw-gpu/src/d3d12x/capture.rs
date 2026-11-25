//! Presented frames written to PNG files, to see what reaches the screen:
//! with `XNW_CAPTURE=<folder>`, every frame whose number is a power of
//! two (the frames the log reports) is saved as `frame-<n>.png`, at half
//! size. What is saved is the host image the window shows: the composited
//! back buffer, or with one plane the plane's host texture that is copied
//! to it; never title memory.
//!
//! Ported from VIctory-base (VIctory main d2a4bae + 26 PRs, branch
//! `xnw/base` @ 5d0516e, GPL-3.0-or-later), `crates/victory-gpu/src/
//! capture.rs` lines 1-17 (`folder`), 65-131 (`wanted`, `save`, `save_as`,
//! `chunk`, `crc32`), 132-241 (`read_back`, `save_read_back`) and the test
//! 636-642. Changed: `XNW_CAPTURE` (was `VICTORY_CAPTURE`); in every build
//! (the base: diagnostics builds only); no frame list, busy-frame or
//! target-dump modes; `tracing` is `xnw_core`; the saved line counts
//! the pixels that are not black.

use std::path::PathBuf;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The folder frames go to, if capturing.
pub(crate) fn folder() -> Option<&'static PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| {
            let dir = std::env::var_os("XNW_CAPTURE").filter(|d| !d.is_empty()).map(PathBuf::from)?;
            match std::fs::create_dir_all(&dir) {
                Ok(()) => Some(dir),
                Err(e) => {
                    xnw_core::warn!("XNW_CAPTURE: cannot use {}: {e}", dir.display());
                    None
                }
            }
        })
        .as_ref()
}

/// Whether frame `n` is saved: powers of two, and with
/// `XNW_CAPTURE_EVERY=<k>` every k-th frame too.
pub(crate) fn wanted(n: u64) -> bool {
    static EVERY: std::sync::LazyLock<u64> =
        std::sync::LazyLock::new(|| std::env::var("XNW_CAPTURE_EVERY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0));
    folder().is_some() && (n.is_power_of_two() || (*EVERY > 0 && n % *EVERY == 0))
}

/// Saves 8-bit RGBA or BGRA rows as `<name>.png`, at half size.
fn save_as(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    save_scaled(name, width, height, pitch, bgra, rows, 2);
}

/// As [`save_as`], every `step`-th pixel of every `step`-th row.
fn save_scaled(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8], step: usize) {
    let Some(folder) = folder() else { return };
    let (w, h) = ((width as usize / step).max(1) as u32, (height as usize / step).max(1) as u32);
    // Filter byte 0 per row, then RGB of every other pixel of every other row.
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    let mut lit = 0u64;
    for y in 0..h as usize {
        raw.push(0);
        let row = &rows[(step * y).min(height as usize - 1) * pitch..];
        for x in 0..w as usize {
            let p = &row[(step * x).min(width as usize - 1) * 4..][..3];
            lit += u64::from(p != [0, 0, 0]);
            if bgra {
                raw.extend([p[2], p[1], p[0]]);
            } else {
                raw.extend_from_slice(p);
            }
        }
    }
    // Compressing and writing took 40-200 ms on the presenting thread
    // (a frame spike at every saved frame); a thread of its own does it.
    let name = name.to_owned();
    let encode = move || {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut header = Vec::new();
        header.extend(w.to_be_bytes());
        header.extend(h.to_be_bytes());
        header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
        chunk(&mut png, b"IHDR", &header);
        chunk(&mut png, b"IDAT", &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 3));
        chunk(&mut png, b"IEND", &[]);
        let path = folder.join(format!("{name}.png"));
        match std::fs::write(&path, png) {
            Ok(()) => xnw_core::info!("capture: {name} ({width}x{height}) saved to {} ({lit} of {} saved pixels not black)", path.display(), u64::from(w) * u64::from(h)),
            Err(e) => xnw_core::warn!("capture: {name} not saved to {}: {e}", path.display()),
        }
    };
    if std::thread::Builder::new().name("xnw capture".into()).spawn(encode).is_err() {
        xnw_core::warn!("capture: no thread to save a frame");
    }
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Records a copy of `texture` (in the copy-source state) into a
/// readback buffer.
pub(crate) fn read_back(device: &ID3D12Device, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    read_back_sub(device, list, texture, 0)
}

/// As [`read_back`], for subresource `sub` (a mip level of an unarrayed
/// texture).
fn read_back_sub(device: &ID3D12Device, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource, sub: u32) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(&desc, sub, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
    }
    footprint.Offset = 0;
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer).ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some((buffer, footprint))
    }
}

/// Saves a frame read back by [`read_back`] (8-bit color only).
pub(crate) fn save_read_back(frame: u64, buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
    save_read_back_as(frame, &format!("frame-{frame}"), 2, buffer, footprint);
}

/// A presented world frame (`XNW_CAPTURE_TARGETS=world+...`), at
/// quarter size as `world-<frame>.png`.
pub(crate) fn save_world_frame(frame: u64, buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
    save_read_back_as(frame, &format!("world-{frame}"), 4, buffer, footprint);
}

fn save_read_back_as(frame: u64, name: &str, step: usize, buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
    let f = footprint.Footprint;
    let bgra = match f.Format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            xnw_core::warn!("capture: frame {frame} not saved: format {} is not 8-bit color", other.0);
            return;
        }
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished (every submission is waited for); the
    // mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let len = f.RowPitch as usize * f.Height as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), len);
        save_scaled(name, f.Width, f.Height, f.RowPitch as usize, bgra, rows, step);
        buffer.Unmap(0, None);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
    }

    /// A D32S8 depth target (format 19/20) copies its depth plane as
    /// R32_TYPELESS (39), which the capture reads as 32-bit float depth.
    /// Measured on the RX 5700 XT: 19, 20, 44, 45, 39 -> 39; 40 -> 40;
    /// 53 -> 53; 55 -> 55 (D24 planes would read as float: not decoded right).
    #[test]
    fn depth_plane_footprints() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        if std::env::var_os("XNW_AGILITY").is_none_or(|d| d.is_empty()) {
            println!("SKIPPED: set XNW_AGILITY");
            return;
        }
        let host = crate::d3d12x::host::host().expect("a host GPU");
        for f in [19, 20, 44, 45, 39, 40, 53, 55] {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 640,
                Height: 360,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT(f),
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                ..Default::default()
            };
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0;
            // SAFETY: a plain query.
            unsafe { host.device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total)) };
            println!("format {f}: plane 0 footprint format {} row pitch {} total {total}", fp.Footprint.Format.0, fp.Footprint.RowPitch);
            if matches!(f, 19 | 20) {
                assert_eq!(fp.Footprint.Format.0, 39);
                assert_eq!(super::bytes_per_texel(fp.Footprint.Format), 4);
                let half = 0.5f32.to_le_bytes();
                assert_eq!(super::texel(fp.Footprint.Format, &half).map(|t| t.0[0]), Some(0.5));
            }
        }
    }
}

// ---- Render targets of one frame (XNW) ----
//
// `XNW_CAPTURE_TARGETS=<n>` saves, in frame `n` (presented frames), with
// `mesh+<k>` in the k-th frame after the first frame with mesh draws, or
// with `rect+<k>` after the first rectangle variant drawn (the world), the
// render and depth targets each time a list rebinds its targets or closes
// after drawing into them: `target-<frame>-<seq>-<rt|ds><i>-<format>.png`,
// in bind order, at most 160 per frame. Colour is scaled to the image's own
// range (so dark HDR targets show); depth is grey from its range. Written
// after the frame's work has run. Each saved target also logs its channel
// minima, maxima, means and NaN/Inf count, and the pass (pipeline) that
// wrote it, so frames compare in the log. `<spec>:<k>` saves `k`
// consecutive frames from the trigger (`at+60:6`); compute dispatches in
// those frames also save the textures they write (`uav<i>`).

/// The frame whose targets are saved, once known.
fn target_frame() -> Option<u64> {
    static SPEC: std::sync::LazyLock<Option<(u8, u64)>> = std::sync::LazyLock::new(|| {
        let v = std::env::var("XNW_CAPTURE_TARGETS").ok()?;
        let v = v.trim();
        let v = v.split_once(':').map_or(v, |(a, _)| a);
        if let Some(k) = v.strip_prefix("mesh+") {
            return k.parse().ok().map(|k| (1, k));
        }
        if let Some(k) = v.strip_prefix("rect+") {
            return k.parse().ok().map(|k| (2, k));
        }
        // `at+<seconds>`: the first frame after that many seconds on the log
        // clock (frame numbers and draw counts shift with the frame rate).
        if let Some(k) = v.strip_prefix("at+") {
            return k.parse().ok().map(|k| (3, k));
        }
        // `flash@<seconds>[:<luma>]`: from that time every frame's
        // targets are copied, and kept only around a presented frame whose
        // maximum luma passes the threshold (default 32): that frame, the
        // one before and the one after are saved.
        if let Some(k) = v.strip_prefix("flash@") {
            return k.parse().ok().map(|k| (4, k));
        }
        // `world+<k>`: k frames after the world starts (see
        // `WorldDetector`); at least 1, the start is known one frame late.
        if let Some(k) = v.strip_prefix("world+") {
            return k.parse().ok().map(|k: u64| (5, k.max(1)));
        }
        v.parse().ok().map(|n| (0, n))
    });
    let (mode, n) = (*SPEC)?;
    folder()?;
    match mode {
        0 => return Some(n),
        2 => {
            let f = super::draw::RECT_VARIANT_FRAME.load(std::sync::atomic::Ordering::Relaxed);
            return (f != u64::MAX).then(|| f + n);
        }
        4 => {
            if xnw_core::uptime().as_secs() < n {
                return None;
            }
            return Some(super::queue::presented());
        }
        5 => return world_start().map(|w| w + n),
        3 => {
            static AT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            if xnw_core::uptime().as_secs() < n {
                return None;
            }
            return Some(*AT.get_or_init(|| super::queue::presented() + 1));
        }
        _ => {}
    }
    static FIRST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    if crate::mesh::MESH_DRAWS.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return None;
    }
    Some(*FIRST.get_or_init(|| super::queue::presented()) + n)
}

/// Consecutive frames saved from the trigger (`<spec>:<k>`, default 1).
fn target_frames() -> u64 {
    static K: std::sync::LazyLock<u64> =
        std::sync::LazyLock::new(|| std::env::var("XNW_CAPTURE_TARGETS").ok().and_then(|v| v.split_once(':').and_then(|(_, k)| k.trim().parse().ok())).unwrap_or(1).clamp(1, 64));
    *K
}

struct Snapshot {
    frame: u64,
    pass: String,
    name: String,
    buffer: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
}

static SNAPSHOTS: std::sync::Mutex<Vec<Snapshot>> = std::sync::Mutex::new(Vec::new());
/// (frame, snapshots taken in it).
static TAKEN: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));
const PER_FRAME: u64 = 640;

/// Whether targets are saved now.
pub(crate) fn targets_wanted() -> bool {
    let Some(f) = target_frame() else { return false };
    let now = super::queue::presented();
    if !flash_mode() && now >= f + 3 {
        drain(now);
    }
    let taken = TAKEN.lock().unwrap_or_else(|e| e.into_inner());
    (f..f + target_frames()).contains(&now) && (taken.0 != now || taken.1 < PER_FRAME)
}

/// Records copies of `texture` (title resource `this`, a render target
/// `rt` or depth) into readback memory on `list`.
pub(crate) fn snapshot(list: &super::host::HostList, this: super::object::This, texture: &ID3D12Resource, label: &str, pass: &str) {
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    if desc.SampleDesc.Count > 1 || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        return;
    }
    // Flash mode copies every frame: square shadow and atlas targets left out.
    if flash_mode() && desc.Width == u64::from(desc.Height) && desc.Width >= 512 {
        return;
    }
    let frame = super::queue::presented();
    let n = {
        let mut taken = TAKEN.lock().unwrap_or_else(|e| e.into_inner());
        if taken.0 != frame {
            *taken = (frame, 0);
        }
        taken.1 += 1;
        taken.1 - 1
    };
    if n >= PER_FRAME {
        return;
    }
    const COPY_SOURCE: u32 = 0x800;
    super::list::ensure_state(list, this, COPY_SOURCE);
    let Some(device) = super::host::host_device() else { return };
    let Some((buffer, footprint)) = read_back(&device.device, &list.list, texture) else { return };
    let name = format!("target-{frame}-{n:02}-{label}-f{}-{}x{}", desc.Format.0, desc.Width, desc.Height);
    SNAPSHOTS.lock().unwrap_or_else(|e| e.into_inner()).push(Snapshot { frame, pass: pass.to_owned(), name, buffer, footprint });
}

/// Whether `XNW_CAPTURE_TARGETS=flash@...`.
fn flash_mode() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_CAPTURE_TARGETS").is_ok_and(|v| v.trim().starts_with("flash@")));
    *ON
}

/// Flash mode, at the luma of presented frame `frame` (its lists were
/// recorded while `frame - 1` frames had been presented): keeps the
/// snapshots of this frame and the one before when `max` passes the
/// threshold, and those of the next frame; drops the older ones.
fn flash_luma(frame: u64, max: u32, lit: f64) -> bool {
    static THRESHOLD: std::sync::LazyLock<u32> =
        std::sync::LazyLock::new(|| std::env::var("XNW_CAPTURE_TARGETS").ok().and_then(|v| v.split_once(':').and_then(|(_, k)| k.trim().parse().ok())).unwrap_or(32));
    static SAVE_NEXT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static EVENTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    // Only once the trigger time has passed (front-end frames are lit).
    if target_frame().is_none() {
        return false;
    }
    // The work shown by presented frame `frame` was recorded two frames
    // earlier (the title records ahead): a test run saved all-zero copies
    // for `frame - 1` (not yet executed) and the lit scene for `frame - 2`.
    let tag = frame.saturating_sub(2);
    // A flash: bright pixels on a mostly black frame (loading screens and
    // the mission card are lit over most of the frame, or dim).
    let flash = max > *THRESHOLD && lit < 50.0 && EVENTS.load(std::sync::atomic::Ordering::Relaxed) < 2;
    // The first frame after the trigger time is saved whole, as the
    // baseline (normally a black one).
    static BASE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let base = max <= 2 && !BASE.swap(true, std::sync::atomic::Ordering::Relaxed);
    let next = SAVE_NEXT.swap(false, std::sync::atomic::Ordering::Relaxed) || base;
    let keep_from = if flash { tag.saturating_sub(1) } else { tag };
    let (save, drop_count): (Vec<Snapshot>, usize) = {
        let mut all = SNAPSHOTS.lock().unwrap_or_else(|e| e.into_inner());
        let taken = std::mem::take(&mut *all);
        let mut save = Vec::new();
        let mut dropped = 0;
        for s in taken {
            if (flash && s.frame + 1 >= tag && s.frame <= tag) || (next && s.frame == tag) {
                save.push(s);
            } else if s.frame >= keep_from {
                all.push(s);
            } else {
                dropped += 1;
            }
        }
        (save, dropped)
    };
    let _ = drop_count;
    if flash {
        EVENTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SAVE_NEXT.store(true, std::sync::atomic::Ordering::Relaxed);
        xnw_core::info!("capture: flash at presented frame {frame} (luma max {max}): saving {} targets of it and the frame before", save.len());
    } else if next {
        xnw_core::info!("capture: frame after the flash, presented {frame} (luma max {max}): saving {} targets", save.len());
    }
    for s in save {
        save_target(&s.name, &s.pass, &s.buffer, &s.footprint);
    }
    flash || next
}

/// Saves the snapshots whose frame has run (two frames back; every
/// submission is waited for).
fn drain(now: u64) {
    let ready: Vec<Snapshot> = {
        let mut all = SNAPSHOTS.lock().unwrap_or_else(|e| e.into_inner());
        // Work recorded while `t` frames had been presented is shown, so
        // has run, by present `t + 2` (the title records two frames ahead,
        // as a test run showed); saved one frame later still.
        let (ready, keep) = std::mem::take(&mut *all).into_iter().partition(|s| s.frame + 3 <= now);
        *all = keep;
        ready
    };
    for s in ready {
        save_target(&s.name, &s.pass, &s.buffer, &s.footprint);
    }
}

/// One texel as up to four floats, by format (typeless forms read as
/// their usual typed one); `None` for formats not decoded.
fn texel(format: DXGI_FORMAT, p: &[u8]) -> Option<([f32; 4], bool)> {
    let u16f = |i: usize| f16(u16::from_le_bytes([p[2 * i], p[2 * i + 1]]));
    let f32at = |i: usize| f32::from_le_bytes([p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]]);
    let w = |i: usize| u32::from_le_bytes([p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]]);
    Some(match format.0 {
        // R8G8B8A8, B8G8R8A8 (and typeless, sRGB)
        27..=32 => ([p[0], p[1], p[2], p[3]].map(|b| f32::from(b) / 255.0), false),
        87 | 88 | 90..=93 => ([p[2], p[1], p[0], p[3]].map(|b| f32::from(b) / 255.0), false),
        // R10G10B10A2
        23..=25 => {
            let v = w(0);
            ([v & 1023, v >> 10 & 1023, v >> 20 & 1023, v >> 30].map(|c| c as f32 / 1023.0), false)
        }
        // R11G11B10 float
        26 => {
            let v = w(0);
            let f = |m: u32, e: u32, mb: u32| if e == 0 { m as f32 / (1 << mb) as f32 * 2f32.powi(-14) } else { (1.0 + m as f32 / (1 << mb) as f32) * 2f32.powi(e as i32 - 15) };
            ([f(v & 63, v >> 6 & 31, 6), f(v >> 11 & 63, v >> 17 & 31, 6), f(v >> 22 & 31, v >> 27 & 31, 5), 1.0], false)
        }
        // R16G16B16A16 float/typeless
        9 | 10 => ([u16f(0), u16f(1), u16f(2), u16f(3)], false),
        // R16G16 float
        33 | 34 => ([u16f(0), u16f(1), 0.0, 1.0], false),
        // R16 float
        // R16 typeless (D16 depth) as unorm, R16 float as float.
        53 => ([f32::from(u16::from_le_bytes([p[0], p[1]])) / 65535.0; 4], true),
        54 => {
            let v = u16f(0);
            ([v, v, v, 1.0], true)
        }
        // R32G32B32A32 float
        1 | 2 => ([f32at(0), f32at(1), f32at(2), f32at(3)], false),
        // R32G32 float
        15 | 16 => ([f32at(0), f32at(1), 0.0, 1.0], false),
        // R32 float, D32, R32 typeless: depth-like grey
        39..=41 => ([f32at(0); 4], true),
        // D32S8 plane / R32G8X24
        19..=22 => ([f32at(0); 4], true),
        // D24S8, R24G8: 24-bit unorm depth
        44..=47 => ([(w(0) & 0xff_ffff) as f32 / 16_777_215.0; 4], true),
        // R8, R8G8
        60..=64 => ([f32::from(p[0]) / 255.0; 4], true),
        48..=52 => ([f32::from(p[0]) / 255.0, f32::from(p[1]) / 255.0, 0.0, 1.0], false),
        // R16 unorm/typeless
        55 | 56 => ([f32::from(u16::from_le_bytes([p[0], p[1]])) / 65535.0; 4], true),
        _ => return None,
    })
}

fn f16(h: u16) -> f32 {
    let (s, e, m) = (h >> 15, (h >> 10) & 31, h & 1023);
    let v = match e {
        0 => f32::from(m) * 2f32.powi(-24),
        31 => f32::NAN,
        _ => (1.0 + f32::from(m) / 1024.0) * 2f32.powi(i32::from(e) - 15),
    };
    if s == 1 { -v } else { v }
}

fn bytes_per_texel(format: DXGI_FORMAT) -> usize {
    match format.0 {
        1 | 2 => 16,
        9 | 10 | 15 | 16 | 19..=22 => 8,
        48..=59 => 2,
        60..=65 => 1,
        _ => 4,
    }
}

/// Saves a snapshot, converted for viewing.
fn save_target(name: &str, pass: &str, buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
    let f = footprint.Footprint;
    let (w, h) = (f.Width as usize, f.Height as usize);
    let bpt = bytes_per_texel(f.Format);
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy has run (two frames on, every submission waited
    // for); the mapping spans the footprint's rows.
    let pixels: Option<Vec<([f32; 4], bool)>> = unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), f.RowPitch as usize * h);
        let out = (0..h).flat_map(|y| (0..w).map(move |x| (y, x))).map(|(y, x)| texel(f.Format, &rows[y * f.RowPitch as usize + x * bpt..][..bpt])).collect();
        buffer.Unmap(0, None);
        out
    };
    let Some(pixels) = pixels else {
        xnw_core::info!("capture: {name} not saved: format {} not decoded", f.Format.0);
        return;
    };
    // Range over finite values (colour channels together).
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for (p, _) in &pixels {
        for &c in &p[..3] {
            if c.is_finite() {
                lo = lo.min(c);
                hi = hi.max(c);
            }
        }
    }
    let grey = pixels.first().is_some_and(|p| p.1);
    let lo = if grey { lo } else { lo.min(0.0) };
    let span = if hi > lo { hi - lo } else { 1.0 };
    let mut rgba = Vec::with_capacity(w * h * 4);
    for (p, _) in &pixels {
        for &c in &p[..3] {
            let v = if c.is_finite() { (c - lo) / span } else { 1.0 };
            rgba.push((v.clamp(0.0, 1.0) * 255.0) as u8);
        }
        rgba.push(255);
    }
    // Alpha separately: zero colour with alpha written means the shader ran
    // over empty inputs; zero alpha too means the write itself was lost.
    let (mut alo, mut ahi) = (f32::INFINITY, f32::NEG_INFINITY);
    for (p, _) in &pixels {
        if p[3].is_finite() {
            alo = alo.min(p[3]);
            ahi = ahi.max(p[3]);
        }
    }
    xnw_core::info!("capture: {name} range {lo} .. {hi}, alpha {alo} .. {ahi}");
    let st = stats(&pixels);
    xnw_core::info!("capture: {name} stats {st} pass {pass}");
    save_as(name, w as u32, h as u32, w * 4, false, &rgba);
}

/// Per channel min/max/mean over finite values, and the NaN/Inf count.
fn stats(pixels: &[([f32; 4], bool)]) -> String {
    let mut lo = [f32::INFINITY; 4];
    let mut hi = [f32::NEG_INFINITY; 4];
    let mut sum = [0f64; 4];
    let mut n = [0u64; 4];
    let mut bad = 0u64;
    for (p, _) in pixels {
        for c in 0..4 {
            if p[c].is_finite() {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
                sum[c] += f64::from(p[c]);
                n[c] += 1;
            } else {
                bad += 1;
            }
        }
    }
    let ch: Vec<String> = (0..4).map(|c| format!("{:.4}/{:.4}/{:.4}", lo[c], sum[c] / n[c].max(1) as f64, hi[c])).collect();
    format!("min/mean/max r {} g {} b {} a {}; nonfinite {bad}", ch[0], ch[1], ch[2], ch[3])
}

/// `XNW_FRAME_LUMA=1`: every presented frame is read back and its
/// maximum and mean luma (8-bit) logged, so frames with a picture show in
/// the log without saving images.
pub(crate) fn luma_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_FRAME_LUMA").is_ok_and(|v| v == "1") || flash_mode());
    *ON
}

/// Logs the luma of a presented frame read back by [`read_back`]; true
/// when flash mode wants the presented frame saved too.
pub(crate) fn log_luma(frame: u64, buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) -> bool {
    let f = footprint.Footprint;
    let bgra = matches!(f.Format, DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB);
    if !bgra && !matches!(f.Format, DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) {
        return false;
    }
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished (the submission was waited for); the
    // mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), f.RowPitch as usize * f.Height as usize);
        let (mut max, mut sum, mut n, mut lit) = (0u32, 0u64, 0u64, 0u64);
        for y in (0..f.Height as usize).step_by(4) {
            for x in (0..f.Width as usize).step_by(4) {
                let p = &rows[y * f.RowPitch as usize + x * 4..][..3];
                let (r, b) = if bgra { (p[2], p[0]) } else { (p[0], p[2]) };
                let l = (u32::from(r) * 54 + u32::from(p[1]) * 183 + u32::from(b) * 19) >> 8;
                max = max.max(l);
                sum += u64::from(l);
                n += 1;
                lit += u64::from(l > 8);
            }
        }
        buffer.Unmap(0, None);
        xnw_core::info!("luma: frame {frame} max {max} mean {:.2} lit {:.2} %", sum as f64 / n.max(1) as f64, 100.0 * lit as f64 / n.max(1) as f64);
        flash_mode() && flash_luma(frame, max, 100.0 * lit as f64 / n.max(1) as f64)
    }
}

// ---- World start ----
//
// Time and frame-number triggers miss the world: the load's length varies.
// The world is recognised by its draws instead. A recorded frame qualifies
// when it has at least `WORLD_TOTAL` draws, at least `WORLD_DRAWS` of them
// into colour targets of one size smaller than the presented output (the
// render size after the title's render scale, e.g. 640x360 under
// 1920x1080) and at least `WORLD_DEPTH` of those with depth writes on; the
// world starts at the first of `WORLD_STREAK` consecutive qualifying frames.
// One frame of the mission card (frame 1335) passed the rule
// (50 render-size colour draws); card frames draw far fewer in total
// (frame-call windows: 7-10 draws per frame on loading screens, against
// 2,943-2,984 per world frame in one run, 1,216-1,680 of them mesh draws).
// Used by `XNW_CAPTURE_TARGETS=world+<k>[:<n>]`, `XNW_DEPTH_LOG=1` and
// `XNW_CULL_DUMP=<k>`.

/// Draws in a world frame (all targets).
const WORLD_TOTAL: u32 = 500;
/// Draws to one render-size colour target.
const WORLD_DRAWS: u32 = 50;
/// Of those, with depth writes on.
const WORLD_DEPTH: u32 = 20;
/// Consecutive qualifying frames.
const WORLD_STREAK: u32 = 3;

/// One closed frame's tally, for the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorldCheck {
    pub tag: u64,
    pub total: u32,
    pub size: (u32, u32),
    pub colour: u32,
    pub depth: u32,
    pub qualifies: bool,
    pub streak: u32,
}

/// Per recorded frame tallies of draws by colour-target size.
#[derive(Default)]
pub(crate) struct WorldDetector {
    tag: u64,
    total: u32,
    /// (size, colour draws, of those depth-writing).
    counts: Vec<((u32, u32), u32, u32)>,
    streak: u32,
    streak_start: u64,
    started: Option<(u64, (u32, u32))>,
}

impl WorldDetector {
    /// A draw recorded while `tag` frames had been presented, into colour
    /// targets whose first is `size` (0x0: none), writing depth or not,
    /// with `output` the presented size. When this begins a new frame, the
    /// previous one is closed: its check is returned.
    pub(crate) fn note(&mut self, tag: u64, size: (u32, u32), depth_write: bool, output: (u32, u32)) -> Option<WorldCheck> {
        if self.started.is_some() {
            return None;
        }
        let mut closed = None;
        if tag != self.tag {
            closed = self.close(output);
            self.tag = tag;
            self.total = 0;
            self.counts.clear();
            if self.started.is_some() {
                return closed;
            }
        }
        self.total += 1;
        if size.0 != 0 && size.1 != 0 {
            match self.counts.iter_mut().find(|c| c.0 == size) {
                Some(c) => {
                    c.1 += 1;
                    c.2 += u32::from(depth_write);
                }
                None => self.counts.push((size, 1, u32::from(depth_write))),
            }
        }
        closed
    }

    /// The finished frame's check; a third consecutive qualifying frame
    /// starts the world at the first of them.
    fn close(&mut self, output: (u32, u32)) -> Option<WorldCheck> {
        if output.0 == 0 || output.1 == 0 || self.total == 0 {
            return None;
        }
        let best = self.counts.iter().filter(|(s, _, _)| s.0 < output.0 && s.1 < output.1 && s.0 >= 256 && s.1 >= 144).max_by_key(|c| c.1).copied().unwrap_or(((0, 0), 0, 0));
        let qualifies = self.total >= WORLD_TOTAL && best.1 >= WORLD_DRAWS && best.2 >= WORLD_DEPTH;
        if qualifies {
            if self.streak > 0 && self.tag == self.streak_start + u64::from(self.streak) {
                self.streak += 1;
            } else {
                self.streak = 1;
                self.streak_start = self.tag;
            }
            if self.streak >= WORLD_STREAK {
                self.started = Some((self.streak_start, best.0));
            }
        } else {
            self.streak = 0;
        }
        Some(WorldCheck { tag: self.tag, total: self.total, size: best.0, colour: best.1, depth: best.2, qualifies, streak: self.streak })
    }

    pub(crate) fn start(&self) -> Option<(u64, (u32, u32))> {
        self.started
    }
}

static WORLD: std::sync::Mutex<Option<WorldDetector>> = std::sync::Mutex::new(None);
/// The presented output size (width << 32 | height).
static OUTPUT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WORLD_START: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// Whether anything wants the world start.
fn world_wanted() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("XNW_CAPTURE_TARGETS").is_ok_and(|v| v.trim().starts_with("world+")) || std::env::var("XNW_DEPTH_LOG").is_ok_and(|v| v == "1") || cull_k().is_some() || super::frameskip::wants_world()
    });
    *ON
}

/// The presented output size, from each present.
pub(crate) fn note_output(width: u32, height: u32) {
    OUTPUT.store(u64::from(width) << 32 | u64::from(height), std::sync::atomic::Ordering::Relaxed);
    if cull_k().is_some() {
        let now = super::queue::presented();
        cull_drain(now);
        drain(now);
    }
}

/// A draw into colour targets whose first is `size` (0x0: none), with
/// depth writes on or off.
pub(crate) fn note_draw(size: (u32, u32), depth_write: bool) {
    if !world_wanted() || WORLD_START.load(std::sync::atomic::Ordering::Relaxed) != u64::MAX {
        return;
    }
    let o = OUTPUT.load(std::sync::atomic::Ordering::Relaxed);
    let output = ((o >> 32) as u32, o as u32);
    let tag = super::queue::presented();
    let mut w = WORLD.lock().unwrap_or_else(|e| e.into_inner());
    let d = w.get_or_insert_with(WorldDetector::default);
    let Some(c) = d.note(tag, size, depth_write, output) else { return };
    // The numbers it decided on, for frames with any real work.
    static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if c.total >= 100 && LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
        xnw_core::info!(
            "capture: world check frame {}: {} draws, {} colour draws at {}x{} ({} writing depth): {} (streak {} of {WORLD_STREAK}; needs {WORLD_TOTAL} draws, {WORLD_DRAWS} colour, {WORLD_DEPTH} depth)",
            c.tag,
            c.total,
            c.colour,
            c.size.0,
            c.size.1,
            c.depth,
            if c.qualifies { "yes" } else { "no" },
            c.streak
        );
    }
    if let Some((frame, (rw, rh))) = d.start() {
        WORLD_START.store(frame, std::sync::atomic::Ordering::Relaxed);
        xnw_core::info!("capture: world starts at frame {frame} (render size {rw}x{rh}; recorded while {frame} frames were presented, shown at present {}; decided at frame {})", frame + 2, c.tag);
    }
}

/// The recording frame the world starts in, once known.
pub(crate) fn world_start() -> Option<u64> {
    let w = WORLD_START.load(std::sync::atomic::Ordering::Relaxed);
    (w != u64::MAX).then_some(w)
}

/// Whether presented frame `frame` is one of the first 40 world frames
/// saved under `XNW_CAPTURE_TARGETS=world+...`.
pub(crate) fn world_frame_wanted(frame: u64) -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_CAPTURE_TARGETS").is_ok_and(|v| v.trim().starts_with("world+")));
    *ON && folder().is_some() && world_start().is_some_and(|w| (w + 2..w + 42).contains(&frame))
}

/// `XNW_DEPTH_LOG`: whether recording frame `tag` is logged: the first 3
/// world frames, then every 64th.
pub(crate) fn depth_log_frame(tag: u64) -> bool {
    world_start().is_some_and(|w| tag >= w && (tag < w + 3 || tag % 64 == 0))
}

#[cfg(test)]
mod world_tests {
    use super::WorldDetector;

    /// A frame's tally: (size, colour draws, of those depth-writing), plus
    /// draws without a colour target.
    type Frame<'a> = (&'a [((u32, u32), u32, u32)], u32);

    /// Feeds frames from tag 100; returns the start.
    fn run(frames: &[Frame], output: (u32, u32)) -> Option<(u64, (u32, u32))> {
        let mut d = WorldDetector::default();
        for (i, (tally, other)) in frames.iter().enumerate() {
            let tag = i as u64 + 100;
            for &(size, n, depth) in *tally {
                for k in 0..n {
                    d.note(tag, size, k < depth, output);
                }
            }
            for _ in 0..*other {
                d.note(tag, (0, 0), false, output);
            }
            d.note(tag, (0, 0), false, output);
        }
        d.note(10_000, (0, 0), false, output);
        d.start()
    }

    const OUT: (u32, u32) = (1920, 1080);
    const UI: Frame = (&[((1920, 1080), 9, 0)], 0);
    const WORLD: Frame = (&[((640, 360), 1200, 900), ((1920, 1080), 30, 0), ((128, 128), 400, 400)], 1300);
    /// The run-38b card frame: 50+ render-size draws, few draws overall.
    const CARD: Frame = (&[((640, 360), 60, 30), ((1920, 1080), 20, 0)], 100);

    #[test]
    fn loading_screens_and_the_card_are_not_the_world() {
        assert_eq!(run(&[UI, UI, CARD, CARD, CARD, UI], OUT), None);
    }

    #[test]
    fn the_world_starts_at_the_first_of_three_world_frames() {
        assert_eq!(run(&[UI, CARD, UI, WORLD, WORLD, WORLD, WORLD], OUT), Some((103, (640, 360))));
    }

    #[test]
    fn a_broken_streak_starts_again() {
        assert_eq!(run(&[WORLD, WORLD, UI, WORLD, WORLD, WORLD], OUT), Some((103, (640, 360))));
        assert_eq!(run(&[WORLD, WORLD, UI, WORLD, WORLD], OUT), None);
    }

    #[test]
    fn world_frames_without_depth_writes_or_output_do_not_count() {
        let flat: Frame = (&[((640, 360), 1200, 0)], 1300);
        assert_eq!(run(&[flat, flat, flat, flat], OUT), None);
        assert_eq!(run(&[WORLD, WORLD, WORLD], (0, 0)), None);
        let shadows: Frame = (&[((128, 128), 2000, 2000), ((240, 135), 600, 600)], 0);
        assert_eq!(run(&[shadows, shadows, shadows], OUT), None);
    }
}

// ---- Culling dump ----
//
// `XNW_CULL_DUMP=<k>`: in the k-th recorded frame after the world start,
// the first 8 compute dispatches whose program name contains "Cull": their
// root constants, every buffer they bind (root views, table views) as the
// CPU held it when the dispatch was recorded and as it is after the frame
// ran (title memory; under `XNW_SHARED=2` that is the GPU's own memory), an
// indirect dispatch's arguments, and the textures they read (each mip's
// stats, mip 0 as PNG: the depth pyramid). Once per dump: the title's
// culling globals through the title base (args ring 0x1484062F0,
// visible lists 0x1484062D0, pyramid 0x148769F58). Each dispatch ends with
// a verdict line by the culling rule.

/// `XNW_CULL_DUMP`'s k.
fn cull_k() -> Option<u64> {
    static K: std::sync::LazyLock<Option<u64>> = std::sync::LazyLock::new(|| std::env::var("XNW_CULL_DUMP").ok().and_then(|v| v.trim().parse().ok()));
    *K
}

/// Whether dispatches recorded now are dumped.
pub(crate) fn cull_dump_frame() -> bool {
    let Some(k) = cull_k() else { return false };
    folder().is_some() && world_start().is_some_and(|w| super::queue::presented() == w + k.max(1))
}

/// One dumped range: recorded bytes now, executed bytes read later.
struct CullBytes {
    tag: u64,
    group: u32,
    label: String,
    address: u64,
    recorded: Vec<u8>,
}

static CULL: std::sync::Mutex<Vec<CullBytes>> = std::sync::Mutex::new(Vec::new());
static CULL_GROUPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Starts the dump of one culling dispatch: its group number, or `None`
/// past the first 8.
pub(crate) fn cull_begin(name: &str, how: &str, constants: &[(u8, Vec<u32>)]) -> Option<u32> {
    let g = CULL_GROUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if g >= 8 {
        return None;
    }
    if g == 0 {
        cull_globals();
    }
    let consts: Vec<String> = constants.iter().map(|(slot, c)| format!("slot {slot}: {}", hex_words(c))).collect();
    xnw_core::info!("cull #{g}: {name} ({how}) frame {}; root constants {}", super::queue::presented(), consts.join("; "));
    Some(g)
}

fn hex_words(w: &[u32]) -> String {
    w.iter().map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(" ")
}

fn hex_bytes(b: &[u8]) -> String {
    let words: Vec<u32> = b.chunks(4).map(|c| {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        u32::from_le_bytes(w)
    }).collect();
    hex_words(&words)
}

/// Reads `len` bytes of title memory at `address` (a range a view of the
/// dispatch names, so mapped).
fn title_bytes(address: u64, len: usize) -> Vec<u8> {
    if address == 0 || len == 0 {
        return Vec::new();
    }
    // Committed and readable all along, else nothing (a GPU address that is
    // not title memory).
    let end = address + len as u64;
    let mut p = address;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: xnw_core::win::MemoryBasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { xnw_core::win::VirtualQueryEx(xnw_core::win::GetCurrentProcess(), p as *const _, &mut info, size_of_val(&info)) };
        // PAGE_NOACCESS 0x01, PAGE_GUARD 0x100.
        if n == 0 || info.state != xnw_core::win::MEM_COMMIT || info.protect & 0x101 != 0 {
            return Vec::new();
        }
        p = (info.base + info.region_size) as u64;
    }
    // SAFETY: committed, readable title memory; read only.
    unsafe { std::slice::from_raw_parts(address as *const u8, len) }.to_vec()
}

/// A buffer range of dispatch `group`: kept as recorded, read again after
/// the frame ran.
pub(crate) fn cull_bytes(group: u32, label: &str, address: u64, len: u64) {
    let len = len.min(576) as usize;
    let recorded = title_bytes(address, len);
    CULL.lock().unwrap_or_else(|e| e.into_inner()).push(CullBytes { tag: super::queue::presented(), group, label: label.to_owned(), address, recorded });
}

/// A texture dispatch `group` reads: every mip's stats, mip 0 as PNG.
pub(crate) fn cull_texture(list: &super::host::HostList, group: u32, this: super::object::This, texture: &ID3D12Resource, label: &str) {
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count > 1 {
        return;
    }
    const COPY_SOURCE: u32 = 0x800;
    super::list::ensure_state(list, this, COPY_SOURCE);
    let Some(device) = super::host::host_device() else { return };
    let frame = super::queue::presented();
    for mip in 0..u32::from(desc.MipLevels).min(14) {
        let Some((buffer, footprint)) = read_back_sub(&device.device, &list.list, texture, mip) else { continue };
        let name = format!("cull-{frame}-{group}-{label}-mip{mip}-f{}-{}x{}", desc.Format.0, footprint.Footprint.Width, footprint.Footprint.Height);
        SNAPSHOTS.lock().unwrap_or_else(|e| e.into_inner()).push(Snapshot { frame, pass: format!("cull #{group}"), name, buffer, footprint });
    }
}

/// The title's culling globals, read through the title base.
fn cull_globals() {
    let base = super::object::title_base() as u64;
    if base == 0 {
        return;
    }
    let at = |va: u64| base + (va - 0x1_4000_0000);
    // SAFETY: the title image's data section; read only.
    let q = |va: u64| unsafe { (at(va) as *const u64).read_unaligned() };
    let ring: Vec<String> = (0..3).map(|i| format!("{:#x}", q(0x1_4840_62F0 + 8 * i))).collect();
    let lists: Vec<String> = (0..2).map(|i| format!("{:#x}", q(0x1_4840_62D0 + 8 * i))).collect();
    xnw_core::info!(
        "cull globals: args ring 2F0[0..3] {}; visible lists 2D0[0..2] {}; pyramid F58 {:#x}, F60 {:#x}",
        ring.join(" "),
        lists.join(" "),
        q(0x1_4876_9F58),
        q(0x1_4876_9F60)
    );
}

/// Logs the dumped ranges whose frame has run, and each dispatch's verdict.
fn cull_drain(now: u64) {
    let ready: Vec<CullBytes> = {
        let mut all = CULL.lock().unwrap_or_else(|e| e.into_inner());
        let (ready, keep) = std::mem::take(&mut *all).into_iter().partition(|c| c.tag + 3 <= now);
        *all = keep;
        ready
    };
    if ready.is_empty() {
        return;
    }
    let mut groups: Vec<u32> = ready.iter().map(|c| c.group).collect();
    groups.dedup();
    for g in groups {
        let items: Vec<&CullBytes> = ready.iter().filter(|c| c.group == g).collect();
        let mut executed_all = Vec::new();
        for c in &items {
            let executed = title_bytes(c.address, c.recorded.len());
            xnw_core::info!("cull #{g} {} {:#x}+{}: recorded {}", c.label, c.address, c.recorded.len(), hex_bytes(&c.recorded[..c.recorded.len().min(128)]));
            xnw_core::info!("cull #{g} {} {:#x}+{}: executed {}", c.label, c.address, c.recorded.len(), hex_bytes(&executed[..executed.len().min(128)]));
            executed_all.push((c.label.as_str(), executed));
        }
        let zero = |b: &[u8]| b.iter().all(|&x| x == 0);
        let block = executed_all.iter().filter(|(l, b)| l.starts_with("cbv") && b.len() >= 16).map(|(_, b)| b).next();
        let outputs: Vec<&Vec<u8>> = executed_all.iter().filter(|(l, _)| l.starts_with("uav")).map(|(_, b)| b).collect();
        let jobs = executed_all.iter().find(|(l, _)| *l == "args").and_then(|(_, b)| b.get(..4)).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let verdict = match block {
            Some(b) if zero(b) => "block zero at execution",
            _ if !outputs.is_empty() && outputs.iter().any(|b| !zero(b)) => "non-zero",
            _ if jobs.is_some_and(|j| j > 0) => "block fine, args zero",
            _ => "plausible, zero",
        };
        if let Some(b) = block {
            let w = |i: usize| b.get(i..i + 4).map_or(0, |x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]));
            xnw_core::info!("cull #{g} block fields (CB = block+96): job list {:#x}, matrices {:#x}, meshlet list {:#x}; plane count {}", w(0), w(4), w(8), w(452));
        }
        xnw_core::info!("cull #{g} verdict: {verdict} (indirect jobs {jobs:?}, {} outputs, block {})", outputs.len(), if block.is_some() { "bound" } else { "not seen" });
    }
}
