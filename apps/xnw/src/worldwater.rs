//! `XNW_WATER_SIM=probe|flat` (default off): the ocean wave textures the
//! title's GPU writes and its CPU samples for buoyancy (WAT1).
//!
//! The wave simulation's last compute pass (0x142E79D90 -> 0x142E66900 ->
//! 0x140294DB0, the external-target variant) writes a 256x256 R32_FLOAT
//! height map per cascade into a ring of 5 slots in the water manager
//! (.bss 0x14A4DC080 + 0x4413D0 + 240*cascade + 48*slot), made by
//! 0x1458AC450 -> 0x141BC5950 [V]. The record keeps {texture +16, CPU view
//! +24, bytes per texel +32, width +36, height +40} (0x141FE2E30, the CPU view
//! from addr_gpu_to_cpu 0x144FD2940) [V]. The water query (0x144F413E0, 138
//! call sites) copies the record pointers (0x144F41A30), builds samplers on
//! the CPU views (0x144F41850) and reads them with wrap-around bilinear taps
//! (0x1451DCFF0 -> 0x1451DE620, 33 sites) [V]. The ring's readiness is CPU
//! frame counting (0x14084A8C0, 0x14084A930), no fence [V]. With
//! `XNW_GPU=null` nothing writes those texels: the CPU reads whatever the
//! allocation held [V reads; content U, this switch's `probe` shows it].
//! The second ring (0x1476B4250.., R32 heights + R32G32 displacements, 5
//! slots) is zeroed by the title at creation (0x145EBEA10) [V].
//!
//! `probe`: one line per ring texture on the first readable frame and every
//! 600 frames after: zero, non-finite and |h| > 100 m texel counts, min, max.
//! `flat`: as probe, and a cascade texture holding a non-zero texel is
//! zero-filled: a flat sea at the base height the streamed height map and
//! the DeepOcean keyframes give (FFT heights are zero-mean, I). Writing
//! title memory here goes through the write guards like a title store.

const IMAGE_BASE: usize = 0x1_4000_0000;
/// The water manager (static object, .bss).
const WATER: usize = 0x1_4A4D_C080;
/// Its CPU-read cascade ring: 240 B per cascade, 48 B per slot.
const RING: usize = 0x44_13D0;
const SLOTS: usize = 5;
/// HIDWORD of qword_14A91D3A8: the last cascade index (0x141D95620 loop).
const LAST_CASCADE: usize = 0x1_4A91_D3AC;
/// The second ring: {width, height} ints, then 32-B records with the CPU
/// view at +0x10; heights 0x1476B4250.., displacements 0x1476B42F0...
const RING2_SIZE: usize = 0x1_476B_41D8;
const RING2_HEIGHTS: usize = 0x1_476B_4250;
const RING2_DISPS: usize = 0x1_476B_42F0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Off,
    Probe,
    Flat,
}

fn mode() -> Mode {
    static M: std::sync::LazyLock<Mode> = std::sync::LazyLock::new(|| match std::env::var("XNW_WATER_SIM").as_deref().map(str::trim) {
        Ok("probe") => Mode::Probe,
        Ok("flat") => Mode::Flat,
        _ => Mode::Off,
    });
    *M
}

/// Texel counts of an f32 texture.
#[derive(Default, Debug, PartialEq)]
pub(crate) struct Stats {
    pub(crate) texels: usize,
    pub(crate) zero: usize,
    pub(crate) nonfinite: usize,
    pub(crate) big: usize,
    pub(crate) min: f32,
    pub(crate) max: f32,
}

pub(crate) fn stats(b: &[u8]) -> Stats {
    let mut s = Stats { min: f32::INFINITY, max: f32::NEG_INFINITY, ..Default::default() };
    for c in b.chunks_exact(4) {
        let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        s.texels += 1;
        if v == 0.0 {
            s.zero += 1;
        }
        if !v.is_finite() {
            s.nonfinite += 1;
            continue;
        }
        if v.abs() > 100.0 {
            s.big += 1;
        }
        s.min = s.min.min(v);
        s.max = s.max.max(v);
    }
    s
}

fn read(addr: u64, len: usize) -> Option<Vec<u8>> {
    if !(0x1_0000..0x8000_0000_0000).contains(&addr) || len == 0 || len > 1 << 24 {
        return None;
    }
    let mut b = vec![0u8; len];
    crate::watch::guarded_read(addr as usize, &mut b).then_some(b)
}

fn le(b: &[u8], at: usize, n: usize) -> u64 {
    b.get(at..at + n).map_or(0, |s| s.iter().rev().fold(0u64, |v, x| v << 8 | u64::from(*x)))
}

/// One ring texture: (name, CPU view, bytes).
fn textures(title: usize) -> Vec<(String, u64, usize)> {
    let real = |va: usize| (title + (va - IMAGE_BASE)) as u64;
    let mut out = Vec::new();
    let last = read(real(LAST_CASCADE), 4).map_or(0, |b| le(&b, 0, 4) as usize).min(2);
    for c in 0..=last {
        for s in 0..SLOTS {
            let Some(r) = read(real(WATER + RING + 240 * c + 48 * s), 48) else { continue };
            let (ptr, bpp, w, h) = (le(&r, 24, 8), le(&r, 32, 4) as usize, le(&r, 36, 4) as usize, le(&r, 40, 4) as usize);
            // The ring's own shape (0x141BC5950: 256x256, format 41).
            if ptr != 0 && bpp == 4 && w == 256 && h == 256 {
                out.push((format!("cascade{c}.slot{s}"), ptr, w * h * bpp));
            }
        }
    }
    if let Some(size) = read(real(RING2_SIZE), 8) {
        let (w, h) = (le(&size, 0, 4) as usize, le(&size, 4, 4) as usize);
        if (1..=1024).contains(&w) && (1..=1024).contains(&h) {
            for (name, base, bpp) in [("height", RING2_HEIGHTS, 4), ("disp", RING2_DISPS, 8)] {
                for s in 0..SLOTS {
                    if let Some(r) = read(real(base + 32 * s + 0x10), 8)
                        && le(&r, 0, 8) != 0
                    {
                        out.push((format!("ring2.{name}{s}"), le(&r, 0, 8), w * h * bpp));
                    }
                }
            }
        }
    }
    out
}

/// Once a frame: from the writer thread (worldprobe's render_frame), or
/// without the render path from the first hook draw of the frame.
pub(crate) fn tick(title: usize, frame: u64) {
    let m = mode();
    if m == Mode::Off {
        return;
    }
    static FIRST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
    let first = FIRST.load(std::sync::atomic::Ordering::Relaxed);
    if first != u64::MAX && !frame.wrapping_sub(first).is_multiple_of(600) {
        return;
    }
    let tex = textures(title);
    if tex.is_empty() {
        return;
    }
    if first == u64::MAX {
        FIRST.store(frame, std::sync::atomic::Ordering::Relaxed);
    }
    for (name, ptr, len) in tex {
        let Some(b) = read(ptr, len) else {
            xnw_core::info!("water sim: frame {frame} {name} {ptr:#x}: unreadable");
            continue;
        };
        let s = stats(&b);
        let fill = m == Mode::Flat && name.starts_with("cascade") && s.zero < s.texels;
        xnw_core::info!(
            "water sim: frame {frame} {name} {ptr:#x}: {} texels, {} zero, {} non-finite, {} beyond 100 m, min {} max {}{}",
            s.texels, s.zero, s.nonfinite, s.big, s.min, s.max, if fill { ": zero-filled (XNW_WATER_SIM=flat)" } else { "" }
        );
        if fill {
            // SAFETY: the whole range was just read, so it is mapped title
            // memory (a texture the title keeps for its lifetime).
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, len) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_count_texels() {
        let v: Vec<u8> = [0.0f32, 1.5, -250.0, f32::NAN, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let s = stats(&v);
        assert_eq!((s.texels, s.zero, s.nonfinite, s.big, s.min, s.max), (5, 2, 1, 1, -250.0, 1.5));
    }
}
