//! The lighting's constant tables made by the title's own draws (LIT4):
//! `world-render <frame> --lighting auto --luts <frame with the LUT draws>`.
//!
//! The title draws its BRDF and area-light tables once (pk7d f1059 sinks
//! 0-5: `VS_screenTransformS_e4d137db` + one pixel program each, 3
//! vertices, one target; none in f1060-1067 [V pack sinks.tsv]). Each runs
//! here through the bridge over its captured bindings, at the size of the
//! view the lighting program reads, and its target 0 goes in that slot in
//! the table's own format. Which draw fills which slot is by target format
//! and name [I]: the only R16G16_UNORM draw for slot 100 (127x128
//! R16G16_UNORM), the R16G16B16A16_FLOAT `PS_IntegrateEnvBRDF` for slot 132
//! (256x256).

use std::collections::HashMap;
use std::path::Path;

use xnw_shader::emit;
use xnw_world::bridge;
use xnw_world::textures::Provided;

use crate::bridge::{Row, load, named};
use crate::pack::Pack;

/// One table: name, the pixel program's container hash, the lighting
/// program's frame-table slot, its DXGI format, its size where the
/// lighting sink names no view there (pk7a: 96 slots; pk7d f1059's TL
/// sink 7497 views [V]).
pub struct Lut {
    pub name: &'static str,
    pub pixel: u128,
    pub slot: u32,
    pub format: u32,
    pub size: (u32, u32),
}

const R16G16_UNORM: u32 = 35;
const R16G16B16A16_FLOAT: u32 = 10;

pub const LUTS: [Lut; 2] = [
    Lut { name: "PS_AreaLightIlluminance", pixel: 0x901b_5b70_0116_8b4c_5a7d_3a14_a1c1_2c30, slot: 100, format: R16G16_UNORM, size: (127, 128) },
    Lut { name: "PS_IntegrateEnvBRDF", pixel: 0xdd22_2c09_efca_e37b_cafa_859b_d2f2_8342, slot: 132, format: R16G16B16A16_FLOAT, size: (256, 256) },
];

/// f32 to IEEE half (round to nearest even), as the bits.
pub fn half(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half_m = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let r = if rest > mid || (rest == mid && half_m & 1 == 1) { half_m + 1 } else { half_m };
        return sign | r as u16;
    }
    let h = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1FFF;
    let r = if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) { h + 1 } else { h };
    sign | r as u16
}

/// Target 0 rows (float RGBA) as texels of `format` (R16G16_UNORM or RGBA16F).
pub fn texels(px: &[[f32; 4]], w: u32, h: u32, format: u32) -> Provided {
    let (bpp, rows): (u32, Vec<u8>) = if format == R16G16_UNORM {
        (4, px.iter().flat_map(|p| [p[0], p[1]].map(|c| (c.clamp(0.0, 1.0) * 65535.0).round() as u16).into_iter().flat_map(u16::to_le_bytes)).collect())
    } else {
        (8, px.iter().flat_map(|p| p.map(half).into_iter().flat_map(u16::to_le_bytes)).collect())
    };
    Provided { width: w, height: h, format, rows, row_bytes: w * bpp, row_count: h, dim: xnw_world::textures::TEX_2D, depth: 1, mips: 1 }
}

/// Sink `sink` (a plain draw of `vertices`) through its own vertex and
/// pixel programs over its captured bindings, at `w` x `h`: target 0. With
/// `held`, a texture the compute runner holds at a view's address
/// stands in for the captured one: what an earlier dispatch wrote.
pub fn run_draw(pack: &Pack, sink: u32, vertices: u32, w: u32, h: u32, host: &xnw_gpu::host::Host, held: Option<&mut xnw_world::compute::Runner>) -> Result<(Vec<[f32; 4]>, f64, Vec<String>), String> {
    let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).ok_or("no such sink")?.pipeline;
    let stage = |name: &str| pack.data.programs.iter().find(|p| p.pipeline == pipeline && p.stage == name).map(|p| p.container).ok_or_else(|| format!("sink {sink}: no {name} program"));
    let code = |h: u128| pack.blob(h).ok_or_else(|| format!("no container bytes for {h:032x}"));
    let (vsc, psc) = (code(stage("vertex")?)?, code(stage("pixel")?)?);
    let vs = emit::translate_with(&vsc, bridge::options()).map_err(|e| format!("sink {sink} vertex program: {e}"))?;
    let ps = emit::translate_with(&psc, bridge::options()).map_err(|e| format!("sink {sink} pixel program: {e}"))?;
    let (mut used, mut wave) = (HashMap::new(), 32);
    for c in [&vsc, &psc] {
        named(c, &mut used, &mut wave)?;
    }
    let layout = emit::mesh::Layout { max_vertices: 0, max_prims: 0, rows: 1, elements: Vec::new() };
    let prog = bridge::Programs { ms: vs, layout, vs: Vec::new(), ps, wave, vertex: Some((vertices, 1, false)) };
    let counts = std::sync::Arc::new([const { std::sync::atomic::AtomicU64::new(0) }; 3]);
    let mut provider = pack.texel_provider(counts);
    let mut loaded = load(pack, sink, &Row::default(), None, &used, &mut provider);
    let mut lacks: Vec<String> = loaded.lacks.iter().filter(|l| !l.starts_with("the group count") && !l.starts_with("the targets") && !l.starts_with("sampler heap")).cloned().collect();
    if let Some(runner) = held {
        let (held, why) = crate::compute::held_textures(pack, sink, runner);
        lacks.extend(why);
        for (root, slot, t) in held {
            lacks.push(format!("root {root} slot {slot}: the runner's {}x{} format {} (an earlier dispatch's output)", t.width, t.height, t.format));
            loaded.set_texture(root, slot, t);
        }
        for (root, slot, kind, bytes) in crate::compute::held_buffers(pack, sink, runner, &used) {
            lacks.push(format!("root {root} slot {slot}: the runner's {} B buffer (an earlier dispatch's output or a held stand-in)", bytes.len()));
            loaded.set_buffer(root, slot, kind, bytes);
        }
    }
    let d = loaded.draw();
    let b = bridge::bind(&prog.ms, &prog.ps, &d)?;
    if let Some(gap) = bridge::gap(&host.caps, &prog.needs()) {
        return Err(format!("not run on {}: {gap}", host.caps.adapter));
    }
    let target = bridge::Target { width: w, height: h, depth: None, limit: if host.caps.software { 600.0 } else { 1.0 }, depth_only: false, records: false };
    crate::replay::watchdog_arm();
    let f = bridge::run(host, &prog, &b, &target);
    crate::replay::watchdog_disarm();
    let f = f?;
    Ok((f.rt0, f.seconds, lacks))
}

/// Every table of [`LUTS`] whose draw `luts` (a capture frame) holds, run at
/// the size of the lighting sink's view `size(slot)`: (slot, texels, target
/// 0, width, height). A picture of each (red, green) goes beside `out`.
pub type Made = (u32, Provided, Vec<[f32; 4]>, u32, u32);

pub fn produce(luts: &Path, size: &dyn Fn(u32) -> Option<(u32, u32)>, host: &xnw_gpu::host::Host, out: &Path) -> Result<Vec<Made>, String> {
    let pack = Pack::open(luts)?.ok_or_else(|| format!("{}: no pack", luts.display()))?;
    let mut made = Vec::new();
    for l in &LUTS {
        let pipes: std::collections::HashSet<u64> = pack.data.programs.iter().filter(|p| p.stage == "pixel" && p.container == l.pixel).map(|p| p.pipeline).collect();
        let Some(s) = pack.data.sinks.iter().find(|s| pipes.contains(&s.pipeline)) else {
            println!("  table {} (slot {}): no draw of {:08x} in {}", l.name, l.slot, (l.pixel >> 96) as u32, luts.display());
            continue;
        };
        let (w, h) = size(l.slot).unwrap_or(l.size);
        let vertices = s.count.max(3);
        let (px, secs, lacks) = run_draw(&pack, s.index, vertices, w, h, host, None)?;
        let untouched = px.iter().filter(|p| **p == bridge::CLEAR).count();
        let bad = px.iter().filter(|p| p.iter().any(|c| !c.is_finite())).count();
        let range = |k: usize| px.iter().filter(|p| **p != bridge::CLEAR).map(|p| p[k]).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
        let (r, g) = (range(0), range(1));
        println!("  table {} (slot {}): sink {} ({} vertices) at {w}x{h} in {secs:.2} s: untouched {untouched}, not finite {bad}; r [{:.4}, {:.4}], g [{:.4}, {:.4}]", l.name, l.slot, s.index, vertices, r.0, r.1, g.0, g.1);
        for x in lacks {
            println!("    lacks: {x}");
        }
        let pic: Vec<u8> = px.iter().flat_map(|p| [p[0], p[1], 0.0].map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8).into_iter().chain([255])).collect();
        let png = out.with_extension(format!("lut{}.png", l.slot));
        crate::replay::write_png(&png, w, h, &pic).map_err(|e| e.to_string())?;
        println!("    picture {} (red = .r, green = .g)", png.display());
        made.push((l.slot, texels(&px, w, h, l.format), px, w, h));
    }
    Ok(made)
}

/// `--run-draw <sink>=<w>x<h>[,...]` (a probe): any captured plain draw of
/// the frame through its own programs at that size; target 0's range and a
/// picture (rgb, clamped) beside `out`.
pub fn probe_draws(dir: &Path, spec: &str, host: &xnw_gpu::host::Host, out: &Path, mut held: Option<&mut xnw_world::compute::Runner>) -> Result<(), String> {
    let pack = Pack::open(dir)?.ok_or("--run-draw needs a v2 capture (pack)")?;
    for item in spec.split(',').filter(|x| !x.is_empty()) {
        let parse = || -> Option<(u32, u32, u32)> {
            let (k, wh) = item.split_once('=')?;
            let (w, h) = wh.split_once('x')?;
            Some((k.parse().ok()?, w.parse().ok()?, h.parse().ok()?))
        };
        let (sink, w, h) = parse().ok_or_else(|| format!("--run-draw {item}: want sink=WxH"))?;
        let count = pack.data.sinks.iter().find(|s| s.index == sink).map_or(3, |s| s.count.max(3));
        let (px, secs, lacks) = run_draw(&pack, sink, count, w, h, host, held.as_deref_mut())?;
        let lit: Vec<&[f32; 4]> = px.iter().filter(|p| **p != bridge::CLEAR && p.iter().all(|c| c.is_finite())).collect();
        let range = |k: usize| lit.iter().map(|p| p[k]).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
        let nz = lit.iter().filter(|p| p[0] != 0.0 || p[1] != 0.0 || p[2] != 0.0).count();
        println!("  draw sink {sink} ({count} vertices) at {w}x{h} in {secs:.2} s: written {}, rgb not zero {nz}; r {:?} g {:?} b {:?} a {:?}", lit.len(), range(0), range(1), range(2), range(3));
        for x in lacks {
            println!("    lacks: {x}");
        }
        let pic: Vec<u8> = px.iter().flat_map(|p| [p[0], p[1], p[2]].map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8).into_iter().chain([255])).collect();
        let png = out.with_extension(format!("sink{sink}.png"));
        crate::replay::write_png(&png, w, h, &pic).map_err(|e| e.to_string())?;
        println!("    picture {}", png.display());
    }
    Ok(())
}

/// Slot 100's falloff as the lighting program reads it (LIT3: .r at u =
/// 1/2 + sign(N.L) sqrt|N.L| / 2 on the last row), against Lambert.
pub fn falloff(px: &[[f32; 4]], w: u32, h: u32) -> String {
    let row = (h - 1) as usize * w as usize;
    [-0.5f32, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        .iter()
        .map(|&d| {
            let u = 0.5 + d.signum() * d.abs().sqrt() / 2.0;
            let x = ((u * w as f32 - 0.5).round() as i64).clamp(0, i64::from(w) - 1) as usize;
            format!("N.L {d}: {:.3} (Lambert {:.3})", px[row + x][0], d.max(0.0))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    #[test]
    fn half_rounds_like_ieee() {
        assert_eq!(super::half(1.0), 0x3C00);
        assert_eq!(super::half(-2.0), 0xC000);
        assert_eq!(super::half(0.5), 0x3800);
        assert_eq!(super::half(65504.0), 0x7BFF);
        assert_eq!(super::half(1e6), 0x7C00);
        assert_eq!(super::half(0.0), 0);
        // The smallest subnormal half, 2^-24.
        assert_eq!(super::half(5.960_464_5e-8), 1);
        assert_eq!(super::half(f32::NAN) & 0x7C00, 0x7C00);
    }

    #[test]
    fn texels_pack_the_formats() {
        let px = [[1.0, 0.5, 0.25, 2.0]];
        let t = super::texels(&px, 1, 1, super::R16G16_UNORM);
        assert_eq!(t.rows, vec![0xFF, 0xFF, 0x00, 0x80]);
        assert_eq!(t.row_bytes, 4);
        let t = super::texels(&px, 1, 1, super::R16G16B16A16_FLOAT);
        assert_eq!(t.rows, vec![0x00, 0x3C, 0x00, 0x38, 0x00, 0x34, 0x00, 0x40]);
    }
}
