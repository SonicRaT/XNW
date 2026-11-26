//! `world-render <frame dir> --lighting <sink|auto> [--gbuf theirs|ours]
//! [--shadow captured] [--luts <frame>|here] [--sky-chain] [--out <sheet png>]` (LIT2): the title's tiled
//! lighting program over one full screen, on the software adapter.
//!
//! One draw of `VS_TiledLighting_FullScreen` cd20ed95 (three vertices from
//! the vertex index, no interpolants; from the shader pack, the capture does
//! not hold it) with `TL_PS_PBR_MATERIAL_STANDARD` 69aabd5d (the capture's
//! lighting sink) over that sink's captured bindings (`bridge::load`):
//! - `--gbuf theirs`: the G-buffer as the capture holds it (frame table
//!   slots 73 albedo, 94 normal, 54 depth: LIT1);
//! - `--gbuf ours`: our pass's G-buffer (`XNW_WORLD_RENDER=...,gbuf=on`,
//!   `xnw_world::gbuf`), at the lighting size (a nearest copy when the
//!   pass is drawn at another size).
//! The screen shadow mask (slot 95) is white unless `--shadow captured`
//! (no shadows: the title's mask is a GPU output no capture holds).
//! Target 0 (fp16 lit rgb, the albedo already multiplied in, + luminance)
//! is read back; the sheet is albedo | normal over lit | lit / albedo, both
//! with our exposure and Hable curve (`xnw_world::lighting`, from the
//! frame's timecycle). The sun's GPU-made inputs stand in (LIT3, see the
//! `--sun` block); `--ps-patch off=word,...` writes GFX10 words over the
//! pixel program before translation (the debug export: `v_mov` into
//! v100..v103 at an `s_waitcnt`, then the export patch's register word at
//! 0x4ecc names them).

use std::path::Path;

use xnw_shader::emit::{self, Options};
use xnw_world::bridge;
use xnw_world::lighting::{Lighting, display};
use xnw_world::textures::Provided;

use crate::bridge::{Row, load, named};
use crate::pack::Pack;

/// `TL_PS_PBR_MATERIAL_STANDARD_42ad1e50` (LIT1).
pub const TL_PS: u128 = 0x69aa_bd5d_5611_9314_605e_3951_c0a2_efd4;
/// `VS_TiledLighting_FullScreen_bd146321`.
pub const TL_VS_FULLSCREEN: u128 = 0xcd20_ed95_23ab_732d_d125_4690_359a_9af1;
/// Frame table (root 0) slots the lighting program reads the G-buffer at
/// (69aabd5d's prologue [V disasm]: 0x920 = slot 73 rgba albedo; 0x1240 =
/// slot 146 rgba, the packed normal decoded at 0x148..0x1e4 (LIT1 named
/// slot 94: that one is read as rgb at 0xf0, another target); 0xf80 = slot
/// 124 one channel, device depth (linearised at 0x39c with the view
/// record's near/far); 0x6c0 = slot 54 .g, the stencil's material bits).
pub const ALBEDO_SLOT: u32 = 73;
pub const NORMAL_SLOT: u32 = 146;
pub const DEPTH_SLOT: u32 = 124;
pub const STENCIL_SLOT: u32 = 54;
pub const SHADOW_SLOT: u32 = 95;
/// b5, the constant buffer whose +0x30 turns the sun's shadow term on.
pub const B5_SLOT: u32 = 178;
/// The 16-B buffer whose first two words scale the light (T8+0x2a0).
pub const LIGHT_SCALE_SLOT: u32 = 21;
/// Its stand-in (a GPU output the capture holds as zero). The writer is
/// `avgLuminanceCompute:ComputeLuxCompressionInfo` 64993692 (not sunk; a
/// direct dispatch): it stores (c, 1/c, k, 1/k), c the lux compression
/// (0x84..0xa0) and k its constant buffer +0x50 (0x54, 0x70..0x74, the
/// store at 0xac) [V disasm]. The lighting program reads c and 1/c; the
/// sky paraboloid draw multiplies its rgb by 1/k (68d222c8 0xf48, 0x11e8)
/// and AmbientCubeIBR multiplies the result back by k (9ae7b3a3 0x158,
/// 0x170) [V disasm]. LIT3's (1, 1, 1, 0) made 1/k zero, and with it the
/// whole sky draw (LIT6). Neutral: c = k = 1.
pub const LIGHT_SCALE_UNIT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
/// The sun's other GPU-made inputs (LIT3, `--ps-patch` probes): a scalar
/// buffer, a 1-D curve, a screen-space texture and a 2-D table.
pub const SUN_SCALE_SLOT: u32 = 29;
pub const SUN_CURVE_SLOT: u32 = 153;
pub const SUN_SCREEN_SLOT: u32 = 101;
pub const SUN_LUT_SLOT: u32 = 100;
/// DXGI formats of the textures we put there.
const R8G8B8A8_UNORM_SRGB: u32 = 29;
const R8G8B8A8_UNORM: u32 = 28;
const R32_FLOAT: u32 = 41;
const R8_UNORM: u32 = 61;

/// A nearest copy of `rows` (`bpp` bytes a pixel, `w` x `h`) at `tw` x `th`.
pub fn nearest(rows: &[u8], bpp: usize, w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(bpp * (tw * th) as usize);
    for y in 0..th {
        let sy = (u64::from(y) * u64::from(h) / u64::from(th)) as usize;
        for x in 0..tw {
            let sx = (u64::from(x) * u64::from(w) / u64::from(tw)) as usize;
            let at = bpp * (sy * w as usize + sx);
            out.extend_from_slice(&rows[at..at + bpp]);
        }
    }
    out
}

fn texture(width: u32, height: u32, format: u32, bpp: u32, rows: Vec<u8>) -> Provided {
    Provided { width, height, format, rows, row_bytes: width * bpp, row_count: height, dim: xnw_world::textures::TEX_2D, depth: 1, mips: 1 }
}

/// The stand-in for slot 100 (LIT3, I): the title reads its .r at
/// u = 1/2 + sign(d) sqrt(|d|) / 2 with d = N.L (0x1b88..0x1bb4) and
/// v = 1 (0x1c3c). Lambert undoes that warp: max(0, 2u - 1)^2 = max(0, N.L).
/// The title's own table (a GPU output, U) may add wrap or scattering.
pub fn lambert_table(w: u32, h: u32) -> Vec<u8> {
    let mut rows = Vec::with_capacity((4 * w * h) as usize);
    for _ in 0..h {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let v = ((2.0 * u - 1.0).max(0.0).powi(2) * 255.0).round() as u8;
            rows.extend_from_slice(&[v, v, v, 255]);
        }
    }
    rows
}

/// sRGB byte to linear.
fn linear(b: u8) -> f32 {
    let c = f32::from(b) / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// The debug export (LIT3): prints the pixel program's export patches
/// (the console's 32-bit target forms) and the words `--ps-patch` writes
/// over its code after them (`emit::translate_patched`).
pub fn show_patches(container: &[u8], words: &[(u32, u32)]) -> Result<(), String> {
    let c = xnw_shader::Container::parse(container).map_err(|e| format!("container: {e}"))?;
    let dxil = c.part(xnw_shader::FourCC::DXIL).ok_or("no DXIL part")?;
    let native = xnw_shader::native_programs(dxil.data).map_err(|e| format!("native: {e}"))?;
    let p = native.iter().find(|p| p.stage == xnw_shader::native::HwStage::Ps).ok_or("no pixel program")?;
    if let xnw_shader::native::StageIo::Ps { patches, .. } = &p.io {
        for x in patches {
            let w: Vec<String> = x.code.iter().map(|w| format!("{w:08x}")).collect();
            println!("  export patch mrt{} at 0x{:x}: {}", x.mrt, 4 * x.offset, w.join(" "));
        }
    }
    let code = p.patched_code();
    for &(off, word) in words {
        let old = code.get(off as usize..off as usize + 4).ok_or(format!("patch 0x{off:x}: outside the code ({} B)", code.len()))?;
        println!("  probe patch 0x{off:x}: {:08x} -> {word:08x}", u32::from_le_bytes([old[0], old[1], old[2], old[3]]));
    }
    Ok(())
}

/// `--ps-patch 0x4ec0=5e04170e,0x4ec4=...` (hex) as (offset, word) pairs.
fn parse_patch(v: &str) -> Result<Vec<(u32, u32)>, String> {
    let hex = |s: &str| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).map_err(|_| format!("--ps-patch {v}: {s} is not hex"));
    v.split(',').filter(|s| !s.is_empty()).map(|kv| kv.split_once('=').ok_or(format!("--ps-patch {kv}: want off=word")).and_then(|(k, w)| Ok((hex(k)?, hex(w)?)))).collect()
}

/// The sink of the frame's first draw with the standard lighting program.
fn lighting_sink(pack: &Pack) -> Option<u32> {
    let pipes: std::collections::HashSet<u64> = pack.data.programs.iter().filter(|p| p.stage == "pixel" && p.container == TL_PS).map(|p| p.pipeline).collect();
    pack.data.sinks.iter().find(|s| pipes.contains(&s.pipeline)).map(|s| s.index)
}

pub fn cli(args: &[String], dir: &Path, host: &xnw_gpu::host::Host, pass: &mut xnw_world::Pass) -> Result<(), String> {
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let pack = Pack::open(dir)?.ok_or("--lighting needs a v2 capture (pack)")?;
    let sink = match opt("--lighting").as_deref() {
        Some(v) if v != "auto" && !v.starts_with("--") => v.parse().map_err(|_| format!("--lighting {v}: not a sink number"))?,
        _ => lighting_sink(&pack).ok_or("no sink of the standard lighting program 69aabd5d in this capture")?,
    };
    let ours = opt("--gbuf").as_deref() == Some("ours");
    println!("lighting: {} sink {sink}, G-buffer {}", dir.display(), if ours { "ours" } else { "theirs (captured)" });

    // The two programs: the pixel program from the capture, the full-screen vertex program from the shader pack.
    let psc = pack.blob(TL_PS).ok_or("the capture holds no bytes of 69aabd5d")?;
    let words = opt("--ps-patch").map_or(Ok(Vec::new()), |v| parse_patch(&v))?;
    show_patches(&psc, &words)?;
    let rpf = xnw_gpu::d3d12x::pack::default_pack().ok_or("the full-screen vertex program comes from shaders.rpf: set XNW_GAME_DRIVE (or XNW_SHADER_PACK)")?;
    let vsc = xnw_world::preskin::container(&rpf, None, TL_VS_FULLSCREEN).ok_or_else(|| format!("{}: no container {TL_VS_FULLSCREEN:032x}", rpf.display()))?;
    let vs = emit::translate_with(&vsc, Options::default()).map_err(|e| format!("vertex program cd20ed95: {e}"))?;
    let ps = emit::translate_patched(&psc, Options::default(), &words).map_err(|e| format!("pixel program 69aabd5d: {e}"))?;
    let (mut used, mut wave) = (std::collections::HashMap::new(), 32);
    for c in [&vsc, &psc] {
        named(c, &mut used, &mut wave)?;
    }
    let layout = emit::mesh::Layout { max_vertices: 0, max_prims: 0, rows: 1, elements: Vec::new() };
    let prog = bridge::Programs { ms: vs, layout, vs: Vec::new(), ps, wave, vertex: Some((3, 1, false)) };

    let counts = std::sync::Arc::new([const { std::sync::atomic::AtomicU64::new(0) }; 3]);
    let mut texels = pack.texel_provider(counts);
    let mut loaded = load(&pack, sink, &Row::default(), None, &used, &mut texels);

    // The lighting size: the captured albedo view's (640 x 360 in pk7a).
    let tables = pack.tables(sink);
    let view = |root: u16, slot: u32| tables.iter().find(|t| t.0 == root).and_then(|t| t.2.iter().find(|v| v.slot == slot).cloned());
    let (w, h) = view(0, ALBEDO_SLOT).map_or((640, 360), |v| (v.width as u32, v.height));
    println!("  lighting size {w}x{h}; named slots (root/slot: view kind format size -> bound):");
    let mut names: Vec<(&(u16, u32), &u32)> = used.iter().collect();
    names.sort();
    let mut line = Vec::new();
    for (&(root, slot), _) in names {
        let v = view(root, slot).map_or("not in the table".into(), |v| format!("kind {} fmt {} {}x{}", v.kind, v.format, v.width, v.height));
        line.push(format!("{root}/{slot}: {v} -> {}", loaded.describe(root, slot)));
    }
    for l in line.chunks(3) {
        println!("    {}", l.join(" | "));
    }
    // The directional record (root 1 slot 0, 340 B): direction +0x110, colour x intensity +0x12C, intensity +0x138, tint +0x148.
    if let Ok((_, b)) = pack.slot_bytes(sink, 1, 0) {
        let f = |o: usize| b.get(o..o + 4).map_or(f32::NAN, |x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]));
        println!("  directional record (captured, used as is): dir ({:.3}, {:.3}, {:.3}), colour x I ({:.3}, {:.3}, {:.3}), I {:.3}, tint {:.3}", f(0x110), f(0x114), f(0x118), f(0x12C), f(0x130), f(0x134), f(0x138), f(0x148));
    }

    // Our G-buffer: the pass drawn with gbuf=on, read back, put at the three slots.
    let mut our_albedo = None;
    if ours {
        if !xnw_world::gbuf::on() {
            return Err("--gbuf ours: draw with XNW_WORLD_RENDER=...,gbuf=on (the pass makes its targets at start)".into());
        }
        let png = opt("--out").map_or_else(|| dir.join("lighting-ours.png"), |o| Path::new(&o).with_extension("pass.png"));
        crate::replay::render_one(args, dir, &png, pass, &host.caps.adapter, 0)?;
        let g = pass.read_gbuf().ok_or("the pass has no G-buffer")??;
        let scale = |rows: &[u8], bpp: usize| if (g.width, g.height) == (w, h) { rows.to_vec() } else { nearest(rows, bpp, g.width, g.height, w, h) };
        let depth: Vec<u8> = g.depth.iter().flat_map(|z| z.to_le_bytes()).collect();
        let (a, n, z) = (scale(&g.albedo, 4), scale(&g.normal, 4), scale(&depth, 4));
        let drawn = g.depth.iter().filter(|z| **z > 0.0).count();
        println!("  our G-buffer {}x{} -> {w}x{h}: {drawn} of {} pixels drawn (depth > 0)", g.width, g.height, g.depth.len());
        loaded.set_texture(0, ALBEDO_SLOT, texture(w, h, R8G8B8A8_UNORM_SRGB, 4, a.clone()));
        loaded.set_texture(0, NORMAL_SLOT, texture(w, h, R8G8B8A8_UNORM, 4, n.clone()));
        loaded.set_texture(0, DEPTH_SLOT, texture(w, h, R32_FLOAT, 4, z.clone()));
        // Stencil (.g of a depth-stencil view): an R32 view reads 0 there (material bits 0).
        loaded.set_texture(0, STENCIL_SLOT, texture(w, h, R32_FLOAT, 4, z));
        our_albedo = Some((a, n));
    }
    // Frame table slot 21 (16 B, T8+0x2a0): dwords 0 and 1 scale the light
    // (69aabd5d 0x4748: s10, s11; s11 multiplies the exported rgb at 0x4dcc..0x4e18).
    // A GPU output (exposure, I) the capture holds as zero, so every pixel is
    // black: 1.0 and 1.0 stand in (neutral) unless `--slot21 captured`.
    if opt("--slot21").as_deref() != Some("captured") {
        let unit: Vec<u8> = LIGHT_SCALE_UNIT.iter().flat_map(|v| v.to_le_bytes()).collect();
        loaded.set_buffer(0, LIGHT_SCALE_SLOT, xnw_world::preskin::Kind::Structured(16), unit);
        println!("  slot {LIGHT_SCALE_SLOT} (light scale, captured zero): {LIGHT_SCALE_UNIT:?} stands in");
    }
    // b5 (frame table slot 178, past the capture's 96 slots: 256 zero bytes
    // stand in): +0x30 == 0 skips the shadow mask at 0xa50 and leaves the
    // sun's visibility v53 at 0, and the sun block (0x19e0: v53 > 0) never
    // runs. +0x30 = 1 stands in unless `--b5 zero` (LIT1: "directional on").
    // `--fill-cb <f32>`: a probe, never a picture: every word of the three
    // uncaptured constant buffers (slots 177, 178, 180) set to the value.
    if let Some(v) = opt("--fill-cb").and_then(|v| v.parse::<f32>().ok()) {
        for slot in [177, B5_SLOT, 180] {
            loaded.set_buffer(0, slot, xnw_world::preskin::Kind::Cbv, v.to_le_bytes().repeat(64 * 16));
        }
        println!("  probe: slots 177, 178, 180 filled with {v}");
    }
    let captured_b5 = pack.slot_bytes(sink, 0, B5_SLOT).ok().map(|(_, b)| b);
    if let Some(b) = &captured_b5 {
        let w = |o: usize| b.get(o..o + 4).map_or(0, |x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]));
        println!("  b5 (slot {B5_SLOT}) captured: {} B, +0x30 = {}", b.len(), w(0x30));
    } else if opt("--b5").as_deref() != Some("zero") && opt("--fill-cb").is_none() {
        let mut b5 = vec![0u8; 256];
        b5[0x30..0x34].copy_from_slice(&1u32.to_le_bytes());
        loaded.set_buffer(0, B5_SLOT, xnw_world::preskin::Kind::Cbv, b5);
        println!("  b5 (slot {B5_SLOT}, not captured): +0x30 = 1 stands in, the rest zero");
    }
    // The sun's remaining zero gates (LIT3, found with --ps-patch): GPU
    // outputs no capture holds, each a factor of the sun's direct term.
    // Neutral stand-ins unless `--sun captured`:
    // - slot 29 (T8+0x3a0, 32 x 12 B) +0x54: a scalar on the sun colour
    //   (0x1b60 s78; 0x1c80..0x1c90 s78 x colour, 0x222c again): 1.0;
    // - slot 153 (T8+0x1320, 16x1 R10G10B10A2, a 1-D curve sampled at
    //   (depth / 8192)^(1/4), 0x1be8 and 0x2218), times the colour: white;
    // - slot 101 (T8+0xca0, 640x360, screen space, 0x1c30), times the sun's
    //   rgb at 0x1d90..0x1db8: white;
    // - slot 100 (T8+0xc80, 127x128 R16G16, .r at 0x1c3c, u = 1/2 + sign(N.L)
    //   sqrt|N.L| / 2, v = 1): the diffuse falloff (`lambert_table`);
    // - slot 21 word 2 (0x1b6c s55, scales the SH sum at 0x1cf8): 1.0.
    // `--sun-skip 29,153,...`: leave those slots as captured (the bisection).
    let mut skip: Vec<u32> = opt("--sun-skip").map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    // `--luts <frame dir>|here` (LIT4): the tables the title's own draws
    // make (`lighting_luts::LUTS`: slots 100 and 132) from the frame that
    // holds those draws (pk7d f1059), in place of their stand-ins.
    // `--run-dispatch <sink>,<sink>,...`: those dispatch sinks in
    // order through the shared compute runner; a `--run-draw` after them
    // reads the textures they wrote.
    let mut runner = None;
    if let Some(v) = opt("--run-dispatch") {
        let pack = crate::pack::Pack::open(dir)?.ok_or("--run-dispatch needs a v2 capture (pack)")?;
        let sinks: Vec<u32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
        let mut r = xnw_world::compute::Runner::new(host)?;
        crate::replay::watchdog_arm();
        let lines = crate::compute::run_chain(&pack, &sinks, &mut r);
        crate::replay::watchdog_disarm();
        lines?;
        println!("  runner: {} ran ({:.3} s on the GPU), dropped {:?}, holds {} addresses", r.ran, r.seconds, r.dropped, r.keys().len());
        runner = Some(r);
    }
    if let Some(v) = opt("--run-draw") {
        let base = opt("--out").map_or_else(|| dir.join("lighting.png"), std::path::PathBuf::from);
        crate::lighting_luts::probe_draws(dir, &v, host, &base, runner.as_mut())?;
    }
    let mut made: Vec<(u64, Provided)> = Vec::new();
    if let Some(v) = opt("--luts") {
        let from = if v == "here" { dir.to_path_buf() } else { std::path::PathBuf::from(&v) };
        let size = |slot: u32| view(0, slot).map(|v| (v.width as u32, v.height));
        let base = opt("--out").map_or_else(|| dir.join("lighting.png"), std::path::PathBuf::from);
        for (slot, t, px, tw, th) in crate::lighting_luts::produce(&from, &size, host, &base)? {
            if slot == SUN_LUT_SLOT {
                println!("  slot {slot} falloff (last row): {}", crate::lighting_luts::falloff(&px, tw, th));
            }
            made.push((view(0, slot).map_or(0, |v| v.addr), Provided { width: t.width, height: t.height, format: t.format, rows: t.rows.clone(), row_bytes: t.row_bytes, row_count: t.row_count, dim: t.dim, depth: t.depth, mips: t.mips }));
            loaded.set_texture(0, slot, t);
            skip.push(slot);
        }
    }
    // `--sky-chain` (LIT5): the title's sky and ambient producers of this
    // frame (`lighting_chain::CHAIN`) on the shared compute runner, after
    // the LUTs above and the light scale stand-in are held at the addresses
    // the lighting program reads them at; then every texture of the
    // lighting program's named slots the runner holds (153, 72, ...) and
    // slot 29's 384 B replace their stand-ins.
    if args.iter().any(|a| a == "--sky-chain") {
        let mut r = xnw_world::compute::Runner::new(host)?;
        let unit: Vec<u8> = LIGHT_SCALE_UNIT.iter().flat_map(|x| x.to_le_bytes()).collect();
        let scale = if opt("--slot21").as_deref() == Some("captured") { None } else { view(0, LIGHT_SCALE_SLOT).map(|v| (v.addr, unit)) };
        made.retain(|m| m.0 != 0);
        let base = opt("--out").map_or_else(|| dir.join("lighting.png"), std::path::PathBuf::from);
        crate::replay::watchdog_arm();
        let lines = crate::lighting_chain::run(&pack, host, &mut r, &made, scale, &base);
        crate::replay::watchdog_disarm();
        println!("  sky chain:");
        for l in lines? {
            println!("    {l}");
        }
        println!("    runner: {} ran ({:.3} s on the GPU), dropped {:?}, holds {} addresses", r.ran, r.seconds, r.dropped, r.keys().len());
        let (held, notes) = crate::compute::held_textures(&pack, sink, &mut r);
        for n in notes {
            println!("    {n}");
        }
        for (root, slot, t) in held {
            if root != 0 || !used.contains_key(&(root, slot)) || skip.contains(&slot) && slot != SUN_LUT_SLOT {
                continue;
            }
            println!("    slot {slot}: the chain's {}x{}x{} format {} ({} B)", t.width, t.height, t.depth, t.format, t.rows.len());
            loaded.set_texture(0, slot, t);
            if !skip.contains(&slot) {
                skip.push(slot);
            }
        }
        if let Some(v) = view(0, SUN_SCALE_SLOT)
            && r.held(crate::compute::key(v.addr)).is_some()
        {
            let b = r.read(crate::compute::key(v.addr))?;
            let f = |o: usize| b.get(o..o + 4).map_or(f32::NAN, |x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]));
            println!("    slot {SUN_SCALE_SLOT}: the chain's {} B, +0x54 = {:.4}, words 0-2 ({:.4}, {:.4}, {:.4})", b.len(), f(0x54), f(0), f(4), f(8));
            loaded.set_buffer(0, SUN_SCALE_SLOT, xnw_world::preskin::Kind::Structured(12), b.get(..384).unwrap_or(&b).to_vec());
            skip.push(SUN_SCALE_SLOT);
        }
    }
    let stand = |slot: u32| !skip.contains(&slot);
    if opt("--sun").as_deref() != Some("captured") {
        let mut b29 = pack.slot_bytes(sink, 0, SUN_SCALE_SLOT).map(|(_, b)| b).unwrap_or_else(|_| vec![0; 384]);
        if b29.len() >= 0x58 {
            b29[0x54..0x58].copy_from_slice(&1.0f32.to_le_bytes());
        }
        if stand(SUN_SCALE_SLOT) {
            loaded.set_buffer(0, SUN_SCALE_SLOT, xnw_world::preskin::Kind::Structured(12), b29);
        }
        let white = |w: u32, h: u32| texture(w, h, R8G8B8A8_UNORM, 4, vec![255; (4 * w * h) as usize]);
        for (slot, tw, th) in [(SUN_CURVE_SLOT, 16, 1), (SUN_SCREEN_SLOT, w, h)] {
            if stand(slot) {
                loaded.set_texture(0, slot, white(tw, th));
            }
        }
        if stand(SUN_LUT_SLOT) {
            loaded.set_texture(0, SUN_LUT_SLOT, texture(127, 128, R8G8B8A8_UNORM, 4, lambert_table(127, 128)));
        }
        println!("  sun stand-ins: slot {SUN_SCALE_SLOT} +0x54 = 1.0; slots {SUN_CURVE_SLOT} (16x1), {SUN_SCREEN_SLOT} ({w}x{h}) white; {SUN_LUT_SLOT} Lambert (127x128); skipped {skip:?}");
    }
    // `--white 57,126` (a probe, LIT6): those frame-table slots white
    // (RGBA8 1.0, the pass's size): which input gates a term.
    for slot in opt("--white").map(|v| v.split(',').filter_map(|x| x.parse::<u32>().ok()).collect::<Vec<_>>()).unwrap_or_default() {
        loaded.set_texture(0, slot, texture(w, h, R8G8B8A8_UNORM, 4, vec![255; (4 * w * h) as usize]));
        println!("  --white: slot {slot} white {w}x{h} (a probe)");
    }
    if opt("--shadow").as_deref() != Some("captured") {
        loaded.set_texture(0, SHADOW_SLOT, texture(w, h, R8_UNORM, 1, vec![255; (w * h) as usize]));
        println!("  shadow mask (slot {SHADOW_SLOT}): white {w}x{h} (no shadows)");
    }
    for l in loaded.lacks.iter().filter(|l| !l.starts_with("the group count") && !l.starts_with("the targets")) {
        println!("  lacks: {l}");
    }

    // The draw.
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
    let px = &f.rt0;
    let untouched = px.iter().filter(|p| **p == bridge::CLEAR).count();
    let bad = px.iter().filter(|p| p.iter().any(|c| !c.is_finite())).count();
    let lit: Vec<&[f32; 4]> = px.iter().filter(|p| **p != bridge::CLEAR && p.iter().all(|c| c.is_finite())).collect();
    let mean = |k: usize| lit.iter().map(|p| f64::from(p[k])).sum::<f64>() / lit.len().max(1) as f64;
    let max = lit.iter().map(|p| p[0].max(p[1]).max(p[2])).fold(0f32, f32::max);
    let black = lit.iter().filter(|p| p[0] <= 0.0 && p[1] <= 0.0 && p[2] <= 0.0).count();
    println!(
        "  lighting draw on {} in {:.2} s: {} px, untouched {untouched}, not finite {bad}, black {black}; light rgb mean ({:.4}, {:.4}, {:.4}), max {max:.3}, target 0 .a mean {:.4}",
        host.caps.adapter,
        f.seconds,
        px.len(),
        mean(0),
        mean(1),
        mean(2),
        mean(3)
    );

    // Per channel: pixels not zero, min, max (the debug export reads these).
    let chan: Vec<String> = (0..4)
        .map(|k| {
            let v = lit.iter().map(|p| p[k]);
            let nz = lit.iter().filter(|p| p[k] != 0.0).count();
            format!("{nz} nz [{:.4}, {:.4}]", v.clone().fold(f32::INFINITY, f32::min), v.fold(f32::NEG_INFINITY, f32::max))
        })
        .collect();
    println!("  target 0 per channel: {}", chan.join(" | "));

    // Target 0 (lit colour, albedo included) with our exposure and curve.
    let tc: Vec<f32> = std::fs::read(dir.join("timecycle.bin")).map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()).unwrap_or_default();
    let (l, _) = Lighting::from_timecycle(&tc, None);
    println!("  compose: exposure {:.3} (EV {:.2} + bias {:.2}, timecycle {}), Hable {:?}", l.exposure, l.ev, l.bias, if tc.is_empty() { "missing: defaults" } else { "of the frame" }, l.curve);
    let (albedo, normal) = match our_albedo {
        Some(an) => an,
        None => {
            // The captured texels at the frame table's handle + 32 x slot, when whole.
            let table = tables.iter().find(|t| t.0 == 0).map(|t| t.1);
            let mut rows = |slot: u32| table.and_then(|at| texels(at + 32 * u64::from(slot))).filter(|p| (p.width, p.height) == (w, h) && p.row_bytes == 4 * w).map(|p| p.rows);
            (rows(ALBEDO_SLOT).unwrap_or_else(|| vec![0; (4 * w * h) as usize]), rows(NORMAL_SLOT).unwrap_or_else(|| vec![0; (4 * w * h) as usize]))
        }
    };
    let c = &l.curve;
    let disp = |x: f32| (display(x * l.exposure, c) * 255.0).round() as u8;
    let n = (w * h) as usize;
    let mut sheet = vec![0u8; 4 * 4 * n];
    let sw = 2 * w as usize;
    let mut put = |qx: usize, qy: usize, i: usize, rgb: [u8; 3]| {
        let (x, y) = (i % w as usize + qx * w as usize, i / w as usize + qy * h as usize);
        sheet[4 * (y * sw + x)..4 * (y * sw + x) + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    };
    for i in 0..n {
        let a = [albedo[4 * i], albedo[4 * i + 1], albedo[4 * i + 2]];
        put(0, 0, i, a);
        let nb = [normal[4 * i], normal[4 * i + 1], normal[4 * i + 2], normal[4 * i + 3]];
        let nv = xnw_world::gbuf::unpack_normal(nb);
        put(1, 0, i, if nb == [0; 4] { [0; 3] } else { nv.map(|v| ((v * 0.5 + 0.5) * 255.0) as u8) });
        let p = px[i];
        let ok = p != bridge::CLEAR && p.iter().all(|c| c.is_finite());
        put(0, 1, i, if ok { [disp(p[0]), disp(p[1]), disp(p[2])] } else { [255, 0, 255] });
        // The program multiplies the albedo itself (slot 73 into v2..v4 at 0x110,
        // 0x4e2c..0x4e48 [V disasm]): the light alone is target 0 / albedo.
        let un = |c: f32, b: u8| c / linear(b).max(1.0 / 255.0);
        put(1, 1, i, if ok { [disp(un(p[0], a[0])), disp(un(p[1], a[1])), disp(un(p[2], a[2]))] } else { [255, 0, 255] });
    }
    let out = opt("--out").map_or_else(|| dir.join("lighting.png"), std::path::PathBuf::from);
    crate::replay::write_png(&out, 2 * w, 2 * h, &sheet).map_err(|e| e.to_string())?;
    println!("  sheet {} ({}x{}): albedo | normal / lit (target 0) | light (target 0 / albedo)", out.display(), 2 * w, 2 * h);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn nearest_halves_and_keeps() {
        let rows: Vec<u8> = (0..16).collect();
        assert_eq!(super::nearest(&rows, 1, 4, 4, 2, 2), vec![0, 2, 8, 10]);
        assert_eq!(super::nearest(&rows, 1, 4, 4, 4, 4), rows);
    }

    #[test]
    fn lambert_table_undoes_the_warp() {
        let t = super::lambert_table(127, 2);
        assert_eq!(t.len(), 4 * 127 * 2);
        // u below 1/2 (N.L < 0): dark; the last texel: N.L near 1.
        assert_eq!(t[4 * 10], 0);
        assert!(t[4 * 126] >= 250);
        // u = 3/4 (texel 95): sqrt(N.L) = 1/2, so N.L = 1/4.
        assert!((i32::from(t[4 * 95]) - 64).abs() <= 3);
    }

    #[test]
    fn parse_patch_reads_hex_pairs() {
        assert_eq!(super::parse_patch("0x4ec0=5e04170e,1d4=7ECE0306").unwrap(), vec![(0x4ec0, 0x5e04_170e), (0x1d4, 0x7ece_0306)]);
        assert!(super::parse_patch("0x10").is_err());
    }
}
