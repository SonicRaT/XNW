//! The sky and ambient chain (LIT5): the title's own producers of the
//! lighting program's slots 153, 29 and 72, run once over one frame's
//! captured bindings on the shared compute runner (`crate::compute`),
//! so `world-render <frame> --lighting auto --sky-chain` lights with what
//! the title's GPU would have made instead of LIT3's stand-ins.
//!
//! The steps are named by container (pk8a f1027-1030 hold each once; the
//! sinks are found by container in any v2.9 frame). The order is the
//! dependency order by title address (`<out>/lit5\chain.py` over
//! refs.bin [V]), not the sink order: the capture numbers sinks in CPU
//! record order, and thread 3's ambient dispatches (sinks 46-94 in f1027)
//! read what thread 1's sky dispatches (848-859) write.
//! - Transmittance (256x64) -> MultipleScattering (32x32) -> SkyView
//!   (2 x 192x108); DirectionalLightTransmittanceLUT reads Transmittance
//!   and writes slot 153 (16x1).
//! - `ps_sky_paraboloid_reflection` (a draw: SkyView, 153) -> the 128x128
//!   RGBA16F texture AmbientCubeIBR_Paraboloid reads at root 1 slot 0 (no
//!   other sink binds it, so the draw's target [I]; draw targets are not
//!   captured) -> slot 29 (32 x 12 B).
//! - LightMap_NoAmbient (29, 153, slot 100) -> ClassifyClear -> Classify
//!   (tile counter, tile list) -> SkyClassified32/16 (indirect: x, y, z
//!   from the counter at +0 and +0x20, as ClassifyClear writes them at
//!   0x48/0x50 and Classify adds to them at 0x414; 32 reads tile x >> 1,
//!   16 tile (n/2)^2 + x [V disasm]) -> LightMap -> LightMapPropagate ->
//!   AmbientMap, whose root 1 slot 4 UAV is slot 72 (128x128x7 RGBA16F).
//!
//! What stands in (each printed): the light scale (slot 21, the title's
//! exposure, a GPU output) is `lighting::LIGHT_SCALE_UNIT` in the runner too; the
//! 384x384 sky probe the SkyClassified pair reads (root 0 slot 65) has no
//! writer in the capture (U) and stays zero; LUTs 100 and 132 come from
//! `--luts` when given.

use xnw_world::compute::{Buffer, Image, Outcome, Res, Runner, Shape};
use xnw_world::preskin::Kind;
use xnw_world::textures::Provided;

use crate::compute::{self, key};
use crate::pack::Pack;

/// One step: a dispatch (container), the paraboloid draw (pixel
/// container), or an indirect dispatch whose x, y, z sit at this offset of
/// the classify counter.
#[derive(Clone, Copy)]
pub enum Step {
    Dispatch(&'static str, u128),
    Draw(&'static str, u128),
    Indirect(&'static str, u128, u64),
}

pub const CHAIN: [Step; 14] = [
    Step::Dispatch("CS_Transmittance", 0xe6d7_475a_b43f_1fb9_f858_2daa_d4f5_ec6b),
    Step::Dispatch("CS_MultipleScattering", 0x8c5f_5cfd_aeeb_e66e_bbad_6fb4_a3d5_11d8),
    Step::Dispatch("CS_SkyView", 0x7679_96c1_ce51_eace_99da_7699_f9e8_1c52),
    Step::Dispatch("CS_DirectionalLightTransmittanceLUT", 0x8344_8872_8ebd_10f6_7b1c_0520_6400_c6fe),
    Step::Draw("ps_sky_paraboloid_reflection", PARABOLOID_PS),
    Step::Dispatch("CS_AmbientCubeIBR_Paraboloid", AMBIENT_CUBE),
    Step::Dispatch("CS_LightMap_NoAmbient", 0x5ce5_d87e_a257_6dc3_a9cf_4de7_bc1e_8a4b),
    Step::Dispatch("CS_AmbientMapClassifyClear", CLASSIFY_CLEAR),
    Step::Dispatch("CS_AmbientMapClassify16_32", 0xd470_f982_b9cc_a53a_4061_9890_7d5c_f904),
    Step::Indirect("CS_AmbientMapSkyClassified32", 0x19a9_d7f5_8de3_56c9_7456_0480_2c11_d4a4, 0),
    Step::Indirect("CS_AmbientMapSkyClassified16", 0x51d5_0a7b_c1e1_7941_d362_4b1b_2fec_8eec, 0x20),
    Step::Dispatch("CS_LightMap", 0xe103_4c21_b824_e1dc_f3f2_7414_1239_18ef),
    Step::Dispatch("CS_LightMapPropagate", 0xd0c9_606b_aabb_bd65_28c4_f0f2_faac_e6ac),
    Step::Dispatch("CS_AmbientMap", 0x0e3d_7b2c_8405_da35_8abc_92fe_2a61_6aaf),
];

const PARABOLOID_PS: u128 = 0x068d_222c_84d2_6d16_dced_8472_2bff_fee8;
const AMBIENT_CUBE: u128 = 0x9ae7_b3a3_23b1_ab2a_eb8b_ba7e_e722_c413;
const CLASSIFY_CLEAR: u128 = 0x2649_8285_36e4_0445_ba30_7e6b_0ac4_88b4;
const R16G16B16A16_FLOAT: u32 = 10;

/// The first sink of the frame whose `stage` program is `container`.
pub fn sink_of(pack: &Pack, stage: &str, container: u128) -> Option<u32> {
    let pipes: std::collections::HashSet<u64> = pack.data.programs.iter().filter(|p| p.stage == stage && p.container == container).map(|p| p.pipeline).collect();
    pack.data.sinks.iter().find(|s| pipes.contains(&s.pipeline)).map(|s| s.index)
}

/// The address of sink `sink`'s view at `root` / `slot` (0: none).
pub fn view_addr(pack: &Pack, sink: u32, root: u16, slot: u32) -> u64 {
    pack.tables(sink).iter().find(|t| t.0 == root).and_then(|t| t.2.iter().find(|v| v.slot == slot)).map_or(0, |v| v.addr)
}

/// A texture the runner holds as an input: a `Provided` at `addr`.
pub fn hold_texture(runner: &mut Runner, addr: u64, t: &Provided) -> Result<(), String> {
    runner.hold(&Res::Image(Image { key: key(addr), shape: Shape::Tex2D, format: t.format, width: t.width, height: t.height, depth: 1, uav: false, texels: &t.rows }))
}

/// `XNW_CHAIN_DUMP=<dir>` (a debug export, off when unset): each step's
/// output bytes as `<dir>/<step>-<key>.bin` (a dispatch: what
/// `Runner::read` gives; the draw: its pixels as f32 rgba).
fn dump(step: &str, key: u64, bytes: &[u8]) {
    let Ok(dir) = std::env::var("XNW_CHAIN_DUMP") else { return };
    let path = std::path::Path::new(&dir).join(format!("{step}-{key:x}.bin"));
    if let Err(e) = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, bytes)) {
        println!("  XNW_CHAIN_DUMP {}: {e}", path.display());
    }
}

/// Indirect group counts from the counter's three words at `at`.
pub fn args_at(counter: &[u8], at: u64) -> Option<[u32; 3]> {
    let w = |i: u64| counter.get((at + 4 * i) as usize..(at + 4 * i + 4) as usize).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    Some([w(0)?, w(1)?, w(2)?])
}

/// Runs [`CHAIN`] over `pack`'s frame on `runner`, after `inputs` (address,
/// texels: the LUTs, the light scale) are held. One line per step.
pub fn run(pack: &Pack, host: &xnw_gpu::host::Host, runner: &mut Runner, inputs: &[(u64, Provided)], scale: Option<(u64, Vec<u8>)>, out: &std::path::Path) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    for (addr, t) in inputs {
        hold_texture(runner, *addr, t)?;
        lines.push(format!("held {addr:#x}: {}x{} format {} (an input)", t.width, t.height, t.format));
    }
    if let Some((addr, bytes)) = &scale {
        runner.hold(&Res::Buffer(Buffer { key: key(*addr), kind: Kind::Structured(16), uav: false, len: bytes.len() as u64, bytes }))?;
        lines.push(format!("held {addr:#x}: the light scale stand-in {:?}", crate::lighting::LIGHT_SCALE_UNIT));
    }
    let mut counter = 0u64;
    for step in CHAIN {
        let (name, container, stage) = match step {
            Step::Dispatch(n, c) | Step::Indirect(n, c, _) => (n, c, "compute"),
            Step::Draw(n, c) => (n, c, "pixel"),
        };
        let Some(sink) = sink_of(pack, stage, container) else {
            lines.push(format!("{name}: no sink of {:08x} in this frame: skipped", (container >> 96) as u32));
            continue;
        };
        if let Step::Draw(..) = step {
            // Its target: the texture AmbientCubeIBR_Paraboloid reads at root 1 slot 0 [I].
            let cube = sink_of(pack, "compute", AMBIENT_CUBE).ok_or("no AmbientCubeIBR_Paraboloid sink for the draw's target")?;
            let at = view_addr(pack, cube, 1, 0);
            let (w, h) = pack.tables(cube).iter().find(|t| t.0 == 1).and_then(|t| t.2.iter().find(|v| v.slot == 0)).map_or((128, 128), |v| (v.width as u32, v.height));
            let (px, secs, lacks) = crate::lighting_luts::run_draw(pack, sink, 3, w, h, host, Some(runner))?;
            let t = crate::lighting_luts::texels(&px, w, h, R16G16B16A16_FLOAT);
            let untouched = px.iter().filter(|p| **p == xnw_world::bridge::CLEAR).count();
            let nz = px.iter().filter(|p| **p != xnw_world::bridge::CLEAR && (p[0] != 0.0 || p[1] != 0.0 || p[2] != 0.0)).count();
            let alpha = px.iter().filter(|p| **p != xnw_world::bridge::CLEAR).map(|p| p[3]).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
            let max = px.iter().map(|p| p[0].max(p[1]).max(p[2])).fold(0f32, f32::max);
            hold_texture(runner, at, &t)?;
            dump(name, key(at), &px.iter().flat_map(|p| p.iter().flat_map(|c| c.to_le_bytes())).collect::<Vec<u8>>());
            lines.push(format!("{name} (draw sink {sink}) at {w}x{h} in {secs:.2} s: untouched {untouched}, rgb not zero {nz} of {}, max {max:.4}, a {alpha:?}; held at {at:#x} (AmbientCubeIBR's root 1 slot 0)", px.len()));
            lines.extend(lacks.into_iter().map(|l| format!("    lacks: {l}")));
            let pic: Vec<u8> = px.iter().flat_map(|p| [p[0], p[1], p[2]].map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8).into_iter().chain([255])).collect();
            let png = out.with_extension("paraboloid.png");
            crate::replay::write_png(&png, w, h, &pic).map_err(|e| e.to_string())?;
            lines.push(format!("    picture {}", png.display()));
            continue;
        }
        if container == CLASSIFY_CLEAR {
            counter = view_addr(pack, sink, 1, 0);
        }
        let p = compute::program(pack, sink)?;
        let mut loaded = compute::load(pack, sink, &p.uses, &p.named);
        if let Step::Indirect(_, _, at) = step {
            // The record the capture names (v2.11), else the classify
            // counter at the step's offset (LIT5, older captures).
            let got = match compute::indirect_groups(pack, sink, runner) {
                Some((g, rec)) => Some((g, format!("the held record at {rec:#x}"))),
                None => {
                    let words = if counter != 0 && runner.held(key(counter)).is_some() { runner.read(key(counter)).ok() } else { None };
                    words.as_deref().and_then(|b| args_at(b, at)).map(|g| (g, format!("the classify counter {counter:#x} +{at:#x}")))
                }
            };
            match got {
                Some((g, from)) => {
                    loaded.groups = g;
                    loaded.lacks.retain(|l| !l.starts_with("the y and z group counts"));
                    lines.push(format!("{name}: indirect groups {g:?} from {from}"));
                }
                None => lines.push(format!("{name}: no classify counter held: groups {:?} (the sink's count)", loaded.groups)),
            }
        }
        let head = format!("{name} (sink {sink}, groups {:?})", loaded.groups);
        println!("  {head}: running");
        match runner.dispatch(&p.shader, p.wave, &p.uses, &loaded.job()).map_err(|e| format!("{head}: {e}"))? {
            Outcome::Dropped(d) => lines.push(format!("{head}: dropped: {} ({})", d.reason, d.detail)),
            Outcome::Ran(r) => {
                let mut wrote = Vec::new();
                for &k in &r.wrote {
                    let b = runner.read(k)?;
                    wrote.push(format!("{k:#x} {} of {} B non-zero", b.iter().filter(|x| **x != 0).count(), b.len()));
                    dump(name, k, &b);
                }
                lines.push(format!("{head}: ran in {:.3} s; wrote [{}]", r.seconds, wrote.join("; ")));
            }
        }
        lines.extend(loaded.lacks.iter().map(|l| format!("    lacks: {l}")));
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    #[test]
    fn args_read_three_words_at_the_offset() {
        let mut c = vec![0u8; 64];
        c[0..12].copy_from_slice(&[7, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
        c[32..44].copy_from_slice(&[3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(super::args_at(&c, 0), Some([7, 1, 1]));
        assert_eq!(super::args_at(&c, 0x20), Some([3, 1, 1]));
        assert_eq!(super::args_at(&c, 60), None);
    }

    #[test]
    fn the_chain_ends_in_the_ambient_map_and_draws_once() {
        let draws = super::CHAIN.iter().filter(|s| matches!(s, super::Step::Draw(..))).count();
        assert_eq!(draws, 1);
        assert!(matches!(super::CHAIN[13], super::Step::Dispatch("CS_AmbientMap", _)));
    }
}
