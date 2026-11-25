//! The frame pack's strand hair draws (HAIR1): each main-view strand draw
//! (vertex program `hair::STRAND_VS`) as its root constants, group record
//! and world, and whether the pool holds its strands' points
//! (`xnw-lab hair <frame dirs...>`); and each strand draw through the
//! title's own vertex program on the bridge (`world-render <frame>
//! --hair-bridge`, depth only: the pixel program writes a visibility buffer).

use std::collections::HashMap;
use std::path::Path;

use xnw_formats::framepack::{self as fp, kind, reason};
use xnw_world::hair::{self, Census, Draw, GROUP_BYTES, Group, WORLD_BYTES};

use crate::pack::Pack;

/// One main-view strand draw of a frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrandDraw {
    pub sink: u32,
    pub draw: Draw,
    pub group: Group,
}

/// The frame's strand draws and the census.
pub fn strands(p: &Pack) -> (Vec<StrandDraw>, Census) {
    let vs: HashMap<u64, u128> = p.data.programs.iter().filter(|g| g.stage == "vertex").map(|g| (g.pipeline, g.container)).collect();
    let mut c = Census::default();
    let mut out = Vec::new();
    let mut pools: HashMap<u64, Vec<u8>> = HashMap::new();
    for s in p.data.sinks.iter().filter(|s| s.op == fp::op::DRAW) {
        match vs.get(&s.pipeline) {
            Some(&h) if h == hair::STRAND_SHADOW_VS => c.shadow_draws += 1,
            Some(&h) if h == hair::STRAND_VS => {
                c.draws += 1;
                let roots = p.data.refs_of(s.index).find(|r| r.kind == kind::ROOTS && r.slot == 2 && reason::has_bytes(r.reason)).and_then(|r| p.blob(r.hash));
                let groups = p.slot_bytes(s.index, 0, hair::GROUPS_SLOT).ok();
                let worlds = p.slot_bytes(s.index, 0, hair::WORLDS_SLOT).ok();
                let (Some(roots), Some((_, groups)), Some((_, worlds))) = (roots, groups, worlds) else {
                    c.unreadable += 1;
                    continue;
                };
                let roots: Vec<u32> = roots.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
                let gi = roots.get(3).copied().unwrap_or(u32::MAX) as usize;
                let Some(gb) = groups.get(gi * GROUP_BYTES..(gi + 1) * GROUP_BYTES) else {
                    c.unreadable += 1;
                    continue;
                };
                let group = Group::parse(gb);
                let wi = group.world as usize;
                let Some(wb) = worlds.get(wi * WORLD_BYTES..(wi + 1) * WORLD_BYTES) else {
                    c.unreadable += 1;
                    continue;
                };
                let draw = Draw::new(&roots, s.count, wb);
                // The pool is one view for every draw (the frame table's slot): read once.
                let at = p.slot_bytes(s.index, 0, hair::POINTS_SLOT).ok().map(|(v, b)| {
                    pools.entry(v.addr).or_insert(b);
                    v.addr
                });
                let pool = at.and_then(|a| pools.get(&a));
                c.strands += u64::from(group.strands);
                for k in 0..group.strands {
                    let (at, len) = group.strand_bytes(draw.lod, k);
                    let Some(b) = pool.and_then(|b| b.get(at as usize..(at + len) as usize)) else { continue };
                    c.captured += 1;
                    c.with_points += u64::from(b.iter().any(|&x| x != 0));
                }
                out.push(StrandDraw { sink: s.index, draw, group });
            }
            _ => {}
        }
    }
    (out, c)
}

/// `xnw-lab hair <frame dirs...> [--draws]`: the census line per frame;
/// `--draws` lists each strand draw (sink, LOD, group, points, strands,
/// box, world translation).
pub fn hair_cmd(args: &[String]) -> Result<(), String> {
    let list = args.iter().any(|a| a == "--draws");
    // Frame directories (the lab appends `--adapter hw`: its value is no directory).
    let dirs: Vec<&String> = args.iter().filter(|a| !a.starts_with("--") && Path::new(a).is_dir()).collect();
    if dirs.is_empty() {
        return Err("usage: xnw-lab hair <frame dirs...> [--draws]".into());
    }
    for d in dirs {
        let Some(p) = Pack::open(Path::new(d))? else {
            println!("{d}: no pack");
            continue;
        };
        let (draws, c) = strands(&p);
        println!("{d}: {}", c.line());
        for s in draws.iter().filter(|_| list) {
            let (g, w) = (&s.group, &s.draw.world);
            println!(
                "  sink {} lod {} group {} vertices {} points {} strands {} box {:.3?} + {:.3?} at ({:.2}, {:.2}, {:.2})",
                s.sink,
                s.draw.lod,
                s.draw.group,
                s.draw.vertices,
                g.layout(s.draw.lod).1,
                g.strands,
                g.lo,
                g.ext,
                w[3],
                w[7],
                w[11]
            );
        }
    }
    Ok(())
}

/// `world-render <frame> --hair-bridge [--out <png>]`: every main-view strand
/// draw of the frame through the title's vertex program (the bridge's
/// plain-draw mode), one instance per strand of its group (the capture
/// holds no instance count: [I]), as a triangle strip [I], into one depth
/// target each; prints the pixels each draw covers. The points are the
/// captured pool's (zero until the smh chain runs: every strand is culled
/// by the program's own length test, 0 pixels); `--hair-probe` puts
/// made-up strands there first ([`probe_points`]).
pub fn bridge_cli(args: &[String], dir: &Path, host: &xnw_gpu::host::Host, size: (u32, u32)) -> Result<(), String> {
    let p = Pack::open(dir)?.ok_or("--hair-bridge needs a v2 capture")?;
    // `--hair-probe`: made-up points in the pool (a placement probe of the instrument, never hair).
    let probe = args.iter().any(|a| a == "--hair-probe");
    if probe {
        println!("hair bridge: --hair-probe: the pool holds made-up straight strands (placement probe, not hair)");
    }
    let (draws, c) = strands(&p);
    println!("{}: {}", dir.display(), c.line());
    let counts = std::sync::Arc::new([const { std::sync::atomic::AtomicU64::new(0) }; 3]);
    let mut texels = p.texel_provider(counts);
    let mut union = vec![false; (size.0 * size.1) as usize];
    for s in &draws {
        let (prog, used, vs) = crate::bridge::vertex_program(&p, s.sink, s.draw.vertices, s.group.strands, true)?;
        let mut loaded = crate::bridge::load(&p, s.sink, &crate::bridge::Row::default(), None, &used, &mut texels);
        if probe && let Some(pool) = loaded.slot_bytes_mut(0, hair::POINTS_SLOT) {
            probe_points(s, pool);
        }
        let lacks: Vec<&String> = loaded.lacks.iter().filter(|l| !l.starts_with("the group count") && !l.starts_with("sampler heap") && !l.starts_with("the targets")).collect();
        let d = loaded.draw();
        let b = xnw_world::bridge::bind(&prog.ms, &prog.ps, &d)?;
        if let Some(gap) = xnw_world::bridge::gap(&host.caps, &prog.needs()) {
            println!("  sink {}: not run: {gap}", s.sink);
            continue;
        }
        // Reverse depth, as the pass: greater-equal, written.
        let target = xnw_world::bridge::Target { width: size.0, height: size.1, depth: Some((true, 7)), limit: if host.caps.software { 120.0 } else { 1.0 }, depth_only: true, records: false };
        crate::replay::watchdog_arm();
        let f = xnw_world::bridge::run(host, &prog, &b, &target);
        crate::replay::watchdog_disarm();
        let f = f?;
        let mut n = 0;
        for (i, z) in f.depth.iter().enumerate() {
            if *z != f.far {
                n += 1;
                union[i] = true;
            }
        }
        println!(
            "  sink {} vs {:032x}: lod {} group {} {} vertices x {} strands on {} in {:.3} s: {n} pixels; vertex constants {:x?}{}",
            s.sink,
            vs,
            s.draw.lod,
            s.draw.group,
            s.draw.vertices,
            s.group.strands,
            host.caps.adapter,
            f.seconds,
            b.mesh.constants,
            if lacks.is_empty() { String::new() } else { format!("; lacks: {}", lacks.iter().map(|l| l.as_str()).collect::<Vec<_>>().join("; ")) }
        );
    }
    if let Some(out) = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)) {
        let rgba: Vec<u8> = union.iter().flat_map(|&m| if m { [255, 140, 20, 255] } else { [0, 0, 0, 255] }).collect();
        crate::replay::write_png(Path::new(out), size.0, size.1, &rgba).map_err(|e| e.to_string())?;
    }
    println!("hair bridge: {} strand draws run, {} pixels covered in all", draws.len(), union.iter().filter(|m| **m).count());
    Ok(())
}

/// Placement probe: straight strands hanging from the top of the group's box
/// over 60 % of its height, spread over its width by a hash of the strand,
/// written into `pool` where the draw's strands lie (16 B a point: three
/// u16 in the pool, f16 arc length; the layout the vertex program reads).
pub fn probe_points(s: &StrandDraw, pool: &mut [u8]) {
    let (n, g) = (s.group.layout(s.draw.lod).1 as u64, &s.group);
    for k in 0..g.strands {
        let (at, _) = g.strand_bytes(s.draw.lod, k);
        let h = k.wrapping_mul(0x9e37_79b9) ^ 0x5bd1_e995;
        let (fx, fy) = ((h & 0xffff) as f32 / 65535.0, (h >> 16) as f32 / 65535.0);
        for i in 0..n {
            let v = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
            let q = [fx, fy, 1.0 - 0.6 * v].map(|u| (u.clamp(0.0, 1.0) * 65535.0).round() as u32);
            let arc = half(v * 0.6 * g.ext[2].abs());
            let o = (at + i * hair::POINT_BYTES as u64) as usize;
            let Some(p) = pool.get_mut(o..o + hair::POINT_BYTES) else { return };
            for c in 0..3 {
                p[4 * c..4 * c + 4].copy_from_slice(&q[c].to_le_bytes());
            }
            p[12..16].copy_from_slice(&u32::from(arc).to_le_bytes());
        }
    }
}

/// f32 to IEEE half (normal range, truncated; arc lengths of a metre or so).
fn half(x: f32) -> u16 {
    if x < 6.2e-5 {
        return 0;
    }
    let b = x.to_bits();
    ((((b >> 23) & 0xff) - 112) << 10 | ((b >> 13) & 0x3ff)) as u16
}

#[cfg(test)]
mod tests {
    #[test]
    fn half_of_arc_lengths() {
        for x in [0.25f32, 0.6, 1.0, 1.5] {
            assert!((xnw_world::hair::f16(super::half(x)) - x).abs() < x * 1e-3, "{x}");
        }
        assert_eq!(super::half(0.0), 0);
    }
}
