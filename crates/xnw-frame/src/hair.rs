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

/// The frame's strand draws and the census (`chain`: the pool the strand
/// chain wrote, at its address, in place of the captured one).
pub fn strands(p: &Pack, chain: Option<(u64, &[u8])>) -> (Vec<StrandDraw>, Census) {
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
                    pools.entry(v.addr).or_insert_with(|| chain.filter(|c| c.0 == v.addr).map_or(b, |c| c.1.to_vec()));
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
        return Err("usage: xnw-lab hair <frame dirs...> [--draws] [--chain-refs]".into());
    }
    if args.iter().any(|a| a == "--chain") {
        return dirs.iter().try_for_each(|d| crate::hair_chain::chain_cmd(Path::new(d)).map(|_| ()));
    }
    if args.iter().any(|a| a == "--chain-refs") {
        return dirs.iter().try_for_each(|d| chain_refs(Path::new(d)));
    }
    for d in dirs {
        let Some(p) = Pack::open(Path::new(d))? else {
            println!("{d}: no pack");
            continue;
        };
        let (draws, c) = strands(&p, None);
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
pub fn bridge_cli(args: &[String], dir: &Path, host: &xnw_gpu::host::Host, size: (u32, u32)) -> Result<Strands, String> {
    let p = Pack::open(dir)?.ok_or("--hair-bridge needs a v2 capture")?;
    // `--hair-probe`: made-up points in the pool (a placement probe of the instrument, never hair).
    let probe = args.iter().any(|a| a == "--hair-probe");
    if probe {
        println!("hair bridge: --hair-probe: the pool holds made-up straight strands (placement probe, not hair)");
    }
    // `--hair-chain`: the title's strand chain first, its pool in place of the captured one.
    let chain = if args.iter().any(|a| a == "--hair-chain") {
        let c = crate::hair_chain::run(&p, host)?;
        for s in c.steps.iter().filter(|s| s.error.is_some()) {
            println!("  chain sink {} {}: {}", s.sink, s.name, s.error.as_deref().unwrap_or(""));
        }
        println!("{}", c.line());
        if c.steps.iter().any(|s| s.error.is_some()) {
            return Err("hair bridge: the strand chain did not run whole: nothing drawn".into());
        }
        Some(c)
    } else {
        None
    };
    let mut vertex = Vertex::new();
    let (union, lines, n) = draw_strands(&p, host, size, chain.as_ref(), probe, &mut vertex)?;
    for l in &lines {
        println!("{l}");
    }
    if let Some(out) = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)) {
        let rgba: Vec<u8> = union.depth.iter().flat_map(|&z| if z != union.far { [255, 140, 20, 255] } else { [0, 0, 0, 255] }).collect();
        crate::replay::write_png(Path::new(out), size.0, size.1, &rgba).map_err(|e| e.to_string())?;
    }
    println!("hair bridge: {n} strand draws run, {} pixels covered in all", union.covered());
    Ok(union)
}

/// Translated strand vertex programs by container hash, with the slots
/// they name (HAIR6: the live path keeps them across frames).
pub type Vertex = HashMap<u128, Result<(xnw_world::bridge::Programs, HashMap<(u16, u32), u32>), String>>;

/// Every main-view strand draw of `p` through the title's vertex program on
/// the bridge's plain-draw mode (one instance per strand of its group, a
/// triangle strip [I]), depth only, reverse depth, into one union of the
/// nearest strand per pixel at `size`. `chain`: its pool in place of the
/// captured one; `probe`: made-up straight strands ([`probe_points`]).
/// Returns the union, the lines (the census, then one per draw) and the
/// strand draws. `world-render --hair-bridge` and the live path (HAIR6)
/// both draw here.
pub fn draw_strands(p: &Pack, host: &xnw_gpu::host::Host, size: (u32, u32), chain: Option<&crate::hair_chain::Chain>, probe: bool, vertex: &mut Vertex) -> Result<(Strands, Vec<String>, usize), String> {
    let (draws, c) = strands(p, chain.map(|c| (c.pool_at, &c.pool[..])));
    let mut lines = vec![format!("{}: {}", p.dir.display(), c.line())];
    let counts = std::sync::Arc::new([const { std::sync::atomic::AtomicU64::new(0) }; 3]);
    let mut texels = p.texel_provider(counts);
    let mut union = Strands { depth: vec![0.0; (size.0 * size.1) as usize], far: 0.0 };
    for s in &draws {
        let vs = crate::bridge::vertex_container(p, s.sink)?;
        let (prog, used) = match vertex.entry(vs).or_insert_with(|| crate::bridge::vertex_program(p, s.sink, 0, 0, true).map(|(prog, used, _)| (prog, used))) {
            Ok((prog, used)) => (xnw_world::bridge::Programs { vertex: Some((s.draw.vertices, s.group.strands, true)), ..prog.clone() }, used.clone()),
            Err(e) => return Err(e.clone()),
        };
        let mut loaded = crate::bridge::load(p, s.sink, &crate::bridge::Row::default(), None, &used, &mut texels);
        if probe && let Some(pool) = loaded.slot_bytes_mut(0, hair::POINTS_SLOT) {
            probe_points(s, pool);
        }
        if let Some(c) = chain
            && let Some(pool) = loaded.slot_bytes_mut(0, hair::POINTS_SLOT)
        {
            pool.clone_from(&c.pool);
        }
        let lacks: Vec<&String> = loaded.lacks.iter().filter(|l| !l.starts_with("the group count") && !l.starts_with("sampler heap") && !l.starts_with("the targets")).collect();
        let d = loaded.draw();
        let b = xnw_world::bridge::bind(&prog.ms, &prog.ps, &d)?;
        if let Some(gap) = xnw_world::bridge::gap(&host.caps, &prog.needs()) {
            lines.push(format!("  sink {}: not run: {gap}", s.sink));
            continue;
        }
        // Reverse depth, as the pass: greater-equal, written.
        let target = xnw_world::bridge::Target { width: size.0, height: size.1, depth: Some((true, 7)), limit: if host.caps.software { 120.0 } else { 1.0 }, depth_only: true, records: false };
        crate::replay::watchdog_arm();
        let f = xnw_world::bridge::run(host, &prog, &b, &target);
        crate::replay::watchdog_disarm();
        let f = f?;
        let mut n = 0;
        union.far = f.far;
        for (i, z) in f.depth.iter().enumerate() {
            if *z != f.far {
                n += 1;
                // Reverse depth: the nearest strand is the greatest.
                union.depth[i] = if union.depth[i] == f.far { *z } else { union.depth[i].max(*z) };
            }
        }
        lines.push(format!(
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
        ));
    }
    Ok((union, lines, draws.len()))
}

/// The strand draws' pixels: per pixel the nearest strand's depth (reverse
/// depth, as the pass: the greatest), `far` where no strand drew.
#[derive(Clone, Debug, Default)]
pub struct Strands {
    pub depth: Vec<f32>,
    pub far: f32,
}

impl Strands {
    pub fn covered(&self) -> usize {
        self.depth.iter().filter(|z| **z != self.far).count()
    }

    /// Whether pixel `i` shows a strand over `scene` (our pass's depth): the
    /// strand passes the title's own strand depth test, greater-equal
    /// (HAIR4: without it, pk8a f1027's group 7, body hair under the hoodie,
    /// lay over the man's upper arm). No scene: every strand pixel.
    pub fn shows(&self, i: usize, scene: Option<&[f32]>) -> bool {
        let z = self.depth.get(i).copied().unwrap_or(self.far);
        z != self.far && scene.and_then(|s| s.get(i)).is_none_or(|s| z >= *s)
    }
}

impl Strands {
    /// The strand pixels as our pass draws them ([`xnw_world::hair::Pixels`]):
    /// (x, y, depth, 0) per covered pixel of a target `width` wide.
    pub fn pixels(&self, width: u32) -> xnw_world::hair::Pixels {
        self.depth.iter().enumerate().filter(|(_, z)| **z != self.far).map(|(i, z)| [(i as u32 % width) as f32, (i as u32 / width) as f32, *z, 0.0]).collect()
    }
}

/// HAIR6 `XNW_HAIR_OVERLAY=pass` (replay; default off): the strand pixels
/// go with the list and our pass draws them under its depth test
/// (`xnw_world::hair::HairPass`, the live path's), not painted over the
/// picture read back.
pub fn overlay_in_pass() -> bool {
    std::env::var("XNW_HAIR_OVERLAY").is_ok_and(|v| v == "pass")
}

/// Takes the pending strand pixels (for [`overlay_in_pass`]).
pub fn take_overlay() -> Option<Strands> {
    OVERLAY.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// The strand pixels `hair=on` paints over the pass's picture (set before the pass draws).
static OVERLAY: std::sync::Mutex<Option<Strands>> = std::sync::Mutex::new(None);

pub fn set_overlay(s: Strands) {
    *OVERLAY.lock().unwrap_or_else(|e| e.into_inner()) = Some(s);
}

/// Whether strand pixels wait to be painted (the caller reads the pass's depth for [`overlay`]).
pub fn overlay_pending() -> bool {
    OVERLAY.lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

/// Paints the strand pixels (a flat hair brown: our simple pixel stage) over
/// `rgba` where they pass the depth test against `scene` (our pass's depth
/// rows; `XNW_HAIR_DEPTH=off`: none, every strand pixel as before HAIR4).
/// Prints the pixels painted and hidden.
pub fn overlay(rgba: &mut [u8], scene: Option<&[f32]>) {
    let Some(s) = OVERLAY.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    let scene = scene.filter(|_| std::env::var("XNW_HAIR_DEPTH").map_or(true, |v| v != "off"));
    let mut shown = 0;
    for (i, px) in rgba.chunks_exact_mut(4).enumerate() {
        if s.shows(i, scene) {
            px.copy_from_slice(&[96, 64, 40, 255]);
            shown += 1;
        }
    }
    println!("hair overlay: {shown} of {} strand pixels painted ({} behind our pass's depth{})", s.covered(), s.covered() - shown, if scene.is_none() { "; no depth test" } else { "" });
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

    /// HAIR4: a strand pixel shows where it is nearer than (reverse depth:
    /// greater than) or as near as the pass's depth; none drawn never shows.
    #[test]
    fn strand_pixels_pass_the_reverse_depth_test() {
        let s = super::Strands { depth: vec![0.0, 0.2, 0.2, 0.1], far: 0.0 };
        let scene = [0.5, 0.1, 0.2, 0.3];
        let shown: Vec<bool> = (0..4).map(|i| s.shows(i, Some(&scene))).collect();
        assert_eq!(shown, [false, true, true, false]);
        assert_eq!((0..4).filter(|i| s.shows(*i, None)).count(), 3);
        assert_eq!(s.covered(), 3);
    }
}

/// `xnw-lab hair <frame> --chain-refs`: every strand-input dispatch sink
/// (`smh_*` by name) with its records: roots, table slots, bytes, strand views.
pub fn chain_refs(dir: &Path) -> Result<(), String> {
    let p = Pack::open(dir)?.ok_or("no pack")?;
    let names = crate::audit::program_names(dir).ok_or("no shader-names.tsv above the frame")?;
    let prog: HashMap<u64, u128> = p.data.programs.iter().filter(|g| g.stage == "compute").map(|g| (g.pipeline, g.container)).collect();
    for s in p.data.sinks.iter().filter(|s| s.op == fp::op::DISPATCH) {
        let Some(h) = prog.get(&s.pipeline) else { continue };
        let Some(n) = names.get(h).filter(|n| n.starts_with("smh_")) else { continue };
        println!("sink {} count {} {:08x} {n}", s.index, s.count, (h >> 96) as u32);
        if let Ok((_, recs)) = p.slot_bytes(s.index, 0, hair::GROUPS_SLOT) {
            for (r, rec) in recs.chunks_exact(GROUP_BYTES).enumerate() {
                let w: Vec<String> = rec.chunks_exact(4).enumerate().map(|(i, b)| format!("{:x}:{:x}", i * 4, u32::from_le_bytes([b[0], b[1], b[2], b[3]]))).collect();
                println!("  record {r}: {}", w.join(" "));
            }
        }
        for r in p.data.refs_of(s.index) {
            let b = p.bytes_of(r);
            let extra = match r.kind {
                kind::ROOTS => b.as_ref().map(|b| format!("{:x?}", b.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect::<Vec<_>>())).unwrap_or_default(),
                kind::TABLE | kind::STRAND_VIEW => b.as_ref().map(|b| fp::decode_slots(b).iter().map(|v| format!("[{} k{} lay{} {:#x} +{} fmt{} w{} {:x?}]", v.slot, v.kind, v.layout, v.addr, v.bytes, v.format, v.width, &v.words[..8])).collect::<Vec<_>>().join(" ")).unwrap_or_default(),
                _ => b.as_ref().map(|b| format!("{} B, nonzero {}", b.len(), b.iter().filter(|x| **x != 0).count())).unwrap_or_default(),
            };
            println!("  {} slot {} addr {:#x} len {} reason {} {extra}", kind::name(r.kind), r.slot, r.addr, r.len, r.reason);
        }
    }
    Ok(())
}
