//! The capture reader: a probe frame (`draws.tsv`, `camera.txt`, `blobs\`)
//! and a live list (`list.tsv`, the shared `blobs\k-*`), the timecycle
//! lighting, the water map and the terrain tiles of a frame directory.
//! Moved verbatim from apps/xnw/src/worldprobe.rs (59afddb);
//! `load_terrain` from apps/xnw/src/worldterrain.rs.

use std::collections::HashMap;

use xnw_world::terrain::{TILE_BYTES, Terrain, parse};
use xnw_world::{Albedo, Camera as RCamera, DrawRecord, FrameList, GeomKey};

use crate::decode::{Mesh, decode_indexed, decode_meshlets, expand_meshlets, fnv, max_meshlet_vertices};
use crate::le32;
use crate::world::{WorldDraw, WorldSrc, camera_position, plausible};

/// A probe frame read back from its files.
pub struct LoadedFrame {
    pub draws: Vec<WorldDraw>,
    pub views: HashMap<u64, [f32; 16]>,
    pub projs: HashMap<u64, [f32; 16]>,
    pub meshes: HashMap<u64, Mesh>,
    pub failed: Vec<String>,
    /// Geometries taken from the old probe's scene-local.obj (no blobs).
    pub from_obj: usize,
    /// Per draw, its palette address (0 none), and the palettes found in blobs\.
    pub draw_palette: Vec<u64>,
    pub draw_stage: Vec<u32>,
    pub draw_caller: Vec<String>,
    pub draw_ghash: Vec<u64>,
    /// Per draw its instance block (inst-<i>.bin), and the blocks (16 floats per record).
    pub draw_inst: Vec<Option<u32>>,
    pub inst_blocks: HashMap<u32, Vec<f32>>,
    /// The model's skinned bit (model+0x35 bit 0) when the dump has it.
    pub draw_skinned: Vec<Option<bool>>,
    pub palettes: HashMap<u64, Vec<f32>>,
    /// Per geometry, its raw vertex buffer and meshlets expanded to
    /// vertex-buffer indices (blobs only), for the shader-decode path.
    pub raw: HashMap<u64, (Vec<u8>, Vec<u32>)>,
    pub max_meshlet_vertices: u32,
}

/// Reads draws.tsv, camera.txt and the geometries: blobs\ when the frame
/// has them, else the probe's scene-local.obj (frames dumped without
/// `raw=`; geometries whose positions reach 1e4 are dropped: the old
/// decoder read float positions as packed ones).
pub fn load_frame(dir: &std::path::Path) -> Result<LoadedFrame, String> {
    let tsv = std::fs::read_to_string(dir.join("draws.tsv")).map_err(|e| format!("draws.tsv: {e}"))?;
    let mut lines = tsv.lines();
    let head: Vec<&str> = lines.next().ok_or("draws.tsv is empty")?.split('\t').collect();
    let col = |n: &str| head.iter().position(|h| *h == n);
    let (cg, ce, cv, cw, cwa, cie) = (col("geometry").ok_or("no geometry column")?, col("effect"), col("viewport"), col("world").ok_or("no world column")?, col("world_alt"), col("ib_elem"));
    let cs = col("world_src");
    let cst = col("stage");
    let ccl = col("caller");
    let (cgh, cpi, cpi2) = (col("ghash"), col("pal_index"), col("pal2_index"));
    let (cib, cic) = (col("inst_block"), col("inst_count"));
    let (cci, ccn) = (col("cur_idx"), col("cur_n"));
    let mut draw_skin: Vec<(u32, u32)> = Vec::new();
    let mut draw_inst: Vec<Option<u32>> = Vec::new();
    let mut draw_ghash = Vec::new();
    let mut draw_skinned = Vec::new();
    let cms = col("model_skinned");
    let mut draw_pal_index = Vec::new();
    let mut draw_pal2_index = Vec::new();
    let mut draw_stage = Vec::new();
    let mut draw_caller = Vec::new();
    let cp = col("palette");
    let mut draw_palette = Vec::new();
    let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0);
    let floats = |s: &str| -> Option<Vec<f32>> {
        let v: Vec<f32> = s.replace('|', " ").split_whitespace().map(|x| x.parse::<f32>().unwrap_or(f32::NAN)).collect();
        (v.len() == 16 && v.iter().all(|x| x.is_finite())).then_some(v)
    };
    let mut draws = Vec::new();
    let mut elems: HashMap<u64, u32> = HashMap::new();
    for (i, l) in lines.enumerate() {
        let f: Vec<&str> = l.split('\t').collect();
        let get = |c: Option<usize>| c.and_then(|c| f.get(c).copied()).unwrap_or("-");
        let g = hex(get(Some(cg)));
        let world = floats(get(Some(cw))).map(|a| {
            let b = floats(get(cwa)).unwrap_or_else(|| a.clone());
            let mut m = [0f32; 32];
            m[..16].copy_from_slice(&a);
            m[16..].copy_from_slice(&b);
            m
        });
        let src = match (world.is_some(), cs.map(|_| WorldSrc::parse(get(cs)))) {
            (false, _) => WorldSrc::None,
            (true, Some(s)) if s != WorldSrc::None && s != WorldSrc::Borrowed => s,
            (true, _) => WorldSrc::Bound,
        };
        elems.insert(g, get(cie).parse().unwrap_or(2));
        draw_palette.push(hex(get(cp)));
        draw_stage.push(get(cst).parse().unwrap_or(0));
        draw_caller.push(get(ccl).to_string());
        draw_ghash.push(hex(get(cgh)));
        draw_inst.push(get(cib).parse::<u32>().ok().filter(|_| get(cic) != "0"));
        draw_skin.push((get(cci).parse().unwrap_or(0), get(ccn).parse().unwrap_or(0)));
        draw_skinned.push(match get(cms) { "1" => Some(true), "0" => Some(false), _ => None });
        draw_pal_index.push(get(cpi).parse::<u32>().ok());
        draw_pal2_index.push(get(cpi2).parse::<u32>().ok());
        draws.push(WorldDraw { index: i, geom: g, effect: hex(get(ce)), viewport: hex(get(cv)), world, src });
    }
    let cam = std::fs::read_to_string(dir.join("camera.txt")).map_err(|e| format!("camera.txt: {e}"))?;
    let mut views: HashMap<u64, [f32; 16]> = HashMap::new();
    let mut projs: HashMap<u64, [f32; 16]> = HashMap::new();
    let mut cur = 0u64;
    for l in cam.lines().map(str::trim) {
        if let Some(r) = l.strip_prefix("viewport ") {
            cur = hex(r.split_whitespace().next().unwrap_or(""));
        } else if let Some(r) = l.strip_prefix("view ") {
            let v: Vec<f32> = r.replace('|', " ").split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if let Ok(a) = <[f32; 16]>::try_from(v.as_slice()) {
                views.insert(cur, a);
            }
        } else if let Some(r) = l.strip_prefix("proj ") {
            let v: Vec<f32> = r.replace('|', " ").split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if let Ok(a) = <[f32; 16]>::try_from(v.as_slice()) {
                projs.insert(cur, a);
            }
        }
    }
    let blobs = dir.join("blobs");
    let mut meshes: HashMap<u64, Mesh> = HashMap::new();
    let mut raw: HashMap<u64, (Vec<u8>, Vec<u32>)> = HashMap::new();
    let mut max_nv = 0;
    let mut failed = Vec::new();
    let mut geoms: Vec<u64> = draws.iter().map(|d| d.geom).collect();
    geoms.sort_unstable();
    geoms.dedup();
    for g in geoms {
        let h = draws.iter().zip(&draw_ghash).find(|(d, _)| d.geom == g).map_or(0, |(_, h)| *h);
        let shared = dir.parent().map(|p| p.join("blobs"));
        let read = |ext: &str| std::fs::read(blobs.join(format!("{g:x}.{ext}"))).ok().or_else(|| std::fs::read(shared.as_ref()?.join(format!("{g:x}-{h:x}.{ext}"))).ok());
        let (mesh_blob, vb_blob) = (read("mesh"), read("vb"));
        if let (Some(m), Some(v)) = (&mesh_blob, &vb_blob) {
            max_nv = max_nv.max(max_meshlet_vertices(m));
            if let Ok(i) = expand_meshlets(m) {
                raw.insert(g, (v.clone(), i));
            }
        }
        let r = match (mesh_blob, vb_blob, read("ib")) {
            (Some(m), Some(v), _) => decode_meshlets(&m, &v),
            (None, Some(v), Some(i)) => decode_indexed(&i, elems.get(&g).copied().unwrap_or(2), &v),
            _ => Err("no blobs".into()),
        };
        match r {
            Ok(m) => {
                meshes.insert(g, m);
            }
            Err(e) => failed.push(format!("{g:#x}: {e}")),
        }
    }
    let mut palettes: HashMap<u64, Vec<f32>> = HashMap::new();
    // Per-draw dumps: one palette per draw by index (keys 1 << 63 | index, so the
    // per-pointer files and these never collide).
    let floats = |b: Vec<u8>| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
    for (i, ix) in draw_pal_index.iter().enumerate() {
        if let Some(ix) = ix {
            let key = 1u64 << 63 | u64::from(*ix);
            if !palettes.contains_key(&key)
                && let Ok(b) = std::fs::read(blobs.join(format!("pal-{ix}.bin")))
            {
                palettes.insert(key, floats(b));
            }
            if let Some(p) = draw_palette.get_mut(i) {
                *p = key;
            }
        }
    }
    let _ = &draw_pal2_index;
    // With palbuf.bin, a skinned draw's palette is its slice
    // [curIndex, curIndex + curCount) of the title's buffer (key 1 << 62 | idx << 20 | n).
    if let Ok(pb) = std::fs::read(blobs.join("palbuf.bin")) {
        let all = floats(pb);
        // Newer captures: palbuf.tsv says where each (cur_idx, cur_n) starts (a window
        // of the ring buffer); without it palbuf.bin starts at bone 0.
        let firsts: HashMap<(u32, u32), u32> = std::fs::read_to_string(blobs.join("palbuf.tsv"))
            .map(|t| {
                t.lines()
                    .skip(1)
                    .filter_map(|l| {
                        let f: Vec<u32> = l.split('\t').filter_map(|v| v.parse().ok()).collect();
                        (f.len() == 3).then(|| ((f[0], f[1]), f[2]))
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (i, (idx, n)) in draw_skin.iter().enumerate() {
            let start = if firsts.is_empty() { Some(*idx) } else { firsts.get(&(*idx, *n)).copied() };
            let Some(start) = start else { continue };
            let (a, b) = (12 * start as usize, 12 * (start + *n) as usize);
            if *n == 0 || b > all.len() {
                continue;
            }
            let key = 1u64 << 62 | u64::from(*idx) << 20 | u64::from(*n);
            palettes.entry(key).or_insert_with(|| all[a..b].to_vec());
            if let Some(p) = draw_palette.get_mut(i) {
                *p = key;
            }
        }
    }
    for p in draw_palette.iter().copied().filter(|p| *p != 0 && *p >> 63 == 0) {
        if !palettes.contains_key(&p)
            && let Ok(b) = std::fs::read(blobs.join(format!("pal-{p:x}.bin")))
        {
            palettes.insert(p, b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
        }
    }
    let mut inst_blocks: HashMap<u32, Vec<f32>> = HashMap::new();
    for b in draw_inst.iter().flatten() {
        if !inst_blocks.contains_key(b)
            && let Ok(v) = std::fs::read(blobs.join(format!("inst-{b}.bin")))
        {
            inst_blocks.insert(*b, floats(v));
        }
    }
    let mut from_obj = 0;
    if meshes.is_empty()
        && let Ok(text) = std::fs::read_to_string(dir.join("scene-local.obj"))
    {
        failed.clear();
        for (g, m) in parse_local_obj(&text) {
            if m.pos.iter().all(|p| p.iter().all(|v| v.abs() < 1e4)) {
                meshes.insert(g, m);
                from_obj += 1;
            } else {
                failed.push(format!("{g:#x}: scene-local.obj positions past 1e4"));
            }
        }
    }
    Ok(LoadedFrame { draws, views, projs, meshes, failed, from_obj, draw_palette, draw_stage, draw_caller, draw_ghash, draw_skinned, draw_inst, inst_blocks, palettes, raw, max_meshlet_vertices: max_nv })
}

/// The objects `g<geometry>` of a scene-local.obj (v, vt, vn, f a/b/c).
pub fn parse_local_obj(text: &str) -> Vec<(u64, Mesh)> {
    let mut out: Vec<(u64, Mesh)> = Vec::new();
    let mut base = 0u32;
    for l in text.lines() {
        let mut w = l.split_whitespace();
        let Some(kind) = w.next() else { continue };
        let nums = |w: std::str::SplitWhitespace| w.filter_map(|x| x.parse::<f32>().ok()).collect::<Vec<_>>();
        match kind {
            "o" => {
                if let Some((_, m)) = out.last() {
                    base += m.pos.len() as u32;
                }
                let g = w.next().and_then(|n| n.strip_prefix('g')).and_then(|h| u64::from_str_radix(h, 16).ok()).unwrap_or(0);
                out.push((g, Mesh::default()));
            }
            "v" | "vn" | "vt" if !out.is_empty() => {
                let v = nums(w);
                let m = &mut out.last_mut().unwrap().1;
                match (kind, v.as_slice()) {
                    ("v", [a, b, c, ..]) => m.pos.push([*a, *b, *c]),
                    ("vn", [a, b, c, ..]) => m.normal.push([*a, *b, *c]),
                    ("vt", [a, b, ..]) => m.uv.push([*a, *b]),
                    _ => {}
                }
            }
            "f" if !out.is_empty() => {
                let i: Vec<u32> = w.filter_map(|x| x.split('/').next()?.parse::<u32>().ok()).collect();
                if let [a, b, c] = i[..] {
                    out.last_mut().unwrap().1.tris.push([a - 1 - base, b - 1 - base, c - 1 - base]);
                }
            }
            _ => {}
        }
    }
    out
}

/// The frame list our pass draws: one record per draw with a plausible
/// transform (camera-relative to its own viewport), keyed by geometry.
pub fn frame_list(f: &LoadedFrame) -> FrameList {
    let cameras = f.views.iter().filter_map(|(vp, view)| Some(RCamera { viewport: *vp, view: *view, proj: *f.projs.get(vp)? })).collect();
    // The main view: 16:9 and the most draws of the whole frame.
    let mut count: HashMap<u64, usize> = HashMap::new();
    for d in &f.draws {
        *count.entry(d.viewport).or_default() += 1;
    }
    let wide = |p: &[f32; 16]| p[0] > 0.0 && (p[5] / p[0] - 16.0 / 9.0).abs() < 0.01;
    let main = f.projs.iter().filter(|(_, p)| wide(p)).max_by_key(|(vp, _)| (count.get(vp).copied().unwrap_or(0), **vp)).map(|(vp, _)| *vp);
    let mut list = FrameList { frame: 0, draws: Vec::new(), cameras, main, ..Default::default() };
    for (i, d) in f.draws.iter().enumerate() {
        // An instanced draw becomes one record per instance (at most 256 offline).
        if let Some(block) = f.draw_inst.get(i).copied().flatten().and_then(|b| f.inst_blocks.get(&b))
            && let Some(view) = f.views.get(&d.viewport)
            && f.meshes.contains_key(&d.geom)
        {
            let origin = camera_position(view);
            for r in block.chunks_exact(16).take(256) {
                let w = xnw_world::instance_world(r.try_into().unwrap(), origin);
                let world = [w[0], w[1], w[2], 0.0, w[3], w[4], w[5], 0.0, w[6], w[7], w[8], 0.0, w[9], w[10], w[11], 1.0];
                list.draws.push(DrawRecord { geom: GeomKey { mesh_at: d.geom, hash: f.draw_ghash.get(i).copied().unwrap_or(0), ..Default::default() }, world, viewport: d.viewport, effect: d.effect, albedo: Albedo::Flat(xnw_world::flat_colour(d.effect)), palette: xnw_world::NO_PALETTE, stage: f.draw_stage.get(i).copied().unwrap_or(0), inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: xnw_world::Material::default(), state: None, remap: xnw_world::NO_REMAP, ped: false });
            }
            continue;
        }
        let Some(w) = d.world else { continue };
        let world: [f32; 16] = w[..16].try_into().unwrap();
        let Some(m) = f.meshes.get(&d.geom) else { continue };
        if !plausible(&world) {
            continue;
        }
        // A skinned mesh with its palette in the dump: its own key (skinned per draw).
        let skinned_model = f.draw_skinned.get(i).copied().flatten() != Some(false);
        let pal = f.draw_palette.get(i).copied().filter(|p| skinned_model && !m.skin.is_empty() && f.palettes.contains_key(p)).unwrap_or(0);
        list.draws.push(DrawRecord { geom: GeomKey { mesh_at: d.geom, vb_at: pal, hash: f.draw_ghash.get(i).copied().unwrap_or(0), ..Default::default() }, world, viewport: d.viewport, effect: d.effect, albedo: Albedo::Flat(xnw_world::flat_colour(d.effect)), palette: xnw_world::NO_PALETTE, stage: f.draw_stage.get(i).copied().unwrap_or(0), inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: xnw_world::Material::default(), state: None, remap: xnw_world::NO_REMAP, ped: false });
    }
    list
}

/// Our sun from the timecycle render copy: direction the final
/// light direction +0x1630 (sun or moon; else the sun direction +0x1610),
/// colour light_dir_col (+0x0) x light_dir_mult (+0xC), as hue (largest
/// channel 1) times a brightness mult / (mult + 4) * 1.25 (14.75 by day: 0.98;
/// our 8-bit target has no exposure of its own [I]); ambient
/// 0.15 + 0.1 * ambient_sky_intensity (+0x330), clamped to 0.1..0.45 [I].
/// [dir xyz, ambient, colour rgb, 1].
pub fn sun_of(tc: &[u8]) -> Option<[f32; 8]> {
    if tc.len() < 0x1640 {
        return None;
    }
    let f = |o: usize| f32::from_bits(le32(tc, o));
    let unit = |o: usize| {
        let v = [f(o), f(o + 4), f(o + 8)];
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        (l.is_finite() && (0.5..1.5).contains(&l)).then(|| v.map(|x| x / l))
    };
    // A night capture: +0x1610 is the sun below the horizon (z -0.72),
    // +0x1630 the moon (z +0.64): the light is the final direction.
    let dir = unit(0x1630).or_else(|| unit(0x1610))?;
    let col = [f(0x0), f(0x4), f(0x8)];
    let mult = Some(f(0xC)).filter(|m| m.is_finite() && *m > 0.0).unwrap_or(1.0);
    if !col.iter().all(|c| c.is_finite() && *c >= 0.0) {
        return None;
    }
    let top = col.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    let bright = mult / (mult + 4.0) * 1.25;
    let amb = Some(f(0x330)).filter(|a| a.is_finite()).map_or(0.25, |a| (0.15 + 0.1 * a).clamp(0.1, 0.45));
    Some([dir[0], dir[1], dir[2], amb, col[0] / top * bright, col[1] / top * bright, col[2] / top * bright, 1.0])
}

/// The lighting of our pass from the timecycle render copy (and
/// the sky stand-in's colours for the ambient); `log`: the table of values
/// read is logged once.
pub fn light_of(tc: &[u8], sky: Option<&xnw_world::Sky>, log: bool) -> xnw_world::lighting::Lighting {
    let floats: Vec<f32> = tc.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let sky = sky.map(|s| ([s.zenith[0], s.zenith[1], s.zenith[2]], [s.horizon[0], s.horizon[1], s.horizon[2]]));
    // `ourlight=on` (LITE1): our own pass's light from the same values.
    let (l, table) = if xnw_world::ourlight::asked() { xnw_world::lighting::Lighting::ours(&floats, sky) } else { xnw_world::lighting::Lighting::from_timecycle(&floats, sky) };
    static ONCE: std::sync::Once = std::sync::Once::new();
    if log {
        ONCE.call_once(|| {
            let rows: Vec<String> = table.iter().map(|(n, i, v, d, tag)| format!("{n} [{i}] = {v}{} ({tag})", if *d { " (default)" } else { "" })).collect();
            xnw_core::info!("world render: lighting: timecycle values: {}", rows.join("; "));
            xnw_core::info!("{}", l.line());
        });
    }
    l
}

/// The sky stand-in from the timecycle render copy [I colours]:
/// zenith = light colour x Rayleigh beta (TC[395..397]) x saturate(sun.z + 0.1)
/// plus a night floor (a night capture rendered near black without it); horizon = lerp(zenith, light colour x 0.9, 0.6); sun
/// disc radius 0.27 deg x sky_sun_disc_size (+0x72C) while the sun is up;
/// moon disc (+0x1620) while the sun is down; fog towards the horizon colour
/// over far_clip (+0x14C4) metres (fog variables are not cracked).
pub fn sky_of(tc: &[u8]) -> Option<xnw_world::Sky> {
    let light = sun_of(tc)?;
    let f = |o: usize| f32::from_bits(le32(tc, o));
    let unit = |o: usize| {
        let v = [f(o), f(o + 4), f(o + 8)];
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        (l.is_finite() && (0.5..1.5).contains(&l)).then(|| v.map(|x| x / l))
    };
    let sun = unit(0x1610).unwrap_or([light[0], light[1], light[2]]);
    let col = [light[4], light[5], light[6]];
    let beta = [f(0x62C), f(0x630), f(0x634)];
    let beta = if beta.iter().all(|b| b.is_finite() && (0.0..=1.0).contains(b)) && beta.iter().any(|b| *b > 0.0) { beta } else { [0.1601, 0.3741, 0.9134] };
    let day = (sun[2] + 0.1).clamp(0.0, 1.0);
    let zenith: [f32; 3] = std::array::from_fn(|i| col[i] * beta[i] * day + [0.04, 0.06, 0.12][i]);
    let horizon: [f32; 3] = std::array::from_fn(|i| (zenith[i] + (col[i] * 0.9 * day.max(0.15) - zenith[i]) * 0.6).max([0.10, 0.11, 0.14][i]));
    let disc = Some(f(0x72C)).filter(|d| d.is_finite() && *d > 0.0 && *d < 20.0).unwrap_or(1.0);
    let cos_disc = (0.27 * disc).to_radians().cos();
    let up = sun[2] > -0.05;
    let moon = unit(0x1620).filter(|m| !up && m[2] > 0.0);
    let far = Some(f(0x14C4)).filter(|d| d.is_finite() && *d > 100.0).unwrap_or(10_000.0);
    Some(xnw_world::Sky {
        sun: [sun[0], sun[1], sun[2], cos_disc],
        sun_col: if up { [col[0], col[1], col[2], 1.0] } else { [0.0; 4] },
        zenith: [zenith[0], zenith[1], zenith[2], 0.0],
        horizon: [horizon[0], horizon[1], horizon[2], 0.0],
        moon: moon.map_or([0.0; 4], |m| [m[0], m[1], m[2], 1.0]),
        fog: [horizon[0], horizon[1], horizon[2], 1.0 / far],
    })
}

/// Grid cells per side at most (the map is sampled every k texels).
pub const WATER_GRID: u32 = 512;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterMap {
    pub map: u64,
    pub data: u64,
    pub w: u32,
    pub h: u32,
    pub aabb: [f32; 4],
    pub s: [f32; 4],
}

/// The water grid of a map: every k-th texel (at most WATER_GRID cells a
/// side), absolute positions x = (u + 0.5 - ox) / sx, y likewise, z from the
/// texel; a cell is kept only when its four corners are water (non-zero).
pub fn water_grid(m: &WaterMap, texels: &[u16]) -> xnw_world::WaterGrid {
    let (w, h) = (m.w as usize, m.h as usize);
    let k = (m.w.max(m.h).div_ceil(WATER_GRID)).max(1) as usize;
    let (gw, gh) = ((w - 1) / k + 1, (h - 1) / k + 1);
    let at = |gx: usize, gy: usize| texels.get((gy * k).min(h - 1) * w + (gx * k).min(w - 1)).copied().unwrap_or(0);
    let mut grid = xnw_world::WaterGrid { id: m.id(), ..Default::default() };
    let mut index: HashMap<(usize, usize), u32> = HashMap::new();
    for gy in 0..gh.saturating_sub(1) {
        for gx in 0..gw.saturating_sub(1) {
            let corners = [(gx, gy), (gx + 1, gy), (gx + 1, gy + 1), (gx, gy + 1)];
            if corners.iter().any(|(x, y)| at(*x, *y) == 0) {
                continue;
            }
            let ids: Vec<u32> = corners
                .iter()
                .map(|&(x, y)| {
                    *index.entry((x, y)).or_insert_with(|| {
                        let (u, v) = ((x * k).min(w - 1) as f32, (y * k).min(h - 1) as f32);
                        grid.vertices.push([(u + 0.5 - m.s[2]) / m.s[0], (v + 0.5 - m.s[3]) / m.s[1], 1600.0 * f32::from(at(x, y)) / 65535.0 - 800.0]);
                        (grid.vertices.len() - 1) as u32
                    })
                })
                .collect();
            grid.indices.extend([ids[0], ids[1], ids[2], ids[0], ids[2], ids[3]]);
        }
    }
    grid
}

/// A dumped water map (the first in `dir`).
pub fn load_water(dir: &std::path::Path) -> Option<(WaterMap, Vec<u16>)> {
    let txt = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "txt"))?;
    let text = std::fs::read_to_string(&txt).ok()?;
    let field = |k: &str| text.lines().find_map(|l| l.strip_prefix(k).map(str::trim).map(str::to_string));
    let list4 = |s: String| -> Option<[f32; 4]> {
        let v: Vec<f32> = s.trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|x| x.trim().parse().ok()).collect();
        v.try_into().ok()
    };
    let m = WaterMap { map: 1, data: 1, w: field("width ")?.parse().ok()?, h: field("height ")?.parse().ok()?, aabb: list4(field("aabb ")?)?, s: list4(field("transform ")?)? };
    let b = std::fs::read(txt.with_extension("bin")).ok()?;
    Some((m, b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()))
}

/// The water tint: a deep colour lit by the light colour, alpha 0.8 [I].
pub fn water_tint(sun: Option<[f32; 8]>) -> [f32; 4] {
    let c = sun.map_or([1.0; 3], |s| [s[4], s[5], s[6]]);
    [0.05 + 0.1 * c[0], 0.16 + 0.2 * c[1], 0.22 + 0.25 * c[2], 0.8]
}

/// A live list dumped by [`dump_list`], read back for the pass.
/// The decoded render state of a draw as one list.tsv field (the replay
/// draws with the state the live pass used; "-" = our defaults):
/// depth_enable,depth_write,func,bias,clamp_bits,slope_bits,blend0..7,cull,front_ccw
/// (the last two since the cull switch: 14 fields read as cull not recorded).
pub fn rstate_text(s: Option<xnw_world::RenderState>) -> String {
    match s {
        None => "-".into(),
        Some(s) => format!("{},{},{},{},{:#x},{:#x},{},{},{}", u8::from(s.depth_enable), u8::from(s.depth_write), s.depth_func, s.depth_bias, s.bias_clamp, s.slope_bias, s.blend.iter().map(u8::to_string).collect::<Vec<_>>().join(","), s.cull, u8::from(s.front_ccw)),
    }
}

pub fn rstate_parse(v: &str) -> Option<xnw_world::RenderState> {
    let f: Vec<&str> = v.split(',').collect();
    if f.len() != 14 && f.len() != 16 {
        return None;
    }
    let n = |i: usize| f[i].trim().parse::<i64>().ok();
    let h = |i: usize| u32::from_str_radix(f[i].trim().trim_start_matches("0x"), 16).ok();
    let mut blend = [0u8; 8];
    for (k, b) in blend.iter_mut().enumerate() {
        *b = n(6 + k)? as u8;
    }
    Some(xnw_world::RenderState { depth_enable: n(0)? != 0, depth_write: n(1)? != 0, depth_func: n(2)? as u8, depth_bias: n(3)? as i32, bias_clamp: h(4)?, slope_bias: h(5)?, blend, cull: if f.len() == 16 { n(14)? as u8 } else { 0 }, front_ccw: f.len() == 16 && n(15)? != 0 })
}

pub fn load_list(fdir: &std::path::Path) -> Result<(FrameList, HashMap<GeomKey, (Vec<u8>, Vec<u32>)>), String> {
    load_list_with(fdir, &|_| false)
}

/// [`load_list`] without reading the blobs of the keys `have` already holds
/// (`xnw burst` loads a sequence's geometries once).
pub fn load_list_with(fdir: &std::path::Path, have: &dyn Fn(&GeomKey) -> bool) -> Result<(FrameList, HashMap<GeomKey, (Vec<u8>, Vec<u32>)>), String> {
    let text = std::fs::read_to_string(fdir.join("list.tsv")).map_err(|e| e.to_string())?;
    let shared = fdir.parent().ok_or("no parent")?.join("blobs");
    let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0);
    let mut list = FrameList::default();
    let mut geoms = HashMap::new();
    for l in text.lines().skip(1) {
        let f: Vec<&str> = l.split('\t').collect();
        if f.len() < 13 {
            continue;
        }
        let geom = GeomKey { mesh_at: hex(f[0]), mesh_len: f[1].parse().unwrap_or(0), vb_at: hex(f[2]), vb_len: f[3].parse().unwrap_or(0), ib: f[4].parse().unwrap_or(0), hash: hex(f[5]) };
        let w: Vec<f32> = f[12].replace('|', " ").split_whitespace().filter_map(|x| x.parse().ok()).collect();
        let Ok(world) = <[f32; 16]>::try_from(w.as_slice()) else { continue };
        if !have(&geom)
            && let std::collections::hash_map::Entry::Vacant(v) = geoms.entry(geom)
        {
            let stem = shared.join(format!("k-{:x}-{:x}", geom.mesh_at, geom.hash));
            if let (Ok(vb), Ok(idx)) = (std::fs::read(stem.with_extension("vb")), std::fs::read(stem.with_extension("idx"))) {
                v.insert((vb, idx.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()));
            }
        }
        list.draws.push(DrawRecord { geom, world, viewport: hex(f[6]), stage: f[7].parse().unwrap_or(0), palette: f[8].parse().unwrap_or(u32::MAX), inst_first: f[9].parse().unwrap_or(0), inst_count: f[10].parse().unwrap_or(0), bone_first: f.get(13).and_then(|v| v.parse().ok()).unwrap_or(0), bone_count: f.get(14).and_then(|v| v.parse().ok()).unwrap_or(0), effect: hex(f[11]), albedo: Albedo::Flat(xnw_world::flat_colour(hex(f[11]))), material: xnw_world::Material { albedo_desc: f.get(15).map_or(0, |v| hex(v)), alpha_clip: f.get(16) == Some(&"1"), layers: std::array::from_fn(|k| f.get(19 + k).map_or(0, |v| hex(v))), stipple: f.get(23).map_or(0, |v| hex(v) as u8), state_index: f.get(24).map(|v| { let s: Vec<u16> = v.split('/').map(|x| x.parse().unwrap_or(0xFFFF)).collect(); [s.first().copied().unwrap_or(0xFFFF), s.get(1).copied().unwrap_or(0xFFFF), s.get(2).copied().unwrap_or(0xFFFF)] }).unwrap_or([0xFFFF; 3]), ..Default::default() }, state: f.get(25).and_then(|v| rstate_parse(v)), remap: f.get(17).and_then(|v| v.parse().ok()).unwrap_or(xnw_world::NO_REMAP), ped: f.get(18) == Some(&"1") });
    }
    let cams = std::fs::read_to_string(fdir.join("camera.txt")).map_err(|e| e.to_string())?;
    let (mut vp, mut view) = (0u64, None);
    for l in cams.lines().map(str::trim) {
        let m = |r: &str| -> Option<[f32; 16]> { r.replace('|', " ").split_whitespace().filter_map(|x| x.parse().ok()).collect::<Vec<f32>>().try_into().ok() };
        if let Some(r) = l.strip_prefix("viewport ") {
            vp = hex(r.split_whitespace().next().unwrap_or(""));
        } else if let Some(r) = l.strip_prefix("view ") {
            view = m(r);
        } else if let Some(r) = l.strip_prefix("proj ")
            && let (Some(v), Some(p)) = (view, m(r))
        {
            list.cameras.push(RCamera { viewport: vp, view: v, proj: p });
        }
    }
    for i in 0.. {
        let Ok(b) = std::fs::read(fdir.join(format!("pal-{i}.bin"))) else { break };
        list.palettes.push(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
    }
    // `XNW_REPLAY_TIMECYCLE=<file>` (offline, LITE1): another frame's timecycle over
    // this frame's draws (a sun-on light on a night capture's street).
    let tc_file = std::env::var_os("XNW_REPLAY_TIMECYCLE").map_or_else(|| fdir.join("timecycle.bin"), std::path::PathBuf::from);
    if let Ok(b) = std::fs::read(tc_file) {
        list.sun = sun_of(&b);
        list.sky = sky_of(&b);
        list.light = Some(light_of(&b, list.sky.as_ref(), false));
    }
    // The sequence's water map (water\<id>.txt/.bin), the first one.
    if let Some(wdir) = fdir.parent().map(|p| p.join("water"))
        && let Some(m) = load_water(&wdir)
        && let Some(cam) = list.main_camera()
    {
        list.water = Some(xnw_world::Water { grid: std::sync::Arc::new(water_grid(&m.0, &m.1)), origin: xnw_world::camera_position(&cam.view), tint: water_tint(list.sun) });
    }
    if let Ok(b) = std::fs::read(fdir.join("instances.bin")) {
        list.instances = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    }
    // Captures before a0e3dcd hold non-transform
    // "instances" (3x3 entries of 1e29..1e37 from the buffer-view fallback):
    // Only the bad instance worlds go (zeroed: a collapsed instance); the rest of
    // their batch draws (a whole batch dropped took palm crowns with it).
    let mut bad = 0usize;
    for w in list.instances.chunks_exact_mut(12) {
        if !(w[..9].iter().all(|v| v.is_finite() && v.abs() < 1e3) && w[9..].iter().all(|v| v.is_finite() && v.abs() < 1e6)) {
            w.fill(0.0);
            bad += 1;
        }
    }
    if bad > 0 {
        eprintln!("  instances: {bad} non-transform instance worlds collapsed");
    }
    // Rigid parts' bones (rigid.tsv), composed by the caller with `--rigid`.
    if let Ok(t) = std::fs::read_to_string(fdir.join("rigid.tsv")) {
        for l in t.lines() {
            let Some((i, v)) = l.split_once('\t') else { continue };
            let f: Vec<f32> = v.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if let (Ok(i), Ok(b)) = (i.parse::<u32>(), <[f32; 12]>::try_from(f.as_slice())) {
                list.rigid.push((i, b));
            }
        }
    }
    // The frame's terrain tile records, when dumped.
    list.terrain = load_terrain(fdir).map(std::sync::Arc::new);
    list.imposters = crate::imposter::load_imposters(fdir).map(std::sync::Arc::new);
    Ok((list, geoms))
}

impl WaterMap {
    /// The change key: map, data, size, transform.
    pub fn id(&self) -> u64 {
        fnv(&[self.map.to_le_bytes(), self.data.to_le_bytes(), (u64::from(self.w) << 32 | u64::from(self.h)).to_le_bytes(), (u64::from(self.s[0].to_bits()) << 32 | u64::from(self.s[1].to_bits())).to_le_bytes(), (u64::from(self.s[2].to_bits()) << 32 | u64::from(self.s[3].to_bits())).to_le_bytes()].concat())
    }
}

/// A dumped tile array: ours (terrain.bin + terrain.txt), else the older raw
/// `blobs\terrain.bin` (no heap base: no slots).
pub fn load_terrain(fdir: &std::path::Path) -> Option<Terrain> {
    let (raw, txt) = match std::fs::read(fdir.join("terrain.bin")) {
        Ok(b) => (b, std::fs::read_to_string(fdir.join("terrain.txt")).unwrap_or_default()),
        Err(_) => (std::fs::read(fdir.join("blobs").join("terrain.bin")).ok()?, String::new()),
    };
    let field = |k: &str| txt.lines().find_map(|l| l.strip_prefix(k)).map(str::trim).and_then(|v| if let Some(h) = v.strip_prefix("0x") { u64::from_str_radix(h, 16).ok() } else { v.parse().ok() });
    let n = (raw.len() / TILE_BYTES) as u32;
    Some(parse(field("cap ").unwrap_or(u64::from(n)) as u32, field("count ").unwrap_or(u64::from(n)) as u32, raw, field("heap_base ").unwrap_or(0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state text keeps the cull mode and winding; a 14-field text
    /// (captures before the cull switch) reads as cull not recorded.
    #[test]
    fn rstate_text_keeps_cull() {
        let s = xnw_world::RenderState { depth_enable: true, depth_write: true, depth_func: 7, depth_bias: -8, bias_clamp: 0, slope_bias: 1.5f32.to_bits(), blend: [0, 2, 1, 1, 2, 1, 1, 15], cull: 3, front_ccw: true };
        let text = rstate_text(Some(s));
        assert_eq!(text, "1,1,7,-8,0x0,0x3fc00000,0,2,1,1,2,1,1,15,3,1");
        assert_eq!(rstate_parse(&text), Some(s));
        assert_eq!(rstate_parse("1,1,7,0,0x0,0x0,0,2,1,1,2,1,1,15").map(|s| (s.cull, s.front_ccw, s.depth_func)), Some((0, false, 7)));
        assert_eq!(rstate_parse("-"), None);
    }
}
