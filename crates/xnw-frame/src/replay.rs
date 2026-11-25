//! The replay driver of our world pass: `xnw world-render` (a probe frame,
//! a live list or a `--seq` directory to PNG) and `xnw world-probe-decode`.
//! Moved verbatim from apps/xnw/src/worldprobe.rs (59afddb).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use xnw_world::GeomKey;

use crate::audit;
use crate::decode::Mesh;
use crate::load::{LoadedFrame, frame_list, load_frame, load_list};
use crate::world::{plausible, rigid_world, write_world};

/// `xnw world-probe-decode <frame dir>`: decodes a probe frame again from
/// its files (draws.tsv, camera.txt, blobs\<geometry>.{mesh,vb,ib}) and
/// rewrites scene.obj / scene-local.obj; no title.
pub fn decode_cli(args: &[String]) -> Result<(), String> {
    let dir = std::path::Path::new(args.first().ok_or("usage: xnw world-probe-decode <frame dir>")?);
    let LoadedFrame { mut draws, views, meshes, failed, max_meshlet_vertices, .. } = load_frame(dir)?;
    let refs: HashMap<u64, &Mesh> = meshes.iter().map(|(g, m)| (*g, m)).collect();
    let st = write_world(dir, 0, &mut draws, &refs, &views).map_err(|e| e.to_string())?;
    let with_n = meshes.values().filter(|m| !m.normal.is_empty()).count();
    let with_uv = meshes.values().filter(|m| !m.uv.is_empty()).count();
    println!("{} draws, {} geometries decoded ({} failed; normals {with_n}, uv {with_uv}; largest meshlet {max_meshlet_vertices} vertices); transforms: {} ({} borrowed, {} without); {} vertices written; world bounds {}", draws.len(), meshes.len(), failed.len(), st.sources, st.borrowed, st.missing, st.vertices, st.bounds.show());
    for f in failed.iter().take(8) {
        println!("  decode failed: {f}");
    }
    Ok(())
}

/// `xnw world-render <frame dir> --adapter warp|hw [--out <png>] [--size WxH]`:
/// draws a probe frame's main view with our D3D12 pass
/// (`xnw_world::Pass`): filled triangles, depth, Lambert,
/// a flat colour per effect; read back to a PNG. WARP only for now.
pub fn render_cli(args: &[String]) -> Result<(), String> {
    // `--write-shader <file>`: the built-in world pass shader as a file to edit.
    if let Some(f) = args.iter().position(|a| a == "--write-shader").and_then(|i| args.get(i + 1)) {
        std::fs::write(f, xnw_world::RAW_SHADER).map_err(|e| format!("{f}: {e}"))?;
        println!("{f}: the built-in world pass shader ({} bytes); draw with it: --shader {f}", xnw_world::RAW_SHADER.len());
        return Ok(());
    }
    let dir = std::path::Path::new(args.first().ok_or("usage: xnw world-render <frame dir | sequence dir --seq> --adapter warp|hw [--out <png>] [--shader <hlsl>] | --write-shader <hlsl>")?);
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    // `--skin-check [all] [--out-tsv f] [--skin-rules hex]`: the skin sanity of the ped (or every skinned) draws, CPU only.
    if let Some(i) = args.iter().position(|a| a == "--skin-check") {
        let (mut list, mut geoms) = load_list(dir)?;
        audit::skin_args(args, dir, &mut list, &mut geoms)?;
        if let Some(n) = opt("--pack-probe").and_then(|v| v.parse().ok()) {
            return audit::pack_probe(dir, &list, n);
        }
        let tsv = PathBuf::from(opt("--out-tsv").unwrap_or_else(|| "skin-check.tsv".into()));
        return audit::skin_check_cli(&list, &geoms, args.get(i + 1).map(String::as_str) != Some("all"), skin_rules_arg(args), &tsv);
    }
    let t_setup = Instant::now();
    let (host, mut pass) = pass_from_args(args)?;
    // `--jitter`: the projection as captured, the title's TAA jitter kept (default: dropped).
    xnw_world::keep_jitter(args.iter().any(|a| a == "--jitter"));
    // `--bridgeps on|<hash>+<hash>` or `XNW_WORLD_RENDER=...,bridgeps=...`: those rows by the title's own programs in our pass.
    let bridgeps = opt("--bridgeps").map_or_else(|| xnw_world::bridge::parse_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default()), |v| xnw_world::bridge::parse_switch(&format!("bridgeps={v}")));
    if !bridgeps.is_empty() {
        return crate::bridge_ps::bridgeps_cli(args, dir, &host, &mut pass, &bridgeps);
    }
    // `--bridge <list index>`: one draw through the title's own programs, against our albedo.
    if let Some(k) = opt("--bridge") {
        return crate::bridge::cli(args, dir, &k, &host, &mut pass);
    }
    // `--hair-bridge`: the frame's strand draws through the title's vertex program (depth only).
    if args.iter().any(|a| a == "--hair-bridge") {
        return crate::hair::bridge_cli(args, dir, &host, (pass.width, pass.height));
    }
    if let Some(side) = opt("--bridge-audit") {
        return crate::bridge::audit_cli(args, dir, &side, &host, &mut pass);
    }
    if args.iter().any(|a| a == "--seq") {
        // Every frame-N of a sequence dump, to <dir>\seq-<N>.png (one pass, geometry kept by key).
        let mut frames: Vec<(u64, PathBuf)> = std::fs::read_dir(dir).map_err(|e| e.to_string())?.flatten().filter_map(|e| Some((e.file_name().to_str()?.strip_prefix("frame-")?.parse().ok()?, e.path()))).collect();
        frames.sort();
        // `--out <dir>` puts the pictures there (CI renders read-only golden captures).
        let out_dir = opt("--out").map_or_else(|| dir.to_path_buf(), PathBuf::from);
        std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
        // `--audit`: the geometry audit instead of the pictures.
        if args.iter().any(|a| a == "--audit") {
            return audit::run(&frames, &out_dir, &mut pass);
        }
        // `--cuts [--cutfast] [--texms N]` (CUT1): the live screen budgets and
        // cut watch, one timing line per frame, no pictures.
        if args.iter().any(|a| a == "--cuts") {
            use xnw_world::{SCREEN_UPLOAD_MS, cuts::CutWatch, geometry::GeomBudget};
            pass.upload_ms_budget = Some(SCREEN_UPLOAD_MS);
            pass.geom.budget = GeomBudget::SCREEN;
            pass.cuts = CutWatch::new(true);
            pass.cuts.fast = args.iter().any(|a| a == "--cutfast");
            pass.tex.opts.frame_ms = opt("--texms").and_then(|v| v.parse().ok()).unwrap_or(0.0);
            println!("cuts replay: {} frames, fast {}, texture reads cap {} ms, textures {:?}", frames.len(), pass.cuts.fast, pass.tex.opts.frame_ms, pass.tex.opts.mode);
        }
        for (n, frame) in frames {
            render_one(args, &frame, &out_dir.join(format!("seq-{n:05}.png")), &mut pass, &host.caps.adapter, n)?;
        }
        return Ok(());
    }
    let out = opt("--out").map_or_else(|| dir.join("w4-d3d12.png"), PathBuf::from);
    let r = render_one(args, dir, &out, &mut pass, &host.caps.adapter, 0);
    eprintln!("  total {:.0} ms before teardown", t_setup.elapsed().as_secs_f64() * 1e3);
    r
}

/// `--skin-rules <hex>`: the skin rule bits (audit::SKIN_*; 0: the default maths).
pub fn skin_rules_arg(args: &[String]) -> u32 {
    args.iter().position(|a| a == "--skin-rules").and_then(|i| args.get(i + 1)).and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()).unwrap_or(0)
}

pub fn render_one(args: &[String], dir: &std::path::Path, out: &std::path::Path, pass: &mut xnw_world::Pass, adapter: &str, frame_no: u64) -> Result<(), String> {
    use xnw_world::{HostGeometry, pack_vertices};
    if dir.join("list.tsv").exists() {
        // A live list (screen mode dump): drawn exactly as the live pass did.
        let (mut list, mut geoms) = load_list(dir)?;
        list.frame = frame_no;
        // Draw-time bones when the capture has them (before --only renumbers rows); the vehicle rules.
        audit::skin_args(args, dir, &mut list, &mut geoms)?;
        // `--preskin`: the pre-skinned ped rows drawn from the capture's dispatches.
        crate::preskin::preskin_arg(args, dir, &mut list, &mut geoms)?;
        // `--preskin-gpu[=check]`: the same rows skinned by our compute pass on the device.
        let preskin_gpu = crate::preskin_gpu::gpu_arg(args, dir, &mut list)?.is_some();
        if preskin_gpu {
            println!("  pre-skin pipeline made in {:.0} ms", pass.preskin_prepare()?);
        }
        list_options(args, &mut list);
        // A v2 frame gives its texels (pack origin or stored bytes) unless
        // `--tex=checker` or `--tex=off`; slots it has none for stay flat.
        let counts = std::sync::Arc::new([const { std::sync::atomic::AtomicU64::new(0) }; 3]);
        let packed = !args.iter().any(|a| a == "--tex=checker" || a == "--tex=off");
        match crate::pack::Pack::open(dir)?.filter(|_| packed) {
            Some(p) => {
                pass.tex.opts.mode = xnw_world::TexMode::Provided;
                pass.tex.set_provider(Some(p.texel_provider(counts.clone())));
            }
            None if pass.tex.opts.mode == xnw_world::TexMode::Provided => {
                pass.tex.opts.mode = xnw_world::TexMode::Checker;
                pass.tex.set_provider(None);
            }
            None => {}
        }
        let mut decode = |k: &GeomKey| -> Option<HostGeometry> {
            let (vb, idx) = geoms.get(k)?;
            Some(HostGeometry { vertices: Vec::new(), indices: idx.clone(), raw_vb: Some(vb.clone()) })
        };
        watchdog_arm();
        let t = Instant::now();
        let st = pass.draw(&list, &mut decode, true)?;
        if args.iter().any(|a| a == "--cuts") {
            watchdog_disarm();
            let ours = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "cuts frame {frame_no}: coverage {:.2}{}, upload {:.1} ms ({} geometries, {} KiB, {} pending), textures {:.1} ms, record {:.1} ms, GPU {:.1} ms, ours {ours:.1} ms{}",
                st.coverage,
                if st.held { " held" } else { "" },
                st.upload_ms,
                st.uploaded_geometries,
                st.uploaded_bytes >> 10,
                st.pending,
                st.tex_ms,
                (st.record_ms - st.upload_ms - st.tex_ms).max(0.0),
                st.gpu_ms,
                pass.cuts.line.take().map(|l| format!("
  {l}")).unwrap_or_default()
            );
            return Ok(());
        }
        let rgba = pass.read_rgba()?;
        watchdog_disarm();
        write_png(out, pass.width, pass.height, &rgba).map_err(|e| e.to_string())?;
        if preskin_gpu {
            println!("  {}", xnw_world::preskin_gpu::line(list.frame, &list.preskin, &pass.preskin_stats()));
        }
        println!("{} (live list): {} records, {} main view; drawn {} ({} triangles, {} skinned, {} instanced with {} instances, {} without geometry); {} geometries; record {:.1} ms, GPU {:.1} ms -> {}", dir.display(), list.draws.len(), list.main_draws().count(), st.draws, st.triangles, st.skinned, st.instanced, st.instances, st.skipped, geoms.len(), st.record_ms, st.gpu_ms, out.display());
        let ts = std::mem::take(&mut pass.tex.stats);
        println!("  textures ({:?}): {} draws textured, {} flat ({}), {} distinct slots of {} asked (table {}); layered draws {}", pass.tex.opts.mode, ts.bound_draws, ts.flat_draws, xnw_world::textures::TexTable::flat_split(&ts), ts.distinct, ts.demand, pass.tex.opts.slots, st.layered);
        if pass.tex.opts.mode == xnw_world::TexMode::Provided {
            let c: Vec<u64> = counts.iter().map(|n| n.load(std::sync::atomic::Ordering::Relaxed)).collect();
            println!("  pack texels: {} slots asked, {} from pack origin, {} from stored texels, {} flat (no view, no row or no texels)", c[0], c[1], c[2], c[0] - c[1] - c[2]);
        }
        if list.terrain.is_some() {
            println!("  {}", pass.terrain.log_lines()[0].trim_start_matches("world render: "));
        }
        if pass.imposters.opts.on {
            println!("  {}", pass.imposters.log_line().trim_start_matches("world render: "));
            // Double drawing: cards within 5 m of an instance the list draws (its worlds are relative to the main camera).
            if let (Some(im), Some(cam)) = (list.imposters.as_deref(), list.main_camera()) {
                use xnw_world::imposter;
                let p = xnw_world::camera_position(&cam.view).map(f64::from);
                let cards = imposter::select(im, p, Some(&imposter::Meshed::new(&list.instances, p)), &mut imposter::ImposterStats::default());
                let near: Vec<String> = cards.iter().take(8).map(|c| format!("{:.0} m ({:.0},{:.0},{:.0}) {:.1}x{:.1}", c.distance, c.centre[0], c.centre[1], c.centre[2], 2.0 * c.half_width, 2.0 * c.half_height)).collect();
                println!("  imposter cards within 5 m of a drawn instance: {} of {}; nearest: {}", imposter::over_meshes(&cards, &list.instances, p, 5.0), cards.len(), near.join(", "));
            }
        }
        // `--pick x,y[;x,y..]`: the draw under each pixel (an ID pass:
        // each draw writes its list index + 1; no sky, no title states).
        if let Some(points) = args.iter().position(|a| a == "--pick").and_then(|i| args.get(i + 1)) {
            let (sky, ids) = (pass.sky, true);
            (pass.sky, pass.id_mode) = (false, ids);
            let mut l2 = list.clone();
            l2.frame = frame_no + 1_000_000;
            watchdog_arm();
            let picked = pass.draw(&l2, &mut decode, true).and_then(|_| pass.read_rgba());
            watchdog_disarm();
            (pass.sky, pass.id_mode) = (sky, false);
            let px = picked?;
            for pt in points.split(';') {
                let Some((x, y)) = pt.split_once(',').and_then(|(a, b)| Some((a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?))) else { continue };
                if x >= pass.width || y >= pass.height {
                    continue;
                }
                let i = 4 * (y * pass.width + x) as usize;
                let id = u32::from(px[i]) | u32::from(px[i + 1]) << 8 | u32::from(px[i + 2]) << 16;
                match id.checked_sub(1).and_then(|k| list.draws.get(k as usize).map(|r| (k, r))) {
                    Some((k, r)) => {
                        let g = r.geom;
                        let word = geoms.get(&g).and_then(|(vb, _)| vb.get(..4)).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
                        println!(
                            "  pick {x},{y}: draw {k}: stage {}, effect {:#x}, mesh {:#x} ({} B), vb {:#x} ({} B, word0 {word:#x}), ib {:#x}, hash {:#x}, instanced {}, palette {}, ped {}, state {:?}, world t ({:.1} {:.1} {:.1})",
                            r.stage, r.effect, g.mesh_at, g.mesh_len, g.vb_at, g.vb_len, g.ib, g.hash, r.inst_count, if r.palette == u32::MAX { "-".into() } else { r.palette.to_string() }, r.ped, r.material.state_index, r.world[12], r.world[13], r.world[14]
                        );
                    }
                    None => println!("  pick {x},{y}: nothing (id {id:#x}: background, sky, water, terrain or a held decal)"),
                }
            }
        }
        return Ok(());
    }
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let (w, h) = (pass.width, pass.height);
    let t0 = Instant::now();
    let frame = load_frame(dir)?;
    let load_s = t0.elapsed().as_secs_f64();
    // Bisect filters: --caller title+0x.. / --not-caller ..., --src cache|bound|bound-buffer
    let (only_caller, not_caller, only_src) = (opt("--caller"), opt("--not-caller"), opt("--src"));
    let mut list = frame_list(&frame);
    if only_caller.is_some() || not_caller.is_some() || only_src.is_some() {
        let keep: Vec<bool> = frame.draws.iter().enumerate().filter(|(_, d)| d.world.is_some_and(|w| plausible(&w[..16].try_into().unwrap())) && frame.meshes.contains_key(&d.geom)).map(|(i, d)| {
            let caller = frame.draw_caller.get(i).map_or("", String::as_str);
            only_caller.as_deref().is_none_or(|c| c == caller) && not_caller.as_deref().is_none_or(|c| c != caller) && only_src.as_deref().is_none_or(|s| s == d.src.name())
        }).collect();
        let mut i = 0;
        list.draws.retain(|_| {
            i += 1;
            keep.get(i - 1).copied().unwrap_or(false)
        });
    }
    let mut list = list;
    list.frame = frame_no;
    let skin_on = !args.iter().any(|a| a == "--skin=off");
    // --skinned-only: the draws with a palette (bisect); --unskinned-only the rest.
    if args.iter().any(|a| a == "--skinned-only") {
        list.draws.retain(|r| r.geom.vb_at != 0);
    }
    if args.iter().any(|a| a == "--unskinned-only") {
        list.draws.retain(|r| r.geom.vb_at == 0);
    }
    let mut skinned = 0;
    let raw_path = args.iter().any(|a| a == "--raw");
    let mut list = list;
    if raw_path {
        // The shader decodes the title's vertex buffer; skinned draws get
        // their palette from the dump (one list palette per pointer).
        let mut index: HashMap<u64, u32> = HashMap::new();
        for r in &mut list.draws {
            if r.geom.vb_at != 0 {
                let pal = r.geom.vb_at;
                let n = list.palettes.len() as u32;
                r.palette = *index.entry(pal).or_insert(n);
                if r.palette == n {
                    list.palettes.push(frame.palettes[&pal].clone());
                }
                r.geom.vb_at = 0;
            }
        }
    }
    let mut decode = |k: &GeomKey| -> Option<HostGeometry> {
        if raw_path {
            let (vb, idx) = frame.raw.get(&k.mesh_at)?;
            return Some(HostGeometry { vertices: Vec::new(), indices: idx.clone(), raw_vb: Some(vb.clone()) });
        }
        let base = frame.meshes.get(&k.mesh_at)?;
        let owned = (k.vb_at != 0 && skin_on).then(|| base.skinned(frame.palettes.get(&k.vb_at)?)).flatten();
        skinned += usize::from(owned.is_some());
        if let Some(o) = &owned
            && args.iter().any(|a| a == "--skin-report")
        {
            let span = |v: &[[f32; 3]]| (0..3).map(|i| v.iter().map(|p| p[i]).fold(f32::MIN, f32::max) - v.iter().map(|p| p[i]).fold(f32::MAX, f32::min)).fold(0f32, f32::max);
            let bones = frame.palettes.get(&k.vb_at).map_or(0, |p| p.len() / 12);
            let max_idx = base.skin.iter().flat_map(|(i, w)| i.iter().zip(w).filter(|(_, w)| **w > 0).map(|(i, _)| *i)).max().unwrap_or(0);
            println!("skin {:#x} palette {:#x} ({bones} bones, max index {max_idx}): bind span {:.2} m, skinned span {:.2} m, {} vertices", k.mesh_at, k.vb_at, span(&base.pos), span(&o.pos), o.pos.len());
        }
        let m = owned.as_ref().unwrap_or(base);
        Some(HostGeometry { vertices: pack_vertices(&m.pos, &m.normal, &m.uv), indices: m.tris.iter().flatten().copied().collect(), raw_vb: None })
    };
    watchdog_arm();
    let first = pass.draw(&list, &mut decode, false)?;
    watchdog_disarm();
    watchdog_arm();
    let again = pass.draw(&list, &mut decode, true)?;
    if raw_path {
        skinned = first.skinned;
    }
    let rgba = pass.read_rgba()?;
    watchdog_disarm();
    let skin_meshes = frame.meshes.values().filter(|m| !m.skin.is_empty()).count();
    write_png(out, w, h, &rgba).map_err(|e| e.to_string())?;
    let cam = list.main_camera().map_or(0, |c| c.viewport);
    println!(
        "{}: {} draws in the frame, {} in the main view ({cam:#x}) with a transform; {skin_meshes} geometries carry skin weights, {} palettes in the dump, {skinned} draws skinned; drawn {} ({} triangles, {} skipped); {} geometries uploaded ({} MiB){}; load {load_s:.1} s; WARP frame: record {:.1} ms + GPU {:.1} ms first, {:.1} + {:.1} ms cached; {} ({}) -> {}",
        dir.display(),
        frame.draws.len(),
        list.main_draws().count(),
        frame.palettes.len(),
        first.draws,
        first.triangles,
        first.skipped,
        first.uploaded_geometries,
        first.uploaded_bytes >> 20,
        if frame.from_obj > 0 { format!(" from scene-local.obj ({} dropped)", frame.failed.len()) } else { String::new() },
        first.record_ms,
        first.gpu_ms,
        again.record_ms,
        again.gpu_ms,
        adapter,
        if hw_adapter() { "hardware" } else { "software" },
        out.display()
    );
    Ok(())
}

/// An RGBA image as PNG (zlib-stored deflate from xnw-formats? no: a minimal
/// stored-block writer; no external crate).
pub fn write_png(path: &std::path::Path, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        for x in 0..w as usize {
            raw.extend_from_slice(&rgba[(y * w as usize + x) * 4..][..3]);
        }
    }
    // zlib with stored (uncompressed) deflate blocks, Adler-32 at the end.
    let mut z = vec![0x78, 0x01];
    for (i, chunk) in raw.chunks(65535).enumerate() {
        let last = (i + 1) * 65535 >= raw.len();
        z.push(u8::from(last));
        z.extend((chunk.len() as u16).to_le_bytes());
        z.extend((!(chunk.len() as u16)).to_le_bytes());
        z.extend_from_slice(chunk);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    z.extend((b << 16 | a).to_be_bytes());
    let crc = |data: &[u8]| -> u32 {
        let mut c = 0xFFFF_FFFFu32;
        for &x in data {
            c ^= u32::from(x);
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    };
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    for (kind, data) in [(&b"IHDR"[..], [w.to_be_bytes(), h.to_be_bytes()].concat().into_iter().chain([8, 2, 0, 0, 0]).collect::<Vec<u8>>()), (b"IDAT", z), (b"IEND", Vec::new())] {
        png.extend((data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend(&data);
        png.extend(&body);
        png.extend(crc(&body).to_be_bytes());
    }
    std::fs::write(path, png)
}

/// Options on a live list, shared by `world-render` and `xnw-lab`:
/// `--only k[,k..]` (just these draws, list indices as --pick names them) and
/// `--rigid` (rigid parts composed with their bone).
pub fn list_options(args: &[String], list: &mut xnw_world::FrameList) {
    // `--only k[,k..]`: just these draws (list indices, as --pick names them).
    if let Some(only) = args.iter().position(|a| a == "--only").and_then(|i| args.get(i + 1)) {
        let keep: Vec<usize> = only.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        let mut i = 0usize;
        list.draws.retain(|_| {
            i += 1;
            keep.contains(&(i - 1))
        });
        // The pre-skin's draws follow their rows.
        let kept: Vec<usize> = (0..i).filter(|k| keep.contains(k)).collect();
        list.preskin = crate::preskin_gpu::subset(&list.preskin, &kept);
    }
    // `--preskin-bits` bit 3 (live `preskinbits`): ped-flagged rows without a ped remap drawn as skinned props.
    let bits = args.iter().position(|a| a == "--preskin-bits").and_then(|i| args.get(i + 1)).and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()).unwrap_or(0);
    if bits & xnw_world::preskin_gpu::BIT_PROPS != 0 {
        let n = xnw_world::preskin_gpu::props_unflag(list);
        println!("  props: {n} ped-flagged rows without a ped remap drawn as skinned props");
    }
    if args.iter().any(|a| a == "--rigid") {
        for (i, b) in list.rigid.clone() {
            if let Some(r) = list.draws.get_mut(i as usize) {
                r.world = rigid_world(&b, &r.world);
            }
        }
        println!("  rigid: {} rigid parts composed with their bone", list.rigid.len());
    }
}

/// The host and our pass as `world-render`'s options say (`--adapter
/// warp|hw` (hw: the watchdog runs), `--size`, `--shader`, `--wire`,
/// `--skin=off`, `--skinned`, `--light=old`, `--layers=`, `--ev=`, `--pass`,
/// `--tex=off`, `--terrain=off`); shared by `world-render` and `xnw-lab`.
pub fn pass_from_args(args: &[String]) -> Result<(xnw_gpu::host::Host, xnw_world::Pass), String> {
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    // `--adapter hw` draws on the hardware adapter (our own pass and HLSL only,
    // never the title's programs), under a 1 s per-frame watchdog.
    let hw = match opt("--adapter").as_deref() {
        Some("warp") => false,
        Some("hw") => true,
        _ => return Err("world-render: pass --adapter warp (software) or --adapter hw (hardware, 1 s per-frame watchdog)".into()),
    };
    // SAFETY: set before any thread of ours reads the environment.
    unsafe { std::env::set_var("XNW_ADAPTER", if hw { "hw" } else { "warp" }) };
    let (w, h) = opt("--size").and_then(|s| s.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))).unwrap_or((1280u32, 720u32));
    let t_setup = Instant::now();
    let host = xnw_gpu::host::Host::new(None).map_err(|e| e.to_string())?;
    let t_device = t_setup.elapsed().as_secs_f64() * 1e3;
    if host.caps.software == hw {
        return Err(format!("asked for the {} adapter, got {}", if hw { "hardware" } else { "software" }, host.caps.adapter));
    }
    if hw {
        // `--watchdog-ms N`: another limit (tests prove the watchdog with a tiny one).
        watchdog_start(Duration::from_millis(opt("--watchdog-ms").and_then(|v| v.parse().ok()).unwrap_or(1000)));
    }
    let mut pass = xnw_world::Pass::new(&host, w, h)?;
    eprintln!("  {}: device {t_device:.0} ms, pass (pipelines) {:.0} ms", host.caps.adapter, t_setup.elapsed().as_secs_f64() * 1e3 - t_device);
    // `--shader <file>`: the world pass shader from a file (default: the built-in one).
    if let Some(f) = opt("--shader") {
        let src = std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?;
        check_shader_table(&src).map_err(|e| format!("{f}: {e}"))?;
        pass.set_raw_shader(&src).map_err(|e| format!("{f}: {e}"))?;
    }
    pass.wire = args.iter().any(|a| a == "--wire");
    pass.skin = !args.iter().any(|a| a == "--skin=off");
    // Skinned non-ped draws only with `--skinned` (as live `skinned=on`).
    pass.skinned = args.iter().any(|a| a == "--skinned");
    // `--peds=off`: ped rows only where the pre-skin gives them positions (as live without `peds=on`).
    pass.peds = !args.iter().any(|a| a == "--peds=off");
    pass.skin_rules = skin_rules_arg(args);
    // `--light=old`: the earlier lighting (before/after pictures).
    pass.light_old = args.iter().any(|a| a == "--light=old");
    // `--layers=c0|c1|off`, `--layers-debug` (fixed layer colours on the ground stages of old dumps).
    if let Some(v) = args.iter().find(|a| a.starts_with("--layers=")) {
        pass.layers = xnw_world::parse_layers(v.trim_start_matches("--"));
    }
    pass.layers_debug = args.iter().any(|a| a == "--layers-debug");
    // `--ev=N`: the exposure bias in EV (default: automatic, night lifted).
    if let Some(b) = args.iter().find_map(|a| a.strip_prefix("--ev=")).and_then(|v| v.trim_start_matches('+').parse::<f32>().ok()) {
        xnw_world::lighting::set_ev_bias(Some(b));
    }
    pass.only_stage = opt("--pass").and_then(|s| s.parse().ok());
    // Dumps carry descriptor slots, not texels: a checker per distinct slot (`--tex=off`: flat).
    if !args.iter().any(|a| a == "--tex=off") {
        pass.tex.opts.mode = xnw_world::TexMode::Checker;
        // Baked cells as checkers; heightmaps have no texels offline (flat tiles).
        pass.terrain.table.opts.mode = xnw_world::TexMode::Checker;
    }
    // `--tex=<n>`: table entries used (as live `tex=<n>`).
    if let Some(n) = args.iter().find_map(|a| a.strip_prefix("--tex=")).and_then(|v| v.parse::<u32>().ok()) {
        pass.tex.opts.slots = n.clamp(1, xnw_world::TEX_SLOTS);
    }
    pass.terrain.opts.on = !args.iter().any(|a| a == "--terrain=off");
    pass.imposters.opts.show = args.iter().any(|a| a == "--imposters-show");
    pass.imposters.opts.on = pass.imposters.opts.show || args.iter().any(|a| a == "--imposters");
    Ok((host, pass))
}

/// A shader file must declare the albedo table at the pass's size
/// (`albedoTex[TEX_SLOTS]`): one written before c770ce0 says 256 and
/// would index past its array. Refused with the sizes named.
pub fn check_shader_table(src: &str) -> Result<(), String> {
    let n = src.split("albedoTex[").nth(1).and_then(|r| r.split(']').next()).and_then(|v| v.trim().parse::<u32>().ok());
    match n {
        Some(n) if n == xnw_world::TEX_SLOTS => Ok(()),
        Some(n) => Err(format!("declares albedoTex[{n}], the pass binds {} (TEX_SLOTS): write it again with --write-shader and carry the edit over", xnw_world::TEX_SLOTS)),
        None => Err("no albedoTex[N] declaration: not a world pass shader".into()),
    }
}

// ---- the hardware-adapter watchdog ----

/// The watchdog: (start, limit) once started; `DEADLINE` is the armed
/// deadline in ms since start (0 = disarmed).
static WATCH: OnceLock<(Instant, Duration)> = OnceLock::new();
static DEADLINE: AtomicU64 = AtomicU64::new(0);

/// Starts the watchdog (hardware adapter only): a frame armed for longer
/// than `limit` exits the process (code 3) before the driver's 2 s timeout.
pub fn watchdog_start(limit: Duration) {
    if WATCH.set((Instant::now(), limit)).is_err() {
        return;
    }
    std::thread::spawn(|| {
        let (t0, limit) = *WATCH.get().unwrap();
        loop {
            std::thread::sleep(Duration::from_millis(20));
            let d = DEADLINE.load(Ordering::Acquire);
            if d != 0 && t0.elapsed().as_millis() as u64 > d {
                eprintln!("xnw: watchdog: a frame on the hardware adapter took over {} ms: exiting before the driver's timeout", limit.as_millis());
                std::process::exit(3);
            }
        }
    });
}

/// True once the watchdog runs (the hardware adapter is in use).
pub fn hw_adapter() -> bool {
    WATCH.get().is_some()
}

/// Arms the watchdog for one frame (no-op on WARP).
pub fn watchdog_arm() {
    if let Some((t0, limit)) = WATCH.get() {
        DEADLINE.store((t0.elapsed() + *limit).as_millis() as u64, Ordering::Release);
    }
}

/// Disarms it after the frame's readback.
pub fn watchdog_disarm() {
    DEADLINE.store(0, Ordering::Release);
}
