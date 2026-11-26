//! `xnw world-render <frame dir> --adapter warp --bridge-audit pixels
//! --bridge-dir <dir>`: every main-view row (static by default,
//! `--bridge-class`, `--bridge-rows` narrow them) through the title's own
//! mesh and pixel program with the capture's bindings (draw words, sampler
//! heap, tables), against our pass's albedo for the same row. Per row: the
//! pixels target 0 covers, the colour verdict, what the record lacked by
//! kind, a removed device or a failure. Two frames: the title's target 0
//! of every row composed by depth, and our pass's albedo of the same rows;
//! a sheet of both (`sheet.png`: theirs above, ours below).
//!
//! A removed device ends a run; `--bridge-resume` continues from
//! `pixels.tsv` (a program that removed the device is not run again), and
//! a process that died inside the adapter leaves `current.txt`: that row is
//! written as `crashed` on the resume.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use xnw_world::bridge;
use xnw_world::bridged::{self, SrcRoot, SrcView, Source, WORDS};
use xnw_world::{DrawRecord, GeomKey, HostGeometry, Pass, RAW_SHADER, TexMode};

use crate::bridge::{Row, load, programs};
use crate::load::load_list;
use crate::pack::Pack;

/// What target 0 of one row says (pixels it covers: RGBA): `plausible`
/// (finite, colour within 0..2, not black everywhere), `black`, `not
/// finite`, `out of range`, `empty` (no pixel).
pub fn colour_verdict(px: &[[f32; 4]]) -> &'static str {
    if px.is_empty() {
        return "empty";
    }
    if px.iter().any(|p| p.iter().any(|v| !v.is_finite())) {
        return "not finite";
    }
    if px.iter().any(|p| p[..3].iter().any(|v| !(-1e-3..=2.0).contains(v))) {
        return "out of range";
    }
    if px.iter().all(|p| p[..3].iter().all(|v| v.abs() < 1e-3)) {
        return "black";
    }
    "plausible"
}

/// The kinds of binding a record's `lacks` lines say are missing (the
/// stand-ins `load` puts in their place), by name; lines that say nothing
/// is missing are not counted.
pub fn lack_kinds(lacks: &[String]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for l in lacks {
        let kind = if l.starts_with("nothing for the bindless words") {
            if l.contains("textures without texels") { Some("word texture without texels") } else { None }
        } else if l.starts_with("nothing for") {
            None
        } else if l.starts_with("the bindless words") || l.starts_with("the draw words") {
            Some("bindless words")
        } else if l.starts_with("sampler heap entries") {
            Some("sampler heap")
        } else if l.contains("a texture the programs name has no texels") || l.starts_with("texels of the view") {
            Some("table texture without texels")
        } else if l.contains("a view the programs name has no bytes") {
            Some("table view without bytes")
        } else if l.contains("constant buffer the programs name") {
            Some("constant buffer zero")
        } else if l.contains("nothing stands in") {
            Some("root view")
        } else if l.starts_with("the group count: no") {
            Some("group count")
        } else {
            None
        };
        if let Some(k) = kind
            && !out.contains(&k)
        {
            out.push(k);
        }
    }
    out
}

/// One row's pixels as the title's target 0 left them: (pixel, RGBA, z).
type Pixels = Vec<(u32, [f32; 4], f32)>;

fn write_pixels(f: &mut std::fs::File, row: u32, px: &Pixels) -> std::io::Result<()> {
    let mut b = Vec::with_capacity(8 + 24 * px.len());
    b.extend(row.to_le_bytes());
    b.extend((px.len() as u32).to_le_bytes());
    for (i, c, z) in px {
        b.extend(i.to_le_bytes());
        for v in c.iter().chain(std::iter::once(z)) {
            b.extend(v.to_le_bytes());
        }
    }
    f.write_all(&b)
}

fn read_pixels(path: &Path) -> HashMap<u32, Pixels> {
    let b = std::fs::read(path).unwrap_or_default();
    let w = |at: usize| b.get(at..at + 4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()));
    let mut out = HashMap::new();
    let mut at = 0;
    while at + 8 <= b.len() {
        let (row, n) = (w(at), w(at + 4) as usize);
        at += 8;
        if at + 24 * n > b.len() {
            break;
        }
        let px = (0..n).map(|k| {
            let e = at + 24 * k;
            let f = |j: usize| f32::from_bits(w(e + 4 + 4 * j));
            (w(e), [f(0), f(1), f(2), f(3)], f(4))
        });
        out.insert(row, px.collect());
        at += 24 * n;
    }
    out
}

/// The title's rows composed by depth (reverse Z: the larger z is nearer)
/// as RGBA8, black where no row drew.
pub fn compose(width: u32, height: u32, rows: &[&Pixels]) -> Vec<u8> {
    let n = (width * height) as usize;
    let (mut z, mut rgba) = (vec![f32::MIN; n], vec![0u8; 4 * n]);
    for px in rows {
        for (i, c, d) in px.iter() {
            let i = *i as usize;
            if i < n && *d >= z[i] {
                z[i] = *d;
                for k in 0..3 {
                    rgba[4 * i + k] = (c[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
                rgba[4 * i + 3] = 255;
            }
        }
    }
    rgba
}

const HEADER: &str = "row\tdraw\tsink\tstage\tpixel\tmesh\tgroups\tseconds\ttheirs\tours\tboth\tdiff\tr\tg\tb\tverdict\tlacks\tnote";

/// `--bridge-audit pixels`.
pub fn audit_pixels(args: &[String], dir: &Path, out: &Path, host: &xnw_gpu::host::Host, pass: &mut Pass) -> Result<(), String> {
    let (mut list, geoms) = load_list(dir)?;
    let pack = Pack::open(dir)?.ok_or("--bridge-audit needs a v2 capture (pack\\manifest.tsv)")?;
    let mut args = args.to_vec();
    if !args.iter().any(|a| a == "--bridge-class") {
        args.extend(["--bridge-class".to_string(), "static".to_string()]);
    }
    let rows = crate::bridge::audit_rows(&args, &list, &pack)?;
    let (tsv_path, px_path, current) = (out.join("pixels.tsv"), out.join("pixels.bin"), out.join("current.txt"));
    let resume = args.iter().any(|a| a == "--bridge-resume");
    let mut lines: Vec<String> = if resume { std::fs::read_to_string(&tsv_path).unwrap_or_default().lines().skip(1).map(str::to_owned).collect() } else { Vec::new() };
    let mut removes: Vec<String> = lines.iter().filter(|l| l.contains("device removed") || l.contains("\tcrashed\t") || l.contains("the submission took over")).filter_map(|l| l.split('\t').nth(4).map(str::to_owned)).collect();
    // A process that died inside the adapter left the row it was on.
    if resume && let Ok(c) = std::fs::read_to_string(&current) {
        let c: Vec<&str> = c.trim().split('\t').collect();
        if c.len() >= 5 && !lines.iter().any(|l| l.split('\t').next() == Some(c[0])) {
            lines.push(format!("{}\t{}\t{}\t{}\t{}\t-\t0\t0\t0\t0\t0\t0\t0\t0\t0\tcrashed\t-\tthe process died inside the adapter", c[0], c[1], c[2], c[3], c[4]));
            removes.push(c[4].to_owned());
        }
    }
    let done: Vec<u32> = lines.iter().filter_map(|l| l.split('\t').next()?.parse().ok()).collect();
    if !resume {
        let _ = std::fs::remove_file(&px_path);
    }
    let save = |lines: &[String]| std::fs::write(&tsv_path, format!("{HEADER}\n{}", lines.iter().map(|l| format!("{l}\n")).collect::<String>())).map_err(|e| format!("{}: {e}", tsv_path.display()));
    save(&lines)?;
    let mut pxf = std::fs::OpenOptions::new().create(true).append(true).open(&px_path).map_err(|e| format!("{}: {e}", px_path.display()))?;

    // Our side: the albedo view with the capture's texels.
    pass.set_raw_shader(&format!("#define LAB_VIEW 6\n{RAW_SHADER}"))?;
    (pass.sky, pass.water, pass.terrain.opts.on) = (false, false, false);
    pass.tex.opts.mode = TexMode::Provided;
    let counts = Arc::new([const { AtomicU64::new(0) }; 3]);
    pass.tex.set_provider(Some(pack.texel_provider(counts.clone())));
    let mut texels = pack.texel_provider(counts);
    let mut decode = |key: &GeomKey| geoms.get(key).map(|(vb, idx)| HostGeometry { vertices: Vec::new(), indices: idx.clone(), raw_vb: Some(vb.clone()), vb_place: 0 });
    let all = std::mem::take(&mut list.draws);
    let mut progs: HashMap<u64, Result<_, String>> = HashMap::new();
    let t0 = std::time::Instant::now();
    let mut lost = false;
    let mut frame_no = 0u64;
    for &(k, draw, sink) in rows.iter().filter(|r| !done.contains(&(r.0 as u32))) {
        if lost {
            break;
        }
        let row: &DrawRecord = &all[k];
        let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).map_or(0, |s| s.pipeline);
        let entry = progs.entry(pipeline).or_insert_with(|| programs(&pack, sink).and_then(|x| crate::bridge::pixel_refused(&pack, &x.2, host.caps.software).map_or(Ok(x), Err)));
        let (ms_hash, ps_hash) = match entry {
            Ok((_, _, h)) => (format!("{:08x}", h[0] >> 96), format!("{:08x}", h[1] >> 96)),
            Err(_) => ("-".into(), "-".into()),
        };
        // Ours: the row alone, albedo and coverage.
        list.draws = vec![row.clone()];
        frame_no += 2;
        list.frame = frame_no - 1;
        pass.draw(&list, &mut decode, true)?;
        let ours = pass.read_rgba()?;
        (list.frame, pass.id_mode) = (frame_no, true);
        let ids = pass.draw(&list, &mut decode, true).and_then(|_| pass.read_rgba());
        pass.id_mode = false;
        let mask: Vec<bool> = ids?.chunks_exact(4).map(|px| px[..3] == [1, 0, 0]).collect();
        let covered = mask.iter().filter(|m| **m).count();
        let head = format!("{k}\t{draw}\t{sink}\t{}\t{ps_hash}\t{ms_hash}", row.stage);
        let fail = |groups: u32, why: &str, verdict: &str, lacks: &str| format!("{head}\t{groups}\t0\t0\t{covered}\t0\t0\t0\t0\t0\t{verdict}\t{lacks}\t{}", why.replace(['\t', '\n'], " "));
        let (p, used) = match entry {
            Ok((p, used, _)) => (&*p, &*used),
            Err(e) => {
                lines.push(fail(0, e, "not run", "-"));
                continue;
            }
        };
        if removes.contains(&ps_hash) {
            lines.push(fail(0, "this pixel program removed the device, crashed or ran past the limit on an earlier row", "not run", "-"));
            continue;
        }
        if let Some(gap) = bridge::gap(&host.caps, &p.needs()) {
            lines.push(fail(0, &gap, "not run", "-"));
            continue;
        }
        let mut loaded = load(&pack, sink, &Row::with_palette(row, &list, &pack, draw), geoms.get(&row.geom).map(|g| &g.0[..]), used, &mut texels);
        if !args.iter().any(|a| a == "--bridge-visible") {
            loaded.direct_meshlets();
        }
        let kinds = lack_kinds(&loaded.lacks);
        let lacks = if kinds.is_empty() { "-".to_string() } else { kinds.join(",") };
        if args.iter().any(|a| a == "--bridge-lacks") {
            for l in &loaded.lacks {
                println!("  row {k}: lacks {l}");
            }
        }
        if loaded.groups == 0 {
            lines.push(fail(0, "no group count", "not run", &lacks));
            continue;
        }
        let d = loaded.draw();
        let b = match bridge::bind(&p.ms, &p.ps, &d) {
            Ok(b) => b,
            Err(e) => {
                lines.push(fail(loaded.groups, &e, "not run", &lacks));
                continue;
            }
        };
        let depth = row.state.filter(|s| s.depth_enable).map(|s| (s.depth_write, i32::from(s.depth_func)));
        let target = bridge::Target { width: pass.width, height: pass.height, depth, limit: if host.caps.software { 120.0 } else { 1.0 }, depth_only: false, records: false };
        std::fs::write(&current, format!("{head}\n")).map_err(|e| e.to_string())?;
        crate::replay::watchdog_arm();
        let frame = bridge::run(host, p, &b, &target);
        crate::replay::watchdog_disarm();
        let frame = match frame {
            Ok(f) => f,
            Err(e) => {
                let removed = e.starts_with("device removed");
                // A program that ran past the limit runs past it again: not run on later rows.
                if e.contains("the submission took over") {
                    removes.push(ps_hash.clone());
                }
                lines.push(fail(loaded.groups, &e, if removed { "device removed" } else { "failed" }, &lacks));
                // SAFETY: the device call takes no arguments.
                lost = unsafe { host.device.GetDeviceRemovedReason() }.is_err();
                save(&lines)?;
                continue;
            }
        };
        let px: Pixels = frame.rt0.iter().zip(&frame.depth).enumerate().filter(|(_, (c, _))| **c != bridge::CLEAR).map(|(i, (c, z))| (i as u32, *c, *z)).collect();
        let colours: Vec<[f32; 4]> = px.iter().map(|p| p.1).collect();
        let verdict = colour_verdict(&colours);
        let mean: [f64; 3] = std::array::from_fn(|c| colours.iter().map(|p| f64::from(p[c])).filter(|v| v.is_finite()).sum::<f64>() / colours.len().max(1) as f64);
        let (both, diff) = bridge::compare(&ours, &mask, &frame.rt0);
        lines.push(format!("{head}\t{}\t{:.3}\t{}\t{covered}\t{both}\t{diff:.4}\t{:.3}\t{:.3}\t{:.3}\t{verdict}\t{lacks}\t", loaded.groups, frame.seconds, px.len(), mean[0], mean[1], mean[2]));
        write_pixels(&mut pxf, k as u32, &px).map_err(|e| e.to_string())?;
        save(&lines)?;
    }
    save(&lines)?;
    let _ = std::fs::remove_file(&current);
    drop(pxf);
    println!("bridge audit (pixels): {} rows, {} written, in {:.1} s on {} -> {}", rows.len(), lines.len(), t0.elapsed().as_secs_f64(), host.caps.adapter, tsv_path.display());
    if lost {
        println!("bridge audit (pixels): the device was removed; run again with --bridge-resume for the rows left");
        return Ok(());
    }
    summary(&lines);

    // The two frames: theirs composed by depth, ours through our pass in one draw.
    let pixels = read_pixels(&px_path);
    let theirs = compose(pass.width, pass.height, &pixels.values().collect::<Vec<_>>());
    list.draws = rows.iter().map(|r| all[r.0].clone()).collect();
    list.frame = frame_no + 1;
    pass.draw(&list, &mut decode, true)?;
    let ours = pass.read_rgba()?;
    let mut sheet = theirs.clone();
    sheet.extend_from_slice(&ours);
    crate::replay::write_png(&out.join("theirs.png"), pass.width, pass.height, &theirs).map_err(|e| e.to_string())?;
    crate::replay::write_png(&out.join("ours.png"), pass.width, pass.height, &ours).map_err(|e| e.to_string())?;
    crate::replay::write_png(&out.join("sheet.png"), pass.width, 2 * pass.height, &sheet).map_err(|e| e.to_string())?;
    println!("bridge audit (pixels): sheet (theirs above, ours below) -> {}", out.join("sheet.png").display());
    Ok(())
}

/// A frame pack sink as the bridged rows' [`Source`] (the live title's is
/// the list sink's tap): its root constants, root views (the list's
/// vertex buffer where the capture has none), tables, draw words and the
/// views they name, sampler heap entries. Buffer bytes are shared across
/// the frame's sinks by (address, length).
pub struct PackSource<'a> {
    pack: &'a Pack,
    sink: u32,
    tables: Vec<(u16, u64, Vec<xnw_formats::framepack::SlotView>)>,
    vb: Option<(u64, Arc<Vec<u8>>)>,
    shared: &'a std::cell::RefCell<HashMap<(u64, u64, u32), Arc<Vec<u8>>>>,
}

impl<'a> PackSource<'a> {
    pub fn new(pack: &'a Pack, sink: u32, vb: Option<(u64, Arc<Vec<u8>>)>, shared: &'a std::cell::RefCell<HashMap<(u64, u64, u32), Arc<Vec<u8>>>>) -> PackSource<'a> {
        PackSource { pack, sink, tables: pack.tables(sink), vb, shared }
    }
}

fn src_view(v: &xnw_formats::framepack::SlotView) -> SrcView {
    // A placed view's words: its resource description (14), then its view
    // description, whose second word is the view's dimension.
    SrcView { kind: v.kind, addr: v.addr, bytes: v.bytes, format: v.format, width: v.width, dim: bridged::title_dim(v.kind, &v.words) }
}

impl Source for PackSource<'_> {
    fn roots(&self) -> Vec<SrcRoot> {
        use xnw_formats::framepack::{kind, reason};
        let mut out: Vec<SrcRoot> = Vec::new();
        let mut put = |i: u16, r: SrcRoot| {
            if out.len() <= usize::from(i) {
                out.resize(usize::from(i) + 1, SrcRoot::None);
            }
            out[usize::from(i)] = r;
        };
        for r in self.pack.data.refs_of(self.sink) {
            match r.kind {
                kind::ROOTS if reason::has_bytes(r.reason) => {
                    if let Some(b) = self.pack.blob(r.hash) {
                        put(r.slot, SrcRoot::Constants(b.chunks_exact(4).map(|c| crate::le32(c, 0)).collect()));
                    }
                }
                kind::ROOT_VIEW => put(r.slot, SrcRoot::View(r.addr)),
                _ => {}
            }
        }
        for (root, handle, _) in &self.tables {
            put(*root, SrcRoot::Table(*handle));
        }
        out
    }

    fn table(&self, root: u16, slot: u32) -> Option<SrcView> {
        self.tables.iter().find(|t| t.0 == root)?.2.iter().find(|v| v.slot == slot).map(src_view)
    }

    fn bytes(&self, root: u16, slot: u32, v: &SrcView) -> Option<Arc<Vec<u8>>> {
        let (_, b) = self.pack.slot_bytes(self.sink, root, slot).ok()?;
        let key = (v.addr, v.bytes, b.len() as u32);
        Some(self.shared.borrow_mut().entry(key).or_insert_with(|| Arc::new(b)).clone())
    }

    fn root_view(&self, root: u16, addr: u64) -> Option<Arc<Vec<u8>>> {
        use xnw_formats::framepack::{kind, reason};
        let r = self.pack.data.refs_of(self.sink).find(|r| r.kind == kind::ROOT_VIEW && r.slot == root)?;
        if reason::has_bytes(r.reason)
            && let Some(b) = self.pack.blob(r.hash)
        {
            return Some(Arc::new(b));
        }
        // The capture cut it: the list's copy of the row's vertex buffer.
        self.vb.as_ref().filter(|(at, _)| addr == *at || addr == *at + 0x4_0000_0000).map(|v| v.1.clone())
    }

    fn words(&self) -> Option<[u32; WORDS]> {
        self.pack.draw_words(self.sink)?.words.ok()
    }

    fn heap(&self, index: u32) -> Option<(u64, SrcView)> {
        let dw = self.pack.draw_words(self.sink)?;
        let v = dw.views.ok()?.into_iter().find(|v| v.slot == index)?;
        Some((dw.heap_start + 32 * u64::from(index), src_view(&v)))
    }

    fn samplers(&self) -> Option<Arc<Vec<(u32, bridge::SamplerDesc)>>> {
        crate::bridge::samplers(self.pack, self.sink, &mut Vec::new()).map(Arc::new)
    }
}

/// `--bridgeps on|<hash>+<hash>` or `XNW_WORLD_RENDER=...,bridgeps=...`:
/// the rows of those pixel programs through the live title's path:
/// each sink as a [`Source`] to [`bridged::take`], the pipelines from
/// [`bridged::cache`], the frame drawn by our pass with the rows recorded
/// on its list ([`xnw_world::bridged::record`]), textures from our texture
/// table by handle. Prints both timings; `--out` writes the sheet (our
/// pass alone above, with the title's rows below).
pub fn bridgeps_cli(args: &[String], dir: &Path, host: &xnw_gpu::host::Host, pass: &mut Pass, wanted: &[u32]) -> Result<(), String> {
    xnw_world::bridge::title_programs_here(host).map_err(|e| format!("bridgeps: {e}"))?;
    let (mut list, geoms) = load_list(dir)?;
    let pack = Pack::open(dir)?.ok_or("--bridgeps needs a v2 capture (pack\\manifest.tsv)")?;
    let mut args = args.to_vec();
    if !args.iter().any(|a| a == "--bridge-class") {
        args.extend(["--bridge-class".to_string(), "static".to_string()]);
    }
    let rows = crate::bridge::audit_rows(&args, &list, &pack)?;
    if !args.iter().any(|a| a == "--bridgeps-lit") {
        pass.set_raw_shader(&format!("#define LAB_VIEW 6\n{RAW_SHADER}"))?;
    }
    (pass.sky, pass.water, pass.terrain.opts.on) = (false, false, false);
    pass.tex.opts.mode = TexMode::Provided;
    let counts = Arc::new([const { AtomicU64::new(0) }; 3]);
    pass.tex.set_provider(Some(pack.texel_provider(counts)));
    let mut decode = |key: &GeomKey| geoms.get(key).map(|(vb, idx)| HostGeometry { vertices: Vec::new(), indices: idx.clone(), raw_vb: Some(vb.clone()), vb_place: 0 });
    // The pipelines, once per pair (here, not on a thread: the replay waits).
    let t1 = std::time::Instant::now();
    let mut taken = Vec::new();
    let mut other = 0u32;
    let shared = std::cell::RefCell::new(HashMap::new());
    let vbs: HashMap<GeomKey, Arc<Vec<u8>>> = HashMap::new();
    let mut vbs = vbs;
    let mut roots_shown = std::collections::HashSet::new();
    for &(k, _, sink) in &rows {
        let row: &DrawRecord = &list.draws[k];
        let Some(pipeline) = pack.data.sinks.iter().find(|s| s.index == sink).map(|s| s.pipeline) else { continue };
        let stage = |name: &str| pack.data.programs.iter().find(|p| p.pipeline == pipeline && p.stage == name).map(|p| p.container);
        let (Some(ms), Some(ps)) = (stage("mesh"), stage("pixel")) else {
            other += 1;
            continue;
        };
        if !wanted.contains(&((ps >> 96) as u32)) {
            other += 1;
            continue;
        }
        let key = (ms, ps);
        if !bridged::cache().asked(key) {
            let (Some(msc), Some(psc)) = (pack.blob(ms), pack.blob(ps)) else { continue };
            bridged::cache().ask(key, msc, psc, &host.device, &host.caps, true);
        }
        let Some(built) = bridged::cache().get(key) else { continue };
        // `--bridgeps-inputs`: which root views each program reads (a root
        // view has no bounds on the card; the mesh form's SRV reads are
        // clamped by their lengths, the pixel program's are not: HANG1).
        if args.iter().any(|a| a == "--bridgeps-inputs") && roots_shown.insert(key) {
            let (m, p) = (&built.programs.ms, &built.programs.ps);
            println!("bridgeps roots: pixel {:08x}: mesh SRVs {:?} CBVs {:?} (SRVs clamped); pixel SRVs {:?} CBVs {:?} (unclamped)", (ps >> 96) as u32, m.root_srvs, m.root_cbvs, p.root_srvs, p.root_cbvs);
        }
        let vb = geoms.get(&row.geom).map(|g| (row.geom.vb_at, vbs.entry(row.geom).or_insert_with(|| Arc::new(g.0.clone())).clone()));
        let mut o = bridged::take(&PackSource::new(&pack, sink, vb, &shared), key, &built.used);
        o.set_groups(row.geom.mesh_at);
        taken.push((k as u32, o));
    }
    let t_take = t1.elapsed().as_secs_f64();
    // `--bridgeps-inputs`: the rows' buffer inputs by (root, slot): rows, the
    // largest, distinct bytes as the capture holds them per sink (before
    // the frame shares them by address) and as the rows get them.
    if args.iter().any(|a| a == "--bridgeps-inputs") {
        let sink_of: HashMap<u32, u32> = rows.iter().map(|&(k, _, sink)| (k as u32, sink)).collect();
        let mut by: std::collections::BTreeMap<(usize, u32), (u32, usize, u64, std::collections::HashSet<u128>, std::collections::HashSet<usize>)> = Default::default();
        for (k, o) in &taken {
            for (r, x) in o.roots.iter().enumerate() {
                let bufs: Vec<(u32, &Arc<Vec<u8>>)> = match x {
                    bridged::ORoot::Table(v) => v.iter().filter_map(|(s, sl)| if let bridged::OSlot::Buffer(_, b) = sl { Some((*s, b)) } else { None }).collect(),
                    bridged::ORoot::View(b) => vec![(u32::MAX, b)],
                    _ => Vec::new(),
                };
                for (slot, b) in bufs {
                    let e = by.entry((r, slot)).or_default();
                    e.0 += 1;
                    e.1 = e.1.max(b.len());
                    e.2 += b.len() as u64;
                    let own = if slot == u32::MAX { None } else { sink_of.get(k).and_then(|&sink| pack.slot_bytes(sink, r as u16, slot).ok()) };
                    e.3.insert(own.map_or_else(|| xnw_core::hash128(&[&b[..]]), |(_, raw)| xnw_core::hash128(&[&raw])));
                    e.4.insert(Arc::as_ptr(b) as usize);
                }
            }
        }
        for ((r, slot), (n, most, all, contents, shared)) in &by {
            let at = if *slot == u32::MAX { format!("root {r} view") } else { format!("root {r} slot {slot}") };
            println!("bridgeps inputs: {at}: {n} rows, largest {most} B, {all} B in all; {} distinct as captured per sink, {} as the rows get them", contents.len(), shared.len());
        }
        // The draw words by position: a texture, another view, or no view
        // at that heap index (with the values seen there most).
        let mut words: Vec<([u32; 3], HashMap<u32, u32>)> = vec![Default::default(); WORDS];
        for &(_, _, sink) in rows.iter().filter(|r| sink_of.values().any(|s| *s == r.2)) {
            let Some(dw) = pack.draw_words(sink) else { continue };
            let (Ok(w), views) = (dw.words, dw.views.unwrap_or_default()) else { continue };
            for (k, x) in w.iter().enumerate() {
                let c = match views.iter().find(|v| v.slot == *x) {
                    Some(v) if v.kind == xnw_formats::framepack::view::TEXTURE || v.kind == xnw_formats::framepack::view::PLACED => 0,
                    Some(_) => 1,
                    None => {
                        *words[k].1.entry(*x).or_default() += 1;
                        2
                    }
                };
                words[k].0[c] += 1;
            }
        }
        for (k, (n, vals)) in words.iter().enumerate() {
            let mut top: Vec<(u32, u32)> = vals.iter().map(|(v, c)| (*c, *v)).collect();
            top.sort_unstable_by(|a, b| b.cmp(a));
            let top: Vec<String> = top.iter().take(4).map(|(c, v)| format!("{v:#x} x{c}")).collect();
            println!("bridgeps words: word {k}: texture {}, other view {}, no view {} [{}]", n[0], n[1], n[2], top.join(", "));
        }
    }
    let n = taken.len();
    list.bridged = Some(Arc::new(bridged::Frame::new(taken)));
    // Our pass alone first, the rows' textures read in the same frame.
    list.frame = 1;
    pass.bridge_warm = true;
    let t0 = std::time::Instant::now();
    crate::replay::watchdog_arm();
    let r = pass.draw(&list, &mut decode, true).and_then(|st| Ok((st, pass.read_rgba()?)));
    crate::replay::watchdog_disarm();
    println!("  {}", crate::replay::watch_line("bridgeps ours"));
    let (st, ours) = r?;
    let t_ours = t0.elapsed().as_secs_f64();
    pass.bridge_warm = false;
    pass.bridge_on = true;
    // `--warm`: the bridged frame once first (bindings, kept inputs), its picture thrown away.
    if args.iter().any(|a| a == "--warm") {
        crate::replay::watchdog_arm();
        let r = pass.draw(&list, &mut decode, true);
        crate::replay::watchdog_disarm();
        println!("  {}", crate::replay::watch_line("bridgeps bridged warm-up"));
        r?;
        // What the warm-up frame drew and bound (its inputs go with the frame; the next one keeps them).
        let warm = pass.read_rgba()?;
        let changed = ours.chunks_exact(4).zip(warm.chunks_exact(4)).filter(|(a, b)| a[..3] != b[..3]).count();
        println!("bridgeps: warm-up: {changed} pixels changed; {}", bridged::line(list.frame, list.bridged.as_deref().unwrap_or(&bridged::Frame::default()), &pass.bridge_stats));
    }
    let t2 = std::time::Instant::now();
    // Every submission of the bridged frame (with `--chunk N`: N of our draws, then N bridged rows, a submission each) under the watchdog.
    crate::replay::watchdog_arm();
    let r = pass.draw(&list, &mut decode, true).and_then(|_| pass.read_rgba());
    crate::replay::watchdog_disarm();
    println!("  {}", crate::replay::watch_line("bridgeps bridged"));
    let theirs = r?;
    let t_with = t2.elapsed().as_secs_f64();
    // `--bridgeps-repeat N`: the same frame N more times (the inputs again
    // as the same bytes, as the live tap gives kept views): each frame's
    // record cost, and whether its picture is the first one's.
    if let Some(n) = args.iter().position(|a| a == "--bridgeps-repeat").and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<u32>().ok()) {
        for k in 1..=n {
            crate::replay::watchdog_arm();
            let r = pass.draw(&list, &mut decode, true).and_then(|_| pass.read_rgba());
            crate::replay::watchdog_disarm();
            println!("  {}", crate::replay::watch_line(&format!("bridgeps repeat {k}")));
            let again = r?;
            let differ = theirs.chunks_exact(4).zip(again.chunks_exact(4)).filter(|(a, b)| a[..3] != b[..3]).count();
            let s = &pass.bridge_stats;
            println!("bridgeps repeat {k}: record {:.2} ms (bindings {:.2}, uploads {:.2}, views {:.2}, list {:.2}); uploaded {:.1} MiB, kept {:.1} MiB ({:.1} MiB new); {differ} pixels differ from the first; {}", s.ms, s.parts[0], s.parts[1], s.parts[2], s.parts[3], s.bytes as f64 / f64::from(1 << 20), s.kept as f64 / f64::from(1 << 20), s.kept_new as f64 / f64::from(1 << 20), bridged::blocks_text(s));
        }
    }
    let changed = ours.chunks_exact(4).zip(theirs.chunks_exact(4)).filter(|(a, b)| a[..3] != b[..3]).count();
    println!("bridgeps: our pass {} draws in {t_ours:.2} s; the title's programs {wanted:08x?}: {n} rows ({other} rows of other programs), pipelines and inputs in {t_take:.2} s, the frame with them in {t_with:.2} s; {changed} pixels changed", st.draws);
    println!("bridgeps: {}", bridged::line(list.frame, list.bridged.as_deref().unwrap_or(&bridged::Frame::default()), &pass.bridge_stats));
    if std::env::var("XNW_GPU_DEBUG").is_ok_and(|v| v.contains("layer")) {
        for l in xnw_world::bridge::layer_messages(&host.device) {
            println!("bridgeps: {l}");
        }
    }
    if let Some(out) = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)) {
        let mut sheet = ours;
        sheet.extend_from_slice(&theirs);
        crate::replay::write_png(Path::new(out), pass.width, 2 * pass.height, &sheet).map_err(|e| e.to_string())?;
        println!("bridgeps: sheet (ours above, with the title's rows below) -> {out}");
    }
    Ok(())
}

/// Rows by verdict, by lacking kind, and by pixel program.
fn summary(lines: &[String]) {
    let mut verdicts: std::collections::BTreeMap<&str, u32> = Default::default();
    let mut kinds: std::collections::BTreeMap<&str, u32> = Default::default();
    let mut programs: std::collections::BTreeMap<&str, (u32, u32)> = Default::default();
    for l in lines {
        let c: Vec<&str> = l.split('\t').collect();
        let (Some(ps), Some(v), Some(lk)) = (c.get(4), c.get(15), c.get(16)) else { continue };
        *verdicts.entry(v).or_default() += 1;
        if *v == "plausible" && *lk == "-" {
            *verdicts.entry("plausible with every binding held").or_default() += 1;
        }
        for k in lk.split(',').filter(|k| *k != "-") {
            *kinds.entry(k).or_default() += 1;
        }
        let e = programs.entry(ps).or_default();
        e.0 += 1;
        e.1 += u32::from(*v == "plausible");
    }
    println!("bridge audit (pixels): by verdict: {}", verdicts.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", "));
    println!("bridge audit (pixels): rows lacking, by kind: {}", kinds.iter().map(|(k, n)| format!("{k} {n}")).collect::<Vec<_>>().join(", "));
    let mut by: Vec<_> = programs.into_iter().collect();
    by.sort_by_key(|p| std::cmp::Reverse(p.1.0));
    println!("bridge audit (pixels): pixel programs {}; most rows: {}", by.len(), by.iter().take(12).map(|(p, (n, ok))| format!("{p} {ok}/{n}")).collect::<Vec<_>>().join(", "));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_verdicts() {
        assert_eq!(colour_verdict(&[]), "empty");
        assert_eq!(colour_verdict(&[[0.5, 0.4, 0.3, 1.0]]), "plausible");
        assert_eq!(colour_verdict(&[[0.0, 0.0, 0.0, 0.02]]), "black");
        assert_eq!(colour_verdict(&[[f32::NAN, 0.0, 0.0, 0.0]]), "not finite");
        assert_eq!(colour_verdict(&[[0.5, 3.0, 0.0, 0.0]]), "out of range");
    }

    #[test]
    fn lacking_kinds_from_lines() {
        let l = |s: &str| s.to_string();
        let lacks = [
            l("nothing for the bindless words: the draw's 16 words at slot 86 + 0x10; 5 textures at their heap indices, 8 buffer views not bound, 2 words name no view (words [1]); textures without texels (1x1 grey stands in): 11683 (no texels)"),
            l("nothing for the sampler heap: 54 entries (0 raw, default there)"),
            l("root 2 slot 3: a texture the programs name has no texels"),
            l("root view 3 at 0x7 (reason 3): the list's vertex buffer 0x3 (8 B) stands in"),
            l("sampler heap entries: 128 default samplers (linear, wrap)"),
        ];
        assert_eq!(lack_kinds(&lacks), vec!["word texture without texels", "table texture without texels", "sampler heap"]);
        let pixels: Pixels = vec![(0, [1.0, 0.0, 0.0, 1.0], 0.2), (1, [0.0, 1.0, 0.0, 1.0], 0.5)];
        let nearer: Pixels = vec![(0, [0.0, 0.0, 1.0, 1.0], 0.4), (1, [0.0, 0.0, 1.0, 1.0], 0.1)];
        let rgba = compose(2, 1, &[&pixels, &nearer]);
        assert_eq!(rgba, vec![0, 0, 255, 255, 0, 255, 0, 255]);
    }
}
