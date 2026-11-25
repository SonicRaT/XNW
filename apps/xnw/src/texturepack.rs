//! `xnw texture-pack <ttpf | dir | rpf> [--out DIR] [--drop-mips N] [--only-trace LOG] [--threads K] [--plan] [--verify N] [--dds N]`
//!
//! Converts the title's `.ttpf` textures to host-ready packs
//! (`xnw_formats::texpack`, "Pack"). One `.xtp` per
//! source file in `--out` (default `out\texpack`); resumable: textures
//! already in a matching pack are skipped. An `.rpf` stands for
//! `<its folder>/data/ttdb`.
//!
//! - `--only-trace LOG`: only textures whose pages a run requested
//!   (`XNW_TRACE=dstorage` lines `dstorage: options .., <path> @ <offset>, ..`).
//! - `--threads K`: workers, 1 to 4 (default 4). The process runs at idle
//!   priority in background mode (low CPU and I/O priority): the machine is
//!   shared.
//! - `--plan`: no conversion; pack sizes with 0, 1 and 2 mips dropped.
//! - `--verify N`: N packed 2D textures (formats in turn) compared byte for
//!   byte with the runtime's path: each page decoded as DirectStorage does
//!   (zlib, then BCPack), the chain laid out from the title-style
//!   description (`texture::Texture::new`, as `xnw-gpu` does) and detiled.
//! - `--dds N`: N packed textures written as `.dds` (stored mips) to
//!   `<out>\dds`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

use xnw_formats::bcpack::{self, Tables};
use xnw_formats::texpack::{self, Writer};
use xnw_formats::texture::{self, Dimension, TextureDesc};
use xnw_formats::ttpf::{self, Ttpf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage: xnw texture-pack <ttpf|dir|rpf> [--out DIR] [--drop-mips N] [--only-trace LOG] [--threads K] [--plan] [--verify N] [--dds N]";

#[derive(Default)]
struct Totals {
    converted: AtomicU64,
    skipped: AtomicU64,
    failed: AtomicU64,
    stored_in: AtomicU64,
    bytes_out: AtomicU64,
    busy_ns: AtomicU64,
}

pub fn run(args: &[String]) -> Res<()> {
    let path = args.first().filter(|a| !a.starts_with("--")).ok_or(USAGE)?;
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1));
    let out = PathBuf::from(flag("--out").map_or("out\\texpack", |s| s.as_str()));
    let drop: u32 = flag("--drop-mips").map_or(Ok(0), |v| v.parse())?;
    let threads: usize = flag("--threads").map_or(Ok(4), |v| v.parse())?;
    let threads = threads.clamp(1, 4);
    let verify: usize = flag("--verify").map_or(Ok(0), |v| v.parse())?;
    let dds: usize = flag("--dds").map_or(Ok(0), |v| v.parse())?;
    let plan_only = args.iter().any(|a| a == "--plan");
    let trace = flag("--only-trace").map(|p| read_trace(Path::new(p))).transpose()?;

    background();
    let mut root = PathBuf::from(path);
    if root.extension().is_some_and(|e| e.eq_ignore_ascii_case("rpf")) {
        root = root.parent().unwrap_or(Path::new(".")).join("data").join("ttdb");
    }
    let mut files = vec![];
    collect(&root, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err(format!("no .ttpf under {}", root.display()).into());
    }
    if plan_only {
        return plan(&files, trace.as_ref());
    }
    let tables = Tables::builtin().map_err(|e| e.0)?;
    if verify == 0 && dds == 0 || args.iter().any(|a| a == "--convert") {
        convert(&files, trace.as_ref(), &out, drop, threads, &tables)?;
    }
    if verify > 0 {
        verify_packs(&files, &out, verify, &tables)?;
    }
    if dds > 0 {
        write_dds(&files, &out, dds)?;
    }
    Ok(())
}

/// Idle CPU priority and background mode (low I/O and memory priority).
fn background() {
    use xnw_core::win::{GetCurrentProcess, IDLE_PRIORITY_CLASS, PROCESS_MODE_BACKGROUND_BEGIN, SetPriorityClass};
    // SAFETY: plain calls on the current process's pseudo handle.
    let ok = unsafe { SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS) != 0 && SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN) != 0 };
    if !ok {
        eprintln!("texture-pack: could not lower the priority");
    }
}

fn collect(p: &Path, out: &mut Vec<PathBuf>) -> Res<()> {
    if p.is_dir() {
        for e in std::fs::read_dir(p)? {
            collect(&e?.path(), out)?;
        }
    } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("ttpf")) {
        out.push(p.to_path_buf());
    }
    Ok(())
}

/// Requested (file key -> offsets) from a DirectStorage trace.
fn read_trace(path: &Path) -> Res<HashMap<String, BTreeSet<u64>>> {
    let text = std::fs::read_to_string(path)?;
    let mut map: HashMap<String, BTreeSet<u64>> = HashMap::new();
    for line in text.lines() {
        let Some(rest) = line.split("dstorage: options ").nth(1) else { continue };
        let Some((_, rest)) = rest.split_once(", ") else { continue };
        let Some((file, rest)) = rest.split_once(" @ ") else { continue };
        if !file.to_ascii_lowercase().ends_with(".ttpf") {
            continue;
        }
        let Some(off) = rest.split(',').next().and_then(|o| u64::from_str_radix(o.trim().trim_start_matches("0x"), 16).ok()) else { continue };
        if let Some(key) = texpack::file_key(Path::new(file)) {
            map.entry(key).or_default().insert(off);
        }
    }
    Ok(map)
}

/// The textures of `t` to convert: all, or those a trace touched.
fn selection(file: &Path, t: &Ttpf, trace: Option<&HashMap<String, BTreeSet<u64>>>) -> Vec<u32> {
    let Some(trace) = trace else { return (0..t.textures.len() as u32).collect() };
    let Some(offsets) = texpack::file_key(file).and_then(|k| trace.get(&k)) else { return vec![] };
    let index = xnw_formats::origin::FileIndex::new(0, file.to_path_buf(), t.clone());
    let set: BTreeSet<u32> = offsets.iter().filter_map(|&o| index.resolve(o)).map(|r| r.texture).filter(|&x| x != xnw_formats::origin::NO_TEXTURE).collect();
    set.into_iter().collect()
}

fn open_ttpf(path: &Path) -> Res<(std::fs::File, Ttpf)> {
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let t = Ttpf::read(&mut f, len)?;
    Ok((f, t))
}

fn plan(files: &[PathBuf], trace: Option<&HashMap<String, BTreeSet<u64>>>) -> Res<()> {
    let mut total = [0u64; 3];
    let (mut textures, mut stored) = (0u64, 0u64);
    for file in files {
        let (_, t) = open_ttpf(file)?;
        let sel = selection(file, &t, trace);
        let mut sizes = [0u64; 3];
        for &i in &sel {
            let x = &t.textures[i as usize];
            stored += t.pages_of(x.addr, x.span).map(|p| t.pages[p].stored).sum::<u64>();
            for (d, s) in sizes.iter_mut().enumerate() {
                if let Some(p) = texpack::plan(x, i, d as u32) {
                    *s = s.next_multiple_of(texpack::block_align(p.entry.block)) + p.entry.block;
                }
            }
        }
        let index = (texpack::HEADER + t.textures.len() as u64 * texpack::RECORD as u64).next_multiple_of(0x1_0000);
        textures += sel.len() as u64;
        for d in 0..3 {
            total[d] += if sel.is_empty() { 0 } else { sizes[d] + index };
        }
    }
    println!(
        "plan: {} files, {textures} textures, source pages {:.1} MB; pack {:.1} MB (all mips), {:.1} MB (drop 1), {:.1} MB (drop 2)",
        files.len(),
        mb(stored),
        mb(total[0]),
        mb(total[1]),
        mb(total[2])
    );
    Ok(())
}

fn mb(b: u64) -> f64 {
    b as f64 / 1e6
}

fn convert(files: &[PathBuf], trace: Option<&HashMap<String, BTreeSet<u64>>>, out: &Path, drop: u32, threads: usize, tables: &Tables) -> Res<()> {
    let started = Instant::now();
    let totals = Totals::default();
    let errors: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());
    for file in files {
        let (_, t) = open_ttpf(file)?;
        let sel = selection(file, &t, trace);
        if sel.is_empty() {
            continue;
        }
        let writer = Writer::open(out, file, t.textures.len() as u32, drop)?;
        let selected = sel.len();
        let todo: Vec<u32> = sel.into_iter().filter(|&i| !writer.has(i)).collect();
        totals.skipped.fetch_add((selected - todo.len()) as u64, Relaxed);
        let next = AtomicUsize::new(0);
        let file_started = Instant::now();
        let before = totals.bytes_out.load(Relaxed);
        std::thread::scope(|s| {
            for _ in 0..threads.min(todo.len()) {
                s.spawn(|| {
                    let Ok(mut src) = std::fs::File::open(file) else { return };
                    loop {
                        let k = next.fetch_add(1, Relaxed);
                        let Some(&i) = todo.get(k) else { break };
                        let began = Instant::now();
                        let x = &t.textures[i as usize];
                        let result = texpack::plan(x, i, drop)
                            .ok_or_else(|| format!("format {} or {} mips not packable", x.format, x.mips))
                            .and_then(|p| texpack::convert(&t, &mut src, i, &p, Some(tables)).map(|b| (p, b)))
                            .and_then(|(p, b)| writer.add(&p, &b).map(|()| b.len()).map_err(|e| e.to_string()));
                        match result {
                            Ok(n) => {
                                totals.converted.fetch_add(1, Relaxed);
                                totals.bytes_out.fetch_add(n as u64, Relaxed);
                                totals.stored_in.fetch_add(t.pages_of(x.addr, x.span).map(|p| t.pages[p].stored).sum::<u64>(), Relaxed);
                            }
                            Err(e) => {
                                totals.failed.fetch_add(1, Relaxed);
                                let kind = format!("{} {}: {}", ttpf_kind(x.kind), x.format, e.split(':').next().unwrap_or(""));
                                let mut m = errors.lock().unwrap_or_else(|e| e.into_inner());
                                let n = m.entry(kind).or_default();
                                if *n == 0 {
                                    eprintln!("  {} texture {i} ({:016x}): {e}", file.display(), x.hash);
                                }
                                *n += 1;
                            }
                        }
                        totals.busy_ns.fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
                    }
                });
            }
        });
        println!(
            "{}: {} converted ({} already in, {} selected of {}), {:.1} MB out, {:.1} s",
            writer.path.display(),
            todo.len(),
            selected - todo.len(),
            selected,
            t.textures.len(),
            mb(totals.bytes_out.load(Relaxed) - before),
            file_started.elapsed().as_secs_f64()
        );
    }
    let busy = Duration::from_nanos(totals.busy_ns.load(Relaxed)).as_secs_f64().max(1e-9);
    let out_mb = mb(totals.bytes_out.load(Relaxed));
    println!(
        "total: {} converted, {} already in, {} failed, in {:.1} MB stored -> out {:.1} MB; {:.1} s wall on {threads} threads, {:.1} MB/s out per busy thread ({:.1} MB/s in), {:.1} MB/s overall",
        totals.converted.load(Relaxed),
        totals.skipped.load(Relaxed),
        totals.failed.load(Relaxed),
        mb(totals.stored_in.load(Relaxed)),
        out_mb,
        started.elapsed().as_secs_f64(),
        out_mb / busy,
        mb(totals.stored_in.load(Relaxed)) / busy,
        out_mb / started.elapsed().as_secs_f64()
    );
    for (k, n) in errors.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        println!("  failed {n}: {k}");
    }
    Ok(())
}

fn ttpf_kind(k: u8) -> &'static str {
    match k {
        ttpf::KIND_2D => "2D",
        ttpf::KIND_ARRAY => "array",
        ttpf::KIND_CUBE => "cube",
        ttpf::KIND_VOLUME => "volume",
        _ => "kind?",
    }
}

/// A page decoded as the DirectStorage worker does: zlib first, then BCPack.
fn runtime_page(src: &std::fs::File, p: &ttpf::Page, tables: &Tables) -> Res<Vec<u8>> {
    use std::os::windows::fs::FileExt;
    let mut raw = vec![0; p.stored as usize];
    src.seek_read(&mut raw, p.offset)?;
    let mid = if p.storage & 8 != 0 { miniz_oxide::inflate::decompress_to_vec_zlib(&raw).map_err(|e| format!("{e:?}"))? } else { raw };
    let mut out = vec![0; p.size as usize];
    match p.storage & 7 {
        0 => {
            let n = mid.len().min(out.len());
            out[..n].copy_from_slice(&mid[..n]);
        }
        bc => bcpack::decode(bc, &mid, &mut out, tables).map_err(|e| e.0)?,
    }
    Ok(out)
}

fn verify_packs(files: &[PathBuf], out: &Path, n: usize, tables: &Tables) -> Res<()> {
    let packs = texpack::open_dir(out)?;
    let mut by_format: BTreeMap<u32, Vec<(PathBuf, u32)>> = BTreeMap::new();
    for file in files {
        let Some(id) = texpack::file_id(file) else { continue };
        let (_, t) = open_ttpf(file)?;
        for (i, x) in t.textures.iter().enumerate() {
            if x.kind == ttpf::KIND_2D && x.layers == 1 && packs.lookup(id, i as u32).is_some() {
                by_format.entry(x.format).or_default().push((file.clone(), i as u32));
            }
        }
    }
    // Formats in turn, spread through each list.
    let mut picks = vec![];
    let mut round = 0;
    while picks.len() < n && by_format.values().any(|v| v.len() > round) {
        for v in by_format.values() {
            if picks.len() < n && v.len() > round {
                let step = (v.len() / (n / by_format.len().max(1)).max(1)).max(1);
                if let Some(p) = v.get((round * step) % v.len()) {
                    picks.push(p.clone());
                }
            }
        }
        round += 1;
    }
    let (mut equal, mut differ, mut mips) = (0, 0, 0);
    let mut per_format: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut cache: HashMap<PathBuf, (std::fs::File, Ttpf)> = HashMap::new();
    for (file, i) in picks {
        if !cache.contains_key(&file) {
            cache.insert(file.clone(), open_ttpf(&file)?);
        }
        let (src, t) = &cache[&file];
        let x = t.textures[i as usize];
        let e = packs.lookup(texpack::file_id(&file).ok_or("id")?, i).ok_or("entry")?;
        // The runtime: decoded pages into title memory, then the layout of
        // the title's description, detiled per mip.
        let layout = texture::Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            width: x.width,
            height: x.height,
            depth_or_array: 1,
            mips: u32::from(e.mips),
            format: x.format,
            samples: 1,
            swizzle: ttpf::swizzle(x.mode),
        })
        .map_err(|e| format!("{e:?}"))?
        .planes[0]
        .layout
        .clone();
        let mut memory = vec![0u8; layout.size.max(x.span) as usize];
        for k in t.pages_of(x.addr, x.span) {
            let p = &t.pages[k];
            let bytes = runtime_page(src, p, tables)?;
            let from = (p.addr - x.addr) as usize;
            let len = bytes.len().min(memory.len() - from);
            memory[from..from + len].copy_from_slice(&bytes[..len]);
        }
        let mut packed = vec![0u8; e.block as usize];
        let n = packs.read_mips(&e, e.stored_mips(), &mut packed)?;
        let base = e.span(e.stored_mips()).ok_or("span")?.0;
        let mut same = true;
        for m in e.stored_mips() {
            let loc = e.mip[m as usize];
            let mut want = vec![0u8; loc.bytes as usize];
            layout.detile(&memory, m, 0, &mut want, loc.row_pitch as usize, 0).map_err(|e| format!("{e:?}"))?;
            let at = (loc.offset - base) as usize;
            same &= at + want.len() <= n && packed[at..at + want.len()] == want[..];
            mips += 1;
        }
        let f = per_format.entry(x.format).or_default();
        if same {
            equal += 1;
            f.0 += 1;
        } else {
            differ += 1;
            f.1 += 1;
            println!("  differs: {} texture {i} ({:016x}) {}x{} fmt {}", file.display(), x.hash, x.width, x.height, x.format);
        }
    }
    println!("verify: {equal} textures equal byte for byte ({mips} mips), {differ} differ");
    println!("  by format: {}", per_format.iter().map(|(f, (a, b))| format!("{f}:{a}/{}", a + b)).collect::<Vec<_>>().join(" "));
    Ok(())
}

fn write_dds(files: &[PathBuf], out: &Path, n: usize) -> Res<()> {
    let packs = texpack::open_dir(out)?;
    let dir = out.join("dds");
    std::fs::create_dir_all(&dir)?;
    let mut written = 0;
    let mut formats: BTreeMap<u32, usize> = BTreeMap::new();
    for file in files {
        let Some(id) = texpack::file_id(file) else { continue };
        let (_, t) = open_ttpf(file)?;
        for (i, x) in t.textures.iter().enumerate() {
            if written >= n {
                return Ok(());
            }
            let Some(e) = packs.lookup(id, i as u32) else { continue };
            if x.width < 256 || e.stored < 2 || formats.get(&x.format).copied().unwrap_or(0) >= 2 || x.kind == ttpf::KIND_VOLUME {
                continue;
            }
            let mut buf = vec![0u8; e.block as usize];
            let len = packs.read_mips(&e, e.stored_mips(), &mut buf)?;
            let base = e.span(e.stored_mips()).ok_or("span")?.0;
            let f = texture::format(x.format).ok_or("format")?;
            let slices = if x.kind == ttpf::KIND_CUBE { 6 } else { x.layers };
            let mut data = vec![];
            // DDS order: slice by slice, mips finest first, tight rows.
            for s in 0..slices {
                for m in e.stored_mips() {
                    let loc = e.mip[m as usize];
                    let row = ((x.width >> m).max(1).div_ceil(f.block) * f.bytes) as usize;
                    let at = (loc.offset - base) as usize + s as usize * loc.slice_pitch as usize;
                    for r in 0..loc.rows as usize {
                        data.extend_from_slice(&buf[at + r * loc.row_pitch as usize..][..row]);
                    }
                }
            }
            let _ = len;
            let first = u32::from(e.first);
            let shown = ttpf::Texture { width: (x.width >> first).max(1), height: (x.height >> first).max(1), ..*x };
            let name = format!("{}_{i}_{:016x}_{}x{}_fmt{}.dds", texpack::pack_name(&texpack::file_key(file).unwrap_or_default()).trim_end_matches(".xtp"), x.hash, shown.width, shown.height, x.format);
            std::fs::write(dir.join(&name), ttpf::dds(&shown, u32::from(e.stored), slices, &data))?;
            println!("  dds {}", dir.join(&name).display());
            *formats.entry(x.format).or_default() += 1;
            written += 1;
        }
    }
    Ok(())
}
