//! `xnw translate <rpf> [--stage vs|ps|cs|ms|gs] [--validate] [--pipelines N]
//! [--fma exact|fast] [--mesh native|fallback] [--jobs N] [--dump DIR]`: every shader of the pack
//! through the translator, on all cores, with what fails counted by reason.
//!
//! Per program: IR ([`ir::build`]), resource tracing, stage I/O, structure,
//! then DXIL ([`emit::emit_with`]). `--validate` runs `dxil.dll`'s validator
//! on each emitted shader (one validator per thread); `--pipelines N`
//! creates N sampled pipelines per stage on the host device afterwards
//! (vertex with a stub pixel shader, pixel with a stub vertex shader,
//! compute alone, mesh with a stub pixel shader). `--dump DIR` writes the
//! first three shaders failing each validator reason.
//!
//! `--mesh fallback` (XNW): mesh programs emit as compute writing
//! records (`emit::mesh`); each also gets its position-only pass-through
//! vertex shader (validated with `--validate`), and `--pipelines N` makes N
//! fallback pairs: the compute pipeline, and the draw pipeline with the
//! pass-through vertex shader and the pack's pixel shader of the same
//! effect hash (a stub where none translates), plus how the pairs' pixel
//! inputs meet the mesh outputs (by row, by name).
//!
//! The walk and the tallies follow VIctory's `apps/victory/src/translate.rs`
//! (commit d2a4bae1f49533ef5cb512fd1b1ba1513f2e1989, GPL-3.0-or-later);
//! reworked for one RPF archive, per-stage counts and timing.

use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use xnw_shader::emit::host::{self, Host};
use xnw_shader::emit::{self, Fma, HostShader, MeshForm, Options, mesh, validator::Validator};
use xnw_shader::rpf9::{Archive, EntryKind};
use xnw_shader::sgd5::{self, Record};
use xnw_shader::{Container, FourCC, HwStage, ShaderError, io, ir, native_programs, resource, structure};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage: xnw translate <rpf> [--stage vs|ps|cs|ms|gs] [--validate] [--pipelines N] [--descriptors plan|gpu] \
                     [--fma exact|fast] [--mesh native|fallback] [--jobs N] [--dump DIR]";

/// Hardware stages in report order; `Es` and `Gs` are reported as `gs`.
const STAGES: [(HwStage, &str); 6] = [
    (HwStage::Vs, "vs"),
    (HwStage::Ps, "ps"),
    (HwStage::Cs, "cs"),
    (HwStage::Ms, "ms"),
    (HwStage::Es, "gs (es)"),
    (HwStage::Gs, "gs"),
];

fn stage_index(s: HwStage) -> usize {
    STAGES.iter().position(|&(x, _)| x == s).unwrap_or(0)
}

/// Counts per stage, one column per step.
#[derive(Default, Clone, Copy)]
struct Steps {
    programs: usize,
    ir: usize,
    resources: usize,
    io: usize,
    structured: usize,
    /// Every step before emission passed.
    ready: usize,
    emitted: usize,
    validated: usize,
    instructions: usize,
    wave64: usize,
    /// Emitted shaders that declare `WaveSize`.
    wave_size: usize,
    /// Emitted shaders that use the exact (double) fma path.
    doubles: usize,
}

impl Steps {
    fn add(&mut self, o: &Steps) {
        self.programs += o.programs;
        self.ir += o.ir;
        self.resources += o.resources;
        self.io += o.io;
        self.structured += o.structured;
        self.ready += o.ready;
        self.emitted += o.emitted;
        self.validated += o.validated;
        self.instructions += o.instructions;
        self.wave64 += o.wave64;
        self.wave_size += o.wave_size;
        self.doubles += o.doubles;
    }
}

#[derive(Default)]
struct Tally {
    records: usize,
    corrupt: usize,
    duplicates: usize,
    stages: [Steps; 6],
    /// Reason → programs stopped by it, and the first one.
    failed: BTreeMap<String, (usize, String)>,
    /// Emitted shaders kept for `--pipelines`, per stage.
    kept: [Vec<(String, HostShader)>; 6],
    /// `--mesh fallback`: the kept mesh programs' record layouts, by label.
    layouts: Vec<(String, mesh::Layout)>,
    /// `--mesh fallback`: pass-through vertex shaders emitted, validated.
    pass_vs: [usize; 2],
    /// Time in translation proper (IR to DXIL), summed over threads.
    busy: Duration,
    /// The same split by step: IR, resources, I/O, structure, emission.
    step: [Duration; 5],
    /// The slowest programs: (time, label, per-step times).
    slowest: Vec<(Duration, String, [Duration; 5])>,
    /// Programs and their summed time by time taken: <1 ms, <10, <100, <1 s, more.
    histogram: [(usize, Duration); 5],
    validate_time: Duration,
    dxil_bytes: usize,
}

impl Tally {
    fn fail(&mut self, reason: String, example: impl FnOnce() -> String) {
        self.failed.entry(reason).or_insert_with(|| (0, example())).0 += 1;
    }

    fn merge(&mut self, o: Tally) {
        self.records += o.records;
        self.corrupt += o.corrupt;
        self.duplicates += o.duplicates;
        for (a, b) in self.stages.iter_mut().zip(&o.stages) {
            a.add(b);
        }
        for (reason, (n, example)) in o.failed {
            self.failed.entry(reason).or_insert((0, example)).0 += n;
        }
        for (a, b) in self.kept.iter_mut().zip(o.kept) {
            a.extend(b);
        }
        self.layouts.extend(o.layouts);
        self.pass_vs[0] += o.pass_vs[0];
        self.pass_vs[1] += o.pass_vs[1];
        self.busy += o.busy;
        for (a, b) in self.step.iter_mut().zip(o.step) {
            *a += b;
        }
        for (a, b) in self.histogram.iter_mut().zip(o.histogram) {
            a.0 += b.0;
            a.1 += b.1;
        }
        self.slowest.extend(o.slowest);
        self.slowest.sort_by(|a, b| b.0.cmp(&a.0));
        self.slowest.truncate(8);
        self.validate_time += o.validate_time;
        self.dxil_bytes += o.dxil_bytes;
    }
}

struct Config {
    stage: Option<&'static str>,
    validate: bool,
    pipelines: usize,
    opts: Options,
    jobs: usize,
    dump: Option<PathBuf>,
}

fn value<'a>(args: &'a [String], flag: &str) -> Res<Option<&'a str>> {
    match args.iter().position(|a| a == flag) {
        Some(i) => Ok(Some(args.get(i + 1).ok_or(USAGE)?.as_str())),
        None => Ok(None),
    }
}

pub fn run(args: &[String]) -> Res<()> {
    let path = args.first().filter(|a| !a.starts_with("--")).ok_or(USAGE)?;
    let stage = match value(args, "--stage")? {
        None => None,
        Some(s) => Some(*["vs", "ps", "cs", "ms", "gs"].iter().find(|&&x| x == s).ok_or(USAGE)?),
    };
    let fma = match value(args, "--fma")? {
        None | Some("exact") => Fma::Exact,
        Some("fast") => Fma::Fast,
        Some(_) => return Err(USAGE.into()),
    };
    let jobs = match value(args, "--jobs")? {
        Some(n) => n.parse::<usize>()?.max(1),
        None => std::thread::available_parallelism().map_or(4, |n| n.get()),
    };
    let pipelines = match value(args, "--pipelines")? {
        Some(n) => n.parse()?,
        None => 0,
    };
    let mesh = match value(args, "--mesh")? {
        None | Some("native") => MeshForm::Native,
        Some("fallback") => MeshForm::Fallback,
        Some(_) => return Err(USAGE.into()),
    };
    // XNW: `--descriptors gpu` as XNW_DESC=gpu translates.
    let descriptors = match value(args, "--descriptors")? {
        None | Some("plan") => xnw_shader::emit::Descriptors::Plan,
        Some("gpu") => xnw_shader::emit::Descriptors::Gpu,
        Some(_) => return Err(USAGE.into()),
    };
    let cfg = Config {
        stage,
        validate: args.iter().any(|a| a == "--validate"),
        pipelines,
        opts: Options { fma, mesh, descriptors, probe: false, loop_counts: None, pagetable: false },
        jobs,
        dump: value(args, "--dump")?.map(PathBuf::from),
    };
    if let Some(d) = &cfg.dump {
        std::fs::create_dir_all(d)?;
    }
    let dll = Validator::default_path();
    if cfg.validate {
        Validator::load(dll.as_deref())?;
        println!("validator  {}", dll.as_ref().map_or("dxil.dll".into(), |p| p.display().to_string()));
    }

    let start = Instant::now();
    let data = std::fs::read(path)?;
    let archive = Archive::parse(&data)?;
    let mut records: Vec<Record<'_>> = Vec::new();
    for (i, e) in archive.entries.iter().enumerate() {
        if e.kind == EntryKind::Sgd5 {
            records.extend(sgd5::records(archive.read(i).expect("entry in bounds")));
        }
    }
    let scanned = start.elapsed();
    println!(
        "archive    {path}: {} records (read + scan {:.2} s), fma {:?}, {} threads{}",
        records.len(),
        scanned.as_secs_f64(),
        fma,
        cfg.jobs,
        cfg.stage.map_or(String::new(), |s| format!(", stage {s}"))
    );

    let walk = Instant::now();
    let total = walk_pack(&records, &cfg, dll.as_deref());
    let wall = walk.elapsed();
    print(&total, &cfg, wall);
    if cfg.pipelines > 0 {
        make_pipelines(&total, &cfg, &records)?;
    }
    Ok(())
}

fn walk_pack(records: &[Record<'_>], cfg: &Config, dll: Option<&std::path::Path>) -> Tally {
    const CHUNK: usize = 16;
    let next = AtomicUsize::new(0);
    let seen = Mutex::new(HashSet::new());
    let mut total = Tally::default();
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..cfg.jobs)
            .map(|_| {
                s.spawn(|| {
                    let mut t = Tally::default();
                    let mut validator = if cfg.validate { Validator::load(dll).ok() } else { None };
                    loop {
                        let first = next.fetch_add(CHUNK, Ordering::Relaxed);
                        if first >= records.len() {
                            break;
                        }
                        for r in &records[first..(first + CHUNK).min(records.len())] {
                            record(r, cfg, &seen, &mut t, validator.as_mut());
                        }
                    }
                    t
                })
            })
            .collect();
        for w in workers {
            total.merge(w.join().expect("translate worker"));
        }
    });
    total
}

/// A 128-bit content key (two differently seeded std hashes) for duplicate
/// DXIL parts; replaces VIctory's blake3.
fn key(bytes: &[u8]) -> (u64, u64) {
    let mut a = std::hash::DefaultHasher::new();
    bytes.hash(&mut a);
    let mut b = std::hash::DefaultHasher::new();
    0x9e37_79b9_7f4a_7c15u64.hash(&mut b);
    bytes.hash(&mut b);
    (a.finish(), b.finish())
}

fn wanted(cfg: &Config, stage: HwStage) -> bool {
    match cfg.stage {
        None => true,
        Some("gs") => matches!(stage, HwStage::Es | HwStage::Gs),
        Some(s) => STAGES[stage_index(stage)].1 == s,
    }
}

fn record(r: &Record<'_>, cfg: &Config, seen: &Mutex<HashSet<(u64, u64)>>, t: &mut Tally, validator: Option<&mut Validator>) {
    let name = || r.name.unwrap_or("?").to_owned();
    t.records += 1;
    let Some(bytes) = r.unpack() else {
        t.corrupt += 1;
        return t.fail("corrupt record".into(), name);
    };
    let Ok(container) = Container::parse(&bytes) else {
        return t.fail("not a DXBC container".into(), name);
    };
    let Some(dxil) = container.part(FourCC::DXIL).map(|p| p.data) else {
        return t.fail("no DXIL part".into(), name);
    };
    let programs = match native_programs(dxil) {
        Ok(p) => p,
        Err(e) => return t.fail(e.to_string(), name),
    };
    if !programs.iter().any(|p| wanted(cfg, p.stage)) {
        return;
    }
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(key(dxil)) {
        t.duplicates += 1;
        return;
    }
    let mut validator = validator;
    for p in programs.iter().filter(|p| wanted(cfg, p.stage)) {
        let si = stage_index(p.stage);
        let label = || format!("{} {:?}", name(), p.stage);
        let began = Instant::now();
        t.stages[si].programs += 1;
        t.stages[si].wave64 += usize::from(p.wave64);
        let mut lap = began;
        let mut mine = [Duration::ZERO; 5];
        let mut split = |t: &mut Tally, k: usize| {
            let now = Instant::now();
            t.step[k] += now - lap;
            mine[k] += now - lap;
            lap = now;
        };
        let built = ir::build(&p.patched_code(), p.wave64, p.vgprs);
        split(t, 0);
        let program = match built {
            Ok(x) => x,
            Err(e) => {
                t.busy += began.elapsed();
                t.fail(reason("ir", e), label);
                continue;
            }
        };
        let s = &mut t.stages[si];
        s.ir += 1;
        s.instructions += program.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
        let tracked = resource::track(&program, &p.layout, p.stage);
        split(t, 1);
        let res = match tracked {
            Ok(r) => {
                t.stages[si].resources += 1;
                Some(r)
            }
            Err(e) => {
                t.fail(reason("resources", e), label);
                None
            }
        };
        let raised = if matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms) {
            match io::raise(&program, p) {
                Ok(r) => {
                    t.stages[si].io += 1;
                    Some(r)
                }
                Err(e) => {
                    t.fail(reason("I/O", e), label);
                    None
                }
            }
        } else {
            t.fail("emit: no ES/GS (merged geometry) path".into(), label);
            None
        };
        split(t, 2);
        let structured = structure::structure(&program);
        split(t, 3);
        let st = match structured {
            Ok(s) => {
                t.stages[si].structured += 1;
                Some(s)
            }
            Err(e) => {
                t.fail(reason("structure", e), label);
                None
            }
        };
        let (Some(res), Some(raised), Some(st)) = (res, raised, st) else {
            t.busy += began.elapsed();
            continue;
        };
        t.stages[si].ready += 1;
        let emitted = emit::emit_with(&container, p, &program, &st, &raised, &res, cfg.opts);
        split(t, 4);
        let spent = began.elapsed();
        let bucket = [1, 10, 100, 1000].iter().position(|&ms| spent < Duration::from_millis(ms)).unwrap_or(4);
        t.histogram[bucket].0 += 1;
        t.histogram[bucket].1 += spent;
        if t.slowest.len() < 8 || spent > t.slowest[t.slowest.len() - 1].0 {
            t.slowest.push((spent, label(), mine));
            t.slowest.sort_by(|a, b| b.0.cmp(&a.0));
            t.slowest.truncate(8);
        }
        t.busy += began.elapsed();
        let mut shader = match emitted {
            Ok(x) => x,
            Err(e) => {
                t.fail(reason("emit", e), label);
                continue;
            }
        };
        let s = &mut t.stages[si];
        s.emitted += 1;
        s.wave_size += usize::from(declares_wave_size(&shader.dxil));
        s.doubles += usize::from(uses_doubles(&shader.dxil));
        t.dxil_bytes += shader.dxil.len();
        if let Some(v) = validator.as_deref_mut() {
            let began = Instant::now();
            let verdict = v.validate(&mut shader.dxil);
            t.validate_time += began.elapsed();
            match verdict {
                Ok(()) => t.stages[si].validated += 1,
                Err(text) => {
                    let first = text
                        .lines()
                        .find(|l| l.contains("error"))
                        .unwrap_or(text.lines().next().unwrap_or(""))
                        .trim()
                        .to_owned();
                    let why = format!("validator: {}", clip(&first, 160));
                    if let Some(dir) = &cfg.dump {
                        let n = t.failed.get(&why).map_or(0, |f| f.0);
                        if n < 3 {
                            let stem: String = first.chars().filter(char::is_ascii_alphanumeric).take(40).collect();
                            let file = dir.join(format!("{stem}_{n}"));
                            let _ = std::fs::write(file.with_extension("dxil"), &shader.dxil);
                            let _ = std::fs::write(file.with_extension("txt"), format!("{}\n{text}", label()));
                        }
                    }
                    t.fail(why, label);
                }
            }
        }
        let fallback = p.stage == HwStage::Ms && cfg.opts.mesh == MeshForm::Fallback;
        let layout = if fallback {
            match mesh::record_layout(&bytes) {
                Ok(l) => Some(l),
                Err(e) => {
                    t.fail(reason("record layout", e), label);
                    None
                }
            }
        } else {
            None
        };
        if let Some(l) = &layout {
            match mesh::passthrough_vs(l, None) {
                Ok(mut vs) => {
                    t.pass_vs[0] += 1;
                    if let Some(v) = validator.as_deref_mut() {
                        match v.validate(&mut vs) {
                            Ok(()) => t.pass_vs[1] += 1,
                            Err(text) => {
                                let first = text.lines().find(|l| l.contains("error")).unwrap_or("").trim().to_owned();
                                t.fail(format!("pass-through vs validator: {}", clip(&first, 160)), label);
                            }
                        }
                    }
                }
                Err(e) => t.fail(reason("pass-through vs", e), label),
            }
        }
        if cfg.pipelines > 0 && t.kept[si].len() < cfg.pipelines && (!fallback || layout.is_some()) {
            if let Some(l) = layout {
                t.layouts.push((label(), l));
            }
            t.kept[si].push((label(), shader));
        }
    }
}

fn clip(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `step: what`, without the instruction offset, so equal causes group.
fn reason(step: &str, e: ShaderError) -> String {
    match e {
        ShaderError::Unsupported { what, .. } => format!("{step}: {what}"),
        ShaderError::Isa { reason, .. } => format!("{step}: {reason}"),
        e => format!("{step}: {e}"),
    }
}

/// The SFI0 feature bits of a container (0 without the part).
fn features(dxil: &[u8]) -> u64 {
    Container::parse(dxil)
        .ok()
        .and_then(|c| c.part(FourCC::SFI0).map(|p| p.data.to_vec()))
        .and_then(|d| d.get(..8).map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes"))))
        .unwrap_or(0)
}

fn uses_doubles(dxil: &[u8]) -> bool {
    features(dxil) & 1 != 0
}

/// `WaveSize` is in the PSV0 runtime info (min = max wave lanes), which our
/// writer fills only when the entry declares it.
fn declares_wave_size(dxil: &[u8]) -> bool {
    let Ok(c) = Container::parse(dxil) else { return false };
    let Some(psv) = c.part(FourCC::PSV0) else { return false };
    // PSV runtime info 0: stage-specific union (16 B), then min and max
    // wave lane counts.
    let d = psv.data;
    let at = 4 + 16;
    let get = |o: usize| d.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")));
    matches!((get(at), get(at + 4)), (Some(lo), Some(hi)) if lo == hi && (lo == 32 || lo == 64))
}

fn print(t: &Tally, cfg: &Config, wall: Duration) {
    println!(
        "records    {} read, {} corrupt, {} duplicate DXIL parts skipped",
        t.records, t.corrupt, t.duplicates
    );
    println!(
        "\n{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8}",
        "stage", "programs", "ir", "res", "io", "struct", "ready", "emitted", "validated", "wavesize", "doubles", "wave64"
    );
    let mut sum = Steps::default();
    for ((_, name), s) in STAGES.iter().zip(&t.stages) {
        if s.programs == 0 {
            continue;
        }
        sum.add(s);
        println!(
            "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8}",
            name,
            s.programs,
            s.ir,
            s.resources,
            s.io,
            s.structured,
            s.ready,
            s.emitted,
            if cfg.validate { s.validated.to_string() } else { "-".into() },
            s.wave_size,
            s.doubles,
            s.wave64
        );
    }
    println!(
        "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>8}",
        "total",
        sum.programs,
        sum.ir,
        sum.resources,
        sum.io,
        sum.structured,
        sum.ready,
        sum.emitted,
        if cfg.validate { sum.validated.to_string() } else { "-".into() },
        sum.wave_size,
        sum.doubles,
        sum.wave64
    );
    if cfg.opts.mesh == MeshForm::Fallback {
        println!(
            "
mesh       fallback form (compute + records); pass-through vertex shaders {} emitted, {} validated",
            t.pass_vs[0],
            if cfg.validate { t.pass_vs[1].to_string() } else { "-".into() }
        );
    }
    println!(
        "\nIR         {} instructions; DXIL {:.1} MB emitted",
        sum.instructions,
        t.dxil_bytes as f64 / 1e6
    );
    let secs = wall.as_secs_f64();
    println!(
        "wall       {:.2} s on {} threads: {:.0} programs/s, {:.0} emitted/s (translation busy {:.1} s, \
         validation {:.1} s, summed over threads)",
        secs,
        cfg.jobs,
        sum.programs as f64 / secs,
        sum.emitted as f64 / secs,
        t.busy.as_secs_f64(),
        t.validate_time.as_secs_f64()
    );
    let st: Vec<String> = ["ir", "resources", "io", "structure", "emit"]
        .iter()
        .zip(t.step)
        .map(|(n, d)| format!("{n} {:.1}", d.as_secs_f64()))
        .collect();
    println!("steps      {} s (summed over threads)", st.join(", "));
    let h: Vec<String> = ["<1ms", "<10ms", "<100ms", "<1s", ">=1s"]
        .iter()
        .zip(t.histogram)
        .map(|(n, (k, d))| format!("{n} {k} ({:.1} s)", d.as_secs_f64()))
        .collect();
    println!("per program {}", h.join(", "));
    for (d, label, steps) in &t.slowest {
        let ms: Vec<String> = steps.iter().map(|d| format!("{:.0}", d.as_secs_f64() * 1e3)).collect();
        println!("  slow     {:.2} s  {label}  (ms per step {})", d.as_secs_f64(), ms.join("/"));
    }
    let mut failed: Vec<_> = t.failed.iter().collect();
    failed.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    println!("\nfailures   {} reasons, top 20 (programs, reason, first example)", failed.len());
    for (reason, (n, example)) in failed.iter().take(20) {
        println!("  {n:>7}  {reason}  (e.g. {example})");
    }
}

/// Creates up to `N` pipelines per stage from the kept shaders, evenly
/// spread over what was kept, on the host device.
fn make_pipelines(t: &Tally, cfg: &Config, records: &[Record<'_>]) -> Res<()> {
    let agility = std::env::var_os("XNW_AGILITY").map(PathBuf::from);
    let gpu = xnw_gpu::Host::new(agility.as_deref())?;
    println!(
        "\npipelines  {} (shader model {}.{}, {})",
        gpu.caps.adapter,
        gpu.caps.shader_model.0,
        gpu.caps.shader_model.1,
        gpu.caps.agility.map_or("inbox".to_string(), |v| format!("Agility SDK {v}"))
    );
    let h = Host::from_device(gpu.device.clone());
    let stub_ps = host::stub_ps();
    for ((stage, name), kept) in STAGES.iter().zip(&t.kept) {
        if kept.is_empty() {
            continue;
        }
        let mut reasons: BTreeMap<String, (usize, String)> = BTreeMap::new();
        let mut made = 0;
        let began = Instant::now();
        // Each thread kept its first N: take N spread over all of them.
        let step = kept.len().div_ceil(cfg.pipelines).max(1);
        if *stage == HwStage::Ms && cfg.opts.mesh == MeshForm::Fallback {
            fallback_pairs(t, cfg, records, &h, kept, step);
            continue;
        }
        for (label, shader) in kept.iter().step_by(step).take(cfg.pipelines) {
            let r = match stage {
                HwStage::Vs => h.pipeline(shader, &stub_ps).map(drop),
                HwStage::Cs => h.compute_pipeline(shader).map(drop),
                HwStage::Ms => h.mesh_pipeline(shader, &stub_ps).map(drop),
                _ => host::stub_vs(shader).and_then(|vs| h.pipeline(&vs, shader).map(drop)),
            };
            match r {
                Ok(()) => made += 1,
                Err(e) => reasons.entry(clip(e.trim(), 160).to_owned()).or_insert((0, label.clone())).0 += 1,
            }
        }
        let n = kept.iter().step_by(step).take(cfg.pipelines).count();
        println!(
            "  {name:<8} {made} of {n} created ({:.1} ms each)",
            began.elapsed().as_secs_f64() * 1e3 / n.max(1) as f64
        );
        for (e, (k, ex)) in &reasons {
            println!("      {k:>5}  {e}  (e.g. {ex})");
        }
    }
    Ok(())
}

/// The effect hash in a record name (`..._<hash>_Wrapped`).
fn effect_hash(name: &str) -> Option<&str> {
    name.rsplit('_').nth(1).filter(|h| h.len() == 8 && h.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// The first pixel program of the pack whose record has effect hash
/// `hash`, translated.
fn paired_ps(records: &[Record<'_>], index: &BTreeMap<String, Vec<usize>>, hash: &str, opts: Options) -> Option<(String, HostShader)> {
    for &i in index.get(hash)? {
        let r = &records[i];
        let Some(bytes) = r.unpack() else { continue };
        if let Ok(ps) = emit::translate_with(&bytes, opts)
            && ps.stage == HwStage::Ps
        {
            return Some((r.name.unwrap_or("?").to_owned(), ps));
        }
    }
    None
}

/// `--mesh fallback --pipelines N`: compute pipeline plus draw pipeline per
/// sampled mesh program, with the pixel program of the same effect.
fn fallback_pairs(t: &Tally, cfg: &Config, records: &[Record<'_>], h: &Host, kept: &[(String, HostShader)], step: usize) {
    let mut index: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, r) in records.iter().enumerate() {
        if let Some(name) = r.name
            && name.contains("PS")
            && let Some(hash) = effect_hash(name)
        {
            index.entry(hash.to_owned()).or_default().push(i);
        }
    }
    let stub = host::stub_ps();
    let (mut made, mut paired, mut n) = (0, 0, 0);
    let mut links = [0usize; 4];
    let mut reasons: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let began = Instant::now();
    for (label, cs) in kept.iter().step_by(step).take(cfg.pipelines) {
        n += 1;
        let Some((_, layout)) = t.layouts.iter().find(|l| &l.0 == label) else { continue };
        let ps = label.split(' ').next().and_then(effect_hash).and_then(|hash| paired_ps(records, &index, hash, cfg.opts));
        if let Some((_, p)) = &ps {
            paired += 1;
            if let Ok(k) = mesh::link_stats(layout, &p.dxil) {
                for (a, b) in links.iter_mut().zip(k) {
                    *a += b;
                }
            }
        }
        let ps = ps.as_ref().map_or(&stub, |p| &p.1);
        let r = mesh::passthrough_vs(layout, Some(&ps.dxil))
            .map_err(|e| format!("pass-through vs: {e}"))
            .and_then(|vs| h.fallback_pipelines(cs, &vs, ps).map(drop));
        match r {
            Ok(()) => made += 1,
            Err(e) => reasons.entry(clip(e.trim(), 160).to_owned()).or_insert((0, label.clone())).0 += 1,
        }
    }
    println!(
        "  ms (fallback) {made} of {n} pairs created, {paired} with the effect's pixel shader ({:.1} ms each)",
        began.elapsed().as_secs_f64() * 1e3 / n.max(1) as f64
    );
    println!(
        "      pixel inputs {}: a mesh output at the same row {}, one of the same name {} (at the same row {})",
        links[0], links[1], links[2], links[3]
    );
    for (e, (k, ex)) in &reasons {
        println!("      {k:>5}  {e}  (e.g. {ex})");
    }
}
