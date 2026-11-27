//! XNW `shader-gate` (PIPE1): translates a selection of the shader pack's
//! programs with the runtime's options (`emit::runtime_options`), writes one
//! manifest row per program, and validates only the DXIL that changed from
//! a baseline manifest (an equal DXIL hash copies the baseline's verdict).
//! Sharded by hash, resumable, stops early on regressions.
//!
//! ```text
//! shader-gate (--all | --drawn | --list F) --out M.tsv [--base B.tsv] [--shard i/n]
//!             [--pack P] [--threads N] [--max-regress N] [--keep DIR] [--no-validate]
//! shader-gate --merge a.tsv b.tsv ... > manifest.tsv
//! ```
//!
//! Row (TSV, appended and flushed one by one): hash, name, stage, status
//! (`ok`, `refused`, `invalid`), DXIL hash (`xnw_core::hash128`), DXIL
//! bytes, instructions, checked (`validated`, `base`: verdict copied from the
//! baseline, `none`), translate and validate microseconds, reason. The first
//! line names the build (git head, hash of this binary and of the switches
//! the translation reads); `--out` holding rows of the same build resumes,
//! another build starts the file again. Exit: 0 no regression, 1
//! regressions under `--max-regress` (25), 3 stopped early at the limit,
//! 2 usage or input error.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use xnw_shader::emit::validator::Validator;
use xnw_shader::rpf9::{Archive, EntryKind};
use xnw_shader::{Container, FourCC, HwStage, sgd5};

const COLUMNS: &str = "hash\tname\tstage\tstatus\tdxil\tbytes\tinsts\tchecked\ttranslate_us\tvalidate_us\treason";
const NCOL: usize = 11;

/// One manifest row.
#[derive(Clone, Debug, Default)]
struct Row {
    hash: String,
    name: String,
    stage: String,
    status: String,
    dxil: String,
    bytes: usize,
    insts: usize,
    checked: String,
    translate_us: u64,
    validate_us: u64,
    reason: String,
    /// CPU cycles of this thread in the translation (not in the manifest):
    /// unlike `translate_us`, the same on a loaded box.
    cycles: u64,
}

impl Row {
    fn line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            self.hash, self.name, self.stage, self.status, self.dxil, self.bytes, self.insts, self.checked, self.translate_us, self.validate_us, self.reason
        )
    }

    fn parse(line: &str) -> Option<Row> {
        let f: Vec<&str> = line.trim_end_matches(['\r', '\n']).split('\t').collect();
        if f.len() != NCOL || f[0].len() != 32 || line.starts_with('#') {
            return None;
        }
        Some(Row {
            hash: f[0].into(),
            name: f[1].into(),
            stage: f[2].into(),
            status: f[3].into(),
            dxil: f[4].into(),
            bytes: f[5].parse().ok()?,
            insts: f[6].parse().ok()?,
            checked: f[7].into(),
            translate_us: f[8].parse().ok()?,
            validate_us: f[9].parse().ok()?,
            reason: f[10].into(),
            cycles: 0,
        })
    }

    fn ok(&self) -> bool {
        self.status == "ok"
    }
}

/// A manifest file: its first line and its rows (lines that do not parse,
/// such as a row cut short by a kill, are left out).
fn read_manifest(path: &std::path::Path) -> std::io::Result<(String, Vec<Row>)> {
    let text = std::fs::read_to_string(path)?;
    let first = text.lines().next().unwrap_or("").to_owned();
    Ok((first, text.lines().filter_map(Row::parse).collect()))
}

/// What a row is against the baseline.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Same,
    ChangedOk,
    NewlyFailing,
    NewlyOk,
    ChangedReason,
    NoBase,
}

fn classify(base: Option<&Row>, now: &Row) -> Class {
    let Some(b) = base else { return Class::NoBase };
    match (b.ok(), now.ok()) {
        (true, true) if b.dxil == now.dxil => Class::Same,
        (true, true) => Class::ChangedOk,
        (true, false) => Class::NewlyFailing,
        (false, true) => Class::NewlyOk,
        (false, false) if b.status == now.status && b.reason == now.reason && b.dxil == now.dxil => Class::Same,
        (false, false) => Class::ChangedReason,
    }
}

#[derive(Default)]
struct Tally {
    rows: usize,
    same: usize,
    changed_ok: usize,
    newly_failing: usize,
    newly_ok: usize,
    changed_reason: usize,
    no_base: usize,
    ok: usize,
    refused: usize,
    invalid: usize,
    validated: usize,
    copied: usize,
    translate_us: u64,
    validate_us: u64,
    cycles: u64,
    /// The newly failing rows, in the order found.
    regressions: Vec<Row>,
}

impl Tally {
    fn add(&mut self, class: Class, r: &Row) {
        self.rows += 1;
        match class {
            Class::Same => self.same += 1,
            Class::ChangedOk => self.changed_ok += 1,
            Class::NewlyFailing => {
                self.newly_failing += 1;
                self.regressions.push(r.clone());
            }
            Class::NewlyOk => self.newly_ok += 1,
            Class::ChangedReason => self.changed_reason += 1,
            Class::NoBase => self.no_base += 1,
        }
        match r.status.as_str() {
            "ok" => self.ok += 1,
            "refused" => self.refused += 1,
            _ => self.invalid += 1,
        }
        match r.checked.as_str() {
            "validated" => self.validated += 1,
            "base" => self.copied += 1,
            _ => {}
        }
        self.translate_us += r.translate_us;
        self.validate_us += r.validate_us;
        self.cycles += r.cycles;
    }

    fn summary(&self, total: usize, secs: f64) -> String {
        format!(
            "shader-gate: [{secs:.1} s] rows {}/{total}: same {}, changed-ok {}, NEWLY FAILING {}, newly ok {}, changed reason {}, no base {}; ok {} refused {} invalid {}; validated {} copied {}; cpu translate {:.1} s ({:.1} G cycles on the thread) validate {:.1} s",
            self.rows,
            self.same,
            self.changed_ok,
            self.newly_failing,
            self.newly_ok,
            self.changed_reason,
            self.no_base,
            self.ok,
            self.refused,
            self.invalid,
            self.validated,
            self.copied,
            self.translate_us as f64 / 1e6,
            self.cycles as f64 / 1e9,
            self.validate_us as f64 / 1e6
        )
    }

    fn print_regressions(&self, n: usize) {
        for r in self.regressions.iter().take(n) {
            eprintln!("  NEWLY FAILING {} {} {} {}: {}", r.hash, r.name, r.stage, r.status, r.reason);
        }
    }
}

struct Args {
    select: Select,
    out: Option<PathBuf>,
    base: Option<PathBuf>,
    pack: Option<PathBuf>,
    shard: (u128, u128),
    threads: usize,
    max_regress: usize,
    keep: Option<PathBuf>,
    validate: bool,
    /// Per program, the milliseconds of each translation step (stderr).
    stages: bool,
}

enum Select {
    All,
    Drawn,
    List(PathBuf),
}

fn usage() -> ! {
    eprintln!("usage: shader-gate (--all | --drawn | --list F) --out M.tsv [--base B.tsv] [--shard i/n] [--pack P] [--threads N] [--max-regress N] [--keep DIR] [--no-validate] [--stages]");
    eprintln!("       shader-gate --merge a.tsv b.tsv ... > manifest.tsv");
    std::process::exit(2)
}

fn fail(msg: &str) -> ! {
    eprintln!("shader-gate: {msg}");
    std::process::exit(2)
}

/// The first existing path of `candidates`.
fn first_existing(candidates: &[&str]) -> Option<PathBuf> {
    candidates.iter().map(PathBuf::from).find(|p| p.exists())
}

/// The 32-digit hashes of a list file (the first word of a line, or the
/// text before `|`; `#` comments).
fn read_list(path: &std::path::Path) -> HashSet<String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| fail(&format!("{}: {e}", path.display())));
    text.lines()
        .filter_map(|l| l.split(['|', '\t', ' ']).next())
        .map(|h| h.trim().to_ascii_lowercase())
        .filter(|h| h.len() == 32 && h.bytes().all(|b| b.is_ascii_hexdigit()))
        .collect()
}

/// The runtime's name of a container's stage (what `runtime_options`
/// takes), or `-` when it has no program the translator takes.
fn stage_of(container: &[u8]) -> &'static str {
    let stage = Container::parse(container)
        .ok()
        .and_then(|c| c.part(FourCC::DXIL).map(|p| p.data))
        .and_then(|d| xnw_shader::native_programs(d).ok())
        .and_then(|ps| ps.into_iter().map(|p| p.stage).find(|s| matches!(s, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms)));
    match stage {
        Some(HwStage::Vs) => "vertex",
        Some(HwStage::Ps) => "pixel",
        Some(HwStage::Cs) => "compute",
        Some(HwStage::Ms) => "mesh",
        _ => "-",
    }
}

/// A reason as one TSV cell: its first line, tabs out, at most 200 chars.
fn cell(text: &str) -> String {
    let line = text.lines().next().unwrap_or("").replace('\t', " ");
    let line = line.trim();
    if line.is_empty() { "-".into() } else { line.chars().take(200).collect() }
}

/// The runtime's options, and with `XNW_ONE_RETURN=1` the bridge's form of a
/// pixel program (`Options::one_return`, KILL1): its returns joined.
/// `XNW_CONFINE=1`: the bridge's confined form (`Options::confine`, BND1).
fn gate_options(stage: &str, h: u128) -> xnw_shader::emit::Options {
    let mut o = xnw_shader::emit::runtime_options(stage, h, false, None);
    o.one_return = stage == "pixel" && std::env::var("XNW_ONE_RETURN").is_ok_and(|v| v.trim() == "1");
    // BND1: `XNW_CONFINE=1`, the bridge's confined form of every program (`Options::confine`); its
    // pixel loop budget is `XNW_LOOP_LIMIT_PIXEL=16`.
    o.confine = std::env::var("XNW_CONFINE").is_ok_and(|v| v.trim() == "1");
    o
}

/// The build line: the git head of the current folder, the hash of this
/// binary and of the switches a translation reads.
fn build_line(select: &str, shard: (u128, u128)) -> (String, String) {
    let exe = std::env::current_exe().ok().and_then(|p| std::fs::read(p).ok()).unwrap_or_default();
    let switches: Vec<String> = ["XNW_DESC", "XNW_FMA", "XNW_FMA_EXACT", "XNW_PAGETABLE", "XNW_HEAP_NONUNIFORM", "XNW_HEAP_UNIFORMITY", "XNW_ADAPTER", "XNW_SHADER_OPT", "XNW_WAVE_SIZE", "XNW_LOOP_LIMIT_PIXEL", "XNW_LOOP_LIMIT_COMPUTE", "XNW_LOOP_LIMIT_MESH", "XNW_LOOP_LIMIT_VERTEX", "XNW_ONE_RETURN", "XNW_CONFINE"]
        .iter()
        .map(|k| format!("{k}={}", std::env::var(k).unwrap_or_default()))
        .collect();
    let build = format!("{:016x}", xnw_core::hash128(&[&exe, switches.join(";").as_bytes()]) as u64);
    let head = std::process::Command::new("git")
        .args(["rev-parse", "--short=10", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(|| "-".into(), |o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    (build.clone(), format!("#shader-gate build={build} head={head} select={select} shard={}/{} version={}", shard.0, shard.1, xnw_shader::emit::VERSION))
}

fn build_of(first: &str) -> Option<&str> {
    first.split(' ').find_map(|w| w.strip_prefix("build="))
}

fn merge(files: &[String]) -> i32 {
    let mut rows: HashMap<String, Row> = HashMap::new();
    let mut builds: Vec<String> = Vec::new();
    let mut first_line = String::new();
    for f in files {
        let (first, rs) = read_manifest(std::path::Path::new(f)).unwrap_or_else(|e| fail(&format!("{f}: {e}")));
        let b = build_of(&first).unwrap_or("-").to_owned();
        if !builds.contains(&b) {
            builds.push(b);
        }
        if first_line.is_empty() {
            first_line = first;
        }
        eprintln!("shader-gate: {f}: {} rows", rs.len());
        for r in rs {
            rows.entry(r.hash.clone()).or_insert(r);
        }
    }
    if builds.len() > 1 {
        fail(&format!("the shards come from different builds: {}", builds.join(", ")));
    }
    let mut rows: Vec<Row> = rows.into_values().collect();
    rows.sort_by(|a, b| a.hash.cmp(&b.hash));
    let first = first_line.split(' ').map(|w| if w.starts_with("shard=") { "shard=merged" } else { w }).collect::<Vec<_>>().join(" ");
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{first}");
    let _ = writeln!(out, "{COLUMNS}");
    for r in &rows {
        let _ = out.write_all(r.line().as_bytes());
    }
    eprintln!("shader-gate: merged {} rows from {} files", rows.len(), files.len());
    0
}

fn parse_args() -> Args {
    let mut a = Args { select: Select::All, out: None, base: None, pack: None, shard: (0, 1), threads: 6, max_regress: 25, keep: None, validate: true, stages: false };
    let mut selected = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--all" => (a.select, selected) = (Select::All, true),
            "--drawn" => (a.select, selected) = (Select::Drawn, true),
            "--list" => (a.select, selected) = (Select::List(val().into()), true),
            "--out" => a.out = Some(val().into()),
            "--base" => a.base = Some(val().into()),
            "--pack" => a.pack = Some(val().into()),
            "--keep" => a.keep = Some(val().into()),
            "--threads" => a.threads = val().parse().unwrap_or_else(|_| usage()),
            "--max-regress" => a.max_regress = val().parse().unwrap_or_else(|_| usage()),
            "--no-validate" => a.validate = false,
            "--stages" => a.stages = true,
            "--shard" => {
                let v = val();
                let (i, n) = v.split_once('/').unwrap_or_else(|| usage());
                a.shard = (i.parse().unwrap_or_else(|_| usage()), n.parse().unwrap_or_else(|_| usage()));
                if a.shard.1 == 0 || a.shard.0 >= a.shard.1 {
                    usage();
                }
            }
            _ => usage(),
        }
    }
    if !selected || a.out.is_none() || a.threads == 0 || a.max_regress == 0 {
        usage();
    }
    a
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().is_some_and(|a| a == "--merge") {
        if argv.len() < 2 {
            usage();
        }
        std::process::exit(merge(&argv[1..]));
    }
    let args = parse_args();
    let started = Instant::now();
    let out_path = args.out.clone().unwrap_or_default();

    // The selection.
    let (select_name, wanted): (&str, Option<HashSet<String>>) = match &args.select {
        Select::All => ("all", None),
        Select::Drawn => {
            let p = std::env::var_os("XNW_SHADER_DRAWN")
                .map(PathBuf::from)
                .or_else(|| first_existing(&[r"out\shader-gate\drawn.txt"]))
                .unwrap_or_else(|| fail("--drawn: no list (XNW_SHADER_DRAWN, out\\shader-gate\\drawn.txt)"));
            ("drawn", Some(read_list(&p)))
        }
        Select::List(p) => ("list", Some(read_list(p))),
    };

    // The baseline.
    let base: HashMap<String, Row> = match &args.base {
        Some(p) => {
            let (first, rows) = read_manifest(p).unwrap_or_else(|e| fail(&format!("--base {}: {e}", p.display())));
            eprintln!("shader-gate: base {} ({} rows; {first})", p.display(), rows.len());
            rows.into_iter().map(|r| (r.hash.clone(), r)).collect()
        }
        None => HashMap::new(),
    };

    // Resume or start the output.
    let (build, header) = build_line(select_name, args.shard);
    let mut done: HashMap<String, Row> = HashMap::new();
    if let Ok((first, rows)) = read_manifest(&out_path) {
        if build_of(&first) == Some(build.as_str()) {
            done = rows.into_iter().map(|r| (r.hash.clone(), r)).collect();
            eprintln!("shader-gate: resuming {} ({} rows of build {build})", out_path.display(), done.len());
        } else {
            eprintln!("shader-gate: {} is of another build ({}): starting again", out_path.display(), build_of(&first).unwrap_or("-"));
        }
    }
    let file = if done.is_empty() {
        if let Some(d) = out_path.parent().filter(|d| !d.as_os_str().is_empty()) {
            let _ = std::fs::create_dir_all(d);
        }
        let mut f = std::fs::File::create(&out_path).unwrap_or_else(|e| fail(&format!("{}: {e}", out_path.display())));
        let _ = writeln!(f, "{header}");
        let _ = writeln!(f, "{COLUMNS}");
        f
    } else {
        let mut f = std::fs::OpenOptions::new().append(true).open(&out_path).unwrap_or_else(|e| fail(&format!("{}: {e}", out_path.display())));
        // A row cut short by a kill: end its line, it parses as nothing.
        if std::fs::read(&out_path).is_ok_and(|t| t.last() != Some(&b'\n')) {
            let _ = f.write_all(b"\n");
        }
        f
    };
    if let Some(k) = &args.keep {
        let _ = std::fs::create_dir_all(k);
    }

    // The pack's records.
    let pack = args
        .pack
        .clone()
        .or_else(|| std::env::var_os("XNW_SHADER_PACK").map(PathBuf::from))
        .or_else(|| first_existing(&[r"game\shaders.rpf"]))
        .unwrap_or_else(|| fail("no pack (--pack, XNW_SHADER_PACK)"));
    let data = std::fs::read(&pack).unwrap_or_else(|e| fail(&format!("{}: {e}", pack.display())));
    let archive = Archive::parse(&data).unwrap_or_else(|e| fail(&format!("{}: {e:?}", pack.display())));
    let mut records = Vec::new();
    for (i, e) in archive.entries.iter().enumerate() {
        if e.kind == EntryKind::Sgd5
            && let Some(bytes) = archive.read(i)
        {
            records.extend(sgd5::records(bytes));
        }
    }
    let read_secs = started.elapsed().as_secs_f64();

    // The validator: one per thread.
    let dxil_dll = Validator::default_path();
    if args.validate {
        match Validator::load(dxil_dll.as_deref()) {
            Ok(_) => eprintln!("shader-gate: validator {}", dxil_dll.as_ref().map_or("dxil.dll".into(), |p| p.display().to_string())),
            Err(e) => fail(&format!("no validator ({e}); XNW_DXIL names dxil.dll, --no-validate runs without")),
        }
    }

    let expected = wanted.as_ref().map(HashSet::len);
    eprintln!(
        "shader-gate: {} records in {} ({read_secs:.1} s), selection {select_name}{}, shard {}/{}, {} threads, {header}",
        records.len(),
        pack.display(),
        expected.map_or(String::new(), |n| format!(" ({n} hashes)")),
        args.shard.0,
        args.shard.1,
        args.threads
    );

    // Rows already written count first.
    let tally = Mutex::new(Tally::default());
    {
        let mut t = tally.lock().unwrap_or_else(|e| e.into_inner());
        let mut rows: Vec<&Row> = done.values().collect();
        rows.sort_by(|a, b| a.hash.cmp(&b.hash));
        for r in rows {
            t.add(classify(base.get(&r.hash), r), r);
        }
    }
    // The work: every selected program of this shard not written yet,
    // the slowest first by the baseline's translation time (the tail of a
    // few programs that take seconds packs into the threads' idle time).
    let seen: Mutex<HashSet<u128>> = Mutex::new(HashSet::new());
    let work: Mutex<Vec<(usize, u128, u64)>> = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..args.threads {
            s.spawn(|| {
                let mut mine = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(r) = records.get(i) else { break };
                    let Some(bytes) = r.unpack() else { continue };
                    let h = xnw_core::hash128(&[&bytes]);
                    if h % args.shard.1 != args.shard.0 {
                        continue;
                    }
                    let hex = format!("{h:032x}");
                    if wanted.as_ref().is_some_and(|w| !w.contains(&hex)) || !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(h) || done.contains_key(&hex) {
                        continue;
                    }
                    mine.push((i, h, base.get(&hex).map_or(0, |b| b.translate_us)));
                }
                work.lock().unwrap_or_else(|e| e.into_inner()).append(&mut mine);
            });
        }
    });
    let mut work = work.into_inner().unwrap_or_else(|e| e.into_inner());
    work.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    let total_hint = work.len() + done.len();
    eprintln!("shader-gate: {} programs to translate, {} written before ({:.1} s)", work.len(), done.len(), started.elapsed().as_secs_f64());
    let file = Mutex::new(file);
    let next = AtomicUsize::new(0);
    // Rows written before may already hold the limit.
    let stop = AtomicBool::new(tally.lock().unwrap_or_else(|e| e.into_inner()).newly_failing >= args.max_regress);
    let selected = AtomicUsize::new(done.len());
    let steps: Mutex<Vec<(&str, f64, u32)>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..args.threads {
            s.spawn(|| {
                let mut validator = if args.validate { Validator::load(dxil_dll.as_deref()).ok() } else { None };
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let Some(&(i, h, _)) = work.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                    let r = &records[i];
                    let Some(bytes) = r.unpack() else { continue };
                    let hex = format!("{h:032x}");
                    selected.fetch_add(1, Ordering::Relaxed);
                    let row = gate_one(&bytes, h, hex.clone(), r.name.unwrap_or("-"), base.get(&hex), validator.as_mut());
                    if args.stages {
                        eprintln!("shader-gate: stages {hex} {}: {:.1} M cycles; {}", r.name.unwrap_or("-"), row.cycles as f64 / 1e6, stage_times(&bytes, h));
                    }
                    let class = classify(base.get(&row.hash), &row);
                    if class == Class::NewlyFailing {
                        eprintln!("shader-gate: NEWLY FAILING {} {} {} {}: {}", row.hash, row.name, row.stage, row.status, row.reason);
                        if let Some(k) = &args.keep {
                            let _ = std::fs::write(k.join(format!("{}.bin", row.hash)), &bytes);
                        }
                    }
                    {
                        let mut f = file.lock().unwrap_or_else(|e| e.into_inner());
                        let _ = f.write_all(row.line().as_bytes());
                        let _ = f.flush();
                    }
                    let mut t = tally.lock().unwrap_or_else(|e| e.into_inner());
                    t.add(class, &row);
                    if t.rows % 1000 == 0 {
                        eprintln!("{}", t.summary(total_hint, started.elapsed().as_secs_f64()));
                    }
                    if t.newly_failing >= args.max_regress && !stop.swap(true, Ordering::Relaxed) {
                        eprintln!("shader-gate: STOPPED: {} programs newly fail (--max-regress {})", t.newly_failing, args.max_regress);
                    }
                }
                let mut all = steps.lock().unwrap_or_else(|e| e.into_inner());
                for (name, ms, n) in xnw_shader::emit::take_laps() {
                    match all.iter_mut().find(|e| e.0 == name) {
                        Some(e) => (e.1, e.2) = (e.1 + ms, e.2 + n),
                        None => all.push((name, ms, n)),
                    }
                }
            });
        }
    });
    let t = tally.into_inner().unwrap_or_else(|e| e.into_inner());
    let secs = started.elapsed().as_secs_f64();
    eprintln!("{}", t.summary(total_hint, secs));
    if !args.stages {
        let mut all = steps.into_inner().unwrap_or_else(|e| e.into_inner());
        all.sort_by(|a, b| b.1.total_cmp(&a.1));
        let all: Vec<String> = all.iter().map(|(n, ms, k)| format!("{n} {:.1} s/{k}", ms / 1e3)).collect();
        eprintln!("shader-gate: steps (cpu, all threads): {}", all.join(", "));
    }
    if let Some(n) = expected
        && args.shard.1 == 1
        && t.rows < n
        && !stop.load(Ordering::Relaxed)
    {
        eprintln!("shader-gate: {} listed hashes are not in the pack", n - t.rows);
    }
    if !t.regressions.is_empty() {
        eprintln!("shader-gate: first {} of {} regressions:", t.regressions.len().min(10), t.regressions.len());
        t.print_regressions(10);
    }
    let code = if stop.load(Ordering::Relaxed) {
        3
    } else if t.newly_failing > 0 {
        1
    } else {
        0
    };
    eprintln!("shader-gate: exit {code} ({secs:.1} s, {} rows this run, out {})", selected.load(Ordering::Relaxed) - done.len(), out_path.display());
    std::process::exit(code);
}

/// XNW: with `XNW_GATE_UNIFORMITY=<file>`, one line per
/// translated program: hash, stage, name, heap handles the uniformity pass
/// declared non-uniform, those of them with a varying index, all heap
/// handles.
fn uniformity_row(row: &Row, (marked, varying, handles): (usize, usize, usize)) {
    static FILE: std::sync::OnceLock<Option<Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let file = FILE.get_or_init(|| {
        let path = std::env::var_os("XNW_GATE_UNIFORMITY")?;
        std::fs::OpenOptions::new().create(true).append(true).open(path).ok().map(Mutex::new)
    });
    if let Some(f) = file
        && let Ok(mut f) = f.lock()
    {
        let _ = writeln!(f, "{}	{}	{}	{marked}	{varying}	{handles}", row.hash, row.stage, row.name);
    }
}

/// Translates one program as the runtime would and validates its DXIL
/// unless the baseline holds the same DXIL with a verdict.
fn gate_one(bytes: &[u8], h: u128, hash: String, name: &str, base: Option<&Row>, validator: Option<&mut Validator>) -> Row {
    let stage = stage_of(bytes);
    let opts = gate_options(stage, h);
    let t0 = Instant::now();
    let c0 = thread_cycles();
    let r = std::panic::catch_unwind(|| xnw_shader::emit::translate_with(bytes, opts));
    let cycles = thread_cycles().saturating_sub(c0);
    let translate_us = t0.elapsed().as_micros() as u64;
    let mut row = Row { hash, name: cell(name), stage: stage.into(), translate_us, dxil: "-".into(), checked: "none".into(), reason: "-".into(), cycles, ..Row::default() };
    let s = match r {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            row.status = "refused".into();
            row.reason = cell(&e.to_string());
            return row;
        }
        Err(p) => {
            row.status = "refused".into();
            row.reason = cell(&format!("panic: {}", p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default()));
            return row;
        }
    };
    row.dxil = format!("{:032x}", xnw_core::hash128(&[&s.dxil]));
    row.bytes = s.dxil.len();
    row.insts = xnw_shader::emit::last_insts();
    uniformity_row(&row, xnw_shader::emit::last_nonuniform());
    if let Some(b) = base.filter(|b| b.dxil == row.dxil && b.checked != "none" && b.status != "refused") {
        row.status = b.status.clone();
        row.reason = b.reason.clone();
        row.checked = "base".into();
        return row;
    }
    row.status = "ok".into();
    if let Some(v) = validator {
        let t = Instant::now();
        let mut copy = s.dxil.clone();
        let r = v.validate(&mut copy);
        row.validate_us = t.elapsed().as_micros() as u64;
        row.checked = "validated".into();
        if let Err(e) = r {
            row.status = "invalid".into();
            row.reason = cell(&e);
        }
    }
    row
}

/// This thread's CPU cycles so far.
fn thread_cycles() -> u64 {
    let mut c = 0;
    // SAFETY: the pseudo-handle of the calling thread and a valid out pointer.
    unsafe { xnw_core::win::QueryThreadCycleTime(xnw_core::win::GetCurrentThread(), &mut c) };
    c
}

/// The milliseconds of each step [`xnw_shader::emit::translate_with`]
/// takes for one program (the same steps in the same order), to find what
/// is slow in a slow translation.
fn stage_times(bytes: &[u8], h: u128) -> String {
    use xnw_shader::emit::Descriptors;
    let opts = gate_options(stage_of(bytes), h);
    let mut t = Instant::now();
    let mut lap = |name: &str, out: &mut Vec<String>| {
        out.push(format!("{name} {:.1}", t.elapsed().as_secs_f64() * 1e3));
        t = Instant::now();
    };
    let mut out = Vec::new();
    let r = (|| -> Result<(), xnw_shader::ShaderError> {
        let c = Container::parse(bytes)?;
        let d = c.part(FourCC::DXIL).ok_or(xnw_shader::ShaderError::Malformed("no DXIL part"))?;
        let native = xnw_shader::native_programs(d.data)?
            .into_iter()
            .find(|p| matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms))
            .ok_or(xnw_shader::ShaderError::Malformed("no program"))?;
        let spilled = native.stage == HwStage::Cs && !native.layout.is_empty() && native.layout.iter().all(|r| native.stage.user_sgpr(r.user_data) >= 16);
        lap("parse", &mut out);
        let _ = xnw_shader::emit::take_laps();
        let p = xnw_shader::ir::build_with(&native.patched_code(), native.wave64, native.vgprs, spilled)?;
        lap("ir", &mut out);
        let res = xnw_shader::resource::track_with(&p, &native.layout, native.stage, opts.descriptors == Descriptors::Gpu)?;
        lap("resources", &mut out);
        let io = xnw_shader::io::raise(&p, &native)?;
        lap("io", &mut out);
        let st = xnw_shader::structure::structure(&p)?;
        lap("structure", &mut out);
        let e = xnw_shader::emit::emit_with(&c, &native, &p, &st, &io, &res, opts);
        lap("emit", &mut out);
        let inner: Vec<String> = xnw_shader::emit::take_laps().iter().map(|(n, ms, k)| if *k > 1 { format!("{n} {ms:.1}/{k}") } else { format!("{n} {ms:.1}") }).collect();
        out.push(format!("[{}]", inner.join(", ")));
        e.map(|_| ())
    })();
    if let Err(e) = r {
        out.push(format!("stops: {e}"));
    }
    out.join(", ")
}
