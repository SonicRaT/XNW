//! `xnw card-sweep` (SWEEP1): which of the title's pipelines the host
//! adapter takes, found before any title run. Every selected pipeline of
//! the pipeline list (`<XNW_CACHE>/pipelines/pipeline-list.txt`) is
//! translated with the runtime's options (`pipeline::options_for`, after
//! the device: its barycentrics) and its host pipeline state CREATED as
//! the prebuild makes it (`HostPipeline::new`; `--bridged`: also the
//! bridge's pair, `xnw_world::bridged::build`): compile only, no draw, no
//! dispatch, the pipeline library bypassed so the driver compiles each.
//!
//! ```text
//! xnw card-sweep [--drawn] [--all] [--list F] [--from-log LOG]... [--captures DIR]
//!                [--bridged] [--out DIR] [--jobs N] [--timeout S] [--abort-after N]
//!                [--shard i/n] [--limit N] [--pack P] [--crash-test P] [--hang-test P]
//! ```
//!
//! Selection: `--drawn` (default) the pipelines with a program the runs
//! drew with (`XNW_SHADER_DRAWN` or `out\shader-gate\drawn.txt`, here or
//! under the data root `XNW_DATA`, plus the containers of every capture's
//! `programs.tsv` under `--captures`, default `<XNW_DATA>\out\w-pk*`); `--all` every listed pipeline; `--list F`
//! keys or program hash prefixes (first word per line); `--from-log LOG`
//! the keys a run log names as compiling at a crash ("still compiling"),
//! put first ("suspect" column). `--shard i/n` keeps one hash shard.
//!
//! Each pipeline is made in a CHILD process (`--jobs` children at once,
//! each one pipeline at a time): the child prints `BEGIN`, its classes
//! (`PRE`) and its row (`ROW`); a crash is caught by the child's
//! unhandled-exception filter (`CRASH` code, module+offset, access) and the
//! child restarted on the rest; a pipeline past `--timeout` S (60) kills
//! the child (`timeout`). `--abort-after N` (5) crashes or timeouts in a
//! row stop the sweep (exit 3). Resumable: rows already in
//! `<out>/sweep.tsv` are not made again. `--crash-test P` / `--hang-test P`
//! make the child crash (write 0x2000000c) / hang before creating the
//! pipelines whose key or a program hash starts with P (the instrument's proof).
//!
//! Children log to `<out>/child-<k>.log`; the adapter is the process's
//! (`XNW_ADAPTER=warp`, `XNW_WARP`, `XNW_AGILITY`), named in
//! `<out>/device.txt` and the TSV's first line.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, Write as _};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use xnw_gpu::d3d12x::pipeline_sweep as sweep;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The TSV columns (child rows and parent rows alike).
const COLUMNS: [&str; 25] = [
    "item", "kind", "key", "suspect", "stages", "names", "programs", "mesh", "rt", "barycentrics", "wave_ops", "wave_size", "sm_declared", "sm_needed", "uavs", "stores", "cube", "tex3d", "kill",
    "gap", "translate", "validate", "create", "create_ms", "translate_ms",
];
const C_CREATE: usize = 22;
const C_MS: usize = 23;
const C_TMS: usize = 24;

fn hex32(h: u128) -> String {
    format!("{h:032x}")
}

fn cell(s: &str) -> String {
    let one: String = s.chars().map(|c| if c == '\t' || c == '\n' || c == '\r' { ' ' } else { c }).collect();
    if one.is_empty() { "-".into() } else { one.chars().take(300).collect() }
}

pub fn run(args: &[String]) -> Res<()> {
    if args.first().map(String::as_str) == Some("--child") {
        return child(&args[1..]);
    }
    parent(args)
}

// ---- The parent ----

struct Item {
    id: String,
    key: String,
    bridged: bool,
    suspect: bool,
}

fn words_of(path: &Path) -> Res<Vec<String>> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .filter_map(|l| {
            let w = l.split('#').next()?.split(['\t', ' ', '|']).next()?.trim().to_ascii_lowercase();
            (!w.is_empty() && w.bytes().all(|b| b.is_ascii_hexdigit())).then_some(w)
        })
        .collect())
}

/// The keys a run log names as compiling at a crash or for long.
fn log_keys(path: &Path) -> Res<Vec<String>> {
    let text = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&text);
    let mut out = Vec::new();
    // "host pipeline still compiling at access violation: ...", the prebuild's
    // "STILL COMPILING"; not "draws dropped ... (still compiling)".
    for line in text.lines().filter(|l| l.contains("still compiling at ") || l.contains("STILL COMPILING")) {
        if let Some(k) = line.split(" key ").nth(1).and_then(|r| r.split_whitespace().next())
            && k.len() == 32
            && !out.iter().any(|o| o == k)
        {
            out.push(k.to_owned());
        }
    }
    Ok(out)
}

/// The data root's `out` folder: `<XNW_DATA>\out` (the folder that holds
/// the captures, as a checkout's own `out` may not), else `out` here.
fn data_out() -> PathBuf {
    std::env::var_os("XNW_DATA").map_or_else(|| PathBuf::from("out"), |d| PathBuf::from(d).join("out"))
}

/// Container hashes of every capture's `programs.tsv` under `root` (the
/// default root [`data_out`]: only its `w-pk*` folders).
fn capture_programs(root: &Path, only_pk: bool, out: &mut HashSet<String>) {
    let mut stack: Vec<PathBuf> = std::fs::read_dir(root).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir() && (!only_pk || p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("w-pk")))).collect();
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "programs.tsv")
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                for line in text.lines().skip(1) {
                    if let Some(c) = line.split('\t').nth(3).filter(|c| c.len() == 32) {
                        out.insert(c.to_ascii_lowercase());
                    }
                }
            }
        }
    }
}

fn value<'a>(args: &'a [String], flag: &str) -> Option<&'a String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1))
}

fn values<'a>(args: &'a [String], flag: &str) -> Vec<&'a String> {
    args.windows(2).filter(|w| w[0] == flag).map(|w| &w[1]).collect()
}

fn number(args: &[String], flag: &str, default: u64) -> Res<u64> {
    value(args, flag).map_or(Ok(default), |v| v.parse().map_err(|_| format!("{flag} {v}: expected a number").into()))
}

/// One child process at work on a shard's items.
struct Shard {
    items: VecDeque<String>,
    child: Option<std::process::Child>,
    /// Bumped at every spawn: lines of a killed child are dropped.
    generation: u64,
    in_flight: Option<(String, Instant)>,
    pre: Option<Vec<String>>,
    crash: Option<String>,
    /// Children in a row that ended before any item.
    dead_starts: u32,
}

enum Msg {
    Line(String),
    End,
}

fn parent(args: &[String]) -> Res<()> {
    let all = args.iter().any(|a| a == "--all");
    let bridged = args.iter().any(|a| a == "--bridged");
    let jobs = number(args, "--jobs", 4)?.max(1) as usize;
    let timeout = Duration::from_secs(number(args, "--timeout", 60)?.max(1));
    let abort_after = number(args, "--abort-after", 5)?.max(1);
    let limit = value(args, "--limit").map(|v| v.parse::<usize>()).transpose().map_err(|_| "--limit: expected a number")?;
    let shard = match value(args, "--shard") {
        Some(s) => {
            let (i, n) = s.split_once('/').ok_or("--shard i/n")?;
            let (i, n): (u32, u32) = (i.parse()?, n.parse()?);
            (n > 0 && i < n).then_some((i, n)).ok_or("--shard i/n with i < n")?
        }
        None => (0, 1),
    };
    let adapter = std::env::var("XNW_ADAPTER").unwrap_or_default();
    let out = value(args, "--out").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!(r"out\card-sweep\{}", if adapter == "warp" { "warp" } else { "hw" })));
    std::fs::create_dir_all(&out)?;

    // The list and the selection.
    let entries = sweep::list();
    if entries.is_empty() {
        return Err("card-sweep: the pipeline list is empty (XNW_CACHE names the cache holding pipelines\\pipeline-list.txt)".into());
    }
    let mut suspects: Vec<String> = Vec::new();
    for log in values(args, "--from-log") {
        for k in log_keys(Path::new(log))? {
            if !suspects.contains(&k) {
                suspects.push(k);
            }
        }
    }
    let listed: Vec<String> = match value(args, "--list") {
        Some(f) => words_of(Path::new(f))?,
        None => Vec::new(),
    };
    let mut drawn: HashSet<String> = HashSet::new();
    let use_drawn = !all && listed.is_empty() && (args.iter().any(|a| a == "--drawn") || values(args, "--from-log").is_empty());
    if use_drawn {
        let found = std::env::var_os("XNW_SHADER_DRAWN").map(PathBuf::from).into_iter().chain([PathBuf::from(r"out\shader-gate\drawn.txt"), data_out().join(r"shader-gate\drawn.txt")]).find(|p| p.is_file());
        let list = found.ok_or(r"card-sweep --drawn: no drawn list (XNW_SHADER_DRAWN, out\shader-gate\drawn.txt, <XNW_DATA>\out\shader-gate\drawn.txt)")?;
        drawn.extend(words_of(&list)?);
        let n_list = drawn.len();
        let captures = value(args, "--captures").map(PathBuf::from).unwrap_or_else(data_out);
        capture_programs(&captures, value(args, "--captures").is_none(), &mut drawn);
        eprintln!("card-sweep: drawn programs {} ({} from {}, {} more from captures under {})", drawn.len(), n_list, list.display(), drawn.len() - n_list, captures.display());
    }
    let mut chosen: Vec<(bool, i64, &sweep::Entry)> = Vec::new();
    let mut no_recipe = 0;
    for e in &entries {
        let Some(recipe) = e.recipe.as_deref() else {
            no_recipe += 1;
            continue;
        };
        let programs = sweep::parse(recipe).unwrap_or_default();
        let hexes: Vec<String> = programs.iter().flat_map(|p| [hex32(p.1), hex32(p.2)]).collect();
        let suspect = suspects.contains(&e.key);
        let named = |w: &String| e.key.starts_with(w.as_str()) || hexes.iter().any(|h| h.starts_with(w.as_str()));
        let wanted = all || suspect || listed.iter().any(named) || (use_drawn && hexes.iter().any(|h| drawn.contains(h)));
        let in_shard = u32::from_str_radix(&e.key[..8.min(e.key.len())], 16).unwrap_or(0) % shard.1 == shard.0;
        if wanted && in_shard {
            chosen.push((suspect, e.ms, e));
        }
    }
    // Suspects first (in the logs' order), then earliest first use.
    chosen.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| if a.0 { suspects.iter().position(|s| *s == a.2.key).cmp(&suspects.iter().position(|s| *s == b.2.key)) } else { a.1.cmp(&b.1) }).then_with(|| a.2.key.cmp(&b.2.key)));
    let mut items: Vec<Item> = Vec::new();
    for (suspect, _, e) in &chosen {
        items.push(Item { id: e.key.clone(), key: e.key.clone(), bridged: false, suspect: *suspect });
        let pair = e.recipe.as_deref().and_then(sweep::parse).is_some_and(|p| p.iter().any(|s| s.0 == "mesh") && p.iter().any(|s| s.0 == "pixel"));
        if bridged && pair {
            items.push(Item { id: format!("{}/bridged", e.key), key: e.key.clone(), bridged: true, suspect: *suspect });
        }
    }
    if let Some(n) = limit {
        items.truncate(n);
    }
    let missing_suspects: Vec<&String> = suspects.iter().filter(|s| !entries.iter().any(|e| &e.key == *s && e.recipe.is_some())).collect();
    if !missing_suspects.is_empty() {
        eprintln!("card-sweep: {} log keys have no recipe in the list: {}", missing_suspects.len(), missing_suspects.iter().map(|s| &s[..8]).collect::<Vec<_>>().join(" "));
    }

    // Resume: rows already written.
    let tsv = out.join("sweep.tsv");
    let mut done: HashSet<String> = HashSet::new();
    let fresh = !tsv.is_file();
    if !fresh {
        for line in std::fs::read_to_string(&tsv)?.lines().filter(|l| !l.starts_with('#') && !l.starts_with("item\t")) {
            if let Some(id) = line.split('\t').next() {
                done.insert(id.to_owned());
            }
        }
    }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&tsv)?;
    if fresh {
        writeln!(file, "# xnw card-sweep v1, adapter {}, started {:?}", if adapter.is_empty() { "hw (default)" } else { &adapter }, std::time::SystemTime::now())?;
        writeln!(file, "{}", COLUMNS.join("\t"))?;
    }
    let todo: Vec<&Item> = items.iter().filter(|i| !done.contains(&i.id)).collect();
    eprintln!(
        "card-sweep: {} listed ({} without recipe), {} chosen items ({} suspects), {} already done, {} to make on {} children, {} s each at most, out {}",
        entries.len(),
        no_recipe,
        items.len(),
        items.iter().filter(|i| i.suspect).count(),
        items.len() - todo.len(),
        todo.len(),
        jobs.min(todo.len().max(1)),
        timeout.as_secs(),
        out.display()
    );
    if todo.is_empty() {
        return summary(&tsv);
    }
    let by_id: HashMap<&str, &Item> = items.iter().map(|i| (i.id.as_str(), i)).collect();

    // Shards: suspects first, round robin.
    let n = jobs.min(todo.len());
    let mut shards: Vec<Shard> = (0..n).map(|_| Shard { items: VecDeque::new(), child: None, generation: 0, in_flight: None, pre: None, crash: None, dead_starts: 0 }).collect();
    for (k, i) in todo.iter().enumerate() {
        shards[k % n].items.push_back(i.id.clone());
    }
    let (tx, rx) = mpsc::channel::<(usize, u64, Msg)>();
    let mut pass: Vec<String> = Vec::new();
    for flag in ["--pack", "--crash-test", "--hang-test"] {
        if let Some(v) = value(args, flag) {
            pass.push(flag.into());
            pass.push(v.clone());
        }
    }
    let spawn = |k: usize, s: &mut Shard| -> Res<()> {
        s.generation += 1;
        let list = out.join(format!("child-{k}.items"));
        let text: String = s.items.iter().map(|id| {
            let i = by_id[id.as_str()];
            format!("{}\t{}\t{}\n", i.id, u8::from(i.suspect), u8::from(i.bridged))
        }).collect();
        std::fs::write(&list, text)?;
        let log = std::fs::OpenOptions::new().create(true).append(true).open(out.join(format!("child-{k}.log")))?;
        let mut child = std::process::Command::new(std::env::current_exe()?)
            .arg("card-sweep")
            .arg("--child")
            .arg(&list)
            .args(&pass)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(log)
            .spawn()?;
        let stdout = child.stdout.take().ok_or("child stdout")?;
        let (tx, generation) = (tx.clone(), s.generation);
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send((k, generation, Msg::Line(line))).is_err() {
                    return;
                }
            }
            let _ = tx.send((k, generation, Msg::End));
        });
        s.child = Some(child);
        s.in_flight = None;
        s.pre = None;
        s.crash = None;
        Ok(())
    };
    for (k, s) in shards.iter_mut().enumerate() {
        spawn(k, s)?;
    }

    let started = Instant::now();
    let total = todo.len();
    let mut written = 0usize;
    let mut in_a_row = 0u64;
    let mut device_noted = false;
    let mut aborted: Option<String> = None;
    let mut last_progress = Instant::now();
    // A row for an item whose child died or was stopped: its classes (PRE)
    // and what happened.
    let dead_row = |s: &Shard, id: &str, what: String, ms: f64| -> String {
        let mut row = s.pre.clone().filter(|p| p.first().map(String::as_str) == Some(id)).unwrap_or_else(|| {
            let i = by_id[id];
            let mut r = vec!["-".to_owned(); COLUMNS.len()];
            r[0] = i.id.clone();
            r[1] = if i.bridged { "bridged" } else { "title" }.into();
            r[2] = i.key.clone();
            r[3] = u8::from(i.suspect).to_string();
            r
        });
        row[C_CREATE] = cell(&what);
        row[C_MS] = format!("{ms:.0}");
        row.join("\t")
    };
    loop {
        if shards.iter().all(|s| s.child.is_none() && s.items.is_empty()) || aborted.is_some() {
            break;
        }
        let msg = rx.recv_timeout(Duration::from_millis(250));
        let mut rows: Vec<String> = Vec::new();
        if let Ok((k, generation, msg)) = msg {
            let s = &mut shards[k];
            if generation == s.generation {
                match msg {
                    Msg::Line(line) => {
                        let (tag, rest) = line.split_once('\t').unwrap_or((line.as_str(), ""));
                        match tag {
                            "DEVICE" if !device_noted => {
                                device_noted = true;
                                std::fs::write(out.join("device.txt"), format!("{rest}\n"))?;
                                writeln!(file, "# device {rest}")?;
                                eprintln!("card-sweep: device {rest}");
                            }
                            "BEGIN" => {
                                s.in_flight = Some((rest.to_owned(), Instant::now()));
                                s.dead_starts = 0;
                            }
                            "PRE" => s.pre = Some(rest.split('\t').map(str::to_owned).collect()),
                            "CRASH" => s.crash = Some(rest.replace('\t', " ")),
                            "ROW" => {
                                let id = rest.split('\t').next().unwrap_or("").to_owned();
                                s.items.retain(|i| *i != id);
                                s.in_flight = None;
                                s.pre = None;
                                rows.push(rest.to_owned());
                            }
                            "FATAL" => aborted = Some(format!("child {k}: {rest}")),
                            _ => {}
                        }
                    }
                    Msg::End => {
                        let status = s.child.take().and_then(|mut c| c.wait().ok());
                        let code = status.and_then(|st| st.code()).map_or(0, |c| c as u32);
                        if let Some((id, at)) = s.in_flight.take() {
                            let what = match &s.crash {
                                Some(c) => format!("crash {c}"),
                                None => format!("crash exit {code:#010x} (no record)"),
                            };
                            rows.push(dead_row(s, &id, what, at.elapsed().as_secs_f64() * 1e3));
                            s.items.retain(|i| *i != id);
                        } else if !s.items.is_empty() {
                            s.dead_starts += 1;
                            if s.dead_starts >= 3 {
                                aborted = Some(format!("child {k} ended 3 times before any item (exit {code:#010x}; see child-{k}.log)"));
                            }
                        }
                        if !s.items.is_empty() && aborted.is_none() {
                            spawn(k, s)?;
                        }
                    }
                }
            }
        }
        // Time limits.
        for (k, s) in shards.iter_mut().enumerate() {
            if let Some((id, at)) = s.in_flight.clone()
                && at.elapsed() > timeout
            {
                if let Some(mut c) = s.child.take() {
                    let _ = c.kill();
                    let _ = c.wait();
                }
                rows.push(dead_row(s, &id, format!("timeout {} s", timeout.as_secs()), at.elapsed().as_secs_f64() * 1e3));
                s.items.retain(|i| *i != id);
                s.in_flight = None;
                if !s.items.is_empty() {
                    spawn(k, s)?;
                } else {
                    s.generation += 1;
                }
            }
        }
        for row in rows {
            writeln!(file, "{row}")?;
            file.flush()?;
            written += 1;
            let create = row.split('\t').nth(C_CREATE).unwrap_or("");
            if create.starts_with("crash") || create.starts_with("timeout") {
                in_a_row += 1;
                eprintln!("card-sweep: {} {}", row.split('\t').next().unwrap_or(""), create);
                if in_a_row >= abort_after {
                    aborted = Some(format!("{in_a_row} crashes or timeouts in a row (--abort-after {abort_after})"));
                }
            } else {
                in_a_row = 0;
            }
        }
        if last_progress.elapsed() > Duration::from_secs(10) {
            last_progress = Instant::now();
            let secs = started.elapsed().as_secs_f64();
            eprintln!("card-sweep: {written}/{total} in {secs:.0} s, about {:.0} s left", if written > 0 { secs / written as f64 * (total - written) as f64 } else { 0.0 });
        }
    }
    for s in shards.iter_mut() {
        if let Some(mut c) = s.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
    eprintln!("card-sweep: {written} rows in {:.1} s", started.elapsed().as_secs_f64());
    summary(&tsv)?;
    if let Some(why) = aborted {
        eprintln!("card-sweep: STOPPED EARLY: {why}");
        std::process::exit(3);
    }
    Ok(())
}

/// Counts of the TSV's create column (first word) and kinds.
fn summary(tsv: &Path) -> Res<()> {
    let text = std::fs::read_to_string(tsv)?;
    let mut by: std::collections::BTreeMap<String, usize> = Default::default();
    let mut rows = 0;
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.starts_with("item\t")) {
        let f: Vec<&str> = line.split('\t').collect();
        let Some(create) = f.get(C_CREATE) else { continue };
        rows += 1;
        let word = create.split([' ', ':']).next().unwrap_or("");
        *by.entry(format!("{} {word}", f.get(1).unwrap_or(&"?"))).or_default() += 1;
    }
    println!("card-sweep: {rows} rows in {}", tsv.display());
    for (k, n) in by {
        println!("  {n:>6}  {k}");
    }
    Ok(())
}

// ---- The child ----

#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetUnhandledExceptionFilter(filter: Option<xnw_core::win::VectoredHandler>) -> *mut std::ffi::c_void;
    fn GetStdHandle(which: u32) -> xnw_core::win::HANDLE;
    fn WriteFile(file: xnw_core::win::HANDLE, data: *const u8, len: u32, written: *mut u32, overlapped: *mut std::ffi::c_void) -> i32;
}

/// A fixed buffer the crash filter writes into (no allocation there).
struct Line {
    buf: [u8; 512],
    n: usize,
}

impl Line {
    fn put(&mut self, b: &[u8]) {
        let k = b.len().min(self.buf.len() - self.n);
        self.buf[self.n..self.n + k].copy_from_slice(&b[..k]);
        self.n += k;
    }
    fn hex(&mut self, v: u64) {
        self.put(b"0x");
        let mut digits = [0u8; 16];
        let mut k = 0;
        let mut v = v;
        loop {
            digits[k] = b"0123456789abcdef"[(v & 15) as usize];
            k += 1;
            v >>= 4;
            if v == 0 {
                break;
            }
        }
        for i in (0..k).rev() {
            self.put(&[digits[i]]);
        }
    }
}

/// The child's unhandled-exception filter: `CRASH <code> <module>+<offset>
/// <read|write|exec> <address>` on stdout, then the process ends with the
/// exception code. Writes into a fixed buffer: the heap may be what broke.
unsafe extern "system" fn crashed(info: *mut xnw_core::win::ExceptionPointers) -> i32 {
    use xnw_core::win::{GetCurrentProcess, GetModuleFileNameW, GetModuleHandleExW, TerminateProcess};
    // SAFETY: the system passes a live record.
    let record = unsafe { &*(*info).record };
    let mut line = Line { buf: [0; 512], n: 0 };
    line.put(b"CRASH\t");
    line.hex(u64::from(record.code));
    line.put(b"\t");
    let mut module = std::ptr::null_mut();
    // FROM_ADDRESS | UNCHANGED_REFCOUNT.
    // SAFETY: plain calls; the name buffer's size is passed.
    let found = unsafe { GetModuleHandleExW(0x4 | 0x2, record.address as *const u16, &mut module) } != 0;
    if found {
        let mut name = [0u16; 260];
        // SAFETY: as above.
        let n = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) } as usize;
        let start = name[..n].iter().rposition(|&c| c == u16::from(b'\\')).map_or(0, |p| p + 1);
        for &c in &name[start..n] {
            line.put(&[if c < 128 { c as u8 } else { b'?' }]);
        }
        line.put(b"+");
        line.hex((record.address - module as usize) as u64);
    } else {
        line.hex(record.address as u64);
    }
    if record.code == 0xc000_0005 && record.param_count >= 2 {
        line.put(match record.params[0] {
            0 => b"\tread\t".as_slice(),
            1 => b"\twrite\t",
            _ => b"\texec\t",
        });
        line.hex(record.params[1] as u64);
    }
    line.put(b"\n");
    let mut written = 0;
    // SAFETY: our own stdout; the buffer outlives the call.
    unsafe {
        WriteFile(GetStdHandle(0xffff_fff5), line.buf.as_ptr(), line.n as u32, &mut written, std::ptr::null_mut());
        TerminateProcess(GetCurrentProcess(), record.code);
    }
    xnw_core::win::EXCEPTION_CONTINUE_SEARCH
}

fn say(tag: &str, text: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{tag}\t{text}");
    let _ = out.flush();
}

/// The wave size a container's `PSV0` fixes (minimum and maximum equal).
fn psv_wave(c: &[u8]) -> Option<u32> {
    let parsed = xnw_shader::Container::parse(c).ok()?;
    let p = parsed.part(xnw_shader::FourCC::PSV0)?.data;
    let word = |at: usize| Some(u32::from_le_bytes(p.get(4 + at..8 + at)?.try_into().ok()?));
    let (min, max) = (word(16)?, word(20)?);
    (min == max && min != 0).then_some(min)
}

/// What a console container's native code does: (UAV descriptors read,
/// store/atomic instructions, cube reads, 3D reads, BVH instructions).
fn native_facts(container: &[u8]) -> (usize, usize, usize, usize, usize) {
    use xnw_shader::gfx10::{Encoding, Fields, decode};
    let mut f = (0, 0, 0, 0, 0);
    let Ok(parsed) = xnw_shader::Container::parse(container) else { return f };
    let Some(part) = parsed.part(xnw_shader::FourCC::DXIL) else { return f };
    for p in xnw_shader::native_programs(part.data).unwrap_or_default() {
        f.0 += p.used.iter().filter(|u| u.kind == xnw_shader::ResourceType::Uav).count();
        for i in decode(&p.patched_code()).unwrap_or_default() {
            let name = i.name().unwrap_or("");
            if matches!(i.encoding, Encoding::Mubuf | Encoding::Mtbuf | Encoding::Mimg | Encoding::Flat | Encoding::Global) && (name.contains("store") || name.contains("atomic")) {
                f.1 += 1;
            }
            if name.contains("bvh") {
                f.4 += 1;
            }
            if let Fields::Image { dim, .. } = i.fields(p.wave64) {
                match dim & 7 {
                    3 => f.2 += 1,
                    2 => f.3 += 1,
                    _ => {}
                }
            }
        }
    }
    f
}

/// The lowest shader model the program's features need.
fn model_needed(n: &xnw_world::bridge::Needs, ops: &[String]) -> (u32, u32) {
    use xnw_world::bridge::feature::*;
    let mut m = (6, 0);
    let mut at_least = |v: (u32, u32)| m = m.max(v);
    if n.features & BARYCENTRICS != 0 || ops.iter().any(|o| o.contains("attributeAtVertex")) {
        at_least((6, 1));
    }
    if ops.iter().any(|o| o.contains("traceRay") || o.contains("dispatchRays")) {
        at_least((6, 3));
    }
    if n.stage == "ms" || n.stage == "as" || ops.iter().any(|o| o.contains("rayQuery")) {
        at_least((6, 5));
    }
    if n.wave.is_some() || n.features & (VIEW_HEAP_INDEXING | SAMPLER_HEAP_INDEXING | ATOMIC64_TYPED | ATOMIC64_HEAP) != 0 || ops.iter().any(|o| o == "dx.op.createHandleFromHeap" || o == "dx.op.annotateHandle") {
        at_least((6, 6));
    }
    m
}

fn child(args: &[String]) -> Res<()> {
    let list = args.first().ok_or("card-sweep --child <items file>")?;
    let pack = value(args, "--pack").map(PathBuf::from);
    let crash_test = value(args, "--crash-test").cloned();
    let hang_test = value(args, "--hang-test").cloned();
    // SAFETY: a filter that lives as long as the process.
    unsafe { SetUnhandledExceptionFilter(Some(crashed)) };
    // The device first: the translation's options read its barycentrics.
    let Some(host) = xnw_gpu::d3d12x::host::host_device() else {
        say("FATAL", "no host device");
        return Err("no host device".into());
    };
    let c = &host.caps;
    say(
        "DEVICE",
        &format!(
            "{} ({} MiB), shader model {}.{}, wave lanes {}..{}, mesh {}, ray tracing {}, barycentrics {}, agility {:?}, software {}{}",
            c.adapter,
            c.vram_bytes >> 20,
            c.shader_model.0,
            c.shader_model.1,
            c.wave_lanes.0,
            c.wave_lanes.1,
            c.mesh_shaders,
            c.ray_tracing,
            c.barycentrics,
            c.agility,
            c.software,
            c.warp_module.as_deref().map(|w| format!(", {w}")).unwrap_or_default()
        ),
    );
    sweep::begin(pack);
    let recipes: HashMap<String, String> = sweep::list().into_iter().filter_map(|e| Some((e.key, e.recipe?))).collect();
    let mut validator = xnw_shader::emit::validator::Validator::load(xnw_shader::emit::validator::Validator::default_path().as_deref()).ok();
    xnw_shader::emit::collect_ops(true);
    for line in std::fs::read_to_string(list)?.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (Some(&id), Some(&suspect), Some(&bridged)) = (f.first(), f.get(1), f.get(2)) else { continue };
        let bridged = bridged == "1";
        let key = id.split('/').next().unwrap_or(id);
        say("BEGIN", id);
        let mut row = vec!["-".to_owned(); COLUMNS.len()];
        row[0] = id.into();
        row[1] = if bridged { "bridged" } else { "title" }.into();
        row[2] = key.into();
        row[3] = suspect.into();
        let Some(recipe) = recipes.get(key) else {
            row[C_CREATE] = "skipped: no recipe".into();
            say("ROW", &row.join("\t"));
            continue;
        };
        let programs = match sweep::programs(recipe) {
            Ok(p) => p,
            Err(e) => {
                row[C_CREATE] = cell(&format!("skipped: {e}"));
                say("ROW", &row.join("\t"));
                continue;
            }
        };
        row[4] = programs.iter().map(|p| p.stage).collect::<Vec<_>>().join("+");
        row[5] = cell(&programs.iter().map(|p| p.name.clone().unwrap_or_else(|| "?".into())).collect::<Vec<_>>().join(" + "));
        row[6] = programs.iter().map(|p| format!("{:08x}", (p.container_hash >> 96) as u32)).collect::<Vec<_>>().join("+");
        row[7] = if programs.iter().any(|p| p.stage == "mesh") { "fallback".into() } else { "0".into() };

        // Our own translation of each program, for the classes (the
        // functions it calls) and the validator; the pipeline below
        // translates through the runtime's path (shader cache).
        let t0 = Instant::now();
        let mut needs = Vec::new();
        let mut ops_all: Vec<String> = Vec::new();
        let mut translate = Vec::new();
        let mut validate = Vec::new();
        let (mut uavs, mut stores, mut cube, mut tex3d, mut bvh) = (0, 0, 0, 0, 0);
        let mut ours: Vec<Option<Vec<u8>>> = Vec::new();
        for p in &programs {
            let Some(container) = &p.container else {
                translate.push(format!("{}: no container", p.stage));
                ours.push(None);
                continue;
            };
            let nf = native_facts(container);
            (uavs, stores, cube, tex3d, bvh) = (uavs + nf.0, stores + nf.1, cube + nf.2, tex3d + nf.3, bvh + nf.4);
            let opts = sweep::options(p.stage, p.hash);
            match std::panic::catch_unwind(|| xnw_shader::emit::translate_with(container, opts)) {
                Ok(Ok(h)) => {
                    let ops = xnw_shader::emit::last_ops();
                    let wave = psv_wave(&h.dxil);
                    if let Some(n) = xnw_world::bridge::needs(&h.dxil, wave) {
                        needs.push((n, ops.clone()));
                    }
                    for o in ops {
                        if !ops_all.contains(&o) {
                            ops_all.push(o);
                        }
                    }
                    validate.push(match validator.as_mut() {
                        Some(v) => match v.validate(&mut h.dxil.clone()) {
                            Ok(()) => "ok".to_owned(),
                            Err(e) => format!("{}: invalid: {e}", p.stage),
                        },
                        None => "no validator".to_owned(),
                    });
                    ours.push(Some(h.dxil));
                }
                Ok(Err(e)) => {
                    translate.push(format!("{}: {e}", p.stage));
                    ours.push(None);
                }
                Err(_) => {
                    translate.push(format!("{}: panic", p.stage));
                    ours.push(None);
                }
            }
        }
        let translate_ms = t0.elapsed().as_secs_f64() * 1e3;
        let any = |bit: u64| needs.iter().any(|(n, _)| n.features & bit != 0);
        use xnw_world::bridge::feature;
        row[8] = if bvh > 0 || ops_all.iter().any(|o| o.contains("traceRay") || o.contains("rayQuery")) { format!("bvh {bvh}") } else { "0".into() };
        row[9] = u8::from(any(feature::BARYCENTRICS)).to_string();
        let wave_ops: Vec<&str> = ops_all.iter().filter(|o| o.contains("wave") || o.contains("quad")).map(|o| o.trim_start_matches("dx.op.")).collect();
        row[10] = if any(feature::WAVE_OPS) || !wave_ops.is_empty() { cell(&wave_ops.join(",")) } else { "0".into() };
        let waves: Vec<String> = needs.iter().filter_map(|(n, _)| n.wave).map(|w| w.to_string()).collect();
        row[11] = if waves.is_empty() { "-".into() } else { waves.join("+") };
        row[12] = needs.iter().map(|(n, _)| n.model).max().map_or("-".into(), |m| format!("{}.{}", m.0, m.1));
        row[13] = needs.iter().map(|(n, o)| model_needed(n, o)).max().map_or("-".into(), |m| format!("{}.{}", m.0, m.1));
        row[14] = uavs.to_string();
        row[15] = stores.to_string();
        row[16] = cube.to_string();
        row[17] = tex3d.to_string();
        row[18] = u8::from(ops_all.iter().any(|o| o == "dx.op.discard")).to_string();
        let plain: Vec<xnw_world::bridge::Needs> = needs.iter().map(|(n, _)| n.clone()).collect();
        row[19] = cell(&xnw_world::bridge::gap(c, &plain).unwrap_or_else(|| "-".into()));
        row[20] = if translate.is_empty() { "ok".into() } else { cell(&translate.join("; ")) };
        row[21] = if validate.is_empty() { "-".into() } else if validate.iter().all(|v| v == "ok") { "ok".into() } else { cell(&validate.into_iter().filter(|v| v != "ok").collect::<Vec<_>>().join("; ")) };
        row[C_CREATE] = "pending".into();
        row[C_TMS] = format!("{translate_ms:.0}");
        say("PRE", &row.join("\t"));

        // The instrument's proof: a crash or a hang where a driver's would be
        // (named by pipeline key or program hash prefix).
        let named = |p: &str| key.starts_with(p) || programs.iter().any(|x| hex32(x.container_hash).starts_with(p) || hex32(x.hash).starts_with(p));
        if crash_test.as_deref().is_some_and(|p| named(p)) {
            // SAFETY: deliberately not: the access violation the filter records.
            unsafe { std::ptr::write_volatile(0x2000_000c as *mut u32, 0) };
        }
        if hang_test.as_deref().is_some_and(|p| named(p)) {
            loop {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        let t1 = Instant::now();
        let create = if bridged {
            let (Some(ms), Some(ps)) = (programs.iter().find(|p| p.stage == "mesh").and_then(|p| p.container.as_deref()), programs.iter().find(|p| p.stage == "pixel").and_then(|p| p.container.as_deref())) else {
                row[C_CREATE] = "skipped: no mesh and pixel containers".into();
                say("ROW", &row.join("\t"));
                continue;
            };
            match xnw_world::bridged::build(&host.device, c, ms, ps, &[]) {
                Ok(_) => "ok".to_owned(),
                Err(e) => format!("refused: {e}"),
            }
        } else {
            match sweep::make(recipe, &programs) {
                Ok(m) => {
                    // The runtime's translation is ours (same options).
                    let differs = m.dxil.iter().zip(&ours).filter(|(r, o)| matches!((r, o), (Ok(r), Some(o)) if r != o)).count();
                    if differs > 0 {
                        row[20] = format!("{} (runtime DXIL differs in {differs})", row[20]);
                    }
                    match m.create {
                        Ok(()) => "ok".to_owned(),
                        Err(e) => format!("refused: {e}"),
                    }
                }
                Err(e) => format!("skipped: {e}"),
            }
        };
        let create_ms = t1.elapsed().as_secs_f64() * 1e3;
        row[C_CREATE] = cell(&create);
        row[C_MS] = format!("{create_ms:.0}");
        row[C_TMS] = format!("{translate_ms:.0}");
        say("ROW", &row.join("\t"));
    }
    Ok(())
}
