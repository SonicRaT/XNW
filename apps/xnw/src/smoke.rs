//! `xnw smoke [--profile <name>] [--secs N] [--record [FILE]] [--exe PATH]`
//!
//! The gate before any long run: the title from frame 0 in null-GPU mode
//! (`XNW_GPU=null`: nothing of the title's reaches a GPU), on the software
//! adapter (`XNW_ADAPTER=warp`), without a visible window
//! (`XNW_HEADLESS=1`), for at most `--secs` (default 20). It fails, with
//! the offending lines, on a `#fail` line of `tools/runs/common.expect` or
//! of the profile (exceptions, unimplemented imports, panics, ...) and on an
//! `#expect` not met by its deadline (frame 1 by 10 s, ...). Profiles are
//! `tools/runs/<name>.env`: `KEY=VALUE` lines (quotes stripped, `export`
//! allowed), `#expect <seconds> <regex>`, `#fail <regex>`, `#fixture <log>`
//! (a log the profile's expectations must pass, `tools/test_instruments.py`
//! and the tests here). The regex subset is Python's: literals, `.`,
//! classes `[..]`/`[^..]`, `\d \w \s \D \W \S \b`, escapes, `* + ?`
//! `{m,n}`, groups, `|`, `^ $`; `tools/expect_check.py` evaluates the same
//! file against a finished log.
//!
//! One title at a time: refused while another title holds `Global\XNW.title`
//! (run.rs) or a live process holds `out/title.lock` of the main checkout
//! (found through a worktree's `.git` file); the lock holds this pid.
//! Exit code 0 pass, 1 fail, 2 refused or not started.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

// ---- regex (backtracking, Python `re.search` semantics for the subset) ----

#[derive(Debug, Clone)]
enum Atom {
    Char(char),
    Any,
    Class(Vec<(char, char)>, bool),
    Group(Vec<Vec<Item>>),
    Start,
    End,
    WordEdge,
}

#[derive(Debug, Clone)]
struct Item {
    atom: Atom,
    min: usize,
    max: usize,
}

#[derive(Debug, Clone)]
pub struct Regex {
    alt: Vec<Vec<Item>>,
    pub text: String,
}

fn class_of(c: char) -> Option<(Vec<(char, char)>, bool)> {
    let digit = vec![('0', '9')];
    let word = vec![('0', '9'), ('a', 'z'), ('A', 'Z'), ('_', '_')];
    let space = vec![(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r'), ('\x0b', '\x0c')];
    Some(match c {
        'd' => (digit, false),
        'D' => (digit, true),
        'w' => (word, false),
        'W' => (word, true),
        's' => (space, false),
        'S' => (space, true),
        _ => return None,
    })
}

fn escaped(c: char) -> char {
    match c {
        't' => '\t',
        'n' => '\n',
        'r' => '\r',
        c => c,
    }
}

impl Regex {
    pub fn new(text: &str) -> Result<Regex, String> {
        let chars: Vec<char> = text.chars().collect();
        let mut at = 0;
        let alt = parse_alt(&chars, &mut at)?;
        if at != chars.len() {
            return Err(format!("regex {text:?}: unbalanced ')' at {at}"));
        }
        Ok(Regex { alt, text: text.to_owned() })
    }

    /// Whether the pattern matches anywhere in `line`.
    pub fn search(&self, line: &str) -> bool {
        let t: Vec<char> = line.chars().take(4000).collect();
        (0..=t.len()).any(|p| m_alt(&self.alt, &t, p, &mut |_| true))
    }
}

fn parse_alt(c: &[char], at: &mut usize) -> Result<Vec<Vec<Item>>, String> {
    let mut alt = vec![Vec::new()];
    while *at < c.len() {
        let ch = c[*at];
        *at += 1;
        let atom = match ch {
            '|' => {
                alt.push(Vec::new());
                continue;
            }
            ')' => {
                *at -= 1;
                return Ok(alt);
            }
            '(' => {
                if c.get(*at) == Some(&'?') && c.get(*at + 1) == Some(&':') {
                    *at += 2;
                }
                let inner = parse_alt(c, at)?;
                if c.get(*at) != Some(&')') {
                    return Err("missing ')'".into());
                }
                *at += 1;
                Atom::Group(inner)
            }
            '.' => Atom::Any,
            '^' => Atom::Start,
            '$' => Atom::End,
            '[' => {
                let neg = c.get(*at) == Some(&'^');
                if neg {
                    *at += 1;
                }
                let mut set = Vec::new();
                let mut first = true;
                loop {
                    let Some(&x) = c.get(*at) else { return Err("missing ']'".into()) };
                    *at += 1;
                    if x == ']' && !first {
                        break;
                    }
                    first = false;
                    let lo = if x == '\\' {
                        let e = *c.get(*at).ok_or("trailing '\\'")?;
                        *at += 1;
                        if let Some((s, false)) = class_of(e) {
                            set.extend(s);
                            continue;
                        }
                        escaped(e)
                    } else {
                        x
                    };
                    if c.get(*at) == Some(&'-') && c.get(*at + 1).is_some_and(|&h| h != ']') {
                        let hi = c[*at + 1];
                        *at += 2;
                        set.push((lo, hi));
                    } else {
                        set.push((lo, lo));
                    }
                }
                Atom::Class(set, neg)
            }
            '\\' => {
                let e = *c.get(*at).ok_or("trailing '\\'")?;
                *at += 1;
                if e == 'b' {
                    Atom::WordEdge
                } else if let Some((s, n)) = class_of(e) {
                    Atom::Class(s, n)
                } else {
                    Atom::Char(escaped(e))
                }
            }
            '*' | '+' | '?' => return Err(format!("nothing to repeat before '{ch}'")),
            ch => Atom::Char(ch),
        };
        let (mut min, mut max) = (1, 1);
        match c.get(*at) {
            Some('*') => (min, max) = (0, usize::MAX),
            Some('+') => (min, max) = (1, usize::MAX),
            Some('?') => (min, max) = (0, 1),
            Some('{') => {
                let close = c[*at..].iter().position(|&x| x == '}').map(|p| *at + p);
                let body: String = close.map(|e| c[*at + 1..e].iter().collect()).unwrap_or_default();
                let nums: Vec<&str> = body.split(',').collect();
                let n = |s: &str| s.trim().parse::<usize>().ok();
                let parsed = match nums.as_slice() {
                    [a] => n(a).map(|a| (a, a)),
                    [a, b] => n(a).map(|a| (a, if b.trim().is_empty() { usize::MAX } else { n(b).unwrap_or(a) })),
                    _ => None,
                };
                match (close, parsed) {
                    (Some(e), Some(mm)) => {
                        (min, max) = mm;
                        *at = e;
                    }
                    _ => {
                        alt.last_mut().expect("one").push(Item { atom, min, max });
                        continue;
                    }
                }
            }
            _ => {
                alt.last_mut().expect("one").push(Item { atom, min, max });
                continue;
            }
        }
        *at += 1;
        alt.last_mut().expect("one").push(Item { atom, min, max });
    }
    Ok(alt)
}

fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn m_alt(alt: &[Vec<Item>], t: &[char], p: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    alt.iter().any(|seq| m_seq(seq, t, p, k))
}

fn m_seq(seq: &[Item], t: &[char], p: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    match seq.split_first() {
        None => k(p),
        Some((it, rest)) => m_item(it, 0, t, p, &mut |q| m_seq(rest, t, q, k)),
    }
}

/// Greedy: one more repetition first (past the minimum never an empty
/// one), then the rest.
fn m_item(it: &Item, n: usize, t: &[char], p: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    if n < it.max && m_atom(&it.atom, t, p, &mut |q| (q != p || n < it.min) && m_item(it, n + 1, t, q, k)) {
        return true;
    }
    n >= it.min && k(p)
}

fn m_atom(a: &Atom, t: &[char], p: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    match a {
        Atom::Group(alt) => m_alt(alt, t, p, k),
        Atom::Start => p == 0 && k(p),
        Atom::End => p == t.len() && k(p),
        Atom::WordEdge => {
            let before = p > 0 && is_word(t[p - 1]);
            let after = p < t.len() && is_word(t[p]);
            before != after && k(p)
        }
        _ => {
            let Some(&c) = t.get(p) else { return false };
            let ok = match a {
                Atom::Char(x) => *x == c,
                Atom::Any => c != '\n',
                Atom::Class(set, neg) => set.iter().any(|&(lo, hi)| lo <= c && c <= hi) != *neg,
                _ => false,
            };
            ok && k(p + 1)
        }
    }
}

// ---- profiles ----

#[derive(Debug, Default)]
pub struct Profile {
    pub env: Vec<(String, String)>,
    /// `(deadline seconds, pattern)`.
    pub expect: Vec<(f64, Regex)>,
    pub fail: Vec<Regex>,
    pub fixtures: Vec<String>,
    /// `tools/runs`, for `include` lines.
    runs: Option<std::path::PathBuf>,
    depth: usize,
    /// This machine's folders: used for
    /// `XNW_EXE`, `XNW_GAME_DRIVE` and `XNW_AGILITY` only when neither the
    /// profile nor the environment sets them.
    pub local: Vec<(String, String)>,
}

impl Profile {
    /// Adds the lines of a profile or expectations file.
    pub fn add(&mut self, text: &str, from: &str) -> Res<()> {
        for (n, raw) in text.lines().enumerate() {
            let line = raw.trim_end_matches('\r');
            let bad = |e: String| format!("{from}:{}: {e}", n + 1);
            if let Some(rest) = line.strip_prefix("#expect ") {
                let (secs, pat) = rest.trim_start().split_once(' ').ok_or_else(|| bad("#expect <seconds> <regex>".into()))?;
                let secs: f64 = secs.parse().map_err(|_| bad(format!("deadline {secs:?}")))?;
                self.expect.push((secs, Regex::new(pat).map_err(bad)?));
            } else if let Some(pat) = line.strip_prefix("#fail ") {
                self.fail.push(Regex::new(pat).map_err(bad)?);
            } else if let Some(f) = line.strip_prefix("#fixture ") {
                self.fixtures.push(f.trim().to_owned());
            } else if line.trim_start().starts_with('#') || line.trim().is_empty() {
                continue;
            } else if let Some(name) = line.trim().strip_prefix("include ") {
                // The grammar of tools/run_profile.sh: the named profile first.
                let runs = self.runs.clone().ok_or_else(|| bad("include outside tools/runs".into()))?;
                if self.depth >= 12 {
                    return Err(bad("include depth over 12".into()).into());
                }
                let file = runs.join(format!("{}.env", name.trim()));
                let text = std::fs::read_to_string(&file).map_err(|e| bad(format!("{}: {e}", file.display())))?;
                self.depth += 1;
                let r = self.add(&text, &format!("{}.env", name.trim()));
                self.depth -= 1;
                r?;
            } else if let Some(k) = line.trim().strip_prefix("unset ") {
                self.env.retain(|(key, _)| key != k.trim());
            } else if let Some(n) = line.trim().strip_prefix("stop_after_world_frames ") {
                // As tools/run_profile.sh: the title ends itself after n world frames.
                self.env.push(("XNW_STOP_WORLD_FRAMES".to_owned(), n.trim().to_owned()));
            } else if ["secs ", "boot ", "expect ", "expectfile ", "forbid ", "forbid_after "].iter().any(|d| line.trim_start().starts_with(d)) {
                // Checks of tools/run_profile.sh; this checker has `#expect`.
                continue;
            } else {
                let kv = line.trim().strip_prefix("export ").unwrap_or(line.trim());
                let (k, v) = kv.split_once('=').ok_or_else(|| bad(format!("not KEY=VALUE: {kv:?}")))?;
                let v = v.trim();
                let v = v.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')).or_else(|| v.strip_prefix('"').and_then(|v| v.strip_suffix('"'))).unwrap_or(v);
                self.env.push((k.trim().to_owned(), v.to_owned()));
            }
        }
        Ok(())
    }

    /// `tools/runs/common.expect`, then `tools/runs/<name>.env` (if named);
    /// `tools/runs/local.vars` (if present) into [`Profile::local`].
    pub fn load(root: &Path, name: Option<&str>) -> Res<Profile> {
        let mut p = Profile::default();
        let local = root.join("tools/runs/local.vars");
        if let Ok(text) = std::fs::read_to_string(&local) {
            p.add(&text, "local.vars")?;
            p.local = std::mem::take(&mut p.env);
        }
        let common = root.join("tools/runs/common.expect");
        p.add(&std::fs::read_to_string(&common).map_err(|e| format!("{}: {e}", common.display()))?, "common.expect")?;
        if let Some(name) = name {
            p.runs = Some(root.join("tools/runs"));
            let file = root.join(format!("tools/runs/{name}.env"));
            p.add(&std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?, &format!("{name}.env"))?;
        }
        Ok(p)
    }
}

/// Seconds at the start of a log line (`    1.818 info  ...`).
fn stamp(line: &str) -> Option<f64> {
    line.split_whitespace().next()?.parse().ok()
}

/// Expectations over a log as it comes: the verdict of each line, and at
/// the end (or a deadline) what is missing.
pub struct Checker<'a> {
    profile: &'a Profile,
    met: Vec<Option<f64>>,
    pub failures: Vec<String>,
    pub now: f64,
    pub frame: u64,
    pub first_frame: Option<f64>,
}

impl<'a> Checker<'a> {
    pub fn new(profile: &'a Profile) -> Checker<'a> {
        Checker { profile, met: vec![None; profile.expect.len()], failures: Vec::new(), now: 0.0, frame: 0, first_frame: None }
    }

    pub fn feed(&mut self, line: &str) {
        if let Some(t) = stamp(line) {
            self.now = self.now.max(t);
        }
        if let Some(rest) = line.split("PresentX frame ").nth(1)
            && let Some(n) = rest.split(|c: char| !c.is_ascii_digit()).next().and_then(|n| n.parse::<u64>().ok())
        {
            self.frame = self.frame.max(n);
            self.first_frame.get_or_insert(self.now);
        }
        for r in &self.profile.fail {
            if r.search(line) {
                self.failures.push(format!("fail /{}/: {}", r.text, line.trim()));
            }
        }
        for (i, (_, r)) in self.profile.expect.iter().enumerate() {
            if self.met[i].is_none() && r.search(line) {
                self.met[i] = Some(self.now);
            }
        }
    }

    /// Expectations not met although the clock passed their deadline
    /// (`at` = the time reached; `end` = the log is finished).
    pub fn missing(&self, at: f64) -> Vec<String> {
        self.profile
            .expect
            .iter()
            .zip(&self.met)
            .filter_map(|((d, r), m)| match m {
                Some(t) if t <= d => None,
                Some(t) => Some(format!("expect /{}/ by {d} s: seen at {t:.1} s, late", r.text)),
                None if at >= *d => Some(format!("expect /{}/ by {d} s: not seen (log reached {at:.1} s)", r.text)),
                None => None,
            })
            .collect()
    }

    pub fn unchecked(&self, at: f64) -> Vec<String> {
        self.profile.expect.iter().zip(&self.met).filter(|((d, _), m)| m.is_none() && at < *d).map(|((d, r), _)| format!("/{}/ by {d} s", r.text)).collect()
    }
}

// ---- the run ----

/// The main checkout's root: a worktree's `.git` file names
/// `<main>/.git/worktrees/<name>`.
fn main_root() -> PathBuf {
    main_of(&std::env::current_dir().unwrap_or_default())
}

/// The main checkout of the checkout at `dir` (`dir` itself unless it is a
/// worktree).
fn main_of(dir: &Path) -> PathBuf {
    if let Ok(text) = std::fs::read_to_string(dir.join(".git"))
        && let Some(gitdir) = text.trim().strip_prefix("gitdir:")
        && let Some(main) = Path::new(gitdir.trim()).ancestors().nth(3)
    {
        return main.to_path_buf();
    }
    dir.to_path_buf()
}

/// The checkout this xnw.exe was built in (`<root>\target\<profile>\xnw.exe`).
fn exe_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let root = exe.ancestors().nth(3)?;
    root.join("Cargo.toml").is_file().then(|| root.to_path_buf())
}

/// `rel` under the first of `bases` that holds it (else `rel` as given).
fn first_existing(rel: &str, bases: &[PathBuf]) -> PathBuf {
    let p = Path::new(rel);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    bases.iter().map(|b| b.join(rel)).find(|c| c.exists()).unwrap_or_else(|| p.to_path_buf())
}

/// Where a smoke run from any worktree finds things. Profiles: the
/// exe's own checkout first (its tools\runs match its code), then the
/// current folder. Game files: the current folder, then the main checkout
/// of it and of the exe's checkout (worktrees hold no game\).
fn search_bases(profiles: bool) -> Vec<PathBuf> {
    let here = std::env::current_dir().unwrap_or_default();
    let exe = exe_root();
    let mut v = Vec::new();
    if profiles {
        v.extend(exe.clone());
        v.push(here.clone());
    } else {
        v.push(here.clone());
        v.push(main_of(&here));
        v.extend(exe.map(|e| main_of(&e)));
    }
    v
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

fn alive(pid: u32) -> bool {
    use xnw_core::win;
    // SAFETY: plain Win32 calls; the handle is closed.
    unsafe {
        let h = win::OpenProcess(win::SYNCHRONIZE, 0, pid);
        if h.is_null() {
            return false;
        }
        let r = win::WaitForSingleObject(h, 0);
        win::CloseHandle(h);
        r == 0x102
    }
}

/// Whether a title holds run.rs's `Global\XNW.title`.
fn title_running() -> bool {
    use xnw_core::win;
    let name = wide("Global\\XNW.title");
    // SAFETY: plain Win32 calls; the mutex is released and closed.
    unsafe {
        let m = win::OpenMutexW(win::SYNCHRONIZE, 0, name.as_ptr());
        if m.is_null() {
            return false;
        }
        let r = win::WaitForSingleObject(m, 0);
        if r == 0 || r == 0x80 {
            win::ReleaseMutex(m);
        }
        win::CloseHandle(m);
        r == 0x102
    }
}

/// The bench lock, shared with tools/title_lock.sh: `out\title.lock`,
/// one line `pid=<Windows pid> holder=<h> run=<r> start=<time> cwd=<folder>`
/// (older smoke locks began with the bare pid).
struct Lock(PathBuf);

/// The holder's pid of a lock line.
fn lock_pid(text: &str) -> Option<u32> {
    let w = text.split_whitespace().next()?;
    w.strip_prefix("pid=").unwrap_or(w).parse().ok()
}

/// Running `image` processes other than `skip`, from tasklist's CSV
/// (`"xnw.exe","1234",...`): `(image, pid)`.
fn running_images(csv: &str, image: &str, skip: u32) -> Vec<(String, u32)> {
    csv.lines()
        .filter_map(|l| {
            let mut f = l.trim().split("\",\"").map(|s| s.trim_matches('"'));
            let (name, pid) = (f.next()?, f.next()?.parse::<u32>().ok()?);
            (name.eq_ignore_ascii_case(image) && pid != skip).then(|| (name.to_owned(), pid))
        })
        .collect()
}

impl Lock {
    /// Refused while a title image other than us runs, or the lock's
    /// holder is alive; a dead holder's lock is stale and replaced. Created
    /// with `create_new`, so two takers cannot both win.
    fn take(path: PathBuf, run: &str) -> Result<Lock, String> {
        Lock::take_for(path, run, &std::env::var("TITLE_IMAGE").unwrap_or_else(|_| "xnw.exe".into()))
    }

    /// [`Lock::take`] with the title image named (tests use one that never runs).
    fn take_for(path: PathBuf, run: &str, image: &str) -> Result<Lock, String> {
        let csv = std::process::Command::new("tasklist").args(["/FI", &format!("IMAGENAME eq {image}"), "/NH", "/FO", "CSV"]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        let busy = running_images(&csv, &image, std::process::id());
        let held = || std::fs::read_to_string(&path).map(|t| t.lines().next().unwrap_or("").to_owned()).unwrap_or_else(|_| "none".into());
        if !busy.is_empty() {
            let list: Vec<String> = busy.iter().map(|(n, p)| format!("{n} {p}")).collect();
            return Err(format!("a title is running ({}); lock: {}", list.join(", "), held()));
        }
        if let Some(d) = path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let start = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let me = format!("pid={} holder=smoke run={run} start=@{start} cwd={}\n", std::process::id(), std::env::current_dir().unwrap_or_default().display());
        for _ in 0..2 {
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut f) => {
                    use std::io::Write as _;
                    f.write_all(me.as_bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
                    return Ok(Lock(path));
                }
                Err(_) => {
                    let text = held();
                    if lock_pid(&text).is_some_and(alive) {
                        return Err(format!("{} is held: {text}", path.display()));
                    }
                    eprintln!("smoke: stale title lock removed ({text})");
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        Err(format!("could not take {}", path.display()))
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        // Only ours: another taker may have replaced a lock it judged stale.
        if std::fs::read_to_string(&self.0).ok().as_deref().and_then(lock_pid) == Some(std::process::id()) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

/// The child's environment: ours, the profile's, then smoke's own
/// (null GPU from frame 0, WARP, headless, no launch prebuild).
fn resolve(profile: &Profile, record: Option<&str>) -> Result<Vec<(String, String)>, String> {
    let mut env: Vec<(String, String)> = profile.env.clone();
    let set = |env: &mut Vec<(String, String)>, k: &str, v: &str| {
        env.retain(|(a, _)| a != k);
        env.push((k.into(), v.into()));
    };
    let get = |env: &[(String, String)], k: &str| env.iter().rev().find(|(a, _)| a == k).map(|(_, v)| v.clone()).or_else(|| std::env::var(k).ok());
    // The options of `null` (`,ui=keep`) are kept; its point is not.
    let options: String = get(&env, "XNW_GPU").map(|g| g.split(',').skip(1).map(|o| format!(",{}", o.trim())).collect()).unwrap_or_default();
    if let Some(g) = get(&env, "XNW_GPU").filter(|g| g.split(',').next().map(str::trim) != Some("null")) {
        println!("smoke: XNW_GPU={g} -> null{options} (null GPU from frame 0)");
    }
    set(&mut env, "XNW_GPU", &format!("null{options}"));
    if get(&env, "XNW_FRAMESKIP").is_some() {
        env.retain(|(a, _)| a != "XNW_FRAMESKIP");
        set(&mut env, "XNW_FRAMESKIP", "");
    }
    let adapter = std::env::var("XNW_SMOKE_ADAPTER").unwrap_or_else(|_| "warp".into());
    set(&mut env, "XNW_ADAPTER", if adapter == "hardware" { "" } else { "warp" });
    // Anything that draws in null mode must not reach the hardware adapter.
    let drawing: Vec<&str> = ["XNW_WORLD_RENDER", "XNW_PRESENT_TEST", "XNW_SHOW"].into_iter().filter(|k| get(&env, k).is_some_and(|v| !v.is_empty())).collect();
    if adapter == "hardware" && !drawing.is_empty() {
        return Err(format!("refused: {} would draw on the hardware adapter in null mode (XNW_SMOKE_ADAPTER=hardware)", drawing.join(", ")));
    }
    for (k, v) in [("XNW_HEADLESS", "1"), ("XNW_PREBUILD_LAUNCH", "off"), ("XNW_GPU_DEADLINE", "0")] {
        set(&mut env, k, v);
    }
    for k in ["XNW_GAME_DRIVE", "XNW_AGILITY"] {
        if get(&env, k).is_none()
            && let Some((_, v)) = profile.local.iter().rev().find(|(a, _)| a == k)
        {
            set(&mut env, k, v);
        }
        if get(&env, k).is_none() {
            return Err(format!("refused: set {k} ({})", if k == "XNW_AGILITY" { "the Agility SDK folder" } else { "the game's package folder" }));
        }
    }
    if let Some(r) = record {
        set(&mut env, "XNW_IMPORT_TRACE", r);
    }
    Ok(env)
}

pub fn run(args: &[String]) -> Res<()> {
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let secs: f64 = arg("--secs").map_or(Ok(20.0), |s| s.parse()).map_err(|_| "--secs N")?;
    let profile_name = arg("--profile");
    let profile_root = search_bases(true).into_iter().find(|b| b.join("tools/runs/common.expect").is_file()).unwrap_or_else(|| PathBuf::from("."));
    let profile = Profile::load(&profile_root, profile_name.as_deref())?;
    // The game's executable: `--exe`, else `XNW_EXE`, else this machine's `local.vars`.
    let exe = arg("--exe")
        .or_else(|| std::env::var("XNW_EXE").ok().filter(|v| !v.is_empty()))
        .or_else(|| profile.local.iter().rev().find(|(k, _)| k == "XNW_EXE").map(|(_, v)| v.clone()))
        .unwrap_or_else(|| refuse("refused: name the game's executable (--exe <file> or XNW_EXE)"));
    // From a worktree (no game\, maybe older tools\runs) as from the main checkout.
    let exe = first_existing(&exe, &search_bases(false)).to_string_lossy().into_owned();
    let record = args.iter().position(|a| a == "--record").map(|i| match args.get(i + 1) {
        Some(f) if !f.starts_with("--") => f.clone(),
        _ => format!(r"out\imports.tsv,until={}", (secs - 2.0).max(1.0)),
    });
    let env = match resolve(&profile, record.as_deref()) {
        Ok(e) => e,
        Err(e) => refuse(&e),
    };
    if title_running() {
        refuse("refused: another title instance holds Global\\XNW.title");
    }
    // A log of this run's own.
    let run = format!("smoke-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()));
    let _lock = match Lock::take(main_root().join(r"out\title.lock"), &run) {
        Ok(l) => l,
        Err(e) => refuse(&format!("refused: {e}")),
    };
    std::fs::create_dir_all("out")?;
    let log_path = PathBuf::from(format!(r"out\{run}.log"));
    let mut log = std::io::BufWriter::new(std::fs::File::create(&log_path)?);
    let mut cmd = std::process::Command::new(std::env::current_exe()?);
    cmd.args(["run", &exe, "--execute"]).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).stdin(std::process::Stdio::null());
    for (k, v) in &env {
        if v.is_empty() {
            cmd.env_remove(k);
        } else {
            cmd.env(k, v);
        }
    }
    let shown: Vec<String> = env.iter().map(|(k, v)| format!("{k}={v}")).collect();
    println!("smoke: {exe}, {secs} s, profile {}, env {}", profile_name.as_deref().unwrap_or("-"), shown.join(" "));
    let started = Instant::now();
    let mut child = cmd.spawn().map_err(|e| format!("starting the title: {e}"))?;
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    for stream in [child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>), child.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>)].into_iter().flatten() {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stream).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
    }
    drop(tx);
    let mut check = Checker::new(&profile);
    let mut exit = None;
    let mut ended = "the time limit";
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(line) => {
                let _ = writeln!(log, "{line}");
                check.feed(&line);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                exit = child.wait().ok();
                ended = "the title's exit";
                break;
            }
            Err(_) => {}
        }
        let wall = started.elapsed().as_secs_f64();
        // The log clock, or the wall clock when the log is silent (a hang).
        let at = check.now.max(wall - 3.0);
        if !check.failures.is_empty() {
            ended = "a fail line";
            break;
        }
        if !check.missing(at).is_empty() {
            ended = "a missed expectation";
            break;
        }
        if wall >= secs {
            break;
        }
        if let Ok(Some(s)) = child.try_wait() {
            exit = Some(s);
            ended = "the title's exit";
            // drain what is left
            while let Ok(line) = rx.recv_timeout(Duration::from_millis(300)) {
                let _ = writeln!(log, "{line}");
                check.feed(&line);
            }
            break;
        }
    }
    if exit.is_none() {
        let _ = std::process::Command::new("taskkill").args(["/PID", &child.id().to_string(), "/T", "/F"]).output();
        let _ = child.wait();
        while let Ok(line) = rx.recv_timeout(Duration::from_millis(200)) {
            let _ = writeln!(log, "{line}");
            check.feed(&line);
        }
    }
    let _ = log.flush();
    drop(log);
    let _ = std::fs::copy(&log_path, r"out\smoke.log");
    let wall = started.elapsed().as_secs_f64();
    let reached = if exit.is_some() { f64::MAX } else { check.now };
    let mut bad = check.failures.clone();
    bad.extend(check.missing(reached));
    if let Some(s) = exit
        && bad.is_empty()
    {
        bad.push(format!("the title ended by itself ({s}) at {:.1} s", check.now));
    }
    println!(
        "smoke: ended by {ended} after {wall:.1} s wall (log {:.1} s); frame {} reached{}; log {}",
        check.now,
        check.frame,
        check.first_frame.map_or(String::new(), |t| format!(", frame 1 at {t:.1} s")),
        log_path.display()
    );
    let open = check.unchecked(reached);
    if !open.is_empty() {
        println!("smoke: not checked (deadline after the end): {}", open.join(", "));
    }
    if bad.is_empty() {
        println!("smoke: PASS");
        return Ok(());
    }
    for b in bad.iter().take(20) {
        println!("smoke: {b}");
    }
    println!("smoke: FAIL ({} problems)", bad.len());
    std::process::exit(1)
}

fn refuse(why: &str) -> ! {
    println!("smoke: {why}");
    std::process::exit(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(p: &str) -> Regex {
        Regex::new(p).unwrap()
    }

    #[test]
    fn the_regex_subset_matches_like_python() {
        assert!(r("PresentX frame 1 \\(").search("    1.818 info  PresentX frame 1 (2 planes): x"));
        assert!(!r("PresentX frame 1 \\(").search("PresentX frame 16 (2 planes)"));
        assert!(r("^\\s*\\d+\\.\\d+ WARN  EXCEPTION 0x[0-9a-f]+").search("    0.939 WARN  EXCEPTION 0x80000003 (breakpoint)"));
        assert!(r("a(b|cd)+e$").search("xxabcdbe"));
        assert!(!r("a(b|cd)+e$").search("xxae"));
        assert!(r("colou?r").search("color") && r("colou?r").search("colour"));
        assert!(r("x{2,3}y").search("axxy") && !r("^x{2,3}y").search("xy"));
        assert!(r("\\bframe\\b").search("a frame b") && !r("\\bframe\\b").search("frames"));
        assert!(r("[^a-c]z").search("az dz") && !r("[^a-c]z").search("az bz"));
        assert!(r("frames? (rendered|skipped)").search("30 frames rendered"));
        assert!(r("").search("anything"));
        assert!(r("a.*b.*c").search(&format!("a{}b{}c", "x".repeat(500), "y".repeat(500))));
        assert!(Regex::new("(ab").is_err() && Regex::new("*a").is_err());
    }

    #[test]
    fn profiles_parse_env_and_expectations() {
        let mut p = Profile::default();
        p.add("# comment\nXNW_A='x y'\nexport XNW_B=\"2\"\nXNW_C=3\n#expect 10 PresentX frame 1 \\(\n#fail panicked at\n#fixture tools/fixtures/x.log\n", "t").unwrap();
        assert_eq!(p.env, [("XNW_A".into(), "x y".into()), ("XNW_B".into(), "2".into()), ("XNW_C".into(), "3".into())]);
        assert_eq!(p.expect.len(), 1);
        assert_eq!(p.expect[0].0, 10.0);
        assert_eq!(p.fail.len(), 1);
        assert_eq!(p.fixtures, ["tools/fixtures/x.log"]);
        assert!(p.add("#expect x y\n", "t").is_err());
    }

    #[test]
    fn smoke_always_resolves_to_null_on_warp() {
        let mut p = Profile::default();
        p.add("XNW_GPU='null@frame+1250'\nXNW_FRAMESKIP=4\nXNW_PATCH=watchdog\nXNW_GAME_DRIVE=d\nXNW_AGILITY=a\n", "t").unwrap();
        let env = resolve(&p, None).unwrap();
        let get = |k: &str| env.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        assert_eq!(get("XNW_GPU"), Some("null"));
        assert_eq!(get("XNW_FRAMESKIP"), Some(""), "removed from the child");
        assert_eq!(get("XNW_HEADLESS"), Some("1"));
        assert_eq!(get("XNW_PATCH"), Some("watchdog"));
        if std::env::var("XNW_SMOKE_ADAPTER").is_err() {
            assert_eq!(get("XNW_ADAPTER"), Some("warp"));
        }
    }

    /// The game and Agility folders: the profile's, else this machine's
    /// `local.vars`; a run with neither is refused before it starts.
    #[test]
    fn smoke_folders_come_from_the_profile_or_local_env() {
        if std::env::var_os("XNW_GAME_DRIVE").is_some() || std::env::var_os("XNW_AGILITY").is_some() {
            return;
        }
        let mut p = Profile::default();
        p.add("XNW_AGILITY=a\n", "t").unwrap();
        assert!(resolve(&p, None).unwrap_err().contains("XNW_GAME_DRIVE"));
        p.local = vec![("XNW_GAME_DRIVE".into(), "d".into()), ("XNW_AGILITY".into(), "other".into())];
        let env = resolve(&p, None).unwrap();
        let get = |k: &str| env.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        assert_eq!((get("XNW_GAME_DRIVE"), get("XNW_AGILITY")), (Some("d"), Some("a")));
    }

    /// The repository root (tests run in `apps/xnw`).
    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// The bench lock shared with tools/title_lock.sh: parse, refuse a
    /// live holder (naming it), replace a dead holder's lock, release only ours.
    #[test]
    fn title_lock_refuses_live_and_replaces_stale() {
        assert_eq!(lock_pid("pid=4242 holder=run_profile run=r1 start=2026-11-24 20:00:00 cwd=y"), Some(4242));
        assert_eq!(lock_pid("4242 xnw smoke here"), Some(4242), "older smoke lock");
        assert_eq!(lock_pid("junk"), None);
        let csv = "\"xnw.exe\",\"100\",\"Console\",\"1\",\"9,000 K\"\r\n\"xnw.exe\",\"200\",\"Console\",\"1\",\"1 K\"\r\n";
        assert_eq!(running_images(csv, "xnw.exe", 100), vec![("xnw.exe".to_string(), 200)]);
        assert!(running_images("INFO: No tasks are running which match the specified criteria.", "xnw.exe", 0).is_empty());
        let dir = std::env::temp_dir().join(format!("xnw-title-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join(r"out\title.lock");
        let image = "xnw-lock-test-none.exe";
        // Live holder: this test process.
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let live = format!("pid={} holder=run_profile run=gate91 start=x cwd=y\n", std::process::id());
        std::fs::write(&path, &live).unwrap();
        let e = Lock::take_for(path.clone(), "s1", image).err().unwrap();
        assert!(e.contains("is held: pid=") && e.contains("holder=run_profile run=gate91"), "{e}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), live, "the holder's lock stays");
        // Dead holder: replaced, and ours is released on drop.
        std::fs::write(&path, "pid=4294967290 holder=smoke run=old start=x cwd=y\n").unwrap();
        {
            let _l = Lock::take_for(path.clone(), "s2", image).unwrap();
            let t = std::fs::read_to_string(&path).unwrap();
            assert!(t.starts_with(&format!("pid={} holder=smoke run=s2 start=@", std::process::id())), "{t}");
            assert!(Lock::take_for(path.clone(), "s3", image).is_err(), "a second taker while we hold it");
        }
        assert!(!path.exists(), "released on drop");
        // A lock someone else wrote after ours is not removed by our drop.
        {
            let _l = Lock::take_for(path.clone(), "s4", image).unwrap();
            std::fs::write(&path, "pid=4294967290 holder=other run=x start=x cwd=y\n").unwrap();
        }
        assert!(path.exists(), "not ours: kept");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A worktree resolves to its main checkout; the game is found in
    /// the first base that holds it; profiles in the exe's own checkout.
    #[test]
    fn worktrees_find_the_main_checkout_and_the_game() {
        let t = std::env::temp_dir().join(format!("xnw-smoke-roots-{}", std::process::id()));
        let main = t.join("main");
        let wt = t.join("wt");
        std::fs::create_dir_all(main.join(".git/worktrees/wt")).unwrap();
        std::fs::create_dir_all(main.join("game")).unwrap();
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(main.join(r"game\title.exe"), b"").unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", main.join(".git/worktrees/wt").display())).unwrap();
        assert_eq!(main_of(&wt), main);
        assert_eq!(main_of(&main), main);
        let rel = r"game\title.exe";
        assert_eq!(first_existing(rel, &[wt.clone(), main_of(&wt)]), main.join(rel));
        assert_eq!(first_existing(r"game\none.exe", &[wt.clone(), main.clone()]), PathBuf::from(r"game\none.exe"));
        assert_eq!(first_existing(r"C:\abs.exe", &[main.clone()]), PathBuf::from(r"C:\abs.exe"));
        // The test binary lives in <root>\target\debug\deps: not an xnw.exe checkout layout,
        // but the profile search still ends at a folder with tools\runs\common.expect here.
        assert!(root().join("tools/runs/common.expect").is_file());
        let _ = std::fs::remove_dir_all(&t);
    }

    /// As tools/expect_check.py: a finished log, deadlines after its end not checked.
    fn check_file(p: &Profile, log: &Path) -> (Vec<String>, Vec<String>, u64) {
        let mut c = Checker::new(p);
        for line in std::fs::read_to_string(log).unwrap().lines() {
            c.feed(line);
        }
        let mut bad = c.failures.clone();
        bad.extend(c.missing(c.now));
        (bad, c.unchecked(c.now), c.frame)
    }

    /// Every profile passes on the fixtures it names; the crash fixture
    /// fails the common expectations (as `tools/test_instruments.py`).
    #[test]
    fn profiles_pass_their_fixtures_and_the_crash_fails() {
        let root = root();
        let mut checked = 0;
        for e in std::fs::read_dir(root.join("tools/runs")).unwrap().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(stem) = name.strip_suffix(".env") else { continue };
            let p = Profile::load(&root, Some(stem)).unwrap();
            for f in &p.fixtures {
                let (bad, _, _) = check_file(&p, &root.join(f));
                assert!(bad.is_empty(), "{stem} on {f}: {bad:?}");
                checked += 1;
            }
        }
        assert!(checked > 0, "no profile names a fixture");
        let common = Profile::load(&root, None).unwrap();
        let (bad, open, frame) = check_file(&common, &root.join("tools/fixtures/crash-boot.log"));
        assert!(bad.iter().any(|b| b.contains("EXCEPTION")), "{bad:?}");
        assert!(open.iter().any(|b| b.contains("PresentX frame 1")), "{open:?}");
        assert_eq!(frame, 0);
    }
}
