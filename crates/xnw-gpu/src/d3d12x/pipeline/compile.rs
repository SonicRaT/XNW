//! Host pipeline compiles watched. Each making of a host pipeline (a
//! worker's `Pending::make`: translation through the cache, then the
//! driver's compile or a library load) is registered while it runs, so:
//!
//! - one taking over [`SLOW_LOG`] is logged with its programs, their DXIL
//!   sizes, its wall time and how many were compiling at once; the
//!   [`SLOWEST`] slowest are logged at exit;
//! - those still running at exit or at a host GPU failure are logged with
//!   their running time (a compile that never finishes is named);
//! - one past the soft deadline (`XNW_COMPILE_DEADLINE_S`, default 20 s) is
//!   logged loudly, its draws stay dropped for the rest of the run even if
//!   it finishes, and its key is written to `<cache>/pipelines/
//!   slow-pipelines.txt`: later runs do not build it in-run (its draws drop
//!   at once) unless `XNW_SLOW_PIPELINES=retry`; `xnw prebuild` builds them
//!   last, one at a time; `xnw prebuild --list-slow` lists them.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Makings logged when they take longer.
const SLOW_LOG: Duration = Duration::from_secs(1);
/// Slowest makings kept for the exit summary.
const SLOWEST: usize = 20;
pub(crate) const SLOW_FILE: &str = "slow-pipelines.txt";

/// The soft deadline of one making.
pub(crate) fn deadline() -> Duration {
    static D: OnceLock<Duration> = OnceLock::new();
    *D.get_or_init(|| Duration::from_secs_f64(std::env::var("XNW_COMPILE_DEADLINE_S").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|&s| s > 0.0).unwrap_or(20.0)))
}

struct Running {
    key: String,
    label: String,
    started: Instant,
    /// Set past the deadline: the pipeline's draws drop for good.
    slow: Arc<AtomicBool>,
    warned: bool,
}

#[derive(Default)]
struct State {
    next: u64,
    running: HashMap<u64, Running>,
    /// (milliseconds, description) of the slowest finished makings.
    slowest: Vec<(u64, String)>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
/// Makings running now.
pub(crate) static RUNNING: AtomicU64 = AtomicU64::new(0);
/// Makings that passed the deadline this run.
pub(crate) static OVERDUE: AtomicU64 = AtomicU64::new(0);

fn state() -> std::sync::MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// A making in progress; finished by [`Ticket::done`].
pub(crate) struct Ticket {
    id: u64,
}

/// Registers a making of the pipeline `key` (programs `label`), with the
/// flag the deadline sets.
pub(crate) fn begin(key: &str, label: String, slow: Arc<AtomicBool>) -> Ticket {
    watchdog();
    let mut s = state();
    let s = s.get_or_insert_with(State::default);
    s.next += 1;
    let id = s.next;
    s.running.insert(id, Running { key: key.to_owned(), label, started: Instant::now(), slow, warned: false });
    RUNNING.fetch_add(1, Ordering::Relaxed);
    // On disk before the compile starts, so a device loss or a crash
    // leaves the suspects named.
    write_inflight(&s.running);
    Ticket { id }
}

/// `<cache>/pipelines/inflight.txt`: the makings running now (key,
/// programs, seconds running at the write), rewritten at each start and end.
fn write_inflight(running: &HashMap<u64, Running>) {
    let Some(dir) = DIR.get() else { return };
    let mut rows: Vec<&Running> = running.values().collect();
    rows.sort_by_key(|r| r.started);
    let mut text = format!("# xnw host pipelines compiling now (translator {})\n", xnw_shader::emit::VERSION);
    for r in rows {
        text.push_str(&format!("{}\t{}\t{:.1} s\n", r.key, r.label, r.started.elapsed().as_secs_f64()));
    }
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(INFLIGHT_FILE), text);
}

pub(crate) const INFLIGHT_FILE: &str = "inflight.txt";
pub(crate) const SUSPECT_FILE: &str = "suspect.txt";

/// The host device was lost while pipelines compiled: those in flight
/// (once per process) are appended to `suspect.txt` with the date and the
/// translator version; both prebuilds and the in-run build skip them until
/// `--retry-suspects` / `XNW_RETRY_SUSPECTS=1`. Returns how many.
pub(crate) fn device_lost(why: &str) -> usize {
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Ordering::Relaxed) {
        return 0;
    }
    let rows: Vec<(String, String, f64)> = state().as_ref().map(|s| s.running.values().map(|r| (r.key.clone(), r.label.clone(), r.started.elapsed().as_secs_f64())).collect()).unwrap_or_default();
    let date = utc_now();
    let mut text = String::new();
    for (key, label, secs) in &rows {
        xnw_core::warn!("d3d12x: SUSPECT (in flight at {why}, {secs:.1} s compiling): {label} key {key}");
        text.push_str(&format!("{key}\t{label}\t{date}\ttranslator {}\t{secs:.1} s\t{why}\n", xnw_shader::emit::VERSION));
    }
    if let Some(dir) = DIR.get()
        && !text.is_empty()
    {
        use std::io::Write;
        let _ = std::fs::create_dir_all(dir);
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(SUSPECT_FILE)) {
            let _ = f.write_all(text.as_bytes());
        }
        SUSPECTS.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
    rows.len()
}

/// The time now as `YYYY-MM-DD hh:mm:ss UTC`.
fn utc_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil from days (H. Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC", rem / 3600, rem % 3600 / 60, rem % 60)
}

static SUSPECTS: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);
static RETRY_SUSPECTS: AtomicBool = AtomicBool::new(false);

/// `--retry-suspects`.
pub(crate) fn retry_suspects(on: bool) {
    RETRY_SUSPECTS.store(on, Ordering::Relaxed);
}

/// Whether `key` was in flight at an earlier device loss (and is not
/// to be retried).
pub(crate) fn listed_suspect(key: &str) -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    if RETRY_SUSPECTS.load(Ordering::Relaxed) || *ENV.get_or_init(|| std::env::var("XNW_RETRY_SUSPECTS").is_ok_and(|v| v == "1")) {
        return false;
    }
    let mut g = SUSPECTS.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_none() {
        let text = DIR.get().and_then(|d| std::fs::read_to_string(d.join(SUSPECT_FILE)).ok()).unwrap_or_default();
        *g = Some(parse_slow(&text));
    }
    g.as_ref().is_some_and(|s| s.contains_key(key))
}

/// Past this a running making is named every 5 s ("STILL COMPILING").
static STILL_AFTER_MS: AtomicU64 = AtomicU64::new(0);

/// `--timeout <s>` (0: the deadline).
pub(crate) fn still_after(d: Duration) {
    STILL_AFTER_MS.store(d.as_millis() as u64, Ordering::Relaxed);
}

fn still_after_now() -> Duration {
    match STILL_AFTER_MS.load(Ordering::Relaxed) {
        0 => deadline(),
        ms => Duration::from_millis(ms),
    }
}

impl Ticket {
    /// The making ended; `detail` names the DXIL sizes and the outcome.
    pub(crate) fn done(self, detail: &str) {
        let at_once = RUNNING.fetch_sub(1, Ordering::Relaxed);
        let mut s = state();
        let Some(s) = s.as_mut() else { return };
        let Some(r) = s.running.remove(&self.id) else { return };
        write_inflight(&s.running);
        let took = r.started.elapsed();
        let line = format!("{} [{detail}] key {}", r.label, r.key);
        if took >= SLOW_LOG {
            xnw_core::warn!("d3d12x: slow host pipeline: {:.1} s, {at_once} making at once: {line}", took.as_secs_f64());
        }
        let ms = took.as_millis() as u64;
        if s.slowest.len() < SLOWEST || s.slowest.last().is_some_and(|l| l.0 < ms) {
            s.slowest.push((ms, line));
            s.slowest.sort_by(|a, b| b.0.cmp(&a.0));
            s.slowest.truncate(SLOWEST);
        }
    }
}

/// The makings past `deadline` among `running`, not warned yet.
fn overdue(running: &HashMap<u64, Running>, now: Instant, deadline: Duration) -> Vec<u64> {
    running.iter().filter(|(_, r)| !r.warned && now.duration_since(r.started) >= deadline).map(|(&id, _)| id).collect()
}

/// Checks the deadline once a second (started with the first making).
fn watchdog() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xnw-compile-watch".into()).spawn(|| {
            let mut ticks = 0u64;
            loop {
                std::thread::sleep(Duration::from_secs(1));
                ticks += 1;
                let mut late = Vec::new();
                {
                    let mut s = state();
                    let Some(s) = s.as_mut() else { continue };
                    // Every 5 s, each making past the timeout is named.
                    if ticks.is_multiple_of(5) {
                        let after = still_after_now();
                        for r in s.running.values().filter(|r| r.started.elapsed() >= after) {
                            xnw_core::warn!("d3d12x: STILL COMPILING {} after {:.0} s: {} ({} making at once)", r.key, r.started.elapsed().as_secs_f64(), r.label, s.running.len());
                        }
                    }
                    for id in overdue(&s.running, Instant::now(), deadline()) {
                        if let Some(r) = s.running.get_mut(&id) {
                            r.warned = true;
                            r.slow.store(true, Ordering::Relaxed);
                            late.push((r.key.clone(), r.label.clone(), r.started.elapsed()));
                        }
                    }
                }
                for (key, label, took) in late {
                    OVERDUE.fetch_add(1, Ordering::Relaxed);
                    xnw_core::warn!(
                        "d3d12x: HOST PIPELINE PAST ITS DEADLINE: {:.0} s and still compiling ({} making at once): {label} key {key}; its draws stay dropped, listed as too slow",
                        took.as_secs_f64(),
                        RUNNING.load(Ordering::Relaxed)
                    );
                    mark_slow(&key, &label);
                }
            }
        });
    });
}

/// Logs the makings still running (`why`: exit, a host GPU failure) and,
/// at exit, the slowest finished.
pub(crate) fn report(why: &str) {
    let s = state();
    let Some(s) = s.as_ref() else { return };
    let mut running: Vec<&Running> = s.running.values().collect();
    running.sort_by_key(|r| r.started);
    for r in &running {
        xnw_core::warn!("d3d12x: host pipeline still compiling at {why}: {:.1} s: {} key {}", r.started.elapsed().as_secs_f64(), r.label, r.key);
    }
    if !s.slowest.is_empty() {
        xnw_core::info!("d3d12x: slowest host pipelines ({why}, {} still running, {} past the deadline):", running.len(), OVERDUE.load(Ordering::Relaxed));
        for (ms, line) in &s.slowest {
            xnw_core::info!("d3d12x:   {:.2} s {line}", *ms as f64 / 1000.0);
        }
    }
}

// ---- the too-slow list -------------------------------------------------

static DIR: OnceLock<PathBuf> = OnceLock::new();
static SLOW: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

/// The folder of the list (the pipeline cache folder), set once.
pub(crate) fn use_dir(dir: PathBuf) {
    let _ = DIR.set(dir);
}

fn slow() -> std::sync::MutexGuard<'static, Option<HashMap<String, String>>> {
    let mut g = SLOW.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_none() {
        let text = DIR.get().and_then(|d| std::fs::read_to_string(d.join(SLOW_FILE)).ok()).unwrap_or_default();
        *g = Some(parse_slow(&text));
    }
    g
}

/// `key \t programs` per line; `#` lines are comments.
pub(crate) fn parse_slow(text: &str) -> HashMap<String, String> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (key, label) = l.split_once('\t').unwrap_or((l, ""));
            let key = key.trim();
            (key.len() == 32 && key.bytes().all(|b| b.is_ascii_hexdigit())).then(|| (key.to_owned(), label.trim().to_owned()))
        })
        .collect()
}

fn slow_text(all: &HashMap<String, String>) -> String {
    let mut rows: Vec<(&String, &String)> = all.iter().collect();
    rows.sort();
    let mut text = String::from("# xnw pipelines past the compile deadline: key, programs\n");
    for (k, l) in rows {
        text.push_str(&format!("{k}\t{l}\n"));
    }
    text
}

fn mark_slow(key: &str, label: &str) {
    let mut g = slow();
    let all = g.get_or_insert_with(HashMap::new);
    if all.insert(key.to_owned(), label.to_owned()).is_none()
        && let Some(dir) = DIR.get()
    {
        let _ = std::fs::create_dir_all(dir);
        let tmp = dir.join(format!(".{SLOW_FILE}.tmp-{}", std::process::id()));
        if std::fs::write(&tmp, slow_text(all)).and_then(|()| std::fs::rename(&tmp, dir.join(SLOW_FILE))).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

/// Whether the list names `key` as too slow to build in-run (always `false`
/// with `XNW_SLOW_PIPELINES=retry`).
pub(crate) fn listed_slow(key: &str) -> bool {
    static RETRY: OnceLock<bool> = OnceLock::new();
    if *RETRY.get_or_init(|| std::env::var("XNW_SLOW_PIPELINES").is_ok_and(|v| v == "retry")) {
        return false;
    }
    slow().as_ref().is_some_and(|s| s.contains_key(key))
}

/// The listed too-slow pipelines (key, programs).
pub(crate) fn slow_list() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = slow().as_ref().map(|s| s.iter().map(|(k, l)| (k.clone(), l.clone())).collect()).unwrap_or_default();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_deadline_finds_each_overdue_making_once() {
        let now = Instant::now();
        let mut running = HashMap::new();
        let run = |ago: u64, warned| Running { key: "k".into(), label: "l".into(), started: now - Duration::from_secs(ago), slow: Arc::new(AtomicBool::new(false)), warned };
        running.insert(1, run(25, false));
        running.insert(2, run(5, false));
        running.insert(3, run(40, true));
        assert_eq!(overdue(&running, now, Duration::from_secs(20)), [1]);
        assert!(overdue(&running, now, Duration::from_secs(60)).is_empty());
    }

    #[test]
    fn the_date_is_a_utc_time() {
        let d = utc_now();
        assert_eq!(d.len(), 23, "{d}");
        assert!(d.starts_with("20") && d.ends_with(" UTC"));
        assert_eq!(&d[4..5], "-");
    }

    #[test]
    fn the_slow_list_round_trips_and_skips_damage() {
        let key = "0123456789abcdef0123456789abcdef";
        let all = HashMap::from([(key.to_owned(), "vertex 1 + pixel 2".to_owned())]);
        let text = slow_text(&all);
        assert_eq!(parse_slow(&text), all);
        let damaged = format!("{text}short\tx\n{key}zz\ty\n");
        assert_eq!(parse_slow(&damaged), all);
    }

    #[test]
    fn a_finished_making_is_kept_among_the_slowest() {
        let t = begin("key-a", "compute 1".into(), Arc::new(AtomicBool::new(false)));
        assert!(RUNNING.load(Ordering::Relaxed) >= 1);
        t.done("1 KiB DXIL, made");
        let s = state();
        assert!(s.as_ref().unwrap().slowest.iter().any(|(_, l)| l.contains("key-a")));
    }
}
