//! Shared plumbing: the log, the per-run output folder, the `perf`
//! counters and the Win32 declarations the other crates use.

pub mod imprec;
pub mod perf;
pub mod syscalls;
pub mod win;

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static START: OnceLock<Instant> = OnceLock::new();

/// A fault that must end the run (an object guard page hit): the
/// crash logger stops the process once it has logged the fault.
/// The title's static TLS index (the loader sets it), for reading
/// another thread's title TLS (the frame threads line names the render
/// and sub-render threads by their context).
pub static TITLE_TLS_INDEX: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub static FATAL_FAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FILE: Mutex<Option<File>> = Mutex::new(None);
static RUN_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The run folder [`init_log`] was given, if any (diagnostic dumps go there).
pub fn run_dir() -> Option<&'static Path> {
    RUN_DIR.get().map(PathBuf::as_path)
}

/// A 128-bit content hash: two SipHash-1-3 passes (std's hasher) with
/// different fixed prefixes. Stable for a pinned toolchain; not
/// cryptographic. Shader-cache keys and logged shader hashes use it.
pub fn hash128(parts: &[&[u8]]) -> u128 {
    use std::hash::Hasher;
    let half = |seed: u64| {
        let mut h = std::hash::DefaultHasher::new();
        h.write_u64(seed);
        for p in parts {
            h.write_usize(p.len());
            h.write(p);
        }
        h.finish()
    };
    (u128::from(half(0x584e_575f_6861_7368)) << 64) | u128::from(half(0x9e37_79b9_7f4a_7c15))
}

/// Creates the next `<root>/run/<n>/` folder and returns it.
pub fn new_run_dir(root: &Path) -> std::io::Result<PathBuf> {
    let runs = root.join("run");
    std::fs::create_dir_all(&runs)?;
    let last = std::fs::read_dir(&runs)?
        .flatten()
        .filter_map(|e| e.file_name().to_str()?.parse::<u32>().ok())
        .max()
        .unwrap_or(0);
    let dir = runs.join(format!("{:04}", last + 1));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Starts the clock and, with a folder, mirrors the log to `<dir>/xnw.log`.
pub fn init_log(dir: Option<&Path>) {
    START.get_or_init(Instant::now);
    if let Some(dir) = dir {
        let _ = RUN_DIR.set(dir.to_path_buf());
        *FILE.lock().unwrap_or_else(|e| e.into_inner()) = File::create(dir.join("xnw.log")).ok();
    }
}

/// Time since the log clock started: the time every log line shows.
pub fn uptime() -> std::time::Duration {
    START.get_or_init(Instant::now).elapsed()
}

/// One log line: seconds since start, tag, text. Goes to stderr and the run
/// log. Never blocks on the file for long: a line is dropped from the file
/// only if another thread holds it for 2 ms (crash handlers log while
/// threads are suspended); two threads logging at once both get in.
pub fn log_line(tag: &str, text: std::fmt::Arguments<'_>) {
    let t = START.get_or_init(Instant::now).elapsed().as_secs_f64();
    let line = format!("{t:9.3} {tag:<5} {text}\n");
    let _ = std::io::stderr().write_all(line.as_bytes());
    let give_up = Instant::now() + std::time::Duration::from_millis(2);
    loop {
        match FILE.try_lock() {
            Ok(mut f) => {
                if let Some(f) = f.as_mut() {
                    let _ = f.write_all(line.as_bytes());
                }
                return;
            }
            Err(std::sync::TryLockError::Poisoned(e)) => {
                if let Some(f) = e.into_inner().as_mut() {
                    let _ = f.write_all(line.as_bytes());
                }
                return;
            }
            Err(std::sync::TryLockError::WouldBlock) if Instant::now() < give_up => std::thread::yield_now(),
            Err(std::sync::TryLockError::WouldBlock) => return,
        }
    }
}

#[macro_export]
macro_rules! info {
    ($($a:tt)*) => { $crate::log_line("info", format_args!($($a)*)) };
}

#[macro_export]
macro_rules! warn {
    ($($a:tt)*) => { $crate::log_line("WARN", format_args!($($a)*)) };
}

/// A traced call or event; same sink, own tag so it can be filtered out.
#[macro_export]
macro_rules! trace {
    ($($a:tt)*) => { $crate::log_line("trace", format_args!($($a)*)) };
}
