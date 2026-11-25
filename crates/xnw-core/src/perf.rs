//! `XNW_TRACE=perf`: cheap counters per runtime area (calls, time, bytes),
//! summed up in the log every 10 s, deltas and totals. Off, a counted call
//! costs one load of a cached flag.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// What is counted. `Peak` areas keep a maximum in their byte field.
#[derive(Clone, Copy)]
pub enum Area {
    /// Title path calls (`CreateFileW`, attributes, CRT `fopen`, ...).
    FileOpen,
    /// DirectStorage file opens.
    DsOpen,
    /// DirectStorage file reads (by workers), bytes read.
    DsRead,
    /// zlib inflation, bytes produced.
    DsInflate,
    /// BCPack decoding, bytes produced.
    DsBcpack,
    /// Writes to physical-page destinations (views mapped per request).
    DsPages,
    /// Whole requests on workers, destination bytes.
    DsRequest,
    /// Title threads blocked in enqueue on a full queue.
    DsBlocked,
    /// Title threads blocked in `Close`.
    DsClose,
    /// Reads waiting in the scheduler, peak (sampled at submit).
    DsDepth,
    /// Reads from files on the full package, bytes.
    DsReadPackage,
    /// `XMemVirtualAlloc`, bytes asked.
    XmemAlloc,
    /// The title's `VirtualAlloc`/`VirtualFree` on its XMem regions.
    XmemFree,
    /// `XMemMapPhysicalPages` and scatter (maps and unmaps), bytes.
    XmemMap,
    /// `XMemGetWorkingSetStatistics`.
    XmemStats,
    /// `XMemAllocatePhysicalPages`/`XMemFreePhysicalPages`.
    XmemPhysical,
}

const NAMES: [&str; 16] = [
    "files: path calls",
    "dstorage: opens",
    "dstorage: file reads",
    "dstorage: zlib",
    "dstorage: bcpack",
    "dstorage: page writes",
    "dstorage: requests",
    "dstorage: enqueue blocked",
    "dstorage: close waits",
    "dstorage: depth (peak reads waiting)",
    "dstorage: reads from the full package",
    "xmem: virtual alloc",
    "xmem: title VirtualAlloc/Free on regions",
    "xmem: map/unmap physical",
    "xmem: working-set statistics",
    "xmem: physical alloc/free",
];

/// Per area: calls, nanoseconds, bytes (or the peak).
static COUNTERS: [[AtomicU64; 3]; NAMES.len()] = [const { [const { AtomicU64::new(0) }; 3] }; NAMES.len()];

/// Whether `XNW_TRACE` names `perf`.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XNW_TRACE").is_ok_and(|t| t.to_ascii_lowercase().split(',').any(|f| f.trim() == "perf"));
        if on {
            let _ = std::thread::Builder::new().name("perf".into()).spawn(report);
        }
        on
    })
}

/// Counts one call of `area` that took `time` and moved `bytes`.
pub fn add(area: Area, time: Duration, bytes: u64) {
    let c = &COUNTERS[area as usize];
    c[0].fetch_add(1, Ordering::Relaxed);
    c[1].fetch_add(time.as_nanos() as u64, Ordering::Relaxed);
    c[2].fetch_add(bytes, Ordering::Relaxed);
}

/// Keeps the largest `value` seen for a `Peak` area.
pub fn peak(area: Area, value: u64) {
    if on() {
        let c = &COUNTERS[area as usize];
        c[0].fetch_add(1, Ordering::Relaxed);
        c[2].fetch_max(value, Ordering::Relaxed);
    }
}

/// Runs `f`, counted under `area` with `bytes` when perf is on.
#[inline]
pub fn time<R>(area: Area, bytes: u64, f: impl FnOnce() -> R) -> R {
    if !on() {
        return f();
    }
    let started = Instant::now();
    let r = f();
    add(area, started.elapsed(), bytes);
    r
}

/// A start time when perf is on, for [`since`].
#[inline]
pub fn start() -> Option<Instant> {
    on().then(Instant::now)
}

/// Counts the time since `started` (from [`start`]) under `area`.
#[inline]
pub fn since(area: Area, started: Option<Instant>, bytes: u64) {
    if let Some(s) = started {
        add(area, s.elapsed(), bytes);
    }
}

/// The summary lines: per area with calls, the last period's calls, time
/// and bytes, then totals since the start.
fn report() {
    let mut last = [[0u64; 3]; NAMES.len()];
    loop {
        std::thread::sleep(Duration::from_secs(10));
        let now: Vec<[u64; 3]> = COUNTERS.iter().map(|c| [0, 1, 2].map(|i| c[i].load(Ordering::Relaxed))).collect();
        for (i, name) in NAMES.iter().enumerate() {
            let [calls, nanos, bytes] = now[i];
            if calls == 0 {
                continue;
            }
            let [dc, dn, db] = [0, 1, 2].map(|k| now[i][k] - last[i][k].min(now[i][k]));
            if i == Area::DsDepth as usize {
                crate::log_line("perf", format_args!("{name}: {bytes} ({calls} samples)"));
                continue;
            }
            crate::log_line(
                "perf",
                format_args!(
                    "{name}: last 10 s {dc} calls {:.3} s {:.1} MB; total {calls} calls {:.3} s {:.1} MB{}",
                    dn as f64 / 1e9,
                    db as f64 / 1e6,
                    nanos as f64 / 1e9,
                    bytes as f64 / 1e6,
                    if nanos > 0 && bytes > 0 { format!(" ({:.0} MB/s per busy thread)", bytes as f64 / 1e6 / (nanos as f64 / 1e9)) } else { String::new() }
                ),
            );
        }
        last.copy_from_slice(&now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_add_and_peak() {
        add(Area::XmemStats, Duration::from_nanos(5), 7);
        add(Area::XmemStats, Duration::from_nanos(5), 7);
        let c = &COUNTERS[Area::XmemStats as usize];
        assert!(c[0].load(Ordering::Relaxed) >= 2 && c[2].load(Ordering::Relaxed) >= 14);
        assert_eq!(NAMES.len(), Area::XmemPhysical as usize + 1);
    }
}
