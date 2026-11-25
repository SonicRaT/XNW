//! The per-submission GPU-time watchdog of the offline lab and replay
//! "Card safety").
//!
//! The earlier watchdog timed a whole frame (CPU recording, uploads,
//! texture reads, pipeline builds and the GPU wait) against 1 s, so a cold
//! first frame whose CPU work alone took over 1 s exited although the card
//! was idle (a test run step 2: pk5 f977 records 1080 ms on the CPU). This one
//! times only what the card runs: each submission from its
//! ExecuteCommandLists to its fence. Every submitter brackets its fence
//! wait with [`begin`] / [`end`]; a thread exits the process (code 3) when
//! one submission's wait passes the budget, well under the driver's 2 s
//! TDR. Exiting does not stop work already queued on the card: what keeps
//! a frame under the TDR is a small submission (the world pass's `chunk`),
//! the watchdog only stops the next one being sent.
//!
//! Every submission is logged (label, fence-wait ms, timestamp ms where the
//! submitter has a pair of timestamps) for the frame line ([`take`]), on
//! either adapter; the exit only once [`start`] ran (the hardware adapter,
//! or a test with a tiny budget).

use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// One submission: who sent it, its fence wait, its GPU time by timestamps.
#[derive(Clone, Copy, Debug)]
pub struct Sub {
    pub label: &'static str,
    pub wait_ms: f64,
    pub stamp_ms: Option<f64>,
}

/// (process start, budget) once started; `SENT` is the submission in flight
/// in microseconds since start + 1 (0: none), `LABEL` its submitter.
static WATCH: OnceLock<(Instant, Duration)> = OnceLock::new();
static SENT: AtomicU64 = AtomicU64::new(0);
static LABEL: Mutex<&'static str> = Mutex::new("");
static LOG: Mutex<Vec<Sub>> = Mutex::new(Vec::new());

/// Starts the exit: a submission whose fence wait passes `budget` ends the
/// process with code 3. Once per process; later calls keep the first budget.
pub fn start(budget: Duration) {
    if WATCH.set((Instant::now(), budget)).is_err() {
        return;
    }
    std::thread::spawn(|| {
        let (t0, budget) = *WATCH.get().expect("set above");
        loop {
            std::thread::sleep(Duration::from_millis(10));
            let s = SENT.load(Ordering::Acquire);
            if s != 0 && t0.elapsed().as_micros() as u64 + 1 > s + budget.as_micros() as u64 {
                let label = LABEL.lock().map_or("?", |g| *g);
                eprintln!("xnw: gpu watchdog: one submission ({label}) has run over {} ms on the GPU (its fence not reached): exiting before the driver's 2 s timeout", budget.as_millis());
                std::process::exit(3);
            }
        }
    });
}

/// The budget once started (the hardware adapter), else None.
pub fn budget() -> Option<Duration> {
    WATCH.get().map(|w| w.1)
}

/// A submission is sent: call right after ExecuteCommandLists, before the wait.
pub fn begin(label: &'static str) -> Instant {
    let now = Instant::now();
    if let Some((t0, _)) = WATCH.get() {
        if let Ok(mut g) = LABEL.lock() {
            *g = label;
        }
        SENT.store(now.duration_since(*t0).as_micros() as u64 + 1, Ordering::Release);
    }
    now
}

/// Its fence was reached: logs the wait (and the timestamps' ms if any).
pub fn end(label: &'static str, sent: Instant, stamp_ms: Option<f64>) -> f64 {
    SENT.store(0, Ordering::Release);
    let wait_ms = sent.elapsed().as_secs_f64() * 1e3;
    if let Ok(mut g) = LOG.lock()
        && g.len() < 1 << 16
    {
        g.push(Sub { label, wait_ms, stamp_ms });
    }
    wait_ms
}

/// The submissions logged since the last call.
pub fn take() -> Vec<Sub> {
    LOG.lock().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default()
}

/// The frame line: `gpu watch: <what>: CPU <ms> ms, <n> submissions: fence wait longest <ms> (<label>), sum <ms>; timestamps longest <ms>; budget <ms> ms | off`.
pub fn line(what: &str, frame_ms: f64, subs: &[Sub]) -> String {
    let sum: f64 = subs.iter().map(|s| s.wait_ms).sum();
    let long = subs.iter().copied().max_by(|a, b| a.wait_ms.total_cmp(&b.wait_ms));
    let stamp = subs.iter().filter_map(|s| s.stamp_ms).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
    let mut labels: Vec<(&str, usize)> = Vec::new();
    for s in subs {
        match labels.iter_mut().find(|(l, _)| *l == s.label) {
            Some(e) => e.1 += 1,
            None => labels.push((s.label, 1)),
        }
    }
    format!(
        "gpu watch: {what}: CPU {:.1} ms (recording, uploads, reads, pipelines), {} submissions ({}): fence wait longest {:.2} ms{}, sum {sum:.2} ms; timestamps longest {}; budget {}",
        (frame_ms - sum).max(0.0),
        subs.len(),
        labels.iter().map(|(l, n)| format!("{l} {n}")).collect::<Vec<_>>().join(", "),
        long.map_or(0.0, |s| s.wait_ms),
        long.map_or(String::new(), |s| format!(" ({})", s.label)),
        stamp.map_or("-".into(), |v| format!("{v:.2} ms")),
        budget().map_or("off (software adapter)".into(), |b| format!("{} ms per submission", b.as_millis())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_splits_cpu_from_submissions() {
        let subs = [Sub { label: "pass", wait_ms: 3.0, stamp_ms: Some(2.5) }, Sub { label: "pass", wait_ms: 5.0, stamp_ms: Some(4.0) }, Sub { label: "cull check", wait_ms: 1.0, stamp_ms: None }];
        let l = line("frame 0", 100.0, &subs);
        assert!(l.contains("CPU 91.0 ms"), "{l}");
        assert!(l.contains("3 submissions (pass 2, cull check 1)"), "{l}");
        assert!(l.contains("longest 5.00 ms (pass), sum 9.00 ms; timestamps longest 4.00 ms"), "{l}");
    }

    #[test]
    fn begin_end_logs_the_wait() {
        let t = begin("t");
        let w = end("t", t, None);
        assert!(w >= 0.0);
        assert!(take().iter().any(|s| s.label == "t"));
    }
}
