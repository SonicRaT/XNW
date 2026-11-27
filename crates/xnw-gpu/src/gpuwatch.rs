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
//!
//! On either adapter a submitter that stops waiting for its fence (its own
//! limit, or [`overdue`]) calls [`abandon`]: the submission still runs, so
//! the process ends a few seconds later instead of waiting behind it.

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
/// The fence and value of the submission in flight, when its
/// submitter does not wait for it at once ([`fence`]), with its `SENT`.
static FENCE: Mutex<Option<(Fence, u64, u64)>> = Mutex::new(None);

/// A fence the watchdog thread reads (D3D12 objects are free-threaded).
struct Fence(windows::Win32::Graphics::Direct3D12::ID3D12Fence);
// SAFETY: ID3D12Fence is free-threaded; only GetCompletedValue is called on it.
unsafe impl Send for Fence {}

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
            // A submission not waited for at once ([`fence`]) ends at its fence.
            if s != 0
                && let Ok(g) = FENCE.lock()
                && let Some((f, v, at)) = g.as_ref()
                && *at == s
                // SAFETY: a plain query on a live fence.
                && unsafe { f.0.GetCompletedValue() } >= *v
            {
                let _ = SENT.compare_exchange(s, 0, Ordering::AcqRel, Ordering::Acquire);
                continue;
            }
            if s != 0 && t0.elapsed().as_micros() as u64 + 1 > s + budget.as_micros() as u64 {
                let label = LABEL.lock().map_or("?", |g| *g);
                eprintln!("xnw: gpu watchdog: one submission ({label}) has run over {} ms on the GPU (its fence not reached): exiting before the driver's 2 s timeout", budget.as_millis());
                std::process::exit(3);
            }
        }
    });
}

/// The longest one submission may run on the software adapter before its
/// submitter gives it up ([`overdue`]): the compute runner's and the
/// lighting draws' own limit there. On the hardware adapter the watchdog
/// thread exits long before.
pub const SOFT_LIMIT_S: f64 = 600.0;

/// How long the process lives after a submission was given up
/// ([`abandon`]): time for the step to print what it has.
pub const GRACE: Duration = Duration::from_secs(10);

/// A submitter gave up on a fence (`label`, after `seconds`) and the
/// adapter still runs that submission. Nothing sent to its queue after it
/// reaches its fence, and releasing the queue waits for it for ever (the
/// software adapter runs it on this process's threads: `xnw-lab hair
/// --chain` on a test frame sat 900 s after its 600 s limit, to the remote
/// call's limit; test `a_stuck_submission_ends_the_process`). So the
/// process ends `grace` from now with code 3, whatever it is doing then;
/// the submitter returns its error meanwhile and the step reports. Later
/// calls keep the first deadline.
pub fn abandon(label: &str, seconds: f64, grace: Duration) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    eprintln!(
        "xnw: gpu watchdog: one submission ({label}) has not reached its fence after {seconds} s and still runs on the adapter: nothing sent after it on its queue will finish; the process ends in {:.1} s (exit 3)",
        grace.as_secs_f64()
    );
    let label = label.to_string();
    ONCE.call_once(move || {
        let spawned = std::thread::Builder::new().name("xnw gpu watch end".into()).spawn({
            let label = label.clone();
            move || {
                std::thread::sleep(grace);
                give_up(&label, seconds);
            }
        });
        // No thread: end at once rather than hang.
        if spawned.is_err() {
            give_up(&label, seconds);
        }
    });
}

/// Whether the submission sent at `sent` has run past [`SOFT_LIMIT_S`]: its
/// submitter stops waiting and returns its error; the process then ends
/// ([`abandon`]). For fence waits that had no limit of their own.
pub fn overdue(label: &str, sent: Instant) -> bool {
    if sent.elapsed().as_secs_f64() <= SOFT_LIMIT_S {
        return false;
    }
    abandon(label, SOFT_LIMIT_S, GRACE);
    true
}

/// Ends the process now with code 3, without unloading anything (the
/// adapter's unload could wait for the submission a thread of ours is stuck
/// behind). stdout is flushed first (stderr and the log are not buffered).
pub fn give_up(label: &str, seconds: f64) -> ! {
    use std::io::Write;
    eprintln!("xnw: gpu watchdog: one submission ({label}) given up after {seconds} s still runs on the adapter: ending the process (exit 3)");
    let _ = std::io::stdout().flush();
    // SAFETY: our own process; nothing of ours runs after this.
    unsafe { xnw_core::win::TerminateProcess(xnw_core::win::GetCurrentProcess(), 3) };
    std::process::abort()
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

/// The submission just sent ([`begin`]) signals `value` on `fence`
/// and its submitter waits for it later (the world pass's `XNW_PASS_LAG`:
/// at the next frame's start, after the title's own frame, which at a
/// load can take over the budget alone: a test run exited at world entry). The
/// watchdog takes it as ended once the fence has reached `value`, so only
/// the card's own time counts against the budget.
pub fn fence(fence: &windows::Win32::Graphics::Direct3D12::ID3D12Fence, value: u64) {
    if WATCH.get().is_some()
        && let Ok(mut g) = FENCE.lock()
    {
        *g = Some((Fence(fence.clone()), value, SENT.load(Ordering::Acquire)));
    }
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

/// The live line (no frame's CPU time to split off): `gpu watch: <what>: <n> submissions: fence wait longest <ms> (<label>); timestamps longest <ms>; budget <ms>`.
pub fn summary(what: &str, subs: &[Sub]) -> String {
    let long = subs.iter().copied().max_by(|a, b| a.wait_ms.total_cmp(&b.wait_ms));
    let stamp = subs.iter().filter_map(|s| s.stamp_ms).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
    format!(
        "gpu watch: {what}: {} submissions: fence wait longest {:.2} ms{}; timestamps longest {}; budget {}",
        subs.len(),
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
    fn summary_names_the_longest() {
        let subs = [Sub { label: "pass", wait_ms: 3.0, stamp_ms: Some(2.5) }, Sub { label: "bridgeps chunk", wait_ms: 7.0, stamp_ms: Some(6.0) }];
        let l = summary("the last 60 frames", &subs);
        assert!(l.contains("2 submissions: fence wait longest 7.00 ms (bridgeps chunk); timestamps longest 6.00 ms"), "{l}");
    }

    /// A given-up submission ends the process with code 3 after the grace,
    /// while the thread that gave it up hangs (here: sleeps). The child is
    /// this test with `XNW_GPUWATCH_CHILD` set.
    #[test]
    fn abandon_ends_the_process() {
        if std::env::var_os("XNW_GPUWATCH_CHILD").is_some() {
            abandon("test", 0.5, Duration::from_millis(300));
            std::thread::sleep(Duration::from_secs(60));
            std::process::exit(0);
        }
        let t = Instant::now();
        let mut child = std::process::Command::new(std::env::current_exe().expect("exe"))
            .args(["gpuwatch::tests::abandon_ends_the_process", "--exact", "--test-threads=1"])
            .env("XNW_GPUWATCH_CHILD", "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("child");
        let code = loop {
            if let Some(st) = child.try_wait().expect("wait") {
                break st.code();
            }
            if t.elapsed() > Duration::from_secs(30) {
                let _ = child.kill();
                panic!("the child was still running after 30 s");
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(code, Some(3));
        assert!(t.elapsed() >= Duration::from_millis(300), "not before the grace: {:?}", t.elapsed());
    }

    /// A submission sent and waited for later ([`fence`]) does not end
    /// the process once its fence is reached (the child: budget 100 ms, a
    /// fence the CPU signals, 400 ms with no wait); one whose fence is not
    /// reached does (exit 3). The child is this test with `XNW_GPUWATCH_FENCE`
    /// set to the value it waits on (1: reached, 2: not); no device: skipped.
    #[test]
    fn a_reached_fence_ends_the_watch() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Direct3D12::{D3D12_FENCE_FLAG_NONE, D3D12CreateDevice, ID3D12Device, ID3D12Fence};
        if let Some(v) = std::env::var_os("XNW_GPUWATCH_FENCE") {
            let mut dev: Option<ID3D12Device> = None;
            // SAFETY: plain creation calls; the objects live to the exit.
            let f: ID3D12Fence = unsafe {
                if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    std::process::exit(0);
                }
                let Some(dev) = dev else { std::process::exit(0) };
                let f: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
                f.Signal(1).expect("signal");
                f
            };
            start(Duration::from_millis(100));
            let _t = begin("t");
            fence(&f, v.to_str().and_then(|x| x.parse().ok()).unwrap_or(2));
            std::thread::sleep(Duration::from_millis(400));
            std::process::exit(0);
        }
        let run = |v: &str| {
            let st = std::process::Command::new(std::env::current_exe().expect("exe"))
                .args(["gpuwatch::tests::a_reached_fence_ends_the_watch", "--exact", "--test-threads=1"])
                .env("XNW_GPUWATCH_FENCE", v)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .expect("child");
            st.code()
        };
        assert_eq!(run("1"), Some(0), "the fence was reached: no exit");
        assert_eq!(run("2"), Some(3), "the fence was not reached: the watchdog's exit");
    }

    #[test]
    fn overdue_is_false_within_the_limit() {
        assert!(!overdue("t", Instant::now()));
    }

    #[test]
    fn begin_end_logs_the_wait() {
        let t = begin("t");
        let w = end("t", t, None);
        assert!(w >= 0.0);
        assert!(take().iter().any(|s| s.label == "t"));
    }
}
