//! PROF1: the pass's record time by section (`XNW_RECORD_PROF=N`, default
//! off: one line every N frames, `on` = 60; `xnw burst` takes 1 for a line
//! a frame). Each section's mean over the window and the heaviest frame's
//! split. Off, a section costs one relaxed load (no clock read).
//!
//! Sections are marked on the render thread only; a [`Lap`] sums its own
//! and adds them to the frame's at its flush (a few mutex takes a frame).

use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

/// The sections, in record order. `rest` is the frame's record time no
/// section names (it should stay near 0).
pub const NAMES: [&str; N] = [
    "filter+sort",
    "geometry",
    "palettes+instances",
    "list start",
    "pre-skin",
    "mesh compute",
    "gpu cull",
    "textures",
    "sky+terrain",
    "draw resolve",
    "draw record",
    "bridgeps",
    "lit+water",
    "tail+close",
    // Inside `bridgeps` (bridged::Gpu::record).
    "b.setup",
    "b.inputs map",
    "b.row checks",
    "b.bind",
    "b.place",
    "b.row push",
    "b.staged",
    "b.uploads",
    "b.views",
    "b.list",
    "b.other",
    "imposters",
    // PERF3 `bridgepar`: the pass waiting for the rows' preparing thread (part of `bridgeps`).
    "b.wait",
    // PERF3: the pass's own share (start, texture snapshot); the preparing thread idle until the snapshot came.
    "b.start",
    "b.tex wait",
];
pub const N: usize = 29;
pub const FILTER: usize = 0;
pub const GEOMETRY: usize = 1;
pub const PALETTES: usize = 2;
pub const LIST_START: usize = 3;
pub const PRESKIN: usize = 4;
pub const MESH: usize = 5;
pub const CULL: usize = 6;
pub const TEXTURES: usize = 7;
pub const SKY: usize = 8;
pub const RESOLVE: usize = 9;
pub const RECORD: usize = 10;
pub const BRIDGEPS: usize = 11;
pub const LIT: usize = 12;
pub const TAIL: usize = 13;
pub const B_SETUP: usize = 14;
pub const B_ARCS: usize = 15;
pub const B_CHECKS: usize = 16;
pub const B_BIND: usize = 17;
pub const B_PLACE: usize = 18;
pub const B_PUSH: usize = 19;
pub const B_STAGED: usize = 20;
pub const B_UPLOADS: usize = 21;
pub const B_VIEWS: usize = 22;
pub const B_LIST: usize = 23;
pub const B_OTHER: usize = 24;
/// Top-level, listed last (added after the bridged parts).
pub const IMPOSTERS: usize = 25;
/// PERF3: inside `bridgeps`, listed after the top-level imposters.
pub const B_WAIT: usize = 26;
pub const B_START: usize = 27;
pub const B_TEXWAIT: usize = 28;
/// Top-level sections (they sum to the record time); the rest are parts of `bridgeps`.
const TOP: usize = 14;

/// Lines every N frames (0: off).
pub fn every() -> u64 {
    static E: OnceLock<u64> = OnceLock::new();
    *E.get_or_init(|| match std::env::var("XNW_RECORD_PROF").as_deref().map(str::trim) {
        Ok("on") => 60,
        Ok(v) => v.parse().unwrap_or(0),
        Err(_) => 0,
    })
}

pub fn on() -> bool {
    every() > 0
}

#[derive(Clone, Copy)]
struct Prof {
    /// This frame's milliseconds by section, the window's sums, the heaviest frame's.
    frame: [f64; N],
    sums: [f64; N],
    worst: [f64; N],
    /// The window's frames, record sums; the heaviest frame (number, record ms).
    frames: u64,
    record: f64,
    worst_at: (u64, f64),
    /// Events counted over the window (bridged: kept restarts, small buffers made).
    events: [u64; 2],
}

static P: Mutex<Prof> = Mutex::new(Prof { frame: [0.0; N], sums: [0.0; N], worst: [0.0; N], frames: 0, record: 0.0, worst_at: (0, 0.0), events: [0; 2] });

fn with<R>(f: impl FnOnce(&mut Prof) -> R) -> R {
    f(&mut P.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Adds `ms` to section `i` of this frame.
pub fn add(i: usize, ms: f64) {
    if on() {
        with(|p| p.frame[i] += ms);
    }
}

/// Section `i`'s milliseconds so far this frame.
pub fn get(i: usize) -> f64 {
    if on() { with(|p| p.frame[i]) } else { 0.0 }
}

/// Counts event `k` (0: kept restart, 1: small buffer made).
pub fn event(k: usize) {
    if on() {
        with(|p| p.events[k] += 1);
    }
}

/// A clock that charges each lap to a section, summed here and added at
/// [`Lap::flush`] or its drop (no clock read when off).
pub struct Lap(Option<(Instant, [f64; N])>);

impl Lap {
    pub fn new() -> Lap {
        Lap(on().then(|| (Instant::now(), [0.0; N])))
    }

    /// Charges the time since the last mark to `i`, less `minus` (ms another section took inside).
    pub fn mark_less(&mut self, i: usize, minus: f64) {
        if let Some((t, ms)) = &mut self.0 {
            let now = Instant::now();
            ms[i] += ((now - *t).as_secs_f64() * 1e3 - minus).max(0.0);
            *t = now;
        }
    }

    pub fn mark(&mut self, i: usize) {
        self.mark_less(i, 0.0);
    }

    /// The laps so far into this frame's sections.
    pub fn flush(&mut self) {
        if let Some((_, ms)) = &mut self.0 {
            let v = std::mem::replace(ms, [0.0; N]);
            with(|p| p.frame.iter_mut().zip(v).for_each(|(a, b)| *a += b));
        }
    }
}

impl Drop for Lap {
    fn drop(&mut self) {
        self.flush();
    }
}

impl Default for Lap {
    fn default() -> Self {
        Lap::new()
    }
}

/// The frame ended with `record` ms of record time: summed into the window;
/// every [`every`] frames the line (None between).
pub fn frame_end(frame: u64, record: f64) -> Option<String> {
    if !on() {
        return None;
    }
    with(|p| {
        let f = std::mem::replace(&mut p.frame, [0.0; N]);
        for (s, v) in p.sums.iter_mut().zip(f) {
            *s += v;
        }
        p.frames += 1;
        p.record += record;
        if record >= p.worst_at.1 {
            p.worst = f;
            p.worst_at = (frame, record);
        }
        if p.frames < every() {
            return None;
        }
        let n = p.frames as f64;
        let top = |v: &[f64; N]| v[..TOP].iter().sum::<f64>() + v[IMPOSTERS];
        let (named, wnamed) = (top(&p.sums), top(&p.worst));
        let part = |v: &[f64; N], k: f64| NAMES.iter().zip(v).filter(|x| *x.1 / k >= 0.05).map(|(name, ms)| format!("{name} {:.2}", ms / k)).collect::<Vec<_>>().join(", ");
        let line = format!(
            "world render: record parts (XNW_RECORD_PROF, {} frames to frame {frame}): mean record {:.2} ms: {}, rest {:.2}; heaviest frame {} record {:.2} ms: {}, rest {:.2}; bridged kept restarts {}, small buffers made {}",
            p.frames,
            p.record / n,
            part(&p.sums, n),
            (p.record - named) / n,
            p.worst_at.0,
            p.worst_at.1,
            part(&p.worst, 1.0),
            p.worst_at.1 - wnamed,
            p.events[0],
            p.events[1]
        );
        *p = Prof { frame: [0.0; N], sums: [0.0; N], worst: [0.0; N], frames: 0, record: 0.0, worst_at: (0, 0.0), events: [0; 2] };
        Some(line)
    })
}
