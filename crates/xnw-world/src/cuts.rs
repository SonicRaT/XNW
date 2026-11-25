//! Camera cuts without a visible break.
//!
//! Measured over two test runs: a cut brings 260-4,320 new geometries
//! (7-71 MiB). Before the budget the presenting thread sat in one upload of up to
//! 0.9 s; with the 4 ms time budget it shows 2-58 partial frames instead
//! (draws without geometry skipped, sky or clear colour in the holes).
//!
//! What this file decides, per frame, from what the pass already knows:
//! - **Coverage**: the share of the main view's triangles whose geometry is
//!   on the device (a missing geometry counts with its queued index count,
//!   else the frame's mean).
//! - **Cut**: the main camera moved more than [`CUT_METRES`] or turned more
//!   than [`CUT_DEGREES`] since the last frame, or coverage fell under
//!   [`CUT_COVERAGE`] (draw-set turnover with the camera still).
//! - **Hold**: after a picture was shown, a frame under [`HOLD_COVERAGE`] is
//!   not shown: the last complete picture stays, for at most [`HOLD_MS`]
//!   (then whatever is drawn is shown: never a freeze).
//! - **Recovery**: from the cut until coverage reaches
//!   [`COMPLETE_COVERAGE`]; while recovering, the upload time budget is
//!   multiplied by [`RECOVERY_BOOST`] (the frame is held anyway, and the
//!   biggest-on-screen geometries go first: geometry.rs plan order).
//! - One log line per cut when it completes (or gives up after
//!   [`RECOVERY_FRAMES`]), with our pass's time in it split by part.
//! - **Fast** (`cutfast=on`, off by default): while a camera cut
//!   recovers, the frame's geometry upload time is what drains the main
//!   view's queue in [`DRAIN_FRAMES`] frames (at most [`DRAIN_MAX_MS`]), and
//!   new texture reads stop after [`CUT_TEX_MS`] of the frame. A test run (card):
//!   cut at frame 770 took 6 frames / 392 ms with 1,949 geometries still
//!   queued after its first frame (12 ms of upload a frame) and 90 ms of that
//!   frame in record (texture reads in the frame).

use super::{Camera, camera_position};

pub const CUT_METRES: f32 = 10.0;
pub const CUT_DEGREES: f32 = 30.0;
/// A focal-length change (proj[5]) of more than this fraction is a cut.
pub const CUT_FOV: f32 = 0.08;
pub const CUT_COVERAGE: f32 = 0.8;
pub const HOLD_COVERAGE: f32 = 0.9;
pub const COMPLETE_COVERAGE: f32 = 0.98;
pub const HOLD_MS: f64 = 250.0;
pub const RECOVERY_BOOST: f64 = 3.0;
pub const RECOVERY_FRAMES: u64 = 300;
/// `cutfast=on`: the frames a camera cut's queue is drained in.
pub const DRAIN_FRAMES: u64 = 2;
/// `cutfast=on`: the most geometry upload time one cut frame takes.
pub const DRAIN_MAX_MS: f64 = 30.0;
/// `cutfast=on`: geometry bytes one cut frame may upload at most.
pub const DRAIN_MAX_BYTES: u64 = 96 << 20;
/// `cutfast=on`: new texture reads in a cut frame stop after this.
pub const CUT_TEX_MS: f64 = 10.0;

/// What the frame should do with its picture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Show {
    /// Show this frame's picture.
    Present,
    /// Keep showing the last one (coverage under the hold mark).
    Hold,
}

/// A cut in progress, for its log line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cut {
    pub frame: u64,
    pub start_ms: f64,
    pub held: u32,
    pub frames: u64,
    pub uploaded: u64,
    pub worst_ms: f64,
    /// Our pass's time over the cut's frames: geometry upload, texture
    /// table (reads, copies, views), the rest of the record, GPU.
    pub parts: [f64; 4],
    pub done: bool,
    /// A camera cut (jump or signal): the hold applies. A coverage drop with
    /// the camera still (mid-shot streaming: ~400 geometries in frames
    /// 1532-1548 of a test run) only raises the upload budget.
    pub camera: bool,
}

#[derive(Debug, Default)]
pub struct CutWatch {
    /// Hold partial pictures (screen mode; offline draws every frame).
    pub hold: bool,
    /// The upload time budget's factor while a cut recovers (1: none).
    pub boost: f64,
    /// `cutfast=on`: drain a camera cut's queue in [`DRAIN_FRAMES`].
    pub fast: bool,
    last_cam: Option<([f32; 3], [f32; 3], f32)>,
    shown_once: bool,
    hold_since: Option<f64>,
    pub cut: Option<Cut>,
    /// Totals: cuts, held frames, hold timeouts.
    pub cuts: u64,
    pub held: u64,
    pub timeouts: u64,
    /// The finished cut's line (taken by the caller to log).
    pub line: Option<String>,
}

fn forward(view: &super::Mat) -> [f32; 3] {
    [view[2], view[6], view[10]]
}

impl CutWatch {
    pub fn new(hold: bool) -> CutWatch {
        CutWatch { hold, boost: RECOVERY_BOOST, ..Default::default() }
    }

    /// Whether a cut is being recovered from (the upload budget is raised).
    pub fn recovering(&self) -> bool {
        self.cut.is_some()
    }

    /// `cutfast=on` and a camera cut recovering: the frames left to drain
    /// its queue in (at least 1).
    pub fn drain_frames(&self) -> Option<u64> {
        self.cut.filter(|c| self.fast && c.camera).map(|c| DRAIN_FRAMES.saturating_sub(c.frames).max(1))
    }

    /// The upload time for this frame from the base `ms`: x boost while a
    /// cut recovers; with `fast`, at least what drains `need_ms` of queued
    /// main-view uploads in the frames left (at most [`DRAIN_MAX_MS`]).
    pub fn upload_ms(&self, ms: f64, need_ms: f64) -> f64 {
        if !self.recovering() {
            return ms;
        }
        let boosted = ms * self.boost;
        match self.drain_frames() {
            Some(n) => boosted.max(need_ms / n as f64).min(DRAIN_MAX_MS.max(boosted)),
            None => boosted,
        }
    }

    /// The frame's texture read time cap (0: none): [`CUT_TEX_MS`] in a
    /// fast camera cut, else `base`.
    pub fn tex_ms(&self, base: f64) -> f64 {
        if self.drain_frames().is_some() {
            if base > 0.0 { base.min(CUT_TEX_MS) } else { CUT_TEX_MS }
        } else {
            base
        }
    }

    /// After the GPU: this frame's parts (upload, textures, rest of the
    /// record, GPU), added to the cut in progress.
    pub fn parts(&mut self, parts: [f64; 4]) {
        if let Some(c) = &mut self.cut {
            for (a, b) in c.parts.iter_mut().zip(parts) {
                *a += b;
            }
        }
    }

    /// Before the uploads: a cut starts here when the camera jumped.
    pub fn camera(&mut self, frame: u64, cam: &Camera, now_ms: f64) {
        let (p, f) = (camera_position(&cam.view), forward(&cam.view));
        let fy = cam.proj[5];
        if let Some((lp, lf, lfy)) = self.last_cam {
            let moved = p.iter().zip(&lp).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
            let dot = (f.iter().zip(&lf).map(|(a, b)| a * b).sum::<f32>() / (f.iter().map(|v| v * v).sum::<f32>().sqrt() * lf.iter().map(|v| v * v).sum::<f32>().sqrt()).max(1e-6)).clamp(-1.0, 1.0);
            // 1275 15.8 m, 122 deg, fov 37.8 -> 33.4; 1440 2.1 m, 92 deg, fov 33.4 -> 27.
            let zoom = lfy.abs() > 1e-6 && ((fy - lfy) / lfy).abs() > CUT_FOV;
            if moved > CUT_METRES || dot.acos().to_degrees() > CUT_DEGREES || zoom {
                self.start(frame, now_ms, true);
            }
        }
        self.last_cam = Some((p, f, fy));
    }

    /// A cut signalled by another source (the title's own camera-cut
    /// flag, when found): the same effect as a detected one. Upload sources
    /// plug in at `GeomPolicy::offer` (the draw hook today; residency events later).
    pub fn signal(&mut self, frame: u64, now_ms: f64) {
        self.start(frame, now_ms, true);
    }

    fn start(&mut self, frame: u64, now_ms: f64, camera: bool) {
        if camera && let Some(c) = &mut self.cut {
            c.camera = true;
        }
        if self.cut.is_none() {
            self.cuts += 1;
            self.cut = Some(Cut { frame, start_ms: now_ms, held: 0, frames: 0, uploaded: 0, worst_ms: 0.0, parts: [0.0; 4], done: false, camera });
        }
    }

    /// After the uploads: this frame's coverage, the bytes it uploaded and
    /// its time so far; what to show.
    pub fn frame(&mut self, frame: u64, coverage: f32, uploaded: u64, frame_ms: f64, now_ms: f64) -> Show {
        if coverage < CUT_COVERAGE && self.shown_once {
            self.start(frame, now_ms, false);
        }
        let mut show = Show::Present;
        if self.hold && self.shown_once && coverage < HOLD_COVERAGE && self.cut.is_some_and(|c| c.camera) {
            let since = *self.hold_since.get_or_insert(now_ms);
            if now_ms - since < HOLD_MS {
                show = Show::Hold;
            } else {
                self.timeouts += 1;
            }
        }
        if show == Show::Present {
            self.hold_since = None;
            self.shown_once = true;
        } else {
            self.held += 1;
        }
        if let Some(c) = &mut self.cut {
            c.frames += 1;
            c.uploaded += uploaded;
            c.held += u32::from(show == Show::Hold);
            c.worst_ms = c.worst_ms.max(frame_ms);
            c.done = coverage >= COMPLETE_COVERAGE;
        }
        show
    }

    /// After the frame (its whole time known): a finished cut's line.
    pub fn end(&mut self, frame_ms: f64, now_ms: f64) {
        let Some(c) = &mut self.cut else { return };
        c.worst_ms = c.worst_ms.max(frame_ms);
        if !c.done && c.frames < RECOVERY_FRAMES {
            return;
        }
        self.line = Some(format!(
            "world render: {} at frame {}: complete after {} frames / {:.0} ms{}, held {}, uploaded {:.1} MiB, worst frame {:.1} ms (cuts {}, held {}, hold timeouts {}); ours: upload {:.0} ms, textures {:.0} ms, record {:.0} ms, GPU {:.0} ms, outside {:.0} ms",
            if c.camera { "cut" } else { "streaming" },
            c.frame,
            c.frames,
            now_ms - c.start_ms,
            if c.done { "" } else { " (gave up: not complete)" },
            c.held,
            c.uploaded as f64 / f64::from(1 << 20),
            c.worst_ms,
            self.cuts,
            self.held,
            self.timeouts,
            c.parts[0],
            c.parts[1],
            c.parts[2],
            c.parts[3],
            (now_ms - c.start_ms - c.parts.iter().sum::<f64>()).max(0.0)
        ));
        self.cut = None;
    }
}

/// Milliseconds since the first call (the cut clock).
pub fn now_ms() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1e3
}

/// Coverage from (index count, on the device) per main-view draw; a missing
/// geometry whose count is unknown (0) counts with the frame's mean.
pub fn coverage(draws: &[(u32, bool)]) -> f32 {
    let known: Vec<u32> = draws.iter().filter(|d| d.0 > 0).map(|d| d.0).collect();
    let mean = if known.is_empty() { 1.0 } else { known.iter().map(|&n| f64::from(n)).sum::<f64>() / known.len() as f64 };
    let (mut have, mut all) = (0f64, 0f64);
    for &(n, resident) in draws {
        let w = if n > 0 { f64::from(n) } else { mean };
        all += w;
        if resident {
            have += w;
        }
    }
    if all == 0.0 { 1.0 } else { (have / all) as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(x: f32, yaw: f32) -> Camera {
        let (s, c) = yaw.to_radians().sin_cos();
        // Rotation about z, then the translation -p R.
        let mut v = [0f32; 16];
        (v[0], v[1], v[4], v[5], v[10], v[15]) = (c, -s, s, c, 1.0, 1.0);
        let p = [x, 0.0, 0.0];
        for j in 0..3 {
            v[12 + j] = -(p[0] * v[j] + p[1] * v[4 + j] + p[2] * v[8 + j]);
        }
        Camera { viewport: 1, view: v, proj: [0.0; 16] }
    }

    #[test]
    fn coverage_weights_by_triangles() {
        assert_eq!(coverage(&[]), 1.0);
        assert!((coverage(&[(300, true), (100, false)]) - 0.75).abs() < 1e-6);
        // Unknown size: the mean of the known (200).
        assert!((coverage(&[(300, true), (100, true), (0, false)]) - 400.0 / 600.0).abs() < 1e-6);
    }

    #[test]
    fn camera_jump_starts_a_cut_held_until_complete() {
        let mut w = CutWatch::new(true);
        w.camera(1, &cam(0.0, 0.0), 0.0);
        assert_eq!(w.frame(1, 0.2, 0, 5.0, 0.0), Show::Present, "nothing shown yet: show what there is");
        w.camera(2, &cam(0.5, 2.0), 33.0);
        assert_eq!(w.frame(2, 1.0, 0, 5.0, 33.0), Show::Present);
        assert!(!w.recovering(), "a small move is not a cut");
        w.camera(3, &cam(500.0, 0.0), 66.0);
        assert!(w.recovering());
        assert_eq!(w.frame(3, 0.3, 10 << 20, 20.0, 66.0), Show::Hold);
        w.camera(4, &cam(500.0, 0.0), 99.0);
        w.end(20.0, 66.0);
        assert_eq!(w.frame(4, 0.95, 10 << 20, 20.0, 99.0), Show::Present, "over the hold mark");
        w.end(20.0, 99.0);
        assert!(w.recovering(), "not complete yet");
        assert_eq!(w.frame(5, 0.99, 0, 8.0, 132.0), Show::Present);
        w.end(8.0, 132.0);
        let l = w.line.take().unwrap();
        assert!(l.starts_with("world render: cut at frame 3: complete after 3 frames / 66 ms, held 1, uploaded 20.0 MiB"), "{l}");
        assert_eq!((w.cuts, w.held), (1, 1));
    }

    #[test]
    fn fast_cut_drains_in_two_frames_and_caps_texture_reads() {
        let mut w = CutWatch::new(true);
        w.camera(1, &cam(0.0, 0.0), 0.0);
        w.frame(1, 1.0, 0, 5.0, 0.0);
        assert_eq!((w.upload_ms(4.0, 50.0), w.tex_ms(0.0)), (4.0, 0.0), "no cut: the base");
        w.camera(2, &cam(500.0, 0.0), 33.0);
        assert_eq!(w.upload_ms(4.0, 50.0), 12.0, "not fast: x3 only");
        assert_eq!(w.drain_frames(), None);
        w.fast = true;
        assert_eq!(w.drain_frames(), Some(2));
        assert_eq!(w.upload_ms(4.0, 50.0), 25.0, "half the queue in the first frame");
        assert_eq!(w.upload_ms(4.0, 200.0), DRAIN_MAX_MS, "capped");
        assert_eq!(w.upload_ms(4.0, 10.0), 12.0, "never under the boost");
        assert_eq!((w.tex_ms(0.0), w.tex_ms(4.0)), (CUT_TEX_MS, 4.0));
        w.frame(2, 0.5, 0, 30.0, 33.0);
        assert_eq!(w.drain_frames(), Some(1), "the rest in the second");
        w.parts([20.0, 10.0, 3.0, 2.0]);
        w.end(30.0, 66.0);
        w.camera(3, &cam(500.0, 0.0), 66.0);
        w.frame(3, 1.0, 0, 30.0, 66.0);
        w.parts([20.0, 10.0, 3.0, 2.0]);
        w.end(30.0, 99.0);
        let l = w.line.take().unwrap();
        assert!(l.ends_with("ours: upload 40 ms, textures 20 ms, record 6 ms, GPU 4 ms, outside 0 ms"), "{l}");
        assert_eq!(w.drain_frames(), None, "over");
    }

    #[test]
    fn a_hold_never_outlasts_its_timeout_and_turnover_is_a_cut() {
        let mut w = CutWatch::new(true);
        w.frame(1, 1.0, 0, 5.0, 0.0);
        // Camera still, the draw set replaced: a cut by coverage.
        let mut t = 0.0;
        let mut shown = Vec::new();
        for f in 2..20 {
            t += 33.0;
            shown.push(w.frame(f, 0.1, 0, 5.0, t));
        }
        assert_eq!(w.cuts, 1);
        assert!(shown.iter().all(|s| *s == Show::Present), "mid-shot streaming: no hold, only the boost: {shown:?}");
        assert!(w.recovering());
        // A camera cut while partial: held 250 ms, then shown.
        let mut w = CutWatch::new(true);
        w.frame(1, 1.0, 0, 5.0, 0.0);
        w.signal(2, 33.0);
        let mut shown = Vec::new();
        let mut t = 0.0;
        for f in 2..20 {
            t += 33.0;
            shown.push(w.frame(f, 0.1, 0, 5.0, t));
        }
        assert!(shown[..7].iter().all(|s| *s == Show::Hold) && shown[8] == Show::Present, "held 250 ms, then shown: {shown:?}");
        let mut off = CutWatch::new(false);
        off.frame(1, 1.0, 0, 5.0, 0.0);
        assert_eq!(off.frame(2, 0.1, 0, 5.0, 33.0), Show::Present, "hold off: offline draws every frame");
    }
}
