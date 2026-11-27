//! Live switches (SW1): render switches flipped inside one running title.
//!
//! One static table of name -> value. Each reader asks [`value`] (one
//! relaxed atomic load); the first ask stores the run's start value, which
//! the reader computes as before (its environment variable or
//! `XNW_WORLD_RENDER` item), so a run that never flips anything behaves as
//! before and the offline tools (replay, lab, burst) see their own
//! environment. Three drivers write the table: the control file (`xnw`
//! `live.rs`, `out\live.txt` polled once a second), the window keys F1-F12
//! (`xnw-hle` `window.rs`) and `XNW_AB_LIVE` (`ab.rs`, at the A/B frame).
//! Every change logs `live: <name> <old> -> <new> at frame N` (N: the
//! presented frame) and restarts the [`luma`] flicker window.
//!
//! A switch is here only when a flip between two frames is safe: no
//! object of the pass is made from it at start, or the reader takes it
//! once per frame (the comment of each entry says which). `cap`: the start
//! value is the most it can be (a switch whose objects are made at start
//! can be turned off and on again, never on when the run started without
//! it; a mask, only bits it started with). Restart only (left out): see
//! "Live switches".

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

/// A value not given yet (its reader has not run).
const UNSET: u32 = u32::MAX;

pub struct Switch {
    pub name: &'static str,
    v: AtomicU32,
    start: AtomicU32,
    /// The largest value (a level), or the bits allowed (a mask).
    pub max: u32,
    /// A bit mask: a key cycles through its start value, each start bit alone, none.
    pub mask: bool,
    /// Never above (outside) the start value.
    pub cap: bool,
    /// What the values mean (the key map and the file's help).
    pub help: &'static str,
}

const fn sw(name: &'static str, max: u32, mask: bool, cap: bool, help: &'static str) -> Switch {
    Switch { name, v: AtomicU32::new(UNSET), start: AtomicU32::new(UNSET), max, mask, cap, help }
}

pub const BRIDGEPS: usize = 0;
pub const BRIDGEFAM: usize = 1;
pub const OURNORMALS: usize = 2;
pub const OURLIGHT: usize = 3;
pub const DECAL: usize = 4;
pub const LAG: usize = 5;
pub const HOLD: usize = 6;
pub const UI: usize = 7;
pub const HAIR: usize = 8;
pub const TEX: usize = 9;
pub const BRIDGECAM: usize = 10;
pub const BRIDGEBIAS: usize = 11;
pub const BRIDGEULPS: usize = 12;
pub const BRIDGESLOPE: usize = 13;
pub const HEAD: usize = 14;
pub const SUNSHADOW: usize = 15;
pub const INSTREC: usize = 16;
pub const LIGHTS: usize = 17;
pub const POST: usize = 18;
pub const EXPOSURE: usize = 19;
pub const HAIRRIBBON: usize = 20;
pub const BIGCOPY: usize = 21;
pub const LIGHTKEEP: usize = 22;

/// The table; F1.. are its entries in this order, then [`KEY_MARK`] and [`KEY_TABLE`].
pub static TABLE: [Switch; 23] = [
    // The list sink (title thread) asks per draw whether to give the draw's
    // inputs to its row; a row not given draws our stand-in, as a row whose
    // pipelines are not made yet. Pipelines and the tap stay as made.
    sw("bridgeps", 1, false, true, "1 the bridged rows as started, 0 every row our stand-in"),
    // As bridgeps, per program of the bridgeps list (bit k: the k-th hash, printed at start).
    sw("bridgefam", 0x7FFF_FFFF, true, true, "bit k: rows of the k-th bridgeps program bridged (hex mask)"),
    // Taken once a frame by the lighting pass: 0 clears the title normal
    // target before the pass reads it, so every pixel takes our normal and
    // roughness (the pass's `own` test fails on a cleared word).
    sw("ournormals", 1, false, true, "1 title normals (as started), 0 ours"),
    // Taken once a frame when the walk starts: 0 marks the frame lit from
    // the start, so no row leaves albedo, the pass is not recorded and the
    // bridged rows draw in the tail; the pass's objects stay made. on/metal: restart only.
    sw("ourlight", 1, false, true, "1 our lighting pass as made (on or metal), 0 off (rows light themselves)"),
    // A bridged decal row's pipeline is made with or without its other
    // targets; the cache keys on the value, so a flip makes (once) or reuses the other pipeline.
    sw("decal", 1, false, false, "1 decal rows blend every target (SHD1), 0 target 0 alone (XNW_DECAL_TARGETS=off)"),
    // Taken once a frame by the pass (the frame start always waits for a
    // frame in flight); the second picture is made at start, hence cap.
    sw("lag", 1, false, true, "1 frame-lag retirement (XNW_PASS_LAG=on), 0 wait each frame"),
    // Taken once a frame by the bridged rows; 0 lets every held binding go,
    // so turning it on again starts from fresh bindings.
    sw("hold", 2, false, false, "0 off, 1 held bindings (XNW_BRIDGE_HOLD=on), 2 check"),
    // Taken once a list: 0 leaves the UI rows out of the list (the pack
    // writer still takes them: its hooks are made at start, hence cap).
    sw("ui", 1, false, true, "1 our UI layer (ui=on), 0 none"),
    // As ui, the strand pixels.
    sw("hair", 1, false, true, "1 our hair strands (hair=on), 0 none"),
    // Asked per draw at the hook: 0 reads no texture (the draw is drawn
    // untextured, as a texture not resident); the table stays made.
    sw("tex", 1, false, true, "1 textures read at the draw hooks, 0 none (tex=off)"),
    // Taken once a frame by the bridged rows (FLK1): what a row does whose
    // view constants are not the list camera's; no object is made from it.
    // These two have no key (F11 and F12 are the mark and the table).
    sw("bridgecam", 2, false, false, "bridged rows whose view constants are another camera's: 0 drawn as they are, 1 left out (our stand-in shows), 2 given the list camera (XNW_BRIDGE_CAM)"),
    // Taken once a frame for the bridged rows' viewport (FLK1): their depth
    // is b + z * (1 - b), b = n * 1e-7, towards the viewer (reverse Z); the
    // pipelines are not remade.
    sw("bridgebias", 100_000, false, false, "n: bridged rows' depth moved n * 1e-7 towards the viewer before the test against ours (0 none; XNW_BRIDGE_BIAS)"),
    // The bridged rows' pixel pipelines carry these two (FLK1); the
    // pipeline map keys on them, so a flip makes (once) or reuses the
    // pipelines of that setting, as decal does.
    sw("bridgeulps", 1_000_000, false, false, "n: bridged rows' depth bias in units of the depth's last place at the primitive's nearest depth (a relative bias, n * 1.2e-7; XNW_BRIDGE_ULPS)"),
    sw("bridgeslope", 64, false, false, "n: bridged rows' slope-scaled depth bias n / 4 (the depth change over n / 4 pixels; XNW_BRIDGE_SLOPE)"),
    // Taken once a list (FACE1): 0 runs no head graph, the faces stay
    // neutral (the pack writer still records the head dispatches: its mode
    // is set at start, hence cap).
    sw("head", 1, false, true, "1 the title's head graph fills the faces (head=on), 0 neutral faces"),
    // Taken once a frame by the pass (SHADOW1): 0 draws no cascade and
    // clears the screen term to 1. The atlas and the term are made at
    // start: a run started without `sunshadow=` has none (no key: past F12).
    sw("sunshadow", 3, false, false, "sun shadows in our lighting pass: 0 off, 1 the title's cascades while it draws them else ours, 2 the title's only, 3 ours only (sunshadow=on|title|own at start)"),
    // Asked per sink at the list sink: 0 reads no instance records, the
    // rows keep the hook's world; the tap stays installed, hence cap. No key.
    sw("instrec", 1, false, true, "1 `*_inst` rows placed by their instance records (default; instrec=off), 0 by the hook's world"),
    // Taken once a frame by the pass (LIGHT2): 0 uploads no light (the
    // pass's loop runs zero times). The pass's lights variant is made at
    // start: a run started without `lights=on` has none; live the array is
    // read only when the run started with it. No key.
    sw("lights", 1, false, false, "the title's local lights in our lighting pass: 1 on, 0 off (lights=on at start)"),
    // Taken once a frame by the pass (LIGHT2 after POST1): 0 the lighting
    // pass tone-maps itself into the 8-bit target, as a run without `post=`.
    // The scene-linear target and the composite are made at start. No key.
    sw("post", 1, false, false, "1 scene-linear lighting and our composite (post=on|key at start), 0 the lighting pass's own tone map"),
    // Taken once a frame with post: the composite's exposure E (a linear multiplier). No key.
    sw("exposure", 100_000, false, false, "n: the composite's exposure E = n / 100 (0: the run's own, the timecycle's EV with post=on, our key with post=key)"),
    // Taken once a list (HAIR10): 0 builds no ribbons (the pack writer still
    // records the strand draws: its mode is set at start, hence cap). No key.
    sw("hairribbon", 1, false, true, "1 our guide ribbons from the title's strand records (hairribbon=on), 0 none"),
    // Asked per big frame-table view at the list sink (PERF5): 1 the sink
    // copies its head and the render thread the rest at the drain cut, 0
    // the sink copies it all (into a pooled buffer). No key.
    sw("bigcopy", 1, false, false, "1 big frame-table views copied at the drain cut (bigcopy=drain), 0 by the sinks (worker, default)"),
    // Taken once a frame by the pass's light selection (FLK2): 0 the plain
    // score each frame (lights pop in and out at the cap). No key.
    sw("lightkeep", 1, false, false, "1 kept lights hold their place at the cap (x1.5 score) and cap entries fade in over 6 frames (lightkeep=on, default), 0 the plain score"),
];

/// A mark in the log; the flicker window restarts.
pub const KEY_MARK: usize = 10;
/// The table in the log.
pub const KEY_TABLE: usize = 11;

/// Changes so far (the flicker window restarts when it moves).
static GEN: AtomicU64 = AtomicU64::new(0);

/// The value of switch `i`; its first ask stores `start()` as the run's start.
#[inline]
pub fn value(i: usize, start: impl FnOnce() -> u32) -> u32 {
    let v = TABLE[i].v.load(Ordering::Relaxed);
    if v != UNSET { v } else { init(i, start()) }
}

/// [`value`] != 0.
#[inline]
pub fn on(i: usize, start: impl FnOnce() -> u32) -> bool {
    value(i, start) != 0
}

#[cold]
fn init(i: usize, s: u32) -> u32 {
    let s = s.min(TABLE[i].max);
    let _ = TABLE[i].start.compare_exchange(UNSET, s, Ordering::Relaxed, Ordering::Relaxed);
    match TABLE[i].v.compare_exchange(UNSET, s, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => s,
        Err(now) => now,
    }
}

/// The index of switch `name`.
pub fn find(name: &str) -> Option<usize> {
    TABLE.iter().position(|s| s.name == name)
}

/// A value as the file, the keys and `XNW_AB_LIVE` write it: on/off,
/// title/ours, check, start, a decimal or 0x hex number.
fn parse_value(s: &str, start: u32) -> Option<u32> {
    match s {
        "on" | "title" => Some(1),
        "off" | "ours" => Some(0),
        "check" => Some(2),
        "start" => Some(start),
        h if h.starts_with("0x") => u32::from_str_radix(&h[2..], 16).ok(),
        d => d.parse().ok(),
    }
}

fn show(s: &Switch, v: u32) -> String {
    if v == UNSET {
        "unset".into()
    } else if s.mask {
        format!("{v:#x}")
    } else {
        v.to_string()
    }
}

/// Whether `v` is allowed for switch `s` started at `start` (Err: why not).
fn allowed(s: &Switch, v: u32, start: u32) -> Result<(), String> {
    if (s.mask && v & !s.max != 0) || (!s.mask && v > s.max) {
        return Err(format!("{v} is out of range ({})", s.help));
    }
    if start == UNSET {
        return Err("not read by this run yet (its part is not in this run)".into());
    }
    if s.cap && ((s.mask && v & !start != 0) || (!s.mask && v > start)) {
        return Err(format!("started at {}: more is restart only", show(s, start)));
    }
    Ok(())
}

/// Sets switch `name` to `text` at presented frame `frame`: the old and
/// new values when it changed (logged), None when it held that value.
pub fn set(name: &str, text: &str, frame: u64) -> Result<Option<(u32, u32)>, String> {
    let i = find(name).ok_or_else(|| format!("no live switch {name} ({})", names()))?;
    let s = &TABLE[i];
    let start = s.start.load(Ordering::Relaxed);
    let v = parse_value(text, start).ok_or_else(|| format!("{name}={text}: not a value ({})", s.help))?;
    allowed(s, v, start).map_err(|e| format!("{name}={text}: {e}"))?;
    let old = s.v.swap(v, Ordering::Relaxed);
    if old == v {
        return Ok(None);
    }
    GEN.fetch_add(1, Ordering::Relaxed);
    crate::info!("live: {name} {} -> {} at frame {frame}", show(s, old), show(s, v));
    Ok(Some((old, v)))
}

/// Every switch name, comma-separated.
pub fn names() -> String {
    TABLE.iter().map(|s| s.name).collect::<Vec<_>>().join(",")
}

/// The next value of a key press on switch `s` at `v` (started at `start`): a
/// level toggles 0 <-> start (1 when it started at 0 and has no cap); a mask
/// cycles start, each start bit alone, 0. None: nothing to toggle to.
fn next_on_key(s: &Switch, v: u32, start: u32) -> Option<u32> {
    if s.mask {
        let mut seq = vec![start];
        seq.extend((0..31).map(|k| 1u32 << k).filter(|b| start & b != 0 && *b != start));
        seq.push(0);
        let at = seq.iter().position(|&x| x == v);
        return Some(at.map_or(start, |k| seq[(k + 1) % seq.len()])).filter(|_| start != 0);
    }
    if v != 0 {
        Some(0)
    } else if start != 0 {
        Some(start)
    } else if !s.cap {
        Some(1)
    } else {
        None
    }
}

/// Window key F(k+1) at presented frame `frame` (the title's thread: no
/// environment lookups here).
pub fn key(k: usize, frame: u64) {
    match k {
        KEY_MARK => {
            GEN.fetch_add(1, Ordering::Relaxed);
            crate::info!("live: mark (F11) at frame {frame}");
        }
        KEY_TABLE => crate::info!("live: table at frame {frame}: {}", table_line()),
        k if k < TABLE.len() => {
            let s = &TABLE[k];
            let (v, start) = (s.v.load(Ordering::Relaxed), s.start.load(Ordering::Relaxed));
            let next = if v == UNSET { None } else { next_on_key(s, v, start) };
            match next {
                Some(n) => {
                    if let Err(e) = set(s.name, &n.to_string(), frame) {
                        crate::warn!("live: F{} {}: {e}", k + 1, s.name);
                    }
                }
                None => crate::warn!("live: F{} {}: started at {}: restart only", k + 1, s.name, show(s, start)),
            }
        }
        _ => {}
    }
}

/// `name=value` of every switch (unset: its reader has not run).
pub fn table_line() -> String {
    TABLE.iter().map(|s| format!("{}={}", s.name, show(s, s.v.load(Ordering::Relaxed)))).collect::<Vec<_>>().join(" ")
}

/// The key map, one line: `F1 bridgeps=1 (help) | ...`.
pub fn key_map() -> String {
    let mut parts: Vec<String> = TABLE.iter().enumerate().take(KEY_MARK).map(|(k, s)| format!("F{} {}={}{}", k + 1, s.name, show(s, s.v.load(Ordering::Relaxed)), if s.cap { "" } else { " (free)" })).collect();
    parts.push("F11 mark".into());
    parts.push("F12 table".into());
    parts.join(" | ")
}

/// `name=value` items of `text`: lines and commas; `#` starts a comment.
pub fn items(text: &str) -> Vec<(String, String)> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or(""))
        .flat_map(|l| l.split(','))
        .filter_map(|p| {
            let (n, v) = p.split_once('=')?;
            let (n, v) = (n.trim().to_ascii_lowercase(), v.trim().to_ascii_lowercase());
            (!n.is_empty() && !v.is_empty()).then_some((n, v))
        })
        .collect()
}

/// The control file's last items: a poll applies only what changed in
/// the file since the last poll (a key press is not undone by the file).
#[derive(Default)]
pub struct Poll {
    seen: HashMap<String, String>,
}

impl Poll {
    /// The items of `text` that differ from the last call's (the first
    /// call's: every item), last one per name.
    pub fn changes(&mut self, text: &str) -> Vec<(String, String)> {
        let mut now: HashMap<String, String> = HashMap::new();
        let mut order = Vec::new();
        for (n, v) in items(text) {
            if !now.contains_key(&n) {
                order.push(n.clone());
            }
            now.insert(n, v);
        }
        let out = order.into_iter().filter(|n| self.seen.get(n) != now.get(n)).map(|n| (n.clone(), now[&n].clone())).collect();
        self.seen = now;
        out
    }
}

// ---- The flicker counter ----

/// Frames in one flicker window.
pub const FLICKER_FRAMES: u32 = 600;
/// A zigzag: the frame's mean luma differs from both neighbours, the same way, by more than this.
pub const ZIG: f64 = 0.5;
/// A big one: by more than this.
pub const BIG: f64 = 3.0;

/// Big zigzags named per window (FLK2: a `flicker frame` line each).
pub const NAMED: u32 = 24;

/// FLK2: what a presented picture was made of. The world pass notes it when
/// a present takes the picture ([`note_shown`]); the presenter notes its UI
/// plane ([`note_ui`]); the flicker counter names a big zigzag's three
/// frames with them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Shown {
    /// The world list drawn into the picture (0: none noted).
    pub world: u64,
    /// The picture's texture (low 16 bits of its address).
    pub pic: u16,
    /// Our rows and the bridged rows drawn.
    pub rows: u32,
    pub bridged: u32,
    /// Lights kept, cap entries faded in (entered), kept lights dropped by
    /// the cap (left), and the kept set's energy at the camera.
    pub lights: u32,
    pub entered: u32,
    pub left: u32,
    pub energy: f32,
    /// The exposure E (0: the pass's own tone map).
    pub e: f32,
    /// The UI plane composed over it (low 16 bits of its address; 0 none).
    pub ui: u16,
    /// FLK3: the present's movie chain (`ui=keep`): bits 1 the Bink draw
    /// ran, 2 the fill, 4 the blit onto the UI plane sampled the movie
    /// target, 8 the target was cleared; and the plane the Bink draw sampled
    /// (low 16 bits; its two plane sets alternate per movie frame).
    pub movie: u8,
    pub plane: u16,
}

impl Shown {
    fn part(&self, again: bool) -> String {
        format!(
            "w{}{} pic {:04x} rows {}+{} lights {} (+{} -{}) energy {:.3} E {:.3} ui {:04x} movie {:x} plane {:04x}",
            self.world,
            if again { " again" } else { "" },
            self.pic,
            self.rows,
            self.bridged,
            self.lights,
            self.entered,
            self.left,
            self.energy,
            self.e,
            self.ui,
            self.movie,
            self.plane
        )
    }
}

static SHOWN: Mutex<Shown> = Mutex::new(Shown { world: 0, pic: 0, rows: 0, bridged: 0, lights: 0, entered: 0, left: 0, energy: 0.0, e: 0.0, ui: 0, movie: 0, plane: 0 });
static UI_PLANE: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(0);
/// FLK3: the present's movie bits and Bink plane ([`note_movie`]).
static MOVIE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// FLK2: the picture a present takes (the world pass, at the take).
pub fn note_shown(s: Shown) {
    *SHOWN.lock().unwrap_or_else(|e| e.into_inner()) = s;
}

/// FLK2: the UI plane a present composes (0: none).
pub fn note_ui(address: u64) {
    UI_PLANE.store(address as u16, Ordering::Relaxed);
}

/// FLK3: the movie chain of the frame being presented (bits, Bink plane).
pub fn note_movie(bits: u8, plane: u16) {
    MOVIE.store(u32::from(bits) << 16 | u32::from(plane), Ordering::Relaxed);
}

/// The counter's state: the last two means (and their frames' notes),
/// frames counted, zigzags, big ones, big ones named, the change count it
/// started at.
#[derive(Default)]
pub struct Flicker {
    last: Vec<f64>,
    notes: Vec<(u64, Shown)>,
    n: u32,
    zig: u32,
    big: u32,
    named: u32,
    epoch: u64,
}

impl Flicker {
    fn line(&self, tag: &str, why: &str) -> String {
        format!("flicker{tag}: {} of {} frames zigzag (>{ZIG}), {} big (>{BIG}){why}", self.zig, self.n, self.big)
    }

    /// Starts again (no frame carried over).
    pub fn restart(&mut self, epoch: u64) {
        *self = Flicker { epoch, ..Default::default() };
    }

    /// One frame's mean luma: the window's line when it is full.
    pub fn push(&mut self, mean: f64, tag: &str) -> Option<String> {
        self.push_noted(mean, 0, Shown::default(), tag).0
    }

    /// [`Flicker::push`] with the present's frame and its picture's note:
    /// (the window's line when full, a `flicker frame` line naming a big
    /// zigzag's three frames, the first [`NAMED`] of a window).
    pub fn push_noted(&mut self, mean: f64, frame: u64, note: Shown, tag: &str) -> (Option<String>, Option<String>) {
        self.last.push(mean);
        self.notes.push((frame, note));
        if self.last.len() < 3 {
            return (None, None);
        }
        let (a, b, c) = (self.last[0], self.last[1], self.last[2]);
        let notes = [self.notes[0], self.notes[1], self.notes[2]];
        self.last.remove(0);
        self.notes.remove(0);
        let (d1, d2) = (b - a, b - c);
        let same = d1 * d2 > 0.0;
        let big = same && d1.abs() > BIG && d2.abs() > BIG;
        self.n += 1;
        self.zig += u32::from(same && d1.abs() > ZIG && d2.abs() > ZIG);
        self.big += u32::from(big);
        let named = (big && self.named < NAMED).then(|| {
            self.named += 1;
            let again = |k: usize| k > 0 && notes[k].1.world != 0 && notes[k].1.world == notes[k - 1].1.world;
            format!("flicker{tag} frame {}: luma {a:.2} {b:.2} {c:.2}; shown {} | {} | {}", notes[1].0, notes[0].1.part(false), notes[1].1.part(again(1)), notes[2].1.part(again(2)))
        });
        if self.n < FLICKER_FRAMES {
            return (None, named);
        }
        let line = self.line(tag, "");
        (self.n, self.zig, self.big, self.named) = (0, 0, 0, 0);
        (Some(line), named)
    }
}

static FLICKER: Mutex<Option<Flicker>> = Mutex::new(None);
/// The A/B phase tag of the flicker lines (0 none, 1 A, 2 B).
static PHASE: AtomicU8 = AtomicU8::new(0);

/// `XNW_AB`: the flicker lines carry ` [A]` or ` [B]` (1, 2; 0 none).
pub fn set_phase(p: u8) {
    PHASE.store(p.min(2), Ordering::Relaxed);
}

/// A presented frame's mean luma (`XNW_FRAME_LUMA=1`'s readback): every
/// [`FLICKER_FRAMES`] counted frames one line; a live change closes the
/// window early (its line says so) and starts a new one.
/// `frame`: the present's; the picture's note ([`note_shown`], [`note_ui`]) goes with it.
pub fn luma(frame: u64, mean: f64) {
    let tag = ["", " [A]", " [B]"][usize::from(PHASE.load(Ordering::Relaxed).min(2))];
    let epoch = GEN.load(Ordering::Relaxed);
    let mut g = FLICKER.lock().unwrap_or_else(|e| e.into_inner());
    let f = g.get_or_insert_with(|| Flicker { epoch, ..Default::default() });
    if f.epoch != epoch {
        if f.n > 0 {
            crate::info!("{}", f.line(tag, " (to the live change)"));
        }
        f.restart(epoch);
    }
    let movie = MOVIE.load(Ordering::Relaxed);
    let note = Shown { ui: UI_PLANE.load(Ordering::Relaxed), movie: (movie >> 16) as u8, plane: movie as u16, ..*SHOWN.lock().unwrap_or_else(|e| e.into_inner()) };
    let (line, named) = f.push_noted(mean, frame, note, tag);
    if let Some(n) = named {
        crate::info!("{n}");
    }
    if let Some(line) = line {
        crate::info!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_parse_and_caps_hold() {
        assert_eq!((parse_value("on", 0), parse_value("off", 1), parse_value("title", 0), parse_value("check", 0)), (Some(1), Some(0), Some(1), Some(2)));
        assert_eq!((parse_value("0x3", 0), parse_value("5", 0), parse_value("start", 7), parse_value("x", 0)), (Some(3), Some(5), Some(7), None));
        let lag = &TABLE[LAG];
        assert!(allowed(lag, 0, 1).is_ok() && allowed(lag, 1, 1).is_ok());
        assert!(allowed(lag, 1, 0).is_err(), "a capped switch started off stays off");
        assert!(allowed(lag, 2, 1).is_err(), "out of range");
        assert!(allowed(&TABLE[HOLD], 2, 0).is_ok(), "hold is free");
        assert!(allowed(&TABLE[BRIDGEFAM], 0b101, 0b111).is_ok() && allowed(&TABLE[BRIDGEFAM], 0b1000, 0b111).is_err());
        assert!(allowed(lag, 0, UNSET).is_err(), "capped and not read yet");
        assert_eq!(TABLE.iter().map(|s| s.name).collect::<std::collections::HashSet<_>>().len(), TABLE.len(), "names are unique");
    }

    #[test]
    fn the_table_starts_from_its_reader_and_logs_changes() {
        // `decal` and `hold` are used by no other test.
        assert_eq!(value(DECAL, || 1), 1);
        assert_eq!(value(DECAL, || 0), 1, "the first ask gives the start");
        assert_eq!(set("decal", "off", 7), Ok(Some((1, 0))));
        assert_eq!(set("decal", "0", 8), Ok(None));
        assert!(!on(DECAL, || 1));
        assert!(set("nope", "1", 0).is_err() && set("decal", "x", 0).is_err());
        assert_eq!(value(HOLD, || 0), 0);
        assert_eq!(set("hold", "check", 9), Ok(Some((0, 2))));
        assert!(table_line().contains("decal=0") && table_line().contains("hold=2"));
        assert!(key_map().starts_with("F1 bridgeps=") && key_map().ends_with("F11 mark | F12 table"));
    }

    #[test]
    fn keys_toggle_levels_and_cycle_masks() {
        let (lag, hold, fam) = (&TABLE[LAG], &TABLE[HOLD], &TABLE[BRIDGEFAM]);
        assert_eq!([next_on_key(lag, 1, 1), next_on_key(lag, 0, 1), next_on_key(lag, 0, 0)], [Some(0), Some(1), None]);
        assert_eq!([next_on_key(hold, 0, 0), next_on_key(hold, 0, 2), next_on_key(hold, 2, 2)], [Some(1), Some(2), Some(0)]);
        let mut v = 0b101;
        let mut seen = Vec::new();
        for _ in 0..4 {
            v = next_on_key(fam, v, 0b101).unwrap();
            seen.push(v);
        }
        assert_eq!(seen, [0b001, 0b100, 0, 0b101]);
        assert_eq!(next_on_key(fam, 0, 0), None);
    }

    #[test]
    fn the_file_applies_what_changed() {
        let mut p = Poll::default();
        assert_eq!(items("# c\nBridgePS = off\n\nhold=on, lag=0 # x\nbad\n=1\n"), [("bridgeps".into(), "off".into()), ("hold".into(), "on".into()), ("lag".into(), "0".into())]);
        let ch = p.changes("bridgeps=off\nhold=on\n");
        assert_eq!(ch.len(), 2);
        assert!(p.changes("bridgeps=off\nhold=on\n").is_empty(), "unchanged: nothing (a key press stays)");
        assert_eq!(p.changes("bridgeps=on\nhold=on\nhold=check\n"), [("bridgeps".to_string(), "on".to_string()), ("hold".to_string(), "check".to_string())]);
        assert!(p.changes("").is_empty(), "lines removed: nothing set");
        assert_eq!(p.changes("hold=check").len(), 1, "back after a removal: set again");
    }

    #[test]
    fn the_file_poll_reads_a_file() {
        let f = std::env::temp_dir().join(format!("xnw-live-test-{}.txt", std::process::id()));
        std::fs::write(&f, "ui=off\n").unwrap();
        let mut p = Poll::default();
        assert_eq!(p.changes(&std::fs::read_to_string(&f).unwrap()), [("ui".to_string(), "off".to_string())]);
        std::fs::write(&f, "ui=off\ntex=0\n").unwrap();
        assert_eq!(p.changes(&std::fs::read_to_string(&f).unwrap()), [("tex".to_string(), "0".to_string())]);
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn the_flicker_counter_counts_zigzags() {
        let mut f = Flicker::default();
        // 600 counted frames need 602 means: alternate 10, 12 (zig, not big), then a run of steady frames.
        let mut out = None;
        for k in 0..602u32 {
            let m = if k < 100 { if k % 2 == 0 { 10.0 } else { 12.0 } } else if k < 110 { if k % 2 == 0 { 10.0 } else { 20.0 } } else { 50.0 };
            if let Some(l) = f.push(m, "") {
                out = Some(l);
            }
        }
        // Middles 1..=99 zig by 2 (99); 100 (10 between 12 and 20: zig, not big);
        // 101..=108 by 10 (8 big); 109 (20 between 10 and 50) and the rest: no.
        assert_eq!(out.as_deref(), Some("flicker: 108 of 600 frames zigzag (>0.5), 8 big (>3)"));
        assert_eq!(f.n, 0, "the window starts again after its line");
        // A restart carries no frame over; a ramp is no zigzag.
        f.restart(1);
        assert_eq!((f.n, f.last.len(), f.epoch), (0, 0, 1));
        for k in 0..10 {
            assert_eq!(f.push(f64::from(k), " [B]"), None);
        }
        assert_eq!((f.n, f.zig), (8, 0));
        assert_eq!(f.line(" [B]", " (to the live change)"), "flicker [B]: 0 of 8 frames zigzag (>0.5), 0 big (>3) (to the live change)");
    }

    /// FLK2: a big zigzag names its three frames' pictures; a picture shown
    /// again (same world list) says so; at most NAMED a window.
    #[test]
    fn big_zigzags_are_named() {
        let mut f = Flicker::default();
        let n = |world, pic, lights| Shown { world, pic, rows: 91, bridged: 40, lights, e: 1.711, ..Default::default() };
        assert_eq!(f.push_noted(28.43, 3067, n(2516, 0xa0, 4), ""), (None, None));
        // FLK3: the movie chain of each present (here: the brighter one a
        // blit of the other plane set).
        assert_eq!(f.push_noted(37.52, 3068, Shown { movie: 5, plane: 0x2000, ..n(2516, 0xb0, 4) }, ""), (None, None));
        let (_, named) = f.push_noted(28.43, 3069, n(2517, 0xa0, 4), "");
        assert_eq!(
            named.as_deref(),
            Some(
                "flicker frame 3068: luma 28.43 37.52 28.43; shown w2516 pic 00a0 rows 91+40 lights 4 (+0 -0) energy 0.000 E 1.711 ui 0000 movie 0 plane 0000 | w2516 again pic 00b0 rows 91+40 lights 4 (+0 -0) energy 0.000 E 1.711 ui 0000 movie 5 plane 2000 | w2517 pic 00a0 rows 91+40 lights 4 (+0 -0) energy 0.000 E 1.711 ui 0000 movie 0 plane 0000"
            )
        );
        let mut lines = 1;
        for k in 0..100u64 {
            lines += u32::from(f.push_noted(if k % 2 == 0 { 37.5 } else { 28.4 }, 3070 + k, n(2518 + k, 0, 4), "").1.is_some());
        }
        assert_eq!((lines, f.big), (NAMED, 101));
    }
}
