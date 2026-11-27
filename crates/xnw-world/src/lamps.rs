//! `XNW_WORLD_RENDER=...,lights=on` (LIGHT2, default off; needs
//! `ourlight=metal`): the title's local lights (street lamps, neon,
//! headlights) in our lighting pass.
//!
//! Read side: the title's CPU scene-light array of the frame (976-byte
//! records, 384 a buffer, at 0x1487BAF40 + 374784 x dword_148FDC4D0, count
//! dword_1488CD81C; built under the null GPU [V]).
//! Live it is read where the frame's timecycle is; a capture's pack holds
//! the same bytes (kind 55). [`decode`] takes the fields we use.
//!
//! Each frame [`select`] keeps up to [`MAX`] lights by score and packs them
//! camera-relative into 64-byte records for the pass's constant buffer
//! ([`CB_BYTES`]); the pass ([`crate::ourlight`], `#define LAMPS`) loops
//! over them per pixel with the title's model (TL_PS 69aabd5d, LIGHT1
//! section 1): attenuation 1 / (d^2 + R^2), window sat(1 - (d^2 / r^2)^2)^2,
//! near ramp sat(d^2 k + o), spot cone t (t - t^2 + t^4 (3 - 2 t^2)) with
//! t = sat((cos - cos outer) / (cos inner - cos outer)); on the BRDF the sun
//! uses (diffuse + GGX). Position from our depth. No shadows.
//!
//! Not the title's: capsule (4) and rect (32) lights are point lights at
//! their centre; the sphere-light N.L table is plain N.L; colour is the
//! record's colour x intensity (the title's colour table is written by a
//! compute, LIGHT1 "Open"); the near ramp's (k, o) from the record's pair
//! is inferred ([`ramp`]).
//!
//! Exposure: the light sum is scaled by [`SCALE`] (`lightscale=N`), see there.

/// Bytes of a scene-light record and records in a buffer of the array.
pub const RECORD: usize = 976;
pub const CAP: u32 = 384;
/// Lights kept a frame (the pass's loop bound).
pub const MAX: usize = 64;
/// The pass's constant buffer: a 16-byte head (.x the count) and [`MAX`] 64-byte records.
pub const CB_BYTES: usize = 16 + 64 * MAX;

/// The light sum's scale (`lightscale=N` replaces it). Our pass's units
/// are relative: a white surface under the full sun at N.L = 1 gives 1,
/// i.e. the sun's irradiance is pi, where the title's default sun is
/// light_dir_mult 14.75 (lighting.rs `TC`). The record's intensities are in
/// the title's units, so they are brought to ours by pi / 14.75 = 0.213: a
/// 500-intensity sodium lamp 5 m up (E = 20 at its foot) then lights white
/// to 1.36 before exposure, about the sunlit level, instead of 6.4.
pub const SCALE: f32 = std::f32::consts::PI / 14.75;

/// `lights=on|off` in `XNW_WORLD_RENDER` (the last one wins; default off).
pub fn parse_switch(v: &str) -> bool {
    v.split(',').filter_map(|p| p.trim().strip_prefix("lights=")).next_back() == Some("on")
}

/// `lightscale=N` in `XNW_WORLD_RENDER` (the last one wins; default [`SCALE`]).
pub fn parse_scale(v: &str) -> f32 {
    v.split(',').filter_map(|p| p.trim().strip_prefix("lightscale=")).next_back().and_then(|n| n.parse::<f32>().ok()).filter(|s| s.is_finite() && *s >= 0.0).unwrap_or(SCALE)
}

/// The switch as the environment asks it now (the frame readers ask it).
pub fn asked() -> bool {
    parse_switch(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default())
}

/// Live `lights` (taken once a frame; start: asked): 0 uploads no light.
pub fn live_on() -> bool {
    xnw_core::live::on(xnw_core::live::LIGHTS, || u32::from(asked()))
}

pub const POINT: u32 = 1;
pub const SPOT: u32 = 2;
pub const CAPSULE: u32 = 4;
pub const RECT: u32 = 32;

/// The fields of a scene-light record we use (offsets: LIGHT1's table).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SceneLight {
    /// +0x84: 1 point, 2 spot, 4 capsule, 32 rect.
    pub kind: u32,
    /// +0x00: world position, absolute.
    pub pos: [f32; 3],
    /// +0x20: axis, unit (a spot's direction).
    pub dir: [f32; 3],
    /// +0x10: colour, normalised; +0x44: intensity.
    pub colour: [f32; 3],
    pub intensity: f32,
    /// +0x40: range r in metres (the window's radius).
    pub range: f32,
    /// +0x6C: source radius R (the falloff's).
    pub radius: f32,
    /// +0x50 / +0x54: cos of the spot's inner / outer half angle.
    pub cos_inner: f32,
    pub cos_outer: f32,
    /// +0x5C: the spot's outer half angle in DEGREES.
    pub outer_deg: f32,
    /// +0x48 / +0x4C: the near ramp's pair on d^2.
    pub ramp: [f32; 2],
    /// +0x80: flags2. Bit [`VEHICLE`] is set on every light at a car
    /// and on no street, sign or interior light of the captures (LGT1).
    pub flags2: u32,
}

/// flags2 bit of a car's lights (head, tail, brake, indicator, interior):
/// LGT1, pk7h f1076-1078 / pk7d f1066-1067 / pk8a f1027-1032, every light
/// within 4 m of a vehicle draw's origin carries it, no other light does [I:
/// 37 frames, the title's writer is not read]. Only the probe line uses it.
pub const VEHICLE: u32 = 0x4000;

/// SIREN1: car lights of a light bar's hues (red or blue, saturated) kept
/// first, up to this many, when the frame has more than [`MAX`] lights
/// (`XNW_LAMP_SIREN=N`, read once; 0: off). Police scenes of the main run
/// (a test run f2380-2568, a test run f2255-2449) have 103-114 car lights of
/// 241-250 and keep 37 [V the probe line]; which of them are light-bar
/// lights is U (no capture holds a siren). Tail and brake lights are red
/// too: they are car lights LGT1 wants kept anyway.
pub const SIREN_RESERVE: usize = 16;

/// A light bar's hues: a car light whose largest channel is red or blue
/// and whose other two channels are below [`HUE_SAT`] of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hue {
    Red,
    Blue,
}

/// The other channels' share of the largest at most (0.5: (1, 0.4, 0.2) is
/// red, (1, 0.6, 0.2) amber is not; (0.1, 0.4, 1) police blue is blue).
pub const HUE_SAT: f32 = 0.5;

/// `XNW_LAMP_SIREN=N` (read once; default [`SIREN_RESERVE`]).
fn siren_reserve() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XNW_LAMP_SIREN").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(SIREN_RESERVE).min(MAX))
}

/// One record (at least [`RECORD`] bytes).
pub fn decode_one(b: &[u8]) -> SceneLight {
    let f = |at: usize| f32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    let v = |at: usize| [f(at), f(at + 4), f(at + 8)];
    SceneLight {
        kind: u32::from_le_bytes([b[0x84], b[0x85], b[0x86], b[0x87]]),
        pos: v(0),
        dir: v(0x20),
        colour: v(0x10),
        intensity: f(0x44),
        range: f(0x40),
        radius: f(0x6C),
        cos_inner: f(0x50),
        cos_outer: f(0x54),
        outer_deg: f(0x5C),
        ramp: [f(0x48), f(0x4C)],
        flags2: u32::from_le_bytes([b[0x80], b[0x81], b[0x82], b[0x83]]),
    }
}

/// The first `count` records of the array's current buffer (`raw` from its
/// first record; fewer when `raw` is short, none over [`CAP`]).
pub fn decode(raw: &[u8], count: u32) -> Vec<SceneLight> {
    if count > CAP {
        return Vec::new();
    }
    raw.chunks_exact(RECORD).take(count as usize).map(decode_one).collect()
}

impl SceneLight {
    /// The cone's (cos inner, cos outer): the record's pair; when it is not
    /// a cone (not finite, outside -1..1, inner not above outer), the outer
    /// from the angle in degrees and an inner 10 % of the way to the axis.
    pub fn cone(&self) -> (f32, f32) {
        let (i, o) = (self.cos_inner, self.cos_outer);
        if i.is_finite() && o.is_finite() && (-1.0..=1.0).contains(&o) && i > o && i <= 1.0 {
            return (i, o);
        }
        let o = if self.outer_deg.is_finite() { self.outer_deg.clamp(0.5, 179.0).to_radians().cos() } else { 0.0 };
        (o + (1.0 - o) * 0.1, o)
    }

    /// SIREN1: a car light ([`VEHICLE`]) of a light bar's hue (None: not a
    /// car light, or white, amber or unsaturated).
    pub fn siren_hue(&self) -> Option<Hue> {
        if self.flags2 & VEHICLE == 0 || !self.colour.iter().all(|c| c.is_finite()) {
            return None;
        }
        let [r, g, b] = self.colour;
        if r > 0.0 && g <= HUE_SAT * r && b <= HUE_SAT * r {
            Some(Hue::Red)
        } else if b > 0.0 && r <= HUE_SAT * b && g <= HUE_SAT * b {
            Some(Hue::Blue)
        } else {
            None
        }
    }

    /// Kept by the pass: a known type with light and a range.
    pub fn lights(&self) -> bool {
        matches!(self.kind, POINT | SPOT | CAPSULE | RECT) && self.intensity.is_finite() && self.intensity > 0.0 && self.range.is_finite() && self.range > 0.0 && self.pos.iter().chain(&self.colour).all(|x| x.is_finite())
    }
}

/// The title's range window at squared distance `d2`: sat(1 - (d^2 / r^2)^2)^2.
pub fn window(d2: f32, range: f32) -> f32 {
    let x = d2 / (range * range);
    let w = (1.0 - x * x).clamp(0.0, 1.0);
    w * w
}

/// The near ramp's (k, o) of sat(d^2 k + o) from the record's pair (a, b):
/// 0 at d^2 = a, 1 at d^2 = b; k at most 1e4, the title's value for the
/// pair (0, 0) [V the value; I the rule].
pub fn ramp(pair: [f32; 2]) -> (f32, f32) {
    let (a, b) = (pair[0], pair[1]);
    if !(a.is_finite() && b.is_finite()) {
        return (1e4, 0.0);
    }
    let k = if b - a > 1e-4 { 1.0 / (b - a) } else { 1e4 };
    (k, -a.max(0.0) * k)
}

/// The spot cone at `cos` (of the angle between the axis and the ray to
/// the pixel): t (t - t^2 + t^4 (3 - 2 t^2)), t = sat((cos - outer) / (inner - outer)).
pub fn cone(cos: f32, inner: f32, outer: f32) -> f32 {
    let t = ((cos - outer) / (inner - outer)).clamp(0.0, 1.0);
    let t2 = t * t;
    t * (t - t2 + t2 * t2 * (3.0 - 2.0 * t2))
}

/// The light's weight at a point `d` metres away, `cos` between its axis
/// and the ray to the point (what the pass multiplies colour x intensity
/// and the BRDF by): attenuation x window x ramp x cone.
pub fn weight(l: &SceneLight, d: f32, cos: f32) -> f32 {
    let d2 = d * d;
    let (k, o) = ramp(l.ramp);
    let spot = if l.kind == SPOT {
        let (i, o) = l.cone();
        cone(cos, i, o)
    } else {
        1.0
    };
    window(d2, l.range) / (d2 + l.radius * l.radius) * (d2 * k + o).clamp(0.0, 1.0) * spot
}

/// The frame's numbers for the probe line.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub total: u32,
    /// Point, spot, capsule, rect.
    pub types: [u32; 4],
    pub kept: u32,
    /// The kept by type (as `types`).
    pub kept_types: [u32; 4],
    /// Lights with flags2 [`VEHICLE`]: (in the array, kept).
    pub vehicle: (u32, u32),
    /// SIREN1: car lights of [`Hue::Blue`] and [`Hue::Red`] (in the array,
    /// kept), car lights the pass cannot use (not [`SceneLight::lights`]:
    /// intensity 0, a flash pattern's off phase or a lamp switched off) and
    /// the hued car lights the reserve swapped in.
    pub blue: (u32, u32),
    pub red: (u32, u32),
    pub car_dark: u32,
    pub reserved: u32,
    /// The nearest record: metres, type, intensity, range.
    pub nearest: Option<(f32, u32, f32, f32)>,
    /// The brightest kept (intensity x largest colour channel) and its distance.
    pub brightest: Option<(f32, f32)>,
    /// Record 0 is among the kept.
    pub first_kept: bool,
    /// FLK2: the kept set's energy at the camera, sum of colour x intensity
    /// x scale x fade / max(d^2, 1) (the instrument's, not the pass's).
    pub energy: f32,
    /// FLK2 `lightkeep` on: the churn at the cap (None: off).
    pub keep: Option<KeepStats>,
}

/// FLK2: the kept set against the last frame's (`lightkeep` on).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KeepStats {
    /// Lit last frame but outside the kept set, kept now (they fade in).
    pub entered: u32,
    /// Kept last frame, lit still, dropped by the cap now.
    pub left: u32,
    /// Kept and not lit last frame (newly on, a flash, streamed in): at once.
    pub fresh: u32,
    /// Kept lights still fading in.
    pub fading: u32,
    /// The camera jumped (a cut): no last frame.
    pub reset: bool,
}

impl Stats {
    /// `lights: N this frame, types {...}, kept K, nearest at d m, brightest I`.
    pub fn line(&self) -> String {
        let t = self.types;
        format!(
            "lights: {} this frame, types {{point {}, spot {}, capsule {}, rect {}}}, kept {} {{{} {} {} {}}}, car lights {} (kept {}; blue {}/{}, red {}/{}, dark {}, reserved {}), nearest at {}, brightest {}",
            self.total,
            t[0],
            t[1],
            t[2],
            t[3],
            self.kept,
            self.kept_types[0],
            self.kept_types[1],
            self.kept_types[2],
            self.kept_types[3],
            self.vehicle.0,
            self.vehicle.1,
            self.blue.0,
            self.blue.1,
            self.red.0,
            self.red.1,
            self.car_dark,
            self.reserved,
            self.nearest.map_or("-".into(), |(d, k, i, r)| format!("{d:.1} m (type {k}, I {i:.0}, r {r:.0})")),
            self.brightest.map_or("-".into(), |(i, d)| format!("{i:.0} at {d:.1} m{}", if self.first_kept { "; record 0 kept" } else { "" })),
        ) + &self.keep.map_or(String::new(), |k| format!("; keep: +{} -{} new {} fading {}{}, energy {:.3}", k.entered, k.left, k.fresh, k.fading, if k.reset { " (cut)" } else { "" }, self.energy))
    }
}

/// The frame's kept lights as the pass's records, best first, and the
/// probe's numbers. `cam`: the main camera's world position; `scale`: the
/// light sum's ([`SCALE`]). Score (LGT1): intensity x largest colour
/// channel / (dn^2 + r^2), dn = max(d - r, 0) the camera's distance to the
/// light's range sphere: about the light's lit pool as the screen sees it
/// (its pixels ~ (r / d)^2, their brightness ~ I / r^2); ties go to the
/// nearer. Before LGT1 the score was I x col x (window at the camera +
/// 0.001) / d: every light holding the camera first, then by I / d, which
/// kept a 50-I lamp 70 m away over a car's 1-m tail or brake light 4 m
/// away (pk7d f1066-1067, pk7h f1077: 1 of 2, 0 of 1, 3 of 5 car lights
/// within 30 m kept; all with this score, the rest of the set the same).
/// `XNW_LAMP_SCORE=old` (A/B, read once): the old score.
///
/// A record (16 floats): position - camera, 1 / r^2; colour x intensity x
/// scale, R^2; axis, cone s; cone o, ramp k, ramp o, type. The cone is
/// sat(cos s + o); a light without one has s 0, o 1.
pub fn select(lights: &[SceneLight], cam: [f32; 3], scale: f32) -> (Vec<[f32; 16]>, Stats) {
    select_with(lights, cam, scale, siren_reserve())
}

/// FLK2 `XNW_WORLD_RENDER=...,lightkeep=on|off` (default on; live switch
/// `lightkeep`). Past the cap the plain score reshuffles the kept set each
/// frame as the camera moves (60-95 lights at the 64 cap in the main run's
/// 85-160 s, POST2), so lights pop in and out. With it:
/// - a light kept last frame scores x [`KEEP_BONUS`] at the cap: a newcomer
///   must beat it by that margin (hysteresis);
/// - a light that was lit last frame but outside the kept set and is kept
///   now fades in over [`FADE_FRAMES`] frames (its colour x the fade);
/// - a light that was not lit last frame (just switched on, a flash
///   pattern's on phase, a siren, streamed in) is kept at full colour at
///   once: the title's own flashes are not damped;
/// - SIREN1's reserve runs after the bonus, on the same order: light-bar
///   lights keep their places in the reserve; their colour is untouched
///   unless they enter by the cap while lit (membership, faded as others).
/// A dropped light goes at once (the cap leaves no slot to fade it out);
/// the bonus makes that rare. A camera jump of [`CUT_M`] starts again.
/// Lights are the same across frames by type and position (nearest within
/// [`MATCH_M`]): the array has no stable index [U].
pub fn parse_keep(v: &str) -> bool {
    v.split(',').filter_map(|p| p.trim().strip_prefix("lightkeep=")).next_back().is_none_or(|v| !matches!(v.trim(), "off" | "0"))
}

/// Live `lightkeep` (taken once a frame; start: `XNW_WORLD_RENDER`'s).
pub fn keep_on() -> bool {
    xnw_core::live::on(xnw_core::live::LIGHTKEEP, || u32::from(parse_keep(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default())))
}

/// A kept light's score factor at the cap (FLK2).
pub const KEEP_BONUS: f32 = 1.5;
/// Frames a cap entry takes to reach full colour (FLK2).
pub const FADE_FRAMES: f32 = 6.0;
/// The same light in the last frame: same type, within this (metres; a car
/// at 216 km/h moves 1 m a frame at 60 frames a second).
pub const MATCH_M: f32 = 1.0;
/// A camera move this long between frames is a cut: no last frame.
pub const CUT_M: f32 = 20.0;

/// FLK2: the last frame's lit lights for [`select_keep`].
#[derive(Clone, Debug, Default)]
pub struct Keep {
    /// World position, type, kept, fade (0..=1), by x.
    prev: Vec<([f32; 3], u32, bool, f32)>,
    cam: Option<[f32; 3]>,
}

impl Keep {
    /// The last frame's light of `kind` nearest `pos` within [`MATCH_M`]: (kept, fade).
    fn find(&self, pos: [f32; 3], kind: u32) -> Option<(bool, f32)> {
        let from = self.prev.partition_point(|p| p.0[0] < pos[0] - MATCH_M);
        let mut best: Option<(f32, bool, f32)> = None;
        for p in self.prev[from..].iter().take_while(|p| p.0[0] <= pos[0] + MATCH_M) {
            let d2 = (0..3).map(|k| (p.0[k] - pos[k]) * (p.0[k] - pos[k])).sum::<f32>();
            if p.1 == kind && d2 <= MATCH_M * MATCH_M && best.is_none_or(|b| d2 < b.0) {
                best = Some((d2, p.2, p.3));
            }
        }
        best.map(|b| (b.1, b.2))
    }
}

/// [`select`] with the kept set held across frames (FLK2, see
/// [`parse_keep`]); `keep` is the pass's, carried from frame to frame.
pub fn select_keep(lights: &[SceneLight], cam: [f32; 3], scale: f32, keep: &mut Keep) -> (Vec<[f32; 16]>, Stats) {
    pick(lights, cam, scale, siren_reserve(), Some(keep))
}

/// [`select`] with the light-bar reserve given (SIREN1): when more than
/// [`MAX`] lights score, up to `reserve` car lights of a light bar's hue
/// ([`SceneLight::siren_hue`]) the cap would drop replace the lowest-scored
/// kept lights that are not such lights; the kept set is then best first
/// again. 0, or at most [`MAX`] lights: the plain score.
pub fn select_with(lights: &[SceneLight], cam: [f32; 3], scale: f32, reserve: usize) -> (Vec<[f32; 16]>, Stats) {
    pick(lights, cam, scale, reserve, None)
}

fn pick(lights: &[SceneLight], cam: [f32; 3], scale: f32, reserve: usize, mut keep: Option<&mut Keep>) -> (Vec<[f32; 16]>, Stats) {
    let mut st = Stats { total: lights.len() as u32, ..Default::default() };
    // FLK2: a camera jump starts the kept set again; each light's last frame (kept, fade).
    let mut ks = KeepStats::default();
    if let Some(k) = keep.as_deref_mut()
        && k.cam.is_some_and(|c| (0..3).map(|j| (c[j] - cam[j]) * (c[j] - cam[j])).sum::<f32>() > CUT_M * CUT_M)
    {
        k.prev.clear();
        ks.reset = true;
    }
    let was: Vec<Option<(bool, f32)>> = keep.as_deref().map_or(Vec::new(), |k| lights.iter().map(|l| k.find(l.pos, l.kind)).collect());
    let mut scored: Vec<(f32, f32, usize)> = Vec::with_capacity(lights.len());
    for (i, l) in lights.iter().enumerate() {
        if let Some(k) = [POINT, SPOT, CAPSULE, RECT].iter().position(|t| *t == l.kind) {
            st.types[k] += 1;
        }
        st.vehicle.0 += u32::from(l.flags2 & VEHICLE != 0);
        st.car_dark += u32::from(l.flags2 & VEHICLE != 0 && !l.lights());
        match l.siren_hue() {
            Some(Hue::Blue) => st.blue.0 += 1,
            Some(Hue::Red) => st.red.0 += 1,
            None => {}
        }
        let d = (0..3).map(|k| (l.pos[k] - cam[k]) * (l.pos[k] - cam[k])).sum::<f32>().sqrt();
        if d.is_finite() && st.nearest.is_none_or(|n| d < n.0) {
            st.nearest = Some((d, l.kind, l.intensity, l.range));
        }
        if !l.lights() || !d.is_finite() {
            continue;
        }
        let power = l.intensity * l.colour.iter().fold(0f32, |a, c| a.max(*c)).max(1e-3);
        let dn = (d - l.range).max(0.0);
        let score = if old_score() { power * (window(d * d, l.range) + 1e-3) / d.max(0.1) } else { power / (dn * dn + l.range * l.range + 1e-3) };
        // FLK2: kept last frame: it holds its place unless out-scored by the margin.
        let bonus = if was.get(i).copied().flatten().is_some_and(|w| w.0) { KEEP_BONUS } else { 1.0 };
        scored.push((score * bonus, d, i));
    }
    let order = |a: &(f32, f32, usize), b: &(f32, f32, usize)| b.0.total_cmp(&a.0).then(a.1.total_cmp(&b.1));
    scored.sort_by(order);
    // FLK2: every lit light (the kept set's churn and the next frame).
    let lit: Vec<usize> = if keep.is_some() { scored.iter().map(|s| s.2).collect() } else { Vec::new() };
    if scored.len() > MAX && reserve > 0 {
        let hued = |i: usize| lights[i].siren_hue().is_some();
        let mut have = scored[..MAX].iter().filter(|s| hued(s.2)).count();
        let mut extra = scored[MAX..].iter().filter(|s| hued(s.2)).copied().collect::<Vec<_>>().into_iter();
        for pos in (0..MAX).rev() {
            if have >= reserve.min(MAX) {
                break;
            }
            if hued(scored[pos].2) {
                continue;
            }
            let Some(e) = extra.next() else { break };
            scored[pos] = e;
            have += 1;
            st.reserved += 1;
        }
        scored.truncate(MAX);
        scored.sort_by(order);
    }
    scored.truncate(MAX);
    st.kept = scored.len() as u32;
    let mut out = Vec::with_capacity(scored.len());
    let mut next: Vec<([f32; 3], u32, bool, f32)> = Vec::new();
    for &(_, d, i) in &scored {
        let l = &lights[i];
        // FLK2: the fade: a cap entry ramps in, a newly lit light is at once.
        let fade = match (keep.is_some(), was.get(i).copied().flatten()) {
            (false, _) => 1.0,
            (true, None) => {
                ks.fresh += 1;
                1.0
            }
            (true, Some((false, _))) => {
                ks.entered += 1;
                1.0 / FADE_FRAMES
            }
            (true, Some((true, f))) => (f + 1.0 / FADE_FRAMES).min(1.0),
        };
        ks.fading += u32::from(fade < 1.0);
        if keep.is_some() {
            next.push((l.pos, l.kind, true, fade));
        }
        st.energy += l.colour.iter().fold(0f32, |a, c| a.max(*c)) * l.intensity * scale * fade / (d * d).max(1.0);
        let power = l.intensity * l.colour.iter().fold(0f32, |a, c| a.max(*c));
        if st.brightest.is_none_or(|b| power > b.0) {
            st.brightest = Some((power, d));
        }
        st.first_kept |= i == 0;
        if let Some(k) = [POINT, SPOT, CAPSULE, RECT].iter().position(|t| *t == l.kind) {
            st.kept_types[k] += 1;
        }
        st.vehicle.1 += u32::from(l.flags2 & VEHICLE != 0);
        match l.siren_hue() {
            Some(Hue::Blue) => st.blue.1 += 1,
            Some(Hue::Red) => st.red.1 += 1,
            None => {}
        }
        let (s, o) = if l.kind == SPOT {
            let (ci, co) = l.cone();
            let s = 1.0 / (ci - co);
            (s, -co * s)
        } else {
            (0.0, 1.0)
        };
        let (k, ro) = ramp(l.ramp);
        let n = (l.dir[0] * l.dir[0] + l.dir[1] * l.dir[1] + l.dir[2] * l.dir[2]).sqrt();
        let axis = if n.is_finite() && n > 0.5 { l.dir.map(|x| x / n) } else { [0.0, 0.0, -1.0] };
        let radius = if l.radius.is_finite() { l.radius.clamp(0.001, 10.0) } else { 0.001 };
        let c = l.colour.map(|x| x.max(0.0) * l.intensity * scale * fade);
        out.push([l.pos[0] - cam[0], l.pos[1] - cam[1], l.pos[2] - cam[2], 1.0 / (l.range * l.range), c[0], c[1], c[2], radius * radius, axis[0], axis[1], axis[2], s, o, k, ro, l.kind as f32]);
    }
    if let Some(k) = keep {
        let mut kept = vec![false; lights.len()];
        for s in &scored {
            kept[s.2] = true;
        }
        for &i in lit.iter().filter(|&&i| !kept[i]) {
            ks.left += u32::from(was.get(i).copied().flatten().is_some_and(|w| w.0));
            next.push((lights[i].pos, lights[i].kind, false, 0.0));
        }
        next.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]));
        (k.prev, k.cam) = (next, Some(cam));
        st.keep = Some(ks);
    }
    (out, st)
}

/// `XNW_LAMP_SCORE=old` (LGT1's A/B): [`select`] scores as before LGT1.
fn old_score() -> bool {
    static OLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OLD.get_or_init(|| std::env::var("XNW_LAMP_SCORE").is_ok_and(|v| v.trim() == "old"))
}

/// The constant buffer's bytes for `records` (at most [`MAX`] are taken).
pub fn constants(records: &[[f32; 16]]) -> Vec<u8> {
    let n = records.len().min(MAX);
    let mut b = vec![0u8; CB_BYTES];
    b[..4].copy_from_slice(&(n as u32).to_le_bytes());
    for (i, r) in records.iter().take(n).enumerate() {
        for (j, v) in r.iter().enumerate() {
            b[16 + 64 * i + 4 * j..20 + 64 * i + 4 * j].copy_from_slice(&v.to_le_bytes());
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A record as the title lays it out (LIGHT1's table), from a lamp spot
    /// of pk8a's kind: sodium colour, pointing down, outer 75 degrees.
    fn record(kind: u32, pos: [f32; 3], intensity: f32, range: f32) -> Vec<u8> {
        let mut b = vec![0u8; RECORD];
        let mut put = |at: usize, v: f32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        for k in 0..3 {
            put(4 * k, pos[k]);
            put(0x10 + 4 * k, [1.0, 0.57, 0.13][k]);
            put(0x20 + 4 * k, [0.0, 0.0, -1.0][k]);
        }
        put(0x40, range);
        put(0x44, intensity);
        put(0x48, 0.0676);
        put(0x4C, 0.122);
        put(0x50, 0.5);
        put(0x54, 0.258_819);
        put(0x5C, 75.0);
        put(0x6C, 0.05);
        b[0x84..0x88].copy_from_slice(&kind.to_le_bytes());
        b
    }

    #[test]
    fn switches() {
        assert!(!parse_switch("screen,ourlight=metal"));
        assert!(parse_switch("screen,ourlight=metal,lights=on"));
        assert!(!parse_switch("lights=on,lights=off"));
        assert_eq!(parse_scale("lights=on"), SCALE);
        assert_eq!(parse_scale("lights=on,lightscale=0.5"), 0.5);
        assert_eq!(parse_scale("lightscale=-1"), SCALE);
        assert!((SCALE - 0.21299).abs() < 1e-4);
    }

    /// The fields at their offsets; the angle at +0x5C is in degrees:
    /// cos(75 degrees) = 0.258819 is the record's cos outer (cos(75 rad) = 0.9218 is not).
    #[test]
    fn record_decodes_and_the_angle_is_degrees() {
        let mut raw = record(SPOT, [1847.0, 147.25, 11.0], 750.0, 15.0);
        raw.extend(record(POINT, [1.0, 2.0, 3.0], 0.0, 10.0));
        let v = decode(&raw, 2);
        assert_eq!(v.len(), 2);
        let l = v[0];
        assert_eq!((l.kind, l.pos, l.dir, l.colour), (SPOT, [1847.0, 147.25, 11.0], [0.0, 0.0, -1.0], [1.0, 0.57, 0.13]));
        assert_eq!((l.intensity, l.range, l.radius, l.outer_deg, l.ramp), (750.0, 15.0, 0.05, 75.0, [0.0676, 0.122]));
        assert!((l.outer_deg.to_radians().cos() - l.cos_outer).abs() < 1e-5, "degrees");
        assert!((l.outer_deg.cos() - l.cos_outer).abs() > 0.5, "not radians");
        assert_eq!(l.cone(), (0.5, 0.258_819));
        // A record without a usable pair: the outer from the degrees.
        let mut bad = l;
        (bad.cos_inner, bad.cos_outer, bad.outer_deg) = (0.0, 0.0, 60.0);
        let (i, o) = bad.cone();
        assert!((o - 0.5).abs() < 1e-5 && (i - 0.55).abs() < 1e-5, "{i} {o}");
        assert!(l.lights() && !v[1].lights(), "intensity 0 is not kept");
        // A count over the array's capacity or bytes that are short: no lights, the records there are.
        assert!(decode(&raw, 385).is_empty());
        assert_eq!(decode(&raw[..RECORD + 10], 2).len(), 1);
    }

    /// Hand values. Window at r = 10: d = 5: x = 0.25, (1 - 0.0625)^2 = 0.878906;
    /// d = 9: x = 0.81, (1 - 0.6561)^2 = 0.118268; d >= 10: 0.
    /// Cone inner 0.5, outer 0.25: cos 0.375: t = 0.5, 0.5 (0.5 - 0.25 + 0.0625 x 2.5) = 0.203125;
    /// cos 0.5: 1; cos 0.25: 0.
    /// Ramp (0.0676, 0.122): k = 1 / 0.0544 = 18.382, o = -1.2426: d^2 = 0.0948: 0.5; pair (0, 0): k 1e4, o 0.
    /// Weight of a point light r = 10, R = 0.05, no ramp, at d = 5: 0.878906 / 25.0025 = 0.035153.
    #[test]
    fn attenuation_matches_hand_values() {
        assert!((window(25.0, 10.0) - 0.878_906).abs() < 1e-6);
        assert!((window(81.0, 10.0) - 0.118_268).abs() < 1e-6);
        assert_eq!((window(100.0, 10.0), window(400.0, 10.0), window(0.0, 10.0)), (0.0, 0.0, 1.0));
        assert!((cone(0.375, 0.5, 0.25) - 0.203_125).abs() < 1e-6);
        assert_eq!((cone(0.5, 0.5, 0.25), cone(0.9, 0.5, 0.25), cone(0.25, 0.5, 0.25), cone(-1.0, 0.5, 0.25)), (1.0, 1.0, 0.0, 0.0));
        let (k, o) = ramp([0.0676, 0.122]);
        assert!((k - 18.382_353).abs() < 1e-3 && (o + 1.242_647).abs() < 1e-4, "{k} {o}");
        assert!((0.0948 * k + o - 0.5).abs() < 1e-3);
        assert_eq!(ramp([0.0, 0.0]), (1e4, 0.0));
        assert_eq!(ramp([0.01, 0.01]).0, 1e4);
        let mut l = decode_one(&record(POINT, [0.0; 3], 100.0, 10.0));
        l.ramp = [0.0, 0.0];
        assert!((weight(&l, 5.0, 1.0) - 0.035_153).abs() < 1e-6, "{}", weight(&l, 5.0, 1.0));
        // The same as a spot (inner 0.5, outer 0.258819) at cos 0.379410 (t = 0.5): x 0.203125.
        l.kind = SPOT;
        assert!((weight(&l, 5.0, 0.379_410) - 0.035_153 * 0.203_125).abs() < 1e-6);
        assert_eq!(weight(&l, 12.0, 1.0), 0.0, "past the range");
    }

    /// The kept set: no light without intensity, best score first, at most
    /// MAX; positions relative to the camera; the cone as (s, o); the probe's numbers.
    #[test]
    fn select_scores_and_packs() {
        let cam = [100.0, 200.0, 2.0];
        let mut raw = record(SPOT, [100.0, 203.0, 6.0], 750.0, 15.0); // 5 m
        raw.extend(record(POINT, [100.0, 200.9, 2.0], 0.0, 10.0)); // 0.9 m, no light
        raw.extend(record(CAPSULE, [100.0, 240.0, 2.0], 5000.0, 12.0)); // 40 m, camera outside its range
        raw.extend(record(RECT, [102.0, 200.0, 2.0], 50.0, 6.0)); // 2 m
        let lights = decode(&raw, 4);
        let (recs, st) = select(&lights, cam, 0.5);
        assert_eq!((st.total, st.types, st.kept, st.first_kept), (4, [1, 1, 1, 1], 3, true));
        let n = st.nearest.unwrap();
        assert!((n.0 - 0.9).abs() < 1e-4 && (n.1, n.2, n.3) == (POINT, 0.0, 10.0), "{n:?}");
        assert_eq!(st.brightest.map(|b| b.0), Some(5000.0));
        assert_eq!(st.line(), "lights: 4 this frame, types {point 1, spot 1, capsule 1, rect 1}, kept 3 {0 1 1 1}, car lights 0 (kept 0; blue 0/0, red 0/0, dark 0, reserved 0), nearest at 0.9 m (type 1, I 0, r 10), brightest 5000 at 40.0 m; record 0 kept");
        // Scores: capsule 5000 / (28^2 + 12^2) = 5.39; spot 750 / 15^2 = 3.33 (the camera inside); rect 50 / 6^2 = 1.39.
        assert_eq!(recs.iter().map(|r| r[15] as u32).collect::<Vec<_>>(), [CAPSULE, SPOT, RECT]);
        let r = &recs[1];
        assert_eq!(&r[..3], &[0.0, 3.0, 4.0]);
        assert!((r[3] - 1.0 / 225.0).abs() < 1e-9 && (r[7] - 0.0025).abs() < 1e-9);
        assert!((r[4] - 375.0).abs() < 1e-3 && (r[5] - 213.75).abs() < 1e-3 && (r[6] - 48.75).abs() < 1e-3, "colour x intensity x scale");
        // s = 1 / (0.5 - 0.258819) = 4.14626, o = -0.258819 s = -1.07313: cos 0.5 -> 1, cos outer -> 0.
        assert!((r[11] - 4.146_26).abs() < 1e-3 && (0.5 * r[11] + r[12] - 1.0).abs() < 1e-5 && (0.258_819 * r[11] + r[12]).abs() < 1e-5);
        assert_eq!((recs[2][11], recs[2][12]), (0.0, 1.0), "no cone on a rect");
        // A car's tail light 4 m away (I 2.5, r 0.8) is kept over a 50-I lamp 70 m away (r 6): 0.15 vs 0.012 (LGT1).
        let mut tail = record(SPOT, [104.0, 200.0, 2.0], 2.5, 0.8);
        tail[0x80..0x84].copy_from_slice(&VEHICLE.to_le_bytes());
        tail.extend(record(SPOT, [170.0, 200.0, 2.0], 50.0, 6.0));
        let two = decode(&tail, 2);
        assert_eq!(two[0].flags2, VEHICLE);
        let mut crowd: Vec<SceneLight> = (0..MAX - 1).map(|i| SceneLight { kind: POINT, pos: [100.0, 300.0 + i as f32, 2.0], intensity: 500.0, range: 15.0, colour: [1.0; 3], ..Default::default() }).collect();
        crowd.extend(two);
        let (_, st) = select(&crowd, cam, 1.0);
        assert_eq!((st.kept, st.vehicle), (MAX as u32, (1, 1)), "the tail light is in, the far lamp out");
        // More than MAX: the best MAX; the buffer holds the count and the records.
        let many: Vec<SceneLight> = (0..100).map(|i| SceneLight { kind: POINT, pos: [100.0 + i as f32, 200.0, 2.0], intensity: 100.0, range: 500.0, colour: [1.0; 3], ..Default::default() }).collect();
        let (recs, st) = select(&many, cam, 1.0);
        assert_eq!((recs.len(), st.kept, st.total), (MAX, MAX as u32, 100));
        assert!(recs.iter().all(|r| r[0] < MAX as f32), "the nearest of equal lights");
        let b = constants(&recs);
        assert_eq!((b.len(), u32::from_le_bytes([b[0], b[1], b[2], b[3]])), (CB_BYTES, MAX as u32));
        assert_eq!(f32::from_le_bytes([b[16 + 64], b[17 + 64], b[18 + 64], b[19 + 64]]), recs[1][0]);
        assert_eq!(constants(&[])[..4], [0; 4]);
    }

    fn car(colour: [f32; 3], at: f32, intensity: f32) -> SceneLight {
        SceneLight { kind: SPOT, pos: [100.0, 200.0 + at, 2.0], colour, intensity, range: 2.0, flags2: VEHICLE, ..Default::default() }
    }

    /// SIREN1: the light-bar hues are saturated red or blue car lights only.
    #[test]
    fn siren_hues() {
        assert_eq!(car([0.1, 0.4, 1.0], 0.0, 1.0).siren_hue(), Some(Hue::Blue));
        assert_eq!(car([1.0, 0.0, 0.0], 0.0, 1.0).siren_hue(), Some(Hue::Red));
        assert_eq!(car([1.0, 0.4, 0.2], 0.0, 1.0).siren_hue(), Some(Hue::Red));
        assert_eq!(car([1.0, 0.6, 0.2], 0.0, 1.0).siren_hue(), None, "amber indicator");
        assert_eq!(car([1.0, 0.57, 0.13], 0.0, 1.0).siren_hue(), None, "sodium");
        assert_eq!(car([1.0, 1.0, 1.0], 0.0, 1.0).siren_hue(), None, "headlight");
        assert_eq!(car([0.0; 3], 0.0, 1.0).siren_hue(), None);
        assert_eq!(car([f32::NAN, 0.0, 1.0], 0.0, 1.0).siren_hue(), None);
        let mut street = car([0.0, 0.0, 1.0], 0.0, 1.0);
        street.flags2 = 0;
        assert_eq!(street.siren_hue(), None, "a blue neon is not a car light");
    }

    /// SIREN1: over MAX lights, light-bar car lights the cap drops replace the
    /// weakest other kept lights, up to the reserve; at most MAX lights or a
    /// reserve of 0: the plain score.
    #[test]
    fn siren_reserve_keeps_light_bars() {
        let cam = [100.0, 200.0, 2.0];
        // MAX street lamps 20-83 m away (I 500, r 15), then 6 light-bar lights 60 m away (I 20, r 2) and a white headlight.
        let mut v: Vec<SceneLight> = (0..MAX).map(|i| SceneLight { kind: POINT, pos: [100.0, 220.0 + i as f32, 2.0], intensity: 500.0, range: 15.0, colour: [1.0, 0.57, 0.13], ..Default::default() }).collect();
        for k in 0..6 {
            v.push(car(if k % 2 == 0 { [0.1, 0.3, 1.0] } else { [1.0, 0.05, 0.05] }, 60.0 + k as f32 * 0.1, 20.0));
        }
        v.push(car([1.0; 3], 61.0, 20.0));
        v[MAX + 1].intensity = 0.0; // a red one in its off phase: counted dark, never kept
        let (_, plain) = select_with(&v, cam, 1.0, 0);
        assert_eq!((plain.kept, plain.vehicle, plain.blue, plain.red, plain.car_dark, plain.reserved), (MAX as u32, (7, 0), (3, 0), (3, 0), 1, 0));
        let (recs, st) = select_with(&v, cam, 1.0, 16);
        assert_eq!((st.kept, st.vehicle, st.blue, st.red, st.car_dark, st.reserved), (MAX as u32, (7, 5), (3, 3), (3, 2), 1, 5));
        assert_eq!(recs.len(), MAX);
        // The farthest lamps went (83 m and down); the nearest stay; best first: the lamps lead.
        assert!(recs.iter().filter(|r| r[15] as u32 == POINT).all(|r| r[1] < 20.0 + (MAX - 5) as f32), "the 5 farthest lamps out");
        assert_eq!(recs[0][15] as u32, POINT);
        // A reserve smaller than the hued: only that many.
        assert_eq!(select_with(&v, cam, 1.0, 2).1.reserved, 2);
        // At most MAX lights: nothing to reserve.
        let few = &v[MAX - 10..];
        let (_, f) = select_with(few, cam, 1.0, 16);
        assert_eq!((f.kept, f.reserved, f.vehicle.1), (16, 0, 6));
        assert!(st.line().contains("car lights 7 (kept 5; blue 3/3, red 3/2, dark 1, reserved 5)"), "{}", st.line());
    }

    /// FLK2: 80 street lamps on a ring around the camera (all past the cap),
    /// each frame's intensity jittered by up to +-10 % (the array as the
    /// camera moves): the plain score swaps kept lights every frame, the
    /// held set none after the first; the kept count stays MAX.
    #[test]
    fn keep_holds_a_churning_set() {
        let cam = [0.0, 0.0, 0.0];
        let mut seed = 12345u32;
        let mut jitter = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            0.9 + 0.2 * (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let ring = |k: usize| [30.0 * (k as f32 * 0.0785).cos(), 30.0 * (k as f32 * 0.0785).sin(), 5.0];
        let (mut keep, mut last_plain, mut last_keep) = (Keep::default(), Vec::<f32>::new(), Vec::<f32>::new());
        let (mut swaps_plain, mut swaps_keep, mut churn) = (0usize, 0usize, 0u32);
        for frame in 0..50 {
            let lights: Vec<SceneLight> = (0..80).map(|k| SceneLight { kind: POINT, pos: ring(k), intensity: 500.0 * jitter(), range: 15.0, colour: [1.0; 3], ..Default::default() }).collect();
            let (plain, _) = pick(&lights, cam, 1.0, 0, None);
            let (held, st) = pick(&lights, cam, 1.0, 0, Some(&mut keep));
            assert_eq!((plain.len(), held.len()), (MAX, MAX));
            // A kept light by its position (x).
            let xs = |r: &Vec<[f32; 16]>| r.iter().map(|r| r[0]).collect::<Vec<f32>>();
            let gone = |now: &Vec<f32>, then: &Vec<f32>| then.iter().filter(|x| !now.contains(x)).count();
            let (p, h) = (xs(&plain), xs(&held));
            if frame > 0 {
                swaps_plain += gone(&p, &last_plain);
                swaps_keep += gone(&h, &last_keep);
                let k = st.keep.unwrap();
                churn += k.entered + k.left;
                assert_eq!((k.fresh, k.fading, k.reset), (0, 0, false));
            } else {
                assert_eq!(st.keep.unwrap().fresh, MAX as u32, "the first frame: all at once");
            }
            (last_plain, last_keep) = (p, h);
        }
        assert!(swaps_plain > 100, "the plain score churns: {swaps_plain}");
        assert_eq!((swaps_keep, churn), (0, 0));
    }

    /// FLK2: a lamp that out-scores a kept one by more than the margin
    /// enters and fades in over FADE_FRAMES frames; the one it replaces goes
    /// (left 1); less than the margin and it stays out.
    #[test]
    fn keep_fades_cap_entries_in() {
        let cam = [0.0, 0.0, 0.0];
        let mut lights: Vec<SceneLight> = (0..=MAX).map(|k| SceneLight { kind: POINT, pos: [10.0 + k as f32, 0.0, 0.0], intensity: 500.0, range: 15.0, colour: [1.0; 3], ..Default::default() }).collect();
        let newcomer = MAX; // the farthest: out at first
        let mut keep = Keep::default();
        let (recs, _) = pick(&lights, cam, 1.0, 0, Some(&mut keep));
        assert!(recs.iter().all(|r| r[0] < 10.0 + MAX as f32));
        // 1.4x the weakest kept: under the margin, still out.
        lights[newcomer].intensity = 500.0 * 1.4;
        let (recs, st) = pick(&lights, cam, 1.0, 0, Some(&mut keep));
        assert!(recs.iter().all(|r| r[0] < 10.0 + MAX as f32) && st.keep.unwrap().entered == 0);
        // 3x: in, at 1/6, then 2/6 .. 1.
        lights[newcomer].intensity = 1500.0;
        let mut fades = Vec::new();
        for frame in 0..8 {
            let (recs, st) = pick(&lights, cam, 1.0, 0, Some(&mut keep));
            let r = recs.iter().find(|r| r[0] == 10.0 + MAX as f32).expect("the newcomer is kept");
            fades.push(r[4] / 1500.0);
            let k = st.keep.unwrap();
            if frame == 0 {
                assert_eq!((k.entered, k.left, k.fading), (1, 1, 1));
                assert!(st.line().ends_with(&format!("; keep: +1 -1 new 0 fading 1, energy {:.3}", st.energy)), "{}", st.line());
            }
        }
        let want = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0].map(|n: f32| n / FADE_FRAMES);
        assert!(fades.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-5), "{fades:?}");
    }

    /// FLK2 with SIREN1: a light bar's flash (dark, then lit) and a lamp
    /// switched on come at full colour at once, in a frame past the cap
    /// with the reserve on; a camera jump past CUT_M starts again (no fade).
    #[test]
    fn keep_lets_flashes_and_cuts_through() {
        let cam = [100.0, 200.0, 0.0];
        let mut lights: Vec<SceneLight> = (0..MAX + 10).map(|k| SceneLight { kind: POINT, pos: [100.0, 210.0 + k as f32, 3.0], intensity: 500.0, range: 15.0, colour: [1.0; 3], ..Default::default() }).collect();
        lights.push(car([0.1, 0.4, 1.0], 30.0, 0.0)); // the bar's blue, dark
        let bar = lights.len() - 1;
        let mut keep = Keep::default();
        let (_, st) = pick(&lights, cam, 1.0, 16, Some(&mut keep));
        assert_eq!((st.kept, st.reserved), (MAX as u32, 0));
        lights[bar].intensity = 20.0;
        let (recs, st) = pick(&lights, cam, 1.0, 16, Some(&mut keep));
        let r = recs.iter().find(|r| r[15] == SPOT as f32).expect("the flash is kept by the reserve");
        assert_eq!((r[6], st.reserved, st.keep.unwrap().fresh, st.keep.unwrap().fading), (20.0, 1, 1, 0), "full colour at once");
        // Off again, on again: at once each time.
        lights[bar].intensity = 0.0;
        let (recs, _) = pick(&lights, cam, 1.0, 16, Some(&mut keep));
        assert!(recs.iter().all(|r| r[15] != SPOT as f32));
        lights[bar].intensity = 20.0;
        let (recs, _) = pick(&lights, cam, 1.0, 16, Some(&mut keep));
        assert_eq!(recs.iter().find(|r| r[15] == SPOT as f32).map(|r| r[6]), Some(20.0));
        // A cut: everything as new, nothing fades.
        let far = [cam[0] + 50.0, cam[1], cam[2]];
        let (_, st) = pick(&lights, far, 1.0, 16, Some(&mut keep));
        let k = st.keep.unwrap();
        assert!(k.reset && k.entered == 0 && k.fading == 0 && k.fresh == MAX as u32, "{k:?}");
        // The switch: on unless off or 0, the last one wins.
        assert!(parse_keep("") && parse_keep("screen,lightkeep=on") && !parse_keep("lightkeep=off") && !parse_keep("lightkeep=on,lightkeep=0"));
    }
}
