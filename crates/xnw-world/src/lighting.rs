//! Our pixel shader's lighting and output chain, as the title's.
//!
//! Title chain (V from its programs): radiance E = light
//! colour x light_dir_mult (no saturate), diffuse albedo / pi x E x
//! saturate(N.L) (N.L = -dot(D, N), D the light vector; our timecycle copy
//! holds the direction TOWARDS the light, +0x1630, so N.L = dot(N, that)),
//! plus ambient; x exposure (exp2 of the fixed EV), Hable(x) / Hable(W),
//! then pow 1/2.2 into a UNORM target. Albedo stays as the title stores it
//! (the sampled value, straight into the G-buffer), so there is
//! one conversion end to end, the final pow.
//!
//! Ambient: the title's is GI (no closed form). Stand-in: the sky stand-in's
//! zenith / horizon colours, linearised, as a hemisphere, scaled so a
//! white diffuse surface in shade sits at [`SHADE_FRACTION`] of the same
//! surface sunlit (N.L = 1), with a floor ([`AMBIENT_FLOOR`]) so that night
//! (moon as the directional light) stays readable.
//!
//! Every timecycle value used is in [`TC`]: name, float index into the
//! render copy (0x16C0 bytes), default, evidence.

/// (name, float index, default, evidence).
pub const TC: [(&str, usize, f32, &str); 28] = [
    ("light_dir_col.r", 0, 1.0, "V"),
    ("light_dir_col.g", 1, 1.0, "V"),
    ("light_dir_col.b", 2, 1.0, "V"),
    ("light_dir_mult", 3, 14.75, "V (default 14.75)"),
    ("ambient_sky_intensity", 204, 1.0, "I TC[204]"),
    ("postfx_exposure", 794, 0.0, "I b6+0x42C"),
    ("exposure_min", 796, -3.5, "I b6+0x420"),
    ("exposure_max", 797, 3.0, "I b6+0x424"),
    ("filmic_a", 838, 0.22, "I b6+0x4A0"),
    ("filmic_b", 839, 0.3, "I"),
    ("filmic_c", 840, 0.1, "I"),
    ("filmic_d", 841, 0.2, "I"),
    ("filmic_e", 842, 0.01, "I"),
    ("filmic_f", 843, 0.3, "I"),
    ("filmic_w", 844, 4.0, "I"),
    ("filmic_a_bright", 847, 0.22, "I b6+0x4C0 (read, not used: the dark/bright blend factor is U)"),
    ("filmic_w_bright", 853, 4.0, "I"),
    ("light_dir.x (final, towards the light)", 0x1630 / 4, 0.0, "V"),
    ("light_dir.y", 0x1630 / 4 + 1, 0.0, "V"),
    ("light_dir.z", 0x1630 / 4 + 2, 1.0, "V"),
    ("sun_dir.z", 0x1610 / 4 + 2, 1.0, "V"),
    // Hash-verified names / the title's own record [V]:
    ("postfx_exposure (second term)", 795, 0.0, "V: EV = 794 + 795"),
    ("directional strength", 0x1650 / 4, 1.0, "V TC+0x1650"),
    ("moon switch", 0x1670 / 4, 0.0, "V TC+0x1670 (> 0: moon set)"),
    ("moon_col.r", 9, 1.0, "V"),
    ("moon_col.g", 10, 1.0, "V"),
    ("moon_col.b", 11, 1.0, "V"),
    ("moon_mult", 12, 1.0, "V"),
];

/// A white surface in shade against the same surface sunlit at N.L = 1.
pub const SHADE_FRACTION: f32 = 0.2;
/// Least ambient luminance AFTER exposure (the floor is divided by the
/// exposure): about 0.2 on screen for white with the default curve, so a
/// night EV of -2 (exposure min = max = -2) stays readable.
pub const AMBIENT_FLOOR: f32 = 0.08;

/// `ourlight=on` ([`Lighting::ours`]; our choices, not the title's values):
/// a white surface facing up in shade against the same surface under a
/// full sun at N.L = 1, sky and ground; their tints (a faint cool sky, a
/// warm ground); the moon set's level, the moon fill with the directional
/// light off; the exposure keys (a white surface facing up maps to about
/// 0.9 on screen by day, 0.7 by night with the default curve) and the sky's.
pub const AMB_SKY: f32 = 0.25;
pub const AMB_GROUND: f32 = 0.10;
pub const SKY_TINT: [f32; 3] = [0.94, 1.0, 1.08];
pub const GROUND_TINT: [f32; 3] = [1.06, 1.0, 0.92];
pub const MOON_LEVEL: f32 = 0.35;
pub const MOON_FILL: f32 = 0.25;
pub const DAY_KEY: f32 = 2.0;
pub const NIGHT_KEY: f32 = 0.7;
pub const SKY_KEY: f32 = 1.6;

/// The default exposure bias (EV) for a timecycle EV `ev`: night-range
/// EVs are lifted (EV -2 -> +1.5, exposure 0.25 -> 0.71), day
/// (EV >= -0.5) is left alone: clamp(-ev - 0.5, 0, 2).
pub fn auto_bias(ev: f32) -> f32 {
    (-ev - 0.5).clamp(0.0, 2.0)
}

static EV_BIAS: std::sync::Mutex<Option<Option<f32>>> = std::sync::Mutex::new(None);

/// Sets the bias (`xnw world-render --ev=N`); None: automatic.
pub fn set_ev_bias(b: Option<f32>) {
    if let Ok(mut g) = EV_BIAS.lock() {
        *g = Some(b);
    }
}

/// The bias asked for: [`set_ev_bias`], else `XNW_WORLD_RENDER=...,ev=+N`;
/// None: [`auto_bias`].
pub fn ev_bias() -> Option<f32> {
    let Ok(mut g) = EV_BIAS.lock() else { return None };
    *g.get_or_insert_with(|| {
        std::env::var("XNW_WORLD_RENDER").ok().and_then(|v| v.split(',').find_map(|p| p.trim().strip_prefix("ev=").and_then(|n| n.trim_start_matches('+').parse::<f32>().ok()))).filter(|b| b.is_finite()).map(|b| b.clamp(-8.0, 8.0))
    })
}

/// The constants of our lighting (the pixel shader's `Light` buffer).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lighting {
    /// Direction towards the light (unit).
    pub dir: [f32; 3],
    /// Radiance: colour x light_dir_mult.
    pub radiance: [f32; 3],
    pub mult: f32,
    /// Ambient sky (up) and ground (down) irradiance, linear.
    pub amb_sky: [f32; 3],
    pub amb_ground: [f32; 3],
    /// exp2(ev + bias): the multiplier the shader applies.
    pub exposure: f32,
    /// The timecycle's EV (794 clamped to 796..797) and our bias (EV).
    pub ev: f32,
    pub bias: f32,
    /// The timecycle's EV range, floats 796..797 (low, high): the title's
    /// b6+0x420/+0x424 [V pk8a]; -32..32 free, low = high pinned (post.rs).
    pub ev_range: [f32; 2],
    /// The directional strength (TC+0x1650) and whether the moon set is in use.
    pub strength: f32,
    pub moon: bool,
    /// The sky stand-in's colours as radiance: x this (sunlit white's level,
    /// lum(radiance) / pi, at least the ambient floor).
    pub sky_scale: f32,
    /// Hable A, B, C, D, E, F, W: timecycle floats 838..844, the title's
    /// set 1 (b6+0x4C0; the one its composite uses, post.rs).
    pub curve: [f32; 7],
    /// The title's set 0 (b6+0x4A0): timecycle floats 847..853 [V pk8a
    /// f1030: both places hold 0.22 0.3 0.1 0.2 0.01 0.3 11.2]. `post=on` only.
    pub curve_dark: [f32; 7],
    /// The fog colour (display value, the sky stand-in's horizon) for
    /// `ourlight=on`'s pass; zero otherwise.
    pub fog: [f32; 3],
}

/// Hable's curve (((x(Ax+CB)+DE)/(x(Ax+B)+DF)) - E/F).
pub fn hable(x: f32, c: &[f32; 7]) -> f32 {
    let [a, b, cc, d, e, f, _] = *c;
    (x * (a * x + cc * b) + d * e) / (x * (a * x + b) + d * f) - e / f
}

/// The display value of a linear (exposed) value: Hable / Hable(W), then 1/2.2.
pub fn display(x: f32, c: &[f32; 7]) -> f32 {
    (hable(x.max(0.0), c) / hable(c[6], c)).clamp(0.0, 1.0).powf(1.0 / 2.2)
}

fn lum(c: &[f32; 3]) -> f32 {
    0.2125 * c[0] + 0.7154 * c[1] + 0.0721 * c[2]
}

impl Lighting {
    /// From the timecycle render copy (as floats) and the sky stand-in's
    /// zenith/horizon colours (display values, as the sky stand-in makes them); every
    /// value missing or not finite takes its default. Also the table of
    /// values read, `(name, index, value, default taken, evidence)`.
    pub fn from_timecycle(tc: &[f32], sky: Option<([f32; 3], [f32; 3])>) -> (Lighting, Vec<(&'static str, usize, f32, bool, &'static str)>) {
        let mut table = Vec::with_capacity(TC.len());
        let mut v = [0f32; TC.len()];
        for (i, (name, at, def, tag)) in TC.iter().enumerate() {
            let read = tc.get(*at).copied().filter(|x| x.is_finite());
            v[i] = read.unwrap_or(*def);
            table.push((*name, *at, v[i], read.is_none(), *tag));
        }
        // The moon set when TC+0x1670 > 0, else the sun set; x the
        // directional strength TC+0x1650; zero below 1e-6 (the title's
        // directional light is then off: a test run's nights).
        let moon = v[23] > 0.0;
        let (col, mult) = if moon { ([v[24], v[25], v[26]], v[27]) } else { ([v[0], v[1], v[2]], v[3]) };
        let col = col.map(|c| c.max(0.0));
        let mult = mult.max(0.0);
        let strength = v[22].max(0.0);
        let mut curve = [v[8], v[9], v[10], v[11], v[12], v[13], v[14]];
        // A curve that cannot be evaluated (zero denominators) takes the defaults.
        if !(hable(curve[6], &curve).is_finite() && hable(curve[6], &curve) > 0.0 && curve[5] != 0.0) {
            curve = [0.22, 0.3, 0.1, 0.2, 0.01, 0.3, 4.0];
        }
        // Set 0: each value missing or not finite takes set 1's; a set that cannot be evaluated is set 1.
        let mut curve_dark: [f32; 7] = std::array::from_fn(|i| tc.get(847 + i).copied().filter(|x| x.is_finite()).unwrap_or(curve[i]));
        if !(hable(curve_dark[6], &curve_dark).is_finite() && hable(curve_dark[6], &curve_dark) > 0.0 && curve_dark[5] != 0.0) {
            curve_dark = curve;
        }
        let (lo, hi) = (v[6].min(v[7]), v[6].max(v[7]));
        let ev = (v[5] + v[21]).clamp(lo, hi);
        let bias = ev_bias().unwrap_or_else(|| auto_bias(ev));
        let exposure = (ev + bias).exp2();
        let d = [v[17], v[18], v[19]];
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let dir = if l.is_finite() && l > 0.5 { d.map(|x| x / l) } else { [0.0, 0.0, 1.0] };
        let radiance = col.map(|c| c * mult * strength);
        let radiance = if lum(&radiance) < 1e-6 { [0.0; 3] } else { radiance };
        let off = lum(&radiance) == 0.0;
        // Ambient: the sky's hue, linearised; magnitude SHADE_FRACTION of the
        // sunlit white (E / pi), at least AMBIENT_FLOOR.
        let (zen, hor) = sky.unwrap_or(([0.35, 0.45, 0.6], [0.5, 0.5, 0.5]));
        let lin = |c: [f32; 3]| c.map(|x| x.max(0.0).powf(2.2));
        let (zl, hl) = (lin(zen), lin(hor));
        let mean = ((lum(&zl) + lum(&hl)) * 0.5).max(1e-4);
        // With the directional light off, the ambient is all there is: twice the floor.
        let floor = if off { 2.0 * AMBIENT_FLOOR } else { AMBIENT_FLOOR };
        let target = (SHADE_FRACTION * lum(&radiance) / std::f32::consts::PI).max(floor / exposure);
        let k = target / mean;
        let lit = Lighting {
            dir,
            radiance,
            mult,
            amb_sky: zl.map(|x| x * k),
            amb_ground: hl.map(|x| x * k * 0.5),
            exposure,
            ev,
            bias,
            ev_range: [lo, hi],
            strength,
            moon,
            sky_scale: (lum(&radiance) / std::f32::consts::PI).max(AMBIENT_FLOOR / exposure),
            curve,
            curve_dark,
            fog: [0.0; 3],
        };
        (lit, table)
    }

    /// `ourlight=on` (LITE1): the light of our own pass from the same
    /// timecycle values ([`Lighting::from_timecycle`] reads them).
    /// - Directional: the title's final light direction (TC+0x1630, towards
    ///   it) in the hue of its colour (sun TC[0..2], or moon TC[9..11] when
    ///   the moon set is in use, TC+0x1670), at its strength TC+0x1650
    ///   (0..1; the moon set at [`MOON_LEVEL`] of it). Units are relative:
    ///   a white surface at N.L = 1 under a full sun gives 1 before exposure
    ///   (the radiance is that x pi), so the title's light_dir_mult (1 in
    ///   every capture, 14.75 by default) does not set the picture's level.
    /// - The title's directional light off (strength 0: night or interior):
    ///   a moon fill at [`MOON_FILL`] from the same direction when it is
    ///   above the horizon, else none; the ambient stays.
    /// - Ambient: a hemisphere, sky [`AMB_SKY`] x [`SKY_TINT`], ground
    ///   [`AMB_GROUND`] x [`GROUND_TINT`], x the timecycle's ambient sky
    ///   intensity TC[204] and colour mult TC[205..207] [I names]. Not the
    ///   sky stand-in's zenith: that hue (light colour x the Rayleigh beta
    ///   TC[395..397] plus a night floor (0.04, 0.06, 0.12)) is what made
    ///   [`Lighting::from_timecycle`]'s ambient deep blue.
    /// - Exposure: [`DAY_KEY`] (directional on) or [`NIGHT_KEY`] over the
    ///   level of a white surface facing up, x exp2(the `ev=` bias): a fixed
    ///   stand-in for the title's GPU exposure. The timecycle EV is not used: its
    ///   clamp is the title's on its own units (pk7a f960: min = max = -2, which
    ///   took two stops off a night already lit in relative units).
    pub fn ours(tc: &[f32], sky: Option<([f32; 3], [f32; 3])>) -> (Lighting, Vec<(&'static str, usize, f32, bool, &'static str)>) {
        let (base, table) = Lighting::from_timecycle(tc, sky);
        let read = |at: usize, def: f32| tc.get(at).copied().filter(|x| x.is_finite()).unwrap_or(def);
        let col = if base.moon { [read(9, 1.0), read(10, 1.0), read(11, 1.0)] } else { [read(0, 1.0), read(1, 1.0), read(2, 1.0)] }.map(|c| c.max(0.0));
        let hue = |c: [f32; 3]| {
            let l = lum(&c);
            if l > 1e-6 { c.map(|x| (x / l).min(3.0)) } else { [0.0; 3] }
        };
        let strength = base.strength.clamp(0.0, 1.0);
        let on = lum(&base.radiance) > 0.0 && lum(&hue(col)) > 0.0;
        let (dir_hue, level) = if on {
            (hue(col), if base.moon { MOON_LEVEL * strength } else { strength })
        } else if base.dir[2] > 0.05 {
            (hue([read(9, 1.0), read(10, 1.0), read(11, 1.0)].map(|c| c.max(0.0))), MOON_FILL)
        } else {
            ([0.0; 3], 0.0)
        };
        let dir_hue = if lum(&dir_hue) > 0.0 { dir_hue } else { [1.0; 3] };
        let amb_i = read(204, 1.0).clamp(0.0, 4.0);
        let amb_c = [read(205, 1.0), read(206, 1.0), read(207, 1.0)].map(|c| c.clamp(0.0, 4.0));
        let amb_sky: [f32; 3] = std::array::from_fn(|i| AMB_SKY * SKY_TINT[i] * amb_i * amb_c[i]);
        let amb_ground: [f32; 3] = std::array::from_fn(|i| AMB_GROUND * GROUND_TINT[i] * amb_i * amb_c[i]);
        let direct = dir_hue.map(|x| x * level);
        let white_up = lum(&direct) * base.dir[2].max(0.25) + lum(&amb_sky);
        let key = if on && !base.moon { DAY_KEY } else { NIGHT_KEY };
        let bias = ev_bias().unwrap_or(0.0);
        let exposure = key / white_up.max(1e-3) * bias.exp2();
        let lit = Lighting {
            radiance: direct.map(|x| x * std::f32::consts::PI),
            amb_sky,
            amb_ground,
            exposure,
            bias,
            sky_scale: SKY_KEY / exposure,
            fog: sky.map_or([0.0; 3], |s| s.1),
            ..base
        };
        (lit, table)
    }

    /// The shader's `Light` buffer: 7 float4 (see RAW_SHADER).
    pub fn constants(&self, old: bool) -> [f32; 28] {
        let c = &self.curve;
        [
            self.dir[0], self.dir[1], self.dir[2], 1.0,
            self.radiance[0], self.radiance[1], self.radiance[2], self.exposure,
            self.amb_sky[0], self.amb_sky[1], self.amb_sky[2], if old { 1.0 } else { 0.0 },
            self.amb_ground[0], self.amb_ground[1], self.amb_ground[2], 0.0,
            c[0], c[1], c[2], c[3],
            c[4], c[5], c[6], 1.0 / hable(c[6], c),
            self.sky_scale, self.fog[0], self.fog[1], self.fog[2],
        ]
    }

    /// The start-up line.
    pub fn line(&self) -> String {
        let c = &self.curve;
        format!(
            "world render: lighting: exposure {:.3} (EV {:.2} + bias {:.2}), directional strength {:.3} ({}), {} set, light mult {:.2}, radiance ({:.2}, {:.2}, {:.2}), dir ({:.2}, {:.2}, {:.2}), ambient sky ({:.3}, {:.3}, {:.3}), tonemap A {} B {} C {} D {} E {} F {} W {}",
            self.exposure, self.ev, self.bias, self.strength, if self.radiance == [0.0; 3] { "off" } else { "on" }, if self.moon { "moon" } else { "sun" }, self.mult, self.radiance[0], self.radiance[1], self.radiance[2], self.dir[0], self.dir[1], self.dir[2], self.amb_sky[0], self.amb_sky[1], self.amb_sky[2], c[0], c[1], c[2], c[3], c[4], c[5], c[6]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEF: [f32; 7] = [0.22, 0.3, 0.1, 0.2, 0.01, 0.3, 4.0];

    /// Hand-computed with the default curve:
    /// hable(4) = (4*(0.88+0.03)+0.002)/(4*(0.88+0.3)+0.06) - 1/30 = 3.642/4.78 - 0.033333 = 0.728592;
    /// hable(1) = (0.22+0.03+0.002)/(0.22+0.3+0.06) - 0.033333 = 0.252/0.58 - 0.033333 = 0.401149;
    /// hable(0.18) = (0.18*(0.0396+0.03)+0.002)/(0.18*(0.0396+0.3)+0.06) - 0.033333 = 0.014528/0.121128 - 0.033333 = 0.086606.
    #[test]
    fn hable_matches_hand_values() {
        assert!((hable(4.0, &DEF) - 0.728592).abs() < 1e-5);
        assert!((hable(1.0, &DEF) - 0.401149).abs() < 1e-5);
        assert!((hable(0.18, &DEF) - 0.086606).abs() < 1e-5);
        assert!(hable(0.0, &DEF).abs() < 1e-6, "black stays black");
        // display(1) = (0.401149/0.728592)^(1/2.2) = 0.550578^0.454545 = 0.762
        assert!((display(1.0, &DEF) - 0.7624).abs() < 1e-3, "{}", display(1.0, &DEF));
        assert_eq!(display(4.0, &DEF), 1.0);
        assert_eq!(display(100.0, &DEF), 1.0);
    }

    /// The moon set when TC+0x1670 > 0; strength TC+0x1650 scales the
    /// radiance; 0 switches the directional light off (a test run's nights) and
    /// the ambient floor doubles.
    #[test]
    fn moon_set_and_strength() {
        let mut tc = vec![0f32; 0x16C0 / 4];
        tc[..4].copy_from_slice(&[1.0, 0.9, 0.8, 10.0]);
        tc[9..13].copy_from_slice(&[0.5, 0.6, 0.7, 2.0]);
        (tc[0x1650 / 4], tc[0x1670 / 4]) = (0.5, 1.0);
        let (l, _) = Lighting::from_timecycle(&tc, None);
        assert!(l.moon && (l.radiance[0] - 0.5).abs() < 1e-6 && (l.radiance[2] - 0.7).abs() < 1e-6, "{:?}", l.radiance);
        tc[0x1650 / 4] = 0.0;
        let (l, _) = Lighting::from_timecycle(&tc, None);
        assert_eq!(l.radiance, [0.0; 3], "directional off");
        let lum3 = |c: [f32; 3]| 0.2125 * c[0] + 0.7154 * c[1] + 0.0721 * c[2];
        let shade = (lum3(l.amb_sky) + lum3(l.amb_ground.map(|x| x * 2.0))) * 0.5;
        assert!((shade * l.exposure - 2.0 * AMBIENT_FLOOR).abs() < 1e-4, "{shade}");
    }

    /// `ourlight=on`: no blue cast in the ambient, the sun's hue kept, the
    /// exposure from the keys; the directional off leaves the moon fill.
    #[test]
    fn ours_is_neutral_and_keyed() {
        let lum3 = |c: [f32; 3]| 0.2125 * c[0] + 0.7154 * c[1] + 0.0721 * c[2];
        let mut tc = vec![0f32; 0x16C0 / 4];
        tc[..4].copy_from_slice(&[0.89, 0.675, 0.392, 1.0]);
        tc[9..13].copy_from_slice(&[0.76, 0.9, 1.0, 0.05]);
        tc[204..208].copy_from_slice(&[1.0; 4]);
        (tc[0x1630 / 4], tc[0x1630 / 4 + 1], tc[0x1630 / 4 + 2]) = (0.07, -0.75, 0.66);
        tc[0x1650 / 4] = 1.0;
        // The a test run sky stand-in's linear zenith was (0.011, 0.049, 0.358): deep blue.
        let sky = Some(([0.13, 0.25, 0.63], [0.4, 0.45, 0.55]));
        let (l, _) = Lighting::ours(&tc, sky);
        assert!(l.amb_sky[2] / l.amb_sky[0] < 1.2, "no blue cast: {:?}", l.amb_sky);
        assert!(l.radiance[0] > l.radiance[2], "warm sun kept: {:?}", l.radiance);
        assert!((lum3(l.radiance) / std::f32::consts::PI - 1.0).abs() < 1e-4, "full sun = 1");
        let white_up = l.dir[2] + lum3(l.amb_sky);
        assert!((white_up * l.exposure - DAY_KEY).abs() < 1e-3, "{}", l.exposure);
        assert_eq!(l.fog, [0.4, 0.45, 0.55]);
        assert_eq!(&l.constants(false)[25..28], &[0.4, 0.45, 0.55]);
        // The title's directional light off: a moon fill, the night key.
        tc[0x1650 / 4] = 0.0;
        let (n, _) = Lighting::ours(&tc, sky);
        assert!((lum3(n.radiance) / std::f32::consts::PI - MOON_FILL).abs() < 1e-4, "{:?}", n.radiance);
        assert!(n.radiance[2] > n.radiance[0], "moon hue");
        assert!(((MOON_FILL * n.dir[2] + lum3(n.amb_sky)) * n.exposure - NIGHT_KEY).abs() < 1e-3);
        // Below the horizon: ambient only.
        tc[0x1630 / 4 + 2] = -0.5;
        let (d, _) = Lighting::ours(&tc, None);
        assert_eq!(d.radiance, [0.0; 3]);
        assert!(d.exposure.is_finite() && d.exposure > 0.0);
        // The old path's constants keep zeros there.
        assert_eq!(&Lighting::from_timecycle(&tc, sky).0.constants(false)[25..28], &[0.0; 3]);
    }

    #[test]
    fn night_bias() {
        assert_eq!(auto_bias(-2.0), 1.5, "night: EV -2 -> exposure 2^-0.5");
        assert_eq!(auto_bias(0.0), 0.0);
        assert_eq!(auto_bias(-5.0), 2.0);
        let mut tc = vec![0f32; 0x16C0 / 4];
        (tc[794], tc[796], tc[797]) = (0.0, -2.0, -2.0);
        let (l, _) = Lighting::from_timecycle(&tc, None);
        assert!((l.exposure - 0.5f32.exp2().recip()).abs() < 1e-6, "{}", l.exposure);
    }

    #[test]
    fn timecycle_defaults_and_reads() {
        // Nothing readable: the defaults (exposure exp2(0) = 1, mult 14.75).
        let (l, t) = Lighting::from_timecycle(&[], None);
        assert_eq!((l.exposure, l.mult, l.curve, l.curve_dark), (1.0, 14.75, DEF, DEF));
        assert!(t.iter().all(|r| r.3), "all defaults");
        let mut tc = vec![0f32; 0x16C0 / 4];
        tc[..4].copy_from_slice(&[0.89, 0.675, 0.392, 10.0]);
        (tc[794], tc[796], tc[797]) = (5.0, -3.5, 3.0); // clamped to 3
        tc[838..845].copy_from_slice(&DEF);
        tc[0x1630 / 4 + 2] = 2.0;
        tc[0x1650 / 4] = 1.0; // Directional strength
        // Set 0 at 847..853 (the title's b6+0x4A0).
        tc[847..854].copy_from_slice(&[0.22, 0.3, 0.1, 0.2, 0.02, 0.3, 11.2]);
        let (l, _) = Lighting::from_timecycle(&tc, Some(([0.2, 0.3, 0.6], [0.6, 0.6, 0.6])));
        assert_eq!((l.curve, l.curve_dark), (DEF, [0.22, 0.3, 0.1, 0.2, 0.02, 0.3, 11.2]));
        assert_eq!((l.exposure, l.ev, l.bias), (8.0, 3.0, 0.0), "day-range EV: no bias");
        assert!((l.radiance[0] - 8.9).abs() < 1e-5 && l.mult == 10.0, "colour x multiplier, no saturate");
        assert_eq!(l.dir, [0.0, 0.0, 1.0]);
        // Shade: a white surface's ambient = 0.2 x sunlit (E/pi), sky hue kept.
        let lum3 = |c: [f32; 3]| 0.2125 * c[0] + 0.7154 * c[1] + 0.0721 * c[2];
        let shade = (lum3(l.amb_sky) + lum3(l.amb_ground.map(|x| x * 2.0))) * 0.5;
        assert!((shade - SHADE_FRACTION * lum3(l.radiance) / std::f32::consts::PI).abs() < 1e-3, "{shade}");
        assert!(l.amb_sky[2] > l.amb_sky[0], "blue sky");
        // Night: a dim moon still leaves the floor.
        tc[3] = 0.05;
        let (l, _) = Lighting::from_timecycle(&tc, None);
        assert!(lum3(l.amb_sky) * l.exposure > 0.05, "the floor holds after exposure");
    }
}
