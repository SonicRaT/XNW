//! HAIR10: our guide ribbons from a frame pack's strand records (live
//! `hairribbon=on`, and `xnw-lab hair <frames> --ribbons` offline). For each
//! main-view strand draw (vertex program `hair::STRAND_VS`): its group
//! record (frame table slot 21, root 2 dword 3), its world (slot 18, record
//! +0x58), the record's remap and strand asset (heap indices +0xe0, +0x120:
//! kinds 52/53) and its current palette window (kind 58, the bones from
//! +0xe8 on). Every guide of the asset is posed on the CPU
//! (`xnw_world::hair::pose_guide`: `CS_UpdatePosePosition`'s 8-weight rule,
//! ed83ecd1) and becomes one ribbon ([`xnw_world::hair::push_ribbon`]),
//! as wide as the asset's guides lie apart. No strand chain, no title
//! program, no pool: the rest shape follows the head's bones (no sim sway,
//! no cut map, one colour). HAIR11: body-hair groups thin (`XNW_HAIR_BODY`),
//! guides cut at the record's cut length (`XNW_HAIR_CUT`), a cheap sway
//! ([`Motion`], `XNW_HAIR_SWAY`).

use std::collections::HashMap;

use xnw_formats::framepack::{self as fp, Ref, kind, reason};
use xnw_world::hair::{self, GROUP_BYTES, WORLD_BYTES};

use crate::pack::Pack;

/// Ribbon half widths (metres): half the guide spacing, within these.
const HALF_MIN: f32 = 0.0005;
const HALF_MAX: f32 = 0.008;
/// A posed guide root farther than this from its group's world origin is
/// not drawn (a palette window that is not the ped's, I).
const ROOT_REACH: f32 = 3.0;
/// HAIR11: body hair's half width at the root (metres): a strand, not a
/// clump of the guide spacing (2-3 mm on pk8a).
const BODY_HALF: f32 = 0.00015;

/// HAIR11 `XNW_HAIR_BODY`: body-hair groups ([`hair::group_kind`]) drawn
/// thin (default), not at all (`off`) or as HAIR10 (`wide`).
fn body_mode() -> u8 {
    static M: std::sync::LazyLock<u8> = std::sync::LazyLock::new(|| match std::env::var("XNW_HAIR_BODY").as_deref() {
        Ok("off") => 0,
        Ok("wide") => 2,
        _ => 1,
    });
    *M
}

/// HAIR11 `XNW_HAIR_CUT=off`: guides keep their whole rest length (default:
/// cut at the record's cut length, [`hair::group_kind`]).
fn cut_on() -> bool {
    static M: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_HAIR_CUT").map_or(true, |v| v != "off"));
    *M
}

/// HAIR11 `XNW_HAIR_SWAY=off`: no sway (default on, [`Motion`]).
fn sway_on() -> bool {
    static M: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_HAIR_SWAY").map_or(true, |v| v != "off"));
    *M
}

/// HAIR11: what the sway keeps across frames. The title's simulated
/// positions are written on the GPU (pool, HAIR9) and its sim compute is
/// not run (upstream: smh_sim loses the device), so a cheap sway: per group
/// an idle wind and the lag of its world origin's velocity, at the tip
/// `t^2` of the guide's length; body hair does not sway.
#[derive(Default)]
pub struct Motion {
    /// Seconds (the caller's clock).
    pub t: f32,
    /// Last frame's groups: (asset key, world origin, velocity, time).
    last: Vec<(u128, [f32; 3], [f32; 3], f32)>,
}

impl Motion {
    /// The smoothed velocity of the group of asset `key` now at `pos`
    /// (world): the nearest last-frame group of the same asset within 1 m.
    fn velocity(&self, key: u128, pos: [f32; 3]) -> [f32; 3] {
        let d2 = |a: &[f32; 3]| (0..3).map(|k| (a[k] - pos[k]).powi(2)).sum::<f32>();
        let Some(prev) = self.last.iter().filter(|e| e.0 == key && d2(&e.1) < 1.0).min_by(|a, b| d2(&a.1).total_cmp(&d2(&b.1))) else { return [0.0; 3] };
        let dt = self.t - prev.3;
        if !(dt > 1e-3 && dt < 0.5) {
            return prev.2;
        }
        std::array::from_fn(|k| prev.2[k] + 0.4 * ((pos[k] - prev.1[k]) / dt - prev.2[k]))
    }

    /// The tip offset of guide `g` (metres) for a guide of length `len`.
    fn tip(&self, vel: [f32; 3], g: u32, len: f32) -> [f32; 3] {
        let ph = g as f32 * 0.37;
        let wind = [(1.7 * self.t + ph).sin(), 0.6 * (1.3 * self.t + 0.7 * ph).cos(), 0.0];
        std::array::from_fn(|k| {
            let lag = (-vel[k] * 0.05).clamp(-0.3, 0.3);
            (0.06 * wind[k] + lag) * len
        })
    }
}

/// One frame's ribbons and what went into them.
#[derive(Default)]
pub struct Built {
    pub points: hair::Ribbons,
    /// Main-view strand draws, their distinct groups, groups posed.
    pub draws: usize,
    pub groups: usize,
    pub posed: usize,
    /// Guides drawn, refused by `pose_guide` (bone past the window, not
    /// finite), refused by the reach test.
    pub guides: usize,
    pub refused: usize,
    pub far: usize,
    /// Groups without: record, world, asset, remap, palette window.
    pub lacks: [usize; 5],
    /// Per posed group: (sink, group, guides, points a guide, half width, root bounds min/max camera-relative).
    pub rows: Vec<(u32, u32, usize, u32, f32, [f32; 3], [f32; 3])>,
    /// The camera the posed points are relative to: the first strand
    /// draw's view block position ([`view_origin`]; None: no view block);
    /// draws whose camera lies more than 1 cm from it.
    pub origin: Option<[f32; 3]>,
    pub other_cameras: usize,
    /// HAIR11: body-hair groups (drawn thin, skipped or wide by
    /// `XNW_HAIR_BODY`), guides cut at the record's cut length.
    pub body: usize,
    pub cut: usize,
}

impl Built {
    pub fn line(&self) -> String {
        format!(
            "{} strand draws, {} groups ({} body hair, {}), {} posed: {} guides ({} points, {} cut), {} refused, {} past {ROOT_REACH} m; groups without record/world/asset/remap/palette {:?}; strand camera {}{}",
            self.draws,
            self.groups,
            self.body,
            ["skipped", "thin", "wide"][usize::from(body_mode())],
            self.posed,
            self.guides,
            self.points.points.len(),
            self.cut,
            self.refused,
            self.far,
            self.lacks,
            self.origin.map_or("not read".to_string(), |o| format!("({:.2}, {:.2}, {:.2})", o[0], o[1], o[2])),
            if self.other_cameras > 0 { format!(" ({} draws of another camera)", self.other_cameras) } else { String::new() }
        )
    }
}

fn le32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
}

/// The frame's strand-record views by heap index: (heap index -> the
/// STRAND_BYTES ref of its view) and the palette windows by record slot.
fn strand_refs(p: &Pack) -> (HashMap<u32, &Ref>, HashMap<u16, &Ref>) {
    let mut views = HashMap::new();
    for r in p.data.refs.iter().filter(|r| r.kind == kind::STRAND_VIEW && reason::has_bytes(r.reason)) {
        let Some(v) = p.blob(r.hash).and_then(|b| fp::decode_slots(&b).into_iter().next()) else { continue };
        if views.contains_key(&v.slot) {
            continue;
        }
        if let Some(b) = p.data.refs_of(r.owner).find(|b| b.kind == kind::STRAND_BYTES && b.slot == r.slot && (reason::has_bytes(b.reason) || Pack::in_place(b))) {
            views.insert(v.slot, b);
        }
    }
    let mut windows = HashMap::new();
    for r in p.data.refs.iter().filter(|r| r.kind == kind::STRAND_PALETTE && r.slot % 2 == 0 && reason::has_bytes(r.reason)) {
        windows.entry(r.slot / 2).or_insert(r);
    }
    (views, windows)
}

/// The frame's ribbons. `widths`: half widths by asset (content hash, or
/// address when left in place), kept by the caller across frames; `motion`
/// the sway's state (HAIR11), its clock set by the caller.
pub fn build(p: &Pack, widths: &mut HashMap<u128, f32>, motion: &mut Motion) -> Built {
    let vs: HashMap<u64, u128> = p.data.programs.iter().filter(|g| g.stage == "vertex").map(|g| (g.pipeline, g.container)).collect();
    let (views, windows) = strand_refs(p);
    let mut b = Built::default();
    let mut seen = std::collections::HashSet::new();
    let mut guide = Vec::new();
    let mut groups_now = Vec::new();
    for s in p.data.sinks.iter().filter(|s| s.op == fp::op::DRAW && vs.get(&s.pipeline) == Some(&hair::STRAND_VS)) {
        b.draws += 1;
        let cam = view_origin(p, s.index);
        if let Some(o) = cam {
            match b.origin {
                None => b.origin = Some(o),
                Some(f) => b.other_cameras += usize::from((0..3).any(|k| (f[k] - o[k]).abs() > 0.01)),
            }
        }
        let roots = p.data.refs_of(s.index).find(|r| r.kind == kind::ROOTS && r.slot == 2 && reason::has_bytes(r.reason)).and_then(|r| p.blob(r.hash));
        let Some(gi) = roots.map(|r| le32(&r, 12)) else {
            b.lacks[0] += 1;
            continue;
        };
        if !seen.insert(gi) {
            continue;
        }
        b.groups += 1;
        let rec = p.slot_bytes(s.index, 0, hair::GROUPS_SLOT).ok().and_then(|(_, g)| g.get(gi as usize * GROUP_BYTES..(gi as usize + 1) * GROUP_BYTES).map(<[u8]>::to_vec));
        let Some(rec) = rec else {
            b.lacks[0] += 1;
            continue;
        };
        let wi = le32(&rec, 0x58) as usize;
        let world = p.slot_bytes(s.index, 0, hair::WORLDS_SLOT).ok().and_then(|(_, w)| w.get(wi * WORLD_BYTES..(wi + 1) * WORLD_BYTES).map(|w| -> [f32; 12] { std::array::from_fn(|i| f32::from_le_bytes([w[0x30 + 4 * i], w[0x31 + 4 * i], w[0x32 + 4 * i], w[0x33 + 4 * i]])) }));
        let Some(world) = world else {
            b.lacks[1] += 1;
            continue;
        };
        let asset_ref = views.get(&le32(&rec, 0x120));
        let Some((asset_ref, asset)) = asset_ref.and_then(|r| Some((*r, p.bytes_of(r)?))) else {
            b.lacks[2] += 1;
            continue;
        };
        let Some(remap) = views.get(&le32(&rec, 0xe0)).and_then(|r| p.bytes_of(r)) else {
            b.lacks[3] += 1;
            continue;
        };
        let Some(palette) = windows.get(&(gi as u16)).and_then(|r| p.bytes_of(r)) else {
            b.lacks[4] += 1;
            continue;
        };
        let key = if asset_ref.hash != 0 { asset_ref.hash } else { u128::from(asset_ref.addr) << 32 | u128::from(asset_ref.len) };
        let (body, cut) = hair::group_kind(&rec);
        b.body += usize::from(body);
        if body && body_mode() == 0 {
            continue;
        }
        let spaced = *widths.entry(key).or_insert_with(|| (0.5 * hair::guide_spacing(&asset)).clamp(HALF_MIN, HALF_MAX));
        let half = if body && body_mode() == 1 { BODY_HALF.min(spaced) } else { spaced };
        let (guides, points) = hair::guide_shape(&asset);
        let origin = [world[3], world[7], world[11]];
        let sway = sway_on() && !body;
        let vel = match cam {
            Some(c) if sway => {
                let pos = std::array::from_fn(|k| origin[k] + c[k]);
                let v = motion.velocity(key, pos);
                groups_now.push((key, pos, v, motion.t));
                v
            }
            _ => [0.0; 3],
        };
        let (mut lo, mut hi, mut drawn) = ([f32::MAX; 3], [f32::MIN; 3], 0);
        for g in 0..guides {
            guide.clear();
            if !hair::pose_guide(&asset, &remap, le32(&rec, 0xe4), &palette, &world, g, &mut guide) {
                b.refused += 1;
                continue;
            }
            let root = guide[0];
            if (0..3).map(|k| (root[k] - origin[k]) * (root[k] - origin[k])).sum::<f32>() > ROOT_REACH * ROOT_REACH {
                b.far += 1;
                continue;
            }
            for k in 0..3 {
                lo[k] = lo[k].min(root[k]);
                hi[k] = hi[k].max(root[k]);
            }
            if cut_on() && hair::cut_guide(&mut guide, cut) {
                b.cut += 1;
            }
            if sway {
                let tip = motion.tip(vel, g, hair::guide_length(&guide));
                hair::sway_guide(&mut guide, tip);
            }
            hair::push_ribbon(&mut b.points.points, &guide, half);
            drawn += 1;
        }
        b.guides += drawn;
        let end = b.points.points.len() as u32;
        let start = b.points.groups.last().map_or(0, |g| g.end);
        if end > start {
            b.points.groups.push(hair::RibbonGroup { start, end, origin });
        }
        if drawn > 0 {
            b.posed += 1;
            b.rows.push((s.index, gi, drawn, points, half, lo, hi));
        }
    }
    if sway_on() {
        motion.last = groups_now;
    }
    b
}

/// The camera a strand draw's camera-relative worlds are relative to: its
/// view block (0x1E0 B at slot 1 of a table root past the frame table):
/// +0xD0 x, y within the 32 m cell, z; +0xDC the cell as two i16 (FLK1,
/// `xnw_world::bridge_cam_delta` reads the same) [V pk8a: our list camera
/// differs from it on f1028-1029, I: a camera cut].
pub fn view_origin(p: &Pack, sink: u32) -> Option<[f32; 3]> {
    let b = p.tables(sink).into_iter().filter(|t| t.0 != 0).filter_map(|(root, _, _)| p.slot_bytes(sink, root, hair::VIEW_SLOT).ok()).find(|(_, b)| b.len() == 0x1e0)?.1;
    let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let cell = |o: usize| f32::from(i16::from_le_bytes([b[o], b[o + 1]])) * 32.0;
    Some([cell(0xdc) + f(0xd0), cell(0xde) + f(0xd4), f(0xd8)])
}

/// Moves `r` (points and group origins) from the camera at `from` to the
/// camera at `to` (both world positions): camera-relative positions plus `from - to`.
pub fn rebase(r: &mut hair::Ribbons, from: [f32; 3], to: [f32; 3]) -> f32 {
    let d = [from[0] - to[0], from[1] - to[1], from[2] - to[2]];
    for p in r.points.iter_mut() {
        for k in 0..3 {
            p[k] += d[k];
        }
    }
    for g in r.groups.iter_mut() {
        for k in 0..3 {
            g.origin[k] += d[k];
        }
    }
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// The camera-relative view-projection a strand draw reads (view block
/// +0x160, row vectors; the first table root past the frame table (root 0
/// slot 1 is the palette) whose slot 1 holds 0x1A0 to 0x1000 B): `xnw-lab
/// hair --ribbons` projects the roots with it.
pub fn draw_view_proj(p: &Pack, sink: u32) -> Option<[f32; 16]> {
    p.tables(sink).into_iter().filter(|t| t.0 != 0).filter_map(|(root, _, _)| p.slot_bytes(sink, root, hair::VIEW_SLOT).ok()).find(|(_, b)| (0x1a0..=0x1000).contains(&b.len())).map(|(_, b)| std::array::from_fn(|i| f32::from_le_bytes([b[0x160 + 4 * i], b[0x161 + 4 * i], b[0x162 + 4 * i], b[0x163 + 4 * i]])))
}

/// `xnw-lab hair <frames> --ribbons`: per frame the build line, then per
/// posed group its guides, width and root box, camera-relative and, through
/// the draw's own view-projection, in pixels of `size`.
pub fn ribbons_cmd(dir: &std::path::Path, size: (u32, u32)) -> Result<(), String> {
    let Some(p) = Pack::open(dir)? else {
        println!("{}: no pack", dir.display());
        return Ok(());
    };
    let t = std::time::Instant::now();
    let b = build(&p, &mut HashMap::new(), &mut Motion::default());
    println!("{}: ribbons: {} in {:.2} ms (asset widths measured once)", dir.display(), b.line(), t.elapsed().as_secs_f64() * 1e3);
    let px = |vp: &[f32; 16], q: [f32; 3]| {
        let c: [f32; 4] = std::array::from_fn(|j| q[0] * vp[j] + q[1] * vp[4 + j] + q[2] * vp[8 + j] + vp[12 + j]);
        ((c[0] / c[3] * 0.5 + 0.5) * size.0 as f32, (0.5 - c[1] / c[3] * 0.5) * size.1 as f32, c[2] / c[3])
    };
    for (sink, g, n, points, half, lo, hi) in &b.rows {
        let shown = draw_view_proj(&p, *sink).map(|vp| {
            let (a, z) = (px(&vp, *lo), px(&vp, *hi));
            format!("; pixels ({:.0}, {:.0}) .. ({:.0}, {:.0}), depth {:.5}..{:.5}", a.0, a.1, z.0, z.1, a.2, z.2)
        });
        println!("  sink {sink} group {g}: {n} guides x {points} points, half width {:.2} mm, roots ({:.3?}) .. ({:.3?}){}", half * 1e3, lo, hi, shown.unwrap_or_default());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebase_moves_onto_the_other_camera() {
        let mut r = hair::Ribbons { points: vec![[1.0, 2.0, 3.0, 0.5], [0.0, 0.0, 0.0, -0.5]], groups: vec![hair::RibbonGroup { start: 0, end: 2, origin: [1.0, 1.0, 1.0] }] };
        let d = rebase(&mut r, [10.0, 0.0, 5.0], [10.0, 3.0, 1.0]);
        assert_eq!(r.points, vec![[1.0, -1.0, 7.0, 0.5], [0.0, -3.0, 4.0, -0.5]]);
        assert_eq!(r.groups[0].origin, [1.0, -2.0, 5.0]);
        assert!((d - 5.0).abs() < 1e-6);
    }
}
