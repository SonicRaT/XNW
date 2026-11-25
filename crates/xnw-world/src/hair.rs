//! Strand hair (HAIR1): the title's strand draws as data, and the strand
//! program's vertex rule on the CPU, read from its native code.
//!
//! The title draws ped hair as strands: plain `DrawInstanced` calls (sink op
//! 4) of `VS_HairStrandGeometry_8e7634e5` (container f76935e0, the main view;
//! the shadow cascades use 58f72c1e) with `PS_HairStrandVBuffer` (7e040f4d),
//! one instance per strand, two vertices per point (a ribbon), then the
//! `hair_resolve` compute pass shades the visibility buffer. None of them is a
//! hook draw, so no list row exists for them.
//!
//! What the vertex program reads (f76935e0, `out\shaders\disasm`, V):
//!
//! | where | what |
//! |---|---|
//! | root 2 constants | LOD, a width divisor scale (1.0), f16 minimum width per metre in the high half, group record index |
//! | table 8 slot 21 (+0x2a0) | group records, 336 B each ([`Group`]) |
//! | table 8 slot 18 (+0x240) | worlds, 176 B each: a 3x4 at +0x30 (camera-relative), width scale at +0x9c |
//! | table 8 slot 15 (+0x1e0) | the points: the 47 MB frame pool, 16 B a point |
//! | table 8 slot 29 (+0x3a0) | width profile image, 64 x 320, 15 steps along a strand |
//! | table 9 slot 1 (+0x20) | the view block: forward at +0xa0, camera-relative view-projection at +0x160 |
//!
//! The points are written on the GPU by the `smh` compute chain
//! (`smh_pose_N:CS_PreSkinSimGuides` and friends, the simulation, then
//! `smh_gen_N:CS_InterpolateStrands` into the pool), so with the title's GPU
//! work nulled they are zero, and so is the profile image (pk3 f1241, pk4
//! f887, pk5 f975, pk6 f1212: 0 non-zero bytes in the 16 MB captured), and
//! the program culls every strand itself (the last point's arc length at most
//! 1e-4, f76935e0 0x2d0-0x308, 0x5fc-0x614). Hair in our pass is the title's
//! own chain through the bridge (the user's direction, 2026-11-25): the smh
//! compute, then this vertex program, per draw; this module is its data.

/// The main-view strand vertex program (container hash).
pub const STRAND_VS: u128 = 0xf76935e03c6e77f2ab85441f194e4240;
/// Its pixel program (the visibility buffer).
pub const STRAND_PS: u128 = 0x7e040f4d1d02c04db8e7fb072ebafe27;
/// The shadow-cascade strand vertex program (no pixel program).
pub const STRAND_SHADOW_VS: u128 = 0x58f72c1e72ee590a7d84ce1cf6538a49;
/// Frame table (root 0) slots the main-view program reads.
pub const POINTS_SLOT: u32 = 15;
pub const WORLDS_SLOT: u32 = 18;
pub const GROUPS_SLOT: u32 = 21;
pub const PROFILE_SLOT: u32 = 29;
/// Root 1 slot of the view block.
pub const VIEW_SLOT: u32 = 1;
pub const GROUP_BYTES: usize = 336;
pub const WORLD_BYTES: usize = 176;
pub const POINT_BYTES: usize = 16;

/// IEEE half to f32.
pub fn f16(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let mag = match e {
        0 => m as f32 * (1.0 / 16_777_216.0),
        31 => {
            if m == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => f32::from_bits((e + 112) << 23 | m << 13),
    };
    if s == 1 { -mag } else { mag }
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).unwrap_or(0)
}

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_bits(u32_at(b, o))
}

/// One strand group record (336 B), the fields the vertex program reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Group {
    /// +0x50 (bit 0: the arc length is scaled by `t_scale`, else by the
    /// strand's own length).
    pub flags: u32,
    /// +0x54 bits 16-23: points a strand at LOD 0; bits 24-31 at LOD > 0.
    pub points: [u32; 2],
    /// +0x58: the world record.
    pub world: u32,
    /// +0x5c: f16 width divisor (low) and arc-length scale (high).
    pub width_div: f32,
    pub t_scale: f32,
    /// +0x60/+0x64/+0x68: per axis the box corner (f16 low) and extent (f16 high).
    pub lo: [f32; 3],
    pub ext: [f32; 3],
    /// +0x6c: LOD 0's first byte in the pool; +0x7c/+0x80 the LOD > 0 base and step.
    pub base: u32,
    pub lod_base: u32,
    pub lod_step: u32,
    /// +0x84 low half: the width profile's image row.
    pub profile_row: u32,
    /// +0x104 + +0x10c: strands (the next group's base is this one's plus
    /// strands x points x 16 B: V on pk4 f887 groups 0 -> 1 -> 2).
    pub strands: u32,
}

impl Group {
    pub fn parse(r: &[u8]) -> Group {
        let w = |o| u32_at(r, o);
        let pair = |o| (f16(w(o) as u16), f16((w(o) >> 16) as u16));
        let (width_div, t_scale) = pair(0x5c);
        let (x, y, z) = (pair(0x60), pair(0x64), pair(0x68));
        Group {
            flags: w(0x50),
            points: [(w(0x54) >> 16) & 0xff, w(0x54) >> 24],
            world: w(0x58),
            width_div,
            t_scale,
            lo: [x.0, y.0, z.0],
            ext: [x.1, y.1, z.1],
            base: w(0x6c),
            lod_base: w(0x7c),
            lod_step: w(0x80),
            profile_row: w(0x84) & 0xffff,
            strands: w(0x104).wrapping_add(w(0x10c)),
        }
    }

    /// (first byte of strand 0 in the pool, points a strand) at `lod`.
    pub fn layout(&self, lod: u32) -> (u64, u32) {
        if lod == 0 {
            (u64::from(self.base), self.points[0])
        } else {
            (u64::from(self.lod_base) + u64::from(self.lod_step) * u64::from(lod - 1), self.points[1])
        }
    }

    /// Point bytes of strand `s` at `lod`: (first byte, length).
    pub fn strand_bytes(&self, lod: u32, s: u32) -> (u64, u64) {
        let (base, n) = self.layout(lod);
        let len = u64::from(n) * POINT_BYTES as u64;
        (base + u64::from(s) * len, len)
    }

    /// A point's position in the group's space: the box corner plus the
    /// three u16 (low halves of dwords 0-2) over 65535 times the extent.
    pub fn position(&self, p: &[u8]) -> [f32; 3] {
        std::array::from_fn(|k| self.lo[k] + (u32_at(p, 4 * k) & 0xffff) as f32 * (1.0 / 65535.0) * self.ext[k])
    }
}

/// One strand draw: its root constants and its world record.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Draw {
    pub lod: u32,
    /// Root dword 1 (1.0 in every capture): multiplies `Group::width_div`.
    pub div_scale: f32,
    /// Root dword 2 high half (f16) / 256: minimum width per metre of distance.
    pub min_width: f32,
    pub group: u32,
    /// Vertices a strand (2 a point).
    pub vertices: u32,
    /// World rows (3x4, camera-relative translation in .w).
    pub world: [f32; 12],
    /// World +0x9c: the width scale.
    pub width_scale: f32,
}

impl Draw {
    /// From root 2's four dwords, the vertex count and the world record.
    pub fn new(roots: &[u32], vertices: u32, world: &[u8]) -> Draw {
        let r = |i: usize| roots.get(i).copied().unwrap_or(0);
        Draw {
            lod: r(0),
            div_scale: f32::from_bits(r(1)),
            min_width: f16((r(2) >> 16) as u16) / 256.0,
            group: r(3),
            vertices,
            world: std::array::from_fn(|i| f32_at(world, 0x30 + 4 * i)),
            width_scale: f32_at(world, 0x9c),
        }
    }
}

/// A frame's strand draws, for the census line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Census {
    /// Main-view strand draws, and shadow-cascade ones.
    pub draws: usize,
    pub shadow_draws: usize,
    /// Strands of the main-view draws (by their groups' counts).
    pub strands: u64,
    /// Strands whose point bytes lie in the captured pool, and of those
    /// with a non-zero byte.
    pub captured: u64,
    pub with_points: u64,
    /// Draws whose group or world record is missing.
    pub unreadable: usize,
}

impl Census {
    pub fn line(&self) -> String {
        format!(
            "hair: strand draws {} (shadow {}), strands {}, in the captured pool {}, with points {}, draws without records {}",
            self.draws, self.shadow_draws, self.strands, self.captured, self.with_points, self.unreadable
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(x: f32) -> u16 {
        // Exact for the values below (small integers and halves).
        let b = x.to_bits();
        if x == 0.0 {
            return 0;
        }
        (((b >> 16) & 0x8000) | ((((b >> 23) & 0xff) - 112) << 10) | ((b >> 13) & 0x3ff)) as u16
    }

    fn group_bytes() -> Vec<u8> {
        let mut r = vec![0u8; GROUP_BYTES];
        let mut put = |o: usize, v: u32| r[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(0x50, 0);
        put(0x54, 3 << 16 | 2 << 24);
        put(0x58, 1);
        put(0x5c, u32::from(h(0.5)) | u32::from(h(2.0)) << 16);
        for k in 0..3 {
            put(0x60 + 4 * k, u32::from(h(-1.0)) | u32::from(h(2.0)) << 16);
        }
        put(0x6c, 0x100);
        put(0x7c, 0x1000);
        put(0x80, 0x200);
        put(0x104, 3);
        put(0x10c, 2);
        r
    }

    #[test]
    fn half_decodes() {
        assert_eq!(f16(0x3c00), 1.0);
        assert_eq!(f16(0xc000), -2.0);
        assert_eq!(f16(0x3800), 0.5);
        assert_eq!(f16(0), 0.0);
    }

    #[test]
    fn group_layout_by_lod() {
        let g = Group::parse(&group_bytes());
        assert_eq!((g.points, g.world, g.strands), ([3, 2], 1, 5));
        assert_eq!((g.width_div, g.t_scale, g.lo, g.ext), (0.5, 2.0, [-1.0; 3], [2.0; 3]));
        assert_eq!(g.layout(0), (0x100, 3));
        assert_eq!(g.layout(2), (0x1200, 2));
        assert_eq!(g.strand_bytes(0, 2), (0x100 + 2 * 48, 48));
        let mut p = [0u8; 16];
        p[0..2].copy_from_slice(&65535u16.to_le_bytes());
        assert_eq!(g.position(&p), [1.0, -1.0, -1.0]);
    }

}
