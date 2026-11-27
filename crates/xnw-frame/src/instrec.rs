//! `instrec` (rows, item 2): rows of the static instanced programs
//! (`*_inst` effects, drawn by a plain or indirect mesh draw, not by the
//! instanced call) take their worlds from the draw's
//! instance records, as the title's program does. The hook's world of
//! such a row is the identity at the world origin: our pass drew it
//! nowhere (geodiff, 93 `ours none` rows in 11 frames).
//!
//! The program (fe53b776 0x61c-0x6b8): instance = group / meshlets per
//! instance (meshlet info dword 0 & 0x1FFFF), record = records + 128 *
//! instance, world as [`crate::audit::inst_world`] reads it (the skin
//! wheels' record), less the camera of its view constants (+0xd0).
//!
//! One reader for the replay ([`record_instances`], over a pack's sink
//! views) and the live list sink (`worldprobe/skinhook.rs`, over the
//! title's memory): [`static_reads`]. Switch: `instrec=off` in
//! `XNW_WORLD_RENDER`, `--instrec=off` in the replay tools, live switch
//! `instrec`.

use std::collections::HashMap;

use xnw_world::{FrameList, NO_PALETTE};

use crate::audit::{InstSlots, inst_camera, inst_world};

/// The static instanced programs by the top 32 bits of the container
/// hash, with their T9 slots (byte offset / 32) read from the code:
/// meshlet info, records, view constants, and `bases` the instance table
/// (0: none, instance k is record k).
/// MS_VS_TransformD fe53b776 (standard_inst: info T9+0x40 at 0xe4, records
/// T9+0x60 at 0x648, camera T9+0xc0+0xd0 at 0x650), 3bac646d and e0f94bb7
/// (standard_emissive_layer(_pxm)_inst: the same slots, 0x6c0 / 0x718),
/// MS_VS_Transform a60b613c and MS_VS_TransformAlphaPrepass 52597401
/// (standard_alpha_inst: T9+0x60, T9+0x80 at 0x718 / 0x7cc, T9+0x100).
/// RIM1: MS_VS_TransformD 9d707828 (standard_vehicle_2lyr_wheel_inst, the
/// deform rims): info T9+0x60 (0xd0), instance table T9+0xa0 (24-B
/// entries: `s1 = T9+0xa0[24 * instance]` at 0x7a0, record = `s1 & 0xff`
/// at 0x7c4, mirror = `s1 & 0x100` at 0x7cc), records T9+0x80 (+128 *
/// record, 0x7d4), camera T9+0x120+0xd0 (0x7b8); the world is
/// [`inst_world`] (no record coarse part: 0x7f8-0x8b4), and the mirror
/// reflects the local x (0x914-0x9a4). T9+0x100 is its remap, not read here.
pub const STATIC_INST: [(u32, InstSlots); 6] = [
    (0xfe53_b776, InstSlots { meshlets: 2, records: 3, bases: 0, view: 6 }),
    (0x3bac_646d, InstSlots { meshlets: 2, records: 3, bases: 0, view: 6 }),
    (0xe0f9_4bb7, InstSlots { meshlets: 2, records: 3, bases: 0, view: 6 }),
    (0xa60b_613c, InstSlots { meshlets: 3, records: 4, bases: 0, view: 8 }),
    (0x5259_7401, InstSlots { meshlets: 3, records: 4, bases: 0, view: 8 }),
    (0x9d70_7828, InstSlots { meshlets: 3, records: 4, bases: 5, view: 9 }),
];

/// An instance-table entry's size; its dword 0 holds the record (bits
/// 0-7) and the mirror flag (bit 8).
pub const ENTRY: usize = 24;

/// Instances read per row at most.
pub const MAX: usize = 256;

/// Why a row was left as it was ([`static_reads`]'s error).
pub const WHY: [&str; 3] = ["slot views missing", "no instances", "unreadable"];

/// The slots of a mesh container hash's static instanced program.
pub fn slots_of(container: u128) -> Option<InstSlots> {
    STATIC_INST.iter().find(|t| t.0 == (container >> 96) as u32).map(|t| t.1)
}

/// The instance worlds (camera-relative, the list's 12 floats) of a
/// direct mesh draw of `groups` groups by a program with `slots` (with an
/// instance table: one per entry, its record, the mirror negating the x axis). `view`
/// gives a table slot's (address, bytes) by (root, slot); `read` the bytes
/// at an address. Err: an index of [`WHY`].
pub fn static_reads(slots: InstSlots, groups: u32, view: &dyn Fn(u16, u32) -> Option<(u64, u64)>, read: &mut dyn FnMut(u64, usize) -> Option<Vec<u8>>) -> Result<Vec<[f32; 12]>, usize> {
    let (Some(info), Some(recs), Some(cam)) = (view(1, slots.meshlets), view(1, slots.records), view(1, slots.view)) else {
        return Err(0);
    };
    let table = if slots.bases == 0 { None } else { Some(view(1, slots.bases).ok_or(0usize)?) };
    let per = read(info.0, 4).map(|b| u32::from_le_bytes(b[..4].try_into().unwrap())).ok_or(2usize)? & 0x1FFFF;
    // `groups` 0: an indirect draw (its group count is a GPU argument): every record (table entry) of the view.
    let have = table.map_or(recs.1 / 128, |t| t.1 / ENTRY as u64) as usize;
    let n = if groups == 0 { usize::MAX } else if per == 0 { 0 } else { (groups / per) as usize }.min(MAX).min(have);
    if n == 0 {
        return Err(1);
    }
    let c = read(cam.0, 0xe0).ok_or(2usize)?;
    let c = inst_camera(&c).ok_or(2usize)?;
    let Some(t) = table else {
        let r = read(recs.0, 128 * n).ok_or(2usize)?;
        return (0..n).map(|k| static_world(&r[128 * k..128 * k + 128], c)).collect::<Option<_>>().ok_or(2);
    };
    // Through the table: instance k's entry names its record and the mirror.
    let e = read(t.0, ENTRY * n).ok_or(2usize)?;
    let e: Vec<u32> = (0..n).map(|k| u32::from_le_bytes(e[ENTRY * k..ENTRY * k + 4].try_into().unwrap())).collect();
    let last = e.iter().map(|w| (w & 0xFF) as usize).max().unwrap_or(0);
    if 128 * (last as u64 + 1) > recs.1 {
        return Err(2);
    }
    let r = read(recs.0, 128 * (last + 1)).ok_or(2usize)?;
    e.iter()
        .map(|w| {
            let at = 128 * (w & 0xFF) as usize;
            let mut m = inst_world(&r[at..at + 128], c)?;
            if w & 0x100 != 0 {
                (m[0], m[1], m[2]) = (-m[0], -m[1], -m[2]);
            }
            Some(m)
        })
        .collect::<Option<_>>()
        .ok_or(2)
}

/// One record's camera-relative world: [`inst_world`] with the record's
/// coarse position added, 32 x the two i16 at +0x64 (x, y; z has none), as
/// the camera's at +0xdc (fe53b776 0x658-0x700: `(bfe(rec+0x64) -
/// bfe(view+0xdc)) << 5` added to `d3 - camera x`, `d7 - camera y`).
pub fn static_world(rec: &[u8], cam: [f64; 3]) -> Option<[f32; 12]> {
    let w = u32::from_le_bytes(rec.get(0x64..0x68)?.try_into().unwrap());
    let coarse = [32.0 * f64::from(w as u16 as i16), 32.0 * f64::from((w >> 16) as u16 as i16)];
    inst_world(rec, [cam[0] - coarse[0], cam[1] - coarse[1], cam[2]])
}

/// `--instrec=off` or `instrec=off` in `XNW_WORLD_RENDER`: the rule is not applied.
pub fn on(args: &[String]) -> bool {
    !args.iter().any(|a| a == "--instrec=off") && !std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|p| p.trim() == "instrec=off"))
}

/// The rule over a loaded list: per row without a palette or instances
/// whose sink is a direct mesh draw by a program of [`STATIC_INST`], the
/// records' worlds become the row's instances (its world the identity).
/// Prints one line.
pub fn record_instances(dir: &std::path::Path, list: &mut FrameList) -> Result<(), String> {
    use xnw_formats::framepack::op;
    let Some(pack) = crate::pack::Pack::open(dir)? else { return Ok(()) };
    let Some(draws) = pack.list_draws().filter(|v| v.len() == list.draws.len()) else { return Ok(()) };
    let (mut rows, mut placed, mut instances, mut indirect, mut why) = (0usize, 0usize, 0usize, 0usize, [0usize; 3]);
    let mut detail: Vec<String> = Vec::new();
    for i in 0..list.draws.len() {
        let r = list.draws[i];
        if r.palette != NO_PALETTE || r.inst_count != 0 {
            continue;
        }
        let slots = |s: &&xnw_formats::framepack::Sink| pack.data.programs.iter().find(|p| p.pipeline == s.pipeline && p.stage == "mesh").and_then(|p| slots_of(p.container));
        let Some((sink, slots)) = pack.data.sinks.iter().filter(|s| s.hook_draw == Some(draws[i]) && matches!(s.op, op::MESH | op::INDIRECT_MESH)).find_map(|s| slots(&s).map(|t| (s, t))) else { continue };
        rows += 1;
        indirect += usize::from(sink.op == op::INDIRECT_MESH);
        let mut views: HashMap<(u16, u32), (u64, u64)> = HashMap::new();
        let mut regions: Vec<(u64, Vec<u8>)> = Vec::new();
        for s in [slots.meshlets, slots.records, slots.view].into_iter().chain((slots.bases != 0).then_some(slots.bases)) {
            if let Ok((v, b)) = pack.slot_bytes(sink.index, 1, s) {
                views.insert((1, s), (v.addr, v.bytes.min(b.len() as u64)));
                regions.push((v.addr, b));
            }
        }
        let mut read = |at: u64, len: usize| regions.iter().find_map(|(a, b)| at.checked_sub(*a).and_then(|o| b.get(o as usize..o as usize + len)).map(<[u8]>::to_vec));
        match static_reads(slots, if sink.op == op::MESH { sink.count } else { 0 }, &|r, s| views.get(&(r, s)).copied(), &mut read) {
            Ok(worlds) => {
                if detail.len() < 12 {
                    detail.push(format!("row {i} sink {} op {} count {} records {} B: {} instances, first at {:.2} {:.2} {:.2}", sink.index, sink.op, sink.count, views.get(&(1, slots.records)).map_or(0, |v| v.1), worlds.len(), worlds[0][9], worlds[0][10], worlds[0][11]));
                }
                let first = (list.instances.len() / 12) as u32;
                for w in &worlds {
                    list.instances.extend_from_slice(w);
                }
                let d = &mut list.draws[i];
                (d.inst_first, d.inst_count) = (first, worlds.len() as u32);
                d.world = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
                placed += 1;
                instances += worlds.len();
            }
            Err(k) => why[k] += 1,
        }
    }
    println!("instance records: {rows} rows of the static instanced programs ({indirect} indirect draws), {placed} placed ({instances} instances); left: {}; {}", WHY.iter().zip(why).map(|(w, n)| format!("{w} {n}")).collect::<Vec<_>>().join(", "), detail.join("; "));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_one_world_per_instance() {
        // Info: 2 meshlets an instance; two records (identity axes, translations 10 and 20 on x); camera (4, 0, 0).
        let slots = STATIC_INST[0].1;
        let rec = |x: f32| -> Vec<u8> {
            let mut f = [0f32; 32];
            (f[0], f[5], f[10], f[3]) = (1.0, 1.0, 1.0, x);
            // Coarse x 2 (64 m), y -1.
            f[25] = f32::from_bits(0xFFFF_0002);
            f.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        let mut cam = vec![0u8; 0xe0];
        cam[0xd0..0xd4].copy_from_slice(&4f32.to_le_bytes());
        let mem: Vec<(u64, Vec<u8>)> = vec![(0x1000, 2u32.to_le_bytes().to_vec()), (0x2000, [rec(10.0), rec(20.0)].concat()), (0x3000, cam)];
        let view = |root: u16, s: u32| match (root, s) {
            (1, 2) => Some((0x1000, 4)),
            (1, 3) => Some((0x2000, 256)),
            (1, 6) => Some((0x3000, 0xe0)),
            _ => None,
        };
        let mut read = |at: u64, len: usize| mem.iter().find_map(|(a, b)| at.checked_sub(*a).and_then(|o| b.get(o as usize..o as usize + len)).map(<[u8]>::to_vec));
        let w = static_reads(slots, 4, &view, &mut read).unwrap();
        assert_eq!(w.len(), 2);
        assert_eq!((w[0][0], w[0][4], w[0][8], w[0][9], w[1][9], w[0][10]), (1.0, 1.0, 1.0, 70.0, 80.0, -32.0));
        assert_eq!(static_reads(slots, 1, &view, &mut read), Err(1), "fewer groups than one instance's meshlets");
        assert_eq!(static_reads(slots, 0, &view, &mut read).map(|w| w.len()), Ok(2), "an indirect draw: every record of the view");
        assert_eq!(static_reads(slots, 4, &|_, _| None, &mut read), Err(0));
        assert!(slots_of(0xfe53_b776u128 << 96).is_some() && slots_of(0x1234_5678u128 << 96).is_none());
    }

    /// RIM1: the deform rims' table (9d707828): entry 0 names record 2, entry 1 record 1 mirrored; no coarse part.
    #[test]
    fn reads_through_the_instance_table() {
        let slots = slots_of(0x9d70_7828u128 << 96).unwrap();
        let rec = |x: f32| -> Vec<u8> {
            let mut f = [0f32; 32];
            (f[0], f[1], f[5], f[10], f[3]) = (0.3, 0.1, 1.0, 1.0, x);
            f[25] = f32::from_bits(0xFFFF_0002);
            f.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        let mut table = vec![0u8; 2 * ENTRY];
        table[0..4].copy_from_slice(&2u32.to_le_bytes());
        table[ENTRY..ENTRY + 4].copy_from_slice(&0x101u32.to_le_bytes());
        let mut cam = vec![0u8; 0xe0];
        cam[0xd0..0xd4].copy_from_slice(&4f32.to_le_bytes());
        let mem: Vec<(u64, Vec<u8>)> = vec![(0x1000, 3u32.to_le_bytes().to_vec()), (0x2000, [rec(10.0), rec(20.0), rec(30.0)].concat()), (0x3000, cam), (0x4000, table)];
        let view = |root: u16, s: u32| match (root, s) {
            (1, 3) => Some((0x1000, 4)),
            (1, 4) => Some((0x2000, 384)),
            (1, 5) => Some((0x4000, 2 * ENTRY as u64)),
            (1, 9) => Some((0x3000, 0xe0)),
            _ => None,
        };
        let mut read = |at: u64, len: usize| mem.iter().find_map(|(a, b)| at.checked_sub(*a).and_then(|o| b.get(o as usize..o as usize + len)).map(<[u8]>::to_vec));
        let w = static_reads(slots, 0, &view, &mut read).unwrap();
        assert_eq!(w.len(), 2, "an indirect draw: every entry of the table");
        assert_eq!((w[0][0], w[0][1], w[0][9], w[0][10]), (0.3, 0.1, 26.0, 0.0), "record 2, camera subtracted, no record coarse");
        assert_eq!((w[1][0], w[1][1], w[1][4], w[1][9]), (-0.3, -0.1, 1.0, 16.0), "record 1, x axis mirrored");
        assert_eq!(static_reads(slots, 6, &view, &mut read).map(|w| w.len()), Ok(2), "6 groups, 3 meshlets an instance");
        assert_eq!(static_reads(slots, 3, &view, &mut read).map(|w| w.len()), Ok(1));
        assert_eq!(static_reads(slots, 0, &|r, s| if s == 5 { None } else { view(r, s) }, &mut read), Err(0), "table view missing");
    }
}
