//! The heightmap terrain's tile records for our
//! world pass, read once per frame on the writer thread (as the water map).
//!
//! Tile array {u32 cap, u32 count, TerrainTileRec*} at 0x148420858, rebuilt
//! every frame by 0x143288320 (count zeroed first, then the records copied
//! from the tile objects) [V]; 372-byte records, at most 4,096 read, with
//! guarded copies. Bindless indices become slots of the title's heap at
//! `qword_14A943AA8` (its base, read with the records) [V 0x144DA90A0 xref,
//! 0x142478440 writes it]. The pass resolves and validates every slot
//! (worldrender/terrain.rs). Read here, after the frame, the array may be
//! mid-rebuild: a short count is possible [I]; the log line says count and
//! records read.
//!
//! Dumps (seq frames): `terrain.bin` (the raw records) and `terrain.txt`
//! (cap, count, heap base); `xnw world-render --seq` draws them (no texels
//! offline: flat tiles at their corner height, checkers for baked cells).

use std::sync::Arc;

use xnw_world::terrain::{MAX_TILES, TILE_BYTES, Terrain, parse};

const IMAGE_BASE: usize = 0x1_4000_0000;
const TILES: usize = 0x1_4842_0858;
const HEAP_BASE: usize = 0x1_4A94_3AA8;

fn guarded(addr: u64, len: usize) -> Option<Vec<u8>> {
    if !(0x1_0000..0x8000_0000_0000).contains(&addr) {
        return None;
    }
    let mut b = vec![0u8; len];
    crate::watch::guarded_read(addr as usize, &mut b).then_some(b)
}

fn le(b: &[u8], at: usize, n: usize) -> u64 {
    b.get(at..at + n).map_or(0, |s| s.iter().rev().fold(0u64, |v, x| v << 8 | u64::from(*x)))
}

/// The tile array is read only with `XNW_WORLD_RENDER=...,terrain=on` (off by default).
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|p| p.trim() == "terrain=on")));
    *ON
}

/// The tile array now (None: header unreadable).
pub(crate) fn now(title: usize) -> Option<Arc<Terrain>> {
    let t0 = std::time::Instant::now();
    let real = |va: usize| (title + (va - IMAGE_BASE)) as u64;
    let h = guarded(real(TILES), 16)?;
    let (cap, count, at) = (le(&h, 0, 4) as u32, le(&h, 4, 4) as u32, le(&h, 8, 8));
    let n = (count as usize).min(MAX_TILES);
    let raw = if n == 0 || at == 0 { Vec::new() } else { guarded(at, n * TILE_BYTES).unwrap_or_default() };
    let base = guarded(real(HEAP_BASE), 8).map_or(0, |b| le(&b, 0, 8));
    let mut t = parse(cap, count, raw, base);
    t.read_ms = t0.elapsed().as_secs_f64() * 1e3;
    Some(Arc::new(t))
}

/// Writes `terrain.bin` and `terrain.txt` into a seq frame directory.
pub(crate) fn dump(fdir: &std::path::Path, t: &Terrain) -> std::io::Result<()> {
    std::fs::write(fdir.join("terrain.bin"), &t.raw)?;
    std::fs::write(fdir.join("terrain.txt"), format!("cap {}\ncount {}\nheap_base {:#x}\ntiles {}\n", t.cap, t.count, t.heap_base, t.tiles.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_frame::load_terrain as load;
    use xnw_world::terrain::synthetic_record;

    #[test]
    fn dump_and_load_round_trip() {
        let dir = std::env::temp_dir().join(format!("xnw-w14-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let raw = [synthetic_record([512.0, 1024.0, 7.0], 3, 0, &[4], 1), synthetic_record([0.0, 0.0, 0.0], 5, 6, &[], 6)].concat();
        let t = parse(9, 2, raw, 0x7000_0000);
        dump(&dir, &t).unwrap();
        let back = load(&dir).unwrap();
        assert_eq!((back.cap, back.count, back.heap_base), (9, 2, 0x7000_0000));
        assert_eq!(back.tiles, t.tiles);
        // A raw dump: records only, no heap base.
        std::fs::remove_file(dir.join("terrain.bin")).unwrap();
        std::fs::create_dir_all(dir.join("blobs")).unwrap();
        std::fs::write(dir.join("blobs").join("terrain.bin"), &t.raw).unwrap();
        let w10 = load(&dir).unwrap();
        assert_eq!((w10.tiles.len(), w10.heap_base, w10.tiles[0].height), (2, 0, 0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reads_a_synthetic_array() {
        // The title's globals at their rebased addresses: a fake image whose
        // "title" base makes real(TILES) and real(HEAP_BASE) land in our buffers.
        let recs = Box::leak([synthetic_record([0.0, 0.0, 0.0], 2, 0, &[], 0), synthetic_record([512.0, 0.0, 0.0], 3, 0, &[], 2)].concat().into_boxed_slice());
        let header = Box::leak(Box::new([0u64; 2]));
        header[0] = 8 | 2 << 32;
        header[1] = recs.as_ptr() as u64;
        let title = header.as_ptr() as usize - (TILES - IMAGE_BASE);
        // The heap base qword sits at real(HEAP_BASE): unreadable here (far
        // off our buffer) or garbage; the slots then may be anything, the
        // pass validates them. Only the records are checked.
        let t = now(title).unwrap();
        assert_eq!((t.cap, t.count, t.tiles.len()), (8, 2, 2));
        assert_eq!(t.tiles[1].rows[3][0], 512.0);
        assert_eq!(t.tiles[1].lod, 2);
    }
}
