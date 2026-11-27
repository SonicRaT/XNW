//! `geomfill` (rows, item 3): the list rows a dump holds no geometry
//! for (no `blobs\k-<mesh>-<hash>.vb`), filled from the pack: the row's
//! sink holds the vertex buffer as a root view and the meshlet buffer as a
//! table view, as the title's mesh program reads them. Says per row why
//! the list writer left the geometry out, by running its two tests
//! ([`crate::decode::vb_ready`], [`crate::decode::expand_meshlets`]) over
//! the pack's bytes: both pass = the list's read of the title's memory
//! failed at its time, not the data. `--geomfill=off` leaves the list as
//! loaded.

use std::collections::{BTreeMap, HashMap};

use xnw_formats::framepack::kind;
use xnw_world::{FrameList, GeomKey};

use crate::decode::{expand_meshlets, vb_ready};
use crate::le32;

pub fn on(args: &[String]) -> bool {
    !args.iter().any(|a| a == "--geomfill=off")
}

/// Fills `geoms` for the main-view rows without geometry. Prints one line.
pub fn fill(dir: &std::path::Path, list: &FrameList, geoms: &mut HashMap<GeomKey, (Vec<u8>, Vec<u32>)>) -> Result<(), String> {
    let Some(pack) = crate::pack::Pack::open(dir)? else { return Ok(()) };
    let Some(draws) = pack.list_draws().filter(|v| v.len() == list.draws.len()) else { return Ok(()) };
    let pair = xnw_core::pair::pair();
    let same = |a: u64, at: u64| a == at || a == pair.to_gpu(at);
    let (mut rows, mut filled) = (0usize, 0usize);
    let mut why: BTreeMap<String, usize> = BTreeMap::new();
    for (i, r) in list.draws.iter().enumerate() {
        if geoms.contains_key(&r.geom) || r.geom.mesh_len == 0 {
            continue;
        }
        rows += 1;
        let Some(sink) = pack.sink_of(draws[i]) else {
            *why.entry("no sink".into()).or_default() += 1;
            continue;
        };
        let vb = pack.data.refs_of(sink).find(|x| x.kind == kind::ROOT_VIEW && same(x.addr, r.geom.vb_at)).and_then(|x| pack.bytes_of(x));
        let mesh = pack.tables(sink).into_iter().flat_map(|t| t.2.into_iter().map(move |v| (t.0, v))).find(|(_, v)| same(v.addr, r.geom.mesh_at)).and_then(|(root, v)| pack.slot_bytes(sink, root, v.slot).ok()).map(|x| x.1);
        let reason = match (&vb, &mesh) {
            (None, _) => "the pack has no vertex buffer either".to_string(),
            (_, None) => "the pack has no meshlet buffer either".to_string(),
            (Some(vb), _) if !vb_ready(vb) => format!("vertex header word {:#x} is not a ready header", le32(vb, 0)),
            (Some(vb), Some(m)) => match expand_meshlets(m) {
                Err(e) => format!("meshlets do not expand ({})", e.split(':').next().unwrap_or("")),
                Ok(idx) => {
                    geoms.insert(r.geom, (vb.clone(), idx));
                    filled += 1;
                    "the pack's bytes pass both tests (the list's own read failed)".to_string()
                }
            },
        };
        *why.entry(reason).or_default() += 1;
    }
    println!("geometry fill: {rows} rows without list geometry, {filled} filled from the pack; {}", why.iter().map(|(w, n)| format!("{n}: {w}")).collect::<Vec<_>>().join("; "));
    Ok(())
}
