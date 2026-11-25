//! The frame pack's imposter records (kind 42, one per call of the title's
//! imposter submit) as the pass's [`Imposters`].
//!
//! Blob (apps/xnw/src/worldprobe/pack.rs `imposter`): 12 u64 {type, stage,
//! r8, r9, a5, a7, IcRec, desc, n, records at, T at, flags (bit 0 IcRec,
//! 1 desc, 2 T, 3 records read)}, then IcRec 0x100 B, desc 0x100 B, T 0x90 B,
//! records 64 B x min(n, 4096). A section that failed to read is zeros.

use std::path::Path;

use xnw_formats::framepack::{self as fp, kind, reason};
use xnw_world::imposter::{Batch, Imposters, MAX_CALLS, RECORD_BYTES, SKIP_CAP, SKIP_NO_RECORDS, SKIP_UNREADABLE};

const HEAD: usize = 96;
const DESC: usize = HEAD + 0x100;
const T: usize = DESC + 0x100;
const RECORDS: usize = T + 0x90;

fn f32_at(b: &[u8], at: usize) -> f32 {
    b.get(at..at + 4).map_or(0.0, |s| f32::from_le_bytes(s.try_into().unwrap()))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    b.get(at..at + 8).map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap()))
}

/// One kind-42 blob as a batch (None: too short or no desc).
pub fn batch_of(b: &[u8]) -> Option<Batch> {
    if b.len() < RECORDS || u64_at(b, 88) & 2 == 0 {
        return None;
    }
    let f3 = |at: usize| -> [f32; 3] { std::array::from_fn(|k| f32_at(b, at + 4 * k)) };
    let have_t = u64_at(b, 88) & 4 != 0;
    let records = b[RECORDS..].chunks_exact(RECORD_BYTES).map(|r| std::array::from_fn(|k| f32_at(r, 4 * k))).collect();
    Some(Batch {
        kind: u64_at(b, 0) as u32,
        stage: u64_at(b, 8) as u32,
        n: u64_at(b, 64) as u32,
        box_min: f3(DESC + 0x10),
        box_max: f3(DESC + 0x20),
        sphere: std::array::from_fn(|k| f32_at(b, DESC + 0x30 + 4 * k)),
        lods: if have_t { std::array::from_fn(|k| f32_at(b, T + 0x48 + 4 * k)) } else { [-1.0; 4] },
        fade: if have_t { [f32_at(b, T + 0x58), f32_at(b, T + 0x5C)] } else { [0.0; 2] },
        records,
        start: 0.0,
    })
}

/// A call's kind-47 record put into its batch: the records of the list L
/// (when kind 42 has none) and the card start distance from the type record
/// R (u32-aligned float R+0x58 [I]). The visible list behind IcRec+0x48 is
/// not used: the culling pass writes it (pk5: all zero, args instance count 0).
pub fn merge(b: &mut Batch, i: &fp::ImpInstances) {
    if b.records.is_empty() && i.reasons[fp::IMP_RECORDS] == reason::OK {
        b.records = i.records.chunks_exact(RECORD_BYTES).map(|r| std::array::from_fn(|k| f32_at(r, 4 * k))).collect();
    }
    if i.reasons[fp::IMP_TYPE] == reason::OK {
        let s = f32_at(&i.type_record, 0x58);
        if s.is_finite() && s > 0.0 && s < 10_000.0 {
            b.start = s;
        }
    }
}

/// One imposter submit call (its kind-42 blob and kind-47 instances, as the
/// pack holds them or as the live pass reads them from title memory) into
/// the frame's batches, counted: the replay and the live pass both add
/// their calls here, so they make the same batches.
pub fn add_call(im: &mut Imposters, blob: &[u8], inst: Option<&fp::ImpInstances>) {
    im.calls += 1;
    if im.calls > MAX_CALLS {
        im.skipped[SKIP_CAP] += 1;
        return;
    }
    let Some(mut b) = batch_of(blob) else {
        im.skipped[SKIP_UNREADABLE] += 1;
        return;
    };
    if let Some(i) = inst {
        merge(&mut b, i);
    }
    if b.n > 0 && b.records.is_empty() {
        im.skipped[SKIP_NO_RECORDS] += 1;
        return;
    }
    im.batches.push(b);
}

/// The frame's imposter batches from its pack (None: no pack or no record):
/// kind 42 per call, its records from kind 47 of the same call (slot).
pub fn load_imposters(fdir: &Path) -> Option<Imposters> {
    let p = crate::pack::Pack::open(fdir).ok()??;
    let inst: std::collections::HashMap<u16, fp::ImpInstances> = p.imposter_instances().into_iter().filter_map(|(s, r)| r.ok().map(|i| (s, i))).collect();
    let mut im = Imposters::default();
    for r in p.data.refs_of(fp::FRAME_SCOPE).filter(|r| r.kind == kind::IMPOSTER) {
        let blob = if reason::has_bytes(r.reason) { p.blob(r.hash).unwrap_or_default() } else { Vec::new() };
        add_call(&mut im, &blob, inst.get(&r.slot));
    }
    (im.calls > 0).then_some(im)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A made-up blob as the writer lays it out: pk4's first batch's table
    /// (box, far fade 400/420, no mesh LODs) with two records.
    fn blob(records: &[[f32; 16]], flags: u64) -> Vec<u8> {
        let mut b = Vec::new();
        for w in [0xb003939u64, 76, 3, 0, records.len() as u64, 0, 0x1000, 0x2000, records.len() as u64, 0x3000, 0x4000, flags] {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b.resize(RECORDS, 0);
        let mut put = |at: usize, v: f32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        for (k, v) in [-5.27f32, -5.0, -0.03].iter().enumerate() {
            put(DESC + 0x10 + 4 * k, *v);
        }
        for (k, v) in [4.87f32, 4.8, 14.5].iter().enumerate() {
            put(DESC + 0x20 + 4 * k, *v);
        }
        for k in 0..4 {
            put(T + 0x48 + 4 * k, -1.0);
        }
        put(T + 0x58, 400.0);
        put(T + 0x5C, 420.0);
        for r in records {
            for v in r {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn blob_parses_into_a_batch() {
        use xnw_world::imposter::synthetic_record;
        let recs = [synthetic_record([10.0, 20.0, 30.0], 1.0), synthetic_record([0.0, 300.0, 0.0], 2.0)];
        let b = batch_of(&blob(&recs, 0xF)).unwrap();
        assert_eq!((b.kind, b.stage, b.n, b.records.len()), (0xb003939, 76, 2, 2));
        assert_eq!((b.box_min, b.box_max, b.lods, b.fade), ([-5.27, -5.0, -0.03], [4.87, 4.8, 14.5], [-1.0; 4], [400.0, 420.0]));
        assert_eq!(b.records[1], recs[1]);
        // pk4 as captured: flags 7 (no records): a batch without records.
        let b = batch_of(&blob(&[], 7)).unwrap();
        assert!(b.records.is_empty());
        assert!(batch_of(&blob(&[], 1)).is_none(), "no desc");
        assert!(batch_of(&[0u8; 40]).is_none(), "short");
    }

    /// A made-up frame: two kind-42 calls (no records, as pk4) and a kind-47
    /// record for call 1 only (its list's two records, a type record).
    #[test]
    fn calls_take_their_records_from_kind_47() {
        use xnw_formats::framepack::{FramePack, Manifest, Ref};
        use xnw_world::imposter::synthetic_record;
        let root = std::env::temp_dir().join(format!("xnw-frame-imposters-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("frame-0011");
        let blobs = fp::blob_dir(&dir);
        let recs = [synthetic_record([10.0, 20.0, 30.0], 1.0), synthetic_record([0.0, 300.0, 0.0], 2.0)];
        let bytes: Vec<u8> = recs.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let inst = fp::ImpInstances { list: 0x3000, records_at: 0x3090, visible_at: 0x5000, count: 2, n: 2, visible_n: 1, visible: vec![0], records: bytes, type_record: { let mut r = vec![0u8; fp::IMP_TYPE_BYTES]; r[0x58..0x5C].copy_from_slice(&90f32.to_le_bytes()); r }, ..fp::ImpInstances::default() };
        fp::write_blob(&blobs, 60, &blob(&[], 7)).unwrap();
        fp::write_blob(&blobs, 61, &inst.encode()).unwrap();
        let r = |kind, slot, hash| Ref { owner: fp::FRAME_SCOPE, kind, slot, addr: 0x1000, len: 0, reason: 0, hash };
        let mut p = FramePack { manifest: Manifest { version: fp::VERSION, frame: 11, ..Manifest::default() }, refs: vec![r(kind::IMPOSTER, 0, 60), r(kind::IMPOSTER, 1, 60), r(kind::IMP_INST, 1, 61)], ..FramePack::default() };
        p.write(&dir).unwrap();
        let im = load_imposters(&dir).unwrap();
        let _ = std::fs::remove_dir_all(root);
        assert_eq!(im.batches.len(), 2);
        assert!(im.batches[0].records.is_empty() && im.batches[0].start == 0.0, "call 0 has no kind 47");
        assert_eq!(im.batches[1].records, recs.to_vec());
        assert_eq!(im.batches[1].start, 90.0);
    }

    /// Calls counted and skipped by reason: unreadable (no desc), n > 0
    /// without records, kept.
    #[test]
    fn calls_are_counted_and_skipped_by_reason() {
        use xnw_world::imposter::synthetic_record;
        let mut im = Imposters::default();
        add_call(&mut im, &blob(&[], 1), None);
        let mut no_recs = blob(&[], 7);
        no_recs[64..72].copy_from_slice(&3u64.to_le_bytes());
        add_call(&mut im, &no_recs, None);
        add_call(&mut im, &blob(&[synthetic_record([1.0, 2.0, 3.0], 1.0)], 0xF), None);
        add_call(&mut im, &[], None);
        assert_eq!((im.calls, im.skipped, im.batches.len()), (4, [2, 1, 0], 1));
        assert_eq!(im.skipped_text(), "skipped 3 (unreadable 2, without records 1)");
    }
}

