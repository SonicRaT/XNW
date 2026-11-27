//! HAIR2: the title's strand chain through the bridge, per dispatch. The
//! strand points the hair draws read (frame table slot 15, the 47 MB pool)
//! are written on the GPU by the `smh` compute programs, which the nulled
//! GPU never runs: `smh_pose_N` (`CS_PreSkinSimNodes`, `CS_PreSkinSimGuides`,
//! `CS_UpdatePosePosition`) and `smh_gen_N:CS_InterpolateStrands`. Here each
//! captured dispatch of them runs translated, in sink order, on the software
//! adapter, with its captured root constants (root 1), frame table (root 0:
//! palette slot 1, group records slot 21, worlds slot 18, ...) and the views
//! its group records name by heap index (v2.6 `strandview`), the pool
//! (slot 36, unordered) carried from one dispatch to the next. The whole
//! chain or nothing (upstream UP3: part of it draws lines across the screen).
//!
//! The palette view (43 MB) lies past the capture's budget: the frame's
//! palette bytes are pieced together from every copy of the view (cut at
//! 16 MB), every hook draw's palette span at its title address and the
//! strand records' own bone windows (v2.9 kind 58).
//!
//! HEAD2: the chain runs on the earlier shared compute runner (`xnw_world::compute`):
//! views bound by the program's declared kind, kept on the device by title
//! address (the pool carried from one dispatch to the next), group counts
//! from kind 60. The inputs the capture holds only in part (the palette,
//! the collision shapes of slot 22 from kind 59, the head table of slot 7)
//! are put at their addresses before the first dispatch.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use xnw_formats::framepack::{self as fp, SlotView, kind, reason, view};
use xnw_world::compute as cr;
use xnw_world::preskin::{self as ps, Kind};

use xnw_core::pair::fold;

use crate::pack::Pack;

/// Frame-table (root 0) slots the chain programs read beside the hair
/// draw's: the bone palette (48 B a bone) and the pool as an unordered view.
pub const PALETTE_SLOT: u32 = 1;
pub const POOL_SLOT: u32 = 36;
/// HAIR4: the sim state `smh_sim` writes (unordered, 5.4 MB); the sim
/// records' InterpolateStrands variants read it as slot 16.
pub const SIM_SLOT: u32 = 33;
/// HAIR3: the head position table `CS_UpdatePosePosition` grows the guides
/// from (record +0x13c: its header's byte), filled on the GPU by the head
/// compute graph, zero in every capture.
pub const SCALP_SLOT: u32 = 7;
/// The collision shapes `CS_UpdatePosePosition` loops over (kind 59 spans).
pub const SPAN_SLOT: u32 = 22;
const BONE: usize = 48;

/// A strand-input program of the chain, by its name. With
/// [`sim_on`] the `smh_sim_N` dispatches (Init, Solve) run between them in
/// captured order: they write the sim state (frame table slot 33) that the
/// sim records' InterpolateStrands variants read back as slot 16.
pub fn is_chain(name: &str) -> bool {
    name.starts_with("smh_pose_") || name.starts_with("smh_gen_") || (name.starts_with("smh_sim_") && sim_on())
}

/// `smh_sim_3:CS_InitSimBuffer`: the Init of the smh_sim_3 records (pk8a
/// f1027 and pk9a f1070-1075 hold none of its dispatches).
pub const INIT_3: u128 = 0xd8f39b4be685c330a42ce9ff24e9aae6;
const INIT_3_BEFORE: &str = "smh_sim_3:CS_SolvePerVertConstraint";

/// `XNW_HAIR_SIM_INIT=on` (default off, needs [`sim_on`]): the stand-in
/// Init [`INIT_3`] before a record's first Solve when no Init ran for it.
pub fn init_on() -> bool {
    sim_on() && std::env::var("XNW_HAIR_SIM_INIT").is_ok_and(|v| v == "on")
}

/// A compute program of the shader pack by its hash, translated as
/// [`crate::compute::program`] translates a captured one.
fn standin_program(hash: u128) -> Result<crate::compute::Program, String> {
    let rpf = xnw_gpu::d3d12x::pack::default_pack().ok_or("the stand-in Init comes from shaders.rpf: set XNW_GAME_DRIVE (or XNW_SHADER_PACK)")?;
    let c = ps::container(&rpf, None, hash).ok_or_else(|| format!("{}: no container {hash:032x}", rpf.display()))?;
    let shader = xnw_shader::emit::translate_with(&c, xnw_world::bridge::options()).map_err(|e| format!("{hash:032x}: {e}"))?;
    let wave = ps::wave_size(&c).ok_or("no wave size")?;
    let uses = cr::uses(&c)?;
    let (mut named, mut w) = (HashMap::new(), 32);
    crate::bridge::named(&c, &mut named, &mut w)?;
    Ok(crate::compute::Program { shader, wave, uses, named, container: hash })
}

/// `XNW_HAIR_SIM=on` (default off, HAIR5): run the captured `smh_sim`
/// dispatches in the chain (PACK v2.10 gives them their tables).
pub fn sim_on() -> bool {
    std::env::var("XNW_HAIR_SIM").is_ok_and(|v| v == "on")
}

/// One dispatch run.
#[derive(Clone, Debug, Default)]
pub struct Step {
    pub sink: u32,
    pub name: String,
    pub groups: [u32; 3],
    pub constants: Vec<u32>,
    pub seconds: f64,
    /// Whether the dispatch wrote the pool.
    pub pool: bool,
    /// Why it was left out on purpose (its record's strands stay unposed).
    pub skipped: Option<String>,
    pub error: Option<String>,
    /// HAIR3, `CS_PreSkinSimGuides` only: the pool's guides against
    /// [`xnw_world::hair::sim_guide`] (guides, within 1e-3, the same bones
    /// at +0xe8 and +0xec, the largest difference).
    pub cpu: Option<(u32, u32, u32, f32)>,
}

/// The chain's result: the pool after the last dispatch, at its title address.
#[derive(Clone, Debug, Default)]
pub struct Chain {
    pub pool_at: u64,
    pub pool: Vec<u8>,
    pub steps: Vec<Step>,
    /// What the capture lacked and what stood in.
    pub lacks: BTreeSet<String>,
    /// Dispatches the runner dropped, by reason.
    pub dropped: BTreeMap<&'static str, u32>,
    /// The device failed: the chain ended there.
    pub removed: bool,
    /// HAIR5 ([`sim_on`]): the sim state (slot 33) after the last dispatch,
    /// at its title address; empty without the sim.
    pub sim_at: u64,
    pub sim: Vec<u8>,
    /// HAIR7 ([`run_on_device`]): the pool's length; it stays on the runner's
    /// device at `pool_at` (`pool` is empty, nothing is read back).
    pub pool_len: u64,
    pub on_device: bool,
}

impl Chain {
    pub fn line(&self) -> String {
        let ok = self.steps.iter().filter(|s| s.error.is_none() && s.skipped.is_none()).count();
        format!(
            "hair chain: {} of {} dispatches run (dropped {:?}{}), {:.2} s on the device, pool {} B at {:#x} ({})",
            ok,
            self.steps.len(),
            self.dropped,
            if self.removed { "; the device failed, the chain ended" } else { "" },
            self.steps.iter().map(|s| s.seconds).sum::<f64>(),
            self.pool_len,
            self.pool_at,
            if self.on_device { "kept on the device".to_string() } else { format!("{} non-zero after", self.pool.iter().filter(|b| **b != 0).count()) }
        )
    }
}

fn le32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
}

/// The frame's chain dispatches in sink order: (sink, groups, program, name).
pub fn dispatches(p: &Pack, names: &HashMap<u128, String>) -> Vec<(u32, u32, u128, String)> {
    let compute: HashMap<u64, u128> = p.data.programs.iter().filter(|g| g.stage == "compute").map(|g| (g.pipeline, g.container)).collect();
    p.data
        .sinks
        .iter()
        .filter(|s| s.op == fp::op::DISPATCH)
        .filter_map(|s| {
            let h = *compute.get(&s.pipeline)?;
            let n = names.get(&h).filter(|n| is_chain(n))?;
            Some((s.index, s.count, h, n.clone()))
        })
        .collect()
}

/// The palette view's bytes: the longest copy of the view in the frame,
/// then every hook draw's palette span placed at its address over it.
fn palette(p: &Pack, v: &SlotView, lacks: &mut BTreeSet<String>) -> Vec<u8> {
    let mut b = p.data.refs.iter().filter(|r| r.kind == kind::BUFFER && r.addr == v.addr && reason::has_bytes(r.reason)).max_by_key(|r| r.len).and_then(|r| p.bytes_of(r)).unwrap_or_default();
    let copied = b.len();
    let (mut spans, mut bones) = (0, 0);
    for r in p.data.refs.iter().filter(|r| r.owner & fp::HOOK != 0 && r.kind == kind::PALETTE) {
        let off = fold(r.addr).wrapping_sub(fold(v.addr));
        let Some(bytes) = p.bytes_of(r) else { continue };
        if off % BONE as u64 != 0 || off + bytes.len() as u64 > v.bytes {
            continue;
        }
        let (off, end) = (off as usize, off as usize + bytes.len());
        if b.len() < end {
            b.resize(end, 0);
        }
        b[off..end].copy_from_slice(&bytes);
        spans += 1;
        bones += bytes.len() / BONE;
    }
    // v2.9 kind 58: the bones each strand record names (from +0xe8 and +0xec).
    let (mut windows, mut wbones) = (0, 0);
    for r in p.data.refs.iter().filter(|r| r.kind == kind::STRAND_PALETTE) {
        let off = fold(r.addr).wrapping_sub(fold(v.addr));
        let Some(bytes) = p.bytes_of(r) else { continue };
        if off + bytes.len() as u64 > v.bytes {
            continue;
        }
        let (off, end) = (off as usize, off as usize + bytes.len());
        if b.len() < end {
            b.resize(end, 0);
        }
        b[off..end].copy_from_slice(&bytes);
        windows += 1;
        wbones += bytes.len() / BONE;
    }
    lacks.insert(format!("the palette (slot 1, {} B at {:#x}): no whole copy; a copy of {copied} B, {spans} hook palette spans ({bones} bones), {windows} strand palette windows ({wbones} bones) placed: {} B", v.bytes, v.addr, b.len()));
    b
}

/// Slot 22's collision shapes (182 MB, past the capture's budget) as the
/// records name them (v2.9 kind 59: rec+0xf8, count rec+0xa8 >> 24), at
/// their offsets; zeros elsewhere.
fn spans(p: &Pack, v: &SlotView, lacks: &mut BTreeSet<String>) -> Vec<u8> {
    let mut b = Vec::new();
    let mut n = 0;
    for r in p.data.refs.iter().filter(|r| r.kind == kind::STRAND_SPAN) {
        let off = fold(r.addr).wrapping_sub(fold(v.addr));
        let Some(bytes) = p.bytes_of(r).filter(|b| !b.is_empty()) else { continue };
        if off + bytes.len() as u64 > v.bytes {
            continue;
        }
        let (off, end) = (off as usize, off as usize + bytes.len());
        if b.len() < end {
            b.resize(end, 0);
        }
        b[off..end].copy_from_slice(&bytes);
        n += 1;
    }
    lacks.insert(format!("slot 22 ({} B at {:#x}, collision shapes): {n} record spans placed (kind 59), zeros elsewhere: {} B", v.bytes, v.addr, b.len()));
    b
}

/// Of the bones the group records name (base +0xe8 and previous base
/// +0xec, up to 4096 past each), how many read non-zero in `palette`.
pub fn bones_held(records: &[u8], palette: &[u8]) -> (usize, usize) {
    let (mut held, mut zero) = (0, 0);
    for rec in records.chunks_exact(xnw_world::hair::GROUP_BYTES) {
        for base in [le32(rec, 0xe8), le32(rec, 0xec)] {
            let at = base as usize * BONE;
            match palette.get(at..at + BONE) {
                Some(b) if b.iter().any(|x| *x != 0) => held += 1,
                _ => zero += 1,
            }
        }
    }
    (held, zero)
}

/// The bone remap (record word +0xe0) as the programs read it: 16-bit
/// elements (`buffer_load_format_x`, then `& 0xfff`; the capture's view
/// carries no format: R16_UINT as the pre-skin's remap, I: its bytes read
/// 0, 1, 2, ... as u16 on pk7a f960).
const REMAP: Kind = Kind::Typed { format: 57, bytes: 2 };

/// A view a group record names by heap index: a buffer (kind, bytes) or
/// a 2D texture (format, width, height, texels).
#[derive(Clone, Debug)]
pub enum Named {
    Buffer(Kind, Vec<u8>),
    Texture(u32, u32, u32, Vec<u8>),
}

/// The views the frame's group records name by heap index (kind 52/53),
/// as (heap index, view); a line in `lacks` for each without bytes. Cut
/// maps without texels take one white texel under [`scalp_on`]; strand
/// assets the frame's budget left out come from another frame ([`borrowed`]).
fn strand_heap(p: &Pack, lacks: &mut BTreeSet<String>) -> Vec<(u32, Named)> {
    let mut out: Vec<(u32, Named)> = Vec::new();
    let mut missing: Vec<(SlotView, u16)> = Vec::new();
    for r in p.data.refs.iter().filter(|r| r.kind == kind::STRAND_VIEW) {
        let (rec, k) = (r.slot / 8, r.slot % 8);
        let Some(v) = p.bytes_of(r).and_then(|b| fp::decode_slots(&b).into_iter().next()) else {
            if r.reason != reason::IMPLAUSIBLE {
                lacks.insert(format!("record {rec} word +{:#x}: index {:#x} names no view (reason {})", fp::STRAND_WORDS[usize::from(k)], r.addr, r.reason));
            }
            continue;
        };
        if out.iter().any(|(i, ..)| *i == v.slot) || v.kind == view::OTHER {
            continue;
        }
        if matches!(v.kind, view::TEXTURE | view::PLACED) {
            match p.view_texels(&v) {
                Ok(t) if t.depth <= 1 => out.push((v.slot, Named::Texture(t.format, t.width, t.height, t.rows))),
                got => {
                    let white = fp::STRAND_WORDS[usize::from(k)] == 0x128 && scalp_on();
                    if white {
                        out.push((v.slot, Named::Texture(28, 1, 1, vec![0xff; 4])));
                    }
                    lacks.insert(format!("heap index {} (record {rec} word +{:#x}): a {}x{} texture format {} without texels ({}){}", v.slot, fp::STRAND_WORDS[usize::from(k)], v.width, v.height, v.format, got.err().unwrap_or("deeper than 1"), if white { ": one white texel (uncut)" } else { ": null" }));
                }
            }
            continue;
        }
        let buffer = matches!(v.kind, view::BUFFER | view::PLACED_BUFFER);
        match p.data.refs_of(r.owner).find(|b| b.kind == kind::STRAND_BYTES && b.slot == r.slot).and_then(|b| p.bytes_of(b)) {
            Some(bytes) if buffer => out.push((v.slot, Named::Buffer(if k == 0 { REMAP } else { Kind::Raw }, bytes))),
            None if buffer && k == ASSET_WORD && !missing.iter().any(|m: &(SlotView, u16)| m.0.slot == v.slot) => missing.push((v, rec)),
            _ => {
                lacks.insert(format!("heap index {} (record {rec} word +{:#x}): a view of kind {} format {} ({} B) without bytes: null", v.slot, fp::STRAND_WORDS[usize::from(k)], v.kind, v.format, v.bytes));
            }
        }
    }
    // HAIR4: assets the frame's budget left out, from the capture's other frames.
    let mut found = borrowed(p, &missing.iter().map(|m| m.0).collect::<Vec<_>>());
    for (v, rec) in &missing {
        match found.remove(&(v.addr, v.bytes)) {
            Some((frame, bytes)) => {
                lacks.insert(format!("heap index {} (record {rec} word +0x120): a strand asset of {} B the frame does not hold: frame {frame}'s copy (same address and length)", v.slot, v.bytes));
                out.push((v.slot, Named::Buffer(Kind::Raw, bytes)));
            }
            None => {
                lacks.insert(format!("heap index {} (record {rec} word +0x120): a strand asset of {} B held by no frame of the capture: null", v.slot, v.bytes));
            }
        }
    }
    out.sort_by_key(|v| v.0);
    out
}

/// `STRAND_WORDS` index of the strand asset word (+0x120).
const ASSET_WORD: u16 = 1;

/// HAIR4: strand assets (record +0x120) the frame lacks, from the other
/// frames of the same capture (nearest first) that hold a view with the same
/// address and length: (address, length) -> (frame, bytes). The assets are
/// read-only: of pk8a's asset addresses held in more than one frame (29),
/// none holds two contents (V, f1027-1032); f1027 lacks three (82-98 KB,
/// reason 4: the frame's budget) that f1028 and f1030-1032 hold.
fn borrowed(p: &Pack, want: &[SlotView]) -> HashMap<(u64, u64), (i64, Vec<u8>)> {
    let mut got = HashMap::new();
    let frame_of = |d: &std::path::Path| d.file_name()?.to_str()?.strip_prefix("frame-")?.parse::<i64>().ok();
    let (Some(me), Some(parent)) = (frame_of(&p.dir), p.dir.parent()) else { return got };
    if want.is_empty() {
        return got;
    }
    let mut frames: Vec<(i64, std::path::PathBuf)> = std::fs::read_dir(parent).into_iter().flatten().flatten().filter_map(|e| Some((frame_of(&e.path())?, e.path()))).filter(|f| f.0 != me).collect();
    frames.sort_by_key(|f| ((f.0 - me).abs(), f.0));
    for (n, dir) in frames {
        if got.len() == want.len() {
            break;
        }
        let Ok(Some(q)) = Pack::open(&dir) else { continue };
        for r in q.data.refs.iter().filter(|r| r.kind == kind::STRAND_VIEW && r.slot % 8 == ASSET_WORD) {
            let Some(v) = q.bytes_of(r).and_then(|b| fp::decode_slots(&b).into_iter().next()) else { continue };
            if got.contains_key(&(v.addr, v.bytes)) || !want.iter().any(|w| w.addr == v.addr && w.bytes == v.bytes) {
                continue;
            }
            if let Some(b) = q.data.refs_of(r.owner).find(|b| b.kind == kind::STRAND_BYTES && b.slot == r.slot).and_then(|b| q.bytes_of(b)).filter(|b| b.len() as u64 == v.bytes) {
                got.insert((v.addr, v.bytes), (n, b));
            }
        }
    }
    got
}

/// Runs the frame's chain on `host` (the software adapter) through the
/// shared compute runner.
pub fn run(p: &Pack, host: &xnw_gpu::host::Host) -> Result<Chain, String> {
    let names = crate::audit::program_names(&p.dir).ok_or("hair chain: no shader-names.tsv above the frame")?;
    let mut runner = cr::Runner::new(host)?;
    run_on(p, &mut runner, &names, &mut Programs::new(), true)
}

/// Translated chain programs by container hash (HAIR6: the live path keeps
/// them across frames; a translation takes far longer than a frame).
pub type Programs = HashMap<u128, Result<crate::compute::Program, String>>;

/// The program names of the shader pack (`shader-names.tsv`), for a pack
/// made in memory (HAIR6).
pub fn names_in(dir: &std::path::Path) -> Option<HashMap<u128, String>> {
    crate::audit::program_names(&dir.join("x"))
}

/// [`run`] on `runner` (its kept resources forgotten first: every frame
/// starts from a zero pool, as a capture's), with the shader pack's
/// `names` and the translations in `programs` (filled as met); `cpu`:
/// each PreSkinSimGuides step's pool read back against the CPU rule.
pub fn run_on(p: &Pack, runner: &mut cr::Runner, names: &HashMap<u128, String>, programs: &mut Programs, cpu: bool) -> Result<Chain, String> {
    run_with(p, runner, names, programs, cpu, true)
}

/// HAIR7: [`run_on`] leaving the pool on the runner's device (the live
/// path: the strand draws read it there, `Runner::resource`): nothing is
/// read back (47 MB a frame).
pub fn run_on_device(p: &Pack, runner: &mut cr::Runner, names: &HashMap<u128, String>, programs: &mut Programs) -> Result<Chain, String> {
    run_with(p, runner, names, programs, false, false)
}

fn run_with(p: &Pack, runner: &mut cr::Runner, names: &HashMap<u128, String>, programs: &mut Programs, cpu: bool, read_pool: bool) -> Result<Chain, String> {
    runner.forget();
    let names = names.clone();
    let list = dispatches(p, &names);
    let first = list.first().ok_or("hair chain: no strand-input dispatch in the frame")?.0;
    let table = |sink: u32| p.tables(sink).into_iter().find(|t| t.0 == 0).map(|t| t.2).unwrap_or_default();
    let pool_view = table(first).into_iter().find(|v| v.slot == POOL_SLOT).ok_or("hair chain: no pool view at slot 36")?;
    let mut chain = Chain { pool_at: pool_view.addr, pool_len: pool_view.bytes, on_device: !read_pool, ..Chain::default() };
    chain.lacks.insert(format!("the pool (slot 36, {} B at {:#x}): an unordered view, not captured: zeros (a first frame without history)", pool_view.bytes, pool_view.addr));
    let mut heap = strand_heap(p, &mut chain.lacks);
    // The runner's heap ends at its bound (card safety); the records name
    // views past it (pk8a f1027: assets at 85264-105438). Those views move
    // to free indices below it and the records' words that name them
    // (kind 52 says which) are rewritten to match: the same view, read by
    // another number.
    let used: BTreeSet<u32> = heap.iter().map(|h| h.0).collect();
    let mut free = (0..cr::HEAP_MAX).rev().filter(|i| !used.contains(i));
    let moved: HashMap<u32, u32> = used.iter().filter(|i| **i >= cr::HEAP_MAX).filter_map(|i| Some((*i, free.next()?))).collect();
    for h in heap.iter_mut() {
        if let Some(n) = moved.get(&h.0) {
            h.0 = *n;
        }
    }
    let patch = |records: &mut Vec<u8>| -> usize {
        let mut n = 0;
        for r in p.data.refs.iter().filter(|r| r.kind == kind::STRAND_VIEW) {
            let (rec, k) = (usize::from(r.slot / 8), usize::from(r.slot % 8));
            let at = rec * xnw_world::hair::GROUP_BYTES + fp::STRAND_WORDS.get(k).copied().unwrap_or(usize::MAX);
            if let Some(&to) = moved.get(&le32(records, at)) {
                records[at..at + 4].copy_from_slice(&to.to_le_bytes());
                n += 1;
            }
        }
        n
    };
    // HAIR5 probes (default off): `XNW_HAIR_LIMIT=<s>` the runner's
    // per-submission limit (600 s), `XNW_HAIR_SKIP=<sink>,..` sinks left out.
    if let Some(s) = std::env::var("XNW_HAIR_LIMIT").ok().and_then(|v| v.trim().parse::<f64>().ok()) {
        runner.limit = s;
    }
    let skip: BTreeSet<u32> = std::env::var("XNW_HAIR_SKIP").unwrap_or_default().split(',').filter_map(|v| v.trim().parse().ok()).collect();
    // Inputs the capture holds in part, put at their addresses once (the
    // runner then binds what it holds at an address over the capture's bytes).
    let mut palettes: HashMap<u64, Vec<u8>> = HashMap::new();
    for (sink, ..) in &list {
        for v in table(*sink) {
            let k = crate::compute::key(v.addr);
            if runner.held(k).is_some() || v.bytes == 0 {
                continue;
            }
            let records = p.slot_bytes(*sink, 0, xnw_world::hair::GROUPS_SLOT).map(|r| r.1).unwrap_or_default();
            let bytes = match v.slot {
                PALETTE_SLOT => {
                    let mut b = palette(p, &v, &mut chain.lacks);
                    if std::env::var("XNW_HAIR_IDENTITY").is_ok() {
                        for (i, w) in b.chunks_exact_mut(4).enumerate() {
                            let one = matches!(i % 12, 0 | 5 | 10);
                            w.copy_from_slice(&(if one { 1.0f32 } else { 0.0 }).to_bits().to_le_bytes());
                        }
                    }
                    let (held, zero) = bones_held(&records, &b);
                    chain.lacks.insert(format!("the palette: of the records' first bones (+0xe8, +0xec) {held} read non-zero, {zero} zero"));
                    palettes.insert(v.addr, b.clone());
                    b
                }
                SPAN_SLOT => spans(p, &v, &mut chain.lacks),
                xnw_world::hair::GROUPS_SLOT if !moved.is_empty() => {
                    let mut b = records.clone();
                    let n = patch(&mut b);
                    chain.lacks.insert(format!("group records (slot 21 at {:#x}): {n} heap-index words moved below the runner's bound ({} views: {moved:?})", v.addr, moved.len()));
                    b
                }
                SCALP_SLOT => match crate::head_graph::table_for(v.addr) {
                    Some(t) => {
                        chain.lacks.insert(format!("the head table (slot 7, {} B at {:#x}): from the head graph (headgraph=on)", t.len(), v.addr));
                        t
                    }
                    None if scalp_on() => {
                        let mut b = p.slot_bytes(*sink, 0, v.slot).map(|r| r.1).unwrap_or_default();
                        scalp(p, &names, &records, &mut b, &mut chain.lacks);
                        b
                    }
                    None => continue,
                },
                _ => continue,
            };
            runner.hold(&cr::Res::Buffer(cr::Buffer { key: k, kind: Kind::Raw, uav: false, len: v.bytes, bytes: &bytes }))?;
        }
    }
    // The runner holds heap indices below its bound (card safety): a record's
    // view past it is left out (the program's read of it takes the null guard).
    for (i, n) in heap.iter().filter(|h| h.0 >= cr::HEAP_MAX) {
        let what = match n {
            Named::Buffer(k, b) => format!("a {k:?} buffer of {} B", b.len()),
            Named::Texture(f, w, h, _) => format!("a {w}x{h} texture format {f}"),
        };
        chain.lacks.insert(format!("heap index {i} ({what}): past the runner's heap bound {}: left out (null)", cr::HEAP_MAX));
    }
    let named: Vec<(u32, cr::Res)> = heap
        .iter()
        .filter(|h| h.0 < cr::HEAP_MAX)
        .map(|(i, n)| {
            let r = match n {
                Named::Buffer(kind, b) => cr::Res::Buffer(cr::Buffer { key: 0, kind: *kind, uav: false, len: b.len() as u64, bytes: b }),
                Named::Texture(format, w, h, t) => cr::Res::Image(cr::Image { key: 0, shape: cr::Shape::Tex2D, format: *format, width: *w, height: *h, depth: 1, uav: false, texels: t }),
            };
            (*i, r)
        })
        .collect();
    let pool_key = crate::compute::key(pool_view.addr);
    // A record whose strand asset (+0x120) the runner does not hold is not
    // run: pk8a f1027 record 6's UpdatePosePosition (sink 685) removed WARP
    // 1.0.22's device with its asset null (records 6, 8 and 15 lacked theirs;
    // HAIR4: borrowed from the capture's other frames, all 55 run).
    let mut records = p.slot_bytes(first, 0, xnw_world::hair::GROUPS_SLOT).map(|r| r.1).unwrap_or_default();
    patch(&mut records);
    let assetless: BTreeSet<u32> = records.chunks_exact(xnw_world::hair::GROUP_BYTES).enumerate().filter(|(_, r)| !named.iter().any(|(i, _)| *i == le32(r, 0x120))).map(|(k, _)| k as u32).collect();
    if !assetless.is_empty() {
        chain.lacks.insert(format!("records {assetless:?}: no strand asset (+0x120) held: their dispatches are not run"));
    }
    // HAIR5: records whose sim state an Init dispatch has written (or the stand-in).
    let mut inited: BTreeSet<u32> = BTreeSet::new();
    let mut standin: Option<Result<crate::compute::Program, String>> = None;
    for (sink, _, _, name) in list {
        let mut step = Step { sink, name, ..Step::default() };
        step.constants = p.data.refs_of(sink).find(|r| r.kind == kind::ROOTS && r.slot == 1).and_then(|r| p.bytes_of(r)).map(|b| b.chunks_exact(4).map(|w| le32(w, 0)).collect()).unwrap_or_default();
        if step.constants.get(3).is_some_and(|r| assetless.contains(r)) {
            step.skipped = Some("not run: its record has no strand asset".into());
            chain.steps.push(step);
            continue;
        }
        if skip.contains(&sink) {
            step.skipped = Some("not run: XNW_HAIR_SKIP".into());
            chain.steps.push(step);
            continue;
        }
        let pipeline = p.data.sinks.iter().find(|s| s.index == sink).map_or(0, |s| s.pipeline);
        let container = p.data.programs.iter().find(|g| g.pipeline == pipeline && g.stage == "compute").map_or(0, |g| g.container);
        let prog = match programs.entry(container).or_insert_with(|| crate::compute::program(p, sink)) {
            Ok(x) => x,
            Err(e) => {
                runner.refuse(cr::why::TRANSLATE, e.clone());
                step.error = Some(format!("translate: {e}"));
                chain.steps.push(step);
                continue;
            }
        };
        let record = step.constants.get(3).copied().unwrap_or(u32::MAX);
        if step.name.contains(":CS_InitSimBuffer") {
            inited.insert(record);
        }
        // HAIR5 `XNW_HAIR_SIM_INIT=on`: a smh_sim_3 Solve of a record no
        // Init has written runs after that variant's Init (d8f39b4b, in no
        // captured frame: from shaders.rpf) over the Solve's own bindings
        // (the two declare the same five: slots 15, 20-22 and 33, V) and
        // group count (I).
        if init_on() && step.name.starts_with(INIT_3_BEFORE) && inited.insert(record) {
            let mut init = Step { sink, name: format!("stand-in {INIT_3:032x} smh_sim_3:CS_InitSimBuffer (record {record})"), constants: step.constants.clone(), ..Step::default() };
            match standin.get_or_insert_with(|| standin_program(INIT_3)) {
                Err(e) => init.error = Some(format!("translate: {e}")),
                Ok(sp) => {
                    let l = crate::compute::load(p, sink, &sp.uses, &sp.named);
                    let mut j = l.job();
                    j.heap = named.clone();
                    init.groups = j.groups;
                    match runner.dispatch(&sp.shader, sp.wave, &sp.uses, &j) {
                        Ok(cr::Outcome::Dropped(d)) => init.error = Some(format!("dropped: {} ({})", d.reason, d.detail)),
                        Ok(cr::Outcome::Ran(r)) => init.seconds = r.seconds,
                        Err(e) => {
                            init.error = Some(e);
                            chain.steps.push(init);
                            chain.removed = true;
                            break;
                        }
                    }
                }
            }
            chain.steps.push(init);
        }
        let loaded = crate::compute::load(p, sink, &prog.uses, &prog.named);
        chain.lacks.extend(loaded.lacks.iter().filter(|l| !l.contains("sampler heap")).cloned());
        let mut job = loaded.job();
        job.heap = named.clone();
        step.groups = job.groups;
        match runner.dispatch(&prog.shader, prog.wave, &prog.uses, &job) {
            Ok(cr::Outcome::Dropped(d)) => step.error = Some(format!("dropped: {} ({})", d.reason, d.detail)),
            Ok(cr::Outcome::Ran(r)) => {
                step.seconds = r.seconds;
                step.pool = r.wrote.contains(&pool_key);
            }
            Err(e) => {
                // A device failure ends the chain; the pool is not read back.
                step.error = Some(e);
                chain.steps.push(step);
                chain.removed = true;
                break;
            }
        }
        if cpu && step.error.is_none() && step.name.contains("CS_PreSkinSimGuides") {
            let mut records = p.slot_bytes(sink, 0, xnw_world::hair::GROUPS_SLOT).map(|r| r.1).unwrap_or_default();
            patch(&mut records);
            let palette = table(sink).iter().find(|v| v.slot == PALETTE_SLOT).and_then(|v| palettes.get(&v.addr)).map_or(&[][..], |b| &b[..]);
            let pool = runner.read(pool_key)?;
            step.cpu = Some(guides_against_cpu(&records, step.constants.get(3).copied().unwrap_or(0), &heap, palette, &pool));
        }
        chain.steps.push(step);
    }
    chain.dropped = runner.dropped.clone();
    if sim_on()
        && let Some(v) = table(first).into_iter().find(|v| v.slot == SIM_SLOT)
    {
        chain.sim_at = v.addr;
        let k = crate::compute::key(v.addr);
        chain.sim = if runner.held(k).is_some() && !chain.removed { runner.read(k)? } else { Vec::new() };
        chain.sim.truncate(v.bytes as usize);
    }
    if !read_pool {
        return Ok(chain);
    }
    chain.pool = if runner.held(pool_key).is_some() && !chain.removed { runner.read(pool_key)? } else { vec![0; pool_view.bytes as usize] };
    chain.pool.truncate(pool_view.bytes as usize);
    Ok(chain)
}

/// `XNW_HAIR_STANDIN=on` (default off, HAIR3): stand-ins for two inputs
/// every capture lacks, without which every strand is one point: the head
/// position table (slot 7) from the frame's variant-11 pre-skin dispatches'
/// own bind positions ([`scalp`]), and the cut map the records name at
/// +0x128 (InterpolateStrands samples its green channel at the strand's
/// scalp point: length = min(guide length, lerp(+0x11c low, high, g)),
/// cb2a35e9 0x624, 0x96c, 0x1154) as one white texel: uncut strands.
pub fn scalp_on() -> bool {
    std::env::var("XNW_HAIR_STANDIN").is_ok_and(|v| v == "on")
}

/// Fills the head position table `t` (slot 7's bytes) at every record's
/// base (+0x13c) whose header reads zero, from the frame's variant-11
/// pre-skin dispatches with that base (b6+0): their vertices' bind
/// positions at their table entries ([`ps::neutral_table`]): a face without
/// blend shapes, a stand-in for the title's head compute graph.
pub fn scalp(p: &Pack, names: &HashMap<u128, String>, records: &[u8], t: &mut Vec<u8>, lacks: &mut BTreeSet<String>) {
    let bases: BTreeSet<u32> = records.chunks_exact(xnw_world::hair::GROUP_BYTES).map(|r| le32(r, 0x13c)).filter(|b| (0..7).all(|k| le32(t, *b as usize + 4 * k) == 0)).collect();
    for base in bases {
        let (mut entries, mut from) = (Vec::new(), 0);
        for d in p.dispatches().iter().filter(|d| names.get(&d.program).and_then(|n| crate::preskin::variant_of(n)) == Some(11)) {
            let Ok(inp) = crate::preskin::load(p, d) else { continue };
            if le32(&inp.b6, 0) != base || le32(&inp.b6, 0xc) == 0 {
                continue;
            }
            entries.extend(ps::neutral_table(&inp.inputs()));
            from += 1;
        }
        if entries.is_empty() {
            lacks.insert(format!("the head table (slot 7) at {base:#x}: zero, and no variant-11 dispatch names it: guides at the origin corner"));
            continue;
        }
        ps::put_table(t, base, &entries);
        lacks.insert(format!("the head table (slot 7) at {base:#x}: zero (GPU-filled): {} entries from {from} variant-11 dispatches' bind positions (no blend shape)", entries.len()));
    }
}

/// HAIR3: record `index`'s guides in `pool` against the CPU rule
/// ([`xnw_world::hair::sim_guide`]) over the same inputs: (guides, within
/// 1e-3, guides whose bones read the same at +0xe8 and +0xec, the largest
/// difference).
pub fn guides_against_cpu(records: &[u8], index: u32, heap: &[(u32, Named)], palette: &[u8], pool: &[u8]) -> (u32, u32, u32, f32) {
    let g = xnw_world::hair::GROUP_BYTES;
    let Some(rec) = records.get(index as usize * g..(index as usize + 1) * g) else { return (0, 0, 0, 0.0) };
    let by = |i: u32| heap.iter().find_map(|h| match &h.1 { Named::Buffer(_, b) if h.0 == i => Some(&b[..]), _ => None }).unwrap_or(&[][..]);
    let (asset, remap) = (by(le32(rec, 0x120)), by(le32(rec, 0xe0)));
    let (out, base, prev) = (le32(rec, 0x144) as usize, le32(rec, 0xe8) as usize, le32(rec, 0xec) as usize);
    let (mut n, mut near, mut same, mut worst) = (0, 0, 0, 0f32);
    while let Some(want) = xnw_world::hair::sim_guide(rec, asset, remap, palette, n) {
        let at = out + 12 * n as usize;
        let got: [f32; 3] = std::array::from_fn(|k| f32::from_bits(le32(pool, at + 4 * k)));
        let d = (0..3).map(|k| (got[k] - want[k]).abs()).fold(0f32, f32::max);
        worst = worst.max(d);
        near += u32::from(d < 1e-3);
        let (bo, ro) = (le32(asset, 0x10) as usize + 8 * n as usize, le32(rec, 0xe4) as usize);
        let bones: Vec<usize> = (0..8).map(|j| {
            let e = 2 * (ro + asset.get(bo + j).copied().unwrap_or(0) as usize);
            (remap.get(e..e + 2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]])) & 0xfff) as usize
        }).collect();
        same += u32::from(bones.iter().all(|k| palette.get((base + k) * BONE..(base + k + 1) * BONE) == palette.get((prev + k) * BONE..(prev + k + 1) * BONE)));
        n += 1;
    }
    (n, near, same, worst)
}

/// `xnw-lab hair <frame> --chain`: runs the chain and prints a line per
/// dispatch, the lacks and the census with the chain's pool.
pub fn chain_cmd(dir: &std::path::Path) -> Result<Chain, String> {
    let p = Pack::open(dir)?.ok_or("no pack")?;
    let host = xnw_gpu::host::Host::new(crate::replay::hw_agility().as_deref()).map_err(|e| format!("hair chain: device: {e}"))?;
    println!("hair chain: adapter {}", host.caps.adapter);
    if let Some(line) = crate::head_graph::prepare(dir, &host) {
        println!("{line}");
    }
    let c = run(&p, &host)?;
    for s in &c.steps {
        println!("  sink {} {} groups {:?} constants {:x?}: {}", s.sink, s.name, s.groups, s.constants, s.error.clone().or(s.skipped.clone()).unwrap_or_else(|| format!("{:.3} s{}", s.seconds, if s.pool { ", wrote the pool" } else { "" })));
        if let Some((n, near, same, worst)) = s.cpu {
            println!("    against the CPU rule: {near} of {n} guides within 1e-3 (largest difference {worst:.2e}); {same} with the same bones at +0xe8 and +0xec");
        }
    }
    for l in &c.lacks {
        println!("  lacks: {l}");
    }
    println!("{}", c.line());
    if sim_on() {
        let fnv = c.sim.iter().fold(0x811c_9dc5u32, |h, b| (h ^ u32::from(*b)).wrapping_mul(0x0100_0193));
        println!("hair sim: state (slot 33) {} B at {:#x}, {} non-zero after the chain, fnv {fnv:08x}", c.sim.len(), c.sim_at, c.sim.iter().filter(|b| **b != 0).count());
    }
    if std::env::var("XNW_HAIR_DUMP").is_ok() {
        for at in [0x1680000usize, 0x1688a60, 0x1689ee0, 0x168aa68, 0x2cf8740, 0x2cfc380] {
            let w: Vec<String> = (0..24).map(|i| { let x = le32(&c.pool, at + 4 * i); format!("{x:08x}={:.3}", f32::from_bits(x)) }).collect();
            println!("  pool {at:#x}: {}", w.join(" "));
        }
        // HAIR3: the scalp positions UpdatePosePosition reads (table slot 7
        // at record +0x13c: a 32 B header, scale/bias/order, then 8 B packed
        // vertices from +0x1c) and the records' boxes.
        let first = dispatches(&p, &crate::audit::program_names(&p.dir).unwrap_or_default()).first().map_or(0, |d| d.0);
        let records = p.slot_bytes(first, 0, xnw_world::hair::GROUPS_SLOT).map(|r| r.1).unwrap_or_default();
        match p.slot_bytes(first, 0, 7) {
            Ok((v, s7)) => {
                println!("  slot 7: {} B captured of {} at {:#x}, {} non-zero", s7.len(), v.bytes, v.addr, s7.iter().filter(|b| **b != 0).count());
                for (i, rec) in records.chunks_exact(xnw_world::hair::GROUP_BYTES).enumerate() {
                    let off = le32(rec, 0x13c) as usize;
                    let hdr: Vec<String> = (0..8).map(|k| format!("{:.4}", f32::from_bits(le32(&s7, off + 4 * k)))).collect();
                    let ent: Vec<String> = (0..6).map(|k| format!("{:08x}{:08x}", le32(&s7, off + 0x1c + 8 * k + 4), le32(&s7, off + 0x1c + 8 * k))).collect();
                    let boxes: Vec<String> = (0..6).map(|k| { let w = le32(rec, 0x60 + 4 * k); format!("{:.3}/{:.3}", xnw_world::hair::f16(w as u16), xnw_world::hair::f16((w >> 16) as u16)) }).collect();
                    println!("  record {i}: +0x13c {off:#x} header {} entries {}; boxes +0x60.. {}; +0xa8 {:#x} +0xfc {:#x} +0x144 {:#x} +0x11c {:.4}/{:.4} +0x128 {}", hdr.join(" "), ent.join(" "), boxes.join(" "), le32(rec, 0xa8), le32(rec, 0xfc), le32(rec, 0x144), xnw_world::hair::f16(le32(rec, 0x11c) as u16), xnw_world::hair::f16((le32(rec, 0x11c) >> 16) as u16), le32(rec, 0x128));
                }
            }
            Err(e) => println!("  slot 7: {e}"),
        }
        // Who could write it: every sink with an unordered view over it.
        if let Ok((v7, _)) = p.slot_bytes(first, 0, 7) {
            let names = crate::audit::program_names(&p.dir).unwrap_or_default();
            let prog: HashMap<u64, Vec<String>> = p.data.programs.iter().fold(HashMap::new(), |mut m, g| {
                m.entry(g.pipeline).or_default().push(names.get(&g.container).cloned().unwrap_or_else(|| format!("{:08x}", g.container >> 96)));
                m
            });
            let (lo, hi) = (fold(v7.addr), fold(v7.addr) + v7.bytes);
            for s in &p.data.sinks {
                for (root, _, views) in p.tables(s.index) {
                    for v in views.iter().filter(|v| v.layout == view::UNORDERED && fold(v.addr) < hi && fold(v.addr) + v.bytes > lo) {
                        println!("  slot 7 range: sink {} op {} root {root} slot {} {:#x}+{} {:?}", s.index, s.op, v.slot, v.addr, v.bytes, prog.get(&s.pipeline));
                    }
                }
            }
        }
    }
    // HAIR5: every record, drawn or not, at LOD 0 and 1: strands with more
    // than one point position, mean root and tip (the record's box space),
    // mean strand length.
    let first = dispatches(&p, &crate::audit::program_names(&p.dir).unwrap_or_default()).first().map_or(0, |d| d.0);
    let records = p.slot_bytes(first, 0, xnw_world::hair::GROUPS_SLOT).map(|r| r.1).unwrap_or_default();
    for (i, rec) in records.chunks_exact(xnw_world::hair::GROUP_BYTES).enumerate() {
        let g = xnw_world::hair::Group::parse(rec);
        for lod in [0, 1] {
            let (_, n) = g.layout(lod);
            let (mut open, mut root, mut tip, mut length) = (0u32, [0f32; 3], [0f32; 3], 0f32);
            for k in 0..g.strands {
                let (at, len) = g.strand_bytes(lod, k);
                let Some(b) = c.pool.get(at as usize..(at + len) as usize) else { continue };
                let pts: Vec<[f32; 3]> = b.chunks_exact(16).map(|q| g.position(q)).collect();
                if pts.iter().all(|q| *q == pts[0]) {
                    continue;
                }
                open += 1;
                for a in 0..3 {
                    root[a] += pts[0][a];
                    tip[a] += pts[pts.len() - 1][a];
                }
                length += pts.windows(2).map(|w| (0..3).map(|a| (w[1][a] - w[0][a]).powi(2)).sum::<f32>().sqrt()).sum::<f32>();
            }
            let m = open.max(1) as f32;
            println!(
                "  record {i} lod {lod} ({} strands x {n} points, world {}): {open} open; root ({:.3},{:.3},{:.3}) tip ({:.3},{:.3},{:.3}) length {:.3}",
                g.strands, g.world, root[0] / m, root[1] / m, root[2] / m, tip[0] / m, tip[1] / m, tip[2] / m, length / m
            );
        }
    }
    let (draws, census) = crate::hair::strands(&p, Some((c.pool_at, &c.pool)));
    println!("{}: with the chain's pool: {}", dir.display(), census.line());
    // HAIR3: strands whose points are not all one (the VS draws only those).
    for s in &draws {
        let (g, lod) = (&s.group, s.draw.lod);
        let open = (0..g.strands).filter(|k| {
            let (at, len) = g.strand_bytes(lod, *k);
            c.pool.get(at as usize..(at + len) as usize).is_some_and(|b| b.chunks_exact(16).any(|q| g.position(q) != g.position(&b[..16])))
        }).count();
        println!("  sink {} group {} lod {lod}: {open} of {} strands with more than one point position", s.sink, s.draw.group, g.strands);
    }
    for s in &draws {
        let (g, lod) = (&s.group, s.draw.lod);
        let (_, n) = g.layout(lod);
        for k in [0, g.strands / 2] {
            let (at, len) = g.strand_bytes(lod, k);
            let Some(b) = c.pool.get(at as usize..(at + len) as usize) else { continue };
            let pts: Vec<String> = b.chunks_exact(16).map(|q| {
                let v = g.position(q);
                format!("({:.3},{:.3},{:.3} | {:08x} {:08x} {:08x} {:08x})", v[0], v[1], v[2], le32(q, 0), le32(q, 4), le32(q, 8), le32(q, 12))
            }).collect();
            println!("  sink {} group {} lod {lod} strand {k} ({n} points at {at:#x}): {}", s.sink, s.draw.group, pts.join(" "));
        }
    }
    Ok(c)
}
