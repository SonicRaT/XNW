//! HEAD1: the title's head compute graph through the bridge. A ped's face
//! (the head position table, frame-table slot 7, that the pre-skin's
//! variant 11 reads as t74/t75 and the strand pose grows guides from) is
//! written on the GPU by the head compute graph: `compute_graph_ops:*`
//! (a small network of matrix products over the rig's controls) and
//! `ped_standard_head_cs:*` (blend-shape translations, sticky lips,
//! normals). The nulled GPU never runs them, so every capture holds the
//! table zero (HAIR3). Here each captured dispatch of the graph runs
//! translated on the software adapter (the card only as a scheduled sample
//! under the GPU watchdog, `XNW_BRIDGE_HW=1`), the network first, with its captured
//! root constants and tables, through the shared compute runner (
//! `xnw_world::compute`, HEAD2): views bound by the program's declared kind
//! (typed buffer UAVs included), results kept by title address, so a later
//! dispatch reads what an earlier one wrote.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Mutex;

use xnw_formats::framepack::{self as fp, kind, reason, view};
use xnw_world::compute as cr;
use xnw_world::preskin::{self as ps, Kind};

use crate::pack::Pack;

const GPU_ALIAS: u64 = 0x4_0000_0000;

/// A head-graph program, by its name.
pub fn is_head(name: &str) -> bool {
    name.starts_with("ped_standard_head_cs:") || name.starts_with("compute_graph_ops:")
}

/// The frame's head-graph dispatches in sink order: (sink, thread, groups, program, name).
pub fn dispatches(p: &Pack, names: &HashMap<u128, String>) -> Vec<(u32, u32, u32, u128, String)> {
    let compute: HashMap<u64, u128> = p.data.programs.iter().filter(|g| g.stage == "compute").map(|g| (g.pipeline, g.container)).collect();
    p.data
        .sinks
        .iter()
        .filter(|s| s.op == fp::op::DISPATCH)
        .filter_map(|s| {
            let h = *compute.get(&s.pipeline)?;
            let n = names.get(&h).filter(|n| is_head(n))?;
            Some((s.index, s.thread, s.count, h, n.clone()))
        })
        .collect()
}

fn short(name: &str) -> &str {
    let n = name.split(':').nth(1).unwrap_or(name);
    n.strip_prefix("CS_").unwrap_or(n)
}

/// `xnw-lab head <frame> --dump`: every head-graph sink's roots and views.
pub fn dump(p: &Pack) -> Result<(), String> {
    let names = crate::audit::program_names(&p.dir).ok_or("head graph: no shader-names.tsv above the frame")?;
    let list = dispatches(p, &names);
    let compute: HashMap<u64, u128> = p.data.programs.iter().filter(|g| g.stage == "compute").map(|g| (g.pipeline, g.container)).collect();
    let indirect: Vec<String> = p.data.sinks.iter().filter(|s| s.op == fp::op::INDIRECT_DISPATCH).filter_map(|s| names.get(compute.get(&s.pipeline)?).filter(|n| is_head(n)).map(|n| format!("{} {}", s.index, short(n)))).collect();
    println!("head graph: {} dispatch sinks; {} indirect head sinks (not run) {:?}", list.len(), indirect.len(), indirect.iter().take(12).collect::<Vec<_>>());
    let mut seen: BTreeSet<u128> = BTreeSet::new();
    for (sink, thread, groups, hash, name) in &list {
        let mut line = format!("sink {sink} t{thread} g{groups} {}", short(name));
        if seen.insert(*hash) {
            if let Some(c) = p.blob(*hash) {
                let (mut used, mut w) = (HashMap::new(), 32);
                let _ = crate::bridge::named(&c, &mut used, &mut w);
                let mut u: Vec<_> = used.into_iter().collect();
                u.sort();
                line += &format!(" [{hash:032x} used {u:?}]");
                match xnw_world::preskin::translate(&c) {
                    Ok(s) => line += &format!(" ud {:?} rootviews {}/{}/{}", s.user_data.iter().map(|u| (u.root, u.slot, u.index)).collect::<Vec<_>>(), s.root_cbvs.len(), s.root_srvs.len(), s.root_uavs.len()),
                    Err(e) => line += &format!(" TRANSLATE: {e}"),
                }
            }
        }
        println!("{line}");
        for r in p.data.refs_of(*sink) {
            match r.kind {
                kind::ROOTS => {
                    let w: Vec<String> = p.bytes_of(r).unwrap_or_default().chunks_exact(4).map(|w| format!("{:x}", u32::from_le_bytes([w[0], w[1], w[2], w[3]]))).collect();
                    println!("  roots {}: {}", r.slot, w.join(" "));
                }
                kind::TABLE => {
                    for v in p.bytes_of(r).map(|b| fp::decode_slots(&b)).unwrap_or_default() {
                        let at = fp::table_slot(u32::from(r.slot), v.slot);
                        let b = p.data.refs_of(*sink).find(|b| (b.kind == kind::BUFFER || b.kind == kind::CBV) && b.slot == at);
                        println!("  table {} slot {}: kind {} layout {} addr {:#x} bytes {} fmt {} w {} -> {}", r.slot, v.slot, v.kind, v.layout, v.addr % GPU_ALIAS, v.bytes, v.format, v.width, b.map_or("no record".to_string(), |b| format!("reason {} len {} non-zero {}", b.reason, b.len, p.bytes_of(b).map_or(0, |x| x.iter().filter(|v| **v != 0).count()))));
                    }
                }
                kind::BUFFER | kind::CBV if std::env::var("XNW_HEADGRAPH_WORDS").is_ok() => {
                    let w: Vec<String> = p.bytes_of(r).unwrap_or_default().chunks_exact(4).take(32).map(|w| format!("{:x}", u32::from_le_bytes([w[0], w[1], w[2], w[3]]))).collect();
                    let all = p.bytes_of(r).unwrap_or_default();
                    let mut distinct: BTreeSet<u32> = BTreeSet::new();
                    for x in all.chunks_exact(4) {
                        distinct.insert(u32::from_le_bytes([x[0], x[1], x[2], x[3]]));
                    }
                    println!("  bytes slot {:?}: {} distinct dwords; {}", fp::table_slot_of(r.slot), distinct.len(), w.join(" "));
                }
                kind::BUFFER | kind::CBV => {}
                k => println!("  ref kind {k} slot {} addr {:#x} len {} reason {}", r.slot, r.addr, r.len, r.reason),
            }
        }
    }
    // Who else binds the graph's buffers: an outside writer (unordered view)
    // of an input the capture holds (near) zero is where its values come from.
    let head: BTreeSet<u32> = list.iter().map(|d| d.0).collect();
    let mut bufs: BTreeMap<u64, u64> = BTreeMap::new();
    for (sink, ..) in &list {
        for (_, _, vs) in p.tables(*sink) {
            for v in vs.iter().filter(|v| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER) && v.bytes > 0) {
                let e = bufs.entry(v.addr % GPU_ALIAS).or_default();
                *e = (*e).max(v.bytes);
            }
        }
    }
    let ops: HashMap<u64, u128> = p.data.programs.iter().map(|g| (g.pipeline, g.container)).collect();
    for (a, len) in &bufs {
        let mut others: BTreeMap<String, u32> = BTreeMap::new();
        for s in p.data.sinks.iter().filter(|s| !head.contains(&s.index)) {
            for (_, _, vs) in p.tables(s.index) {
                for v in vs.iter().filter(|v| v.bytes > 0 && v.addr % GPU_ALIAS < a + len && v.addr % GPU_ALIAS + v.bytes > *a) {
                    let n = ops.get(&s.pipeline).and_then(|h| names.get(h)).map_or("?", |n| n.as_str());
                    *others.entry(format!("{} {} layout {}", fp::op::name(s.op), n, v.layout)).or_default() += 1;
                }
            }
        }
        println!("buffer {a:#x} ({len} B): bound outside the graph by {}", if others.is_empty() { "none".to_string() } else { format!("{others:?}") });
    }
    Ok(())
}

/// The graph's result.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    /// One line per dispatch (and its lacks), as `compute::run_chain` made them.
    pub lines: Vec<String>,
    pub ran: u32,
    pub dropped: BTreeMap<&'static str, u32>,
    pub seconds: f64,
    pub lacks: BTreeSet<String>,
    /// The head position table (the HSS extract's output view): address and bytes.
    pub table: Option<(u64, Vec<u8>)>,
    /// The network's output the extract reads (slot 2): address, bytes, non-zero bytes, a hash.
    pub network: Option<(u64, usize, usize, u32)>,
}

impl Graph {
    pub fn line(&self) -> String {
        let t = self.table.as_ref().map_or("none".to_string(), |(a, b)| format!("{} B at {a:#x}, {} non-zero, hash {:08x}", b.len(), b.iter().filter(|x| **x != 0).count(), xnw_core::hash128(&[b]) as u32));
        let n = self.network.map_or("none".to_string(), |(a, len, nz, h)| format!("{len} B at {a:#x}, {nz} non-zero, hash {h:08x}"));
        format!("head graph: {} dispatches run, dropped {:?}, {:.2} s on the device; network output {n}; head table {t}", self.ran, self.dropped, self.seconds)
    }
}

/// The longest captured copy of the buffer at `addr` in the frame.
fn captured(p: &Pack, addr: u64) -> Vec<u8> {
    p.data.refs.iter().filter(|r| r.kind == kind::BUFFER && r.addr % GPU_ALIAS == addr && reason::has_bytes(r.reason)).max_by_key(|r| r.len).and_then(|r| p.bytes_of(r)).unwrap_or_default()
}

/// The order the graph runs in: the network's dispatches (`compute_graph_ops:*`)
/// first, then the head programs that read its output (the HSS extract
/// reads the last Gather's output, I); thread by thread in the order the
/// threads first record, sink order within a thread. By program, not by
/// thread: pk8a f1027 records the head programs' thread (4, sinks 36-41)
/// before the network's (1, sinks 89-834), pk7h f1076 the other way round.
pub fn order(list: &[(u32, u32, u32, u128, String)]) -> Vec<(u32, u32, u32, u128, String)> {
    let mut threads: Vec<u32> = Vec::new();
    for d in list {
        if !threads.contains(&d.1) {
            threads.push(d.1);
        }
    }
    let net = |d: &(u32, u32, u32, u128, String)| d.4.starts_with("compute_graph_ops:");
    let by_thread = |want: bool| threads.iter().flat_map(move |t| list.iter().filter(move |d| d.1 == *t && net(d) == want).cloned());
    by_thread(true).chain(by_thread(false)).collect()
}

/// Runs the frame's head graph on `host` (the software adapter) through
/// the shared compute runner (views bound by the program's declared
/// kind, typed buffer UAVs, results kept by title address): the dispatches
/// `keep` (sink, name) passes; `ped_standard_head_cs:*` alone leaves the
/// network's output as the capture holds it.
pub fn run(p: &Pack, host: &xnw_gpu::host::Host, keep: &dyn Fn(u32, &str) -> bool) -> Result<Graph, String> {
    let names = crate::audit::program_names(&p.dir).ok_or("head graph: no shader-names.tsv above the frame")?;
    let list = order(&dispatches(p, &names));
    if list.is_empty() {
        return Err("head graph: no head-graph dispatch sink in the frame".into());
    }
    let mut runner = cr::Runner::new(host)?;
    let mut g = Graph::default();
    let extract = list.iter().find(|d| d.4.contains("HSSGraphOutputExtract")).map(|d| d.0);
    let views = extract.and_then(|s| p.tables(s).into_iter().find(|t| t.0 == 0)).map(|t| t.2).unwrap_or_default();
    // The mapping runs once per ped; a frame without its dispatch holds it cleared.
    if let Some(sink) = extract.filter(|_| !list.iter().any(|d| d.4.contains("ComputeBlendshapeMappings"))) {
        mappings(p, sink, &views, &mut runner, &mut g)?;
    }
    let reads = std::env::var("XNW_HEADGRAPH_READS").is_ok();
    let mut programs: HashMap<u64, Result<crate::compute::Program, String>> = HashMap::new();
    for (sink, _, _, _, name) in list.iter().filter(|d| keep(d.0, &d.4)) {
        let pipeline = p.data.sinks.iter().find(|s| s.index == *sink).map_or(0, |s| s.pipeline);
        let prog = match programs.entry(pipeline).or_insert_with(|| crate::compute::program(p, *sink)) {
            Ok(x) => x,
            Err(e) => {
                let d = runner.refuse(cr::why::TRANSLATE, e.clone());
                g.lines.push(format!("sink {sink} {}: dropped: {} ({})", short(name), d.reason, d.detail));
                continue;
            }
        };
        let loaded = crate::compute::load(p, *sink, &prog.uses, &prog.named);
        let mut job = loaded.job();
        if name.starts_with("compute_graph_ops:") {
            d16_as_float(&mut job);
        }
        let head = format!("sink {sink} {} groups {:?}", short(name), job.groups);
        match runner.dispatch(&prog.shader, prog.wave, &prog.uses, &job).map_err(|e| format!("{head}: {e}"))? {
            cr::Outcome::Dropped(d) => g.lines.push(format!("{head}: dropped: {} ({})", d.reason, d.detail)),
            cr::Outcome::Ran(r) => {
                let mut wrote: Vec<String> = Vec::new();
                for k in &r.wrote {
                    let nz = if reads {
                        let b = runner.read(*k)?;
                        format!("{}, hash {:08x}", b.iter().filter(|x| **x != 0).count(), xnw_core::hash128(&[&b]) as u32)
                    } else {
                        "?".into()
                    };
                    wrote.push(format!("{k:#x} ({nz} non-zero)"));
                }
                g.lines.push(format!("{head}: ran in {:.3} s; read kept {:x?}; wrote [{}]", r.seconds, r.read_kept, wrote.join(", ")));
                // `XNW_HEADGRAPH_REPEAT=<sink>:<n>`: the sink run n more times on the same inputs, its written buffers hashed each time.
                if let Some(n) = std::env::var("XNW_HEADGRAPH_REPEAT").ok().and_then(|v| v.split_once(':').and_then(|(a, b)| (a.parse() == Ok(*sink)).then(|| b.parse::<u32>().ok()).flatten())) {
                    g.lines.push(format!("repeat sink {sink}: host {}, dxil {:08x}", psv_text(&prog.shader.dxil), xnw_core::hash128(&[&prog.shader.dxil]) as u32));
                    if let Ok(f) = std::env::var("XNW_HEADGRAPH_DXIL") {
                        std::fs::write(&f, &prog.shader.dxil).map_err(|e| format!("{f}: {e}"))?;
                    }
                    let dx: BTreeSet<u32> = (0..n).filter_map(|_| crate::compute::program(p, *sink).ok()).map(|q| xnw_core::hash128(&[&q.shader.dxil]) as u32).collect();
                    g.lines.push(format!("repeat sink {sink}: {n} more translations give dxil {dx:08x?}"));
                    for i in 0..n {
                        runner.dispatch(&prog.shader, prog.wave, &prog.uses, &job).map_err(|e| format!("{head}: {e}"))?;
                        let mut hs = Vec::new();
                        for k in &r.wrote {
                            hs.push(format!("{k:#x} {:08x}", xnw_core::hash128(&[&runner.read(*k)?]) as u32));
                        }
                        g.lines.push(format!("repeat sink {sink} #{i}: {}", hs.join(", ")));
                    }
                }
                // `XNW_HEADGRAPH_ONEHOT=<sink>:<key>:<in>:<n>:<out>:<rows>:<file>` (hex key and byte offsets):
                // the sink run once per unit input vector (f16 1.0 at half k of n at byte <in> of buffer <key>, the rest of
                // the vector 0); the <rows> output halves at byte <out> of each run written to <file> (n x rows f16)..
                if let Some(v) = std::env::var("XNW_HEADGRAPH_ONEHOT").ok().filter(|v| v.split(':').next().is_some_and(|a| a.parse() == Ok(*sink))) {
                    let f: Vec<&str> = v.split(':').collect();
                    let hex = |i: usize| f.get(i).and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()).ok_or_else(|| format!("XNW_HEADGRAPH_ONEHOT: field {i}"));
                    let (key, at, n, out, rows) = (hex(1)?, hex(2)? as usize, hex(3)? as usize, hex(4)? as usize, hex(5)? as usize);
                    let file = f.get(6..).map(|s| s.join(":")).ok_or("XNW_HEADGRAPH_ONEHOT: file")?;
                    let base = runner.read(key)?;
                    let mut m = Vec::new();
                    for k in 0..n {
                        let mut b = base.clone();
                        b[at..at + 2 * n].fill(0);
                        b[at + 2 * k..at + 2 * k + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
                        runner.hold(&cr::Res::Buffer(cr::Buffer { key, kind: Kind::Raw, uav: true, len: b.len() as u64, bytes: &b }))?;
                        runner.dispatch(&prog.shader, prog.wave, &prog.uses, &job).map_err(|e| format!("{head}: {e}"))?;
                        m.extend_from_slice(&runner.read(key)?[out..out + 2 * rows]);
                    }
                    runner.hold(&cr::Res::Buffer(cr::Buffer { key, kind: Kind::Raw, uav: true, len: base.len() as u64, bytes: &base }))?;
                    std::fs::write(&file, &m).map_err(|e| format!("{file}: {e}"))?;
                    g.lines.push(format!("one-hot sink {sink}: {n} runs, {rows} rows each, to {file}"));
                }
                // `XNW_HEADGRAPH_STOP=<sink>:<file>`: every buffer this sink wrote saved (<file>.<address>), and the chain ends.
                if let Some((at, file)) = std::env::var("XNW_HEADGRAPH_STOP").ok().and_then(|v| v.split_once(':').map(|(a, b)| (a.to_string(), b.to_string()))).filter(|(a, _)| a.parse() == Ok(*sink)) {
                    for k in &r.wrote {
                        std::fs::write(format!("{file}.{k:x}"), runner.read(*k)?).map_err(|e| format!("{file}: {e}"))?;
                    }
                    g.lines.push(format!("stopped after sink {at}: {:x?} saved to {file}.<address>", r.wrote));
                    break;
                }
            }
        }
        g.lacks.extend(loaded.lacks.into_iter().filter(|l| !l.contains("sampler heap")).map(|l| format!("{}: {l}", short(name))));
    }
    (g.ran, g.dropped, g.seconds) = (runner.ran, runner.dropped.clone(), runner.seconds);
    // The network's output (the extract's slot 2) as the head programs read it.
    if let Some(v) = views.iter().find(|v| v.slot == 2) {
        let k = crate::compute::key(v.addr);
        if runner.held(k).is_some() {
            let b = runner.read(k)?;
            g.network = Some((k, b.len(), b.iter().filter(|x| **x != 0).count(), xnw_core::hash128(&[&b]) as u32));
        }
    }
    if let Some(v) = views.iter().find(|v| v.slot == TABLE_OUT) {
        let k = crate::compute::key(v.addr);
        if runner.held(k).is_some() {
            let mut b = runner.read(k)?;
            b.truncate(views_len(p, k).max(v.bytes) as usize);
            g.table = Some((k, b));
        }
    }
    Ok(g)
}

/// The network's programs read and write their buffers only through
/// 16-bit typed accesses (`buffer_load/store_format_d16_*`: every
/// `compute_graph_ops` program in the pack, 0 others), which
/// the translator lowers as float loads and stores of halves (f16 <-> f32).
/// The title views those buffers in integer formats (R16G16B16A16_UINT 12,
/// R16_UINT 57): an integer view under a float load hands the integer's
/// bits over as a float (a denormal) and the f16 of it is 0, so every
/// activation read zero (pk8a f1027: network output 0 non-zero). Viewed in
/// the float format of the same layout the halves pass through unchanged
/// (f16 -> f32 -> f16 is exact, NaN payloads aside, I).
fn d16_as_float(job: &mut cr::Job) {
    for a in job.roots.iter_mut() {
        let cr::Arg::Table(t) = a else { continue };
        for (_, r) in t.iter_mut() {
            if let cr::Res::Buffer(b) = r
                && let Kind::Typed { format, bytes } = b.kind
            {
                let f = match format {
                    12 | 14 => 10,
                    36 | 38 => 34,
                    57 | 59 => 54,
                    x => x,
                };
                b.kind = Kind::Typed { format: f, bytes };
            }
        }
    }
}

/// `CS_ComputeBlendshapeMappings` (220c0303, code 0x24-0xf8) on the
/// CPU: per vertex v below cb[0] whose 16-B index record (at cb[0x18] +
/// cb[0x1c] * v of the face's vertex buffer) carries tag 0xb6 in dword 3,
/// mapping[(dword1 >> 16) - 1] = ((dword0 >> 16) - 1) << 16 | v (0xffff
/// as the blend index when dword0 >> 16 is 0xffff). The title runs it once
/// per ped (no pk7h frame dispatches it, I), so captured memory holds the
/// mapping as cleared, all 0xff, and the HSS extract then reads vertex
/// 0xffff for every entry. Same vertex buffer (HSS slot 1), mapping (slot
/// 3) and constant block (slot 7) as the extract binds.
fn mappings(p: &Pack, sink: u32, views: &[fp::SlotView], runner: &mut cr::Runner, g: &mut Graph) -> Result<(), String> {
    let Some(map) = views.iter().find(|v| v.slot == 3) else { return Ok(()) };
    let (Some(vb), Ok((_, cb))) = (views.iter().find(|v| v.slot == 1), p.slot_bytes(sink, 0, 7)) else { return Ok(()) };
    let (ma, va) = (map.addr % GPU_ALIAS, vb.addr % GPU_ALIAS);
    let mut m = captured(p, ma);
    m.resize(map.bytes as usize, 0xff);
    if m.iter().any(|b| *b != 0xff) {
        return Ok(());
    }
    let v = captured(p, va);
    let (count, at, stride) = (crate::le32(&cb, 0), crate::le32(&cb, 0x18) as usize, crate::le32(&cb, 0x1c) as usize);
    let mut n = 0;
    for k in 0..count as usize {
        let r = at + stride * k;
        let (d0, d1, d3) = (crate::le32(&v, r), crate::le32(&v, r + 4), crate::le32(&v, r + 12));
        if d3 & 0xff00_0000 != 0xb600_0000 {
            continue;
        }
        let blend = if d0 >> 16 != 0xffff { (d0 >> 16).wrapping_sub(1) << 16 } else { 0xffff_0000 };
        let e = ((d1 >> 16) as usize).wrapping_sub(1);
        if let Some(w) = m.get_mut(4 * e..4 * e + 4) {
            w.copy_from_slice(&(blend | k as u32).to_le_bytes());
            n += 1;
        }
    }
    runner.hold(&cr::Res::Buffer(cr::Buffer { key: ma, kind: Kind::Raw, uav: false, len: map.bytes, bytes: &m }))?;
    g.lacks.insert(format!("blend-shape mapping {ma:#x} ({} B): all 0xff in memory and no CS_ComputeBlendshapeMappings in the frame (it runs once per ped): {n} of {count} vertices mapped on the CPU by its rule", map.bytes));
    Ok(())
}

/// The HSS extract's table slot that is the head position table (slot 6:
/// 1,404,980 B at 0x164529840 on pk7h, frame-table slot 7, t74/t75).
const TABLE_OUT: u32 = 6;

/// A translated compute program's `PSV0` wave range and thread-group size.
fn psv_text(dxil: &[u8]) -> String {
    let Some(d) = xnw_shader::Container::parse(dxil).ok().and_then(|c| c.part(xnw_shader::FourCC(*b"PSV0")).map(|p| p.data.to_vec())) else {
        return "no PSV0".into();
    };
    let w = |o: usize| d.get(4 + o..8 + o).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    format!("wave lanes {}..{}, threads [{}, {}, {}]", w(16), w(20), w(36), w(40), w(44))
}

fn views_len(p: &Pack, addr: u64) -> u64 {
    p.data.refs.iter().filter(|r| r.kind == kind::BUFFER && r.addr % GPU_ALIAS == addr).map(|r| u64::from(r.len)).max().unwrap_or(0)
}

/// The filled head position table for the replay (`headgraph=on`): the
/// title address (modulo the alias) and bytes, set once per frame.
static TABLE: Mutex<Option<(u64, Vec<u8>)>> = Mutex::new(None);

/// `headgraph=on` in `XNW_WORLD_RENDER` (default off, HEAD1).
pub fn switch_on(v: &str) -> bool {
    v.split(',').map(str::trim).filter_map(|p| p.strip_prefix("headgraph=")).last().is_some_and(|s| s != "off")
}

pub fn set_table(t: Option<(u64, Vec<u8>)>) {
    *TABLE.lock().unwrap_or_else(|e| e.into_inner()) = t;
}

/// The filled table's bytes when `addr` (any alias) is its view.
pub fn table_for(addr: u64) -> Option<Vec<u8>> {
    let t = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    t.as_ref().filter(|(a, _)| *a == addr % GPU_ALIAS).map(|(_, b)| b.clone())
}

/// Runs the graph for the frame and keeps its table for the replay, when
/// the switch is on; a line for the log. `XNW_HEADGRAPH_TABLE=<file>`: the
/// table `xnw-lab head --save` wrote instead (the pass may run on another
/// adapter than the title's programs need), at the extract's output view.
pub fn prepare(dir: &std::path::Path, host: &xnw_gpu::host::Host) -> Option<String> {
    if !switch_on(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default()) {
        return None;
    }
    let p = match Pack::open(dir) {
        Ok(Some(p)) => p,
        Ok(None) => return Some("head graph: no pack".into()),
        Err(e) => return Some(format!("head graph: {e}")),
    };
    let p = &p;
    if let Ok(f) = std::env::var("XNW_HEADGRAPH_TABLE") {
        let names = crate::audit::program_names(&p.dir).unwrap_or_default();
        let at = dispatches(p, &names).into_iter().filter(|d| d.4.contains("HSSGraphOutputExtract")).find_map(|d| p.tables(d.0).into_iter().find(|t| t.0 == 0)?.2.into_iter().find(|v| v.slot == TABLE_OUT));
        return Some(match (std::fs::read(&f), at) {
            (Ok(b), Some(v)) => {
                let line = format!("head graph: table from {f}: {} B at {:#x}, {} non-zero", b.len(), v.addr % GPU_ALIAS, b.iter().filter(|x| **x != 0).count());
                set_table(Some((v.addr % GPU_ALIAS, b)));
                line
            }
            (Err(e), _) => format!("head graph: {f}: {e}"),
            (_, None) => "head graph: no HSS extract sink in the frame".into(),
        });
    }
    let only = std::env::var("XNW_HEADGRAPH_ONLY").is_ok_and(|v| v == "head");
    Some(match run(p, host, &|_, n| !only || n.starts_with("ped_standard_head_cs:")) {
        Ok(g) => {
            let line = g.line();
            set_table(g.table);
            line
        }
        Err(e) => format!("head graph: {e}"),
    })
}

/// `xnw-lab head <frame> [--adapter warp|hw] [--watchdog-ms N] [--runs N] [--dump] [--only-head] [--sinks a,b] [--save <file>] [--against <file>]`:
/// the graph run, a line per dispatch, the lacks, the table's entries per face.
/// `--adapter` (the lab's default is hw; `XNW_ADAPTER` when absent): the
/// hardware adapter only as a scheduled card sample, `XNW_BRIDGE_HW=1`,
/// under the per-submission GPU watchdog (one dispatch a submission).
/// `--runs N`: the graph N times in this process from the captured inputs,
/// the network output's and the table's hash each time, and how many differ.
pub fn head_cmd(args: &[String]) -> Result<(), String> {
    let dir = args.first().ok_or("usage: xnw-lab head <frame dir> [--adapter warp|hw] [--runs N] [--dump] [--only-head] [--save <file>]")?;
    let p = Pack::open(std::path::Path::new(dir))?.ok_or("no pack")?;
    if args.iter().any(|a| a == "--dump") {
        return dump(&p);
    }
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let hw = match opt("--adapter").as_deref() {
        Some("warp") => Some(false),
        Some("hw") => Some(true),
        Some(o) => return Err(format!("head graph: --adapter warp|hw, not {o}")),
        None => None,
    };
    if let Some(hw) = hw {
        // SAFETY: set before any thread of ours reads the environment.
        unsafe { std::env::set_var("XNW_ADAPTER", if hw { "hw" } else { "warp" }) };
        // The hardware adapter needs the scheduled-sample switch: asked before the device is made.
        if hw && std::env::var("XNW_BRIDGE_HW").ok().as_deref() != Some("1") {
            return Err("head graph: the title's translated programs run on the software adapter; --adapter hw needs XNW_BRIDGE_HW=1 (a scheduled card sample)".into());
        }
    }
    let host = xnw_gpu::host::Host::new(None).map_err(|e| format!("head graph: device: {e}"))?;
    if hw.is_some_and(|hw| hw == host.caps.software) {
        return Err(format!("head graph: asked for the {} adapter, got {}", if host.caps.software { "hardware" } else { "software" }, host.caps.adapter));
    }
    if !host.caps.software {
        crate::replay::watchdog_start(std::time::Duration::from_millis(opt("--watchdog-ms").and_then(|v| v.parse().ok()).unwrap_or(crate::replay::WATCHDOG_MS)));
    } else if let Some(ms) = opt("--watchdog-ms").and_then(|v| v.parse().ok()) {
        crate::replay::watchdog_start(std::time::Duration::from_millis(ms));
    }
    println!("head graph: adapter {}", host.caps.adapter);
    let only = args.iter().any(|a| a == "--only-head");
    let sinks: Vec<u32> = opt("--sinks").map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let keep = |s: u32, n: &str| (!only || n.starts_with("ped_standard_head_cs:")) && (sinks.is_empty() || sinks.contains(&s));
    crate::replay::watchdog_arm();
    let g = run(&p, &host, &keep);
    crate::replay::watchdog_disarm();
    println!("  {}", crate::replay::watch_line("head graph"));
    let g = g?;
    for l in &g.lines {
        println!("  {l}");
    }
    for l in &g.lacks {
        println!("  lacks: {l}");
    }
    println!("{}", g.line());
    if let Some(n) = opt("--runs").and_then(|v| v.parse::<u32>().ok()).filter(|n| *n > 1) {
        let hash = |g: &Graph| (g.network.map(|x| x.3), g.table.as_ref().map(|(_, b)| xnw_core::hash128(&[b]) as u32));
        let first = hash(&g);
        let mut seen: BTreeSet<(Option<u32>, Option<u32>)> = BTreeSet::from([first]);
        for k in 2..=n {
            crate::replay::watchdog_arm();
            let again = run(&p, &host, &keep);
            crate::replay::watchdog_disarm();
            println!("  {}", crate::replay::watch_line(&format!("head graph run {k}")));
            let again = again?;
            let h = hash(&again);
            seen.insert(h);
            println!("head graph run {k}: {} dispatches, network output hash {}, head table hash {}{}", again.ran, h.0.map_or("-".into(), |v| format!("{v:08x}")), h.1.map_or("-".into(), |v| format!("{v:08x}")), if h == first { "" } else { " (differs from run 1)" });
        }
        println!("head graph: {n} runs on {}: {} distinct (network, table) hashes", host.caps.adapter, seen.len());
    }
    // `XNW_GPU_DEBUG=layer[,gbv=state]`: the validation layer's messages (Card safety 1).
    if std::env::var("XNW_GPU_DEBUG").is_ok_and(|v| v.contains("layer")) {
        for l in xnw_world::bridge::layer_messages(&host.device) {
            println!("head graph: {l}");
        }
    }
    if let Some((a, t)) = &g.table {
        let other = match args.iter().position(|a| a == "--against").and_then(|i| args.get(i + 1)) {
            Some(f) => Some(std::fs::read(f).map_err(|e| format!("--against {f}: {e}"))?),
            None => None,
        };
        faces(&p, *a, t, other.as_deref());
        if let Some(i) = args.iter().position(|a| a == "--save") {
            let f = args.get(i + 1).ok_or("--save <file>")?;
            std::fs::write(f, t).map_err(|e| e.to_string())?;
            println!("  table saved to {f}");
        }
    }
    Ok(())
}

/// The faces in the table: per variant-11 pre-skin dispatch whose t74 is
/// the table, its base (b6+0), header and the first positions decoded.
pub fn faces(p: &Pack, addr: u64, t: &[u8], against: Option<&[u8]>) {
    let names = crate::audit::program_names(&p.dir).unwrap_or_default();
    for d in p.dispatches().iter().filter(|d| names.get(&d.program).and_then(|n| crate::preskin::variant_of(n)) == Some(11)) {
        let Ok(inp) = crate::preskin::load(p, d) else { continue };
        let base = crate::le32(&inp.b6, 0) as usize;
        let t74 = p.tables(d.sink).into_iter().find(|t| t.0 == 1).and_then(|t| t.2.into_iter().find(|v| v.slot == 0));
        let same = t74.is_some_and(|v| v.addr % GPU_ALIAS == addr);
        let hdr: Vec<String> = (0..7).map(|k| format!("{:.4}", f32::from_bits(crate::le32(t, base + 4 * k)))).collect();
        // Against the bind positions of the same entries: the expression's offsets.
        let bind = ps::neutral_table(&inp.inputs());
        let (mut worst, mut sum, mut moved) = (0f32, 0f64, 0);
        for (e, b) in &bind {
            let q = ps::table_pos(t, base as u32, *e);
            let d = ((q[0] - b[0]).powi(2) + (q[1] - b[1]).powi(2) + (q[2] - b[2]).powi(2)).sqrt();
            worst = worst.max(d);
            sum += f64::from(d);
            moved += usize::from(d > 1e-4);
        }
        let first: Vec<String> = bind.iter().take(3).map(|(e, b)| { let q = ps::table_pos(t, base as u32, *e); format!("{e}: table ({:.4},{:.4},{:.4}) bind ({:.4},{:.4},{:.4})", q[0], q[1], q[2], b[0], b[1], b[2]) }).collect();
        println!("  face: sink {} base {base:#x} (t74 is the table: {same}); header {}; {} entries: {moved} off their bind position by > 0.1 mm, mean {:.4} m, largest {:.4} m; {}", d.sink, hdr.join(" "), bind.len(), sum / bind.len().max(1) as f64, worst, first.join("; "));
        if let Some(o) = against.filter(|_| !bind.is_empty()) {
            moves(t, o, base as u32, &bind);
        }
    }
}

/// Where table `t` differs from table `o` (another run's, `--against`)
/// over a face's entries: how many move by more than 1 mm, and those
/// entries' bind positions as fractions of the face's box (x, y, z; 0 the
/// box's low side), as a 3 x 3 x 3 count and the ten largest.
fn moves(t: &[u8], o: &[u8], base: u32, bind: &[(u32, [f32; 3])]) {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for (_, b) in bind {
        for k in 0..3 {
            lo[k] = lo[k].min(b[k]);
            hi[k] = hi[k].max(b[k]);
        }
    }
    let frac = |b: &[f32; 3]| -> [f32; 3] { std::array::from_fn(|k| (b[k] - lo[k]) / (hi[k] - lo[k]).max(1e-6)) };
    let mut moved: Vec<(f32, u32, [f32; 3])> = Vec::new();
    for (e, b) in bind {
        let (q, r) = (ps::table_pos(t, base, *e), ps::table_pos(o, base, *e));
        let d = ((q[0] - r[0]).powi(2) + (q[1] - r[1]).powi(2) + (q[2] - r[2]).powi(2)).sqrt();
        if d > 1e-3 {
            moved.push((d, *e, frac(b)));
        }
    }
    moved.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut grid = [[[0u32; 3]; 3]; 3];
    for (_, _, f) in &moved {
        let c = |v: f32| ((v * 3.0) as usize).min(2);
        grid[c(f[0])][c(f[1])][c(f[2])] += 1;
    }
    let top: Vec<String> = moved.iter().take(10).map(|(d, e, f)| format!("{e}: {:.1} mm at ({:.2},{:.2},{:.2})", d * 1e3, f[0], f[1], f[2])).collect();
    println!("    against the other table: {} of {} entries move by more than 1 mm (box x {:.3}..{:.3}, y {:.3}..{:.3}, z {:.3}..{:.3}); count by box third [x][y][z] {grid:?}; largest {}", moved.len(), bind.len(), lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], top.join("; "));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_views_of_16_bit_integers_read_as_float() {
        let b = |format| cr::Res::Buffer(cr::Buffer { key: 1, kind: Kind::Typed { format, bytes: 8 }, uav: true, len: 8, bytes: &[] });
        let mut job = cr::Job { roots: vec![cr::Arg::Table(vec![(0, b(12)), (1, b(57)), (2, b(41)), (3, b(10))])], ..cr::Job::default() };
        d16_as_float(&mut job);
        let cr::Arg::Table(t) = &job.roots[0] else { panic!("table") };
        let got: Vec<i32> = t.iter().map(|(_, r)| match r { cr::Res::Buffer(cr::Buffer { kind: Kind::Typed { format, .. }, .. }) => *format, _ => -1 }).collect();
        assert_eq!(got, [10, 54, 41, 10]);
    }

    #[test]
    fn switch_grammar() {
        assert!(switch_on("hair=on,headgraph=on"));
        assert!(!switch_on("headgraph=off"));
        assert!(!switch_on("hair=on"));
    }

    #[test]
    fn network_thread_runs_before_the_head_programs() {
        let d = |s, t, n: &str| (s, t, 1, 0u128, n.to_string());
        let got: Vec<u32> = order(&[d(1, 1, "compute_graph_ops:a"), d(2, 5, "ped_standard_head_cs:b"), d(3, 1, "compute_graph_ops:c"), d(4, 5, "ped_standard_head_cs:d")]).iter().map(|x| x.0).collect();
        assert_eq!(got, [1, 3, 2, 4]);
        // pk8a: the head programs' thread records first; the network still runs first.
        let got: Vec<u32> = order(&[d(36, 4, "ped_standard_head_cs:map"), d(38, 4, "ped_standard_head_cs:extract"), d(89, 1, "compute_graph_ops:formula"), d(834, 1, "compute_graph_ops:gather")]).iter().map(|x| x.0).collect();
        assert_eq!(got, [89, 834, 36, 38]);
    }
}
