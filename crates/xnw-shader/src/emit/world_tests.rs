//! XNW: the compute programs the title used in its first world frames
//! that once failed to translate, by the 128-bit hash the pipeline log names
//! them with (`xnw_core::hash128` of the container). Containers come from a
//! run's census dumps (`XNW_RUN_DIR`, else `<out>/run`, every
//! `*\census\*-<hash>.bin`), else from the pack (`XNW_SHADER_PACK`) by the
//! same hash. Each must translate and validate; with `XNW_AGILITY`, a host
//! compute pipeline is made too. Ignored by default (reads the pack);
//! `cargo test -p xnw-shader world_programs -- --ignored --nocapture`.

use std::collections::HashMap;
use std::path::PathBuf;

use super::host::Host;
use super::validator::Validator;
use crate::rpf9::{Archive, EntryKind};
use crate::sgd5;

const WORLD: &[&str] = &[
    "0451d340bf7dc5684682004ee641b868",
    "1fb4b3722dde4c7cb5ad9a2c9cd1b4a6",
    "278c4561995dd9537c3c79259d5010bd",
    "2a8122977d3bca87b3dec8ced1f5c4e9",
    "5fa3b9ec81b4e4b5f3ddc88412d74c09",
    "a5e8b19097679d315ba60f7cafa16ac9",
    "c20df6d5d8081e453236a82a137b9c56",
    "48e658a0db8468b59f6fc84a651fa171",
    "9c5904029d3738e81f43901eb6529c53",
    "a49e33ea638e8a90c4224adfab34d17a",
    "818485c2df101acc425f1b1a53f6fe17",
    "feddfe91e9e56fe45860b30bde7844ec",
    "168936d7fea333aa8d041c7e2f231315",
];

/// The containers named by `hashes`: census dumps first, then the pack.
fn containers(hashes: &[&str]) -> HashMap<String, Vec<u8>> {
    let mut out = HashMap::new();
    let runs = std::env::var_os("XNW_RUN_DIR").map_or_else(|| PathBuf::from(r"out\run"), Into::into);
    for run in std::fs::read_dir(runs).into_iter().flatten().flatten() {
        for f in std::fs::read_dir(run.path().join("census")).into_iter().flatten().flatten() {
            let name = f.file_name().to_string_lossy().into_owned();
            if let Some(h) = hashes.iter().find(|h| name.ends_with(&format!("-{h}.bin")))
                && let Ok(b) = std::fs::read(f.path())
            {
                out.insert((*h).to_owned(), b);
            }
        }
    }
    if out.len() == hashes.len() {
        return out;
    }
    let pack = std::env::var_os("XNW_SHADER_PACK").map_or_else(|| PathBuf::from(r"game\shaders.rpf"), Into::into);
    let Ok(data) = std::fs::read(pack) else { return out };
    let Ok(archive) = Archive::parse(&data) else { return out };
    for (i, e) in archive.entries.iter().enumerate() {
        if e.kind != EntryKind::Sgd5 {
            continue;
        }
        let Some(bytes) = archive.read(i) else { continue };
        for r in sgd5::records(bytes) {
            let Some(b) = r.unpack() else { continue };
            let h = format!("{:032x}", xnw_core::hash128(&[&b]));
            if hashes.contains(&h.as_str()) {
                out.entry(h).or_insert_with(|| b.to_vec());
            }
        }
    }
    out
}

/// The native program of `container` in SP3 text, offset per line, to `path`.
fn disasm(container: &[u8], path: &std::path::Path) {
    let Ok(c) = crate::Container::parse(container) else { return };
    let Some(d) = c.part(crate::FourCC::DXIL) else { return };
    let Ok(programs) = crate::native_programs(d.data) else { return };
    let mut out = String::new();
    for p in programs {
        // XNW: the root layout and used resources first, `#` lines.
        for r in &p.layout {
            out += &format!("# param {r:?}\n");
        }
        for u in &p.used {
            out += &format!("# used {u:?}\n");
        }
        let code = p.patched_code();
        for i in crate::gfx10::decode(&code).unwrap_or_default() {
            out += &format!("{:#06x}  {}
", i.offset, i.text(p.wave64));
        }
    }
    let _ = std::fs::write(path, out);
}

/// A host pipeline for `s`: compute (mesh programs in their fallback form,
/// with the fallback's record views) or a graphics one behind a stub vertex
/// shader (pixel programs).
fn host_pipeline(host: &Host, s: &super::HostShader) -> Result<(), String> {
    use crate::native::HwStage;
    use windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_ALL;
    match s.stage {
        HwStage::Cs => host.compute_pipeline(s).map(drop),
        HwStage::Ms => {
            use super::mesh::{INDEX_REGISTER, RECORD_REGISTER};
            let root = host.root_signature_records(&[(s, D3D12_SHADER_VISIBILITY_ALL)], &[(true, RECORD_REGISTER), (true, INDEX_REGISTER)])?;
            host.compute_pipeline_with(&root, &s.dxil).map(drop)
        }
        HwStage::Ps => host.pipeline(&super::host::stub_vs(s)?, s).map(drop),
        other => Err(format!("no host pipeline test for {other:?}")),
    }
}

#[test]
#[ignore = "reads run dumps and the pack"]
fn world_programs() {
    // XNW_WORLD_LIST: a file of `hash|...` lines (a run's failures) in place
    // of WORLD; then only the counts are checked by eye, not asserted.
    let listed: Option<Vec<String>> = std::env::var_os("XNW_WORLD_LIST")
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| t.lines().filter_map(|l| l.split('|').next()).map(str::to_owned).collect());
    let names: Vec<&str> = listed.as_ref().map_or_else(|| WORLD.to_vec(), |l| l.iter().map(String::as_str).collect());
    let found = containers(&names);
    let mut validator = Validator::load(Validator::default_path().as_deref()).ok();
    // XNW_WORLD_NO_GPU: translation and validation only (no device).
    let host = if std::env::var_os("XNW_WORLD_NO_GPU").is_some() { None } else { Host::for_tests() };
    let mut bad = 0;
    // Instruction counts before and after the clean-up passes and DXIL
    // bytes, summed over the programs that validate.
    let (mut sum_before, mut sum_after, mut sum_bytes, mut ok) = (0usize, 0usize, 0usize, 0usize);
    for h in &names {
        let Some(c) = found.get(*h) else {
            println!("{h} not found");
            bad += 1;
            continue;
        };
        if let Some(dir) = std::env::var_os("XNW_WORLD_DISASM") {
            disasm(c, &std::path::Path::new(&dir).join(format!("{h}.txt")));
        }
        let mut table = false;
        let mut bounded = false;
        let mut stats = (0, 0, 0);
        // The emitted DXIL's hash, so two trees' outputs compare per program.
        let mut dxil_hash = 0u128;
        // What it asks of an adapter: the wave size it declares, its feature flags.
        let (mut wave, mut sfi0) = (None, 0u64);
        // Mesh programs as the fallback's compute form: this card has no
        // mesh shaders (XNW).
        // XNW_WORLD_GPU: GPU-side descriptors.
        let descriptors = if std::env::var_os("XNW_WORLD_GPU").is_some() { super::Descriptors::Gpu } else { super::Descriptors::Plan };
        // XNW_WORLD_PAGETABLE: the page-table form (with XNW_WORLD_GPU).
        let pagetable = std::env::var_os("XNW_WORLD_PAGETABLE").is_some();
        let opts = super::Options { mesh: super::MeshForm::Fallback, descriptors, pagetable, ..Default::default() };
        let r = super::translate_with(c, opts).map_err(|e| e.to_string()).and_then(|s| {
            table = s.reads_stride_table();
            bounded = s.reads_heap_bounds();
            let (b, a) = super::shader::OPT_STATS.with(|c| c.get());
            stats = (b, a, s.dxil.len());
            dxil_hash = xnw_core::hash128(&[&s.dxil]);
            (wave, sfi0) = (super::container::declared_wave(&s.dxil), super::container::feature_flags(&s.dxil).unwrap_or(0));
            if let Some(dir) = std::env::var_os("XNW_WORLD_DISASM") {
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{h}.dxil")), &s.dxil);
            }
            if let Some(v) = validator.as_mut() {
                let mut copy = s.dxil.clone();
                v.validate(&mut copy)?;
            }
            match &host {
                Some(host) => host_pipeline(host, &s).map(|()| "validated, pipeline made"),
                None => Ok(if validator.is_some() { "validated" } else { "translated" }),
            }
        });
        match r {
            Ok(how) => {
                println!("{h} {how}{}{} insts {}->{} bytes {} dxil {dxil_hash:032x} wave {} sfi0 {sfi0:#x}", if table { " (runtime stride table)" } else { "" }, if bounded { " (heap-bounded)" } else { "" }, stats.0, stats.1, stats.2, wave.map_or("any".to_owned(), |w| w.to_string()));
                (sum_before, sum_after, sum_bytes, ok) = (sum_before + stats.0, sum_after + stats.1, sum_bytes + stats.2, ok + 1);
            }
            Err(e) => {
                println!("{h} FAILS {}", if std::env::var_os("XNW_WORLD_FULL").is_some() { e.as_str() } else { e.lines().next().unwrap_or("") });
                bad += 1;
            }
        }
    }
    println!("{} of {} world programs translate", names.len() - bad, names.len());
    println!("passing {ok}: insts {sum_before} -> {sum_after}, DXIL bytes {sum_bytes}");
    if listed.is_none() {
        assert_eq!(bad, 0, "{bad} of {} world programs fail", names.len());
    }
}

/// XNW: the mesh programs' data layouts. For every MS program of the
/// pack (or `XNW_WORLD_LIST`), the backward slice of each export (position,
/// parameters by row) over the SSA IR: the memory reads it depends on, each
/// as `name@descriptor+offset` (descriptor: the root parameter and byte
/// offset of the scalar load that fetched it; `H` the view heap; `R` a
/// root view; `C` code), and the scalar loads it reads. Programs with the
/// same reads are one layout class. Writes `XNW_MESH_SURVEY` (a directory):
/// `mesh-layouts.tsv` (hash, name, class) and `classes.txt` (per class:
/// members, examples, the reads of each export). `cargo test -p xnw-shader
/// mesh_layout_survey -- --ignored`.
#[test]
#[ignore = "reads the pack"]
fn mesh_layout_survey() {
    use std::collections::{BTreeMap, BTreeSet, HashSet};
    use std::fmt::Write as _;
    let Some(out) = std::env::var_os("XNW_MESH_SURVEY").map(PathBuf::from) else { return };
    let pack = std::env::var_os("XNW_SHADER_PACK").map_or_else(|| PathBuf::from(r"game\shaders.rpf"), Into::into);
    let data = std::fs::read(pack).expect("pack");
    let archive = Archive::parse(&data).expect("archive");
    let listed: Option<HashSet<String>> = std::env::var_os("XNW_WORLD_LIST")
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| t.lines().filter_map(|l| l.split('|').next()).map(|s| s.trim().to_owned()).collect());
    let mut rows: Vec<(String, String, String, String)> = Vec::new();
    let mut seen = HashSet::new();
    for (i, e) in archive.entries.iter().enumerate() {
        if e.kind != EntryKind::Sgd5 {
            continue;
        }
        let Some(bytes) = archive.read(i) else { continue };
        for r in sgd5::records(bytes) {
            let name = r.name.unwrap_or("-");
            if listed.is_none() && !name.starts_with("MS") {
                continue;
            }
            let Some(b) = r.unpack() else { continue };
            let h = format!("{:032x}", xnw_core::hash128(&[&b]));
            if listed.as_ref().is_some_and(|l| !l.contains(&h)) || !seen.insert(h.clone()) {
                continue;
            }
            let (sig, detail) = mesh_signature(&b).unwrap_or_else(|e| (format!("FAILS {e}"), String::new()));
            rows.push((h, name.to_owned(), sig, detail));
        }
    }
    let mut classes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (n, r) in rows.iter().enumerate() {
        classes.entry(r.2.clone()).or_default().push(n);
    }
    let mut order: Vec<(&String, &Vec<usize>)> = classes.iter().collect();
    order.sort_by_key(|(_, m)| std::cmp::Reverse(m.len()));
    let mut id_of: BTreeMap<&String, usize> = BTreeMap::new();
    let mut text = String::new();
    for (k, (sig, members)) in order.iter().enumerate() {
        id_of.insert(sig, k);
        let mut names: BTreeSet<String> = BTreeSet::new();
        for &m in members.iter() {
            names.insert(rows[m].1.clone());
        }
        let examples: Vec<String> = names.iter().take(6).cloned().collect();
        let _ = writeln!(text, "== class {k}: {} programs, {} names; first {}; e.g. {}", members.len(), names.len(), rows[members[0]].0, examples.join(", "));
        let _ = writeln!(text, "{sig}");
        let _ = writeln!(text, " -- all reads of the first member:\n{}", rows[members[0]].3);
    }
    let mut tsv = String::from("hash\tname\tclass\n");
    for r in &rows {
        let _ = writeln!(tsv, "{}\t{}\t{}", r.0, r.1, id_of[&r.2]);
    }
    std::fs::create_dir_all(&out).expect("dir");
    std::fs::write(out.join("mesh-layouts.tsv"), tsv).expect("tsv");
    std::fs::write(out.join("classes.txt"), text).expect("classes");
    println!("{} mesh programs, {} layout classes", rows.len(), classes.len());
}

/// The reads each export of the mesh program in `container` depends on (see
/// [`mesh_layout_survey`]), as text: one line per export.
fn mesh_signature(container: &[u8]) -> Result<(String, String), String> {
    use std::collections::{BTreeSet, HashMap};
    use crate::gfx10::Fields;
    use crate::ir::Reg;
    let c = crate::Container::parse(container).map_err(|e| e.to_string())?;
    let d = c.part(crate::FourCC::DXIL).ok_or("no DXIL")?;
    let programs = crate::native_programs(d.data).map_err(|e| e.to_string())?;
    let native = programs.into_iter().find(|p| p.stage == crate::HwStage::Ms).ok_or("no mesh program")?;
    let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).map_err(|e| e.to_string())?;
    let res = crate::resource::track_with(&p, &native.layout, native.stage, true).map_err(|e| e.to_string())?;
    let io = crate::io::raise(&p, &native).map_err(|e| e.to_string())?;
    let mut def: HashMap<crate::ir::Value, (usize, usize)> = HashMap::new();
    let mut phi: HashMap<crate::ir::Value, Vec<crate::ir::Value>> = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for ph in &block.phis {
            phi.insert(ph.value, ph.args.clone());
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for w in &inst.writes {
                def.insert(w.1, (b, i));
            }
        }
    }
    let scalar: HashMap<(usize, usize), &crate::resource::Address> = res.scalar.iter().map(|(s, a)| (*s, a)).collect();
    let param = |a: &crate::resource::Address| -> String {
        if a.param == crate::resource::CODE {
            return format!("C+{:#x}", a.offset);
        }
        match native.layout.get(a.param) {
            Some(r) => {
                let kind = match r.root {
                    crate::RootType::Table => "T",
                    crate::RootType::ResourceHeap => "H",
                    crate::RootType::Descriptor => "R",
                    crate::RootType::Constants => "K",
                    _ => "?",
                };
                format!("{kind}{}+{:#x}{}", r.user_data, a.offset, if a.dynamic.is_some() { "+dyn" } else { "" })
            }
            None => format!("P{}+{:#x}", a.param, a.offset),
        }
    };
    let descriptor = |site: (usize, usize)| -> String {
        let Some(d) = res.descriptors.iter().find(|d| d.site == site && d.role == crate::resource::Role::Resource) else { return "?".into() };
        match &d.source {
            crate::resource::Source::Loaded { load, first } => match scalar.get(load) {
                Some(a) => format!("{}#{first}", param(a)),
                None => "L?".into(),
            },
            crate::resource::Source::Built { address, .. } => format!("B[{}]", param(address)),
            crate::resource::Source::Phi { .. } => "phi".into(),
            _ => "x".into(),
        }
    };
    let mut lines = Vec::new();
    let mut detail: Vec<String> = Vec::new();
    for e in &io.exports {
        let what = match e.target {
            crate::io::Target::Position(n) => format!("pos{n}"),
            crate::io::Target::Param { row, .. } => format!("param{row}"),
            crate::io::Target::Primitive => "prim".into(),
            _ => continue,
        };
        let mut stack: Vec<crate::ir::Value> = e.values.iter().flatten().copied().collect();
        let mut seen = std::collections::HashSet::new();
        let mut reads: BTreeSet<String> = BTreeSet::new();
        let mut ops = 0usize;
        while let Some(v) = stack.pop() {
            if !seen.insert(v) {
                continue;
            }
            if let Some(args) = phi.get(&v) {
                stack.extend(args.iter().copied());
                continue;
            }
            let Some(&(b, i)) = def.get(&v) else { continue };
            let inst = &p.blocks[b].insts[i];
            ops += 1;
            let name = inst.name();
            match &inst.fields {
                Fields::Buffer { offset, format, .. } => {
                    let f = format.map_or(String::new(), |f| format!("/f{f}"));
                    reads.insert(format!("{name}{f}@{}+{offset}", descriptor((b, i))));
                }
                Fields::Smem { offset, .. } => {
                    let at = if name.starts_with("s_buffer_load") {
                        descriptor((b, i))
                    } else {
                        scalar.get(&(b, i)).map_or("?".into(), |a| param(a))
                    };
                    reads.insert(format!("{name}@{at}+{offset:#x}"));
                }
                Fields::Image { .. } => {
                    reads.insert(format!("{name}@{}", descriptor((b, i))));
                }
                Fields::Ds { .. } => {
                    reads.insert(name.to_string());
                }
                _ => {}
            }
            for r in &inst.reads {
                if !matches!(r.0, Reg::S(126) | Reg::S(127)) {
                    stack.push(r.1);
                }
            }
        }
        let _ = ops;
        if reads.is_empty() {
            continue;
        }
        // The class key is the geometry stream only: reads through the
        // root view (vertex and meshlet data) and the heap (instance index);
        // the constant-buffer reads (transform, material) go to the detail.
        // Union over the exports, so which output reads what does not split
        // a class.
        for r in reads.iter().filter(|r| r.contains("@B[") || r.contains("@H")) {
            lines.push(format!("  {r}"));
        }
        detail.push(format!("  {what}: {}", reads.iter().map(|r| r.as_str()).collect::<Vec<_>>().join(" ")));
    }
    lines.sort();
    lines.dedup();
    detail.sort();
    Ok((lines.join("\n"), detail.join("\n")))
}
/// XNW: the clean-up passes over the whole pack (or `XNW_WORLD_LIST`),
/// translated once per form (`XNW_SWEEP_FORMS`: `plan`, `pt` for GPU-side
/// descriptors with the page table; default both; mesh programs as the
/// fallback) and finished with no pass, each pass alone and all. Per form
/// and pass set: validated, validator failures by reason, instructions,
/// DXIL bytes, the seconds finishing and validating took, the programs
/// whose validation differs from no pass, and the semantic differential's
/// flags (`Module::differential`). `XNW_SWEEP_OUT` (a directory):
/// `summary.txt`, `flags.txt`, `diffs.txt`. `XNW_PACK_THREADS` (6).
/// `cargo test -p xnw-shader opt_sweep -- --ignored --nocapture`.
#[test]
#[ignore = "translates the whole pack"]
fn opt_sweep() {
    use super::bitcode::opt::{DEAD, FACTS, FOLD, LOADS, MODULE, REUSE};
    use super::shader::{SWEEP, Sweep};
    use std::collections::{BTreeMap, HashSet};
    use std::fmt::Write as _;
    use std::sync::Mutex;
    let Some(out) = std::env::var_os("XNW_SWEEP_OUT").map(PathBuf::from) else { return };
    const SETS: [(u32, &str); 7] = [(0, "off"), (DEAD, "dead"), (FOLD, "fold"), (REUSE, "reuse"), (FACTS, "facts"), (LOADS, "loads"), (MODULE, "all")];
    let forms: Vec<bool> = std::env::var("XNW_SWEEP_FORMS").unwrap_or_else(|_| "plan,pt".into()).split(',').map(|f| f == "pt").collect();
    let pack = std::env::var_os("XNW_SHADER_PACK").map_or_else(|| PathBuf::from(r"game\shaders.rpf"), Into::into);
    let data = std::fs::read(pack).expect("pack");
    let archive = Archive::parse(&data).expect("archive");
    let listed: Option<HashSet<String>> = std::env::var_os("XNW_WORLD_LIST")
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| t.lines().filter_map(|l| l.split('|').next()).map(|s| s.trim().to_owned()).collect());
    let mut records = Vec::new();
    let mut seen = HashSet::new();
    for (i, e) in archive.entries.iter().enumerate() {
        if e.kind != EntryKind::Sgd5 {
            continue;
        }
        let Some(bytes) = archive.read(i) else { continue };
        for r in sgd5::records(bytes) {
            let Some(b) = r.unpack() else { continue };
            let h = format!("{:032x}", xnw_core::hash128(&[&b]));
            if listed.as_ref().is_some_and(|l| !l.contains(&h)) || !seen.insert(h.clone()) {
                continue;
            }
            records.push((h, b));
        }
    }
    // Per (form, set): validated, reasons, instructions, bytes, micros,
    // flagged programs; translation failures per form.
    #[derive(Default, Clone)]
    struct Tally {
        ok: usize,
        reasons: BTreeMap<String, usize>,
        insts: u64,
        bytes: u64,
        micros: u64,
        flagged: usize,
        merged: usize,
    }
    let tallies = Mutex::new(vec![vec![Tally::default(); SETS.len()]; forms.len()]);
    let refused = Mutex::new(vec![BTreeMap::<String, usize>::new(); forms.len()]);
    let flags = Mutex::new(String::new());
    let diffs = Mutex::new(String::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::env::var("XNW_PACK_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(6);
    let started = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            let (next, records, forms, tallies, refused, flags, diffs) = (&next, &records, &forms, &tallies, &refused, &flags, &diffs);
            s.spawn(move || {
                let mut v = Validator::load(Validator::default_path().as_deref()).ok();
                loop {
                    let n = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((h, b)) = records.get(n) else { break };
                    if n % 2000 == 0 {
                        eprintln!("{n} of {} at {} s", records.len(), started.elapsed().as_secs());
                    }
                    for (fi, &pt) in forms.iter().enumerate() {
                        let descriptors = if pt { super::Descriptors::Gpu } else { super::Descriptors::Plan };
                        let opts = super::Options { mesh: super::MeshForm::Fallback, descriptors, pagetable: pt, ..Default::default() };
                        SWEEP.with(|w| *w.borrow_mut() = Some(Sweep { sets: SETS[1..].iter().map(|x| x.0).collect(), out: Vec::new() }));
                        let t0 = std::time::Instant::now();
                        let r = super::translate_with(b, opts);
                        let off_micros = t0.elapsed().as_micros() as u64;
                        let sweep = SWEEP.with(|w| w.borrow_mut().take()).unwrap_or_default();
                        let s = match r {
                            Ok(s) => s,
                            Err(e) => {
                                let e = e.to_string();
                                let reason = e.split(": ").skip(1).collect::<Vec<_>>().join(": ");
                                *refused.lock().unwrap()[fi].entry(reason.chars().take(90).collect()).or_default() += 1;
                                continue;
                            }
                        };
                        let mut validate = |dxil: &[u8]| -> (Result<(), String>, u64) {
                            let t = std::time::Instant::now();
                            let r = v.as_mut().map_or(Ok(()), |v| v.validate(&mut dxil.to_vec()).map(drop));
                            (r.map_err(|e| e.lines().next().unwrap_or("").chars().take(90).collect()), t.elapsed().as_micros() as u64)
                        };
                        let mut rows = Vec::new();
                        let (r0, m0) = validate(&s.dxil);
                        let before = sweep.out.first().map_or(0, |o| o.before);
                        rows.push((0usize, r0, before, s.dxil.len(), off_micros + m0, (Vec::new(), 0)));
                        for o in &sweep.out {
                            let k = SETS.iter().position(|x| x.0 == o.passes).unwrap_or(0);
                            let (r, m) = validate(&o.dxil);
                            rows.push((k, r, o.after, o.dxil.len(), o.micros + m, o.flags.clone()));
                        }
                        let ok0 = rows[0].1.is_ok();
                        let mut t = tallies.lock().unwrap();
                        for (k, r, insts, bytes, micros, (fl, merged)) in &rows {
                            let x = &mut t[fi][*k];
                            match r {
                                Ok(()) => x.ok += 1,
                                Err(e) => *x.reasons.entry(e.clone()).or_default() += 1,
                            }
                            if r.is_ok() || !ok0 {
                                x.insts += *insts as u64;
                                x.bytes += *bytes as u64;
                            }
                            x.micros += micros;
                            x.merged += merged;
                            let form = if pt { "pt" } else { "plan" };
                            if !fl.is_empty() {
                                x.flagged += 1;
                                let mut f = flags.lock().unwrap();
                                for l in fl.iter().take(5) {
                                    let _ = writeln!(f, "{form} {} {h} {l}", SETS[*k].1);
                                }
                            }
                            if r.is_ok() != ok0 {
                                let _ = writeln!(diffs.lock().unwrap(), "{form} {} {h} {}", SETS[*k].1, match r { Ok(()) => "validates (off does not)".to_owned(), Err(e) => e.clone() });
                            }
                        }
                    }
                }
            });
        }
    });
    let mut sum = String::new();
    let _ = writeln!(sum, "{} programs, {} s, {threads} threads", records.len(), started.elapsed().as_secs());
    let tallies = tallies.into_inner().unwrap();
    let refused = refused.into_inner().unwrap();
    for (fi, &pt) in forms.iter().enumerate() {
        let form = if pt { "pt" } else { "plan" };
        let _ = writeln!(sum, "== {form}: translation refused {} (same for every set)", refused[fi].values().sum::<usize>());
        for (k, x) in tallies[fi].iter().enumerate() {
            let _ = writeln!(sum, "{form} {:<5} validated {:>6} failed {:>5} insts {:>11} bytes {:>12} s {:>7.1} flagged {:>5} merges-ok {}", SETS[k].1, x.ok, x.reasons.values().sum::<usize>(), x.insts, x.bytes, x.micros as f64 / 1e6, x.flagged, x.merged);
            let mut rs: Vec<_> = x.reasons.iter().collect();
            rs.sort_by_key(|x| std::cmp::Reverse(*x.1));
            for (r, c) in rs.iter().take(4) {
                let _ = writeln!(sum, "    {c} {r}");
            }
        }
        let mut rs: Vec<_> = refused[fi].iter().collect();
        rs.sort_by_key(|x| std::cmp::Reverse(*x.1));
        for (r, c) in rs.iter().take(8) {
            let _ = writeln!(sum, "  refused {c} {r}");
        }
    }
    std::fs::create_dir_all(&out).expect("dir");
    std::fs::write(out.join("summary.txt"), &sum).expect("summary");
    std::fs::write(out.join("flags.txt"), flags.into_inner().unwrap()).expect("flags");
    std::fs::write(out.join("diffs.txt"), diffs.into_inner().unwrap()).expect("diffs");
    print!("{sum}");
}

/// Every record of the pack: `<n> <name> ok` or `<n> <name> <reason>`, one
/// line each, sorted, to `XNW_PACK_LIST` (for a before/after comparison of
/// two trees). `cargo test -p xnw-shader pack_listing -- --ignored`.
#[test]
#[ignore = "translates the whole pack"]
fn pack_listing() {
    let Some(out) = std::env::var_os("XNW_PACK_LIST") else { return };
    let pack = std::env::var_os("XNW_SHADER_PACK").map_or_else(|| PathBuf::from(r"game\shaders.rpf"), Into::into);
    let data = std::fs::read(pack).expect("pack");
    let archive = Archive::parse(&data).expect("archive");
    let mut records = Vec::new();
    for (i, e) in archive.entries.iter().enumerate() {
        if e.kind == EntryKind::Sgd5 {
            records.extend(sgd5::records(archive.read(i).expect("entry")));
        }
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let lines = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        let threads = std::env::var("XNW_PACK_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(12);
        let validate = std::env::var_os("XNW_PACK_VALIDATE").is_some();
        for _ in 0..threads {
            let (next, records, lines) = (&next, &records, &lines);
            s.spawn(move || {
                // XNW_PACK_VALIDATE: `ok` only for DXIL dxil.dll accepts.
                let mut v = if validate { Validator::load(Validator::default_path().as_deref()).ok() } else { None };
                loop {
                    let n = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(r) = records.get(n) else { break };
                    let started = std::time::Instant::now();
                    if let Ok(only) = std::env::var("XNW_PACK_ONLY")
                        && !only.split(',').any(|k| k.parse() == Ok(n))
                    {
                        continue;
                    }
                    let what = match r.unpack() {
                        Some(b) => match super::translate(&b) {
                            Ok(sh) => match v.as_mut().map(|v| v.validate(&mut sh.dxil.clone())) {
                                Some(Err(e)) => format!("validator: {}", e.lines().next().unwrap_or("")),
                                _ => "ok".to_owned(),
                            },
                            Err(e) => e.to_string().split(": ").skip(1).collect::<Vec<_>>().join(": "),
                        },
                        None => "unpack".to_owned(),
                    };
                    if started.elapsed().as_secs() >= 30 {
                        eprintln!("slow: {n:05} {} {} s", r.name.unwrap_or("-"), started.elapsed().as_secs());
                    }
                    let line = format!("{n:05} {} {what}", r.name.unwrap_or("-"));
                    lines.lock().unwrap().push(line);
                }
            });
        }
    });
    let mut lines = lines.into_inner().unwrap();
    lines.sort();
    std::fs::write(out, lines.join("\n")).expect("write");
}

/// The runtime stride table as xnw-gpu fills it (VIctory form: no
/// raw-view pool, every entry names its own slot): a load of element 2,
/// byte 4 of a buffer whose stride is unknown at translation, stored at
/// element 1, byte 8, read through the title's own view (structured of
/// 12, 16, 20 bytes, typed RGBA32F, raw) on this host's GPU. Program and
/// expectation from VIctory (mod.rs tests, 42538cd, GPL-3.0+).
#[test]
fn stride_table_reads_own_views() {
    use super::host::{Dispatch, Root, Stride, View};
    use super::StrideEntry;
    let words: &[u32] = &[
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x7E00_0282, // v_mov_b32 v0, 2
        0x7E02_0284, // v_mov_b32 v1, 4
        0xE030_3000,
        0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
        0x7E00_0281, // v_mov_b32 v0, 1
        0x7E02_0288, // v_mov_b32 v1, 8
        0xE070_3000,
        0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
        0xBF81_0000, // s_endpgm
    ];
    let used = |kind, table_offset_dwords| crate::UsedResource {
        kind,
        root: crate::RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let param = |root, sgprs, user_data, slot| crate::RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
    let native = crate::NativeProgram {
        stage: crate::HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![param(crate::RootType::Constants, 4, 0, 0), param(crate::RootType::Table, 1, 4, 1)],
        used: vec![used(crate::ResourceType::Srv, 0), used(crate::ResourceType::Uav, 8)],
        io: crate::native::StageIo::Cs(crate::native::ComputeIo { rsrc2: 5 << 1, threads: [1, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let l = super::lower::lower(&super::lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        fallback: false,
        fma: super::Fma::Exact,
        gpu_desc: false,
        probe: false,
        loop_counts: None,
        pagetable: false,
    })
    .unwrap();
    let cs = super::HostShader {
        stage: crate::HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        address_map: l.address_map,
        descriptor_words: l.descriptor_words,
    };
    assert!(cs.reads_stride_table());
    let Some(host) = Host::for_tests() else { return };
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    let data: Vec<u32> = (100..164).collect();
    for (kind, view, stride) in [
        ("raw", Stride::Raw, 4),
        ("structured 12", Stride::Structured(12), 12),
        ("structured 16", Stride::Structured(16), 16),
        ("structured 20", Stride::Structured(20), 20),
        ("typed RGBA32F", Stride::Typed, 16),
    ] {
        let views = [View::Buffer { data: &data, stride: view }, View::RawUav(64)];
        // Both entries carry the structure's stride (the UAV slot's raw
        // view is indexed by the same element size).
        let table: Vec<u32> = [stride, stride]
            .into_iter()
            .flat_map(|s| StrideEntry { stride: s, ..StrideEntry::default() }.words())
            .collect();
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(vec![0, 0, 0, 0, 0])), (1, Root::View { srv: true, data: table })],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        let mut want = vec![0; 64];
        want[((stride + 8) / 4) as usize] = data[((2 * stride + 4) / 4) as usize];
        assert_eq!(out[0], want, "{kind}");
        println!("{kind}: ok");
    }
}

/// XNW: micro-test for (b): 64 uniform constant
/// loads per thread through a root CBV against the same loads through a
/// raw view at a root-constant byte offset. Kernels in
/// tools/microbench/const_{cbv,raw}.hlsl, compiled by dxc to the files
/// `XNW_CONST_BENCH` names (`<dir>`, holding const_cbv.dxil and
/// const_raw.dxil). `cargo test -p xnw-shader constant_loads_cbv_vs_raw
/// -- --ignored --nocapture`.
#[test]
#[ignore = "a GPU timing"]
fn constant_loads_cbv_vs_raw() {
    use super::host::{Dispatch, Root, View};
    let Some(dir) = std::env::var_os("XNW_CONST_BENCH") else { return };
    let Some(host) = Host::for_tests() else { return };
    let user = (0..4).map(|k| super::UserData { index: k, slot: 0, root: crate::RootType::Constants, dword: k }).collect::<Vec<_>>();
    let data: Vec<u32> = (0..1024 + 64).collect();
    for (name, srv) in [("cbv", false), ("raw", true)] {
        let dxil = std::fs::read(std::path::Path::new(&dir).join(format!("const_{name}.dxil"))).unwrap();
        let s = super::HostShader {
            stage: crate::HwStage::Cs,
            dxil,
            user_data: user.clone(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: if srv { vec![] } else { vec![0] },
            root_srvs: if srv { vec![0] } else { vec![] },
            root_uavs: Vec::new(),
            address_map: false,
            descriptor_words: false,
        };
        let (root, pso) = host.compute_pipeline(&s).unwrap();
        let views = [View::RawUav(64)];
        let d = Dispatch {
            // The raw view reads from byte 16, as a table at a 16-byte offset.
            roots: vec![(0, Root::Constants(vec![if srv { 16 } else { 0 }, 0, 0, 0])), (1, Root::View { srv, data: data.clone() })],
            views: &views,
            samplers: &[],
            groups: [16384, 1, 1],
        };
        let mut best = f64::MAX;
        for _ in 0..5 {
            let (_, t) = host.dispatch_timed(&root, &pso, &d, 20).unwrap();
            best = best.min(t);
        }
        println!("{name}: {:.3} ms per dispatch of 1M threads x 64 loads", best / 20.0 * 1e3);
    }
}

/// XNW: GPU-side descriptors. A program loads two buffer records from
/// its table (`s_load_dwordx4` at the table and 32 bytes on), reads dword 1
/// of the first buffer and stores it at dword 2 of the second. The table's
/// records sit in a "window" (a raw view, heap slot 0) at byte `at`, which
/// the host passes as user data 32 (view) and 33 (offset). Records carry the
/// heap index in dword 0 and the tag in dword 3; a record without the tag
/// reads the null of its kind, slot 2 for a raw SRV (here a buffer of 900s,
/// to tell it apart).
/// XNW: stages a, c, d): with GPU-side descriptors and the
/// page table on, a vector buffer load, a scalar buffer load (constants),
/// a store and an atomic through records go through the page table only
/// (`t0` space 4, listed in PSV0; the container's PRIV part says nothing
/// still needs host copies); off, the program is as before and has no
/// PRIV part. Both validate. No device.
#[test]
fn page_table_reads_tagged_buffer_records() {
    let words: &[u32] = &[
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x7E02_0284, // v_mov_b32 v1, 4
        0xE030_1000,
        0x8002_0201, // buffer_load_dword v2, v1, s[8:11], 0 offen
        0xF420_0404,
        0xFA00_0004, // s_buffer_load_dword s16, s[8:11], 0x4
        0x4A04_0410, // v_add_nc_u32 v2, s16, v2
        0x7E02_0288, // v_mov_b32 v1, 8
        0xE070_1000,
        0x8003_0201, // buffer_store_dword v2, v1, s[12:15], 0 offen
        0xE0C8_5000,
        0x8003_0201, // buffer_atomic_add v2, v1, s[12:15], 0 offen glc
        0xBF81_0000, // s_endpgm
    ];
    let used = |kind, table_offset_dwords| crate::UsedResource { kind, root: crate::RootType::Table, table_offset_dwords, lower_bound: 0, stride: 0, dwords: 4, user_data: 4, slot: 1 };
    let param = |root, sgprs, user_data, slot| crate::RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
    let native = crate::NativeProgram {
        stage: crate::HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![param(crate::RootType::Constants, 4, 0, 0), param(crate::RootType::Table, 1, 4, 1)],
        used: vec![used(crate::ResourceType::Srv, 0), used(crate::ResourceType::Uav, 8)],
        io: crate::native::StageIo::Cs(crate::native::ComputeIo { rsrc2: 5 << 1, threads: [1, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = crate::resource::track_with(&p, &native.layout, native.stage, true).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let mut validator = Validator::load(Validator::default_path().as_deref()).ok();
    let mut sizes = Vec::new();
    for pagetable in [false, true] {
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            fma: super::Fma::Exact,
            fallback: false,
            gpu_desc: true,
            probe: false,
            loop_counts: None,
            pagetable,
        })
        .unwrap();
        let cs = super::HostShader { stage: crate::HwStage::Cs, dxil: l.dxil, user_data: l.user_data, samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: l.root_cbvs, root_srvs: l.root_srvs, root_uavs: l.root_uavs, address_map: l.address_map, descriptor_words: l.descriptor_words };
        assert_eq!(cs.reads_page_table(), pagetable);
        assert_eq!(cs.host_copies(), pagetable.then_some(0));
        // Every heap index is bounded by the heap-limit constants.
        assert!(cs.reads_heap_bounds());
        if let Some(v) = validator.as_mut() {
            v.validate(&mut cs.dxil.clone()).unwrap();
        }
        sizes.push(cs.dxil.len());
    }
    println!("DXIL bytes without / with the page table: {sizes:?}");
}

/// XNW: stage b): a constant the code reads through a V# it builds
/// around a root SRV (the root descriptor's address in s[4:5]) goes through
/// the page table at the address the runtime puts in user data (root type
/// Descriptor, dwords 0 and 1), not a root SRV binding, with the switch on;
/// off, the root SRV stays. Both validate. No device.
#[test]
fn page_table_reads_root_views_by_address() {
    let words: &[u32] = &[
        0xBE88_0304, // s_mov_b32 s8, s4
        0xBE89_0305, // s_mov_b32 s9, s5
        0xBE8A_03C1, // s_mov_b32 s10, -1
        0xBE8B_03FF,
        0x1101_4FAC, // s_mov_b32 s11, 0x11014fac
        0xF420_0304,
        0xFA00_0010, // s_buffer_load_dword s12, s[8:11], 0x10
        0xBF06_800C, // s_cmp_eq_u32 s12, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xBF81_0000, // s_endpgm
    ];
    let native = crate::NativeProgram {
        stage: crate::HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            crate::RootParam { root: crate::RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
            crate::RootParam { root: crate::RootType::Descriptor, table_offset_dwords: 0, sgprs: 2, base_register: 0, user_data: 4, slot: 1 },
        ],
        used: vec![crate::UsedResource { kind: crate::ResourceType::Srv, root: crate::RootType::Descriptor, table_offset_dwords: 0, lower_bound: 0, stride: 0, dwords: 4, user_data: 4, slot: 1 }],
        io: crate::native::StageIo::Cs(crate::native::ComputeIo { rsrc2: 6 << 1, threads: [1, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = crate::resource::track_with(&p, &native.layout, native.stage, true).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let mut validator = Validator::load(Validator::default_path().as_deref()).ok();
    for pagetable in [false, true] {
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            fma: super::Fma::Exact,
            fallback: false,
            gpu_desc: true,
            probe: false,
            loop_counts: None,
            pagetable,
        })
        .unwrap_or_else(|e| panic!("pagetable {pagetable}: {e:?}"));
        let cs = super::HostShader { stage: crate::HwStage::Cs, dxil: l.dxil, user_data: l.user_data, samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: l.root_cbvs, root_srvs: l.root_srvs, root_uavs: l.root_uavs, address_map: l.address_map, descriptor_words: l.descriptor_words };
        assert_eq!(cs.reads_page_table(), pagetable);
        if pagetable {
            assert!(cs.root_srvs.is_empty() && cs.root_cbvs.is_empty(), "{:?} {:?}", cs.root_srvs, cs.root_cbvs);
            let addr = cs.user_data.iter().filter(|u| u.root == crate::RootType::Descriptor).count();
            assert_eq!(addr, 2, "{:?}", cs.user_data);
            assert_eq!(cs.host_copies(), Some(0));
        }
        if let Some(v) = validator.as_mut() {
            v.validate(&mut cs.dxil.clone()).unwrap();
        }
    }
}
/// XNW: a bindless texture read (a T# loaded from a table, plan
/// mode: a heap index) bounds its heap index by the heap-limit constants
/// (`b0` in HEAP_BOUNDS_SPACE, listed in PSV0); the DXIL validates.
#[test]
fn bindless_texture_reads_bound_their_heap_index() {
    let words: &[u32] = &[
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0202,
        0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
        0xF000_1F08,
        0x0002_0000, // image_load v[0:3], v[0:1], s[8:15] dmask:0xf dim:2D unorm
        0xBF81_0000, // s_endpgm
    ];
    let param = |root, sgprs, user_data, slot| crate::RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
    let native = crate::NativeProgram {
        stage: crate::HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![param(crate::RootType::Constants, 4, 0, 0), param(crate::RootType::Table, 1, 4, 1)],
        used: vec![crate::UsedResource { kind: crate::ResourceType::Srv, root: crate::RootType::Table, table_offset_dwords: 0, lower_bound: 0, stride: 0, dwords: 8, user_data: 4, slot: 1 }],
        io: crate::native::StageIo::Cs(crate::native::ComputeIo { rsrc2: 5 << 1, threads: [1, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let l = super::lower::lower(&super::lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        fma: super::Fma::Exact,
        fallback: false,
        gpu_desc: false,
        probe: false,
        loop_counts: None,
        pagetable: false,
    })
    .unwrap();
    let cs = super::HostShader { stage: crate::HwStage::Cs, dxil: l.dxil, user_data: l.user_data, samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: l.root_cbvs, root_srvs: l.root_srvs, root_uavs: l.root_uavs, address_map: l.address_map, descriptor_words: l.descriptor_words };
    assert!(cs.reads_heap_bounds());
    if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
        v.validate(&mut cs.dxil.clone()).unwrap();
    }
}
#[test]
fn gpu_descriptor_records() {
    use super::host::{Dispatch, Root, Stride, View};
    let words: &[u32] = &[
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x7E02_0284, // v_mov_b32 v1, 4
        0xE030_1000,
        0x8002_0201, // buffer_load_dword v2, v1, s[8:11], 0 offen
        0x7E02_0288, // v_mov_b32 v1, 8
        0xE070_1000,
        0x8003_0201, // buffer_store_dword v2, v1, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let used = |kind, table_offset_dwords| crate::UsedResource {
        kind,
        root: crate::RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let param = |root, sgprs, user_data, slot| crate::RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
    let native = crate::NativeProgram {
        stage: crate::HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![param(crate::RootType::Constants, 4, 0, 0), param(crate::RootType::Table, 1, 4, 1)],
        used: vec![used(crate::ResourceType::Srv, 0), used(crate::ResourceType::Uav, 8)],
        io: crate::native::StageIo::Cs(crate::native::ComputeIo { rsrc2: 5 << 1, threads: [1, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = crate::resource::track_with(&p, &native.layout, native.stage, true).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let l = super::lower::lower(&super::lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        fma: super::Fma::Exact,
        fallback: false,
        gpu_desc: true,
        probe: false,
        loop_counts: None,
        pagetable: false,
    })
    .unwrap();
    let cs = super::HostShader {
        stage: crate::HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        address_map: l.address_map,
        descriptor_words: l.descriptor_words,
    };
    // Window dwords right after the table parameter (user data 4, 1 SGPR).
    assert!(cs.user_data.iter().any(|u| u.index == 5 && u.dword == super::WINDOW_DWORD), "{:?}", cs.user_data);
    if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
        v.validate(&mut cs.dxil.clone()).unwrap();
    }
    let Some(host) = Host::for_tests() else { return };
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    let data: Vec<u32> = (100..164).collect();
    let nulls: Vec<u32> = (900..964).collect();
    let tag = super::RECORD_TAG;
    for (case, tagged, at) in [("records at 0", true, 0u32), ("records at 64", true, 64), ("untagged: the null", false, 0)] {
        let mut window = vec![0u32; 64];
        let rec = |w: &mut Vec<u32>, at: usize, index: u32, kind: u32| {
            w[at..at + 8].copy_from_slice(&[index, 7, 1, if tagged { tag } else { 0x1234 }, kind, !0, !0, !0]);
        };
        rec(&mut window, at as usize / 4, 1, 2);
        rec(&mut window, at as usize / 4 + 8, 3, 6);
        // The UAV's record keeps its tag in the untagged case.
        window[at as usize / 4 + 8 + 3] = tag;
        let views = [
            View::Buffer { data: &window, stride: Stride::Raw },
            View::Buffer { data: &data, stride: Stride::Raw },
            View::Buffer { data: &nulls, stride: Stride::Raw },
            View::RawUav(64),
        ];
        let mut ud = vec![0u32; cs.user_data_dwords() as usize];
        ud[5] = 0;
        ud[6] = at;
        let out = host
            .dispatch(&root, &pso, &Dispatch { roots: vec![(0, Root::Constants(ud))], views: &views, samplers: &[], groups: [1, 1, 1] })
            .unwrap();
        let want = if tagged { 101 } else { 901 };
        assert_eq!(out[0][2], want, "{case}");
        println!("{case}: {}", out[0][2]);
    }
}
