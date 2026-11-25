//! `xnw shader-info <rpf> [--census] [--disasm <name-or-hash>] [--dump <dir> [--threads N]]`:
//! what is in the title's shader pack.
//!
//! Without options: the archive header, its entries and the record count.
//! `--census` reads every record on all cores: unpack, container, DXIL part,
//! native GFX10 programs, instruction decode, and prints the totals.
//! `--disasm` prints the native programs of the records whose name contains
//! the given text (a name part or a hash), in SP3 syntax.
//! `--dump` writes every distinct record (by the 128-bit hash of the
//! unpacked container) to `<dir>/<hash[..2]>/<hash>.txt`: the `--disasm`
//! text plus one `read` line per memory read with its traced descriptor.
//! Files already there are skipped, so a stopped run resumes; each worker
//! appends `hash, name, kind, offset` of every record to `records-<w>.tsv`.
//!
//! The census follows the shape of VIctory's `shader-info`
//! (`apps/victory/src/shader.rs`, commit d2a4bae1, GPL-3.0-or-later); the code
//! here is new.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use xnw_shader::gfx10::{self, Encoding, Fields, Instruction, Op};
use xnw_shader::rpf9::{self, Archive, EntryKind, ZlibRun};
use xnw_shader::sgd5::{self, Record};
use xnw_shader::{Container, DxilProgram, FourCC, HwStage, NativeProgram, ShaderError, native_programs, xdat_entries};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage: xnw shader-info <rpf> [--census] [--disasm <name-or-hash>] [--dump <dir> [--threads N]]";
/// Export target of `exp prim` (NGG primitive export).
const EXP_PRIM: u8 = 20;

pub fn run(args: &[String]) -> Res<()> {
    let path = args.first().ok_or(USAGE)?;
    let census = args.iter().any(|a| a == "--census");
    let disasm = match args.iter().position(|a| a == "--disasm") {
        Some(i) => Some(args.get(i + 1).ok_or(USAGE)?.as_str()),
        None => None,
    };
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1));
    let dump_dir = value("--dump");
    let threads = match value("--threads") {
        Some(t) => t.parse::<usize>().map_err(|_| USAGE)?,
        None => std::thread::available_parallelism().map_or(4, |n| n.get()),
    };

    let start = Instant::now();
    let data = std::fs::read(path)?;
    // One container (`XNW_SHADER_DUMP`'s files): its native programs.
    if path.ends_with(".dxbc") {
        let container = Container::parse(&data)?;
        let dxil = container.part(FourCC::DXIL).ok_or("no DXIL part")?;
        for n in &native_programs(dxil.data)? {
            print_program(n);
        }
        return Ok(());
    }
    let read_time = start.elapsed();
    let archive = Archive::parse(&data)?;
    let h = &archive.header;
    println!("archive    {path} ({} bytes, read in {:.2} s)", data.len(), read_time.as_secs_f64());
    println!(
        "header     toc [0x40, {:#x}) split {:#x}, trailer {} B, id {:#018x}, flags {:#x}, tag {:#x}, {:#x}",
        h.toc_end, h.toc_split, h.trailer_len, h.archive_id, h.flags, h.tag, h.stride_flag
    );
    println!(
        "entries    {} found, header says {}{}",
        archive.entries.len(),
        h.entry_count,
        if archive.count_matches() { "" } else { "  (MISMATCH)" }
    );
    let mut records: Vec<(usize, Record<'_>)> = Vec::new();
    for (i, e) in archive.entries.iter().enumerate() {
        let bytes = archive.read(i).expect("entry in bounds");
        let mut line = format!("  #{i} {:?} at {:#x}, {} B", e.kind, e.offset, e.size);
        if e.kind == EntryKind::Sgd5 {
            let before = records.len();
            records.extend(sgd5::records(bytes).map(|r| (e.offset, r)));
            line += &format!(", {} records", records.len() - before);
        }
        println!("{line}");
    }
    println!("records    {} (scan {:.2} s)", records.len(), start.elapsed().as_secs_f64());

    if let Some(pattern) = disasm {
        disassemble(&records, pattern);
    }
    if let Some(dir) = dump_dir {
        dump(&records, Path::new(dir), threads.max(1), start)?;
    }
    if census {
        let zlib: Vec<&[u8]> = archive
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.kind == EntryKind::Zlib)
            .map(|(i, _)| archive.read(i).expect("entry in bounds"))
            .collect();
        run_census(&records, &zlib, start);
    }
    Ok(())
}

fn disassemble(records: &[(usize, Record<'_>)], pattern: &str) {
    let needle = pattern.to_ascii_lowercase();
    let mut shown = 0;
    for (base, r) in records {
        let Some(name) = r.name.filter(|n| n.to_ascii_lowercase().contains(&needle)) else {
            continue;
        };
        shown += 1;
        println!("\n{name} @{:#x} kind {:#x}", base + r.offset, r.kind);
        let mut text = String::new();
        match r.unpack() {
            Some(bytes) => record_text(&bytes, false, &mut text),
            None => text.push_str("  corrupt record\n"),
        }
        print!("{text}");
    }
    println!("\n{shown} records match {pattern:?}");
}

/// The `--disasm` text of one unpacked record; with `reads`, each
/// program's memory reads follow its instructions ([`reads_text`]).
fn record_text(bytes: &[u8], reads: bool, out: &mut String) {
    let container = match Container::parse(bytes) {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(out, "  {e}");
            return;
        }
    };
    let parts: Vec<String> = container.parts.iter().map(|p| format!("{} {}", p.fourcc, p.data.len())).collect();
    let _ = writeln!(out, "  parts  {}", parts.join(", "));
    let Some(dxil) = container.part(FourCC::DXIL) else {
        return;
    };
    match DxilProgram::parse(dxil.data) {
        Ok(p) => {
            let x = p.xbox_extras();
            let _ = writeln!(
                out,
                "  dxil   {:?} sm{}.{}, bitcode {}, xdat {}, xbop {}",
                p.kind,
                p.shader_model.0,
                p.shader_model.1,
                if p.stripped { "stripped".to_owned() } else { format!("{} B", p.bitcode.len()) },
                x.xdat,
                x.xbop
            );
        }
        Err(e) => _ = writeln!(out, "  dxil   {e}"),
    }
    match native_programs(dxil.data) {
        Ok(programs) => {
            for n in &programs {
                program_text(n, out);
                if reads {
                    reads_text(n, out);
                }
            }
        }
        Err(e) => _ = writeln!(out, "  native {e}"),
    }
}

fn print_program(n: &NativeProgram) {
    let mut text = String::new();
    program_text(n, &mut text);
    print!("{text}");
}

fn program_text(n: &NativeProgram, out: &mut String) {
    let _ = writeln!(
        out,
        "  native {:?} wave{} vgprs {} code {} B",
        n.stage,
        if n.wave64 { 64 } else { 32 },
        n.vgprs,
        n.code.len()
    );
    for p in &n.layout {
        let _ = writeln!(out, "    root {:?}", p);
    }
    for u in &n.used {
        let _ = writeln!(out, "    used {:?}", u);
    }
    match gfx10::decode(&n.code) {
        Ok(insts) => {
            for i in &insts {
                let words: Vec<String> = i.words.iter().map(|w| format!("{w:08X}")).collect();
                let text = match i.op {
                    Op::Known(_) => i.text(n.wave64),
                    Op::Unknown => format!("<unknown {:?} {:#x}>", i.encoding, i.opcode),
                };
                let _ = writeln!(out, "    {:06x}: {:<44} {text}", i.offset, words.join(" "));
            }
        }
        Err(e) => _ = writeln!(out, "    {e}"),
    }
}

/// One line per memory instruction of `n`, with the descriptor or
/// address it uses traced to the root signature (as a survey of the programs names
/// them): `read <offset> <mnemonic> <where> +<immediate>[+s]`, where is
/// `T9+0x80#0` (dwords 0.. of what a scalar load reads at byte 0x80 of
/// the table in user data 9), `B[R11+0x0]` (a descriptor built around a
/// root view), `H..` (heap), `C+..` (the code's own constants), `K..`
/// (root constants), `G` (an address only known at run time), `?` (not
/// traced); `+s` when an SGPR offset is added. Images add `samp <where>`.
/// A program the tracker refuses gets `reads untraced: <why>` instead.
fn reads_text(n: &NativeProgram, out: &mut String) {
    use xnw_shader::RootType;
    use xnw_shader::resource::{self, Address, Role, Source};
    let traced = xnw_shader::ir::build(&n.code, n.wave64, n.vgprs)
        .map_err(|e| e.to_string())
        .and_then(|p| resource::track_with(&p, &n.layout, n.stage, true).map(|r| (p, r)).map_err(|e| e.to_string()));
    let (p, res): (xnw_shader::ir::Program, resource::Resources) = match traced {
        Ok(t) => t,
        Err(e) => {
            let _ = writeln!(out, "    reads untraced: {e}");
            return;
        }
    };
    let scalar: HashMap<(usize, usize), &Address> = res.scalar.iter().map(|(s, a)| (*s, a)).collect();
    let param = |a: &Address| -> String {
        match a.param {
            resource::CODE => return format!("C+{:#x}", a.offset),
            resource::GLOBAL => return "G".into(),
            _ => {}
        }
        let dynamic = if a.dynamic.is_some() || a.rel.is_some() { "+dyn" } else { "" };
        match n.layout.get(a.param) {
            Some(r) => {
                let kind = match r.root {
                    RootType::Table => "T",
                    RootType::ResourceHeap => "H",
                    RootType::Descriptor => "R",
                    RootType::Constants => "K",
                    _ => "P",
                };
                format!("{kind}{}+{:#x}{dynamic}", r.user_data, a.offset)
            }
            None => format!("P{}+{:#x}{dynamic}", a.param, a.offset),
        }
    };
    let descriptor = |site: (usize, usize), role: Role| -> Option<String> {
        let d = res.descriptors.iter().find(|d| d.site == site && d.role == role)?;
        let name = |s: &Source| match s {
            Source::Loaded { load, first } => scalar.get(load).map_or("L?".into(), |a| format!("{}#{first}", param(a))),
            Source::Built { address, .. } => format!("B[{}]", param(address)),
            Source::Constant(_) => "const".into(),
            Source::Runtime => "runtime".into(),
            Source::Phi { .. } => "phi".into(),
        };
        Some(d.source.leaves().into_iter().map(name).collect::<Vec<_>>().join("|"))
    };
    let sgpr = |o: &gfx10::Operand| matches!(o, gfx10::Operand::Scalar { code, .. } if *code != gfx10::NULL);
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            let at = inst.machine.offset;
            let line = match &inst.fields {
                Fields::Smem { offset, soffset, .. } => {
                    if !name.starts_with("s_buffer_") {
                        scalar.get(&(b, i)).map_or("?".into(), |a| param(a))
                    } else {
                        let place = descriptor((b, i), Role::Resource).unwrap_or_else(|| "?".into());
                        format!("{place} +{offset:#x}{}", if sgpr(soffset) { "+s" } else { "" })
                    }
                }
                Fields::Buffer { offset, soffset, format, .. } => {
                    let place = descriptor((b, i), Role::Resource).unwrap_or_else(|| "?".into());
                    let f = format.map_or(String::new(), |f| format!(" f{f}"));
                    format!("{place} +{offset:#x}{}{f}", if sgpr(soffset) { "+s" } else { "" })
                }
                Fields::Image { .. } => {
                    let mut t = descriptor((b, i), Role::Resource).unwrap_or_else(|| "?".into());
                    if let Some(s) = descriptor((b, i), Role::Sampler) {
                        t += &format!(" samp {s}");
                    }
                    t
                }
                Fields::Flat { .. } => scalar.get(&(b, i)).map_or("?".into(), |a| param(a)),
                _ => continue,
            };
            let _ = writeln!(out, "    read {at:06x} {name} {line}");
        }
    }
}

/// `--dump`: every distinct record to `dir` on `threads` workers.
fn dump(records: &[(usize, Record<'_>)], dir: &Path, threads: usize, start: Instant) -> Res<()> {
    std::fs::create_dir_all(dir)?;
    let next = AtomicUsize::new(0);
    let claimed = Mutex::new(HashSet::new());
    let [written, skipped, same] = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)];
    std::thread::scope(|s| -> Res<()> {
        let workers: Vec<_> = (0..threads)
            .map(|w| {
                let (next, claimed, written, skipped, same) = (&next, &claimed, &written, &skipped, &same);
                s.spawn(move || -> std::io::Result<()> {
                    let mut list = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("records-{w}.tsv")))?;
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        let Some((base, r)) = records.get(k) else { break };
                        let Some(bytes) = r.unpack() else { continue };
                        let hash = xnw_core::hash128(&[&bytes]);
                        writeln!(list, "{hash:032x}\t{}\t{:#x}\t{:#x}", r.name.unwrap_or("-"), r.kind, base + r.offset)?;
                        if !claimed.lock().expect("claimed").insert(hash) {
                            same.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        let path = dump_path(dir, hash);
                        if path.exists() {
                            skipped.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        let mut text = format!("# {hash:032x} {} kind {:#x} @{:#x}\n", r.name.unwrap_or("-"), r.kind, base + r.offset);
                        record_text(&bytes, true, &mut text);
                        std::fs::create_dir_all(path.parent().expect("parent"))?;
                        let tmp = path.with_extension("tmp");
                        std::fs::write(&tmp, text)?;
                        std::fs::rename(&tmp, &path)?;
                        let n = written.fetch_add(1, Ordering::Relaxed) + 1;
                        if n % 1000 == 0 {
                            println!("dump  {n} written, {k} of {} records, {:.0} s", records.len(), start.elapsed().as_secs_f64());
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        for w in workers {
            w.join().expect("dump worker")?;
        }
        Ok(())
    })?;
    let line = format!(
        "dump  {} records: {} written, {} already there, {} repeats; {:.1} s on {threads} threads\n",
        records.len(),
        written.into_inner(),
        skipped.into_inner(),
        same.into_inner(),
        start.elapsed().as_secs_f64()
    );
    print!("{line}");
    std::fs::write(dir.join("done.txt"), line)?;
    Ok(())
}

/// Where `--dump` puts the record with this hash: 256 folders by the first byte.
fn dump_path(dir: &Path, hash: u128) -> std::path::PathBuf {
    let h = format!("{hash:032x}");
    dir.join(&h[..2]).join(format!("{h}.txt"))
}

/// Features to count programs by.
#[derive(Default, Clone, Copy)]
struct Features {
    exp_prim: bool,
    atomic64: bool,
    ray: bool,
}

#[derive(Default)]
struct Totals {
    records: usize,
    unpacked_bytes: u64,
    record_kinds: BTreeMap<u8, usize>,
    errors: BTreeMap<String, usize>,
    dxil_kinds: BTreeMap<String, usize>,
    models: BTreeMap<(u8, u8), usize>,
    xdat: usize,
    xbop: usize,
    stripped: usize,
    xdat_kinds: BTreeMap<u32, usize>,
    ray_stages: BTreeMap<HwStage, usize>,
    with_native: usize,
    programs: usize,
    stages: BTreeMap<HwStage, usize>,
    wave: [usize; 2],
    stage_wave: BTreeMap<(HwStage, bool), usize>,
    /// Records whose programs are all wave32, all wave64, or mixed.
    record_wave: [usize; 3],
    instructions: u64,
    code_bytes: u64,
    mnemonics: HashMap<&'static str, u64>,
    unknown: HashMap<(Encoding, u16), u64>,
    /// Programs and records using each feature.
    mesh: [usize; 2],
    geometry: [usize; 2],
    ngg_prim: [usize; 2],
    atomic64: [usize; 2],
    ray: [usize; 2],
}

impl Totals {
    fn error(&mut self, what: String) {
        *self.errors.entry(what).or_default() += 1;
    }

    fn merge(&mut self, o: Totals) {
        fn add<K: Ord>(a: &mut BTreeMap<K, usize>, b: BTreeMap<K, usize>) {
            for (k, v) in b {
                *a.entry(k).or_default() += v;
            }
        }
        fn addh<K: std::hash::Hash + Eq>(a: &mut HashMap<K, u64>, b: HashMap<K, u64>) {
            for (k, v) in b {
                *a.entry(k).or_default() += v;
            }
        }
        let pair = |a: &mut [usize; 2], b: [usize; 2]| {
            a[0] += b[0];
            a[1] += b[1];
        };
        self.records += o.records;
        self.unpacked_bytes += o.unpacked_bytes;
        add(&mut self.record_kinds, o.record_kinds);
        add(&mut self.errors, o.errors);
        add(&mut self.dxil_kinds, o.dxil_kinds);
        add(&mut self.models, o.models);
        self.xdat += o.xdat;
        self.xbop += o.xbop;
        self.stripped += o.stripped;
        add(&mut self.xdat_kinds, o.xdat_kinds);
        add(&mut self.ray_stages, o.ray_stages);
        self.with_native += o.with_native;
        self.programs += o.programs;
        add(&mut self.stages, o.stages);
        pair(&mut self.wave, o.wave);
        add(&mut self.stage_wave, o.stage_wave);
        for i in 0..3 {
            self.record_wave[i] += o.record_wave[i];
        }
        self.instructions += o.instructions;
        self.code_bytes += o.code_bytes;
        addh(&mut self.mnemonics, o.mnemonics);
        addh(&mut self.unknown, o.unknown);
        pair(&mut self.mesh, o.mesh);
        pair(&mut self.geometry, o.geometry);
        pair(&mut self.ngg_prim, o.ngg_prim);
        pair(&mut self.atomic64, o.atomic64);
        pair(&mut self.ray, o.ray);
    }

    fn record(&mut self, r: &Record<'_>) {
        self.records += 1;
        *self.record_kinds.entry(r.kind).or_default() += 1;
        let Some(bytes) = r.unpack() else {
            return self.error("record does not inflate".into());
        };
        self.unpacked_bytes += bytes.len() as u64;
        let container = match Container::parse(&bytes) {
            Ok(c) => c,
            Err(e) => return self.error(format!("container: {e}")),
        };
        let Some(dxil) = container.part(FourCC::DXIL) else {
            return self.error("container has no DXIL part".into());
        };
        match DxilProgram::parse(dxil.data) {
            Ok(p) => {
                *self.dxil_kinds.entry(format!("{:?}", p.kind)).or_default() += 1;
                *self.models.entry(p.shader_model).or_default() += 1;
                let x = p.xbox_extras();
                self.xdat += usize::from(x.xdat);
                self.xbop += usize::from(x.xbop);
                self.stripped += usize::from(p.stripped);
            }
            Err(e) => self.error(format!("dxil program: {e}")),
        }
        match xdat_entries(dxil.data) {
            Ok(list) => {
                for (kind, ..) in list {
                    *self.xdat_kinds.entry(kind).or_default() += 1;
                }
            }
            Err(e) => self.error(format!("xdat: {e}")),
        }
        let programs = match native_programs(dxil.data) {
            Ok(p) => p,
            Err(e) => return self.error(format!("native: {e}")),
        };
        if programs.is_empty() {
            return self.error("no native program".into());
        }
        self.with_native += 1;
        let waves64 = programs.iter().filter(|p| p.wave64).count();
        self.record_wave[match waves64 {
            0 => 0,
            n if n == programs.len() => 1,
            _ => 2,
        }] += 1;
        let mut any = Features::default();
        let (mut mesh, mut geometry) = (false, false);
        for n in &programs {
            self.programs += 1;
            *self.stages.entry(n.stage).or_default() += 1;
            self.wave[usize::from(n.wave64)] += 1;
            *self.stage_wave.entry((n.stage, n.wave64)).or_default() += 1;
            self.code_bytes += n.code.len() as u64;
            let is_mesh = n.stage == HwStage::Ms;
            let is_geometry = n.stage == HwStage::Gs;
            self.mesh[0] += usize::from(is_mesh);
            self.geometry[0] += usize::from(is_geometry);
            mesh |= is_mesh;
            geometry |= is_geometry;
            let insts = match gfx10::decode(&n.code) {
                Ok(i) => i,
                Err(e) => {
                    self.error(format!("decode: {}", reason(e)));
                    continue;
                }
            };
            let f = self.instructions(&insts);
            self.ngg_prim[0] += usize::from(f.exp_prim);
            self.atomic64[0] += usize::from(f.atomic64);
            self.ray[0] += usize::from(f.ray);
            if f.ray {
                *self.ray_stages.entry(n.stage).or_default() += 1;
            }
            any.exp_prim |= f.exp_prim;
            any.atomic64 |= f.atomic64;
            any.ray |= f.ray;
        }
        self.mesh[1] += usize::from(mesh);
        self.geometry[1] += usize::from(geometry);
        self.ngg_prim[1] += usize::from(any.exp_prim);
        self.atomic64[1] += usize::from(any.atomic64);
        self.ray[1] += usize::from(any.ray);
    }

    fn instructions(&mut self, insts: &[Instruction]) -> Features {
        let mut f = Features::default();
        self.instructions += insts.len() as u64;
        for i in insts {
            match i.op {
                Op::Known(name) => {
                    *self.mnemonics.entry(name).or_default() += 1;
                    f.atomic64 |= is_atomic64(name);
                    f.ray |= name.contains("_bvh");
                    if i.encoding == Encoding::Exp
                        && let Fields::Exp { target, .. } = i.fields(true)
                    {
                        f.exp_prim |= target == EXP_PRIM;
                    }
                }
                Op::Unknown => *self.unknown.entry((i.encoding, i.opcode)).or_default() += 1,
            }
        }
        f
    }
}

/// 64-bit atomics: `_x2` buffer/flat/global/scalar forms, `_u64`/`_i64`/`_b64`/`_f64` LDS forms.
fn is_atomic64(name: &str) -> bool {
    if name.contains("atomic") {
        return name.ends_with("_x2");
    }
    name.starts_with("ds_")
        && !name.contains("read")
        && !name.contains("write")
        && !name.contains("wrxchg")
        && ["_u64", "_i64", "_b64", "_f64"].iter().any(|t| name.ends_with(t))
}

fn reason(e: ShaderError) -> String {
    match e {
        ShaderError::Isa { reason, .. } => reason.to_owned(),
        e => e.to_string(),
    }
}

fn run_census(records: &[(usize, Record<'_>)], zlib: &[&[u8]], start: Instant) {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = AtomicUsize::new(0);
    const CHUNK: usize = 64;
    let mut total = Totals::default();
    let mut zlib_results = Vec::new();
    std::thread::scope(|s| {
        let zl: Vec<_> = zlib.iter().map(|z| s.spawn(move || rpf9::zlib_run(z))).collect();
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut t = Totals::default();
                    loop {
                        let first = next.fetch_add(CHUNK, Ordering::Relaxed);
                        if first >= records.len() {
                            break;
                        }
                        for (_, r) in &records[first..(first + CHUNK).min(records.len())] {
                            t.record(r);
                        }
                    }
                    t
                })
            })
            .collect();
        for w in workers {
            total.merge(w.join().expect("census worker"));
        }
        zlib_results = zl.into_iter().map(|h| h.join().expect("zlib worker")).collect();
    });
    let wall = start.elapsed();
    print_census(&total, &zlib_results);
    println!("\nwall time  {:.2} s on {threads} threads (read + scan + census)", wall.as_secs_f64());
}

fn print_census(t: &Totals, zlib: &[ZlibRun]) {
    println!("\ncensus");
    println!("  records              {} ({} MB unpacked)", t.records, t.unpacked_bytes >> 20);
    for (k, n) in &t.record_kinds {
        println!("    record kind {k:#04x}   {n}");
    }
    let failures: usize = t.errors.values().sum();
    println!("  read failures        {failures}");
    for (e, n) in &t.errors {
        println!("    {n:>8}  {e}");
    }
    println!("  DXIL stage (records)");
    for (k, n) in &t.dxil_kinds {
        println!("    {k:<18} {n}");
    }
    let models: Vec<String> = t.models.iter().map(|((a, b), n)| format!("sm{a}.{b} {n}")).collect();
    println!("  shader models        {}", models.join(", "));
    println!("  XDAT / xbop          {} / {}", t.xdat, t.xbop);
    println!("  bitcode stripped     {} (NOBC)", t.stripped);
    let kinds: Vec<String> = t.xdat_kinds.iter().map(|(k, n)| format!("{k:#04x} {n}")).collect();
    println!("  XDAT entry kinds     {}", kinds.join(", "));
    println!("  with native code     {} records, {} programs", t.with_native, t.programs);
    println!("  hardware stage (programs)       wave32   wave64");
    for (stage, n) in &t.stages {
        let w = |w64| t.stage_wave.get(&(*stage, w64)).copied().unwrap_or(0);
        println!("    {:<12} {n:>8}  {:>12} {:>8}", format!("{stage:?}"), w(false), w(true));
    }
    println!("  programs wave32/64   {} / {}", t.wave[0], t.wave[1]);
    println!(
        "  records wave32/64    {} / {} (mixed {})",
        t.record_wave[0], t.record_wave[1], t.record_wave[2]
    );
    println!("  instructions         {} ({} MB of code)", t.instructions, t.code_bytes >> 20);
    let mut top: Vec<_> = t.mnemonics.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!("  distinct mnemonics   {}", top.len());
    println!("  top 40 mnemonics");
    for (name, n) in top.iter().take(40) {
        println!("    {name:<28} {n}");
    }
    let mut unknown: Vec<_> = t.unknown.iter().collect();
    unknown.sort_by(|a, b| b.1.cmp(a.1).then(format!("{:?}", a.0).cmp(&format!("{:?}", b.0))));
    let unknown_total: u64 = t.unknown.values().sum();
    println!("  unknown opcodes      {} distinct, {unknown_total} instructions", unknown.len());
    for ((enc, op), n) in &unknown {
        println!("    {:<8} {op:#05x}  {n}", format!("{enc:?}"));
    }
    println!("  features             programs  records");
    for (what, v) in [
        ("mesh stage (MS)", t.mesh),
        ("geometry stage (GS)", t.geometry),
        ("NGG exp prim", t.ngg_prim),
        ("64-bit atomics", t.atomic64),
        ("ray tracing (bvh)", t.ray),
    ] {
        println!("    {what:<20} {:>8} {:>8}", v[0], v[1]);
    }
    let ray: Vec<String> = t.ray_stages.iter().map(|(s, n)| format!("{s:?} {n}")).collect();
    println!("    ray tracing by stage: {}", ray.join(", "));
    for (i, z) in zlib.iter().enumerate() {
        println!(
            "  zlib entry {i}         {} streams, {} MB inflated, {} DXBC magics{}",
            z.streams,
            z.inflated >> 20,
            z.dxbc,
            z.error.map_or(String::new(), |at| format!(", STOPS at {at:#x}"))
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_shader::{ResourceType, RootParam, RootType, UsedResource};

    /// The dump path, and the read lines of a small compute program
    /// (a hand-assembled one): a table descriptor at bytes 0 and 0x20,
    /// a vector and a scalar buffer load through the first.
    #[test]
    fn dump_reads() {
        assert_eq!(dump_path(Path::new("d"), 0xab << 120), Path::new("d").join("ab").join(format!("ab{}.txt", "0".repeat(30))));
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
            0xE070_1000,
            0x8003_0201, // buffer_store_dword v2, v1, s[12:15], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let used = |kind, table_offset_dwords| UsedResource { kind, root: RootType::Table, table_offset_dwords, lower_bound: 0, stride: 0, dwords: 4, user_data: 4, slot: 1 };
        let param = |root, sgprs, user_data, slot| RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot };
        let n = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![param(RootType::Constants, 4, 0, 0), param(RootType::Table, 1, 4, 1)],
            used: vec![used(ResourceType::Srv, 0), used(ResourceType::Uav, 8)],
            io: xnw_shader::native::StageIo::Cs(xnw_shader::native::ComputeIo { rsrc2: 5 << 1, threads: [1, 1, 1] }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut text = String::new();
        reads_text(&n, &mut text);
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        assert_eq!(
            lines,
            [
                "read 000004 s_load_dwordx4 T4+0x0",
                "read 00000c s_load_dwordx4 T4+0x20",
                "read 000018 buffer_load_dword T4+0x0#0 +0x0",
                "read 000020 s_buffer_load_dword T4+0x0#0 +0x4",
                "read 000028 buffer_store_dword T4+0x20#0 +0x0",
            ]
        );
    }
}
