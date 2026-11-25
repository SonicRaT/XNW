//! `xnw pack-info <frame dir> [--pack shaders.rpf]`: the coverage of
//! a frame pack v2: sink draws paired with hook draws,
//! draws whose mesh-or-vertex and pixel programs both name a pack record,
//! the same for the main view's stage 66-82 draws (the coverage figure),
//! textures with texels or a pack origin, and bytes by kind.

use std::collections::HashMap;
use std::path::Path;

use xnw_formats::framepack::{Coverage, FramePack};
use xnw_gpu::d3d12x::pack::PackIndex;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// v1 draws.tsv: (stage, viewport) per draw index.
fn v1_draws(dir: &Path) -> Res<Vec<(u32, String)>> {
    let text = std::fs::read_to_string(dir.join("draws.tsv"))?;
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap_or_default().split('\t').collect();
    let col = |n: &str| head.iter().position(|h| *h == n).ok_or_else(|| format!("draws.tsv: no `{n}` column"));
    let (stage, viewport) = (col("stage")?, col("viewport")?);
    Ok(lines
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f.get(stage).and_then(|s| s.parse().ok()).unwrap_or(0), f.get(viewport).copied().unwrap_or("-").to_string())
        })
        .collect())
}

pub fn run(args: &[String]) -> Res<()> {
    let Some(dir) = args.first().map(Path::new) else {
        return Err("usage: xnw pack-info <frame dir> [--pack shaders.rpf]".into());
    };
    let rpf = args.iter().position(|a| a == "--pack").and_then(|i| args.get(i + 1)).map(std::path::PathBuf::from).or_else(xnw_gpu::d3d12x::pack::default_pack);
    let pack = FramePack::open(dir)?.ok_or("no pack\\manifest.tsv: a v1 frame")?;
    let m = &pack.manifest;
    println!("manifest   frame {} v{}, hook draws {}, sink draws {}, hook draws with 0/1/>1 sinks {}/{}/{}, unpaired sinks {}", m.frame, m.version, m.hook_draws, m.sink_draws, m.pairs[0], m.pairs[1], m.pairs[2], m.unpaired_sinks);
    for (k, r, n) in &m.reasons {
        println!("reason     kind {k} reason {r}: {n}");
    }
    let index = match &rpf {
        Some(p) => Some(PackIndex::open(p, Some(Path::new("out/cache/pack-index.bin")))?),
        None => None,
    };
    // Each program's pack name (by container hash, else by its first 24 bytes).
    let mut names: HashMap<u128, Option<String>> = HashMap::new();
    for p in &pack.programs {
        names.entry(p.container).or_insert_with(|| {
            let i = index.as_ref()?;
            i.by_hash(p.container).or_else(|| i.by_code(&p.prefix)).map(|pl| pl.name.clone())
        });
    }
    let named = |h: u128| names.get(&h).is_some_and(Option::is_some);
    let cov = Coverage::of(&pack, &named);
    if index.is_none() {
        println!("note       no shader pack found (--pack): program names not joined");
    }
    // The exit figure: main-view stage 66-82 hook draws whose (1:1) sink has both programs named.
    let draws = v1_draws(dir).unwrap_or_default();
    let mut per_vp: HashMap<&str, u32> = HashMap::new();
    for (s, vp) in &draws {
        if (66..=82).contains(s) {
            *per_vp.entry(vp.as_str()).or_default() += 1;
        }
    }
    let main = per_vp.iter().max_by_key(|(_, n)| **n).map(|(v, _)| v.to_string()).unwrap_or_default();
    let mut sinks_of: HashMap<u32, Vec<u64>> = HashMap::new();
    for s in &pack.sinks {
        if let Some(d) = s.hook_draw {
            sinks_of.entry(d).or_default().push(s.pipeline);
        }
    }
    let both = |pipe: u64| {
        let stage = |pixel: bool| pack.programs.iter().any(|g| g.pipeline == pipe && (g.stage == "pixel" || g.stage == "PS") == pixel && named(g.container));
        stage(false) && stage(true)
    };
    let (mut total, mut ok) = (0u32, 0u32);
    for (i, (s, vp)) in draws.iter().enumerate() {
        if (66..=82).contains(s) && *vp == main {
            total += 1;
            if sinks_of.get(&(i as u32)).is_some_and(|p| p.len() == 1 && both(p[0])) {
                ok += 1;
            }
        }
    }
    let pct = if total == 0 { 0.0 } else { 100.0 * f64::from(ok) / f64::from(total) };
    println!("{}", cov.line());
    println!("pack: main view {main} stage 66-82 draws {total}, both programs named {ok} ({pct:.1} %)");
    albedo_line(dir);
    preskin_lines(dir, &|h| names.get(&h).and_then(|n| n.as_deref()).is_some_and(|n| n.contains("PreSkinD")));
    imposter_line(dir);
    hair_line(&pack, &names);
    wheel_line(&pack);
    bridge_line(dir);
    strand_lines(dir);
    let unnamed: Vec<String> = pack.programs.iter().filter(|p| !named(p.container)).take(5).map(|p| format!("{} {:032x}", p.stage, p.container)).collect();
    if !unnamed.is_empty() {
        println!("unnamed    first {}: {}", unnamed.len(), unnamed.join(", "));
    }
    Ok(())
}

/// The replay's albedo coverage: distinct list.tsv albedo and layer
/// handles, how many the pack resolves to a view, to a texture row, and to
/// texels or a pack origin (what `world-render` can show).
fn albedo_line(dir: &Path) {
    let Ok(text) = std::fs::read_to_string(dir.join("list.tsv")) else {
        println!("albedo     no list.tsv");
        return;
    };
    let Ok(Some(pack)) = xnw_frame::pack::Pack::open(dir) else { return };
    let views = pack.views_by_handle();
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap_or_default().split('\t').collect();
    let cols: Vec<(usize, bool)> = head.iter().enumerate().filter_map(|(i, h)| match *h {
        "albedo_desc" => Some((i, true)),
        "layer0" | "layer1" | "layer2" | "layer3" => Some((i, false)),
        _ => None,
    }).collect();
    let mut rows = 0u32;
    let mut handles: HashMap<u64, bool> = HashMap::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        for (i, albedo) in &cols {
            let h = f.get(*i).and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()).unwrap_or(0);
            if h != 0 {
                rows += u32::from(*albedo);
                *handles.entry(h).or_default() |= *albedo;
            }
        }
    }
    let count = |albedo: bool| {
        let (mut n, mut view, mut row, mut held) = (0u32, 0u32, 0u32, 0u32);
        for (h, _) in handles.iter().filter(|(_, a)| **a == albedo) {
            n += 1;
            let Some(v) = views.get(h) else { continue };
            view += 1;
            let Some(t) = pack.texture(v) else { continue };
            row += 1;
            held += u32::from(t.origin.is_some() || t.texels != 0);
        }
        format!("{n} distinct, {view} resolved to a view, {row} to a texture row, {held} with texels or origin")
    };
    println!("albedo     list rows with an albedo handle {rows}; albedo handles: {}; layer handles: {}", count(true), count(false));
}

/// The imposter calls (kind 42) and their instances (kind 47): how many
/// hold L's records, a visible list and the indirect arguments, and the
/// instance totals.
fn imposter_line(dir: &Path) {
    use xnw_formats::framepack::{FRAME_SCOPE, kind, reason};
    let Ok(Some(pack)) = xnw_frame::pack::Pack::open(dir) else { return };
    let calls = pack.data.refs_of(FRAME_SCOPE).filter(|r| r.kind == kind::IMPOSTER).count();
    let inst = pack.imposter_instances();
    let ok: Vec<_> = inst.iter().filter_map(|i| i.1.as_ref().ok()).collect();
    let with = |p: usize| ok.iter().filter(|i| i.reasons[p] == reason::OK).count();
    let mut why: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    for i in &inst {
        match &i.1 {
            Err(r) => *why.entry(format!("record reason {r}")).or_default() += 1,
            Ok(i) => {
                for (p, name) in ["head", "args", "visible", "records", "handler"].iter().enumerate() {
                    if i.reasons[p] != reason::OK {
                        *why.entry(format!("{name} reason {}", i.reasons[p])).or_default() += 1;
                    }
                }
                if i.reasons[0] == reason::OK && i.count != i.n {
                    *why.entry("count differs from n".into()).or_default() += 1;
                }
            }
        }
    }
    if inst.len() < calls {
        why.insert("calls without an instance record (a capture before v2.3)".into(), (calls - inst.len()) as u32);
    }
    let records: usize = ok.iter().map(|i| i.records.len() / 64).sum();
    let n: u32 = ok.iter().map(|i| i.n).sum();
    let why: Vec<String> = why.iter().map(|(k, v)| format!("{k}: {v}")).collect();
    println!(
        "imposter   calls {calls}, with instances {}: L read {}, records {} ({records} instances of n {n}), visible list {}, args {}, handler {}; {}",
        inst.len(),
        with(0),
        with(3),
        with(2),
        with(1),
        with(4),
        if why.is_empty() { "nothing missing".into() } else { why.join(", ") }
    );
}

/// The instanced wheel draws' palette windows (kind 48): draws that have
/// them, windows with bytes (cut ones among them), and reasons.
fn wheel_line(pack: &xnw_formats::framepack::FramePack) {
    use xnw_formats::framepack::{kind, reason};
    let mut draws = std::collections::HashSet::new();
    let (mut whole, mut cut) = (0u32, 0u32);
    let mut why: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for r in pack.refs.iter().filter(|r| r.kind == kind::PAL_WINDOW) {
        draws.insert(r.owner);
        match r.reason {
            reason::OK => whole += 1,
            reason::CUT => cut += 1,
            other => *why.entry(other).or_default() += 1,
        }
    }
    let why: Vec<String> = why.iter().map(|(r, n)| format!("reason {r}: {n}")).collect();
    println!("wheels     instanced wheel draws with palette windows {}, windows with bytes {} ({cut} cut at the view end); without: {}", draws.len(), whole + cut, if why.is_empty() { "none".into() } else { why.join(", ") });
}

/// The bridge inputs (v2.4): draw sinks with their draw words (kind 49)
/// and views resolved, the words that name a texture with texels or
/// origin, the sampler heaps the sinks name and their entries by form, and
/// the root views without bytes.
fn bridge_line(dir: &Path) {
    use xnw_formats::framepack::{self as fp, kind, reason};
    let Ok(Some(pack)) = xnw_frame::pack::Pack::open(dir) else { return };
    let draws: Vec<u32> = pack.data.sinks.iter().filter(|s| s.op != fp::op::DISPATCH).map(|s| s.index).collect();
    let (mut with, mut views, mut textures, mut held) = (0u32, 0usize, 0u32, 0u32);
    let mut why: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    let mut recorded = 0u32;
    for &s in &draws {
        let Some(w) = pack.draw_words(s) else { continue };
        recorded += 1;
        match (&w.words, &w.views) {
            (Err(r), _) => *why.entry(format!("words reason {r}")).or_default() += 1,
            (Ok(_), Err(r)) => {
                with += 1;
                *why.entry(format!("views reason {r}")).or_default() += 1;
            }
            (Ok(_), Ok(v)) => {
                with += 1;
                views += v.len();
                for v in v.iter().filter(|v| v.kind == fp::view::TEXTURE || v.kind == fp::view::PLACED) {
                    textures += 1;
                    held += u32::from(pack.texture(v).is_some_and(|t| t.texels != 0 || t.origin.is_some()));
                }
            }
        }
    }
    if (recorded as usize) < draws.len() {
        why.insert("sinks without the record (a capture before v2.4)".into(), (draws.len() - recorded as usize) as u32);
    }
    let mut heaps = std::collections::HashMap::new();
    let mut no_heap = 0u32;
    for s in &pack.data.sinks {
        match pack.samplers(s.index) {
            Some((at, Ok(e))) => {
                heaps.insert(at, e);
            }
            Some((_, Err(reason::NULL))) => no_heap += 1,
            Some((_, Err(r))) => *why.entry(format!("samplers reason {r}")).or_default() += 1,
            None => {}
        }
    }
    let form = |f: u32| heaps.values().flatten().filter(|e| e.form == f).count();
    let roots: Vec<&fp::Ref> = pack.data.refs.iter().filter(|r| r.kind == kind::ROOT_VIEW).collect();
    let without = roots.iter().filter(|r| !reason::has_bytes(r.reason)).count();
    let cut = roots.iter().filter(|r| r.reason == reason::CUT).count();
    let why: Vec<String> = why.iter().map(|(k, v)| format!("{k}: {v}")).collect();
    println!(
        "bridge     draw words {with} of {} draw sinks ({views} views, {textures} textures, {held} with texels or origin); sampler heaps {} ({} entries: console {}, d3d12 {}, raw {}; {no_heap} sinks without a heap); root views {} ({} without bytes, {cut} cut); {}",
        draws.len(),
        heaps.len(),
        heaps.values().map(Vec::len).sum::<usize>(),
        form(fp::SAMPLER_CONSOLE),
        form(fp::SAMPLER_D3D12),
        form(fp::SAMPLER_RAW),
        roots.len(),
        without,
        if why.is_empty() { "nothing missing".into() } else { why.join(", ") }
    );
}

/// The hair census: sinks whose programs name hair (strand geometry, the
/// hair resolve, hair materials), by program name: sinks, how many paired
/// with a hook draw (a list row: our pass sees it), ops, and the stages
/// that name hair. Hair our pass never sees is the unpaired count.
fn hair_line(pack: &FramePack, names: &HashMap<u128, Option<String>>) {
    let mut by: std::collections::BTreeMap<String, (u32, u32, std::collections::BTreeSet<u8>)> = Default::default();
    for s in &pack.sinks {
        for p in pack.programs.iter().filter(|p| p.pipeline == s.pipeline) {
            let Some(Some(n)) = names.get(&p.container) else { continue };
            if !n.to_ascii_lowercase().contains("hair") {
                continue;
            }
            let short = n.trim_end_matches("_Wrapped");
            let short = short.rsplit_once('_').filter(|(_, h)| h.len() == 8 && h.chars().all(|c| c.is_ascii_hexdigit())).map_or(short, |(a, _)| a);
            if std::env::var("XNW_HAIR_DRAWS").is_ok() && let Some(d) = s.hook_draw {
                println!("hair draw  {d} sink {} op {} {short}", s.index, s.op);
            }
            let e = by.entry(format!("{}:{short}", p.stage)).or_default();
            e.0 += 1;
            e.1 += u32::from(s.hook_draw.is_some());
            e.2.insert(s.op);
        }
    }
    let parts: Vec<String> = by.iter().map(|(n, (k, h, ops))| format!("{n} {k} ({h} with a hook draw, ops {ops:?})")).collect();
    println!("hair       sinks by program: {}", if parts.is_empty() { "none".into() } else { parts.join("; ") });
}

/// v2.6: the Draw/DrawIndexed sinks' arguments (DRAW_ARGS) and the
/// strand-hair inputs: dispatch sinks holding strand group records, the
/// records, their heap views (with a view, without by reason) and the
/// distinct bytes behind them.
fn strand_lines(dir: &Path) {
    use xnw_formats::framepack::{kind, reason};
    let Ok(Some(pack)) = xnw_frame::pack::Pack::open(dir) else { return };
    let refs = &pack.data.refs;
    let args: Vec<[u32; 4]> = refs
        .iter()
        .filter(|r| r.kind == kind::DRAW_ARGS)
        .filter(|r| reason::has_bytes(r.reason))
        .filter_map(|r| pack.blob(r.hash))
        .filter(|b| b.len() >= 16)
        .map(|b| std::array::from_fn(|i| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])))
        .collect();
    let inst: u64 = args.iter().map(|a| u64::from(a[0])).sum();
    let multi = args.iter().filter(|a| a[0] > 1).count();
    println!("drawargs   Draw/DrawIndexed sinks with their arguments {}: instances {inst} in all, {multi} with more than one, most {}", args.len(), args.iter().map(|a| a[0]).max().unwrap_or(0));
    let views: Vec<&xnw_formats::framepack::Ref> = refs.iter().filter(|r| r.kind == kind::STRAND_VIEW).collect();
    let sinks: std::collections::BTreeSet<u32> = views.iter().map(|r| r.owner).collect();
    let records: std::collections::BTreeSet<(u32, u16)> = views.iter().filter(|r| r.reason != reason::NULL).map(|r| (r.owner, r.slot / 8)).collect();
    let mut why: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for r in views.iter().filter(|r| r.reason != reason::OK) {
        *why.entry(r.reason).or_default() += 1;
    }
    let mut seen = std::collections::HashSet::new();
    let (mut n, mut bytes, mut output) = (0u32, 0u64, 0u32);
    for r in refs.iter().filter(|r| r.kind == kind::STRAND_BYTES) {
        if r.reason == reason::GPU_OUTPUT {
            output += 1;
        } else if reason::has_bytes(r.reason) && seen.insert(r.hash) {
            n += 1;
            bytes += u64::from(r.len);
        }
    }
    let why: Vec<String> = why.iter().map(|(r, n)| format!("reason {r}: {n}")).collect();
    println!(
        "strand     sinks with strand group records {}, records {}, heap views {} ({} with a view; without: {}); bytes {n} distinct objects, {:.1} MB; {output} GPU output views (address only)",
        sinks.len(),
        records.len(),
        views.len(),
        views.iter().filter(|r| r.reason == reason::OK).count(),
        if why.is_empty() { "none".into() } else { why.join(", ") },
        bytes as f64 / 1e6
    );
}

/// The ped pre-skin coverage: dispatches recorded (of the frame's
/// direct dispatches), how many hold b7, their item and all inputs, their
/// bytes by kind, and the PreSkinD draws whose range a recorded dispatch
/// wrote (the coverage figure), with the two output views of the draws' T8.
fn preskin_lines(dir: &Path, is_preskin: &dyn Fn(u128) -> bool) {
    use xnw_formats::framepack::{self as fp, kind, reason};
    let Ok(Some(pack)) = xnw_frame::pack::Pack::open(dir) else { return };
    let ds = pack.dispatches();
    let m = &pack.data.manifest;
    let flag = |f: u32| ds.iter().filter(|d| d.rec.flags & f != 0).count();
    let checks: Vec<Result<(), String>> = ds.iter().map(|d| pack.dispatch_complete(d)).collect();
    let first = ds.iter().zip(&checks).find_map(|(d, c)| c.as_ref().err().map(|why| format!("; first incomplete: sink {} ({why})", d.sink))).unwrap_or_default();
    println!("preskin    dispatches recorded {} of {} direct dispatches; with b7 {}, with item {}, with all inputs {}{first}", ds.len(), m.dispatches[1], flag(fp::B7_FOUND), flag(fp::ITEM_FOUND), checks.iter().filter(|c| c.is_ok()).count());
    // Distinct blobs the dispatch sinks reference, by kind; records without bytes, by reason.
    let owners: std::collections::HashSet<u32> = ds.iter().map(|d| d.sink).collect();
    let mut by: std::collections::BTreeMap<u16, (u32, u64)> = std::collections::BTreeMap::new();
    let mut why: std::collections::BTreeMap<(u16, u32), u32> = std::collections::BTreeMap::new();
    let mut seen = std::collections::HashSet::new();
    for r in pack.data.refs.iter().filter(|r| owners.contains(&r.owner)) {
        if !reason::has_bytes(r.reason) {
            *why.entry((r.kind, r.reason)).or_default() += 1;
        } else if seen.insert(r.hash) {
            let e = by.entry(r.kind).or_default();
            e.0 += 1;
            e.1 += u64::from(r.len);
        }
    }
    if !ds.is_empty() {
        let bytes: Vec<String> = by.iter().map(|(k, (n, b))| format!("{} {b} ({n})", kind::name(*k))).collect();
        let none: Vec<String> = why.iter().map(|((k, r), n)| format!("{} reason {r}: {n}", kind::name(*k))).collect();
        println!("preskin    dispatch bytes by kind (distinct objects): {}; without bytes: {}", bytes.join(", "), if none.is_empty() { "none".into() } else { none.join(", ") });
        let mut programs: HashMap<u128, u32> = HashMap::new();
        for d in &ds {
            *programs.entry(d.program).or_default() += 1;
        }
        let mut programs: Vec<String> = programs.iter().map(|(h, n)| format!("{:012x} x{n}", h >> 80)).collect();
        programs.sort();
        println!("preskin    dispatch programs: {}", programs.join(", "));
    }
    // v2.4: displacing dispatches' maps (variants 8, 9).
    let maps: Vec<(u16, u32, Result<fp::SlotView, u32>)> = ds.iter().flat_map(|d| pack.displacement_maps(d.sink)).collect();
    if !maps.is_empty() {
        let views: Vec<&fp::SlotView> = maps.iter().filter_map(|m| m.2.as_ref().ok()).collect();
        let held = views.iter().filter(|v| pack.texture(v).is_some_and(|t| t.texels != 0 || t.origin.is_some())).count();
        let mut indices: Vec<u32> = maps.iter().map(|m| m.1).collect();
        indices.sort_unstable();
        indices.dedup();
        let mut why: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
        for m in &maps {
            if let Err(r) = m.2 {
                *why.entry(r).or_default() += 1;
            }
        }
        let why: Vec<String> = why.iter().map(|(r, n)| format!("reason {r}: {n}")).collect();
        println!("preskin    displacement maps: dispatches {}, elements {}, with a view {} ({held} textures with texels or origin), heap indices {:?}; without: {}", ds.iter().filter(|d| pack.displacement_maps(d.sink).first().is_some()).count(), maps.len(), views.len(), &indices[..indices.len().min(8)], if why.is_empty() { "none".into() } else { why.join(", ") });
    }
    let draws = pack.preskin_join(is_preskin);
    let views = draws.iter().find(|d| d.views.iter().any(Option::is_some)).map_or([None; 2], |d| d.views);
    let view = |v: Option<fp::SlotView>| v.map_or("-".to_string(), |v| format!("{} B (view kind {}, resource {:#x})", v.bytes, v.kind, v.resource));
    println!(
        "preskin    PreSkinD draws {} (with a range word {}), whose range has a recorded dispatch {}; T8+0x420 {}, T8+0xa80 {}",
        draws.len(),
        draws.iter().filter(|d| d.word.is_some()).count(),
        draws.iter().filter(|d| d.dispatch.is_some()).count(),
        view(views[0]),
        view(views[1])
    );
}
