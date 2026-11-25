//! `xnw texture-info --trace <log>`: replays a run's `XNW_TRACE=dstorage`
//! log through the texture origin index (`xnw_formats::origin`), offline.
//!
//! Each `.ttpf` request (`options .., <name> @ <offset>, <src> -> <dst>
//! bytes`) is resolved to its file page and texture as the run-time registry
//! does; the host file comes from the log's `opened <name> (<host>)` line.
//! Prints: requests resolved, whether sizes and options match the page
//! (one request per whole page), the mips asked for first, how much is
//! mip 0, requests per texture, and how often a (file, offset) is asked for
//! again (what a cold cache keyed by origin would save).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

use xnw_formats::origin::{NO_TEXTURE, Registry};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Default)]
struct Tex {
    requests: u64,
    bytes: u64,
    pages: HashSet<u32>,
    first: Option<(u8, u8)>,
    mip0: bool,
    /// Every request at a higher file address than the one before.
    ascending: bool,
    last_at: u64,
}

/// Per group (ttdb folder) and in total: requests, bytes, mip-0 bytes
/// (the request's pages hold mip 0), mip-0-only bytes, repeats, repeat bytes.
#[derive(Default, Clone, Copy)]
struct Sum {
    requests: u64,
    bytes: u64,
    mip0: u64,
    mip0_only: u64,
    repeats: u64,
    repeat_bytes: u64,
}

impl Sum {
    fn add(&mut self, bytes: u64, mips: (u8, u8), repeat: bool) {
        self.requests += 1;
        self.bytes += bytes;
        if mips.0 == 0 {
            self.mip0 += bytes;
        }
        if mips == (0, 0) {
            self.mip0_only += bytes;
        }
        if repeat {
            self.repeats += 1;
            self.repeat_bytes += bytes;
        }
    }

    fn line(&self, name: &str) {
        let pct = |a: u64, b: u64| 100.0 * a as f64 / b.max(1) as f64;
        println!(
            "  {name:<12} {:>6} req {:>8.1} MB | mip 0 held {:>5.1} % of bytes (mip 0 only {:>5.1} %) | again {:>5.1} % of req, {:>5.1} % of bytes",
            self.requests,
            self.bytes as f64 / 1e6,
            pct(self.mip0, self.bytes),
            pct(self.mip0_only, self.bytes),
            pct(self.repeats, self.requests),
            pct(self.repeat_bytes, self.bytes)
        );
    }
}

/// `options 0x200001, G:\.\data\ttdb\x\y.ttpf @ 0x17500, 16 -> 16 bytes ...`
fn parse_request(line: &str) -> Option<(f64, u64, &str, u64, u64, u64)> {
    let time = line.split_whitespace().next()?.parse().ok()?;
    let rest = &line[line.find("trace dstorage: options ")? + 24..];
    let (options, rest) = rest.split_once(", ")?;
    let (name, rest) = rest.split_once(" @ ")?;
    let (offset, rest) = rest.split_once(", ")?;
    let (src, rest) = rest.split_once(" -> ")?;
    let dst = rest.split_once(' ')?.0;
    let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok();
    Some((time, hex(options)?, name, hex(offset)?, src.parse().ok()?, dst.parse().ok()?))
}

pub fn run(log: &str) -> Res<()> {
    let text = std::fs::read_to_string(log)?;
    let registry = Registry::new(0);
    let mut hosts: HashMap<String, PathBuf> = HashMap::new();
    let mut ids: HashMap<String, Option<u32>> = HashMap::new();
    let (mut header, mut unresolved, mut no_texture, mut size_ok, mut options_ok) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut resolved = 0u64;
    let mut textures: BTreeMap<(u32, u32), Tex> = BTreeMap::new();
    let mut seen: HashSet<(u32, u64)> = HashSet::new();
    let mut groups: BTreeMap<String, Sum> = BTreeMap::new();
    let mut total = Sum::default();
    let mut first_unresolved = Vec::new();
    let (mut t0, mut t1) = (f64::MAX, 0f64);
    for line in text.lines() {
        if let Some(at) = line.find("dstorage: opened ") {
            let rest = &line[at + 17..];
            if let Some((name, host)) = rest.split_once(" (")
                && let Some((host, _)) = host.split_once("), iid")
            {
                hosts.insert(name.to_owned(), PathBuf::from(host));
            }
            continue;
        }
        if !line.contains(".ttpf @ ") {
            continue;
        }
        let Some((time, options, name, offset, src, dst)) = parse_request(line) else {
            continue;
        };
        let id = *ids.entry(name.to_owned()).or_insert_with(|| {
            let host = hosts.get(name)?;
            registry.open(host).map_err(|e| println!("{}: {e}", host.display())).ok()
        });
        let Some(id) = id else {
            unresolved += 1;
            continue;
        };
        let index = registry.file(id).ok_or("registry")?;
        if offset < index.ttpf.header_bytes {
            header += 1;
            continue;
        }
        (t0, t1) = (t0.min(time), t1.max(time));
        let Some(r) = index.resolve(offset) else {
            unresolved += 1;
            if first_unresolved.len() < 5 {
                first_unresolved.push(format!("{name} @ {offset:#x}"));
            }
            continue;
        };
        resolved += 1;
        let page = index.ttpf.pages[r.page as usize];
        if dst == page.size && (src == page.stored || page.storage == 0) {
            size_ok += 1;
        }
        if options & 0xff == page.request_options() {
            options_ok += 1;
        }
        if r.texture == NO_TEXTURE {
            no_texture += 1;
            continue;
        }
        let mips = index.mips_in(r.texture, r.at, r.at + dst);
        let repeat = !seen.insert((id, offset));
        let group = name.to_ascii_lowercase().split("\\ttdb\\").nth(1).and_then(|s| s.split('\\').next()).filter(|s| !s.ends_with(".ttpf")).unwrap_or("(ttdb)").to_owned();
        groups.entry(group).or_default().add(dst, mips, repeat);
        total.add(dst, mips, repeat);
        let t = textures.entry((id, r.texture)).or_insert_with(|| Tex { ascending: true, ..Default::default() });
        if t.requests > 0 && !repeat && r.at <= t.last_at {
            t.ascending = false;
        }
        t.last_at = r.at;
        t.requests += 1;
        t.bytes += dst;
        t.pages.insert(r.page);
        t.first.get_or_insert(mips);
        t.mip0 |= mips.0 == 0;
    }
    let requests = resolved + unresolved;
    println!("{log}: {} files, .ttpf requests {:.1}-{:.1} s", ids.len(), t0, t1);
    println!(
        "  index reads (header) {header}; page requests {requests}: resolved {resolved} ({:.2} %), unresolved {unresolved}, before any texture {no_texture}",
        100.0 * resolved as f64 / requests.max(1) as f64
    );
    println!(
        "  whole page (dst = page size, src = stored): {size_ok} of {resolved}; options = the page's codec: {options_ok} of {resolved}"
    );
    for u in &first_unresolved {
        println!("  unresolved: {u}");
    }
    println!("by folder (bytes = destination bytes):");
    for (g, s) in &groups {
        s.line(g);
    }
    total.line("total");

    // Per texture.
    let n = textures.len() as u64;
    let mut per: BTreeMap<u64, u64> = BTreeMap::new();
    let (mut whole, mut with_mip0, mut ascending, mut multi, mut first_tail) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut first_finest: BTreeMap<u8, u64> = BTreeMap::new();
    let mut mipped = 0u64;
    for (&(id, ti), t) in &textures {
        *per.entry(t.requests.min(16)).or_default() += 1;
        let index = registry.file(id).ok_or("registry")?;
        let tex = index.texture(ti).ok_or("texture")?;
        let span_pages = index.ttpf.pages_of(tex.addr, tex.span).len();
        whole += u64::from(t.pages.len() >= span_pages);
        with_mip0 += u64::from(t.mip0);
        if span_pages > 1 {
            multi += 1;
            ascending += u64::from(t.ascending);
        }
        if tex.mips > 1 {
            mipped += 1;
            let first = t.first.unwrap_or_default();
            *first_finest.entry(first.0).or_default() += 1;
            first_tail += u64::from(first.1 == (tex.mips - 1).min(255) as u8);
        }
    }
    println!(
        "textures touched {n}: all pages read {whole} ({:.1} %), mip 0 read {with_mip0} ({:.1} %); multi-page {multi}, read in file order (tail first, mip 0 last) {ascending}",
        100.0 * whole as f64 / n.max(1) as f64,
        100.0 * with_mip0 as f64 / n.max(1) as f64
    );
    println!(
        "  with mips > 1: {mipped}; first request holds the coarsest mip {first_tail}; finest mip of the first request: {}",
        first_finest.iter().map(|(m, c)| format!("{m}: {c}")).collect::<Vec<_>>().join(", ")
    );
    println!("  requests per texture: {}", per.iter().map(|(k, c)| format!("{}{k}: {c}", if *k == 16 { ">=" } else { "" })).collect::<Vec<_>>().join(", "));
    Ok(())
}
