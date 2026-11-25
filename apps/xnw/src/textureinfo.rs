//! `xnw texture-info <ttpf | dir | rpf> [--census] [--list N] [--dump N] [--only HASH] [--sample K] [--out DIR]`
//! `xnw texture-info --trace <log>`: replay of a DirectStorage trace (`texturetrace.rs`).
//!
//! The title's textures as an offline converter sees them
//! (`xnw_formats::ttpf`). An `.rpf` stands for its
//! package: its textures are the `.ttpf` files under `<its folder>/data/ttdb`
//! (RPF9 archives hold models and name the textures by hash; they store no
//! texture pages). A folder is walked for `.ttpf` files.
//!
//! - `--list N`: the first N textures of each file with their fields, span,
//!   pages, file range and storage.
//! - `--census`: per file and in total: pages and bytes by storage, textures
//!   by kind and format, whether the layout predicts each span, where mip 0
//!   starts relative to pages, and host-pack sizes (linear, all mips and
//!   without mip 0).
//! - `--sample K`: decodes every K-th page (zlib and BCPack, as the
//!   title's DirectStorage requests do) and deflates the result, for the
//!   compressed size of a host pack.
//! - `--dump N`: N 2D textures (formats in turn) decoded, detiled and
//!   written with all mips to `<out>/<file>_<hash>_<w>x<h>_<fmt>.dds`
//!   (default `out\textures`).
//!
//! Read only on the package.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use xnw_formats::bcpack::Tables;
use xnw_formats::ttpf::{self, Texture, Ttpf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Default)]
struct Census {
    files: u64,
    pages: u64,
    by_storage: BTreeMap<u8, [u64; 3]>, // pages, stored, virtual
    textures: u64,
    by_kind: BTreeMap<u8, [u64; 3]>, // textures, layout matches, layout fails
    by_mode: BTreeMap<u8, u64>,
    by_format: BTreeMap<u32, [u64; 5]>, // textures, span, linear, linear without mip 0, with mips > 1
    mismatch: Vec<String>,
    span: u64,
    linear: u64,
    linear_drop: u64,
    // 2D/array textures with mips > 1 whose mip 0 starts on a page start,
    // and the stored bytes of the pages wholly inside mip 0.
    mip0_aligned: [u64; 2],
    mip0_pages_stored: u64,
    tex_on_page_start: u64,
    /// Layouts shorter than the span (the rest is padding to the next texture).
    padded: u64,
    /// Pages with no stored bytes.
    empty_pages: u64,
    // Sample: pages, decoded bytes, deflated bytes, stored bytes, failures.
    sample: BTreeMap<u8, [u64; 5]>,
}

pub fn run(args: &[String]) -> Res<()> {
    if let Some(i) = args.iter().position(|a| a == "--trace") {
        return crate::texturetrace::run(args.get(i + 1).ok_or("--trace <dstorage log>")?);
    }
    let path = args.first().ok_or("usage: xnw texture-info <ttpf|dir|rpf> [--census] [--list N] [--dump N] [--only HASH] [--sample K] [--out DIR]")?;
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1));
    let census = args.iter().any(|a| a == "--census");
    let list: usize = flag("--list").map_or(Ok(0), |v| v.parse())?;
    let dump: usize = flag("--dump").map_or(Ok(0), |v| v.parse())?;
    let sample: u64 = flag("--sample").map_or(Ok(0), |v| v.parse())?;
    let only = flag("--only").map(|h| u64::from_str_radix(h.trim_start_matches("0x"), 16)).transpose()?;
    let out = PathBuf::from(flag("--out").map_or("out\\textures", |s| s.as_str()));

    let mut root = PathBuf::from(path);
    if root.extension().is_some_and(|e| e.eq_ignore_ascii_case("rpf")) {
        let dir = root.parent().unwrap_or(Path::new(".")).join("data").join("ttdb");
        println!("{}: RPF9 stores no texture pages; its package's textures are {}", root.display(), dir.display());
        root = dir;
    }
    let mut files = vec![];
    collect(&root, &mut files)?;
    files.sort();
    if files.is_empty() {
        return Err(format!("no .ttpf under {}", root.display()).into());
    }
    let tables = if sample > 0 || dump > 0 || only.is_some() { Some(Tables::builtin().map_err(|e| e.0)?) } else { None };
    let started = Instant::now();
    let mut c = Census::default();
    let mut dumped = 0usize;
    let mut formats_dumped: BTreeMap<u32, usize> = BTreeMap::new();
    for file in &files {
        let mut f = std::fs::File::open(file)?;
        let len = f.metadata()?.len();
        let t = match Ttpf::read(&mut f, len) {
            Ok(t) => t,
            Err(e) => {
                println!("{}: {e}", file.display());
                continue;
            }
        };
        let name = file.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned());
        if census || list > 0 {
            let stored: u64 = t.pages.iter().map(|p| p.stored).sum();
            let virt: u64 = t.pages.iter().map(|p| p.size).sum();
            println!(
                "{}: {:.1} MB, {} pages ({:.1} MB stored -> {:.1} MB), {} textures, header {} KB",
                file.display(),
                mb(len),
                t.pages.len(),
                mb(stored),
                mb(virt),
                t.textures.len(),
                t.header_bytes / 1024
            );
        }
        for x in t.textures.iter().take(list) {
            print_texture(&t, x);
        }
        if census {
            add(&mut c, &t);
        }
        if sample > 0 {
            for (i, p) in t.pages.iter().enumerate() {
                if i as u64 % sample != 0 {
                    continue;
                }
                let s = c.sample.entry(p.storage).or_default();
                s[0] += 1;
                s[3] += p.stored;
                match Ttpf::decode_page(&mut f, p, tables.as_ref()) {
                    Ok(bytes) => {
                        s[1] += bytes.len() as u64;
                        s[2] += miniz_oxide::deflate::compress_to_vec_zlib(&bytes, 6).len() as u64;
                    }
                    Err(e) => {
                        s[4] += 1;
                        if s[4] == 1 {
                            println!("  sample: page at {:#x} (storage {}): {e}", p.offset, p.storage);
                        }
                    }
                }
            }
        }
        for x in &t.textures {
            let pick = match only {
                Some(h) => x.hash == h,
                None => {
                    dumped < dump
                        && x.kind == ttpf::KIND_2D
                        && x.mips > 1
                        && x.width >= 256
                        && x.height >= 256
                        && formats_dumped.get(&x.format).copied().unwrap_or(0) < dump.div_ceil(3).max(1)
                }
            };
            if !pick {
                continue;
            }
            match write_dds(&t, &mut f, x, tables.as_ref(), &out, &name) {
                Ok(p) => {
                    dumped += 1;
                    *formats_dumped.entry(x.format).or_default() += 1;
                    println!("  dumped {}", p.display());
                }
                Err(e) => println!("  dump {:016x}: {e}", x.hash),
            }
        }
        c.files += 1;
    }
    if census {
        print_census(&c, sample);
    }
    println!("{} files in {:.1} s", files.len(), started.elapsed().as_secs_f64());
    Ok(())
}

fn collect(p: &Path, out: &mut Vec<PathBuf>) -> Res<()> {
    if p.is_dir() {
        for e in std::fs::read_dir(p)? {
            collect(&e?.path(), out)?;
        }
    } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("ttpf")) {
        out.push(p.to_path_buf());
    }
    Ok(())
}

fn mb(b: u64) -> f64 {
    b as f64 / 1e6
}

fn storage_name(s: u8) -> String {
    match (s & 8 != 0, s & 7) {
        (false, 0) => "stored".into(),
        (true, 0) => "zlib".into(),
        (false, bc) => format!("BCPack BC{bc}"),
        (true, bc) => format!("zlib+BCPack BC{bc}"),
    }
}

fn format_name(f: u32) -> String {
    let n = match f {
        2 => "RGBA32F",
        10 => "RGBA16F",
        11 => "RGBA16",
        24 => "RGB10A2",
        26 => "R11G11B10F",
        28 => "RGBA8",
        29 => "RGBA8_SRGB",
        34 => "RG16F",
        35 => "RG16",
        41 => "R32F",
        49 => "RG8",
        54 => "R16F",
        56 => "R16",
        61 => "R8",
        65 => "A8",
        71 => "BC1",
        72 => "BC1_SRGB",
        74 => "BC2",
        77 => "BC3",
        78 => "BC3_SRGB",
        80 => "BC4",
        81 => "BC4_SNORM",
        83 => "BC5",
        84 => "BC5_SNORM",
        87 => "BGRA8",
        91 => "BGRA8_SRGB",
        95 => "BC6H_UF16",
        96 => "BC6H_SF16",
        98 => "BC7",
        99 => "BC7_SRGB",
        _ => "",
    };
    if n.is_empty() { format!("fmt{f}") } else { n.into() }
}

fn kind_name(k: u8) -> &'static str {
    match k {
        ttpf::KIND_2D => "2D",
        ttpf::KIND_CUBE => "cube",
        ttpf::KIND_VOLUME => "volume",
        ttpf::KIND_ARRAY => "2D array",
        _ => "other",
    }
}

/// Linear host bytes of all mips, and without mip 0 (when there are more).
fn linear(x: &Texture) -> (u64, u64) {
    let mips = if x.kind == ttpf::KIND_CUBE { x.mips + 1 - cube_copies(x) } else { x.mips }.max(1);
    let all: u64 = (0..mips).map(|m| ttpf::linear_mip_bytes(x, m)).sum();
    let drop = if mips > 1 { all - ttpf::linear_mip_bytes(x, 0) } else { all };
    (all, drop)
}

fn cube_copies(x: &Texture) -> u32 {
    if x.mode & 8 != 0 { u32::from(x.mode & 7).max(1) } else { 1 }
}

fn print_texture(t: &Ttpf, x: &Texture) {
    let pages = t.pages_of(x.addr, x.span);
    let (first, last) = (t.pages.get(pages.start), pages.end.checked_sub(1).and_then(|i| t.pages.get(i)));
    let mut kinds: BTreeMap<u8, u32> = BTreeMap::new();
    for p in &t.pages[pages.clone()] {
        *kinds.entry(p.storage).or_default() += 1;
    }
    let place = ttpf::place(x);
    let (layout, mip0) = match &place {
        Ok(p) => (
            if p.size == x.span { "=span".to_string() } else { format!("{:#x}/span", p.size) },
            p.chain.mips.first().map_or(0, |m| m.offset) * u64::from(if x.kind == ttpf::KIND_ARRAY { x.layers } else { 1 }),
        ),
        Err(e) => (e.clone(), 0),
    };
    println!(
        "  {:016x} {:>5}x{:<5} {:<10} mips {:>2} {:<8} layers {:>3} mode {:#04x} addr {:#011x} span {:#9x} layout {} mip0 at +{:#x} pages {} file {:#x}..{:#x} {}",
        x.hash,
        x.width,
        x.height,
        format_name(x.format),
        x.mips,
        kind_name(x.kind),
        x.layers,
        x.mode,
        x.addr,
        x.span,
        layout,
        mip0,
        pages.len(),
        first.map_or(0, |p| p.offset),
        last.map_or(0, |p| p.offset + p.stored),
        kinds.iter().map(|(k, n)| format!("{}x{n}", storage_name(*k))).collect::<Vec<_>>().join(",")
    );
}

fn add(c: &mut Census, t: &Ttpf) {
    c.pages += t.pages.len() as u64;
    c.empty_pages += t.pages.iter().filter(|p| p.stored == 0).count() as u64;
    for p in &t.pages {
        let s = c.by_storage.entry(p.storage).or_default();
        s[0] += 1;
        s[1] += p.stored;
        s[2] += p.size;
    }
    for x in &t.textures {
        c.textures += 1;
        *c.by_mode.entry(x.mode).or_default() += 1;
        let (all, drop) = linear(x);
        let f = c.by_format.entry(x.format).or_default();
        f[0] += 1;
        f[1] += x.span;
        f[2] += all;
        f[3] += drop;
        f[4] += u64::from(x.mips > 1);
        c.span += x.span;
        c.linear += all;
        c.linear_drop += drop;
        if t.page_at(x.addr).is_some_and(|i| t.pages[i].addr == x.addr) {
            c.tex_on_page_start += 1;
        }
        let k = c.by_kind.entry(x.kind).or_default();
        k[0] += 1;
        match ttpf::place(x) {
            Ok(p) if p.size <= x.span => {
                k[1] += 1;
                if p.size < x.span {
                    c.padded += 1;
                }
                if matches!(x.kind, ttpf::KIND_2D | ttpf::KIND_ARRAY) && x.mips > 1 {
                    let slices = u64::from(p.slices);
                    let at = x.addr + p.chain.mips[0].offset * slices;
                    let aligned = t.page_at(at).is_some_and(|i| t.pages[i].addr == at);
                    c.mip0_aligned[usize::from(!aligned)] += 1;
                    for i in t.pages_of(at, x.addr + x.span - at) {
                        let pg = &t.pages[i];
                        if pg.addr >= at && pg.addr + pg.size <= x.addr + x.span {
                            c.mip0_pages_stored += pg.stored;
                        }
                    }
                }
            }
            r => {
                k[2] += 1;
                if c.mismatch.len() < 12 {
                    c.mismatch.push(format!(
                        "{:016x} {}x{} {} {} mips {} layers {} mode {:#x}: span {:#x}, layout {}",
                        x.hash,
                        x.width,
                        x.height,
                        format_name(x.format),
                        kind_name(x.kind),
                        x.mips,
                        x.layers,
                        x.mode,
                        x.span,
                        r.map_or_else(|e| e, |p| format!("{:#x}", p.size))
                    ));
                }
            }
        }
    }
}

fn print_census(c: &Census, sample: u64) {
    println!("\ncensus: {} files, {} pages, {} textures", c.files, c.pages, c.textures);
    println!("  storage             pages      stored MB    virtual MB  (title DirectStorage options)");
    for (s, v) in &c.by_storage {
        let p = ttpf::Page { addr: 0, size: 0, offset: 0, stored: 0, storage: *s };
        println!("  {:<18} {:>8} {:>12.1} {:>12.1}  {:#x}", storage_name(*s), v[0], mb(v[1]), mb(v[2]), p.request_options());
    }
    println!("  kind        textures  layout fits  other   ({} fit with padding after; {} pages store no bytes)", c.padded, c.empty_pages);
    for (k, v) in &c.by_kind {
        println!("  {:<10} {:>9} {:>12} {:>6}", kind_name(*k), v[0], v[1], v[2]);
    }
    println!("  layout modes: {}", c.by_mode.iter().map(|(m, n)| format!("{m:#x}:{n}")).collect::<Vec<_>>().join(" "));
    println!("  format        textures  mips>1   console MB  linear MB  linear-no-mip0 MB");
    for (f, v) in &c.by_format {
        println!("  {:<12} {:>9} {:>7} {:>12.1} {:>10.1} {:>12.1}", format_name(*f), v[0], v[4], mb(v[1]), mb(v[2]), mb(v[3]));
    }
    println!(
        "  total: console spans {:.1} MB, linear host {:.1} MB, without mip 0 {:.1} MB",
        mb(c.span),
        mb(c.linear),
        mb(c.linear_drop)
    );
    println!(
        "  textures starting a page {} of {}; 2D/array mip 0 on a page start {} of {}; pages wholly in mip 0 {:.1} MB stored",
        c.tex_on_page_start,
        c.textures,
        c.mip0_aligned[0],
        c.mip0_aligned[0] + c.mip0_aligned[1],
        mb(c.mip0_pages_stored)
    );
    for m in &c.mismatch {
        println!("  layout differs: {m}");
    }
    if sample > 0 {
        println!("  sample (every {sample}th page)  pages  stored MB  decoded MB  deflated MB  ratio  failed");
        for (s, v) in &c.sample {
            println!(
                "  {:<26} {:>6} {:>9.1} {:>10.1} {:>11.1} {:>6.3} {:>6}",
                storage_name(*s),
                v[0],
                mb(v[3]),
                mb(v[1]),
                mb(v[2]),
                v[2] as f64 / v[1].max(1) as f64,
                v[4]
            );
        }
        let (d, z, s): (u64, u64, u64) = c.sample.values().fold((0, 0, 0), |a, v| (a.0 + v[1], a.1 + v[2], a.2 + v[3]));
        println!("  sample total: decoded {:.1} MB, deflated {:.1} MB ({:.3}), stored {:.1} MB ({:.3})", mb(d), mb(z), z as f64 / d.max(1) as f64, mb(s), s as f64 / d.max(1) as f64);
    }
}

fn write_dds(t: &Ttpf, f: &mut std::fs::File, x: &Texture, tables: Option<&Tables>, out: &Path, file: &str) -> Res<PathBuf> {
    let place = ttpf::place(x)?;
    if place.size > x.span || !matches!(x.kind, ttpf::KIND_2D | ttpf::KIND_ARRAY) {
        return Err(format!("{} with span {:#x}, layout {:#x}: not dumped", kind_name(x.kind), x.span, place.size).into());
    }
    let mut span = t.read_span(f, x.addr, x.span, tables)?;
    let chain = &place.chain;
    span.resize(span.len().max((chain.size * u64::from(place.slices)) as usize), 0);
    let bytes = xnw_formats::texture::format(x.format).ok_or("format")?.bytes as usize;
    let mut data = vec![];
    for s in 0..place.slices {
        let one = if place.slices > 1 { ttpf::gather_slice(chain, place.slices, &span, s) } else { span.clone() };
        for m in 0..x.mips {
            let mip = &chain.mips[m as usize];
            let pitch = mip.width as usize * bytes;
            let mut lin = vec![0; pitch * mip.height as usize];
            chain.detile(&one, m, 0, &mut lin, pitch, 0).map_err(|e| format!("detile mip {m}: {e:?}"))?;
            data.extend_from_slice(&lin);
        }
    }
    std::fs::create_dir_all(out)?;
    let path = out.join(format!("{file}_{:016x}_{}x{}_{}.dds", x.hash, x.width, x.height, format_name(x.format)));
    std::fs::write(&path, ttpf::dds(x, x.mips, place.slices, &data))?;
    Ok(path)
}
