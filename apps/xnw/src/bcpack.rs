//! `xnw bcpack-check <ttpf or folder> [--limit N] [--jobs N]`
//!
//! Decodes real BCPack pages of the title's texture page files (`.ttpf`)
//! with `xnw_formats::bcpack`, the decoder DirectStorage uses: the title
//! issues no BCPack request before the world, so this is how the decoder
//! meets retail data. Read only; nothing decoded is written.
//!
//! Container (five-star-recomp, rpf9-research):
//! `RDS5`, n at 0x04, page entries `{lo, hi}` from 0x18
//! (n - 1 of them); `lo` bits 0-23 file offset / 0x100, bits 24-31 a flag
//! byte (bits 2-4 BCPack codec = `DSTORAGE_BCPACK_MODE`, bit 5 zlib, bits
//! 6-7 add 0x1000 steps to the address); `hi` bits 0-19 address / 0x4000. A
//! page's stored bytes run to the next higher file offset, its virtual size
//! (the decoded size) to the next page's address. Then `ETPF` at
//! `align(0x18 + 8 (n - 1), 0x100)`: count at 0x08, 16-byte descriptors from
//! 0x10 whose third word holds the texture's address / 0x1000 in bits 0-20;
//! a texture spans to the next texture's address.
//!
//! Sampling: per file and per codec, the first `--limit` textures (default
//! 10) that have a BCPack page; every BCPack page of a sampled texture is
//! decoded once. Per codec it prints textures, pages, blocks, failures and
//! decode time per block (one thread's time per page; pages run on
//! `--jobs` threads), plus format-level sanity counts on the output:
//! BC1 blocks with c0 == c1 that use index 3 (a plain BC1 encoder never
//! writes one; five-star-recomp counts 0 in 2.86 million plain blocks),
//! BC7 blocks with the reserved mode byte 0, BC6H blocks with a reserved
//! mode.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use xnw_formats::bcpack::{self, Tables};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
struct Page {
    addr: u64,
    offset: u64,
    stored: u64,
    size: u64,
    codec: u8,
    zlib: bool,
}

struct Pool {
    path: PathBuf,
    pages: Vec<Page>,
    /// Texture start addresses, sorted.
    textures: Vec<u64>,
}

fn u32_at(b: &[u8], at: usize) -> Res<u32> {
    Ok(u32::from_le_bytes(b.get(at..at + 4).ok_or("truncated")?.try_into()?))
}

fn parse(path: &Path) -> Res<Pool> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    let mut head = [0u8; 0x18];
    file.read_exact(&mut head)?;
    if &head[..4] != b"RDS5" {
        return Err(format!("{}: not RDS5", path.display()).into());
    }
    let n = u32_at(&head, 4)? as usize;
    let entries = n.saturating_sub(1);
    let mut table = vec![0u8; 8 * entries];
    file.read_exact(&mut table)?;
    let mut raw: Vec<Page> = (0..entries)
        .map(|i| {
            let lo = u32::from_le_bytes(table[8 * i..8 * i + 4].try_into().unwrap());
            let hi = u32::from_le_bytes(table[8 * i + 4..8 * i + 8].try_into().unwrap());
            let flag = (lo >> 24) as u8;
            Page {
                addr: u64::from(hi & 0xf_ffff) * 0x4000 + u64::from(flag >> 6) * 0x1000,
                offset: u64::from(lo & 0xff_ffff) * 0x100,
                stored: 0,
                size: 0,
                codec: (flag >> 2) & 7,
                zlib: flag & 0x20 != 0,
            }
        })
        .collect();
    // Stored bytes: to the next higher file offset, or the end of the file.
    let mut offsets: Vec<u64> = raw.iter().map(|p| p.offset).collect();
    offsets.sort_unstable();
    offsets.dedup();
    for p in &mut raw {
        let i = offsets.partition_point(|&o| o <= p.offset);
        p.stored = offsets.get(i).copied().unwrap_or(len) - p.offset;
    }
    raw.retain(|p| p.stored > 0);
    raw.sort_by_key(|p| p.addr);
    for i in 0..raw.len() {
        raw[i].size = raw.get(i + 1).map_or(0x4000, |q| q.addr - raw[i].addr);
    }
    // ETPF descriptors.
    let etpf = (0x18 + 8 * entries as u64).div_ceil(0x100) * 0x100;
    let mut section = [0u8; 16];
    file.seek(SeekFrom::Start(etpf))?;
    file.read_exact(&mut section)?;
    if &section[..4] != b"ETPF" {
        return Err(format!("{}: no ETPF at {etpf:#x}", path.display()).into());
    }
    let count = u32_at(&section, 8)? as usize;
    let mut desc = vec![0u8; 16 * count];
    file.read_exact(&mut desc)?;
    let mut textures: Vec<u64> = (0..count).map(|i| u64::from(u32_at(&desc, 16 * i + 8).unwrap_or(0) & 0x1f_ffff) * 0x1000).collect();
    textures.sort_unstable();
    Ok(Pool { path: path.to_path_buf(), pages: raw, textures })
}

#[derive(Default)]
struct Tally {
    textures: u64,
    pages: u64,
    blocks: u64,
    stored: u64,
    failed_pages: u64,
    failed_textures: u64,
    time: Duration,
    /// Format sanity: BC1 c0 == c1 with index 3, BC7 mode byte 0, BC6H reserved mode.
    odd: u64,
    first_error: Option<String>,
}

fn block_size(codec: u8) -> u64 {
    if matches!(codec, 1 | 4) { 8 } else { 16 }
}

/// Blocks a plain encoder would not write (see the module comment).
fn odd_blocks(codec: u8, out: &[u8]) -> u64 {
    match codec {
        1 => out
            .chunks_exact(8)
            .filter(|b| {
                let (c0, c1) = (u16::from_le_bytes([b[0], b[1]]), u16::from_le_bytes([b[2], b[3]]));
                let ix = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                c0 == c1 && (0..16).any(|p| ix >> (2 * p) & 3 == 3)
            })
            .count() as u64,
        7 => out.chunks_exact(16).filter(|b| b[0] == 0).count() as u64,
        6 => out
            .chunks_exact(16)
            .filter(|b| {
                let m = b[0] & 3;
                let m5 = b[0] & 0x1f;
                m >= 2 && matches!(m5, 0x13 | 0x17 | 0x1b | 0x1f)
            })
            .count() as u64,
        _ => 0,
    }
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) -> Res<()> {
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        for e in entries {
            collect(&e, out)?;
        }
    } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ttpf")) {
        out.push(path.to_path_buf());
    }
    Ok(())
}

/// One page to decode, and the sampled textures that read it.
struct Job {
    file: usize,
    page: Page,
    textures: Vec<u64>,
}

pub fn run(args: &[String]) -> Res<()> {
    let target = args.first().ok_or("usage: xnw bcpack-check <ttpf or folder> [--limit N] [--jobs N]")?;
    let opt = |name: &str, default: usize| -> Res<usize> {
        Ok(match args.iter().position(|a| a == name) {
            Some(i) => args.get(i + 1).ok_or(format!("{name} needs a value"))?.parse()?,
            None => default,
        })
    };
    let limit = opt("--limit", 10)?;
    let jobs = opt("--jobs", std::thread::available_parallelism().map_or(4, |n| n.get()))?.max(1);
    let mut files = Vec::new();
    collect(Path::new(target), &mut files)?;
    if files.is_empty() {
        return Err(format!("no .ttpf under {target}").into());
    }
    let tables = Tables::builtin()?;
    let mut pools = Vec::new();
    let mut work: Vec<Job> = Vec::new();
    for path in &files {
        let pool = parse(path)?;
        let file = pools.len();
        let mut taken: BTreeMap<u8, usize> = BTreeMap::new();
        let mut by_page: BTreeMap<u64, usize> = BTreeMap::new();
        for (t, &start) in pool.textures.iter().enumerate() {
            let end = pool.textures.get(t + 1).copied().unwrap_or(u64::MAX);
            if end == start {
                continue; // duplicate address
            }
            let first = pool.pages.partition_point(|p| p.addr + p.size <= start);
            let mut pages = pool.pages[first..].iter().take_while(|p| p.addr < end).filter(|p| p.codec != 0).peekable();
            let Some(codec) = pages.peek().map(|p| p.codec) else {
                continue;
            };
            let n = taken.entry(codec).or_default();
            if *n >= limit {
                continue;
            }
            *n += 1;
            for page in pages {
                let at = *by_page.entry(page.addr).or_insert_with(|| {
                    work.push(Job { file, page: *page, textures: Vec::new() });
                    work.len() - 1
                });
                work[at].textures.push(start);
            }
        }
        pools.push(pool);
    }
    println!("{} files, {} BCPack pages of sampled textures, {jobs} threads", files.len(), work.len());

    // Decode on `jobs` threads; each result: (page index, time, error, odd blocks).
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<(usize, Duration, Option<String>, u64)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..jobs)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    let mut opened: BTreeMap<usize, std::fs::File> = BTreeMap::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(job) = work.get(i) else {
                            break done;
                        };
                        let p = job.page;
                        let mut read = || -> Res<Vec<u8>> {
                            use std::os::windows::fs::FileExt;
                            let file = match opened.entry(job.file) {
                                std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
                                std::collections::btree_map::Entry::Vacant(e) => e.insert(std::fs::File::open(&pools[job.file].path)?),
                            };
                            let mut src = vec![0u8; p.stored as usize];
                            file.seek_read(&mut src, p.offset)?;
                            Ok(src)
                        };
                        let (time, error, odd) = match read() {
                            Err(e) => (Duration::ZERO, Some(format!("read: {e}")), 0),
                            Ok(_) if p.zlib => (Duration::ZERO, Some("zlib + BCPack page (none expected)".into()), 0),
                            Ok(src) => {
                                let mut out = vec![0u8; p.size as usize];
                                let t = Instant::now();
                                let r = bcpack::decode(p.codec, &src, &mut out, &tables);
                                let time = t.elapsed();
                                match r {
                                    Ok(()) => (time, None, odd_blocks(p.codec, &out)),
                                    Err(e) => (time, Some(e.to_string()), 0),
                                }
                            }
                        };
                        done.push((i, time, error, odd));
                    }
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    });

    let mut tally: BTreeMap<u8, Tally> = BTreeMap::new();
    let mut texture_failed: BTreeMap<(usize, u64), bool> = BTreeMap::new();
    let mut texture_codec: BTreeMap<(usize, u64), u8> = BTreeMap::new();
    for (i, time, error, odd) in &results {
        let job = &work[*i];
        let t = tally.entry(job.page.codec).or_default();
        t.pages += 1;
        t.blocks += job.page.size / block_size(job.page.codec);
        t.stored += job.page.stored;
        t.time += *time;
        t.odd += odd;
        if let Some(e) = error {
            t.failed_pages += 1;
            if t.first_error.is_none() {
                t.first_error = Some(format!("{} @ {:#x} ({} -> {} bytes): {e}", pools[job.file].path.display(), job.page.offset, job.page.stored, job.page.size));
            }
        }
        for &tex in &job.textures {
            *texture_failed.entry((job.file, tex)).or_default() |= error.is_some();
            texture_codec.entry((job.file, tex)).or_insert(job.page.codec);
        }
    }
    for (key, codec) in &texture_codec {
        let t = tally.entry(*codec).or_default();
        t.textures += 1;
        if texture_failed[key] {
            t.failed_textures += 1;
        }
    }
    println!("{:<6} {:>8} {:>7} {:>10} {:>9} {:>8} {:>8} {:>8} {:>9} {:>6}", "mode", "textures", "pages", "blocks", "stored MB", "tex fail", "pg fail", "ns/block", "MB/s out", "odd");
    let mut total = Tally::default();
    for (codec, t) in &tally {
        let name = match codec {
            1 => "BC1",
            3 => "BC3",
            4 => "BC4",
            5 => "BC5",
            6 => "BC6H",
            7 => "BC7",
            _ => "other",
        };
        let ns = t.time.as_nanos() as f64 / t.blocks.max(1) as f64;
        let mbs = (t.blocks * block_size(*codec)) as f64 / t.time.as_secs_f64().max(1e-9) / 1e6;
        println!(
            "{name:<6} {:>8} {:>7} {:>10} {:>9.1} {:>8} {:>8} {:>8.0} {:>9.1} {:>6}",
            t.textures,
            t.pages,
            t.blocks,
            t.stored as f64 / 1e6,
            t.failed_textures,
            t.failed_pages,
            ns,
            mbs,
            t.odd
        );
        total.textures += t.textures;
        total.pages += t.pages;
        total.blocks += t.blocks;
        total.failed_textures += t.failed_textures;
        total.failed_pages += t.failed_pages;
    }
    println!("total  {:>8} {:>7} {:>10} {:>9} {:>8} {:>8}", total.textures, total.pages, total.blocks, "", total.failed_textures, total.failed_pages);
    for (codec, t) in &tally {
        if let Some(e) = &t.first_error {
            println!("codec {codec}: first failure: {e}");
        }
    }
    Ok(())
}
