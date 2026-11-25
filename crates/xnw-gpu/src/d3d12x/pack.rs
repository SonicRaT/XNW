//! An index of the title's shader pack (`shaders.rpf`): where each shader
//! container lies, found by the content hash of the container (what slots
//! 10, 11, 62 and 86 pass) or by the first 24 bytes of one of its GFX10
//! programs (what slots 57 and 67 pass: a code block of the title's
//! pipeline cache starts with the same 24 bytes as the FSSM program it was
//! compiled from; five-star-recomp "Shaders",
//! GPL-3.0, measured there on 21,502 of 21,522 cache shaders).
//!
//! The index keeps places, not bytes: a container is read back from the
//! file and inflated when a pipeline needs it. It is built once (about 3 s
//! on 12 threads) and saved next to the shader cache, keyed by the pack's
//! size and modification time.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use xnw_shader::rpf9::{Archive, EntryKind};
use xnw_shader::{Container, FourCC, sgd5};

/// One container's place in the pack file.
#[derive(Clone, Debug)]
pub struct Place {
    pub offset: u64,
    pub packed: u32,
    pub unpacked: u32,
    pub name: String,
}

pub struct PackIndex {
    file: std::fs::File,
    places: Vec<Place>,
    /// Each place's container hash (cache code's container, unread).
    hashes: Vec<u128>,
    by_hash: HashMap<u128, u32>,
    by_prefix: HashMap<[u8; 24], u32>,
}

const MAGIC: &[u8; 4] = b"XPI1";

fn stamp(path: &Path) -> std::io::Result<(u64, u64)> {
    let m = std::fs::metadata(path)?;
    let t = m.modified()?.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    Ok((m.len(), t))
}

/// One record's index entries: its container hash and each program's prefix.
type Found = (Place, u128, Vec<[u8; 24]>);

impl PackIndex {
    /// The index of `pack`, from `saved` if that was made from the same
    /// file, else built (and saved there).
    pub fn open(pack: &Path, saved: Option<&Path>) -> std::io::Result<Self> {
        let (size, time) = stamp(pack)?;
        if let Some(saved) = saved
            && let Ok(found) = load(saved, size, time)
        {
            return Ok(Self::from(std::fs::File::open(pack)?, found));
        }
        let found = build(pack)?;
        if let Some(saved) = saved {
            let _ = store(saved, size, time, &found);
        }
        Ok(Self::from(std::fs::File::open(pack)?, found))
    }

    fn from(file: std::fs::File, found: Vec<Found>) -> Self {
        let mut index = PackIndex { file, places: Vec::with_capacity(found.len()), hashes: Vec::with_capacity(found.len()), by_hash: HashMap::new(), by_prefix: HashMap::new() };
        for (place, hash, prefixes) in found {
            let i = index.places.len() as u32;
            index.places.push(place);
            index.hashes.push(hash);
            index.by_hash.entry(hash).or_insert(i);
            for p in prefixes {
                index.by_prefix.entry(p).or_insert(i);
            }
        }
        index
    }

    pub fn len(&self) -> usize {
        self.places.len()
    }

    pub fn is_empty(&self) -> bool {
        self.places.is_empty()
    }

    /// The container whose bytes hash to `hash` ([`xnw_core::hash128`]).
    pub fn by_hash(&self, hash: u128) -> Option<&Place> {
        self.by_hash.get(&hash).map(|&i| &self.places[i as usize])
    }

    /// The container one of whose programs starts with `code`'s 24 bytes.
    pub fn by_code(&self, code: &[u8]) -> Option<&Place> {
        let prefix: [u8; 24] = code.get(..24)?.try_into().ok()?;
        self.by_prefix.get(&prefix).map(|&i| &self.places[i as usize])
    }

    /// The hash of the container [`PackIndex::by_code`] finds for
    /// `code`, without reading it.
    pub fn hash_of_code(&self, code: &[u8]) -> Option<u128> {
        let prefix: [u8; 24] = code.get(..24)?.try_into().ok()?;
        self.by_prefix.get(&prefix).map(|&i| self.hashes[i as usize])
    }

    /// A container's bytes, read back and inflated.
    pub fn container(&self, place: &Place) -> Option<Vec<u8>> {
        let mut packed = vec![0; place.packed as usize];
        let mut at = 0;
        while at < packed.len() {
            match self.file.seek_read(&mut packed[at..], place.offset + at as u64) {
                Ok(0) | Err(_) => return None,
                Ok(n) => at += n,
            }
        }
        sgd5::inflate(&packed, place.unpacked as usize)
    }
}

/// Reads the pack and indexes every record, on all cores.
fn build(pack: &Path) -> std::io::Result<Vec<Found>> {
    let data = std::fs::read(pack)?;
    let archive = Archive::parse(&data).map_err(|e| std::io::Error::other(format!("{e:?}")))?;
    let mut records = Vec::new();
    for (i, e) in archive.entries.iter().enumerate() {
        if e.kind == EntryKind::Sgd5
            && let Some(bytes) = archive.read(i)
        {
            records.extend(sgd5::records(bytes).map(|r| (e.offset, r)));
        }
    }
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = Mutex::new(Vec::with_capacity(records.len()));
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| {
                let mut mine = Vec::new();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((base, r)) = records.get(i) else { break };
                    let Some(bytes) = r.unpack() else { continue };
                    let prefixes = Container::parse(&bytes)
                        .ok()
                        .and_then(|c| c.part(FourCC::DXIL).map(|p| p.data))
                        .and_then(|d| xnw_shader::native_programs(d).ok())
                        .map(|ps| ps.iter().filter_map(|p| p.code.get(..24)?.try_into().ok()).collect())
                        .unwrap_or_default();
                    let place = Place {
                        offset: (base + r.packed_offset()) as u64,
                        packed: r.packed_size() as u32,
                        unpacked: r.unpacked_size() as u32,
                        name: r.name.unwrap_or("?").to_owned(),
                    };
                    mine.push((place, xnw_core::hash128(&[&bytes]), prefixes));
                }
                out.lock().unwrap_or_else(|e| e.into_inner()).append(&mut mine);
            });
        }
    });
    let mut found = out.into_inner().unwrap_or_else(|e| e.into_inner());
    found.sort_by_key(|f| f.0.offset);
    Ok(found)
}

fn store(path: &Path, size: u64, time: u64, found: &[Found]) -> std::io::Result<()> {
    let mut b = Vec::with_capacity(found.len() * 96);
    b.extend_from_slice(MAGIC);
    for v in [size, time, found.len() as u64] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for (p, hash, prefixes) in found {
        b.extend_from_slice(&p.offset.to_le_bytes());
        b.extend_from_slice(&p.packed.to_le_bytes());
        b.extend_from_slice(&p.unpacked.to_le_bytes());
        b.extend_from_slice(&hash.to_le_bytes());
        b.extend_from_slice(&(p.name.len() as u16).to_le_bytes());
        b.extend_from_slice(p.name.as_bytes());
        b.push(prefixes.len() as u8);
        for x in prefixes {
            b.extend_from_slice(x);
        }
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::File::create(&tmp)?.write_all(&b)?;
    std::fs::rename(tmp, path)
}

fn load(path: &Path, size: u64, time: u64) -> std::io::Result<Vec<Found>> {
    let mut b = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut b)?;
    let bad = || std::io::Error::other("bad pack index");
    let mut at = 0;
    let mut take = |n: usize| -> std::io::Result<&[u8]> {
        let s = b.get(at..at + n).ok_or_else(bad)?;
        at += n;
        Ok(s)
    };
    let u64_ = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap_or_default());
    if take(4)? != MAGIC || u64_(take(8)?) != size || u64_(take(8)?) != time {
        return Err(bad());
    }
    let count = u64_(take(8)?) as usize;
    let mut found = Vec::with_capacity(count);
    for _ in 0..count {
        let offset = u64_(take(8)?);
        let packed = u32::from_le_bytes(take(4)?.try_into().unwrap_or_default());
        let unpacked = u32::from_le_bytes(take(4)?.try_into().unwrap_or_default());
        let hash = u128::from_le_bytes(take(16)?.try_into().unwrap_or_default());
        let n = usize::from(u16::from_le_bytes(take(2)?.try_into().unwrap_or_default()));
        let name = String::from_utf8_lossy(take(n)?).into_owned();
        let k = usize::from(take(1)?[0]);
        let mut prefixes = Vec::with_capacity(k);
        for _ in 0..k {
            prefixes.push(take(24)?.try_into().unwrap_or_default());
        }
        found.push((Place { offset, packed, unpacked, name }, hash, prefixes));
    }
    Ok(found)
}

/// Where the pack is for a run: `XNW_SHADER_PACK`, else `shaders.rpf` on
/// the game drive (`XNW_GAME_DRIVE`).
pub fn default_pack() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("XNW_SHADER_PACK") {
        return Some(PathBuf::from(p));
    }
    let drive = std::env::var_os("XNW_GAME_DRIVE")?;
    Some(PathBuf::from(drive).join("shaders.rpf")).filter(|p| p.exists())
}
