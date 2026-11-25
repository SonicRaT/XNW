//! Texture pages the GPU takes from the pack are not read.
//!
//! `XNW_TEXBYPASS=on`: a DirectStorage page request of a `.ttpf` texture
//! whose mips the texture pack (`xnw_formats::texpack`) holds completes at
//! once, without reading, decoding or writing the page. The origin registry
//! records the destination frames as that page with the bytes absent, so a
//! placement finds its origin and the GPU layer uploads from the pack. Index
//! reads, pages of textures the pack lacks or holds without those mips, and
//! volumes go the normal way. Whoever needs the bytes in title memory after
//! all (a fallback upload, a capture) calls
//! `origin::Registry::ensure_present`, which comes back here ([`fill`]) to
//! read, decode and write the page as the request would have.
//! `XNW_TEXBYPASS=probe` bypasses nothing; the GPU layer traps CPU reads of
//! texture pages it took from the registry and counts them (the question
//! the bypass depends on: does the title's CPU read texture pages?).
//!
//! Preload: from the first open of a story pack's file (`pack\story000\...`,
//! the loading screen) for `XNW_PRELOAD_SECS` (default 90), every `.ttpf`
//! page request notes (host `.ttpf`, texture, finest mip asked); at the end
//! of that window they go to `out\cache\preload-<pack>.txt`. On a later run
//! the same open hands that list to the GPU layer, which makes and uploads
//! those textures from the pack ahead of the title (`XNW_PRELOAD_MB`).
//!
//! The write watch needs nothing: page destinations are physical pages,
//! written through views of our own, which the watch never sees either way.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use xnw_formats::origin::{self, NO_TEXTURE};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    Probe,
    On,
}

/// `XNW_TEXBYPASS`: `on`, `probe`, else off.
pub(crate) fn mode() -> Mode {
    static MODE: std::sync::LazyLock<Mode> = std::sync::LazyLock::new(|| match std::env::var("XNW_TEXBYPASS").ok().as_deref().map(str::trim) {
        Some("on") => Mode::On,
        Some("probe") => Mode::Probe,
        _ => Mode::Off,
    });
    *MODE
}

/// A bypassed request: where its page would have gone.
struct Bypassed {
    pages: Vec<usize>,
    page_offset: usize,
}

static BYPASSED: Mutex<Option<HashMap<(u32, u32), Bypassed>>> = Mutex::new(None);

/// Decoding rates for the time a bypass saves, MB/s per worker (L1 run
/// 0001: zlib 1,688 MB in 6.3 s, BCPack 11 MB in 0.9 s).
const ZLIB_MB_S: u64 = 268;
const BCPACK_MB_S: u64 = 12;

/// Called for a page-destination request of registry file `file` before it
/// runs: true if it is complete (bypassed). Also counts every page request
/// and notes it for the preload list.
pub(crate) fn bypass(file: u32, offset: u64, pages: &[usize], page_offset: usize, len: usize) -> bool {
    let registry = origin::registry();
    let Some(index) = registry.file(file) else { return false };
    let Some(r) = index.resolve(offset) else { return false };
    let Some(page) = index.ttpf.pages.get(r.page as usize).copied() else { return false };
    if r.texture == NO_TEXTURE {
        return false;
    }
    let mips = index.mips_in(r.texture, r.at, r.at + page.size);
    note_preload(&index.path, r.texture, u32::from(mips.0));
    let bypassable = mode() == Mode::On
        && origin::bypass_allowed()
        && len as u64 == page.size
        && r.at == page.addr
        && index.texture(r.texture).is_some_and(|t| t.kind != xnw_formats::ttpf::KIND_VOLUME)
        && index.pack_id.and_then(|id| origin::pack()?.lookup(id, r.texture)).is_some_and(|e| e.stored_mips().contains(&u32::from(mips.0)));
    if !bypassable {
        registry.count_request(false, 0, 0, 0);
        return false;
    }
    registry.forget_pages(pages, page_offset, len);
    if registry.record_absent(file, offset, pages, page_offset, len).is_none() {
        registry.count_request(false, 0, 0, 0);
        return false;
    }
    let rate = if page.bcpack() != 0 { BCPACK_MB_S } else if page.zlib() { ZLIB_MB_S } else { 0 };
    let us = if rate == 0 { 0 } else { page.size / rate };
    registry.count_request(true, page.stored, page.size, us);
    let mut b = BYPASSED.lock().unwrap_or_else(|e| e.into_inner());
    b.get_or_insert_with(HashMap::new).insert((file, r.page), Bypassed { pages: pages.to_vec(), page_offset });
    true
}

/// The registry's fill hook: writes a bypassed page into title memory now
/// (read, decode, write, recorded present). True if it did, or the frames
/// no longer hold it.
pub(crate) fn fill(file: u32, page: u32) -> bool {
    let Some(b) = BYPASSED.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|m| m.remove(&(file, page))) else {
        return true;
    };
    let registry = origin::registry();
    if !registry.held_absent(file, page, &b.pages, b.page_offset) {
        return true;
    }
    let Some(index) = registry.file(file) else { return false };
    let Some(p) = index.ttpf.pages.get(page as usize).copied() else { return false };
    let tables = if p.bcpack() != 0 { crate::dstorage::bcpack_tables().ok() } else { None };
    let data = std::fs::File::open(&index.path).map_err(|e| e.to_string()).and_then(|mut f| xnw_formats::ttpf::Ttpf::decode_page(&mut f, &p, tables));
    match data {
        Ok(data) if crate::xmem::write_physical(&b.pages, b.page_offset, &data) => {
            registry.record_pages(file, p.offset, &b.pages, b.page_offset, data.len());
            true
        }
        Ok(_) => false,
        Err(why) => {
            xnw_core::warn!("dstorage: bypassed page {page} of {} not filled: {why}", index.path.display());
            false
        }
    }
}

/// Registers [`fill`] with the registry (once).
pub(crate) fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        origin::set_fill_hook(fill);
        if mode() != Mode::Off {
            xnw_core::info!("dstorage: XNW_TEXBYPASS={}; texture pack {}", if mode() == Mode::On { "on" } else { "probe" }, if origin::pack().is_some() { "found" } else { "missing (nothing is bypassed)" });
        }
    });
}

// Preload lists.

struct Recording {
    key: String,
    until: Instant,
    /// (host `.ttpf`, texture) -> finest mip asked.
    seen: HashMap<(PathBuf, u32), u32>,
}

static RECORDING: Mutex<Option<Recording>> = Mutex::new(None);

fn preload_path(key: &str) -> Option<PathBuf> {
    let dir = std::env::var_os("XNW_CACHE").map_or_else(|| PathBuf::from("out").join("cache"), PathBuf::from);
    (dir.as_os_str() != "off").then(|| dir.join(format!("preload-{key}.txt")))
}

/// A DirectStorage file opened: the first file of a story pack
/// (`...\pack\story000\...`) starts the preload of its list, if one was
/// written, and the recording of this run's.
pub(crate) fn opened(title_path: &str) {
    let lower = title_path.to_ascii_lowercase().replace('/', "\\");
    let Some(key) = story_key(&lower) else { return };
    let mut r = RECORDING.lock().unwrap_or_else(|e| e.into_inner());
    if r.is_some() {
        return;
    }
    let secs = std::env::var("XNW_PRELOAD_SECS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(90u64);
    xnw_core::info!("dstorage: story pack {key} opened: preload list recorded for {secs} s");
    *r = Some(Recording { key: key.clone(), until: Instant::now() + std::time::Duration::from_secs(secs), seen: HashMap::new() });
    drop(r);
    if let Some(path) = preload_path(&key)
        && let Ok(text) = std::fs::read_to_string(&path)
    {
        let _ = std::thread::Builder::new().name("preload list".into()).spawn(move || load(&key, &text));
    }
}

/// The story pack a path lies in (`...\pack\story000\...` -> `story000`).
fn story_key(path: &str) -> Option<String> {
    let lower = path.to_ascii_lowercase().replace('/', "\\");
    let at = lower.find("\\pack\\story")?;
    Some(lower[at + 6..].split('\\').next()?.to_owned()).filter(|k| !k.is_empty())
}

/// Notes a request for the preload list; past the window, writes it.
fn note_preload(host: &std::path::Path, texture: u32, mip: u32) {
    let mut r = RECORDING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(rec) = r.as_mut() else { return };
    if rec.seen.is_empty() && rec.key.is_empty() {
        return; // written already
    }
    if Instant::now() < rec.until {
        let e = rec.seen.entry((host.to_owned(), texture)).or_insert(mip);
        *e = (*e).min(mip);
        return;
    }
    let key = std::mem::take(&mut rec.key);
    let seen = std::mem::take(&mut rec.seen);
    drop(r);
    write_list(&key, &seen);
}

/// `out\cache\preload-<key>.txt`: a comment line, then one line per
/// texture: host `.ttpf` path, texture index, finest mip asked (tabs).
fn write_list(key: &str, seen: &HashMap<(PathBuf, u32), u32>) {
    let Some(path) = preload_path(key) else { return };
    let mut lines: Vec<String> = seen.iter().map(|((p, t), m)| format!("{}\t{t}\t{m}", p.display())).collect();
    lines.sort();
    let text = format!("# xnw preload v1 {key}: host .ttpf, texture, finest mip asked\n{}\n", lines.join("\n"));
    let _ = path.parent().map(std::fs::create_dir_all);
    match std::fs::write(&path, text) {
        Ok(()) => xnw_core::info!("dstorage: preload list {} written: {} textures", path.display(), lines.len()),
        Err(e) => xnw_core::warn!("dstorage: preload list {} not written: {e}", path.display()),
    }
}

/// Reads a preload list and hands it to the GPU layer (registry file ids;
/// each `.ttpf` index is opened here, as DirectStorage would).
fn load(key: &str, text: &str) {
    let registry = origin::registry();
    let mut items = Vec::new();
    let mut ids: HashMap<String, Option<u32>> = HashMap::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut f = line.split('\t');
        let (Some(path), Some(Ok(texture)), Some(Ok(mip))) = (f.next(), f.next().map(str::parse::<u32>), f.next().map(str::parse::<u32>)) else { continue };
        let id = *ids.entry(path.to_owned()).or_insert_with(|| registry.open(std::path::Path::new(path)).ok());
        if let Some(id) = id {
            items.push((id, texture, mip));
        }
    }
    xnw_core::info!("dstorage: preload list for {key}: {} textures in {} files", items.len(), ids.len());
    xnw_gpu::d3d12x::preload_textures(items);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn story_packs_name_their_preload_list() {
        assert_eq!(story_key(r"G:\.\pack\story000\data\ttdb\maps\x.ttpf").as_deref(), Some("story000"));
        assert_eq!(story_key(r"G:\.\data\ttdb\maps\x.ttpf"), None);
    }
}
