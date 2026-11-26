//! Translated shaders kept on disk: one file per translation under
//! `<cache>/shaders/hh/rest`, written to a temporary file and renamed, so
//! a reader never sees half an entry; an entry that does not read back is
//! translated again.
//!
//! Ported from VIctory `crates/victory-gpu/src/shader_cache.rs` 1-359 (the
//! `VSH1` entry encoding 134-278, the cache flow 102-132, its tests 280-359)
//! and `crates/victory-pipeline/src/cache.rs` 12-63 (content-addressed
//! store, temp file + rename), xnw/base 5d0516e (VIctory d2a4bae + PRs, no
//! PR touches these files), GPL-3.0-or-later. Changed:
//! the key is `hash128(format version, VERSION_BREAKING, policy, form,
//! hash of the console container translated)`, not the hash of our
//! executable (which changed on every rebuild); no `anyhow`, no `blake3`,
//! no `victory-pipeline`.
//!
//! One cache per translator generation (D-cache rule from VIctory,
//! `crates/victory-gpu/src/pipeline/library.rs` at upstream, GPL-3.0-or-later:
//! the pipeline library keyed by the translator's hash, the newest two kept,
//! the older deleted, after a translator edit overwrote a 926 MB library).
//! Here the generation is `VERSION_BREAKING`: entries live under
//! `<cache>/shaders/g<N>/hh/rest`; [`evict`] keeps the two highest `g<N>`
//! and deletes the rest, so a bisect back one generation still finds its
//! cache. Folders of two hex digits straight under `shaders/` are the cache
//! from before generations: left alone (the format-1 lookup in
//! [`first_version`] reads them); delete them by hand once no run needs them.
//!
//! The policy is the multiply-add mode (`fast` by default,
//! `XNW_FMA=exact` for all, `XNW_FMA_EXACT=<hash>,...` for listed shaders)
//! and the loop limit.
//!
//! The key rule (stated for shader changes in `xnw_shader::emit`
//! `VERSION`): the key holds `VERSION_BREAKING` (and `MESH_VERSION` for
//! mesh programs), not `VERSION`. A translation that failed is kept as
//! `<entry>.fail` holding the `VERSION` it failed under and the reason: the
//! same `VERSION` gives the same failure without translating, another
//! retries. A translation that succeeded is kept until `VERSION_BREAKING`
//! moves. The key names the container (for cache code: the pack container
//! it was found in), so `xnw prebuild` can fill the cache without the title.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use xnw_shader::emit::{Fma, HostShader, Options, StaticSampler, UserData};
use xnw_shader::{HwStage, RootType};

/// Raised when the entry encoding or the key's layout changes.
/// 2: container hash, `VERSION_BREAKING`.
const FORMAT: u32 = 2;

static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
pub(crate) static HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static TRANSLATED: AtomicU64 = AtomicU64::new(0);
pub(crate) static FAILED: AtomicU64 = AtomicU64::new(0);
/// Of [`FAILED`]: answered by a failure entry of this `VERSION`.
pub(crate) static FAILED_KEPT: AtomicU64 = AtomicU64::new(0);

/// The cache folder: `XNW_CACHE`, else `out/cache` under the current
/// folder; `XNW_CACHE=off` keeps nothing.
pub(crate) fn dir() -> Option<&'static Path> {
    DIR.get_or_init(|| {
        let dir = match std::env::var_os("XNW_CACHE") {
            Some(v) if v == "off" => None,
            Some(v) => Some(PathBuf::from(v)),
            None => Some(PathBuf::from("out").join("cache")),
        };
        if let Some(d) = &dir {
            let gone = evict(d);
            if !gone.is_empty() {
                xnw_core::info!("shader cache: generations {gone:?} deleted (newest {KEEP} kept)");
            }
        }
        dir
    })
    .as_deref()
}

/// Generations kept on disk: the current and the one before.
const KEEP: usize = 2;

/// Deletes all but the newest [`KEEP`] generation folders of `dir` (the
/// current one counts as the newest even if no entry is in it yet); returns
/// the generations deleted.
pub(crate) fn evict(dir: &Path) -> Vec<u32> {
    evict_at(dir, xnw_shader::emit::VERSION_BREAKING)
}

fn evict_at(dir: &Path, current: u32) -> Vec<u32> {
    let root = dir.join("shaders");
    let mut gens: Vec<u32> = std::fs::read_dir(&root)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok()?.file_name().to_str()?.strip_prefix('g')?.parse().ok())
        .filter(|&g| g != current)
        .collect();
    gens.sort_unstable_by(|a, b| b.cmp(a));
    let old = gens.split_off(gens.len().min(KEEP - 1));
    for g in &old {
        let _ = std::fs::remove_dir_all(root.join(format!("g{g}")));
    }
    old
}

/// The cache key of a translation of the console container whose bytes
/// hash (`xnw_core::hash128`) to `container`.
pub(crate) fn key(form: u8, container: u128, opts: Options) -> u128 {
    key_at(xnw_shader::emit::VERSION_BREAKING, form, container, opts)
}

/// The key under `breaking` (a `VERSION_BREAKING`; 0 names the program
/// itself, for its first-version record).
fn key_at(breaking: u32, form: u8, container: u128, opts: Options) -> u128 {
    let (fma, fixed, desc) = key_parts(breaking, FORMAT, opts);
    // Mesh programs in the fallback form carry its own version.
    let mesh = if opts.mesh == xnw_shader::emit::MeshForm::Fallback { xnw_shader::emit::MESH_VERSION } else { 0 };
    xnw_core::hash128(&[b"xnw shader", &fixed, &[fma, form, desc], &mesh.to_le_bytes(), env!("CARGO_PKG_VERSION").as_bytes(), &container.to_le_bytes()])
}

/// The policy parts of a key: multiply-add mode, the fixed words, and the
/// descriptor-mode/probe byte.
fn key_parts(version: u32, format: u32, opts: Options) -> (u8, Vec<u8>, u8) {
    let fma: u8 = match opts.fma {
        Fma::Exact => 0,
        Fma::Fast => 1,
    };
    // XNW: the loop limits count only when a switch changes them.
    let fixed = [&format.to_le_bytes()[..], &version.to_le_bytes(), &xnw_shader::emit::DEFAULT_LOOP_LIMIT.to_le_bytes(), &xnw_shader::emit::loop_limit_key(), &xnw_shader::emit::shader_opt_key()].concat();
    // XNW: the descriptor mode is part of the key; so is the probe.
    // So are the loop counts, and the host's barycentrics: a key
    // without them is the key from before.
    let desc = u8::from(opts.descriptors == xnw_shader::emit::Descriptors::Gpu) | u8::from(opts.probe) << 1 | u8::from(opts.loop_counts.is_some()) << 2 | u8::from(opts.pagetable) << 3 | u8::from(opts.barycentrics) << 4 | u8::from(opts.nonuniform_heap) << 5 | u8::from(opts.keep_uniform_heap) << 6 | u8::from(opts.masked_lanes) << 7;
    (fma, fixed, desc)
}

/// The key the cache had before format 2 (format 1: keyed by `VERSION` and the
/// title's bytes), to read the first-version facts from entries made then.
fn legacy_key(version: u32, form: u8, console: &[u8], opts: Options) -> u128 {
    let (fma, fixed, desc) = key_parts(version, 1, opts);
    if opts.mesh == xnw_shader::emit::MeshForm::Fallback {
        let mesh = xnw_shader::emit::MESH_VERSION.to_le_bytes();
        return xnw_core::hash128(&[b"xnw shader", &fixed, &[fma, form, desc], &mesh, env!("CARGO_PKG_VERSION").as_bytes(), console]);
    }
    xnw_core::hash128(&[b"xnw shader", &fixed, &[fma, form, desc], env!("CARGO_PKG_VERSION").as_bytes(), console])
}

/// XNW: the translator version that first translated this program
/// (`XNW_REFUSE_NEW`): kept in `<cache>/first/hh/rest` (one little-endian
/// u32, keyed as `VERSION_BREAKING` 0) from its first translation on; for
/// programs translated before that file existed, the oldest version whose
/// format-1 cache entry (keyed by `console`, the title's bytes) is still on
/// disk. `None`: never translated here.
pub(crate) fn first_version(dir: &Path, form: u8, container: u128, console: &[u8], opts: Options) -> Option<u32> {
    let side = first_path(dir, form, container, opts);
    if let Some(v) = std::fs::read(&side).ok().and_then(|b| Some(u32::from_le_bytes(b.get(..4)?.try_into().ok()?))) {
        return Some(v);
    }
    let v = (1..=xnw_shader::emit::VERSION).find(|&v| flat_path(dir, legacy_key(v, form, console, opts)).exists())?;
    let _ = write_atomically(&side, &v.to_le_bytes());
    Some(v)
}

/// Records the current translator version as the first to translate the
/// program, unless one is known ([`first_version`]).
pub(crate) fn note_first(dir: &Path, form: u8, container: u128, console: &[u8], opts: Options) {
    if first_version(dir, form, container, console, opts).is_none() {
        let _ = write_atomically(&first_path(dir, form, container, opts), &xnw_shader::emit::VERSION.to_le_bytes());
    }
}

fn first_path(dir: &Path, form: u8, container: u128, opts: Options) -> PathBuf {
    let hex = format!("{:032x}", key_at(0, form, container, opts));
    dir.join("first").join(&hex[..2]).join(&hex[2..])
}

fn path(dir: &Path, key: u128) -> PathBuf {
    let hex = format!("{key:032x}");
    dir.join("shaders").join(format!("g{}", xnw_shader::emit::VERSION_BREAKING)).join(&hex[..2]).join(&hex[2..])
}

/// Where a format-1 entry lay (before generations).
fn flat_path(dir: &Path, key: u128) -> PathBuf {
    let hex = format!("{key:032x}");
    dir.join("shaders").join(&hex[..2]).join(&hex[2..])
}

fn fail_path(dir: &Path, key: u128) -> PathBuf {
    let mut p = path(dir, key).into_os_string();
    p.push(".fail");
    PathBuf::from(p)
}

/// A console container the title passed that the pack does not hold
/// (slots 10/11/62/86 code from the executable), kept by its hash for the
/// prebuild: `<cache>/containers/hh/rest`.
pub(crate) fn keep_container(dir: &Path, hash: u128, bytes: &[u8]) {
    let hex = format!("{hash:032x}");
    let p = dir.join("containers").join(&hex[..2]).join(&hex[2..]);
    if !p.is_file() {
        let _ = write_atomically(&p, bytes);
    }
}

/// A container [`keep_container`] kept, if it reads back whole.
pub(crate) fn kept_container(dir: &Path, hash: u128) -> Option<Vec<u8>> {
    let hex = format!("{hash:032x}");
    let bytes = std::fs::read(dir.join("containers").join(&hex[..2]).join(&hex[2..])).ok()?;
    (xnw_core::hash128(&[&bytes]) == hash).then_some(bytes)
}

/// What the cache holds under `key` (no translation).
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Held {
    Translated,
    /// Failed under this `VERSION`, for this reason.
    Failed(String),
    /// Nothing, or a failure of another `VERSION`: to translate.
    Missing,
}

/// What `dir` holds under `key` for translator `version`.
pub(crate) fn held(dir: &Path, key: u128, version: u32) -> Held {
    if path(dir, key).is_file() {
        return Held::Translated;
    }
    match std::fs::read_to_string(fail_path(dir, key)).ok().and_then(|t| parse_fail(&t)) {
        Some((v, reason)) if v == version => Held::Failed(reason),
        _ => Held::Missing,
    }
}

fn parse_fail(text: &str) -> Option<(u32, String)> {
    let (v, reason) = text.split_once('\t')?;
    Some((v.trim().parse().ok()?, reason.to_owned()))
}

/// The translation under `key`: from the cache in `dir` when there (a
/// failure only if it failed under this `VERSION`), else `translate`d and
/// stored.
pub(crate) fn cached(dir: Option<&Path>, key: u128, translate: impl FnOnce() -> Result<HostShader, String>) -> Result<HostShader, String> {
    cached_at(dir, key, xnw_shader::emit::VERSION, translate)
}

fn cached_at(dir: Option<&Path>, key: u128, version: u32, translate: impl FnOnce() -> Result<HostShader, String>) -> Result<HostShader, String> {
    if let Some(dir) = dir {
        let p = path(dir, key);
        if let Ok(bytes) = std::fs::read(&p) {
            if let Some(shader) = decode(&bytes) {
                HITS.fetch_add(1, Ordering::Relaxed);
                return Ok(shader);
            }
            let _ = std::fs::remove_file(&p);
        }
        if let Held::Failed(reason) = held(dir, key, version) {
            FAILED.fetch_add(1, Ordering::Relaxed);
            FAILED_KEPT.fetch_add(1, Ordering::Relaxed);
            return Err(reason);
        }
    }
    let shader = match translate() {
        Ok(s) => s,
        Err(e) => {
            FAILED.fetch_add(1, Ordering::Relaxed);
            if let Some(dir) = dir {
                let _ = write_atomically(&fail_path(dir, key), format!("{version}\t{e}").as_bytes());
            }
            return Err(e);
        }
    };
    TRANSLATED.fetch_add(1, Ordering::Relaxed);
    if let Some(dir) = dir {
        let _ = write_atomically(&path(dir, key), &encode(&shader));
        let _ = std::fs::remove_file(fail_path(dir, key));
    }
    Ok(shader)
}

fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = path.parent().expect("hashed path has a parent");
    std::fs::create_dir_all(dir)?;
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!(".tmp-{}-{n}", std::process::id()));
    if let Err(e) = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

const MAGIC: &[u8; 4] = b"VSH1";
const STAGES: [HwStage; 8] = [HwStage::Ls, HwStage::Hs, HwStage::Es, HwStage::Gs, HwStage::Vs, HwStage::Ps, HwStage::Ms, HwStage::Cs];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [t.filter, u, v, w, t.mip_lod_bias.to_bits(), t.max_anisotropy, t.comparison, t.border, t.min_lod.to_bits(), t.max_lod.to_bits()];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, u32::from(s.address_map) | u32::from(s.descriptor_words) << 1);
    word(&mut out, s.half_views.len() as u32);
    for &(slot, dword) in &s.half_views {
        word(&mut out, u32::from(slot));
        word(&mut out, dword);
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData { index: r.word()?, slot: u8::try_from(r.word()?).ok()?, root: root_type(r.word()?), dword: r.word()? });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let bits = r.word()?;
    let (address_map, descriptor_words) = (bits & 1 != 0, bits & 2 != 0);
    let n = r.word()? as usize;
    let mut half_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        half_views.push((u8::try_from(r.word()?).ok()?, r.word()?));
    }
    r.0.is_empty().then_some(HostShader { stage, dxil, user_data, samplers, static_samplers, root_cbvs, root_srvs, root_uavs, half_views, address_map, descriptor_words })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData { index: 8, slot: 1, root: RootType::Table, dword: 0 }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([0x400, 0xc0_0000, 0x18f0_0000, 0])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: Vec::new(),
            half_views: vec![(1, 8)],
            address_map: false,
            descriptor_words: false,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xnw-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// The newest two generations stay; the legacy flat folders are left alone.
    #[test]
    fn eviction_keeps_the_newest_two_generations() {
        let dir = scratch("evict");
        for g in ["g3", "g5", "g6", "g9", "ab"] {
            std::fs::create_dir_all(dir.join("shaders").join(g)).unwrap();
        }
        let mut gone = evict_at(&dir, 7);
        gone.sort_unstable();
        assert_eq!(gone, [3, 5, 6], "the current 7 and the newest other, 9, stay");
        let left: Vec<_> = ["g3", "g5", "g6", "g9", "ab"].iter().filter(|g| dir.join("shaders").join(g).is_dir()).collect();
        assert_eq!(left, [&"g9", &"ab"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// XNW: the first translating version, for `XNW_REFUSE_NEW`.
    #[test]
    fn the_first_translating_version_is_kept() {
        let dir = scratch("first-version");
        let o = Options::default();
        assert_eq!(first_version(&dir, 0, 1, b"new", o), None);
        note_first(&dir, 0, 1, b"new", o);
        assert_eq!(first_version(&dir, 0, 1, b"new", o), Some(xnw_shader::emit::VERSION));
        // An older version's format-1 entry on disk, no side file: that version.
        let old = flat_path(&dir, legacy_key(11, 0, b"old", o));
        write_atomically(&old, &encode(&shader())).unwrap();
        note_first(&dir, 0, 2, b"old", o);
        assert_eq!(first_version(&dir, 0, 2, b"old", o), Some(11));
        // The record does not move with VERSION_BREAKING (keyed as 0).
        assert_ne!(first_path(&dir, 0, 2, o), path(&dir, key(0, 2, o)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = scratch("shader-cache");
        let runs = AtomicU64::new(0);
        let translate = || {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let fast = Options { fma: Fma::Fast, ..Default::default() };
        let k = key(0, 0xc0ffee, fast);
        assert_eq!(cached(Some(&dir), k, translate), Ok(shader()));
        assert_eq!(cached(Some(&dir), k, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another policy or form is another entry.
        assert_ne!(k, key(0, 0xc0ffee, Options { fma: Fma::Exact, ..Default::default() }));
        assert_ne!(k, key(1, 0xc0ffee, fast));

        // A damaged entry is a miss, translated and stored again.
        std::fs::write(path(&dir, k), b"VSH1 broken").unwrap();
        assert_eq!(cached(Some(&dir), k, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert_eq!(cached(Some(&dir), k, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A failure is kept as a failure entry, not a translation.
        let k2 = key(0, 0xbad, fast);
        assert!(cached(Some(&dir), k2, || Err("unsupported".to_owned())).is_err());
        assert!(!path(&dir, k2).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The rule: VERSION moves, translations stay and failures are retried;
    /// VERSION_BREAKING moves, everything is another entry.
    #[test]
    fn version_retries_failures_and_breaking_moves_every_key() {
        let dir = scratch("shader-cache-rule");
        let fast = Options { fma: Fma::Fast, ..Default::default() };
        let (good, bad) = (key_at(7, 0, 1, fast), key_at(7, 0, 2, fast));
        let runs = AtomicU64::new(0);
        let ok = || {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let fail = || {
            runs.fetch_add(1, Ordering::Relaxed);
            Err("refused: v_foo".to_owned())
        };
        // Version 13: one translates, one fails.
        assert_eq!(cached_at(Some(&dir), good, 13, ok), Ok(shader()));
        assert_eq!(cached_at(Some(&dir), bad, 13, fail), Err("refused: v_foo".to_owned()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert_eq!(held(&dir, bad, 13), Held::Failed("refused: v_foo".into()));
        // Same version: the failure is answered from disk, not retried.
        assert_eq!(cached_at(Some(&dir), bad, 13, fail), Err("refused: v_foo".to_owned()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        // Version 14: the translation stays, the failure is retried (and now translates).
        assert_eq!(held(&dir, good, 14), Held::Translated);
        assert_eq!(held(&dir, bad, 14), Held::Missing);
        assert_eq!(cached_at(Some(&dir), good, 14, fail), Ok(shader()));
        assert_eq!(cached_at(Some(&dir), bad, 14, ok), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert!(!fail_path(&dir, bad).exists(), "a success removes the failure entry");
        // VERSION_BREAKING: every key moves.
        assert_ne!(good, key_at(8, 0, 1, fast));
        assert_eq!(held(&dir, key_at(8, 0, 1, fast), 14), Held::Missing);
        // Mesh programs carry MESH_VERSION: another key than the native form.
        let mesh = Options { mesh: xnw_shader::emit::MeshForm::Fallback, ..fast };
        assert_ne!(good, key_at(7, 0, 1, mesh));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }
}
