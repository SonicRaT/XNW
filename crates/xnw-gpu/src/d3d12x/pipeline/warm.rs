//! The shader cache and the host pipeline library warmed from the
//! pipeline list, without the title: `xnw prebuild` and the launch prebuild
//! (`xnw run --execute`, before the title's entry point) share this.
//!
//! A listed pipeline's recipe (`library.rs`) names each shader by stage,
//! form, the hash of the bytes the title passed (what the pipeline key and
//! the multiply-add list name) and the hash of the console container it
//! translates (what the shader cache key names; cache code is found in the
//! pack by it, as the run finds it by prefix), then the graphics state
//! (`GraphicsState::encode`, hex) or `-` for compute:
//! `vertex:0:<hash>:<container>,pixel:1:<hash>:<container>;<state hex>`.
//!
//! Every distinct listed shader whose translation the cache does not hold
//! for this build (`shader_cache::held`) is read from the pack and
//! translated on all cores; then every listed pipeline is made, earliest
//! first use first (`HostPipeline::new`, our root signature from the
//! translations, the state from the recipe: a load when the host library
//! holds it, else a compile; the library names a pipeline by its DXIL,
//! so any translator change needs them compiled again) and the library is
//! saved; `pipelines/prebuilt.stamp` then names the list, translator and
//! library length this was done for, so a pass with the same does nothing. A run then finds the translations on
//! disk and loads the pipelines from the library. Restartable: whatever
//! finished is on disk; a second pass finds it held.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::host::{GraphicsState, HostPipeline};
use super::super::shader_cache;
use super::{Form, Shader, library, make};

const STAGES: [&str; 8] = ["vertex", "pixel", "compute", "mesh", "fetch", "domain", "hull", "geometry"];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    (text.len().is_multiple_of(2)).then_some(())?;
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok()).collect()
}

/// The recipe of a pipeline made from `shaders` and `g`, once every
/// shader's container is known (its translation ran).
pub(crate) fn recipe_of(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<String> {
    let mut parts = Vec::with_capacity(shaders.len());
    for s in shaders {
        parts.push(format!("{}:{}:{:032x}:{:032x}", s.stage, s.form as u8, s.hash, s.container_hash()?));
    }
    let state = g.map_or_else(|| "-".to_owned(), |g| hex(&g.encode()));
    Some(format!("{};{state}", parts.join(",")))
}

/// The recipe of a title pipeline before anything is translated: cache
/// code's container found in the pack index by prefix (the in-run
/// prebuild, when it matches a listed pipeline that has no recipe yet).
pub(crate) fn early_recipe(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<String> {
    let mut parts = Vec::with_capacity(shaders.len());
    for s in shaders {
        let container = match (s.container_hash(), s.form) {
            (Some(c), _) => c,
            (None, Form::Container) => s.hash,
            (None, Form::CacheCode) => super::pack()?.hash_of_code(s.bytes())?,
        };
        parts.push(format!("{}:{}:{:032x}:{container:032x}", s.stage, s.form as u8, s.hash));
    }
    let state = g.map_or_else(|| "-".to_owned(), |g| hex(&g.encode()));
    Some(format!("{};{state}", parts.join(",")))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ShaderRef {
    stage: &'static str,
    form: u8,
    hash: u128,
    container: u128,
}

struct Recipe {
    shaders: Vec<ShaderRef>,
    graphics: Option<GraphicsState>,
}

fn parse_recipe(text: &str) -> Option<Recipe> {
    let (shaders, state) = text.split_once(';')?;
    let shaders = shaders
        .split(',')
        .map(|s| {
            let mut f = s.split(':');
            let name = f.next()?;
            let stage = *STAGES.iter().find(|&&n| n == name)?;
            let form: u8 = f.next()?.parse().ok().filter(|&f| f <= 1)?;
            let hash = u128::from_str_radix(f.next()?, 16).ok()?;
            let container = u128::from_str_radix(f.next()?, 16).ok()?;
            Some(ShaderRef { stage, form, hash, container })
        })
        .collect::<Option<Vec<_>>>()?;
    let mut graphics = if state == "-" { None } else { Some(GraphicsState::decode(&unhex(state)?)?) };
    // As the run makes it (`Desc::graphics_state`): no blending on integer
    // targets. Recipes listed before 1efe0565 keep the title's blend there
    // (SWEEP1: pipeline 80e918ca, WindshieldWiperMask, targets R32_FLOAT,
    // B4G4R4A4_UNORM, R8_UINT with blend on target 2, refused 0x80070057 by
    // the RX 5700 XT and WARP in `card-sweep`).
    if let Some(g) = graphics.as_mut() {
        g.apply_integer_blend_bypass();
    }
    Some(Recipe { shaders, graphics })
}

/// A pipeline pass that finished leaves the stamp of what it was made
/// for: the list's recipes and the translator. Written only after a
/// whole pass (no `--only`/`--skip`/`--from`/`--to`), no device loss and the library
/// saved with every pipeline stored; `v2` in the hash: stamps written before
/// One was written after a device loss: are not trusted.
const STAMP_FILE: &str = "prebuilt.stamp";

fn stamp_of(recipes: &[String], library_len: u64) -> u128 {
    let mut sorted: Vec<&str> = recipes.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let fixed = [xnw_shader::emit::VERSION, xnw_shader::emit::VERSION_BREAKING, xnw_shader::emit::MESH_VERSION, u32::from(super::super::descriptor::host::gpu_mode())].map(u32::to_le_bytes).concat();
    let loops = xnw_shader::emit::loop_limit_key();
    let len = library_len.to_le_bytes();
    let mut parts: Vec<&[u8]> = vec![b"xnw prebuilt v2", &fixed, &loops, &len, env!("CARGO_PKG_VERSION").as_bytes()];
    parts.extend(sorted.iter().map(|s| s.as_bytes()));
    xnw_core::hash128(&parts)
}

/// How a prebuild runs.
pub struct Settings {
    /// Threads.
    pub jobs: usize,
    /// Also make the host pipelines (else translations only).
    pub pipelines: bool,
    /// The shader pack (else `XNW_SHADER_PACK`, `XNW_GAME_DRIVE`).
    pub pack: Option<PathBuf>,
    /// Do nothing unless more than this many listed shaders are missing.
    pub threshold: usize,
    /// A progress line on the console, rewritten in place.
    pub console: bool,
    /// Only these pipelines (key or program hash prefixes; empty: all).
    pub only: Vec<String>,
    /// Not these (key or program hash prefixes).
    pub skip: Vec<String>,
    /// Positions in the earliest-first order, `from..=to`.
    pub from: Option<usize>,
    pub to: Option<usize>,
    /// Name a pipeline every 5 s once it compiles this long.
    pub timeout: Option<Duration>,
    /// Build the suspects of an earlier device loss too.
    pub retry_suspects: bool,
}

impl Settings {
    /// All pipelines, `jobs` threads.
    pub fn new(jobs: usize, pipelines: bool, pack: Option<PathBuf>, threshold: usize, console: bool) -> Settings {
        Settings { jobs, pipelines, pack, threshold, console, only: Vec::new(), skip: Vec::new(), from: None, to: None, timeout: None, retry_suspects: false }
    }

    fn whole(&self) -> bool {
        self.only.is_empty() && self.skip.is_empty() && self.from.is_none() && self.to.is_none()
    }
}

/// What a prebuild found and did.
#[derive(Default, Debug)]
pub struct Outcome {
    /// Listed pipelines, and those with a recipe.
    pub listed: usize,
    pub with_recipe: usize,
    /// Distinct listed shaders; of them held translated, held failed (this
    /// version), and missing (to translate).
    pub shaders: usize,
    pub cached: usize,
    pub failed_before: usize,
    pub missing: usize,
    /// This pass: translated, failed by reason.
    pub translated: usize,
    pub failed: Vec<(String, usize)>,
    /// Pipelines over a newly translated shader: made (compiled), loaded
    /// from the library, refused, and recipes whose key is not the listed one.
    pub pipelines: usize,
    pub made: u64,
    pub loaded: u64,
    pub refused: usize,
    pub key_mismatch: usize,
    /// Whether anything was to do (else nothing ran).
    pub needed: bool,
    /// Seconds the pipeline pass took.
    pub pipeline_seconds: f64,
    /// The host device was lost during the pass (exit non-zero).
    pub device_lost: bool,
    /// Suspects of an earlier device loss skipped; pipelines left out
    /// by the filters.
    pub suspects_skipped: usize,
    pub filtered_out: usize,
    /// Why pipelines were not made, when they were not.
    pub note: String,
    pub seconds: f64,
}

impl Outcome {
    pub fn failed_total(&self) -> usize {
        self.failed.iter().map(|f| f.1).sum()
    }
}

/// Progress for the console line and the log.
struct Progress {
    what: &'static str,
    total: usize,
    done: AtomicUsize,
    failed: AtomicUsize,
    started: Instant,
}

impl Progress {
    fn line(&self) -> String {
        let (done, failed) = (self.done.load(Ordering::Relaxed), self.failed.load(Ordering::Relaxed));
        let secs = self.started.elapsed().as_secs_f64();
        let left = if done > 0 { secs / done as f64 * (self.total - done.min(self.total)) as f64 } else { 0.0 };
        format!("prebuild {done}/{} {}, {failed} failed, {secs:.0} s, about {left:.0} s left", self.total, self.what)
    }
}

/// Runs `work(i)` for `i` in `0..n` on `jobs` threads, with the console
/// line (5 times a second) and a log line every 5 s.
fn on_all_cores(p: &Progress, jobs: usize, console: bool, work: impl Fn(usize) -> bool + Sync) {
    let next = AtomicUsize::new(0);
    let over = AtomicBool::new(false);
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= p.total {
                        break;
                    }
                    if !work(i) {
                        p.failed.fetch_add(1, Ordering::Relaxed);
                    }
                    p.done.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        s.spawn(|| {
            let mut logged = Instant::now();
            while !over.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                if console {
                    eprint!("\r{:<78}", p.line());
                }
                if logged.elapsed() >= Duration::from_secs(5) {
                    logged = Instant::now();
                    if console {
                        eprint!("\r{:78}\r", "");
                    }
                    xnw_core::info!("{}", p.line());
                }
                if p.done.load(Ordering::Relaxed) >= p.total {
                    break;
                }
            }
        });
        // The workers end when the work does; the printer then sees it.
        while p.done.load(Ordering::Relaxed) < p.total {
            std::thread::sleep(Duration::from_millis(20));
        }
        over.store(true, Ordering::Relaxed);
    });
    if console {
        eprint!("\r{:78}\r", "");
    }
}

/// Short reasons: the first line, the first 120 characters.
fn reason(e: &str) -> String {
    let first = e.lines().next().unwrap_or_default();
    first.chars().take(120).collect()
}

/// The prebuild: see the module comment.
pub fn warm(s: &Settings) -> Result<Outcome, String> {
    let started = Instant::now();
    if let Some(p) = &s.pack {
        let _ = super::PACK_PATH.set(p.clone());
    }
    super::init_cache_dir();
    let dir = shader_cache::dir().ok_or("XNW_CACHE=off: no shader cache to fill")?;
    let rows = library::rows();
    let mut out = Outcome { listed: rows.len(), ..Outcome::default() };
    // Earliest first use first (what the title draws first is made first).
    let mut rows = rows;
    rows.sort_by(|a, b| a.1.ms.cmp(&b.1.ms).then_with(|| a.0.cmp(&b.0)));
    let mut recipes: Vec<(String, Recipe)> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    for (key, row) in rows {
        if let Some(text) = row.recipe.as_deref()
            && let Some(r) = parse_recipe(text)
        {
            texts.push(format!("{key}\t{text}"));
            recipes.push((key, r));
        }
    }
    out.with_recipe = recipes.len();
    // The diagnostic filters, over the earliest-first order; suspects
    // of an earlier device loss are left out unless retried.
    super::compile::retry_suspects(s.retry_suspects);
    if let Some(t) = s.timeout {
        super::compile::still_after(t);
    }
    let pos: HashMap<String, usize> = recipes.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect();
    let named = |key: &str, r: &Recipe, pats: &[String]| pats.iter().any(|p| key.starts_with(p.as_str()) || r.shaders.iter().any(|x| format!("{:032x}", x.hash).starts_with(p.as_str())));
    let before = recipes.len();
    recipes.retain(|(k, r)| {
        let i = pos[k];
        s.from.is_none_or(|f| i >= f) && s.to.is_none_or(|t| i <= t) && (s.only.is_empty() || named(k, r, &s.only)) && !named(k, r, &s.skip)
    });
    out.filtered_out = before - recipes.len();
    let suspects_before = recipes.len();
    recipes.retain(|(k, _)| !super::compile::listed_suspect(k));
    out.suspects_skipped = suspects_before - recipes.len();
    if out.suspects_skipped > 0 {
        xnw_core::warn!("prebuild: {} suspects of an earlier device loss skipped (suspect.txt; --retry-suspects builds them)", out.suspects_skipped);
    }
    // With the library file's length: a run that stored pipelines (or a
    // driver that refused the library) changes it.
    let library_len = || std::fs::metadata(library::library_file(&dir.join("pipelines"))).map_or(0, |m| m.len());
    let stamp = || format!("{:032x}", stamp_of(&texts, library_len()));
    let stamp_file = dir.join("pipelines").join(STAMP_FILE);

    // Distinct shaders and what the cache holds of them.
    let mut seen = HashSet::new();
    let mut missing = Vec::new();
    for r in recipes.iter().flat_map(|(_, r)| &r.shaders) {
        if !seen.insert(*r) {
            continue;
        }
        let key = shader_cache::key(r.form, r.container, super::options_for(r.stage, r.hash));
        match shader_cache::held(dir, key, xnw_shader::emit::VERSION) {
            shader_cache::Held::Translated => out.cached += 1,
            shader_cache::Held::Failed(_) => out.failed_before += 1,
            shader_cache::Held::Missing => missing.push(*r),
        }
    }
    out.shaders = seen.len();
    out.missing = missing.len();
    // The pipelines are made for every listed recipe (the library names
    // a pipeline by its DXIL, so any translator change needs them all made
    // again); a pass that finished for this very list and translator leaves
    // a stamp, and the next finds nothing to do.
    let pipelines_done = s.whole() && std::fs::read_to_string(&stamp_file).is_ok_and(|t| t.trim() == stamp());
    if missing.len() <= s.threshold && (!s.pipelines || pipelines_done || recipes.is_empty()) {
        out.seconds = started.elapsed().as_secs_f64();
        return Ok(out);
    }
    out.needed = true;
    let pack = super::pack().ok_or("no shader pack index (--pack, XNW_SHADER_PACK or XNW_GAME_DRIVE)")?;
    let form = |f: u8| if f == 0 { Form::Container } else { Form::CacheCode };
    let load = |r: &ShaderRef| -> Result<Arc<Shader>, String> {
        let bytes = match pack.by_hash(r.container) {
            Some(place) => pack.container(place).ok_or("pack record unreadable")?,
            None => shader_cache::kept_container(dir, r.container).ok_or("listed container neither in the pack nor kept")?,
        };
        Ok(Shader::packed(r.stage, form(r.form), r.hash, bytes))
    };

    // Translations.
    let failures: Mutex<HashMap<String, usize>> = Mutex::new(HashMap::new());
    let made: Mutex<HashMap<ShaderRef, Arc<Shader>>> = Mutex::new(HashMap::new());
    let p = Progress { what: "shaders", total: missing.len(), done: AtomicUsize::new(0), failed: AtomicUsize::new(0), started: Instant::now() };
    xnw_core::info!("prebuild: {} listed pipelines ({} with recipes), {} shaders, {} held, {} failed under this version, {} to translate on {} threads", out.listed, out.with_recipe, out.shaders, out.cached, out.failed_before, missing.len(), s.jobs);
    on_all_cores(&p, s.jobs, s.console, |i| {
        let r = &missing[i];
        let result = load(r).and_then(|shader| match shader.translation() {
            Ok(_) => Ok(shader),
            Err(e) => Err(e.clone()),
        });
        match result {
            Ok(shader) => {
                made.lock().unwrap_or_else(|e| e.into_inner()).insert(*r, shader);
                true
            }
            Err(e) => {
                *failures.lock().unwrap_or_else(|e| e.into_inner()).entry(reason(&e)).or_default() += 1;
                false
            }
        }
    });
    let made = made.into_inner().unwrap_or_else(|e| e.into_inner());
    out.translated = made.len();
    let mut failed: Vec<(String, usize)> = failures.into_inner().unwrap_or_else(|e| e.into_inner()).into_iter().collect();
    failed.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.failed = failed;
    xnw_core::info!("prebuild: {} shaders translated, {} failed ({:.1} s)", out.translated, out.failed_total(), p.started.elapsed().as_secs_f64());

    // Host pipelines over the new translations.
    if s.pipelines {
        if super::super::host::host_device().is_none() {
            out.note = "no host device: pipelines left to the run's prebuild".into();
        } else {
            // Pipelines an earlier run found too slow are made last, one
            // at a time.
            let slow: HashSet<String> = super::compile::slow_list().into_iter().map(|(k, _)| k).collect();
            let (late, todo): (Vec<&(String, Recipe)>, Vec<&(String, Recipe)>) = recipes.iter().partition(|(k, _)| slow.contains(k));
            out.pipelines = todo.len() + late.len();
            let (made0, loaded0) = (library::MADE.load(Ordering::Relaxed), library::LOADED.load(Ordering::Relaxed));
            let refused = AtomicUsize::new(0);
            let mismatch = AtomicU64::new(0);
            let build = |(key, r): &(String, Recipe)| -> bool {
                let at = Instant::now();
                let shaders: Result<Vec<Arc<Shader>>, String> = r.shaders.iter().map(|x| made.get(x).cloned().map_or_else(|| load(x), Ok)).collect();
                let Ok(shaders) = shaders else {
                    refused.fetch_add(1, Ordering::Relaxed);
                    return false;
                };
                if make::key_of(&shaders, r.graphics.as_ref()) != *key {
                    mismatch.fetch_add(1, Ordering::Relaxed);
                }
                // Watched as in a run (slow log; no deadline effect here);
                // A start and a finish line each, in the order's numbers.
                let label = make::label(&shaders);
                let n = pos.get(key).copied().unwrap_or(0);
                xnw_core::info!("prebuild: start #{n} {key} {label}");
                let ticket = super::compile::begin(key, label, Arc::new(AtomicBool::new(false)));
                let ok = HostPipeline::new(&shaders, r.graphics.as_ref()).is_some();
                ticket.done(if ok { "made offline" } else { "refused offline" });
                xnw_core::info!("prebuild: finish #{n} {key} {:.2} s {}", at.elapsed().as_secs_f64(), if ok { "ok" } else { "refused" });
                if !ok {
                    refused.fetch_add(1, Ordering::Relaxed);
                }
                ok
            };
            let p = Progress { what: "pipelines", total: todo.len(), done: AtomicUsize::new(0), failed: AtomicUsize::new(0), started: Instant::now() };
            on_all_cores(&p, s.jobs, s.console, |i| build(todo[i]));
            if !late.is_empty() {
                xnw_core::info!("prebuild: {} pipelines listed as too slow, one at a time", late.len());
                let q = Progress { what: "slow pipelines", total: late.len(), done: AtomicUsize::new(0), failed: AtomicUsize::new(0), started: Instant::now() };
                on_all_cores(&q, 1, s.console, |i| build(late[i]));
            }
            out.made = library::MADE.load(Ordering::Relaxed) - made0;
            out.loaded = library::LOADED.load(Ordering::Relaxed) - loaded0;
            out.refused = refused.load(Ordering::Relaxed);
            out.key_mismatch = mismatch.load(Ordering::Relaxed) as usize;
            // A lost device: what was in flight is a suspect; the library
            // is saved only if it still serialises (else the old file stays).
            let host = super::super::host::host_device();
            // SAFETY: a plain query.
            if let Some(Err(e)) = host.map(|h| unsafe { h.device.GetDeviceRemovedReason() }) {
                out.device_lost = true;
                super::compile::device_lost(&format!("host device removed ({:#010x})", e.code().0 as u32));
                xnw_core::warn!("prebuild: HOST DEVICE REMOVED during the pass ({:#010x}): no stamp; suspects in suspect.txt", e.code().0 as u32);
            }
            library::save(true);
            let secs = p.started.elapsed().as_secs_f64();
            out.pipeline_seconds = secs;
            xnw_core::info!("prebuild: {} pipelines: {} made, {} loaded, {} refused ({secs:.1} s, {:.1} made per second on {} threads)", out.pipelines, out.made, out.loaded, out.refused, out.made as f64 / secs.max(0.001), s.jobs);
            // The stamp says "done for this list": only after a whole
            // pass, no device loss, every stored pipeline on disk.
            let unsaved = library::unsaved();
            if !s.whole() || out.device_lost || unsaved > 0 || out.suspects_skipped > 0 {
                xnw_core::info!("prebuild: no stamp (whole pass {}, device lost {}, {unsaved} pipelines not saved, {} suspects skipped)", s.whole(), out.device_lost, out.suspects_skipped);
            } else if let Err(e) = std::fs::create_dir_all(stamp_file.parent().unwrap_or(dir)).and_then(|()| std::fs::write(&stamp_file, stamp())) {
                xnw_core::warn!("prebuild: stamp not written: {e}");
            }
        }
    } else {
        out.note = "translations only: pipelines left to the run's prebuild".into();
    }
    out.seconds = started.elapsed().as_secs_f64();
    Ok(out)
}

pub mod sweep {
    //! SWEEP1: the card sweep's view of the pipeline list (`xnw card-sweep`,
    //! apps/xnw `cardsweep.rs`): the listed pipelines, their programs read
    //! from the pack, and one host pipeline made from a recipe exactly as
    //! the prebuild makes it ([`HostPipeline::new`]), the driver compiling
    //! it here and now (the library bypassed).
    use super::*;

    /// One listed pipeline: key, first use (ms after the story load), recipe.
    pub struct Entry {
        pub key: String,
        pub ms: i64,
        pub recipe: Option<String>,
    }

    /// One program of a recipe.
    pub struct Program {
        pub stage: &'static str,
        /// The hash of the title's bytes (pipeline key, multiply-add list).
        pub hash: u128,
        /// The console container's hash, its pack name and bytes (None:
        /// neither in the pack nor kept).
        pub container_hash: u128,
        pub name: Option<String>,
        pub container: Option<Vec<u8>>,
    }

    /// The made pipeline: per program its translated DXIL (or why not),
    /// then the creation (Ok, or the refusal) and its milliseconds.
    pub struct Made {
        pub dxil: Vec<Result<Vec<u8>, String>>,
        pub create: Result<(), String>,
        pub create_ms: f64,
    }

    /// The sweep's setup: the pack (else `XNW_SHADER_PACK`/`XNW_GAME_DRIVE`),
    /// the cache folder, and the library bypassed from now on.
    pub fn begin(pack: Option<PathBuf>) {
        if let Some(p) = pack {
            let _ = crate::d3d12x::pipeline::PACK_PATH.set(p);
        }
        crate::d3d12x::pipeline::init_cache_dir();
        library::bypass();
    }

    /// The listed pipelines (`<cache>/pipelines/pipeline-list.txt`).
    pub fn list() -> Vec<Entry> {
        crate::d3d12x::pipeline::init_cache_dir();
        library::rows().into_iter().map(|(key, r)| Entry { key, ms: r.ms, recipe: r.recipe }).collect()
    }

    /// The recipe's programs: (stage, title hash, container hash) only.
    pub fn parse(recipe: &str) -> Option<Vec<(&'static str, u128, u128)>> {
        Some(parse_recipe(recipe)?.shaders.iter().map(|s| (s.stage, s.hash, s.container)).collect())
    }

    /// The recipe's programs with their pack names and bytes.
    pub fn programs(recipe: &str) -> Result<Vec<Program>, String> {
        let r = parse_recipe(recipe).ok_or("recipe does not parse")?;
        let pack = crate::d3d12x::pipeline::pack();
        let dir = shader_cache::dir();
        Ok(r.shaders
            .iter()
            .map(|s| {
                let place = pack.and_then(|p| p.by_hash(s.container));
                let container = place.and_then(|pl| pack?.container(pl)).or_else(|| dir.and_then(|d| shader_cache::kept_container(d, s.container)));
                Program { stage: s.stage, hash: s.hash, container_hash: s.container, name: place.map(|pl| pl.name.clone()), container }
            })
            .collect())
    }

    /// The options the runtime translates a `stage` program with title
    /// hash `hash` (`options_for`: the device's barycentrics included).
    pub fn options(stage: &str, hash: u128) -> xnw_shader::emit::Options {
        crate::d3d12x::pipeline::options_for(stage, hash)
    }

    /// Makes the recipe's host pipeline as the prebuild does.
    pub fn make(recipe: &str, programs: &[Program]) -> Result<Made, String> {
        let r = parse_recipe(recipe).ok_or("recipe does not parse")?;
        let form = |f: u8| if f == 0 { Form::Container } else { Form::CacheCode };
        let mut shaders = Vec::new();
        for (x, p) in r.shaders.iter().zip(programs) {
            let bytes = p.container.clone().ok_or("listed container neither in the pack nor kept")?;
            shaders.push(Shader::packed(x.stage, form(x.form), x.hash, bytes));
        }
        let dxil = shaders.iter().map(|s| s.translation().map(|t| t.host.dxil.clone()).map_err(String::clone)).collect();
        crate::d3d12x::pipeline::LAST_FAILURE.with(|l| l.borrow_mut().clear());
        let at = Instant::now();
        let ok = HostPipeline::new(&shaders, r.graphics.as_ref()).is_some();
        let create_ms = at.elapsed().as_secs_f64() * 1e3;
        let why = crate::d3d12x::pipeline::LAST_FAILURE.with(|l| l.borrow().clone());
        Ok(Made { dxil, create: if ok { Ok(()) } else { Err(if why.is_empty() { "refused (no reason noted)".into() } else { why }) }, create_ms })
    }
}

/// The launch prebuild (`xnw run --execute`, before the title starts):
/// `XNW_PREBUILD_LAUNCH=off` skips it, `=shaders` translates only; on by
/// default when more than 20 listed shaders are missing; all cores but one.
pub fn at_launch() {
    let mode = std::env::var("XNW_PREBUILD_LAUNCH").unwrap_or_default();
    if mode == "off" {
        xnw_core::info!("prebuild at launch: off (XNW_PREBUILD_LAUNCH=off)");
        return;
    }
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(1).max(1);
    let s = Settings::new(jobs, mode != "shaders", None, 20, true);
    match warm(&s) {
        Ok(o) if !o.needed => xnw_core::info!(
            "prebuild at launch: not needed ({} listed, {} shaders: {} held, {} failed under this version, {} missing; pipelines made for this list and translator)",
            o.listed,
            o.shaders,
            o.cached,
            o.failed_before,
            o.missing
        ),
        Ok(o) => xnw_core::info!("prebuild at launch: {}", summary(&o)),
        Err(e) => xnw_core::warn!("prebuild at launch: {e}"),
    }
}

/// The outcome as one line.
pub fn summary(o: &Outcome) -> String {
    let failed: Vec<String> = o.failed.iter().take(8).map(|(r, n)| format!("{n} x {r}")).collect();
    format!(
        "listed {} ({} with recipes); shaders {}: already cached {}, failed before {}, translated {}, failed {} [{}]; pipelines {}: made {}, loaded {}, refused {}, key mismatches {}{}; {:.1} s",
        o.listed,
        o.with_recipe,
        o.shaders,
        o.cached,
        o.failed_before,
        o.translated,
        o.failed_total(),
        failed.join("; "),
        o.pipelines,
        o.made,
        o.loaded,
        o.refused,
        o.key_mismatch,
        if o.note.is_empty() { String::new() } else { format!(" ({})", o.note) },
        o.seconds
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipes_round_trip_with_the_state() {
        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 2;
        g.targets.RTFormats[0].0 = 28;
        g.targets.RTFormats[1].0 = 10;
        g.depth_format.0 = 40;
        g.rasterizer.DepthBiasClamp = 0.25;
        g.blend.RenderTarget[3].RenderTargetWriteMask = 5;
        let a = Shader::packed("vertex", Form::CacheCode, 7, b"DXBC vertex".to_vec());
        let b = Shader::packed("pixel", Form::Container, 9, b"DXBC pixel".to_vec());
        let text = recipe_of(&[a.clone(), b.clone()], Some(&g)).expect("containers known");
        let r = parse_recipe(&text).expect("parses");
        assert_eq!(r.shaders.len(), 2);
        assert_eq!(r.shaders[0], ShaderRef { stage: "vertex", form: 1, hash: 7, container: xnw_core::hash128(&[b"DXBC vertex"]) });
        assert_eq!(r.graphics.as_ref().map(GraphicsState::describe), Some(g.describe()));
        // The pipeline key from the recipe is the key of the original.
        assert_eq!(make::key_of(&[a.clone(), b], r.graphics.as_ref()), make::key_of(&[a.clone(), Shader::packed("pixel", Form::Container, 9, Vec::new())], Some(&g)));
        // Compute: no state.
        let c = recipe_of(std::slice::from_ref(&a), None).unwrap();
        assert!(c.ends_with(";-"));
        assert!(parse_recipe(&c).unwrap().graphics.is_none());
        // Damage is refused, not misread.
        assert!(parse_recipe(&text[..text.len() - 1]).is_none());
        assert!(parse_recipe("vertex:2:1:1;-").is_none());
    }

    /// A measurement, not a check (ignored by default): a pipeline list made
    /// from a run's census (`XNW_CENSUS=<run>\census`; graphics pipelines
    /// get one target of format 28, so their keys are not the title's) in
    /// a scratch cache (`XNW_CACHE`, must not be `out\cache`), prebuilt cold
    /// and then again. Needs the pack (`XNW_SHADER_PACK`/`XNW_GAME_DRIVE`)
    /// and `XNW_AGILITY`. `XNW_PREBUILD_JOBS` sets the threads (default 6).
    #[test]
    #[ignore]
    fn census_prebuild_rate() {
        let (Some(census), Some(cache)) = (std::env::var_os("XNW_CENSUS"), std::env::var_os("XNW_CACHE")) else {
            return;
        };
        assert!(!PathBuf::from(&cache).ends_with(r"out\cache"), "a scratch cache, please");
        let pack = super::super::pack().expect("the pack");
        let mut groups: std::collections::BTreeMap<String, Vec<(&'static str, Vec<u8>)>> = Default::default();
        for e in std::fs::read_dir(&census).expect("census folder").flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(stage) = name.split('-').nth(1).and_then(|s| STAGES.iter().find(|&&n| n == s)) else { continue };
            if name.ends_with(".bin") && *stage != "fetch" {
                groups.entry(name[..2].to_owned()).or_default().push((stage, std::fs::read(e.path()).unwrap()));
            }
        }
        let mut text = String::from("# census list\n");
        for (i, files) in groups.values_mut().enumerate() {
            files.sort_by_key(|(s, _)| *s != "vertex" && *s != "mesh");
            let mut shaders = Vec::new();
            for (stage, bytes) in files.iter() {
                let form = if bytes.starts_with(b"DXBC") { Form::Container } else { Form::CacheCode };
                let hash = xnw_core::hash128(&[bytes]);
                let Some(container) = (if form == Form::Container { Some(hash) } else { pack.hash_of_code(bytes) }) else { continue };
                if form == Form::Container && pack.by_hash(hash).is_none() {
                    // As a run keeps it.
                    shader_cache::keep_container(shader_cache::dir().unwrap(), hash, bytes);
                }
                shaders.push((Shader::packed(stage, form, hash, Vec::new()), container));
            }
            let compute = shaders.iter().any(|(s, _)| s.stage == "compute");
            let mut g = GraphicsState::default();
            g.topology = windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            g.targets.NumRenderTargets = 1;
            g.targets.RTFormats[0].0 = 28;
            let g = (!compute).then_some(g);
            let arcs: Vec<Arc<Shader>> = shaders.iter().map(|(s, _)| s.clone()).collect();
            let parts: Vec<String> = shaders.iter().map(|(s, c)| format!("{}:{}:{:032x}:{c:032x}", s.stage, s.form as u8, s.hash)).collect();
            let state = g.as_ref().map_or_else(|| "-".to_owned(), |g| hex(&g.encode()));
            text.push_str(&format!("{}\t{i}\t{};{state}\n", make::key_of(&arcs, g.as_ref()), parts.join(",")));
        }
        let dir = PathBuf::from(&cache).join("pipelines");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("pipeline-list.txt"), text).unwrap();
        let jobs = std::env::var("XNW_PREBUILD_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        let s = Settings::new(jobs, true, None, 0, false);
        let cold = warm(&s).expect("prebuild");
        println!("cold: {}", summary(&cold));
        let warm_again = warm(&s).expect("prebuild");
        println!("again: {}", summary(&warm_again));
        assert_eq!(warm_again.missing, 0, "restartable: everything is held after a pass");
    }

    #[test]
    fn hex_round_trips() {
        assert_eq!(unhex(&hex(&[0, 1, 0xab, 0xff])), Some(vec![0, 1, 0xab, 0xff]));
        assert_eq!(unhex("abc"), None);
        assert_eq!(unhex("zz"), None);
        assert!(parse_recipe("x;-").is_none());
    }
}
