//! `xnw run <exe> [--execute] [--unimplemented trap|zero]`
//!
//! Maps the title and the DLLs it ships, binds every import and reports who
//! provides what. With `--execute` it then runs the title's entry point.
//! `XNW_TRACE` is a comma-separated list of substrings of `dll!symbol`
//! (lower case; `*` for all): calls to matching imports are logged.
//! `XNW_PATCH` names title patches to apply (`patches.rs`).
//! `XNW_CONSOLE` is the console the title sees: `series-s` (default) or
//! `series-x`.
//!
//! A title DLL the title asks for by name at run time (`LoadLibrary`) is
//! mapped, bound and attached here too ([`load_late`]), with the title DLLs
//! it needs first; after VIctory `apps/victory/src/run.rs` (`LateDlls`,
//! `LoaderLock`) and `crates/victory-loader/src/windows.rs`
//! (`load_title_dll`) at d032b8a, GPL-3.0-or-later.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread::ThreadId;

use xnw_loader::{Loaded, Provider, Unimplemented};
use xnw_pe::Symbol;

use xnw_hle::{Route, route};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Maps the title DLL `name` from `game` with the title DLLs it imports
/// (dependencies first, `name` last), unless `have` already has it or no
/// such file is there. Runs none of their code.
fn map_dll(name: &str, game: &Path, cache: &Path, have: &[&'static Loaded], out: &mut Vec<&'static Loaded>) -> Res<()> {
    let name = name.to_ascii_lowercase();
    let file = game.join(&name);
    let known = |l: &&Loaded| l.name.eq_ignore_ascii_case(&name);
    if route(&name) != Route::Title || have.iter().any(known) || out.iter().any(known) || !file.is_file() {
        return Ok(());
    }
    let dll: &'static Loaded = Box::leak(Box::new(xnw_loader::load(&file, cache)?));
    for m in &dll.imports {
        map_dll(&m.dll, game, cache, have, out)?;
    }
    out.push(dll);
    Ok(())
}

/// The import resolver every image is bound with: our HLE first, the host
/// by routing (through our overrides), then the exports of the title DLLs
/// in `dlls`; `None` gets a naming stub.
fn resolve(dlls: &[&'static Loaded], dll: &str, symbol: &Symbol) -> Option<(usize, Provider)> {
    let name = dll.to_ascii_lowercase();
    match route(&name) {
        Route::Hle => xnw_hle::lookup(&name, &symbol_key(symbol)).map(|a| (a, Provider::Hle)),
        Route::Unknown => None,
        Route::Title => dlls.iter().find(|l| l.name.eq_ignore_ascii_case(&name))?.export(symbol).map(|a| (a, Provider::Title)),
        Route::Host | Route::HostFiltered => match xnw_hle::host_override(&name, &symbol_key(symbol)) {
            Some(a) => Some((a, Provider::Hle)),
            None => xnw_loader::host_export(dll, symbol)
                .or_else(|| name.starts_with("ext-ms-win-rtcore-ntuser-").then(|| xnw_loader::host_export("user32.dll", symbol)).flatten())
                .map(|a| (a, Provider::Host)),
        },
    }
}

/// Binds `image` against `dlls` and describes the result (report lines).
fn bind(image: &Loaded, dlls: &[&'static Loaded], filters: &[String]) -> Res<String> {
    let trace = |dll: &str, symbol: &str| {
        let full = format!("{dll}!{symbol}").to_ascii_lowercase();
        filters.iter().any(|f| f == "*" || full.contains(f.as_str()))
    };
    let bound = image.bind(&|dll, symbol| resolve(dlls, dll, symbol), &trace)?;
    let mut by_dll: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (dll, symbol) in &bound.missing {
        by_dll.entry(dll).or_default().push(symbol);
    }
    let mut report = String::new();
    writeln!(report, "{}: {} hle, {} host, {} title-dll, {} missing", image.name, bound.hle, bound.host, bound.title, bound.missing.len())?;
    for (dll, symbols) in by_dll {
        writeln!(report, "  missing {dll} ({}): {}", symbols.len(), symbols.join(" "))?;
    }
    Ok(report)
}

/// The Windows loader lock's shape: one thread at a time, re-entered by
/// the thread that holds it (a `DllMain` that loads another DLL).
#[derive(Default)]
struct LoaderLock {
    owner: Mutex<(Option<ThreadId>, usize)>,
    free: Condvar,
}

impl LoaderLock {
    fn hold<T>(&self, f: impl FnOnce() -> T) -> T {
        let me = std::thread::current().id();
        let lock = || self.owner.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut owner = lock();
            while owner.0.is_some_and(|o| o != me) {
                owner = self.free.wait(owner).unwrap_or_else(|e| e.into_inner());
            }
            *owner = (Some(me), owner.1 + 1);
        }
        let result = f();
        let mut owner = lock();
        owner.1 -= 1;
        if owner.1 == 0 {
            owner.0 = None;
            self.free.notify_one();
        }
        result
    }
}

/// What mapping title DLLs needs, for the ones the title loads itself.
struct Late {
    game: PathBuf,
    cache: PathBuf,
    filters: Vec<String>,
    /// Every title DLL mapped so far (initialised or being initialised),
    /// to bind new ones against.
    dlls: Mutex<Vec<&'static Loaded>>,
    lock: LoaderLock,
}

static LATE: OnceLock<Late> = OnceLock::new();

/// `LoadLibrary` of a title DLL nothing mapped yet (the FidelityFX loader
/// asking for its upscaler). Under the loader lock, as Windows does it: a
/// thread asking while another initialises waits; a `DllMain` that loads
/// another DLL re-enters; a DLL asked for again is returned as it is. The
/// DLL and the title DLLs it needs are mapped, bound, and attached on the
/// calling thread, dependencies first; only once all are initialised do
/// other threads find them by name. If a `DllMain` fails, as `LoadLibrary`
/// does: every DLL this call attached hears `DLL_PROCESS_DETACH` (the
/// failed one too) and all of them are unmapped.
fn load_late(file: &str) -> xnw_hle::TitleLoad {
    use xnw_hle::TitleLoad;
    let Some(late) = LATE.get() else { return TitleLoad::NoFile };
    late.lock.hold(|| {
        let list = || late.dlls.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(dll) = list().iter().find(|l| l.name.eq_ignore_ascii_case(file)) {
            return TitleLoad::Loaded(dll.base());
        }
        let mut new = Vec::new();
        let all = {
            let mut dlls = list();
            if let Err(e) = map_dll(file, &late.game, &late.cache, &dlls, &mut new) {
                xnw_core::warn!("LoadLibrary(\"{file}\"): mapping failed: {e}");
                return TitleLoad::Failed(format!("mapping failed: {e}"));
            }
            if new.is_empty() {
                return TitleLoad::NoFile;
            }
            // Visible before their `DllMain`s run, which may load more.
            dlls.extend(new.iter().copied());
            dlls.clone()
        };
        for dll in &new {
            match bind(dll, &all, &late.filters) {
                Ok(report) => {
                    for line in report.lines() {
                        xnw_core::info!("late {line}");
                    }
                }
                Err(e) => return TitleLoad::Failed(format!("binding {}: {e}", dll.name)),
            }
        }
        for (i, dll) in new.iter().enumerate() {
            xnw_core::info!("attaching {} (loaded by the title)", dll.name);
            // SAFETY: bound above; dependencies come first in `new`.
            let Err(e) = (unsafe { dll.attach() }) else { continue };
            xnw_core::warn!("LoadLibrary(\"{file}\"): {e}; detaching and unmapping {} DLL(s)", new.len());
            list().retain(|l| !new.iter().any(|n| n.base() == l.base()));
            for done in new[..=i].iter().rev() {
                // SAFETY: attached (or failing to) on this thread just now.
                unsafe { done.detach() };
            }
            for dll in &new {
                // SAFETY: detached; no title code holds a handle to it yet.
                unsafe { dll.unload() };
            }
            return TitleLoad::Failed(e.to_string());
        }
        for dll in &new {
            xnw_hle::add_image(&dll.name, dll.base(), dll.size);
        }
        let last = new[new.len() - 1];
        xnw_core::info!("{} mapped, bound and attached at {:#x}", last.name, last.base());
        TitleLoad::Loaded(last.base())
    })
}

/// A symbol as our tables name it: its name, or `#ordinal`.
fn symbol_key(symbol: &Symbol) -> String {
    match symbol {
        Symbol::Name(n) => n.clone(),
        Symbol::Ordinal(o) => format!("#{o}"),
    }
}

/// One title on the machine at a time: instances share 8 GB of video
/// memory, so a second one waits here until the first has gone. The mutex
/// is held until the process ends.
fn wait_for_other_instance() {
    use xnw_core::win;
    let name: Vec<u16> = "Global/XNW.title".replace('/', "\\").encode_utf16().chain(Some(0)).collect();
    // SAFETY: plain Win32 calls with a valid, terminated name.
    unsafe {
        let mutex = win::CreateMutexW(std::ptr::null(), 0, name.as_ptr());
        if mutex.is_null() {
            return;
        }
        if win::WaitForSingleObject(mutex, 0) == 0x102 {
            xnw_core::info!("another title instance is running; waiting for it to end");
            win::WaitForSingleObject(mutex, u32::MAX);
        }
    }
}

pub fn run(args: &[String]) -> Res<()> {
    let exe = PathBuf::from(args.first().ok_or("usage: xnw run <exe> [--execute] [--unimplemented trap|zero]")?);
    let execute = args.iter().any(|a| a == "--execute");
    let policy = match args.iter().position(|a| a == "--unimplemented").and_then(|i| args.get(i + 1)).map(String::as_str) {
        Some("zero") => Unimplemented::Zero,
        _ => Unimplemented::Trap,
    };
    let out = PathBuf::from("out");
    let run_dir = xnw_core::new_run_dir(&out)?;
    xnw_core::init_log(Some(&run_dir));
    xnw_core::info!("run folder {}", run_dir.display());
    if execute {
        wait_for_other_instance();
    }
    xnw_loader::crash::install();
    xnw_loader::crash::report_exceptions();
    xnw_loader::crash::on_report(xnw_gpu::d3d12x::log_memory);
    xnw_loader::crash::on_fault(xnw_gpu::d3d12x::descriptor::slotwatch::on_fault);
    // Every thread's stack once when no frame is presented for 10 s
    // (with the frame statistics, or XNW_STALL_DUMP=1).
    if std::env::var("XNW_FRAME_STATS").is_ok_and(|v| v == "1") || std::env::var("XNW_STALL_DUMP").is_ok_and(|v| v == "1") {
        xnw_loader::crash::watch_stalls(xnw_gpu::d3d12x::frames_presented, xnw_gpu::d3d12x::lock_owners);
    }
    xnw_loader::set_policy(policy);

    let cache = out.join("cache").join("images");
    let game = exe.parent().unwrap_or(Path::new(".")).to_path_buf();
    let title: &'static Loaded = Box::leak(Box::new(xnw_loader::load(&exe, &cache)?));
    let mut dlls = Vec::new();
    for m in &title.imports {
        map_dll(&m.dll, &game, &cache, &[], &mut dlls)?;
    }
    xnw_hle::set_title(title.base());
    // The console the title sees: `XNW_CONSOLE` = series-s (default) | series-x.
    let console = match std::env::var("XNW_CONSOLE") {
        Ok(v) => xnw_hle::Console::parse(&v).ok_or_else(|| format!("XNW_CONSOLE={v}: expected series-s or series-x"))?,
        Err(_) => xnw_hle::Console::default(),
    };
    xnw_hle::set_console(console);
    xnw_core::info!("console: {} (XNW_CONSOLE)", console.name());
    xnw_hle::set_game_dir(&game);
    // The title's package drive: the exe's folder first (local copies on a
    // fast disk), then each folder in `XNW_GAME_DRIVE` (`;`-separated).
    let mut drive = vec![game.clone()];
    if let Some(more) = std::env::var_os("XNW_GAME_DRIVE") {
        drive.extend(std::env::split_paths(&more).filter(|p| *p != game));
    }
    xnw_hle::set_game_drive(&drive);
    xnw_hle::set_overlay(&out.join("data"));
    let exe_file = exe.file_name().unwrap_or_default().to_string_lossy();
    xnw_hle::add_image(&exe_file, title.base(), title.size);
    for dll in &dlls {
        xnw_hle::add_image(&dll.name, dll.base(), dll.size);
    }
    xnw_hle::set_diagnostics(
        xnw_hle::Diagnostics { locate: xnw_loader::crash::locate, log_stack: xnw_loader::crash::log_stack_here },
        policy == Unimplemented::Trap,
    );
    xnw_hle::set_thread_calls(xnw_loader::disable_thread_calls);

    // `perf` names the runtime's counters (`xnw_core::perf`), not imports.
    let filters: Vec<String> = std::env::var("XNW_TRACE").unwrap_or_default().split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty() && s != "perf").collect();
    xnw_core::perf::on();
    let mut report = String::new();
    for image in dlls.iter().copied().chain([title]) {
        report.push_str(&bind(image, &dlls, &filters)?);
    }
    std::fs::write(run_dir.join("report.txt"), &report)?;
    print!("{report}");

    if xnw_loader::rewrite::list_only() {
        // SAFETY: the title is mapped; `list` changes nothing.
        unsafe { xnw_loader::rewrite::apply(title.base(), title.size) };
        println!("XNW_CPU_REWRITE=list: the plan is in the log; nothing was run");
        return Ok(());
    }
    if !execute {
        println!("mapped and bound; nothing was run (add --execute to run the title)");
        return Ok(());
    }
    xnw_hle::use_console_timers();
    // Every switch of the thread, wait and sampler shims read now,
    // before the title runs (an env lookup inside a shim set last error 203
    // over the host's 122).
    xnw_hle::read_shim_switches();
    // XNW_HEADLESS read before any title call.
    xnw_hle::read_window_switches();
    // Listed shaders (and their host pipelines) a translator change left
    // out of the cache, made before any title code runs.
    xnw_gpu::d3d12x::prebuild_at_launch();
    // SAFETY: the title is mapped and none of its code has run.
    unsafe { crate::patches::apply(title.base()) };
    // The Zen 2 instructions this CPU lacks, rewritten before any title code runs.
    // SAFETY: as above.
    unsafe { xnw_loader::rewrite::apply(title.base(), title.size) };
    // SAFETY: as above.
    unsafe { crate::detours::install(title.base()) };
    // SAFETY: as above (`XNW_TITLE_TIME`: the main loop's call sites; off when unset).
    unsafe { crate::titlemeter::install(title.base()) };
    // SAFETY: as above (`XNW_LOD_SCALE`: the LOD scale getter's call sites; off when unset).
    unsafe { crate::levers::lod_install(title.base()) };
    // SAFETY: as above (`XNW_ENCODE_TIME`, `XNW_PATCH=noencode`: the geometry draws' inner call sites; off when unset).
    unsafe { crate::encode::install(title.base()) };
    // SAFETY: as above.
    unsafe { crate::watch::start(title.base()) };
    // SAFETY: as above (`XNW_DWATCH`, a DR0 write watch; off when unset).
    unsafe { crate::dwatch::start(title.base()) };
    // SAFETY: as above.
    unsafe { crate::worldprobe::start(title.base()) };
    // SW1: the live switches' start values, the key map, the control file's poll.
    crate::live::start();
    // SAFETY: as above.
    unsafe { crate::dirtyfilter::start(title.base()) };
    let late = LATE.get_or_init(|| Late { game: game.clone(), cache: cache.clone(), filters, dlls: Mutex::new(dlls.clone()), lock: LoaderLock::default() });
    xnw_hle::set_title_loader(load_late);
    // Under the loader lock, as the DLLs the title loads later.
    late.lock.hold(|| -> Res<()> {
        for dll in &dlls {
            xnw_core::info!("attaching {}", dll.name);
            // SAFETY: imports are bound and dependencies come first in `dlls`.
            unsafe { dll.attach()? };
        }
        Ok(())
    })?;
    // SAFETY: everything is bound and attached; this runs the title.
    let code = unsafe { title.run()? };
    xnw_core::info!("title entry point returned {code}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    #[test]
    fn loader_lock_is_reentrant_and_exclusive() {
        let lock = Arc::new(super::LoaderLock::default());
        assert_eq!(lock.hold(|| lock.hold(|| 7)), 7);
        let order = Arc::new(Mutex::new(Vec::new()));
        let other = lock.hold(|| {
            let (theirs, their_order) = (lock.clone(), order.clone());
            let other = std::thread::spawn(move || theirs.hold(|| their_order.lock().unwrap().push(2)));
            std::thread::sleep(std::time::Duration::from_millis(50));
            order.lock().unwrap().push(1);
            other
        });
        other.join().unwrap();
        assert_eq!(*order.lock().unwrap(), [1, 2]);
    }
}
