//! Import conformance from a recording of the title's own calls
//! (`XNW_IMPORT_TRACE`, `xnw smoke --record`; README.md beside this file).
//!
//! 1. Replay: every distinct pure call (`xnw_core::imprec::PURE`) is issued
//!    again against what the binder binds (our shim if we have one, else
//!    the host's function) with the recorded arguments, input bytes and
//!    last error; result, last error after and the bytes written must be
//!    the recording's. Where we bind a shim over a host function, the host
//!    function gets the same call and the shim must answer as it does.
//! 2. Invariants over every record: `GetLastError` returns the last error
//!    it found; a shim's `BOOL` is 0 or 1; a shim's success leaves the last
//!    error as it was (unless the call documents otherwise); a shim never
//!    leaves our own environment lookups' error (203) behind.
//!
//! The recording is game-derived: without one the
//! test says so and passes.

use std::path::PathBuf;

use xnw_core::imprec::{Arg, PURE, Record, Ret};
use xnw_hle::Route;

fn recording() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("XNW_IMPORT_RECORDING") {
        return Some(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut candidates = vec![root.join("out/imports.tsv")];
    // A worktree: the main checkout's out/ too.
    if let Ok(git) = std::fs::read_to_string(root.join(".git"))
        && let Some(dir) = git.trim().strip_prefix("gitdir:")
        && let Some(main) = std::path::Path::new(dir.trim()).ancestors().nth(3)
    {
        candidates.push(main.join("out/imports.tsv"));
    }
    candidates.into_iter().find(|p| p.is_file())
}

fn host_export(dll: &str, symbol: &str) -> Option<usize> {
    let d = format!("{dll}\0");
    let s = format!("{symbol}\0");
    // SAFETY: NUL-terminated names of a system DLL and its export.
    unsafe {
        let m = xnw_core::win::LoadLibraryA(d.as_ptr());
        if m.is_null() {
            return None;
        }
        let p = xnw_core::win::GetProcAddress(m, s.as_ptr()) as usize;
        (p != 0).then_some(p)
    }
}

/// What the binder binds for `dll!symbol` (run.rs `resolve`): `(our shim,
/// the host's function)`.
fn resolve(dll: &str, symbol: &str) -> (Option<usize>, Option<usize>) {
    let name = dll.to_ascii_lowercase();
    match xnw_hle::route(&name) {
        Route::Hle => (xnw_hle::lookup(&name, symbol), None),
        Route::Host | Route::HostFiltered => (xnw_hle::host_override(&name, symbol), host_export(dll, symbol)),
        _ => (None, None),
    }
}

/// One call's answer: result, last error after, the bytes of each buffer
/// argument after.
#[derive(Debug, PartialEq)]
struct Answer {
    ret: u64,
    le: u32,
    out: Vec<(usize, Vec<u8>)>,
}

fn last_error() -> u32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32
}

/// Issues `r` (a pure call) to `f`; `None` if it cannot be rebuilt.
fn issue(f: usize, r: &Record, spec: &xnw_core::imprec::Pure, fill: u8) -> Option<Answer> {
    if r.args.len() != spec.args.len() {
        return None;
    }
    let mut bufs: Vec<Vec<u8>> = vec![Vec::new(); spec.args.len()];
    let mut args = [0u64; 8];
    for (i, a) in spec.args.iter().enumerate() {
        let recorded = r.args[i];
        if *a == Arg::S {
            args[i] = recorded?;
            continue;
        }
        if recorded == Some(0) {
            continue;
        }
        let n = match *a {
            Arg::Out(n) | Arg::Scratch(n) => n,
            Arg::OutR { len, mult } => (r.args[len as usize]? as u32 as usize * mult as usize).min(1 << 20),
            Arg::OutLenPtr { len } => r.input(len as usize).map(|b| u32::from_le_bytes(b.try_into().unwrap_or([0; 4])) as usize)?,
            Arg::InOut4 | Arg::InA(_) | Arg::InW(_) => {
                bufs[i] = r.input(i)?.to_vec();
                bufs[i].push(0);
                bufs[i].push(0);
                0
            }
            Arg::S => 0,
        };
        if bufs[i].is_empty() {
            bufs[i] = vec![fill; n.max(1)];
        }
    }
    for (i, b) in bufs.iter_mut().enumerate() {
        if !b.is_empty() {
            args[i] = b.as_mut_ptr() as u64;
        }
    }
    type F = extern "system" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64;
    // SAFETY: `f` is the export the title calls with these arguments (extra
    // ones are ignored by the x64 calling convention); every pointer is a
    // buffer of the size the call is told.
    let (ret, le) = unsafe {
        xnw_core::win::SetLastError(r.le_before);
        let ret = std::mem::transmute::<usize, F>(f)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
        (ret, last_error())
    };
    let out = r.bytes.iter().filter(|(input, _, _)| !input).map(|(_, i, b)| (*i, bufs[*i].get(..b.len()).unwrap_or(&bufs[*i]).to_vec())).collect();
    Some(Answer { ret, le, out })
}

/// The call issued twice, its buffers filled with 0x00 and 0xFF first:
/// the answer, and which output bytes the call wrote (the same in both).
fn answer(f: usize, r: &Record, spec: &xnw_core::imprec::Pure) -> Option<(Answer, Vec<Vec<bool>>)> {
    let a = issue(f, r, spec, 0)?;
    let b = issue(f, r, spec, 0xFF)?;
    let written = a.out.iter().zip(&b.out).map(|((_, x), (_, y))| x.iter().zip(y).map(|(p, q)| p == q).collect()).collect();
    Some((a, written))
}

/// Equal results, last errors and written bytes.
fn same(spec: &xnw_core::imprec::Pure, want: &Answer, got: &Answer, written: &[Vec<bool>]) -> bool {
    let out = want.out.len() == got.out.len()
        && want.out.iter().zip(&got.out).zip(written).all(|(((i, w), (j, g)), m)| i == j && w.len() == g.len() && w.iter().zip(g).zip(m).all(|((a, b), &m)| !m || a == b));
    let ret = match spec.ret {
        Ret::Exact => want.ret == got.ret,
        Ret::Bool => want.ret as u32 == got.ret as u32,
        Ret::Volatile | Ret::Void => true,
    };
    ret && want.le == got.le && out
}

#[test]
fn imports_answer_as_recorded() {
    let Some(path) = recording() else {
        eprintln!("import_replay: skipped: no recording (out/imports.tsv or XNW_IMPORT_RECORDING); make one with `xnw smoke --record`");
        return;
    };
    let started = std::time::Instant::now();
    let text = std::fs::read_to_string(&path).unwrap();
    let records: Vec<Record> = text.lines().filter_map(Record::parse).collect();
    let calls: u64 = text.lines().filter_map(|l| l.strip_prefix("C\t")).filter_map(|l| l.split('\t').nth(2)?.parse::<u64>().ok()).sum();
    assert!(!records.is_empty(), "{}: no records", path.display());
    let mut bad = Vec::new();
    let (mut replayed, mut against_host, mut unresolved) = (0, 0, 0);
    let mut done = std::collections::HashSet::new();
    for r in &records {
        let Some((_, spec)) = xnw_core::imprec::pure(&r.symbol) else { continue };
        let key = (r.symbol.clone(), r.args.clone(), r.bytes.iter().filter(|b| b.0).cloned().collect::<Vec<_>>(), r.le_before);
        if !done.insert(key) {
            continue;
        }
        let (shim, host) = resolve(&r.dll, &r.symbol);
        let Some(f) = shim.or(host) else {
            unresolved += 1;
            continue;
        };
        let want = Answer { ret: r.ret, le: r.le_after, out: r.bytes.iter().filter(|b| !b.0).map(|(_, i, b)| (*i, b.clone())).collect() };
        let Some((got, written)) = answer(f, r, spec) else { continue };
        replayed += 1;
        if !same(spec, &want, &got, &written) {
            bad.push(format!("replay {}!{} from {} ({}): recorded {want:?}, now {got:?}", r.dll, r.symbol, r.caller, if shim.is_some() { "our shim" } else { "host" }));
        }
        if let (Some(_), Some(h)) = (shim, host)
            && let Some((theirs, theirs_written)) = answer(h, r, spec)
        {
            against_host += 1;
            let both: Vec<Vec<bool>> = written.iter().zip(&theirs_written).map(|(a, b)| a.iter().zip(b).map(|(x, y)| *x && *y).collect()).collect();
            if !same(spec, &theirs, &got, &both) {
                bad.push(format!("shim != host for {}!{} from {}: host {theirs:?}, shim {got:?}", r.dll, r.symbol, r.caller));
            }
        }
    }
    // Invariants over every record.
    let mut checked = 0;
    for r in &records {
        let name = r.symbol.as_str();
        let what = format!("{}!{} from {} ({})", r.dll, name, r.caller, if r.provider == 'H' { "our shim" } else { "host" });
        if name == "GetLastError" {
            checked += 1;
            if r.ret as u32 != r.le_before {
                bad.push(format!("{what}: returned {} with last error {}", r.ret as u32, r.le_before));
            }
        }
        if r.provider != 'H' {
            continue;
        }
        let is_bool = xnw_core::imprec::BOOLS.contains(&name) || PURE.iter().any(|p| p.name == name && p.ret == Ret::Bool);
        if is_bool {
            checked += 1;
            if r.ret as u32 > 1 {
                bad.push(format!("{what}: BOOL result {:#x}", r.ret as u32));
            }
            if r.ret as u32 == 1 && r.le_after != r.le_before && !xnw_core::imprec::SETS_ERROR_ON_SUCCESS.contains(&name) {
                bad.push(format!("{what}: succeeded but changed last error {} -> {}", r.le_before, r.le_after));
            }
        }
        if r.le_after == 203 && r.le_before != 203 && !name.contains("Environment") {
            checked += 1;
            bad.push(format!("{what}: left last error 203 (ERROR_ENVVAR_NOT_FOUND: an env lookup of ours after the host call)"));
        }
    }
    let shims = records.iter().filter(|r| r.provider == 'H').map(|r| &r.symbol).collect::<std::collections::HashSet<_>>().len();
    println!(
        "import_replay: {}: {} records ({calls} calls), {shims} shimmed imports; replayed {replayed} pure calls ({against_host} also on the host), {unresolved} unresolved; {checked} invariant checks; {} problems; {:.2} s",
        path.display(),
        records.len(),
        bad.len(),
        started.elapsed().as_secs_f64()
    );
    assert!(bad.is_empty(), "{} problems:\n{}", bad.len(), bad.join("\n"));
}

/// The invariants and the replay rebuild on a hand-made record (no
/// recording needed): GetSystemInfo through the host answers as itself.
#[test]
fn a_pure_call_rebuilds_and_matches_itself() {
    let (_, spec) = xnw_core::imprec::pure("GetNativeSystemInfo").unwrap();
    let f = host_export("kernel32.dll", "GetNativeSystemInfo").unwrap();
    let r = Record { dll: "kernel32.dll".into(), symbol: "GetNativeSystemInfo".into(), provider: 'W', caller: "-".into(), args: vec![None], ret: 0, le_before: 0, le_after: 0, bytes: vec![(false, 0, vec![0; 48])] };
    let (a, written) = answer(f, &r, spec).unwrap();
    let b = issue(f, &r, spec, 0).unwrap();
    assert_eq!(a, b);
    assert!(written[0].iter().all(|&w| w), "GetNativeSystemInfo writes all 48 bytes");
    assert_eq!(a.out[0].1.len(), 48);
    assert_ne!(a.out[0].1, vec![0; 48], "the call wrote its structure");
}
