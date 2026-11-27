//! `xnw shader-reduce <pixel hash> --frame <capture dir> [--at N] [--row R]
//! [--also h+h] [--steps bind,out,code] [--only <switch>] [--stop]
//! [--secs S] [--out DIR] [--switches a,b=c]`: why the software adapter
//! fails on a bridged pixel program (KILL1). `--only bind=tex:0/code=kill+mask=1`
//! runs those steps instead of the list (`+` between steps, `/` between
//! the switches of one step); `--all-inputs` steps the mesh program's
//! table slots too; `--stop` ends at the first step that stops the failure;
//! `--go` runs the steps when the baseline passed (an intermittent failure);
//! `--trials N` runs a step that passes again, up to N times (a failure
//! ends it: a pass is weak evidence, a hang is not).
//!
//! Each step is this executable again as `xnw burst <dir> --adapter warp
//! --steady --from N --frames 1 --png ..` with `XNW_REDUCE` set
//! (`xnw_world::reduce`): the program's rows over our pass, one thing
//! changed. A child, because a removed device ends its process. The steps:
//! - reference: none of the program's rows (the picture the others compare
//!   with), twice: the second one's count is what two equal runs differ by,
//!   and a step is counted against the nearer of the two;
//! - baseline: its rows as they are; without `--row`, then each row alone
//!   until one fails by itself (that row is the one reduced);
//! - bind: every input the pixel program names, one at a time, replaced by
//!   its neutral stand-in (grey texture, zeroed buffer, default samplers);
//!   first all textures and all buffers at once;
//! - out: each of our three targets masked, the depth test off, the depth
//!   test passing always;
//! - code: the program translated without its discard, with every image
//!   read a constant, each image read alone, each colour export a constant.
//!
//! A step's verdict: `REMOVED` (the baseline's failure), `ok` with the
//! rows drawn and the pixels that differ from the reference, `hang` (no
//! exit in `--secs`, default 120; the child is ended), `no pipeline` (the
//! stubbed program did not translate or build), `error`. The summary names
//! the first change that stops the removal and every one that does.
//! Software adapter only: a program that fails there stays off the card.
//! `XNW_WARP`, `XNW_AGILITY`, `XNW_GAME_DRIVE` as for a burst.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The switches a bridged row's burst runs with (WIDE1's, batch159 minus `geomasync`, `ui`).
const SWITCHES: &str = "drawbones=on,skinrules=1eb,preskin=on,preskinbits=f,cutfast=on,texms=12,imposters=on,geomcopy=direct,gpucull=on,gbuf=on,ourlight=metal,ournormals=title";
const USAGE: &str = "usage: xnw shader-reduce <pixel hash> --frame <capture dir> [--at N] [--row R] [--also h+h] [--steps bind,out,code] [--only <switch>] [--stop] [--go] [--trials N] [--secs S] [--out DIR] [--switches a,b=c]";

#[derive(Debug, Clone, PartialEq)]
enum Verdict {
    Removed,
    /// Rows drawn, pixels unlike the reference (None: no reference picture).
    Ok(u32, Option<u64>),
    Hang,
    NoPipeline(String),
    Error(String),
}

impl Verdict {
    fn text(&self) -> String {
        match self {
            Verdict::Removed => "REMOVED".into(),
            Verdict::Ok(d, px) => format!("ok (rows drawn {d}, {} px unlike the reference)", px.map_or("?".into(), |p| p.to_string())),
            Verdict::Hang => "hang".into(),
            Verdict::NoPipeline(w) => format!("no pipeline ({w})"),
            Verdict::Error(w) => format!("error ({w})"),
        }
    }

    /// The step ran the row to the end: the change stops the failure.
    fn stops(&self) -> bool {
        matches!(self, Verdict::Ok(d, _) if *d > 0)
    }
}

/// What a finished child's log says (`code`: its exit code; None: ended by us).
fn verdict(log: &str, code: Option<i32>) -> Verdict {
    if log.contains("device removed") {
        return Verdict::Removed;
    }
    if let Some(l) = log.lines().find(|l| l.contains("its rows stay ours")) {
        return Verdict::NoPipeline(l.split("): ").nth(1).unwrap_or(l).trim().chars().take(120).collect());
    }
    match code {
        None => Verdict::Hang,
        Some(0) => {
            let drawn = log.lines().rev().find_map(|l| l.split("bridgeps: rows ").nth(1)?.split("(drawn ").nth(1)?.split(',').next()?.trim().parse().ok());
            Verdict::Ok(drawn.unwrap_or(0), None)
        }
        Some(c) => Verdict::Error(log.lines().rev().find(|l| l.starts_with("xnw: ")).map_or(format!("exit {c}"), |l| l.chars().take(160).collect())),
    }
}

/// `reduce: frame N: rows of H: [a, b]; kept [..]`: the program's rows.
fn rows_of(log: &str) -> Vec<u32> {
    let Some(l) = log.lines().find(|l| l.starts_with("reduce: frame ") && l.contains(": rows of ")) else { return Vec::new() };
    let list = l.split('[').nth(1).and_then(|x| x.split(']').next()).unwrap_or("");
    list.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

/// The inputs of the kept rows as `bind=` names them; `named`: only the
/// ones the pixel program names (`*`), textures, root views and samplers.
fn inputs_of(log: &str, named: bool) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for l in log.lines().filter(|l| l.starts_with("reduce: frame ") && l.contains(") inputs: ")) {
        for w in l.split(") inputs: ").nth(1).unwrap_or("").split(' ') {
            let name = w.split(['*', '(']).next().unwrap_or("");
            let starred = w[name.len()..].starts_with('*');
            if name.is_empty() || named && name.starts_with("slot:") && !starred {
                continue;
            }
            if !out.iter().any(|x| x == name) {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// `reduce: code H: kill image:1e0 export:0`: the program's code sites.
fn sites_of(log: &str) -> Vec<String> {
    log.lines().find(|l| l.starts_with("reduce: code ")).and_then(|l| l.split(": ").nth(2)).map(|s| s.split(' ').filter(|x| !x.is_empty()).map(str::to_string).collect()).unwrap_or_default()
}

/// The first `reduce: fault:` line, without its register dump; the
/// runtime's own throw of the removal (`_com_error` 0x887a0005) is its
/// report, not a fault before it.
fn fault_of(log: &str) -> Option<&str> {
    log.lines().find(|l| l.starts_with("reduce: fault: ") && !(l.contains("_com_error") && l.contains("0x887a0005"))).map(|l| l.split("; code ").next().unwrap_or(l))
}

/// The steps after the baseline: (group, the `XNW_REDUCE` switch).
fn steps(inputs: &[String], sites: &[String], groups: &[&str]) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    if groups.contains(&"bind") {
        out.push(("bind", "bind=textures".to_string()));
        out.push(("bind", "bind=buffers".to_string()));
        out.extend(inputs.iter().map(|i| ("bind", format!("bind={i}"))));
    }
    if groups.contains(&"out") {
        out.extend(["mask=0", "mask=1", "mask=2", "depth=off", "depth=always"].map(|s| ("out", s.to_string())));
    }
    if groups.contains(&"code") {
        if sites.iter().any(|s| s == "kill") {
            out.push(("code", "code=kill".to_string()));
        }
        if sites.iter().any(|s| s.starts_with("image:")) {
            out.push(("code", "code=images".to_string()));
        }
        if sites.iter().any(|s| s.starts_with("export:")) {
            out.push(("code", "code=exports".to_string()));
        }
        out.extend(sites.iter().filter(|s| s.starts_with("image:") || s.starts_with("export:")).map(|s| ("code", format!("code={s}"))));
    }
    out
}

/// Pixels of two RGBA pictures that differ (None: unlike sizes or unreadable).
fn unlike(a: &Path, b: &Path) -> Option<u64> {
    let (a, b) = (std::fs::read(a).ok()?, std::fs::read(b).ok()?);
    (a.len() == b.len()).then(|| a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(x, y)| x != y).count() as u64)
}

struct Runner {
    exe: PathBuf,
    dir: String,
    at: u64,
    switches: String,
    hash: String,
    out: PathBuf,
    secs: u64,
    /// The reference pictures (raw RGBA): a step's count is against the nearer one.
    reference: Vec<PathBuf>,
}

impl Runner {
    /// One burst with `XNW_REDUCE=<hash>,<switch>`; its log goes to `<out>/<name>.log`.
    fn run(&self, name: &str, switch: &str) -> Result<(Verdict, String, f64), String> {
        let t0 = Instant::now();
        let png = self.out.join(name);
        let _ = std::fs::remove_dir_all(&png);
        let log_path = self.out.join(format!("{name}.log"));
        let log = std::fs::File::create(&log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;
        let reduce = if switch.is_empty() { self.hash.clone() } else { format!("{},{switch}", self.hash) };
        let mut child = Command::new(&self.exe)
            .args(["burst", &self.dir, "--adapter", "warp", "--steady", "--from", &self.at.to_string(), "--frames", "1", "--switches", &self.switches, "--png"])
            .arg(&png)
            .env("XNW_REDUCE", &reduce)
            .env("XNW_REDUCE_DUMP", self.out.join("dxil").join(name))
            .env("XNW_ADAPTER", "warp")
            .stdin(Stdio::null())
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log)
            .spawn()
            .map_err(|e| format!("{}: {e}", self.exe.display()))?;
        let code = loop {
            match child.try_wait().map_err(|e| e.to_string())? {
                Some(s) => break s.code(),
                None if t0.elapsed() > Duration::from_secs(self.secs) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                None => std::thread::sleep(Duration::from_millis(50)),
            }
        };
        let text = String::from_utf8_lossy(&std::fs::read(&log_path).map_err(|e| e.to_string())?).into_owned();
        let mut v = verdict(&text, code);
        // The picture as raw RGBA (the burst writes it beside the PNG under `XNW_REDUCE`).
        let raw = png.join(format!("seq-{:05}.rgba", self.at));
        if let Verdict::Ok(d, _) = v {
            v = Verdict::Ok(d, self.reference.iter().filter_map(|r| unlike(r, &raw)).min());
        }
        if !name.starts_with("reference") {
            let _ = std::fs::remove_file(&raw);
        }
        Ok((v, text, t0.elapsed().as_secs_f64()))
    }
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let hash = args.first().filter(|a| a.len() >= 8 && u32::from_str_radix(&a[..8], 16).is_ok()).map(|a| a[..8].to_lowercase()).ok_or(USAGE)?;
    let dir = opt("--frame").ok_or(USAGE)?;
    let at: u64 = match opt("--at") {
        Some(v) => v.parse().map_err(|_| USAGE)?,
        None => std::fs::read_dir(&dir).map_err(|e| format!("{dir}: {e}"))?.flatten().filter_map(|e| e.file_name().to_str()?.strip_prefix("frame-")?.parse().ok()).min().ok_or(format!("{dir}: no frame-N"))?,
    };
    let out = PathBuf::from(opt("--out").unwrap_or(format!("out/reduce/{hash}")));
    std::fs::create_dir_all(&out)?;
    let programs = match opt("--also") {
        Some(a) => format!("{a}+{hash}"),
        None => hash.clone(),
    };
    let switches = format!("{},bridgeps={programs}", opt("--switches").unwrap_or(SWITCHES.to_string()));
    let groups_text = opt("--steps").unwrap_or("bind,out,code".to_string());
    let groups: Vec<&str> = groups_text.split(',').collect();
    let stop = args.iter().any(|a| a == "--stop");
    let mut r = Runner { exe: std::env::current_exe()?, dir: dir.clone(), at, switches, hash: hash.clone(), out: out.clone(), secs: opt("--secs").and_then(|v| v.parse().ok()).unwrap_or(120), reference: Vec::new() };
    println!("shader-reduce: {hash} on frame {at} of {dir}, software adapter; logs and pictures in {}", out.display());
    let t_all = Instant::now();
    let show = |name: &str, switch: &str, v: &Verdict, secs: f64| println!("  {name:<10} {:<24} {} [{secs:.1} s]", if switch.is_empty() { "-" } else { switch }, v.text());

    // The reference: none of the program's rows (a row number no list has).
    let (v, text, secs) = r.run("reference", "row=4294967295")?;
    show("reference", "none of its rows", &v, secs);
    let all_rows = rows_of(&text);
    if !matches!(v, Verdict::Ok(..)) {
        return Err(format!("the frame fails without the program's rows: {} ({})", v.text(), out.join("reference.log").display()).into());
    }
    if all_rows.is_empty() {
        return Err(format!("frame {at} has no row of {hash} (bridgeps reads none: {})", out.join("reference.log").display()).into());
    }
    r.reference.push(out.join("reference").join(format!("seq-{at:05}.rgba")));
    // The same run again: the pixels two equal runs differ by. A capture whose picture has two states
    // (pk7c f1400: 0 or about 7,000 px apart, run by run) is counted against the nearer reference.
    let (v, _, secs) = r.run("reference-again", "row=4294967295")?;
    show("reference", "again (its px: noise)", &v, secs);
    r.reference.push(out.join("reference-again").join(format!("seq-{at:05}.rgba")));

    // The baseline, and the row to reduce.
    let mut row: Option<u32> = opt("--row").and_then(|v| v.parse().ok());
    let base_switch = |row: Option<u32>| row.map_or(String::new(), |w| format!("row={w}"));
    let (mut base, mut base_text, secs) = r.run("baseline", &base_switch(row))?;
    show("baseline", &row.map_or(format!("its {} rows", all_rows.len()), |w| format!("row={w}")), &base, secs);
    if row.is_none() && !base.stops() {
        for (k, &w) in all_rows.iter().enumerate().take(opt("--rows").and_then(|v| v.parse().ok()).unwrap_or(12)) {
            let (v, text, secs) = r.run(&format!("row-{w}"), &format!("row={w}"))?;
            show("row", &format!("row={w}"), &v, secs);
            if !v.stops() {
                (row, base, base_text) = (Some(w), v, text);
                break;
            }
            if k + 1 == all_rows.len().min(12) {
                println!("  no row fails alone: the steps keep all {} rows", all_rows.len());
            }
        }
    }
    // `--go` (SKIN2): an intermittent failure's steps run even when this baseline passed.
    if base.stops() && !args.iter().any(|a| a == "--go") {
        println!("shader-reduce: the baseline does not fail ({}): nothing to reduce", base.text());
        return Ok(());
    }
    match fault_of(&base_text) {
        Some(f) => println!("  {f}"),
        None if base == Verdict::Removed => println!("  no fault in the process before the removal: the adapter ended the device by itself"),
        None => {}
    }
    let (inputs, sites) = (inputs_of(&base_text, !args.iter().any(|a| a == "--all-inputs")), sites_of(&base_text));
    println!("  inputs: {}", inputs.join(" "));
    println!("  code: {}", sites.join(" "));
    let mut list = steps(&inputs, &sites, &groups);
    if let Some(only) = opt("--only") {
        list = only.split('+').map(|s| ("only", s.replace('/', ","))).collect();
    }
    let mut stopping: Vec<String> = Vec::new();
    let mut other: Vec<String> = Vec::new();
    for (k, (group, switch)) in list.iter().enumerate() {
        let full = match row {
            Some(w) => format!("row={w},{switch}"),
            None => switch.clone(),
        };
        // `--trials N` (SKIN2): a step that passes is run again, up to N times; a failure ends it.
        let trials = opt("--trials").and_then(|v| v.parse().ok()).unwrap_or(1u32).max(1);
        let mut t = 0;
        let v = loop {
            let (v, _, secs) = r.run(&format!("{group}-{k:02}-{t}"), &full)?;
            show(group, &if trials > 1 { format!("{switch} #{}", t + 1) } else { switch.clone() }, &v, secs);
            t += 1;
            if !v.stops() || t == trials {
                break v;
            }
        };
        if v.stops() {
            stopping.push(switch.clone());
            if stop {
                break;
            }
        } else if v != base {
            other.push(format!("{switch}: {}", v.text()));
        }
    }
    println!("shader-reduce: {hash} row {}: baseline {}; {} steps in {:.0} s", row.map_or("all".into(), |w| w.to_string()), base.text(), list.len(), t_all.elapsed().as_secs_f64());
    match stopping.first() {
        Some(f) => println!("shader-reduce: first change that stops it: {f}; all that stop it: {}", stopping.join(" ")),
        None => println!("shader-reduce: no single change stops it"),
    }
    if !other.is_empty() {
        println!("shader-reduce: steps that end another way: {}", other.join("; "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = "reduce: code 070aaba8: kill image:1e0 image:41c export:0 export:1\n\
reduce: frame 1030: rows of 070aaba8: [12, 40]; kept [12]\n\
reduce: frame 1030: row 12 (groups 3, lacks 0x0) inputs: tex:0* tex:2* slot:0.5(Cbv,64B) slot:8.163*(texture) root:3(view,16B) samplers(13)\n\
world render: frame 1030: bridgeps: rows 1 (drawn 1, no pipelines 0, refused 0; draws not ready 0, unread 0)\n";

    #[test]
    fn a_log_gives_rows_inputs_sites_and_verdict() {
        assert_eq!(rows_of(LOG), [12, 40]);
        assert_eq!(inputs_of(LOG, true), ["tex:0", "tex:2", "slot:8.163", "root:3", "samplers"]);
        assert_eq!(inputs_of(LOG, false).len(), 6);
        assert_eq!(sites_of(LOG), ["kill", "image:1e0", "image:41c", "export:0", "export:1"]);
        assert_eq!(verdict(LOG, Some(0)), Verdict::Ok(1, None));
        assert!(verdict(LOG, Some(0)).stops());
        assert_eq!(verdict(&format!("{LOG}xnw: readback Map: device removed: reason 0x887a0005\n"), Some(1)), Verdict::Removed);
        assert_eq!(verdict(LOG, None), Verdict::Hang);
        assert!(matches!(verdict("WARN bridgeps: pixel program 070aaba8 (mesh 07d0a626): pixel program: refused: its rows stay ours\n", Some(0)), Verdict::NoPipeline(_)));
        assert!(!Verdict::Ok(0, Some(0)).stops(), "a row that was not drawn stops nothing");
        let thrown = "reduce: fault: a C++ throw of .?AV_com_error@@ (32 B: 0x33fc4be0 0x7ffb 0x887a0005 0x0) by code at 0x7ffbd75f5369 in KERNELBASE.dll+0x25369; code 0f\n";
        assert_eq!(fault_of(thrown), None, "the runtime reporting the removal");
        assert_eq!(fault_of(&format!("{thrown}reduce: fault: access violation reading 0x388 by code at 0x1 in no module (generated code); code 0f\n")), Some("reduce: fault: access violation reading 0x388 by code at 0x1 in no module (generated code)"));
    }

    #[test]
    fn steps_follow_the_inputs_and_sites() {
        let s = steps(&inputs_of(LOG, true), &sites_of(LOG), &["bind", "out", "code"]);
        let names: Vec<&str> = s.iter().map(|x| x.1.as_str()).collect();
        assert_eq!(names, ["bind=textures", "bind=buffers", "bind=tex:0", "bind=tex:2", "bind=slot:8.163", "bind=root:3", "bind=samplers", "mask=0", "mask=1", "mask=2", "depth=off", "depth=always", "code=kill", "code=images", "code=exports", "code=image:1e0", "code=image:41c", "code=export:0", "code=export:1"]);
        assert_eq!(steps(&[], &[], &["out"]).len(), 5);
    }
}
