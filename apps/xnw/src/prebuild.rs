//! `xnw prebuild [--jobs N] [--shaders-only] [--pack <shaders.rpf>] [--list-slow]
//!   [--only <key|hash prefix>,...] [--skip <key|hash prefix>,...|@file]
//!   [--from N] [--to M] [--timeout S] [--retry-suspects]`
//!
//! Without the title: reads the pipeline list (`out/cache/pipelines/
//! pipeline-list.txt`, or under `XNW_CACHE`), translates every listed
//! shader the cache does not hold for this build from the shader pack, and
//! makes the host pipelines over them into the host pipeline library, where
//! a run finds both. Restartable: what finished is on disk. The same code
//! runs at launch (`xnw run --execute`, `XNW_PREBUILD_LAUNCH`).
//!
//! a diagnostic tool: each pipeline logs a start and a finish line with
//! its number in the earliest-first order (`#n`) and its wall time;
//! `--only`/`--skip` pick by pipeline key or program hash prefix (`@file`:
//! one per line, first word; `#` comments), `--from`/`--to` by number;
//! `--timeout S` names a compile every 5 s once it runs S seconds ("STILL
//! COMPILING"); `pipelines/inflight.txt` holds the compiles running now, and
//! those running at a device loss are appended to `pipelines/suspect.txt`
//! and skipped from then on (here, in-run prebuild and in-run builds) until
//! `--retry-suspects` (`XNW_RETRY_SUSPECTS=1` in a run). A device loss
//! exits non-zero and writes no stamp.

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// `a,b,@file` as prefixes (lower case hex).
fn prefixes(list: Option<&String>) -> Res<Vec<String>> {
    let mut out = Vec::new();
    for item in list.map_or("", String::as_str).split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let text = match item.strip_prefix('@') {
            Some(file) => std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?,
            None => item.to_owned(),
        };
        for line in text.lines() {
            let word = line.split('#').next().unwrap_or("").split(['\t', ' ', '|']).next().unwrap_or("").trim().to_ascii_lowercase();
            if !word.is_empty() && word.bytes().all(|b| b.is_ascii_hexdigit()) {
                out.push(word);
            }
        }
    }
    Ok(out)
}

pub fn run(args: &[String]) -> Res<()> {
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1));
    let number = |flag: &str| -> Res<Option<usize>> { value(flag).map(|v| v.parse::<usize>().map_err(|_| format!("{flag} {v}: expected a number").into())).transpose() };
    let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
    let jobs = number("--jobs")?.map_or(logical.saturating_sub(2).max(1), |j| j.max(1));
    // The run's pack (`XNW_SHADER_PACK`, `XNW_GAME_DRIVE`) when set, so the
    // saved pack index stays the run's; else `--pack <file>`.
    let pack = value("--pack").map(std::path::PathBuf::from);
    if pack.is_none() && xnw_gpu::d3d12x::pack::default_pack().is_none() && !args.iter().any(|a| a == "--list-slow") {
        return Err("prebuild: name the shader pack (--pack <file>, XNW_SHADER_PACK, or XNW_GAME_DRIVE holding it)".into());
    }
    // `--list-slow`: the pipelines past the compile deadline in a run.
    if args.iter().any(|a| a == "--list-slow") {
        let slow = xnw_gpu::d3d12x::slow_pipelines();
        println!("{} pipelines listed as too slow to compile in-run", slow.len());
        for (key, programs) in slow {
            println!("  {key}  {programs}");
        }
        return Ok(());
    }
    let mut settings = xnw_gpu::d3d12x::PrebuildSettings::new(jobs, !args.iter().any(|a| a == "--shaders-only"), pack, 0, true);
    settings.only = prefixes(value("--only"))?;
    settings.skip = prefixes(value("--skip"))?;
    settings.from = number("--from")?;
    settings.to = number("--to")?;
    settings.timeout = number("--timeout")?.map(|s| std::time::Duration::from_secs(s as u64));
    settings.retry_suspects = args.iter().any(|a| a == "--retry-suspects");
    let o = xnw_gpu::d3d12x::prebuild(&settings)?;
    println!("listed       {} pipelines ({} with recipes; {} left out by filters, {} suspects skipped)", o.listed, o.with_recipe, o.filtered_out, o.suspects_skipped);
    println!("shaders      {} distinct: already cached {}, failed before (this version) {}, missing {}", o.shaders, o.cached, o.failed_before, o.missing);
    println!("translated   {}", o.translated);
    println!("failed       {}", o.failed_total());
    for (reason, n) in &o.failed {
        println!("  {n:>5}  {reason}");
    }
    println!("pipelines    {}: made {}, loaded {}, refused {}, key mismatches {}", o.pipelines, o.made, o.loaded, o.refused, o.key_mismatch);
    if !o.note.is_empty() {
        println!("note         {}", o.note);
    }
    println!("seconds      {:.1} on {jobs} threads", o.seconds);
    if o.device_lost {
        return Err("the host device was removed during the pass: suspects appended to out\\cache\\pipelines\\suspect.txt, no stamp written".into());
    }
    Ok(())
}
