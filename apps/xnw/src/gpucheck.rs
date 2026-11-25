//! `xnw gpu-check`: can this machine's GPU run what the translated shaders
//! need? Creates the host device (with the Agility SDK when found) and runs
//! small compute shaders, compiled here with the Windows SDK's `dxc.exe`.
//! DXC is a tool of this check only; the runtime never uses it.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use xnw_gpu::Host;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const PROBE: &str = r#"
RWStructuredBuffer<uint> o : register(u0);
#if WAVE64
[WaveSize(64)]
#endif
[numthreads(64, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint lanes = WaveGetLaneCount();
    uint active = WaveActiveCountBits(true);
    o[8 + t.x] = WaveGetLaneIndex();
    if (t.x == 0) {
        o[0] = lanes;
        o[1] = active;
        // Run-time values the compiler cannot fold: 2^127 * 2 - 2^127.
        uint zero = lanes & 0x100u;
        float a = asfloat(0x7F000000u + zero);
        float b = 2.0 + zero;
        float c = -a;
        o[2] = asuint(mad(a, b, c));
        precise float split = a * b;
        o[3] = asuint(split + c);
#if DOUBLES
        o[4] = asuint((float)fma((double)a, (double)b, (double)c));
#endif
    }
}
"#;

/// Newest `...\<name>` folder under `base` that contains `file`.
fn newest_with(base: &Path, file: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(base).ok()?.flatten().map(|e| e.path()).collect();
    dirs.sort();
    dirs.into_iter().rev().find(|d| d.join(file).is_file())
}

pub fn agility_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("XNW_AGILITY") {
        return Some(PathBuf::from(d)).filter(|d| !d.as_os_str().is_empty());
    }
    let exe = std::env::current_exe().ok()?;
    let mut bases = vec![PathBuf::from("deps/agility")];
    // target\<profile>\xnw.exe -> workspace root
    bases.extend(exe.ancestors().nth(3).map(|r| r.join("deps/agility")));
    bases.iter().find_map(|b| newest_with(b, "D3D12Core.dll")).and_then(|d| d.canonicalize().ok()).map(|d| {
        // CreateDeviceFactory takes an ANSI path; drop the \\?\ prefix.
        PathBuf::from(d.to_string_lossy().trim_start_matches(r"\\?\").to_string())
    })
}

fn dxc() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("XNW_DXC") {
        return Some(PathBuf::from(d));
    }
    let kits = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(kits).ok()?.flatten().map(|e| e.path().join("x64")).collect();
    dirs.sort();
    dirs.into_iter().rev().map(|d| d.join("dxc.exe")).find(|p| p.is_file())
}

fn compile(dxc: &Path, dir: &Path, name: &str, defines: &[&str]) -> Res<Vec<u8>> {
    compile_src(dxc, dir, name, PROBE, "cs_6_6", defines)
}

fn compile_src(dxc: &Path, dir: &Path, name: &str, source: &str, profile: &str, defines: &[&str]) -> Res<Vec<u8>> {
    let src = dir.join(format!("{name}.hlsl"));
    std::fs::write(&src, source)?;
    let out = dir.join(format!("{name}.dxil"));
    let mut cmd = Command::new(dxc);
    cmd.args(["-nologo", "-T", profile, "-E", "main", "-Fo"]).arg(&out);
    for d in defines {
        cmd.arg("-D").arg(d);
    }
    let run = cmd.arg(&src).output()?;
    if !run.status.success() {
        return Err(format!("dxc failed for {name}: {}", String::from_utf8_lossy(&run.stderr)).into());
    }
    Ok(std::fs::read(out)?)
}

fn fused(bits: u32) -> &'static str {
    match bits {
        0x7F00_0000 => "fused (one rounding, as the console)",
        0x7F80_0000 => "NOT fused (+inf)",
        _ => "unexpected",
    }
}

/// `--shared-memory`: can the GPU read title memory in place (`xnw_gpu::shared`)?
fn shared_memory(agility: Option<&Path>, dxc: &Path, dir: &Path) -> Res<()> {
    let src = xnw_gpu::shared::SHADERS;
    let shaders = xnw_gpu::shared::Shaders {
        cs: compile_src(dxc, dir, "shared_cs", src, "cs_6_6", &[])?,
        vs: compile_src(dxc, dir, "shared_vs", src, "vs_6_6", &["VS=1"])?,
        tex: compile_src(dxc, dir, "shared_tex", src, "cs_6_6", &["TEX=1"])?,
    };
    Ok(xnw_gpu::shared::run(agility, &shaders)?)
}

/// `--health`: is the GPU and its driver in a state to run the title?
/// 200 fence-waited submissions of a fixed small workload (`Host::health`).
/// Thresholds: on a healthy RX 5700 XT the workload is a few tenths of a
/// millisecond; a 5 ms median is ten times that, and 250 ms is where
/// `d3d12x` logs a slow host submission. A degraded driver state
/// (after a reset) showed 416 to 774 ms on 1-2-draw frames where 7.6 ms
/// was normal: both thresholds catch it.
fn health(host: &Host) -> Res<()> {
    const N: u32 = 200;
    const MEDIAN_MS: f64 = 5.0;
    const WORST_MS: f64 = 250.0;
    let mut t = match host.health(N) {
        Ok(t) => t,
        Err(e) => {
            println!("health: FAILED: {e}");
            println!("verdict: DEGRADED (the workload did not complete; reboot before a title run)");
            return Err(e.into());
        }
    };
    t.sort_by(f64::total_cmp);
    let (median, worst, p95) = (t[t.len() / 2], t[t.len() - 1], t[t.len() * 95 / 100]);
    println!("health: {N} submissions (512x512 clear, 4 MiB UAV clear, 4 MiB copy, each fence-waited): median {median:.3} ms, p95 {p95:.3} ms, worst {worst:.3} ms");
    if median < MEDIAN_MS && worst < WORST_MS {
        println!("verdict: HEALTHY (median under {MEDIAN_MS} ms, worst under {WORST_MS} ms)");
        Ok(())
    } else {
        println!("verdict: DEGRADED (median {median:.1} ms against {MEDIAN_MS}, worst {worst:.1} ms against {WORST_MS}): reboot before a title run");
        Err("GPU degraded".into())
    }
}

pub fn run(args: &[String]) -> Res<()> {
    let shared = args.iter().any(|a| a == "--shared-memory");
    // `--adapter warp` is `XNW_ADAPTER=warp` for this process.
    if args.windows(2).any(|w| w[0] == "--adapter" && w[1].eq_ignore_ascii_case("warp")) {
        // SAFETY: at the start of the command, before any other thread.
        unsafe { std::env::set_var("XNW_ADAPTER", "warp") };
    }
    let healthy = args.iter().any(|a| a == "--health");
    let agility = agility_dir();
    match &agility {
        Some(d) => println!("agility    {}", d.display()),
        None => println!("agility    not found (inbox D3D12)"),
    }
    let host = Host::new(agility.as_deref())?;
    let c = &host.caps;
    println!("adapter    {} ({:.1} GiB)", c.adapter, c.vram_bytes as f64 / (1u64 << 30) as f64);
    println!("runtime    {}", c.agility.map_or("inbox".to_string(), |v| format!("Agility SDK {v}")));
    if let Some(m) = &c.warp_module {
        println!("warp       {m}");
    }
    println!("shader model {}.{}   wave lanes {}..{}", c.shader_model.0, c.shader_model.1, c.wave_lanes.0, c.wave_lanes.1);
    println!("mesh shaders {}   ray tracing {}   double ops {}", c.mesh_shaders, c.ray_tracing, c.double_ops);
    println!(
        "software {}   feature level {:#x}   resource binding tier {}   tiled resources tier {}   int64 atomics typed {} heap {}   enhanced barriers {}",
        c.software, c.feature_level, c.binding_tier, c.tiled_tier, c.int64_typed, c.int64_heap, c.enhanced_barriers
    );
    if healthy {
        return health(&host);
    }

    let mut verdict = Vec::new();
    verdict.push(("shader model 6.6", c.shader_model >= (6, 6)));
    if c.shader_model < (6, 6) {
        if shared {
            match host.probe_open_heap() {
                Ok(s) => println!("shared memory (no shaders): {s}"),
                Err(e) => println!("shared memory (no shaders): FAIL: {e}"),
            }
        }
        println!("\nFAIL: shader model 6.6 is required; the compute probes were not run");
        return Err("no shader model 6.6".into());
    }
    let Some(dxc) = dxc() else {
        println!("\ndxc.exe not found (set XNW_DXC); compute probes skipped");
        return Ok(());
    };
    let dir = std::env::temp_dir().join("xnw-gpu-check");
    std::fs::create_dir_all(&dir)?;
    let doubles = if c.double_ops { "DOUBLES=1" } else { "DOUBLES=0" };

    for (label, wave64) in [("default wave size", false), ("WaveSize(64)", true)] {
        let name = if wave64 { "wave64" } else { "wavedef" };
        let dxil = compile(&dxc, &dir, name, &[if wave64 { "WAVE64=1" } else { "WAVE64=0" }, doubles])?;
        println!("\nprobe: {label} ({} bytes of DXIL)", dxil.len());
        match host.run_compute(&dxil, 8 + 64) {
            Ok(o) => {
                let distinct: BTreeSet<u32> = o[8..72].iter().copied().collect();
                println!("  lane count {}   active lanes {}   distinct lane indices over 64 threads {}", o[0], o[1], distinct.len());
                println!("  mad(a,b,c)        {:#010x}  {}", o[2], fused(o[2]));
                println!("  precise a*b + c   {:#010x}  {}", o[3], fused(o[3]));
                if c.double_ops {
                    println!("  fma via double    {:#010x}  {}", o[4], fused(o[4]));
                }
                if wave64 {
                    verdict.push(("WaveSize(64) gives one 64-lane wave", o[0] == 64 && o[1] == 64 && distinct.len() == 64));
                    verdict.push(("a fused multiply-add is available", o[2] == 0x7F00_0000 || o[4] == 0x7F00_0000));
                }
            }
            Err(e) => {
                println!("  failed: {e}");
                if wave64 {
                    verdict.push(("WaveSize(64) gives one 64-lane wave", false));
                }
            }
        }
    }
    verdict.push(("mesh shaders (not required: compute fallback)", c.mesh_shaders));
    println!();
    for (what, ok) in &verdict {
        println!("{}  {what}", if *ok { "PASS" } else { "FAIL" });
    }
    if shared {
        drop(host);
        shared_memory(agility.as_deref(), &dxc, &dir)?;
    }
    Ok(())
}
