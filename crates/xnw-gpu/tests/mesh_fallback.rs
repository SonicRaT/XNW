//! Proof: a mesh program run as compute plus a vertex shader draws the
//! right picture on the host. Needs the Agility SDK
//! (`XNW_AGILITY`, for shader model 6.6) and DXC (Windows SDK, or `XNW_DXC`);
//! prints SKIPPED and passes without them.
//!
//! The reference picture is analytic: every quad covers whole pixels, so
//! colour and depth must match exactly; alpha (interpolated) within 1/255.
//! Timings print with `cargo test -p xnw-gpu --test mesh_fallback -- --nocapture`.

use std::path::{Path, PathBuf};
use std::process::Command;

use xnw_gpu::Host;
use xnw_gpu::mesh::{DrawMode, Frame, Layout, MeshFallback};

const SRC: &str = include_str!("mesh_fallback.hlsl");

/// A stand-in for the title's culling pass: writes the group counts of
/// an indirect mesh dispatch on the GPU.
const CULL: &str = r#"
cbuffer Cull : register(b0) { uint groups; };
RWByteAddressBuffer args : register(u0);
[RootSignature("RootConstants(num32BitConstants=1, b0), UAV(u0)")]
[numthreads(1, 1, 1)]
void cull_main() { args.Store3(0, uint3(groups, 1, 1)); }
"#;
const RT: u32 = 1024;

/// Position plus `2 + extra` float4 attributes per vertex.
fn layout(extra: u32) -> Layout {
    Layout { vertex_stride: 48 + 16 * extra, max_vertices: 128, max_prims: 128 }
}

/// Builds the program and draw for one configuration.
fn build(host: &Host, dxc: &Path, dir: &Path, mode: DrawMode, extra: u32, ignore_counts: bool) -> MeshFallback {
    let defines = [
        format!("INDEXED={}", u8::from(mode == DrawMode::Indexed)),
        format!("EXTRA={extra}"),
        format!("IGNORE_COUNTS={}", u8::from(ignore_counts)),
    ];
    let cs = compile(dxc, dir, SRC, "cs_6_6", "cs_main", &defines);
    let vs = compile(dxc, dir, SRC, "vs_6_6", "vs_main", &defines);
    let ps = compile(dxc, dir, SRC, "ps_6_6", "ps_main", &defines);
    MeshFallback::new(host, &cs, &vs, &ps, mode, layout(extra), 4096, RT, RT).expect("pipelines")
}

fn dxc() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("XNW_DXC") {
        return Some(PathBuf::from(d)).filter(|p| p.is_file());
    }
    let kits = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(kits).ok()?.flatten().map(|e| e.path().join("x64")).collect();
    dirs.sort();
    dirs.into_iter().rev().map(|d| d.join("dxc.exe")).find(|p| p.is_file())
}

fn compile(dxc: &Path, dir: &Path, source: &str, profile: &str, entry: &str, defines: &[String]) -> Vec<u8> {
    let src = dir.join(format!("{entry}.hlsl"));
    std::fs::write(&src, source).unwrap();
    let out = dir.join(format!("{entry}.dxil"));
    let mut cmd = Command::new(dxc);
    cmd.args(["-nologo", "-O3", "-T", profile, "-E", entry]);
    for d in defines {
        cmd.arg("-D").arg(d);
    }
    let run = cmd
        .arg("-Fo")
        .arg(&out)
        .arg(&src)
        .output()
        .unwrap();
    assert!(run.status.success(), "dxc {entry}: {}", String::from_utf8_lossy(&run.stderr));
    std::fs::read(out).unwrap()
}

/// Program constants `c[4..12]`, as in `mesh_fallback.hlsl`.
#[derive(Clone, Copy)]
struct Scene {
    groups: u32,
    grid_w: u32,
    tile: u32,
    /// 0: every group has `nv`/`np`; 1: `np = 2 * (g % 65)`, `nv = 81` or 0.
    count_mode: u32,
    np: u32,
    nv: u32,
    /// 0: none; 1: every primitive; 2: quads `q % 3 == 1` of groups `g % 7 == 3`.
    cull_mode: u32,
}

impl Scene {
    fn consts(&self) -> [u32; 8] {
        [self.grid_w, self.tile, self.count_mode, self.np, self.nv, self.cull_mode, RT, RT]
    }

    fn counts(&self, g: u32) -> (u32, u32) {
        if self.count_mode == 0 {
            return (self.nv, self.np);
        }
        let np = 2 * (g % 65);
        (if np > 0 { 81 } else { 0 }, np)
    }

    /// The analytic picture: colour (alpha in the top byte) and depth.
    fn expected(&self) -> (Vec<u32>, Vec<f32>) {
        let n = (RT * RT) as usize;
        let (mut color, mut depth) = (vec![0u32; n], vec![1.0f32; n]);
        let cell = self.tile / 8;
        for g in 0..self.groups {
            let (_, np) = self.counts(g);
            let (tx, ty) = ((g % self.grid_w) * self.tile, (g / self.grid_w) * self.tile);
            for q in 0..64 {
                // np is even in every scene: whole quads only.
                let culled = self.cull_mode == 1 || (self.cull_mode == 2 && g % 7 == 3 && q % 3 == 1);
                if 2 * q + 1 >= np || culled {
                    continue;
                }
                let (x0, y0) = (tx + (q % 8) * cell, ty + (q / 8) * cell);
                let rgb = (q * 4) | ((g & 255) << 8) | ((((g >> 8) & 15) * 16 + 15) << 16);
                for y in y0..y0 + cell {
                    for x in x0..x0 + cell {
                        let u = (x as f32 + 0.5 - tx as f32) / self.tile as f32;
                        let v = (y as f32 + 0.5 - ty as f32) / self.tile as f32;
                        let a = (0.5 * (u + v) * 255.0).round() as u32;
                        let i = (y * RT + x) as usize;
                        color[i] = rgb | (a << 24);
                        depth[i] = 0.5;
                    }
                }
            }
        }
        (color, depth)
    }
}

/// Compares a frame with the analytic picture; returns the mismatch count
/// and prints the first few.
fn compare(name: &str, scene: &Scene, f: &Frame, words: usize) -> usize {
    let (color, depth) = scene.expected();
    let mut bad = 0;
    for i in 0..color.len() {
        let (got, want) = (f.color[i], color[i]);
        let alpha_off = (got >> 24).abs_diff(want >> 24);
        if (got & 0xFF_FFFF) != (want & 0xFF_FFFF) || alpha_off > 1 || f.depth[i] != depth[i] {
            if bad < 5 {
                let (x, y) = (i as u32 % RT, i as u32 / RT);
                println!("  {name}: pixel ({x},{y}) got {got:#010x} depth {} want {want:#010x} depth {}", f.depth[i], depth[i]);
            }
            bad += 1;
        }
    }
    for g in 0..scene.groups {
        let (nv, np) = scene.counts(g);
        let h = &f.records[g as usize * words..g as usize * words + 4];
        if h != [nv, np, 64, 0x4D53_4631 ^ g] {
            if bad < 5 {
                println!("  {name}: group {g} header {h:?} want [{nv}, {np}, 64, ..]");
            }
            bad += 1;
        }
    }
    let covered = color.iter().filter(|&&c| c != 0).count();
    println!("{} {name}: {bad} mismatches over {} pixels ({covered} covered) and {} records", if bad == 0 { "PASS" } else { "FAIL" }, color.len(), scene.groups);
    bad
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
fn mesh_fallback() {
    let agility = std::env::var_os("XNW_AGILITY").map(PathBuf::from).filter(|d| !d.as_os_str().is_empty());
    let host = match Host::new(agility.as_deref()) {
        Ok(h) if h.caps.shader_model >= (6, 6) => h,
        Ok(h) => {
            println!("SKIPPED: shader model {:?} < 6.6 (set XNW_AGILITY to the Agility SDK folder)", h.caps.shader_model);
            return;
        }
        Err(e) => {
            println!("SKIPPED: no D3D12 device ({e})");
            return;
        }
    };
    let Some(dxc) = dxc() else {
        println!("SKIPPED: dxc.exe not found (set XNW_DXC)");
        return;
    };
    println!("adapter {}, mesh shaders {}", host.caps.adapter, host.caps.mesh_shaders);
    let dir = std::env::temp_dir().join(format!("xnw-mesh-fallback-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let one = |count_mode, nv, np, cull_mode| Scene { groups: 1, grid_w: 1, tile: 64, count_mode, nv, np, cull_mode };
    let many = |count_mode, cull_mode| Scene { groups: 4096, grid_w: 64, tile: 16, count_mode, nv: 128, np: 128, cull_mode };
    let scenes = [
        ("full group (128 v, 128 t)", one(0, 128, 128, 0)),
        ("partial group (81 v, 50 t)", one(0, 81, 50, 0)),
        ("group that culls everything", one(0, 128, 128, 1)),
        ("4096 groups, counts 0..128, some culled", many(1, 2)),
        ("4096 full groups", many(0, 0)),
    ];

    let mut failures = 0;
    for (mode, extra) in [(DrawMode::Indexed, 0), (DrawMode::VertexId, 0), (DrawMode::Indexed, 16), (DrawMode::VertexId, 16)] {
        let fb = build(&host, &dxc, &dir, mode, extra, false);
        let l = fb.layout;
        let index = if mode == DrawMode::Indexed { l.index_bytes() } else { 0 };
        println!(
            "
mode {mode:?}, {} attributes: record {} bytes per group ({} header + 128 x {} vertex + 128 x 4 primitive) + {index} index bytes",
            2 + extra,
            l.record_bytes(),
            Layout::HEADER_BYTES,
            l.vertex_stride,
        );
        let words = (l.record_bytes() / 4) as usize;
        for (name, scene) in &scenes {
            let frame = fb.frame(&host, scene.groups, &scene.consts()).expect("frame");
            failures += compare(name, scene, &frame, words);
        }
        let timed = many(0, 0);
        for groups in [1, 256, 4096] {
            let mut best = (f64::MAX, f64::MAX);
            let mut line = String::new();
            for _ in 0..3 {
                let t = fb.time(&host, groups, &timed.consts(), 50).expect("timing");
                let cs = median(t.iter().map(|t| t.compute_us).collect());
                let draw = median(t.iter().map(|t| t.draw_us).collect());
                line += &format!(" {cs:.1}+{draw:.1}");
                if cs + draw < best.0 + best.1 {
                    best = (cs, draw);
                }
            }
            println!(
                "  time {groups:>4} groups: program {:8.1} us  draw {:8.1} us  per group {:.4} us   (best of 3 medians of 50; all:{line})",
                best.0,
                best.1,
                (best.0 + best.1) / f64::from(groups)
            );
        }
    }

    // The indirect form. A culling-like pass writes the group count on
    // the GPU; the writer clamps it into the segment and two
    // ExecuteIndirect run the program and the draw.
    {
        let fb = build(&host, &dxc, &dir, DrawMode::Indexed, 0, false);
        let cull = compile(&dxc, &dir, CULL, "cs_6_0", "cull_main", &[]);
        let words = (fb.layout.record_bytes() / 4) as usize;
        let scene = |groups| Scene { groups, ..many(0, 2) };
        // (count written, capacity, ring bytes, groups drawn, writer
        // counters: run, clamped, empty, groups, without ring room). A group
        // takes 6,672 record and 1,536 index bytes.
        const RING: u64 = 128 << 20;
        for (written, capacity, ring, drawn, want) in [
            (92, 4096, RING, 92, [1, 0, 0, 92, 0]),
            (5000, 300, RING, 300, [1, 1, 0, 300, 0]),
            (0, 4096, RING, 0, [1, 0, 1, 0, 0]),
            // A ring of 100 groups clamps a count of 150 to what fits.
            (150, 4096, 100 * 8208, 99, [1, 1, 0, 99, 0]),
        ] {
            let (frame, stats) = fb.frame_indirect(&host, &cull, written, capacity, ring, &scene(written).consts()).expect("indirect frame");
            let name = format!("indirect: {written} groups written, capacity {capacity}, ring {ring} bytes");
            failures += compare(&name, &scene(drawn), &frame, words);
            println!("  writer counters {stats:?} (dispatches, clamped, empty, groups, without room), want {want:?}");
            if stats != want {
                failures += 1;
            }
        }
    }

    // Negative control: a draw that ignores the primitive count must fail
    // the partial-group picture (the program wrote all 128 primitives).
    for mode in [DrawMode::Indexed, DrawMode::VertexId] {
        let fb = build(&host, &dxc, &dir, mode, 0, true);
        let (name, scene) = &scenes[1];
        let frame = fb.frame(&host, scene.groups, &scene.consts()).expect("frame");
        println!("control {mode:?}, counts ignored (must FAIL):");
        let bad = compare(name, scene, &frame, (fb.layout.record_bytes() / 4) as usize);
        if bad == 0 {
            failures += 1;
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(failures, 0, "picture comparisons failed");
}
