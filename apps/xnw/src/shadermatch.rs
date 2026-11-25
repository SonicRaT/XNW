//! `xnw shader-match <rpf> <dir>`: finds the pack record of each shader a
//! run's census dumped: DXBC containers by
//! content hash, pipeline-cache code blocks by their first 24 bytes; then
//! translates the record (fast multiply-add) and reports the result.

use std::path::Path;

use xnw_gpu::d3d12x::pack::PackIndex;
use xnw_shader::emit::{Fma, Options};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn run(args: &[String]) -> Res<()> {
    let (Some(pack), Some(dir)) = (args.first(), args.get(1)) else {
        return Err("usage: xnw shader-match <rpf> <census dir>".into());
    };
    let start = std::time::Instant::now();
    let index = PackIndex::open(Path::new(pack), Some(Path::new("out/cache/pack-index.bin")))?;
    println!("index      {} containers in {:.2} s", index.len(), start.elapsed().as_secs_f64());
    let mut files: Vec<_> = std::fs::read_dir(dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "bin")).collect();
    files.sort();
    for f in files {
        let bytes = std::fs::read(&f)?;
        let name = f.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let dxbc = bytes.starts_with(b"DXBC");
        let place = if dxbc { index.by_hash(xnw_core::hash128(&[&bytes])) } else { index.by_code(&bytes) };
        let Some(place) = place else {
            let own = if dxbc {
                match xnw_shader::emit::translate_with(&bytes, Options { fma: Fma::Fast, ..Default::default() }) {
                    Ok(h) => format!("; its own bytes -> {:?} DXIL {} bytes, {} user dwords", h.stage, h.dxil.len(), h.user_data_dwords()),
                    Err(e) => format!("; its own bytes FAIL: {e}"),
                }
            } else {
                String::new()
            };
            println!("{name:<56} {} not in the pack{own}", if dxbc { "container" } else { "cache code" });
            continue;
        };
        let how = if dxbc { "same bytes" } else { "same 24-byte prefix" };
        let Some(container) = index.container(place) else {
            println!("{name:<56} {how}: {} (unreadable)", place.name);
            continue;
        };
        let t = std::time::Instant::now();
        let result = xnw_shader::emit::translate_with(&container, Options { fma: Fma::Fast, ..Default::default() });
        let ms = t.elapsed().as_secs_f64() * 1e3;
        match result {
            Ok(h) => println!("{name:<56} {how}: {} -> {:?} DXIL {} bytes, {} user dwords ({ms:.1} ms)", place.name, h.stage, h.dxil.len(), h.user_data_dwords()),
            Err(e) => println!("{name:<56} {how}: {} -> FAILS: {e}", place.name),
        }
    }
    Ok(())
}
