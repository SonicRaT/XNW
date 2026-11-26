//! `xnw`: command-line front end.
//!
//!   xnw inspect <exe|dll> [--imports]   headers, sections, import routing
//!   xnw gpu-check [--shared-memory | --health]     host device, shader model, wave size;
//!                                       GPU reads of process memory in place
//!   xnw run <exe> [--execute]           map, bind and (with --execute) run the title
//!   xnw smoke [--profile P] [--secs N]  the title for N s, null GPU, WARP, headless; pass/fail
//!   xnw shader-info <rpf> [--census] [--disasm <name>]  shader pack contents
//!   xnw shader-match <rpf> <dir>        pack records of a run's census shaders
//!   xnw pack-info <frame dir> [--pack <rpf>]  coverage of a frame pack v2
//!   xnw translate <rpf> [--stage ..] [--validate] [--pipelines N] [--fma exact|fast]  pack to DXIL
//!   xnw prebuild [--jobs N] [--shaders-only]  listed shaders and pipelines into the cache
//!   xnw shader-warm [--drawn|--all]          the shaders only, for the drawn list (default) or all listed
//!   xnw movie-info <movies.rpf> [--list] [--entry I --frame N --out f.bmp]  Bink movies, CPU decode
//!   xnw texture-info <ttpf|dir|rpf> [--census] [--list N] [--dump N] [--sample K]  texture page files
//!   xnw texture-pack <ttpf|dir|rpf> [--out DIR] [--drop-mips N] [--only-trace LOG] [--threads K]  host texture packs

mod ab;
mod bcpack;
mod gpucheck;
mod movie;
mod packinfo;
mod detours;
mod fencewait;
mod dirtyfilter;
mod patches;
mod prebuild;
mod run;
mod dwatch;
mod levers;
mod watch;
mod worldahead;
mod worldprobe;
mod worldterrain;
mod worldwater;
mod worldtex;
mod shaderinfo;
mod smoke;
mod stagemeter;
mod shadermatch;
mod textureinfo;
mod texturepack;
mod texturetrace;
mod titlemeter;
mod translate;

use std::collections::BTreeMap;
use std::process::ExitCode;

use xnw_hle::{Route, route};
use xnw_pe::Image;

const USAGE: &str = "usage: xnw inspect <exe|dll> [--imports]
       xnw gpu-check [--shared-memory | --health]
       xnw run <exe> [--execute] [--unimplemented trap|zero]
       xnw smoke [--profile <name>] [--secs N] [--record [FILE[,until=S]]] [--exe PATH]
       xnw shader-info <rpf> [--census] [--disasm <name-or-hash>]
       xnw shader-match <rpf> <census dir>
       xnw pack-info <frame dir> [--pack <rpf>]
       xnw burst <capture dir> --adapter hw|warp [--from N] [--frames K] [--switches a,b=c] [--png DIR] [--png-at N,..]
       xnw texture-info <ttpf|dir|rpf> [--census] [--list N] [--dump N] [--only HASH] [--sample K] [--out DIR]
       xnw texture-info --trace <dstorage log>
       xnw texture-pack <ttpf|dir|rpf> [--out DIR] [--drop-mips N] [--only-trace LOG] [--threads K] [--plan] [--verify N] [--dds N]
       xnw translate <rpf> [--stage vs|ps|cs|ms|gs] [--validate] [--pipelines N] [--fma exact|fast] [--jobs N] [--dump DIR]
       xnw shader-warm [--drawn|--all] [--jobs N] [--pack <shaders.rpf>]
       xnw prebuild [--jobs N] [--shaders-only] [--pack <shaders.rpf>] [--list-slow] [--only P,..] [--skip P,..|@file] [--from N] [--to M] [--timeout S] [--retry-suspects]";

fn main() -> ExitCode {
    // The audio recorder a run starts for `XNW_AUDIO_CAPTURE` is this
    // executable again.
    if xnw_hle::audio_capture_child() {
        return ExitCode::SUCCESS;
    }
    // Our world pass's picture on screen (`XNW_WORLD_RENDER=screen`) reaches the presenter through this.
    xnw_world::install_screen();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("inspect") if args.len() >= 2 => {
            inspect(&args[1], args.iter().any(|a| a == "--imports"))
        }
        Some("gpu-check") => gpucheck::run(&args[1..]),
        Some("run") => run::run(&args[1..]),
        Some("smoke") => smoke::run(&args[1..]),
        Some("shader-info") => shaderinfo::run(&args[1..]),
        Some("translate") => translate::run(&args[1..]),
        Some("prebuild") => prebuild::run(&args[1..]),
        Some("shader-warm") => prebuild::warm(&args[1..]),
        Some("shader-match") => shadermatch::run(&args[1..]),
        Some("pack-info") => packinfo::run(&args[1..]),
        Some("movie-info") => movie::run(&args[1..]),
        Some("bcpack-check") => bcpack::run(&args[1..]),
        Some("texture-info") => textureinfo::run(&args[1..]),
        Some("texture-pack") => texturepack::run(&args[1..]),
        Some("world-probe-decode") => xnw_frame::decode_cli(&args[1..]).map_err(Into::into),
        Some("world-render") => xnw_frame::render_cli(&args[1..]).map_err(Into::into),
        Some("burst") => worldprobe::burst::run(&args[1..]).map_err(Into::into),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xnw: {e}");
            ExitCode::FAILURE
        }
    }
}

fn inspect(path: &str, list_imports: bool) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let img = Image::parse(&data)?;

    println!("image      {path}");
    println!("base       {:#x}  size {:#x}  entry rva {:#x}", img.image_base, img.size_of_image, img.entry_rva);
    println!("timestamp  {:#x}  subsystem {}  dllchars {:#x}", img.timestamp, img.subsystem, img.dll_characteristics);
    println!("functions  {} (unwind entries)", img.function_count());
    if let Some(pdb) = img.pdb_path()? {
        println!("pdb        {pdb}");
    }
    if let Some(tls) = img.tls()? {
        println!(
            "tls        template {:#x} bytes + {:#x} zero, callbacks at rva {:#x}",
            tls.raw_end_va - tls.raw_start_va,
            tls.zero_fill,
            img.va_to_rva(tls.callbacks_va)
        );
    }
    println!("\nsections");
    for s in &img.sections {
        println!("  {:<9} rva {:#10x}  vsize {:#10x}  raw {:#10x}  flags {:#010x}", s.name, s.rva, s.virtual_size, s.file_size, s.flags);
    }

    let modules = img.imports()?;
    let mut totals: BTreeMap<Route, (usize, usize)> = BTreeMap::new();
    println!("\nimports ({} modules)", modules.len());
    for m in &modules {
        let r = route(&m.dll);
        let t = totals.entry(r).or_default();
        t.0 += 1;
        t.1 += m.imports.len();
        println!("  {:<12} {:>4}  {}{}", r.label(), m.imports.len(), m.dll, if m.delay { "  (delay)" } else { "" });
        if list_imports {
            for i in &m.imports {
                println!("      {:#10x}  {}", i.slot_rva, i.symbol);
            }
        }
    }
    println!("\nrouting");
    for (r, (mods, funcs)) in &totals {
        println!("  {:<12} {:>3} modules  {:>4} functions", r.label(), mods, funcs);
    }
    let exports = img.exports()?;
    if !exports.is_empty() {
        println!("\nexports ({})", exports.len());
        for e in exports.iter().take(16) {
            println!("  #{:<4} {:#10x}  {}", e.ordinal, e.rva, e.name.as_deref().unwrap_or("-"));
        }
    }
    Ok(())
}
