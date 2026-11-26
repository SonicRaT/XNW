//! a frame pack's UI draws (Scaleform, Bink) through the title's own
//! translated vertex and pixel programs into a UI layer, composed over the
//! replay's picture (`XNW_WORLD_RENDER=...,ui=on`, `world-render`;
//! `xnw-lab ui <frame dirs...>` lists the rows). The device side is
//! `xnw_world::ui`.
//!
//! A row is a Draw or DrawIndexed sink whose pixel program is of the UI
//! class (uikeep.rs's names: Scaleform `F[A-Z]*`, `PS_Bink*`, `PS_Minimap*`,
//! `*_IM_*`, `PS_BlitUI*`), in sink order (the title's recording order).
//! It needs the pack's v2.12 records: `IA` (views, pass phase and stage),
//! `IA_DATA` (index and vertex bytes) and the pipeline's `GSTATE` (blend,
//! input layout). Captures before v2.12 have none: every row is then
//! "no IA bytes" and nothing is drawn.

use std::collections::HashMap;
use std::path::Path;

use xnw_formats::framepack::{self as fp, kind, reason};
use xnw_gpu::d3d12x::tap::PipeState;
use xnw_world::bridge::Programs;
use xnw_world::ui::{Input, Row};

use crate::pack::Pack;

/// A UI-class program name (uikeep.rs `ui_name`, the same rule).
pub fn ui_name(name: &str) -> bool {
    let n = name.rsplit(':').next().unwrap_or(name);
    let mut c = n.chars();
    (c.next() == Some('F') && c.next().is_some_and(|x| x.is_ascii_uppercase())) || n.starts_with("PS_Bink") || n.starts_with("PS_Minimap") || n.contains("_IM_") || n.starts_with("PS_BlitUI")
}

/// One UI draw of a frame.
#[derive(Clone, Debug, Default)]
pub struct UiSink {
    pub sink: u32,
    pub op: u8,
    pub count: u32,
    pub pipeline: u64,
    pub vs: u128,
    pub ps: u128,
    pub names: [String; 2],
    /// DRAW_ARGS: instances, first vertex or index, base vertex, first instance.
    pub args: [u32; 4],
    pub ia: Option<fp::Ia>,
    pub state: Option<PipeState>,
}

/// The frame's UI draws, in sink order, with `names` (container -> name).
pub fn rows_of(p: &Pack, names: &HashMap<u128, String>) -> Vec<UiSink> {
    let stage = |pipeline: u64, s: &str| p.data.programs.iter().find(|g| g.pipeline == pipeline && g.stage == s).map(|g| g.container);
    let short = |h: u128| names.get(&h).map(|n| n.trim_end_matches("_Wrapped").rsplit_once('_').map_or(n.as_str(), |x| x.0).to_string()).unwrap_or_else(|| format!("{:08x}", (h >> 96) as u32));
    let states: HashMap<u64, PipeState> = p.data.refs_of(fp::FRAME_SCOPE).filter(|r| r.kind == kind::GSTATE && reason::has_bytes(r.reason)).filter_map(|r| Some((r.addr, PipeState::decode(&p.blob(r.hash)?)?))).collect();
    let mut out = Vec::new();
    for s in p.data.sinks.iter().filter(|s| matches!(s.op, fp::op::DRAW | fp::op::INDEXED)) {
        let (Some(vs), Some(ps)) = (stage(s.pipeline, "vertex"), stage(s.pipeline, "pixel")) else { continue };
        let pn = short(ps);
        if !names.get(&ps).is_some_and(|n| ui_name(n)) {
            continue;
        }
        let rec = |k: u16| p.data.refs_of(s.index).find(|r| r.kind == k && reason::has_bytes(r.reason)).and_then(|r| p.blob(r.hash));
        let args = rec(kind::DRAW_ARGS).map(|b| std::array::from_fn(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap_or_default()))).unwrap_or([1, 0, 0, 0]);
        let ia = rec(kind::IA).and_then(|b| fp::Ia::decode(&b));
        out.push(UiSink { sink: s.index, op: s.op, count: s.count, pipeline: s.pipeline, vs, ps, names: [short(vs), pn], args, ia, state: states.get(&s.pipeline).cloned() });
    }
    out
}

/// The row's line: sink, op and count, programs, stage, IA views.
pub fn row_line(r: &UiSink) -> String {
    let stage = r.ia.map_or("-".to_string(), |ia| if ia.phase == u32::MAX { format!("outside a pass, TLS stage {}", ia.stage) } else { format!("phase {} stage {}", ia.phase, ia.stage) });
    let layout = r.state.as_ref().map_or("-".to_string(), |s| s.inputs.iter().map(|i| format!("{}{}:{}@{}+{}", i.name, i.index, i.format, i.slot, i.offset)).collect::<Vec<_>>().join(" "));
    let blend = r.state.as_ref().map_or("-".to_string(), |s| format!("{:?}", s.blend));
    format!("sink {} {} {} x{} {}/{}; {}; layout {layout}; blend {blend}; ia {}", r.sink, fp::op::name(r.op), r.count, r.args[0], r.names[0], r.names[1], stage, r.ia.map_or("none (capture before v2.12)".into(), |ia| format!("index {:#x}+{} f{}, vertex {:#x}+{} stride {}", ia.index.0, ia.index.1, ia.index.2, ia.vertex[0].0, ia.vertex[0].1, ia.vertex[0].2)))
}

/// Whether a row's TLS stage draws the UI plane: stage 30 (HUD, menus,
/// Scaleform; sub-pass 178), or no stage read (u32::MAX).
pub fn on_plane(stage: u32) -> bool {
    stage == 30 || stage == u32::MAX
}

/// Whether a row is a triangle strip: the recorded topology (D3D 5), or,
/// for records without one, a plain 4-vertex Draw (the Bink quad).
pub fn strip(topology: u32, op: u8, count: u32) -> bool {
    topology == 5 || topology == 0 && op == fp::op::DRAW && count == 4
}

/// Translated UI pairs by (vertex, pixel) container, with the slots they
/// name (kept across frames by the live path).
pub type Translated = HashMap<(u128, u128), Result<(Programs, HashMap<(u16, u32), u32>), String>>;

fn translate(p: &Pack, r: &UiSink) -> Result<(Programs, HashMap<(u16, u32), u32>), String> {
    let (prog, mut used, _) = crate::bridge::vertex_program(p, r.sink, 0, 0, false)?;
    let psc = p.blob(r.ps).ok_or_else(|| format!("no container bytes for {:032x}", r.ps))?;
    let ps = xnw_shader::emit::translate_with(&psc, xnw_world::bridge::options()).map_err(|e| format!("pixel program {:032x}: {e}", r.ps))?;
    let mut wave = 32;
    crate::bridge::named(&psc, &mut used, &mut wave)?;
    Ok((Programs { ps, ..prog }, used))
}

/// What one frame's UI came to.
pub struct Drawn {
    pub layer: Option<xnw_world::ui::Layer>,
    pub lines: Vec<String>,
    pub rows: usize,
}

/// Draws the frame's UI rows into one layer of `size` on `host`.
pub fn draw(p: &Pack, host: &xnw_gpu::host::Host, size: (u32, u32), names: &HashMap<u128, String>, cache: &mut Translated) -> Result<Drawn, String> {
    let rows = rows_of(p, names);
    let mut lines = Vec::new();
    let counts = std::sync::Arc::new([const { std::sync::atomic::AtomicU64::new(0) }; 3]);
    let mut texels = p.texel_provider(counts);
    // Owned per row first (bytes, loads), then borrowed by the device rows.
    struct Owned {
        prog: Programs,
        loaded: crate::bridge::Loaded,
        inputs: Vec<Input>,
        vertices: Vec<(u32, Vec<u8>, u32)>,
        indices: Option<(Vec<u8>, u32)>,
        blend: [u32; 8],
        cull: u32,
        draw: (u32, u32, u32, i32),
        strip: bool,
    }
    let mut owned = Vec::new();
    for r in &rows {
        let (Some(ia), Some(state)) = (r.ia, r.state.as_ref()) else {
            lines.push(format!("  ui sink {}: no IA bytes or pipeline state (capture before v2.12): not drawn", r.sink));
            continue;
        };
        if !on_plane(ia.stage) {
            // Stage 6 Bink frames into a movie texture, stage 8 Scaleform to a texture: not the UI plane.
            lines.push(format!("  ui sink {} {}/{}: TLS stage {}: draws into a texture, not the UI plane: not drawn", r.sink, r.names[0], r.names[1], ia.stage));
            continue;
        }
        let t = cache.entry((r.vs, r.ps)).or_insert_with(|| translate(p, r));
        let (prog, used) = match t {
            Ok(x) => x.clone(),
            Err(e) => {
                lines.push(format!("  ui sink {} {}/{}: not translated: {e}", r.sink, r.names[0], r.names[1]));
                continue;
            }
        };
        let data = |slot: u16| p.data.refs_of(r.sink).find(|x| x.kind == kind::IA_DATA && x.slot == slot && reason::has_bytes(x.reason)).and_then(|x| p.blob(x.hash));
        let indexed = r.op == fp::op::INDEXED;
        let indices = if indexed {
            match data(0) {
                Some(b) => Some((b, ia.index.2)),
                None => {
                    lines.push(format!("  ui sink {}: no index bytes: not drawn", r.sink));
                    continue;
                }
            }
        } else {
            None
        };
        let vertices: Vec<(u32, Vec<u8>, u32)> = (0..fp::Ia::default().vertex.len() as u16).filter_map(|k| Some((u32::from(k), data(1 + k)?, ia.vertex[usize::from(k)].2))).collect();
        let inputs: Vec<Input> = state.inputs.iter().map(|i| Input { name: i.name.clone(), index: i.index, format: i.format, slot: i.slot, offset: i.offset, per_instance: i.per_instance, step: i.step }).collect();
        let loaded = crate::bridge::load(p, r.sink, &crate::bridge::Row::default(), None, &used, &mut texels);
        // `XNW_UI_LACKS=1`: what the bridge's binding of each row lacks (stand-ins, nulls).
        if std::env::var("XNW_UI_LACKS").is_ok_and(|v| v == "1") && !loaded.lacks.is_empty() {
            lines.push(format!("  ui sink {} {}/{} lacks: {}", r.sink, r.names[0], r.names[1], loaded.lacks.iter().cloned().collect::<Vec<_>>().join("; ")));
        }
        let [instances, first, base, _] = r.args;
        owned.push(Owned { prog, loaded, inputs, vertices, indices, blend: state.blend, cull: state.cull, draw: (r.count, instances, if indexed { 0 } else { first }, base as i32), strip: strip(ia.topology, r.op, r.count) });
    }
    let mut dev_rows = Vec::new();
    for (i, o) in owned.iter().enumerate() {
        let d = o.loaded.draw();
        match xnw_world::bridge::bind(&o.prog.ms, &o.prog.ps, &d) {
            Ok(binding) => dev_rows.push(Row { programs: &o.prog, binding, blend: o.blend, cull: o.cull, inputs: &o.inputs, vertices: o.vertices.iter().map(|v| (v.0, &v.1[..], v.2)).collect(), indices: o.indices.as_ref().map(|x| (&x.0[..], x.1)), draw: o.draw, strip: o.strip }),
            Err(e) => lines.push(format!("  ui row {i}: not bound: {e}")),
        }
    }
    if dev_rows.is_empty() {
        return Ok(Drawn { layer: None, lines, rows: rows.len() });
    }
    let limit = if host.caps.software { 120.0 } else { 1.0 };
    crate::replay::watchdog_arm();
    let layer = xnw_world::ui::draw_layer(host, &dev_rows, size, limit);
    crate::replay::watchdog_disarm();
    let layer = layer?;
    lines.extend(layer.lines.iter().cloned());
    Ok(Drawn { layer: Some(layer), lines, rows: rows.len() })
}

thread_local! {
    static CAPS: std::cell::RefCell<Option<xnw_gpu::host::Caps>> = const { std::cell::RefCell::new(None) };
}

/// The replay's device capabilities, for [`compose_frame`] (world-render sets them).
pub fn set_caps(caps: &xnw_gpu::host::Caps) {
    CAPS.with(|c| *c.borrow_mut() = Some(caps.clone()));
}

/// `ui=on` in the replay: the frame's UI layer over `rgba` (its size).
/// Prints the census, the row lines and the result.
pub fn compose_frame(dir: &Path, pass: &xnw_world::Pass, rgba: &mut [u8]) -> Result<(), String> {
    let caps = CAPS.with(|c| c.borrow().clone()).ok_or("ui: the replay's device capabilities were not set")?;
    let host = &xnw_world::ui::host_of(pass.device(), caps)?;
    let size = (pass.width, pass.height);
    let Some(p) = Pack::open(dir)? else {
        println!("ui: {} is not a v2 capture: no UI rows", dir.display());
        return Ok(());
    };
    let names = crate::audit::program_names(dir).unwrap_or_default();
    let mut cache = Translated::new();
    let d = draw(&p, host, size, &names, &mut cache)?;
    for l in &d.lines {
        println!("{l}");
    }
    match d.layer {
        Some(layer) => {
            xnw_world::ui::compose(rgba, &layer.rgba);
            println!("ui: {} rows, {} drawn in {:.3} s on {}; layer pixels covered {}", d.rows, layer.drawn, layer.seconds, host.caps.adapter, xnw_world::ui::covered(&layer.rgba));
        }
        None => println!("ui: {} rows, 0 drawn", d.rows),
    }
    Ok(())
}

/// `xnw-lab ui <frame dirs...>`: each frame's UI rows (no device).
pub fn ui_cmd(args: &[String]) -> Result<(), String> {
    for a in args.iter().filter(|a| !a.starts_with("--") && Path::new(a).is_dir()) {
        let dir = Path::new(a);
        let Some(p) = Pack::open(dir)? else {
            println!("{}: not a v2 capture", dir.display());
            continue;
        };
        let names = crate::audit::program_names(dir).ok_or("no shader-names.tsv above the frame")?;
        let rows = rows_of(&p, &names);
        println!("{}: {} UI rows ({} with IA bytes)", dir.display(), rows.len(), rows.iter().filter(|r| r.ia.is_some()).count());
        for r in &rows {
            println!("  {}", row_line(r));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_stage_30_reaches_the_plane() {
        assert!(on_plane(30) && on_plane(u32::MAX));
        assert!(!on_plane(6) && !on_plane(8) && !on_plane(16));
    }

    #[test]
    fn strips_by_topology_or_the_four_vertex_quad() {
        assert!(strip(5, fp::op::INDEXED, 6));
        assert!(!strip(4, fp::op::DRAW, 4));
        assert!(strip(0, fp::op::DRAW, 4));
        assert!(!strip(0, fp::op::INDEXED, 4));
    }

    #[test]
    fn ui_names_are_uikeeps() {
        for n in ["FText_a4ab6e63_Wrapped", "FVertexCxformEAlpha_dc4780ad_Wrapped", "PS_BinkNoAlpha_1_Wrapped", "scaleform:FTexTGCxformEAlpha"] {
            assert!(ui_name(n), "{n}");
        }
        for n in ["PS_DeferredTextured_edc1bca3_Wrapped", "Foo", "VText_a4ab6e63_Wrapped", "PS_SSSResolve"] {
            assert!(!ui_name(n), "{n}");
        }
    }
}
