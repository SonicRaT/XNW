//! Captured dispatches through the shared compute runner (
//! `xnw_world::compute`): a dispatch sink's record as a [`Job`], its
//! program, and a chain of sinks run in order on one runner, so a later
//! sink (or a draw, `lighting_luts::run_draw`) reads what an earlier one
//! wrote, by title address.
//!
//! `world-render <frame> --adapter warp --lighting auto --run-dispatch
//! <sink>,<sink>,...` runs them and prints, per sink, what it read that an
//! earlier one kept and what it wrote (non-zero bytes read back).
//!
//! What the capture gives a slot, and what stands in:
//! - the group counts: kind 60 (v2.9) x, y, z; before it, the sink's count
//!   as x and y = z = 1 (a line in `lacks`);
//! - a texture's dimension: a placed view's resource and view description
//!   (words 0 and 15); a resource view's dimension is not captured before
//!   it is recorded there: depth 1 is 2D, deeper takes the program's read
//!   (array or 3D) [I];
//! - texels: the pack's 2D mip 0 where it has them, else zeros;
//! - unordered views and their first bytes: what the capture read (often
//!   none: reason 10, a GPU output), else zeros.

use std::collections::HashMap;

use xnw_formats::framepack::{self as fp, kind, view};
use xnw_shader::emit;
use xnw_world::compute::{self as cr, Arg, Buffer, Image, Job, Outcome, Res, Runner, Shape, Use};
use xnw_world::preskin::{self, Kind};

use crate::le32;
use crate::pack::Pack;

/// A buffer's address as the title's GPU sees it, past its CPU address.
const GPU_ALIAS: u64 = 0x4_0000_0000;
/// Kind 60 (`GROUPS`, v2.9): x, y, z group counts of a dispatch sink.
pub const GROUPS: u16 = 60;
/// In `named`'s map: a slot read as a constant buffer.
const NAMED_CBV: u32 = u32::MAX;

/// The runner's key of a title address (CPU and GPU aliases agree).
pub fn key(addr: u64) -> u64 {
    if addr == 0 { 0 } else { addr % GPU_ALIAS }
}

enum LRes {
    Buffer { key: u64, kind: Kind, uav: bool, len: u64, bytes: Vec<u8> },
    Image { key: u64, shape: Shape, format: u32, width: u32, height: u32, depth: u32, uav: bool, texels: Vec<u8> },
}

#[derive(Default)]
enum LArg {
    #[default]
    None,
    Constants(Vec<u32>),
    Table(Vec<(u32, LRes)>),
    View(Vec<u8>),
}

/// A dispatch sink's record as owned bytes, and what it lacked.
#[derive(Default)]
pub struct Loaded {
    roots: Vec<LArg>,
    pub samplers: Vec<(u32, xnw_world::bridge::SamplerDesc)>,
    pub groups: [u32; 3],
    pub lacks: Vec<String>,
}

impl Loaded {
    pub fn job(&self) -> Job<'_> {
        fn res(r: &LRes) -> Res<'_> {
            match r {
                LRes::Buffer { key, kind, uav, len, bytes } => Res::Buffer(Buffer { key: *key, kind: *kind, uav: *uav, len: *len, bytes }),
                LRes::Image { key, shape, format, width, height, depth, uav, texels } => Res::Image(Image { key: *key, shape: *shape, format: *format, width: *width, height: *height, depth: *depth, uav: *uav, texels }),
            }
        }
        let roots = self
            .roots
            .iter()
            .map(|a| match a {
                LArg::None => Arg::None,
                LArg::Constants(c) => Arg::Constants(c.clone()),
                LArg::Table(v) => Arg::Table(v.iter().map(|(k, r)| (*k, res(r))).collect()),
                LArg::View(b) => Arg::View(b),
            })
            .collect();
        Job { roots, heap: Vec::new(), samplers: self.samplers.clone(), sampler_slots: 128, groups: self.groups }
    }
}

/// An indirect dispatch sink's group counts from its argument record
/// (`INDIRECT_ARGS`, v2.11): the 12 B there in what the runner holds (an
/// earlier dispatch wrote them), and the record's address. None before
/// v2.11 or when nothing held covers the record (the GPU wrote it).
pub fn indirect_groups(pack: &Pack, sink: u32, runner: &mut Runner) -> Option<([u32; 3], u64)> {
    let at = pack.data.refs_of(sink).find(|r| r.kind == kind::INDIRECT_ARGS)?.addr;
    let b = runner.read_at(key(at), 12)?;
    Some(([le32(&b, 0), le32(&b, 4), le32(&b, 8)], at))
}

/// Sink `sink`'s group counts: kind 60, else (count, 1, 1) and a lack.
pub fn groups(pack: &Pack, sink: u32) -> ([u32; 3], Option<String>) {
    if let Some(b) = pack.data.refs_of(sink).find(|r| r.kind == GROUPS).and_then(|r| pack.bytes_of(r)).filter(|b| b.len() >= 12) {
        return ([le32(&b, 0), le32(&b, 4), le32(&b, 8)], None);
    }
    let s = pack.data.sinks.iter().find(|s| s.index == sink);
    let x = s.map_or(1, |s| s.count);
    if s.is_some_and(|s| s.op == fp::op::INDIRECT_DISPATCH) {
        // The count is the most commands; the GPU writes x, y, z (`indirect_groups`).
        return ([x, 1, 1], Some(format!("the y and z group counts of an indirect dispatch (GPU-written; held only when its producer ran): {x} x 1 x 1")));
    }
    ([x, 1, 1], Some(format!("the y and z group counts (no kind 60 before v2.9): {x} x 1 x 1")))
}

/// A texture view's shape and view format from its captured words; a
/// resource view's depth beyond 1 takes `read` (the program's MIMG
/// dimension) when the words do not say.
fn shape(v: &fp::SlotView, read: Option<u8>) -> Option<(Shape, u32)> {
    let format = if v.kind == view::PLACED && v.words[14] != 0 { v.words[14] } else { v.format };
    // Resource dimension (D3D12: 3 2D, 4 3D), view dimension (4 2D, 5 2D array, 8 3D).
    let by_words = match (v.words[0], v.words[15]) {
        (4, _) | (_, 8) => Some(Shape::Tex3D),
        (2, _) => Some(Shape::Tex1D),
        (3, 5) => Some(Shape::Tex2DArray),
        (3, 4) => Some(Shape::Tex2D),
        (3, _) => Some(if v.depth > 1 { Shape::Tex2DArray } else { Shape::Tex2D }),
        _ => None,
    };
    let s = by_words.or_else(|| match (v.depth, read) {
        (0 | 1, Some(0)) if v.height <= 1 => Some(Shape::Tex1D),
        (0 | 1, _) => Some(Shape::Tex2D),
        (_, Some(2)) => Some(Shape::Tex3D),
        (_, Some(5)) => Some(Shape::Tex2DArray),
        _ => None,
    })?;
    Some((s, format))
}

/// The record of dispatch sink `sink` for a program that reads `uses`
/// (table slots, by kind) and `named` (table slots with strides).
pub fn load(pack: &Pack, sink: u32, uses: &[Use], named: &HashMap<(u16, u32), u32>) -> Loaded {
    let mut out = Loaded::default();
    let put = |roots: &mut Vec<LArg>, i: u16, a: LArg| {
        if roots.len() <= usize::from(i) {
            roots.resize_with(usize::from(i) + 1, LArg::default);
        }
        roots[usize::from(i)] = a;
    };
    for r in pack.data.refs_of(sink) {
        match r.kind {
            kind::ROOTS => {
                if let Some(b) = pack.bytes_of(r) {
                    put(&mut out.roots, r.slot, LArg::Constants(b.chunks_exact(4).map(|c| le32(c, 0)).collect()));
                }
            }
            kind::ROOT_VIEW => match pack.bytes_of(r) {
                Some(b) => put(&mut out.roots, r.slot, LArg::View(b)),
                None => out.lacks.push(format!("root view {} at {:#x} (reason {}): nothing stands in", r.slot, r.addr, r.reason)),
            },
            _ => {}
        }
    }
    for (root, _, views) in pack.tables(sink) {
        let mut slots = Vec::new();
        for v in views {
            let read = uses.iter().find(|u| u16::from(u.root) == root && u.slot == v.slot);
            let stride = named.get(&(root, v.slot));
            if read.is_none() && stride.is_none() {
                continue;
            }
            let uav = v.layout == view::UNORDERED && v.kind == view::BUFFER;
            match v.kind {
                view::CONSTANT | view::BUFFER | view::PLACED_BUFFER => {
                    let bytes = pack.slot_bytes(sink, root, v.slot).ok().map(|(_, b)| b).unwrap_or_default();
                    let kind = match (v.kind, v.format, stride) {
                        (view::CONSTANT, ..) | (_, _, Some(&NAMED_CBV)) => Kind::Cbv,
                        (view::BUFFER, f, _) if f != 0 && f != 39 => Kind::Typed { format: f as i32, bytes: v.width.max(1) as u32 },
                        (_, _, Some(&s)) if s > 4 => Kind::Structured(s),
                        _ => Kind::Raw,
                    };
                    if bytes.is_empty() && !uav {
                        out.lacks.push(format!("root {root} slot {}: {} B ({:?}) without bytes: zeros", v.slot, v.bytes, kind));
                    }
                    let bytes = if kind == Kind::Cbv && bytes.is_empty() { vec![0; 256] } else { bytes };
                    slots.push((v.slot, LRes::Buffer { key: key(v.addr), kind, uav, len: v.bytes.max(bytes.len() as u64), bytes }));
                }
                view::TEXTURE | view::PLACED => {
                    let Some((shape, format)) = shape(&v, read.and_then(|u| u.image).map(|i| i.dim)) else {
                        out.lacks.push(format!("root {root} slot {}: a {}x{}x{} texture of unknown dimension: left out", v.slot, v.width, v.height, v.depth));
                        continue;
                    };
                    let texels = match (shape, pack.view_texels(&v)) {
                        (Shape::Tex2D, Ok(t)) if t.width as u64 == v.width && t.height == v.height => t.rows,
                        (_, got) => {
                            if !read.is_some_and(|u| u.store) {
                                out.lacks.push(format!("root {root} slot {}: {shape:?} {}x{}x{} format {format}: no texels ({}): zeros", v.slot, v.width, v.height, v.depth, got.err().unwrap_or("not 2D")));
                            }
                            Vec::new()
                        }
                    };
                    slots.push((v.slot, LRes::Image { key: key(v.addr), shape, format, width: v.width as u32, height: v.height, depth: u32::from(v.depth.max(1)), uav, texels }));
                }
                _ => {}
            }
        }
        put(&mut out.roots, root, LArg::Table(slots));
    }
    let (g, lack) = groups(pack, sink);
    out.groups = g;
    out.lacks.extend(lack);
    match crate::bridge::samplers(pack, sink, &mut out.lacks) {
        Some(s) => out.samplers = s,
        None => out.lacks.push("sampler heap entries: default samplers (linear, wrap)".into()),
    }
    out
}

/// A dispatch sink's program: translated, its wave, what it reads.
pub struct Program {
    pub shader: emit::HostShader,
    pub wave: u32,
    pub uses: Vec<Use>,
    pub named: HashMap<(u16, u32), u32>,
    pub container: u128,
}

pub fn program(pack: &Pack, sink: u32) -> Result<Program, String> {
    let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).ok_or("no such sink")?.pipeline;
    let container = pack.data.programs.iter().find(|p| p.pipeline == pipeline && p.stage == "compute").map(|p| p.container).ok_or_else(|| format!("sink {sink}: no compute program"))?;
    let c = pack.blob(container).ok_or_else(|| format!("no container bytes for {container:032x}"))?;
    let words = cs_patch(container);
    if !words.is_empty() {
        println!("  XNW_CS_PATCH: sink {sink} program {:08x}: {} words over its code", (container >> 96) as u32, words.len());
    }
    let shader = emit::translate_patched(&c, xnw_world::bridge::options(), &words).map_err(|e| format!("sink {sink} program {container:032x}: {e}"))?;
    let wave = preskin::wave_size(&c).ok_or("no wave size")?;
    let uses = cr::uses(&c)?;
    let (mut named, mut w) = (HashMap::new(), 32);
    crate::bridge::named(&c, &mut named, &mut w)?;
    Ok(Program { shader, wave, uses, named, container })
}

/// The compute-side debug export (LIT6; LIT3's `--ps-patch` for
/// dispatches): `XNW_CS_PATCH=<container hex prefix>:<off>=<word>,...[;...]`
/// writes GFX10 words over the code of the programs whose container hash
/// starts with the prefix (offsets as the listing prints them). Off
/// when unset; a malformed entry is skipped with a line.
pub fn cs_patch(container: u128) -> Vec<(u32, u32)> {
    let Ok(v) = std::env::var("XNW_CS_PATCH") else { return Vec::new() };
    parse_cs_patch(&v, container)
}

fn parse_cs_patch(v: &str, container: u128) -> Vec<(u32, u32)> {
    let hex = |s: &str| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    let name = format!("{container:032x}");
    let mut out = Vec::new();
    for part in v.split(';').filter(|p| !p.trim().is_empty()) {
        let Some((prefix, words)) = part.split_once(':') else {
            println!("  XNW_CS_PATCH {part}: want <container prefix>:<off>=<word>,...");
            continue;
        };
        if !name.starts_with(&prefix.trim().to_ascii_lowercase()) {
            continue;
        }
        for kv in words.split(',').filter(|s| !s.trim().is_empty()) {
            match kv.split_once('=').and_then(|(k, w)| Some((hex(k)?, hex(w)?))) {
                Some(p) => out.push(p),
                None => println!("  XNW_CS_PATCH {kv}: want <off>=<word> (hex)"),
            }
        }
    }
    out
}

/// Runs `sinks` in order on `runner`; one line per sink (and its lacks),
/// each printed as it is made (a device failure ends the chain).
pub fn run_chain(pack: &Pack, sinks: &[u32], runner: &mut Runner) -> Result<Vec<String>, String> {
    let mut lines = Lines(Vec::new());
    let mut programs: HashMap<u64, Program> = HashMap::new();
    for &sink in sinks {
        let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).map(|s| s.pipeline).ok_or_else(|| format!("no sink {sink}"))?;
        if let std::collections::hash_map::Entry::Vacant(e) = programs.entry(pipeline) {
            match program(pack, sink) {
                Ok(p) => {
                    e.insert(p);
                }
                Err(x) => {
                    // A program that does not translate never reaches a dispatch: counted as a drop.
                    let d = runner.refuse(cr::why::TRANSLATE, x);
                    lines.push(format!("sink {sink}: dropped: {} ({})", d.reason, d.detail));
                    continue;
                }
            }
        }
        let p = &programs[&pipeline];
        let mut loaded = load(pack, sink, &p.uses, &p.named);
        if let Some((g, at)) = indirect_groups(pack, sink, runner) {
            loaded.groups = g;
            loaded.lacks.retain(|l| !l.starts_with("the y and z group counts"));
            lines.push(format!("sink {sink}: indirect groups {g:?} from the held record at {at:#x}"));
        }
        let images = p.uses.iter().filter(|u| u.image.is_some()).count();
        let head = format!("sink {sink} ({:08x}, groups {:?}, {} slots read, {images} as images)", (p.container >> 96) as u32, loaded.groups, p.uses.len());
        // The line goes out before the dispatch: a device removal names its sink.
        println!("  {head}: running");
        let got = runner.dispatch(&p.shader, p.wave, &p.uses, &loaded.job()).map_err(|e| format!("{head}: {e}"))?;
        match got {
            Outcome::Dropped(d) => lines.push(format!("{head}: dropped: {} ({})", d.reason, d.detail)),
            Outcome::Ran(r) => {
                let mut wrote = Vec::new();
                for &k in &r.wrote {
                    let b = runner.read(k)?;
                    wrote.push(format!("{k:#x} {:?} {} of {} B non-zero", runner.held(k).map(held_text).unwrap_or_default(), b.iter().filter(|x| **x != 0).count(), b.len()));
                }
                lines.push(format!("{head}: ran in {:.3} s; read kept {:x?}; wrote [{}]", r.seconds, r.read_kept, wrote.join("; ")));
            }
        }
        lines.extend(loaded.lacks.iter().map(|l| format!("    lacks: {l}")));
    }
    Ok(lines.0)
}

/// Lines printed as they are pushed.
struct Lines(Vec<String>);

impl Lines {
    fn push(&mut self, l: String) {
        println!("  {l}");
        self.0.push(l);
    }
    fn extend(&mut self, ls: impl Iterator<Item = String>) {
        for l in ls {
            self.push(l);
        }
    }
}

fn held_text(h: cr::Held) -> String {
    match h {
        cr::Held::Buffer { len } => format!("buffer {len} B"),
        cr::Held::Image { shape, format, width, height, depth } => format!("{shape:?} {width}x{height}x{depth} format {format}"),
    }
}

/// The buffers of draw sink `sink`'s tables the runner holds (LIT5): what
/// an earlier dispatch wrote, or a stand-in put there with
/// [`Runner::hold`], for the draw bridge's `Loaded::set_buffer`: (root,
/// slot, kind, bytes). `named` (the draw programs' slots with strides)
/// gives the kind: structured past 4 B, else raw; constant views are left
/// to the capture.
pub fn held_buffers(pack: &Pack, sink: u32, runner: &mut Runner, named: &HashMap<(u16, u32), u32>) -> Vec<(u16, u32, Kind, Vec<u8>)> {
    let mut out = Vec::new();
    for (root, _, views) in pack.tables(sink) {
        for v in views.iter().filter(|v| v.kind == view::BUFFER || v.kind == view::PLACED_BUFFER) {
            let Some(&stride) = named.get(&(root, v.slot)) else { continue };
            if stride == NAMED_CBV || !matches!(runner.held(key(v.addr)), Some(cr::Held::Buffer { .. })) {
                continue;
            }
            let Ok(b) = runner.read(key(v.addr)) else { continue };
            let kind = if stride > 4 { Kind::Structured(stride) } else { Kind::Raw };
            out.push((root, v.slot, kind, b));
        }
    }
    out
}

/// The textures of draw sink `sink`'s tables the runner holds, in the
/// draw's view format: (root, slot, texels) for the draw bridge's
/// `Loaded::set_texture`, and why others were left as captured. A 1D
/// texture goes in only where the draw's programs read that slot 2D.
pub fn held_textures(pack: &Pack, sink: u32, runner: &mut Runner) -> (Vec<(u16, u32, xnw_world::textures::Provided)>, Vec<String>) {
    let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).map(|s| s.pipeline);
    let mut reads: Vec<Use> = Vec::new();
    for p in pack.data.programs.iter().filter(|p| Some(p.pipeline) == pipeline) {
        if let Some(u) = pack.blob(p.container).and_then(|c| cr::uses(&c).ok()) {
            reads.extend(u);
        }
    }
    let (mut out, mut lacks) = (Vec::new(), Vec::new());
    for (root, _, views) in pack.tables(sink) {
        for v in views.iter().filter(|v| v.kind == view::TEXTURE || v.kind == view::PLACED) {
            if runner.held(key(v.addr)).is_none() {
                continue;
            }
            let format = if v.kind == view::PLACED && v.words[14] != 0 { v.words[14] } else { v.format };
            let read = reads.iter().find(|u| u16::from(u.root) == root && u.slot == v.slot).and_then(|u| u.image).map(|i| i.dim);
            let held = runner.held(key(v.addr));
            let shape = match held {
                Some(cr::Held::Image { shape, .. }) => shape,
                _ => {
                    lacks.push(format!("root {root} slot {}: a buffer held where the draw has a texture: not taken", v.slot));
                    continue;
                }
            };
            // The draw's programs must read it in the dimension it is given (hang1).
            let flat = shape == Shape::Tex1D && read == Some(1);
            if read.is_some_and(|d| d != shape.dim() && !flat) {
                lacks.push(format!("root {root} slot {}: held {shape:?}, the draw reads dim {read:?}: not taken", v.slot));
                continue;
            }
            match runner.provided_as(key(v.addr), Some(format), flat) {
                Ok(t) => out.push((root, v.slot, t)),
                Err(e) => lacks.push(format!("root {root} slot {}: held, not taken: {e}", v.slot)),
            }
        }
    }
    (out, lacks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_from_words_and_reads() {
        let v = |kind, depth, w0, w15| fp::SlotView { kind, depth, words: std::array::from_fn(|i| match i { 0 => w0, 15 => w15, _ => 0 }), ..fp::SlotView::default() };
        assert_eq!(shape(&v(view::PLACED, 1, 3, 4), None).map(|s| s.0), Some(Shape::Tex2D));
        assert_eq!(shape(&v(view::PLACED, 32, 3, 5), None).map(|s| s.0), Some(Shape::Tex2DArray));
        assert_eq!(shape(&v(view::PLACED, 1, 4, 8), None).map(|s| s.0), Some(Shape::Tex3D));
        assert_eq!(shape(&v(view::TEXTURE, 1, 0, 0), None).map(|s| s.0), Some(Shape::Tex2D));
        assert_eq!(shape(&v(view::TEXTURE, 32, 0, 0), None), None, "deeper than 1 without words or a read: unknown");
        assert_eq!(shape(&v(view::TEXTURE, 32, 0, 0), Some(2)).map(|s| s.0), Some(Shape::Tex3D));
        assert_eq!(key(0x5_0000_1000), 0x1_0000_1000);
    }

    #[test]
    fn cs_patch_picks_the_program_by_prefix() {
        let c = 0x7679_96c1_ce51_eace_99da_7699_f9e8_1c52u128;
        assert_eq!(parse_cs_patch("767996c1:0xd00=7E040302,d2c=BF800000", c), [(0xd00, 0x7e04_0302), (0xd2c, 0xbf80_0000)]);
        assert_eq!(parse_cs_patch("deadbeef:10=1;767996:20=2", c), [(0x20, 2)]);
        assert!(parse_cs_patch("767996c1:zz=1,nokey", c).is_empty());
    }
}
