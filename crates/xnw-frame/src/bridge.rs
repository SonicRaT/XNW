//! `xnw world-render <frame dir> --adapter warp --bridge <list index>`: one
//! captured draw through the title's own mesh and pixel program
//! (`xnw_world::bridge`), against our pass's albedo for
//! the same draw. Here: the capture side (which sink, which programs, the
//! record's bytes as a binding input) and the comparison's two pictures.
//!
//! What a v2 capture lacks for such a draw, and what stands in ([`load`]):
//! - the mesh program's root view (the vertex buffer by its GPU address):
//!   the list's own copy of that buffer;
//! - the group count (the title's indirect arguments are a GPU output):
//!   the meshlet buffer's count;
//! - the bindless words (texture heap indices kept behind the material's
//!   constants, in a buffer the capture cuts): the list row's albedo slot at
//!   every word, so target 0 finds the albedo and the other maps do not;
//! - a skinned draw's bones past the palette copy's 16 MB cut: the hook's
//!   span of the draw's bones and the list's palette window around it;
//! - its bone mask words at slot 86 + record +0x74 (past that cut too):
//!   the capture's arena record;
//! - the bindless words and sampler heap entries before v2.4 (pk6): the
//!   albedo's index at every word, one default sampler;
//! - the targets' formats: 32-bit float, as the pixel program exports.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use xnw_core::pair::Pair;
use xnw_formats::framepack::{self as fp, kind, reason, view};
use xnw_world::bridge::{self, Draw, Root, Slot};
use xnw_world::preskin::{Kind, View};
use xnw_world::textures::{Provided, Provider};
use xnw_world::{DECAL_STAGES, DrawRecord, FrameList, GeomKey, HostGeometry, NO_PALETTE, Pass, RAW_SHADER, TexMode};
use xnw_shader::emit;

use crate::le32;
use crate::load::load_list;
use crate::pack::Pack;

/// The frame table's slot of the buffer holding the bindless words.
const BINDLESS_SLOT: u32 = 86;
/// Bindless words written when the capture has none.
const BINDLESS_WORDS: usize = 16;
/// A table view no program's resource list names is still bound when it
/// is at most this long (programs read past their declared ranges).
const SMALL: usize = 64 << 10;
/// In the map of table slots the programs name: a slot read as a constant
/// buffer (other values: the stride a buffer is read at).
const NAMED_CBV: u32 = u32::MAX;
/// Sampler heap slots, each the default sampler.
const SAMPLER_SLOTS: u32 = 128;

enum LSlot {
    Buffer(Kind, Vec<u8>),
    Texture(Provided),
}

#[derive(Default)]
enum LRoot {
    #[default]
    None,
    Constants(Vec<u32>),
    Table(Vec<(u32, LSlot)>),
    View(Vec<u8>),
}

/// A draw's binding input as owned bytes, and what the record lacked.
#[derive(Default)]
pub struct Loaded {
    roots: Vec<LRoot>,
    heap: Vec<(u32, LSlot)>,
    pub groups: u32,
    pub lacks: Vec<String>,
    /// Captured sampler heap entries at their indices (v2.4).
    pub samplers: Vec<(u32, bridge::SamplerDesc)>,
    /// The title address of each captured table buffer view (root, slot,
    /// address; ZC2).
    addrs: Vec<(u16, u32, u64)>,
    /// Slots whose bytes `load` wrote over or past the record's copy, by
    /// (root, slot): the copy's length and the ranges written (the row's own
    /// words; [`bridge::TitleView`], ZC3).
    own: HashMap<(u16, u32), (usize, Vec<(usize, usize)>)>,
    /// The paired reservation the row's addresses fold by ([`Row::pair`]).
    pair: Pair,
}

/// Range `r` of slot `key`'s bytes written by `load` over a copy of
/// `before` bytes (the first write of the slot fixes the copy's length).
fn own(marks: &mut HashMap<(u16, u32), (usize, Vec<(usize, usize)>)>, key: (u16, u32), before: usize, r: (usize, usize)) {
    marks.entry(key).or_insert((before, Vec::new())).1.push(r);
}

/// Bit of a mesh program's first root constant: set, group k draws the
/// meshlet a visible-meshlet list names at k; clear, meshlet k.
const VISIBLE_LIST: u32 = 1 << 28;

impl Loaded {
    /// The title address of the binding's table buffer views, by the
    /// address of their bytes here (`bridge_ms::Batch::add`, ZC2); the GPU
    /// view of the paired reservation folded onto the CPU view.
    /// Each view's copied length and the ranges `load` wrote as the row's
    /// own words go with it (ZC3): the rest of its bytes is padding.
    pub fn addrs(&self) -> Vec<(usize, bridge::TitleView)> {
        let fold = |a: u64| self.pair.fold(a);
        self.addrs
            .iter()
            .filter_map(|&(root, slot, addr)| match self.roots.get(usize::from(root)) {
                Some(LRoot::Table(v)) => v.iter().find(|s| s.0 == slot).and_then(|s| {
                    let LSlot::Buffer(_, b) = &s.1 else { return None };
                    let (captured, overlays) = self.own.get(&(root, slot)).cloned().unwrap_or((b.len(), Vec::new()));
                    Some((b.as_ptr() as usize, bridge::TitleView { addr: fold(addr), captured, overlays }))
                }),
                _ => None,
            })
            .collect()
    }

    /// The title ranges (address, bytes, unit) of the binding's table buffer
    /// views of at least `min` bytes, as [`Loaded::addrs`] folds them.
    pub fn ranges(&self, min: usize) -> Vec<(u64, u64, u64)> {
        let fold = |a: u64| self.pair.fold(a);
        self.addrs
            .iter()
            .filter_map(|&(root, slot, addr)| match self.roots.get(usize::from(root)) {
                Some(LRoot::Table(v)) => v.iter().find(|s| s.0 == slot).and_then(|s| if let LSlot::Buffer(k, b) = &s.1 { (b.len() >= min && *k != Kind::Cbv).then_some((fold(addr), b.len() as u64, bridge::unit(*k))) } else { None }),
                _ => None,
            })
            .collect()
    }

    /// Every meshlet in order instead of the title's visible-meshlet list
    /// (an output of its GPU culling, which no capture holds).
    pub fn direct_meshlets(&mut self) {
        for r in &mut self.roots {
            if let LRoot::Constants(c) = r {
                if let Some(w) = c.first_mut().filter(|w| **w & VISIBLE_LIST != 0) {
                    self.lacks.push(format!("the visible-meshlet list (slot {} of the frame table from entry {}): groups draw the meshlets in order", *w & 0x3FF, (*w >> 10 & 0x1FFFF) * 16));
                    *w &= !VISIBLE_LIST;
                }
            }
        }
    }

    /// The bytes of table slot `slot` of root `root`, to change in place
    /// (HAIR1's placement probe); None when the record holds none.
    pub fn slot_bytes_mut(&mut self, root: u16, slot: u32) -> Option<&mut Vec<u8>> {
        match self.roots.get_mut(usize::from(root))? {
            LRoot::Table(v) => v.iter_mut().find(|s| s.0 == slot).and_then(|s| if let LSlot::Buffer(_, b) = &mut s.1 { Some(b) } else { None }),
            _ => None,
        }
    }

    /// Puts texture `t` at table slot `slot` of root `root` in place of what
    /// the record gave there (LIT2: our G-buffer under the title's lighting).
    pub fn set_texture(&mut self, root: u16, slot: u32, t: Provided) {
        if self.roots.len() <= usize::from(root) {
            self.roots.resize_with(usize::from(root) + 1, LRoot::default);
        }
        if !matches!(self.roots[usize::from(root)], LRoot::Table(_)) {
            self.roots[usize::from(root)] = LRoot::Table(Vec::new());
        }
        if let LRoot::Table(v) = &mut self.roots[usize::from(root)] {
            v.retain(|s| s.0 != slot);
            v.push((slot, LSlot::Texture(t)));
            v.sort_by_key(|s| s.0);
        }
    }

    /// Puts buffer bytes at table slot `slot` of root `root` in place of
    /// what the record gave there (LIT2: a GPU output the capture holds as zero).
    pub fn set_buffer(&mut self, root: u16, slot: u32, kind: Kind, bytes: Vec<u8>) {
        if let Some(LRoot::Table(v)) = self.roots.get_mut(usize::from(root)) {
            v.retain(|s| s.0 != slot);
            v.push((slot, LSlot::Buffer(kind, bytes)));
            v.sort_by_key(|s| s.0);
        }
    }

    /// What table slot `slot` of root `root` holds for the binding, as a
    /// short text ("-" when nothing: the slot reads through the guard).
    pub fn describe(&self, root: u16, slot: u32) -> String {
        let Some(LRoot::Table(v)) = self.roots.get(usize::from(root)) else { return "-".into() };
        match v.iter().find(|s| s.0 == slot).map(|s| &s.1) {
            None => "-".into(),
            Some(LSlot::Buffer(k, b)) => format!("buffer {k:?} {} B{}", b.len(), if b.iter().all(|x| *x == 0) { " (all zero)" } else { "" }),
            Some(LSlot::Texture(t)) => format!("texture {}x{} format {}{}", t.width, t.height, t.format, if t.rows.iter().all(|x| *x == 0) { " (all zero)" } else { "" }),
        }
    }

    pub fn draw(&self) -> Draw<'_> {
        fn slot(s: &LSlot) -> Slot<'_> {
            match s {
                LSlot::Buffer(kind, bytes) => Slot::Buffer(View { kind: *kind, bytes }),
                LSlot::Texture(t) => Slot::Texture(t),
            }
        }
        let roots = self
            .roots
            .iter()
            .map(|r| match r {
                LRoot::None => Root::None,
                LRoot::Constants(c) => Root::Constants(c.clone()),
                LRoot::Table(v) => Root::Table(v.iter().map(|(k, s)| (*k, slot(s))).collect()),
                LRoot::View(b) => Root::View(b),
            })
            .collect();
        Draw { roots, heap: self.heap.iter().map(|(k, s)| (*k, slot(s))).collect(), samplers: self.samplers.clone(), sampler_slots: SAMPLER_SLOTS, groups: self.groups }
    }
}

/// What the list row says of a draw: its meshlet and vertex buffers, and
/// the descriptor handles of the textures it names (the albedo first).
#[derive(Clone, Debug, Default)]
pub struct Row {
    pub mesh_at: u64,
    pub vb_at: u64,
    pub handles: Vec<u64>,
    /// A skinned draw's bones as the hook read them (title address of the
    /// draw's first bone, 48 B a bone): see [`PALETTE`].
    pub palette: Option<(u64, Vec<u8>)>,
    /// The list's palette window the draw's bones lie in (title address of
    /// its first bone, bytes): the bones around the draw's own (wheels'
    /// instance bases).
    pub window: Option<(u64, Vec<u8>)>,
    /// The paired reservation its addresses fold by: the one published
    /// (`xnw_core::pair`, ALIAS1), else the default.
    pub pair: Pair,
}

impl Row {
    pub fn of(r: &DrawRecord) -> Row {
        Row { mesh_at: r.geom.mesh_at, vb_at: r.geom.vb_at, handles: std::iter::once(r.material.albedo_desc).chain(r.material.layers).collect(), palette: None, window: None, pair: xnw_core::pair::pair() }
    }

    /// The same, with the hook's palette span of hook draw `draw` and the
    /// list palette it sits in (placed by the span: its address less 48 B
    /// a bone of the row's `bone_first`).
    pub fn with_palette(r: &DrawRecord, list: &FrameList, pack: &Pack, draw: u32) -> Row {
        let [current, _] = pack.hook_palettes(draw);
        let window = current.as_ref().zip(list.palettes.get(r.palette as usize)).map(|((at, _), p)| (at.wrapping_sub(48 * u64::from(r.bone_first)), p[..p.len() / 12 * 12].iter().flat_map(|f| f.to_le_bytes()).collect()));
        Row { palette: current, window, ..Row::of(r) }
    }
}

/// The skin palette view (root 0 slot 5, 48 B a bone: `T8+0xA0` of the
/// TransformSkin programs). The stage-14 MSR programs read slot 4
/// (`T8+0x80`) instead: a small whole copy there (f887: 6800 B), not this.
const PALETTE: (u16, u32) = (0, 5);

/// The made-up address of a bindless stand-in the record has no view for
/// (far above any title address): such stand-ins share one span.
const STAND_IN_AT: u64 = 0xF000_0000_0000;

/// The record of sink `sink` as a binding input. `used` says which table
/// slots the two programs name, with the stride they read them at; `vb` is
/// the list's copy of the row's vertex buffer; `texels` gives a texture by
/// its descriptor handle.
pub fn load(pack: &Pack, sink: u32, row: &Row, vb: Option<&[u8]>, used: &HashMap<(u16, u32), u32>, texels: &mut Provider) -> Loaded {
    let mut out = Loaded { pair: row.pair, ..Loaded::default() };
    let put = |roots: &mut Vec<LRoot>, i: u16, r: LRoot| {
        if roots.len() <= usize::from(i) {
            roots.resize_with(usize::from(i) + 1, LRoot::default);
        }
        roots[usize::from(i)] = r;
    };
    // HAIR7: `bytes_of` also reads a view a live sink left in place.
    let bytes = |r: &fp::Ref| pack.bytes_of(r);
    for r in pack.data.refs_of(sink) {
        match r.kind {
            kind::ROOTS => {
                if let Some(b) = bytes(r) {
                    put(&mut out.roots, r.slot, LRoot::Constants(b.chunks_exact(4).map(|c| le32(c, 0)).collect()));
                }
            }
            kind::ROOT_VIEW => match (bytes(r), vb) {
                (Some(b), _) => put(&mut out.roots, r.slot, LRoot::View(b)),
                (None, Some(vb)) if r.addr == row.vb_at || r.addr == row.pair.to_gpu(row.vb_at) => {
                    out.lacks.push(format!("root view {} at {:#x} (reason {}): the list's vertex buffer {:#x} ({} B) stands in", r.slot, r.addr, r.reason, row.vb_at, vb.len()));
                    put(&mut out.roots, r.slot, LRoot::View(vb.to_vec()));
                }
                _ => out.lacks.push(format!("root view {} at {:#x} (reason {}): nothing stands in", r.slot, r.addr, r.reason)),
            },
            _ => {}
        }
    }
    for (root, handle, views) in pack.tables(sink) {
        let mut slots = Vec::new();
        for v in views {
            let wanted = used.get(&(root, v.slot));
            match v.kind {
                view::CONSTANT | view::BUFFER | view::PLACED_BUFFER => {
                    // HAIR7: a large view a live sink left in place that the
                    // programs do not name is not read (as a large copy is not bound).
                    let at = fp::table_slot(u32::from(root), v.slot);
                    if wanted.is_none() && pack.data.refs_of(sink).any(|r| r.kind == kind::BUFFER && r.slot == at && Pack::in_place(r)) {
                        if v.addr != 0 {
                            out.addrs.push((root, v.slot, v.addr));
                        }
                        continue;
                    }
                    let got = pack.slot_bytes(sink, root, v.slot).ok().map(|(_, b)| b);
                    let got = if (root, v.slot) == PALETTE {
                        let (before, mut placed) = (got.as_ref().map_or(0, Vec::len), Vec::new());
                        let b = windows(pack.palette_windows(sink), palette(row, &v, got, &mut out.lacks, &mut placed), &mut out.lacks, &mut placed);
                        for r in placed {
                            own(&mut out.own, PALETTE, before, r);
                        }
                        b
                    } else {
                        got
                    };
                    // Every view's address, bytes or not: a stand-in `load`
                    // puts at the slot later is bound at it (ZC3).
                    if v.addr != 0 {
                        out.addrs.push((root, v.slot, v.addr));
                    }
                    let Some(b) = got else {
                        if wanted == Some(&NAMED_CBV) {
                            // A program that loads a constant buffer where the heap holds another kind of view removes the device.
                            out.lacks.push(format!("root {root} slot {}: a constant buffer the programs name has no bytes: 256 zero bytes stand in", v.slot));
                            slots.push((v.slot, LSlot::Buffer(Kind::Cbv, vec![0; 256])));
                        } else if wanted.is_some() {
                            out.lacks.push(format!("root {root} slot {}: a view the programs name has no bytes", v.slot));
                        }
                        continue;
                    };
                    if v.addr >= row.mesh_at && row.pair.same(v.addr, row.mesh_at) {
                        out.groups = le32(&b, 0) & 0x1FFFF;
                    }
                    if wanted.is_none() && b.len() > SMALL {
                        continue;
                    }
                    if wanted.is_some() && (b.len() as u64) < v.bytes {
                        out.lacks.push(format!("root {root} slot {}: the copy is cut at {} of the view's {} B (the rest reads as zero)", v.slot, b.len(), v.bytes));
                    }
                    // R32_TYPELESS (39) is a raw view.
                    let kind = match (v.kind, v.format, wanted) {
                        (view::CONSTANT, ..) => Kind::Cbv,
                        (view::BUFFER, f, _) if f != 0 && f != 39 => Kind::Typed { format: f as i32, bytes: v.width.max(1) as u32 },
                        (_, _, Some(&NAMED_CBV)) => Kind::Cbv,
                        (_, _, Some(&stride)) if stride > 4 => Kind::Structured(stride),
                        _ => Kind::Raw,
                    };
                    slots.push((v.slot, LSlot::Buffer(kind, b)));
                }
                view::TEXTURE | view::PLACED if wanted.is_some() => match texels(handle + 32 * u64::from(v.slot)) {
                    Some(t) => slots.push((v.slot, LSlot::Texture(t))),
                    None => out.lacks.push(format!("root {root} slot {}: a texture the programs name has no texels", v.slot)),
                },
                _ => {}
            }
        }
        for (&(_, slot), _) in used.iter().filter(|(k, kind)| k.0 == root && **kind == NAMED_CBV) {
            if !slots.iter().any(|s| s.0 == slot) {
                out.lacks.push(format!("root {root} slot {slot}: a constant buffer the programs name is no view of the table: 256 zero bytes stand in"));
                slots.push((slot, LSlot::Buffer(Kind::Cbv, vec![0; 256])));
            }
        }
        slots.sort_by_key(|s| s.0);
        put(&mut out.roots, root, LRoot::Table(slots));
    }
    if out.groups == 0 {
        out.lacks.push(format!("the group count: no table view over the meshlet buffer {:#x}", row.mesh_at));
    } else {
        out.lacks.push(format!("the group count (indirect arguments): the meshlet buffer's count {} stands in", out.groups));
    }
    bindless(pack, sink, row, texels, &mut out);
    arena(pack, sink, &mut out);
    match samplers(pack, sink, &mut out.lacks) {
        Some(s) => out.samplers = s,
        None => out.lacks.push(format!("sampler heap entries: {SAMPLER_SLOTS} default samplers (linear, wrap)")),
    }
    out.lacks.push("the targets' formats: 32-bit float targets and D32".into());
    out
}

/// The palette view's bytes with the draw's bones at their place where
/// the copy lacks them (it is cut at 16 MB): first the list's palette
/// window past the copy's end, then the hook's span of the draw's own
/// bones over it unless the copy holds them whole. A line in `lacks` says which.
fn palette(row: &Row, v: &fp::SlotView, got: Option<Vec<u8>>, lacks: &mut Vec<String>, placed: &mut Vec<(usize, usize)>) -> Option<Vec<u8>> {
    let Some((at, bones)) = row.palette.as_ref().filter(|p| !p.1.is_empty()) else { return got };
    let place = |at: u64, len: usize| {
        let off = row.pair.fold(at).wrapping_sub(row.pair.fold(v.addr));
        (off % 48 == 0 && off + len as u64 <= v.bytes).then_some(off as usize)
    };
    let Some(off) = place(*at, bones.len()) else {
        lacks.push(format!("the palette: the hook's bones at {at:#x} lie outside the view at {:#x} ({} B): not placed", v.addr, v.bytes));
        return got;
    };
    let (first, n, end) = (off / 48, bones.len() / 48, off + bones.len());
    let mut b = got.unwrap_or_default();
    let copied = b.len();
    let held = copied.saturating_sub(off).min(bones.len()) / 48;
    let zero = b.get(off..end.min(copied)).map_or(0, |s| s.chunks(48).filter(|c| c.iter().all(|x| *x == 0)).count());
    // The list's window: only the bytes past the copy's end.
    let mut window = 0;
    if let Some((wat, w)) = &row.window
        && let Some(woff) = place(*wat, w.len())
        && woff + w.len() > copied
    {
        let from = copied.max(woff);
        if b.len() < woff + w.len() {
            b.resize(woff + w.len(), 0);
        }
        b[from..woff + w.len()].copy_from_slice(&w[from - woff..]);
        placed.push((from, woff + w.len()));
        window = (woff + w.len() - from) / 48;
    }
    if held == n && zero == 0 {
        // Read at the sink, later than the hook: the copy is what the program saw.
        let same = b[off..end].chunks(48).zip(bones.chunks(48)).filter(|(a, h)| a == h).count();
        lacks.push(format!("nothing for the palette: the copy holds the draw's {n} bones at bone {first}, {same} equal to the hook's; {window} bones of the list's window placed past the copy's end"));
        return Some(b);
    }
    lacks.push(format!("the palette: the copy ({copied} of the view's {} B) holds {held} of the draw's {n} bones at bone {first} ({zero} of them zero, the rest cut): the hook's span stands in, and {window} bones of the list's window past the copy's end", v.bytes));
    if b.len() < end {
        b.resize(end, 0);
    }
    b[off..end].copy_from_slice(bones);
    placed.push((off, end));
    Some(b)
}

/// An instanced draw's palette windows (wheels: bones at the per-instance
/// bases of `T9+0x120`), each at its place in the palette view's
/// bytes when the copy lacks them there (the copy is cut at 16 MB).
fn windows(w: Vec<(u32, Result<Vec<u8>, u32>)>, got: Option<Vec<u8>>, lacks: &mut Vec<String>, placed_at: &mut Vec<(usize, usize)>) -> Option<Vec<u8>> {
    if w.is_empty() {
        return got;
    }
    let mut b = got.unwrap_or_default();
    let (mut placed, mut held) = (0, 0);
    for (base, bytes) in &w {
        let Ok(bytes) = bytes.as_ref() else { continue };
        let (off, end) = (48 * *base as usize, 48 * *base as usize + bytes.len());
        if *base == u32::MAX {
            continue;
        }
        if b.get(off..end) == Some(&bytes[..]) {
            held += 1;
            continue;
        }
        if b.len() < end {
            b.resize(end, 0);
        }
        b[off..end].copy_from_slice(bytes);
        placed_at.push((off, end));
        placed += 1;
    }
    lacks.push(format!("the palette windows: {} records, {held} held by the copy, {placed} placed from their records", w.len()));
    Some(b)
}

/// The bone mask words of a skinned draw (word k = mask16 | rank <<
/// 16 for bones 16k.., the compaction into the draw's palette): read at
/// the frame table's slot 86 from the draw record's +0x74 on (-1: none).
/// The capture cuts slot 86 at 16 MB; the words are its ARENA record. Without
/// them every bone index reads entry 0 of the draw's palette, a zero bone,
/// and the program puts every vertex at the world origin of the draw.
fn arena(pack: &Pack, sink: u32, out: &mut Loaded) {
    let Ok((_, rec)) = pack.slot_bytes(sink, 1, 0) else { return };
    let at = if rec.len() >= 0x78 { le32(&rec, 0x74) } else { u32::MAX };
    if at == u32::MAX {
        return;
    }
    let Some(words) = pack.data.refs_of(sink).find(|r| r.kind == kind::ARENA && reason::has_bytes(r.reason)).and_then(|r| pack.blob(r.hash)) else {
        out.lacks.push(format!("the bone mask words at slot {BINDLESS_SLOT} + {at:#x}: no arena record, the copy's bytes stand"));
        return;
    };
    let Some(LRoot::Table(slots)) = out.roots.get_mut(0) else { return };
    let (at, end) = (at as usize, at as usize + words.len());
    match slots.iter_mut().find(|s| s.0 == BINDLESS_SLOT) {
        Some((_, LSlot::Buffer(_, b))) if b.get(at..end) == Some(&words[..]) => return,
        Some((_, LSlot::Buffer(_, b))) => {
            out.lacks.push(format!("the bone mask words at slot {BINDLESS_SLOT} + {at:#x} (the copy holds {} B): the arena record's {} B stand in", b.len(), words.len()));
            own(&mut out.own, (0, BINDLESS_SLOT), b.len(), (at, end));
            if b.len() < end {
                b.resize(end, 0);
            }
            b[at..end].copy_from_slice(&words);
        }
        _ => {
            out.lacks.push(format!("the bone mask words at slot {BINDLESS_SLOT} + {at:#x} (no view there): the arena record's {} B stand in", words.len()));
            let mut b = vec![0u8; end];
            b[at..].copy_from_slice(&words);
            slots.push((BINDLESS_SLOT, LSlot::Buffer(Kind::Raw, b)));
            slots.sort_by_key(|s| s.0);
        }
    }
}

/// `--bridge-records`: what the mesh program wrote, over all groups:
/// groups that requested vertices, vertices and primitives requested,
/// primitives culled (bit 31), positions not finite, w <= 0, and group 0's
/// first clip position.
pub fn records_summary(rec: &[u8], lay: &xnw_shader::emit::mesh::Layout) -> String {
    let (size, stride) = (lay.record_bytes() as usize, lay.vertex_stride() as usize);
    let Some(row) = lay.position_row() else { return "no SV_Position row".into() };
    let f = |b: &[u8], at: usize| f32::from_bits(le32(b, at));
    let (mut live, mut verts, mut prims, mut culled, mut bad, mut behind) = (0, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut first = None;
    let mut bbox = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for g in rec.chunks_exact(size) {
        let (nv, np) = (le32(g, 0) as usize, le32(g, 4) as usize);
        live += usize::from(nv > 0);
        verts += nv as u64;
        prims += np as u64;
        for v in 0..nv.min(lay.max_vertices as usize) {
            let at = 16 + v * stride + 16 * row as usize;
            let p = [f(g, at), f(g, at + 4), f(g, at + 8), f(g, at + 12)];
            first.get_or_insert(p);
            bad += u64::from(p.iter().any(|x| !x.is_finite()));
            behind += u64::from(p[3] <= 0.0);
            if p[3] > 0.0 {
                let (x, y) = (p[0] / p[3], p[1] / p[3]);
                bbox = [bbox[0].min(x), bbox[1].min(y), bbox[2].max(x), bbox[3].max(y)];
            }
        }
        for p in 0..np.min(lay.max_prims as usize) {
            culled += u64::from(le32(g, lay.prim_offset() as usize + 4 * p) >> 31 == 1);
        }
    }
    format!("records: {live} of {} groups requested vertices; {verts} vertices, {prims} primitives ({culled} culled); positions not finite {bad}, w <= 0 {behind}; NDC box {bbox:?}; first {first:?}", rec.len() / size.max(1))
}

/// The bindless words: at the frame table's slot 86, from the byte offset
/// in the second root constant on (right after the material's constants).
/// A capture that cuts the buffer before them gets the row's albedo heap
/// index at every word, and that texture at its heap slot.
fn bindless(pack: &Pack, sink: u32, row: &Row, texels: &mut Provider, out: &mut Loaded) {
    let Some(at) = out.roots.iter().find_map(|r| if let LRoot::Constants(c) = r { c.get(1).copied() } else { None }) else { return };
    if draw_words(pack, sink, at, out) {
        return;
    }
    let Some(start) = pack.data.refs.iter().find(|r| r.kind == kind::HEAP_SLOT && r.owner == fp::FRAME_SCOPE && r.addr != 0).map(|r| r.addr) else {
        out.lacks.push("the view heap's start (no bindless record in the frame): no texture bound".into());
        return;
    };
    let handles: Vec<u64> = row.handles.iter().copied().filter(|h| *h >= start).collect();
    for &h in &handles {
        match texels(h) {
            Some(t) => out.heap.push((((h - start) / 32) as u32, LSlot::Texture(t))),
            None => out.lacks.push(format!("texels of the view at {h:#x}")),
        }
    }
    out.heap.sort_by_key(|h| h.0);
    out.heap.dedup_by_key(|h| h.0);
    let end = at as usize + 4 * BINDLESS_WORDS;
    for (ri, r) in out.roots.iter_mut().enumerate() {
        let LRoot::Table(slots) = r else { continue };
        let held = slots.iter().position(|s| s.0 == BINDLESS_SLOT);
        let whole = held.is_some_and(|i| matches!(&slots[i].1, LSlot::Buffer(_, b) if b.len() >= end));
        // The frame table is the one whose other slots hold frame buffers; a
        // per-draw table's slot 86 is a neighbour's small record.
        let frame = slots.len() > 64 && !whole;
        if !frame {
            continue;
        }
        let Some(&albedo) = handles.first() else { continue };
        let index = ((albedo - start) / 32) as u32;
        let mut words = vec![0u8; end];
        for k in 0..BINDLESS_WORDS {
            words[at as usize + 4 * k..][..4].copy_from_slice(&index.to_le_bytes());
        }
        out.lacks.push(format!("the bindless words at slot {BINDLESS_SLOT} + {at:#x} (the capture cuts that buffer): the albedo's heap index {index} at {BINDLESS_WORDS} words"));
        match held {
            Some(i) => slots[i].1 = LSlot::Buffer(Kind::Raw, words),
            None => {
                slots.push((BINDLESS_SLOT, LSlot::Buffer(Kind::Raw, words)));
                // No view in the record: the stand-in still shares one buffer
                // with every other row's (ZC3: 64 rows of pk3 f1241, 52 MB of
                // zeros each for 64 B of words).
                out.addrs.push((ri as u16, BINDLESS_SLOT, STAND_IN_AT));
            }
        }
        // The stand-in replaces the copy: only its words are the row's.
        out.own.insert((ri as u16, BINDLESS_SLOT), (0, vec![(at as usize, end)]));
        return;
    }
}

/// v2.4 captures (pk6 on): the draw's own 16 bindless words at the frame
/// table's slot 86 + `at`, and every texture a word names at its heap
/// index. False when the capture has no words for the sink (the caller's
/// stand-in follows). A `lacks` line counts what the words name.
fn draw_words(pack: &Pack, sink: u32, at: u32, out: &mut Loaded) -> bool {
    let Some(dw) = pack.draw_words(sink) else { return false };
    let Ok(words) = dw.words else {
        out.lacks.push(format!("the draw words (reason {}): the stand-in follows", dw.words.err().unwrap_or(0)));
        return false;
    };
    let Some(LRoot::Table(slots)) = out.roots.get_mut(0) else { return false };
    let (from, end) = (at as usize, at as usize + 4 * BINDLESS_WORDS);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    match slots.iter_mut().find(|s| s.0 == BINDLESS_SLOT) {
        Some((_, LSlot::Buffer(_, b))) => {
            own(&mut out.own, (0, BINDLESS_SLOT), b.len(), (from, end));
            if b.len() < end {
                b.resize(end, 0);
            }
            b[from..end].copy_from_slice(&bytes);
        }
        _ => {
            let mut b = vec![0u8; end];
            b[from..].copy_from_slice(&bytes);
            slots.push((BINDLESS_SLOT, LSlot::Buffer(Kind::Raw, b)));
            slots.sort_by_key(|s| s.0);
        }
    }
    let (mut tex, mut buf, mut none, mut missing) = (0, 0, 0, Vec::new());
    for k in 0..BINDLESS_WORDS {
        let Some(v) = dw.view(k) else {
            none += 1;
            continue;
        };
        if out.heap.iter().any(|h| h.0 == v.slot) {
            continue;
        }
        match v.kind {
            view::TEXTURE | view::PLACED => match pack.view_texels_shaped(&v) {
                Ok(t) => {
                    out.heap.push((v.slot, LSlot::Texture(t)));
                    tex += 1;
                }
                Err(why) => {
                    // A texture the program samples must be a texture view:
                    // a 1x1 grey (0.5) stands in.
                    out.heap.push((v.slot, LSlot::Texture(Provided { width: 1, height: 1, format: 28, rows: vec![128, 128, 128, 255], row_bytes: 4, row_count: 1, dim: xnw_world::textures::TEX_2D, depth: 1, mips: 1 })));
                    missing.push(format!("{} ({why})", v.slot));
                }
            },
            _ => buf += 1,
        }
    }
    out.heap.sort_by_key(|h| h.0);
    out.lacks.push(format!("nothing for the bindless words: the draw's 16 words at slot {BINDLESS_SLOT} + {at:#x}; {tex} textures at their heap indices, {buf} buffer views not bound, {none} words name no view (words {words:?}){}", if missing.is_empty() { String::new() } else { format!("; textures without texels (1x1 grey stands in): {}", missing.join(", ")) }));
    true
}

/// v2.4 captures: the sink's sampler heap entries at their indices, as
/// host sampler descriptions (console or D3D12 form); None without them.
pub fn samplers(pack: &Pack, sink: u32, lacks: &mut Vec<String>) -> Option<Vec<(u32, bridge::SamplerDesc)>> {
    let (_, entries) = pack.samplers(sink)?;
    let entries = match entries {
        Ok(e) => e,
        Err(r) => {
            lacks.push(format!("sampler heap entries (reason {r}): default samplers stand in"));
            return None;
        }
    };
    let mut raw = 0;
    let out: Vec<(u32, bridge::SamplerDesc)> = entries
        .iter()
        .filter_map(|e| {
            let d = bridge::sampler_desc(e.form, &e.desc);
            raw += usize::from(d.is_none());
            Some((e.index, d?))
        })
        .collect();
    let top = entries.iter().map(|e| e.index).max().unwrap_or(0);
    let past = entries.iter().filter(|e| e.index >= bridge::SAMPLER_HEAP_MAX).count();
    if past > 0 {
        lacks.push(format!("sampler heap entries past {}: {past} of {} left out (indices up to {top})", bridge::SAMPLER_HEAP_MAX, entries.len()));
    } else {
        lacks.push(format!("nothing for the sampler heap: {} entries, indices up to {top} ({raw} raw, default there)", entries.len()));
    }
    Some(out)
}

/// The translated pair of sink `sink`, the table slots the two programs
/// name (root, slot) with their strides, and the two container hashes.
/// Whether the pixel program may run on an adapter is [`pixel_refused`]'s
/// to say, asked where it is about to run: the mesh half alone is not
/// refused (CARD5 c0: with the refusal here, `--bridgems b31c5ed5` on the
/// card prepared 0 rows, its only pixel program being on the deny list).
pub fn programs(pack: &Pack, sink: u32) -> Result<(bridge::Programs, HashMap<(u16, u32), u32>, [u128; 2]), String> {
    let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).ok_or("no such sink")?.pipeline;
    let stage = |name: &str| pack.data.programs.iter().find(|p| p.pipeline == pipeline && p.stage == name).ok_or_else(|| format!("sink {sink}: the pipeline has no {name} program"));
    let (msr, psr) = (stage("mesh")?, stage("pixel")?);
    let code = |h: u128| pack.blob(h).ok_or_else(|| format!("no container bytes for {h:032x}"));
    let (msc, psc) = (code(msr.container)?, code(psr.container)?);

    // bridgeps's own translation (`XNW_HEAP_NONUNIFORM` included), so that
    // `--bridge-dump` writes the programs a bridgeps row runs (HANG2).
    let (p, used) = xnw_world::bridged::translate(&msc, &psc).map_err(|e| format!("{e} ({:032x}, {:032x})", msr.container, psr.container))?;
    Ok((p, used, [msr.container, psr.container]))
}

/// Why the title's pixel program of the pair `hashes` (mesh, pixel
/// containers) is not run on this adapter by the single-draw bridge,
/// bridgems or the pixel audit (None: it may): it is on the bridge's deny
/// list (`bridge::DENY_HW`), or the pair samples cube images, for which
/// these three bind 2D views and 2D nulls (bridgeps binds a stand-in of the
/// title's dimension).
pub fn pixel_refused(pack: &Pack, hashes: &[u128; 2], software: bool) -> Option<String> {
    if software {
        return None;
    }
    if let Some(why) = bridge::denied(hashes[1], software) {
        return Some(why);
    }
    let (msc, psc) = (pack.blob(hashes[0])?, pack.blob(hashes[1])?);
    (xnw_world::bridged::dims_read(&[&msc, &psc]) & 1 << 9 != 0).then(|| format!("pixel program {:08x} (mesh {:08x}) samples cube images and this path binds 2D views only: not run on the hardware adapter (bridgeps binds the title's dimension)", hashes[1] >> 96, hashes[0] >> 96))
}

/// The table slots a console container's programs name, with the widest
/// stride each is read at, into `used`; `wave` becomes 64 for a mesh
/// program of 64-lane waves.
pub fn named(container: &[u8], used: &mut HashMap<(u16, u32), u32>, wave: &mut u32) -> Result<(), String> {
    let parsed = xnw_shader::Container::parse(container).map_err(|e| e.to_string())?;
    let part = parsed.part(xnw_shader::FourCC::DXIL).ok_or("no DXIL part")?;
    for p in xnw_shader::native_programs(part.data).map_err(|e| e.to_string())? {
        if p.stage == xnw_shader::HwStage::Ms && p.wave64 {
            *wave = 64;
        }
        for u in p.used.iter().filter(|u| u.root == xnw_shader::RootType::Table) {
            let e = used.entry((u16::from(u.slot), u.table_offset_dwords / 8)).or_insert(0);
            *e = (*e).max(if u.kind == xnw_shader::ResourceType::Cbv { NAMED_CBV } else { u.stride });
        }
    }
    Ok(())
}

/// The mesh half of sink `sink`'s pipeline alone, for a depth-only run: the
/// mesh program in compute form, a pass-through vertex shader of position
/// and clip distances, and no pixel program.
pub fn mesh_program(pack: &Pack, container: u128) -> Result<(bridge::Programs, HashMap<(u16, u32), u32>), String> {
    let msc = pack.blob(container).ok_or_else(|| format!("no container bytes for {container:032x}"))?;
    let (ms, layout) = emit::mesh::translate_fallback(&msc, bridge::options()).map_err(|e| format!("mesh program {container:032x}: {e}"))?;
    let vs = emit::mesh::passthrough_vs(&layout, None).map_err(|e| format!("pass-through vertex shader: {e}"))?;
    let (mut used, mut wave) = (HashMap::new(), 32);
    named(&msc, &mut used, &mut wave)?;
    let ps = emit::HostShader { stage: xnw_shader::HwStage::Ps, dxil: Vec::new(), user_data: Vec::new(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false };
    Ok((bridge::Programs { ms, layout, vs, ps, wave, vertex: None }, used))
}

/// Sink `sink`'s vertex program alone for a depth-only plain draw of
/// `vertices` x `instances` (a triangle strip when `strip`): the title's
/// vertex program translated as it is, no pixel program (HAIR1: the strand
/// draws, whose pixel program writes the visibility buffer).
pub fn vertex_program(pack: &Pack, sink: u32, vertices: u32, instances: u32, strip: bool) -> Result<(bridge::Programs, HashMap<(u16, u32), u32>, u128), String> {
    let container = vertex_container(pack, sink)?;
    let vsc = pack.blob(container).ok_or_else(|| format!("no container bytes for {container:032x}"))?;
    let ms = emit::translate_with(&vsc, bridge::options()).map_err(|e| format!("vertex program {container:032x}: {e}"))?;
    let (mut used, mut wave) = (HashMap::new(), 32);
    named(&vsc, &mut used, &mut wave)?;
    let none = || emit::HostShader { stage: xnw_shader::HwStage::Ps, dxil: Vec::new(), user_data: Vec::new(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false };
    let layout = emit::mesh::Layout { max_vertices: 0, max_prims: 0, rows: 1, elements: Vec::new() };
    Ok((bridge::Programs { ms, layout, vs: Vec::new(), ps: none(), wave, vertex: Some((vertices, instances, strip)) }, used, container))
}

/// The container hash of a draw sink's vertex program.
pub fn vertex_container(pack: &Pack, sink: u32) -> Result<u128, String> {
    let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).ok_or("no such sink")?.pipeline;
    Ok(pack.data.programs.iter().find(|p| p.pipeline == pipeline && p.stage == "vertex").ok_or_else(|| format!("sink {sink}: the pipeline has no vertex program"))?.container)
}

/// The G-buffer stage: what a decal row is drawn as for its depth.
const OPAQUE_STAGE: u32 = 70;

/// The pixels one draw covers, each with its depth: (y * width + x, z).
pub type Cover = Vec<(u32, f32)>;

/// One side of the geometry audit as a file: per row, `row`, the pixel
/// count, then (pixel, depth) pairs; all little-endian 32-bit.
pub fn write_side(path: &Path, rows: &[(u32, Cover)]) -> Result<(), String> {
    let mut out = Vec::new();
    for (row, cover) in rows {
        out.extend(row.to_le_bytes());
        out.extend((cover.len() as u32).to_le_bytes());
        for (px, z) in cover {
            out.extend(px.to_le_bytes());
            out.extend(z.to_le_bytes());
        }
    }
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn read_side(path: &Path) -> Result<HashMap<u32, Cover>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut at, mut out) = (0usize, HashMap::new());
    while at + 8 <= b.len() {
        let (row, n) = (le32(&b, at), le32(&b, at + 4) as usize);
        let pairs = b.get(at + 8..at + 8 + 8 * n).ok_or_else(|| format!("{}: row {row} is cut", path.display()))?;
        out.insert(row, pairs.chunks_exact(8).map(|c| (le32(c, 0), f32::from_bits(le32(c, 4)))).collect());
        at += 8 + 8 * n;
    }
    Ok(out)
}

/// Two covers of one draw against each other.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Agreement {
    pub theirs: usize,
    pub ours: usize,
    pub both: usize,
    /// Over the pixels both cover: the median of z theirs - z ours, and the
    /// median and largest of |z theirs - z ours - offset| over the larger z
    /// (reverse Z: the relative difference in distance), `offset` being the
    /// caller's (what the two projections differ by everywhere).
    pub median_dz: f32,
    pub median_rel: f32,
    pub max_rel: f32,
}

impl Agreement {
    /// The draws agree: at most 8 pixels or 2 % of the union are covered by
    /// one side only, and the median relative depth difference is at most
    /// 1e-3. Two empty covers do not say either way (`None`).
    pub fn agrees(&self) -> Option<bool> {
        self.verdict().map(|v| v == "agree")
    }

    /// `agree`, or what differs: `theirs none` / `ours none` (one side
    /// covers nothing), `cover` (the pixel sets), `depth` (the same pixels
    /// at another depth). None: both empty.
    pub fn verdict(&self) -> Option<&'static str> {
        let union = self.theirs + self.ours - self.both;
        let single = union - self.both;
        Some(match (self.theirs, self.ours) {
            (0, 0) => return None,
            (0, _) => "theirs none",
            (_, 0) => "ours none",
            _ if single > 8 && single * 50 > union => "cover",
            _ if self.median_rel > 1e-3 => "depth",
            _ => "agree",
        })
    }
}

/// Both covers sorted by pixel.
pub fn agreement(theirs: &Cover, ours: &Cover, offset: f32) -> Agreement {
    let (mut i, mut j, mut rel, mut dzs) = (0, 0, Vec::new(), Vec::new());
    let mut a = Agreement { theirs: theirs.len(), ours: ours.len(), ..Agreement::default() };
    while i < theirs.len() && j < ours.len() {
        match theirs[i].0.cmp(&ours[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let (t, o) = (theirs[i].1, ours[j].1);
                dzs.push(t - o);
                rel.push((t - o - offset).abs() / t.abs().max(o.abs()).max(f32::MIN_POSITIVE));
                (i, j) = (i + 1, j + 1);
            }
        }
    }
    a.both = rel.len();
    rel.sort_by(f32::total_cmp);
    dzs.sort_by(f32::total_cmp);
    a.median_dz = dzs.get(dzs.len() / 2).copied().unwrap_or(0.0);
    (a.median_rel, a.max_rel) = (rel.get(rel.len() / 2).copied().unwrap_or(0.0), rel.last().copied().unwrap_or(0.0));
    a
}

/// The mean pixel of a cover in a picture `width` wide (x, y); None: empty.
pub fn centroid(cover: &Cover, width: u32) -> Option<(f64, f64)> {
    let n = cover.len() as f64;
    (!cover.is_empty()).then(|| (cover.iter().map(|p| f64::from(p.0 % width)).sum::<f64>() / n, cover.iter().map(|p| f64::from(p.0 / width)).sum::<f64>() / n))
}

/// What kind of draw a list row is, as our pass treats it.
pub fn class(r: &DrawRecord) -> &'static str {
    if r.ped {
        "ped"
    } else if r.palette != NO_PALETTE {
        "skinned"
    } else if r.inst_count != 0 {
        "instanced"
    } else {
        "static"
    }
}

/// Our pass's shader with the pixel's depth as its colour: the bits of
/// `SV_Position.z`, one byte per channel (wireframe and id draws keep
/// their colours).
pub fn depth_shader() -> Result<String, String> {
    const ENTRY: &str = "float4 ps(VsOut i) : SV_Target { return shade(i); }";
    const BITS: &str = "float4 ps(VsOut i) : SV_Target {
    if ((flags & 5) == 0) { uint zb = asuint(i.pos.z); return float4(zb & 255, (zb >> 8) & 255, (zb >> 16) & 255, zb >> 24) / 255.0; }
    return shade(i);
}";
    if RAW_SHADER.matches(ENTRY).count() != 1 {
        return Err("the world pass shader has no single `ps` entry to put the depth view in".into());
    }
    Ok(RAW_SHADER.replace(ENTRY, BITS))
}

/// The rows the audit takes: main-view rows paired 1:1 with a sink, as
/// (row, hook draw, sink); `--bridge-rows a,b,c` or `a-b` and
/// `--bridge-class static,instanced,skinned,ped` narrow them.
pub(crate) fn audit_rows(args: &[String], list: &FrameList, pack: &Pack) -> Result<Vec<(usize, u32, u32)>, String> {
    let draws = pack.list_draws().ok_or("no list-draws.tsv")?;
    let main: Vec<*const DrawRecord> = list.main_draws().map(|d| d as *const DrawRecord).collect();
    let wanted: Option<Vec<(usize, usize)>> = args.iter().position(|a| a == "--bridge-rows").and_then(|i| args.get(i + 1)).map(|v| {
        v.split(',').filter_map(|p| match p.split_once('-') {
            Some((a, b)) => Some((a.trim().parse().ok()?, b.trim().parse().ok()?)),
            None => p.trim().parse().ok().map(|k| (k, k)),
        }).collect()
    });
    let classes: Option<Vec<&str>> = args.iter().position(|a| a == "--bridge-class").and_then(|i| args.get(i + 1)).map(|v| v.split(',').collect());
    Ok(list.draws.iter().enumerate().filter(|(k, r)| main.contains(&(*r as *const DrawRecord)) && wanted.as_ref().is_none_or(|w| w.iter().any(|(a, b)| (*a..=*b).contains(k))) && classes.as_ref().is_none_or(|c| c.contains(&class(r)))).filter_map(|(k, _)| {
        let draw = *draws.get(k)?;
        Some((k, draw, pack.sink_of(draw)?))
    }).collect())
}

fn tsv(path: &Path, lines: &[String]) -> Result<(), String> {
    std::fs::write(path, lines.join("\n") + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// `--bridge-audit ours`: every audited row alone through our pass, its
/// coverage (an id draw) and depth (a depth-bits draw) to `<out>\ours.bin`.
fn audit_ours(args: &[String], dir: &Path, out: &Path, pass: &mut Pass) -> Result<(), String> {
    let (mut list, mut geoms) = load_list(dir)?;
    crate::audit::skin_args(args, dir, &mut list, &mut geoms)?;
    crate::preskin::preskin_arg(args, dir, &mut list, &mut geoms)?;
    let pack = Pack::open(dir)?.ok_or("--bridge-audit needs a v2 capture (pack\\manifest.tsv)")?;
    let rows = audit_rows(args, &list, &pack)?;
    // The title's programs draw with the frame's TAA jitter in their view
    // constants: ours keeps it too (`--no-jitter` drops it, as the live pass does).
    xnw_world::keep_jitter(!args.iter().any(|a| a == "--no-jitter"));
    pass.set_raw_shader(&depth_shader()?)?;
    (pass.sky, pass.water, pass.terrain.opts.on) = (false, false, false);
    let mut decode = |key: &GeomKey| geoms.get(key).map(|(vb, idx)| HostGeometry { vertices: Vec::new(), indices: idx.clone(), raw_vb: Some(vb.clone()), vb_place: 0 });
    let all = std::mem::take(&mut list.draws);
    let (mut sides, mut lines) = (Vec::new(), vec!["row\tclass\tstage\tdrawn\ttriangles\tinstances\tskinned\tpixels".to_string()]);
    let t0 = std::time::Instant::now();
    for (n, &(k, _, _)) in rows.iter().enumerate() {
        // Our defaults instead of the title's state: depth GREATER with write,
        // no blend; a decal as an opaque draw (no depth bias of ours).
        let stage = if DECAL_STAGES.contains(&all[k].stage) { OPAQUE_STAGE } else { all[k].stage };
        list.draws = vec![DrawRecord { state: None, stage, ..all[k].clone() }];
        list.frame = 2 * n as u64 + 1;
        crate::replay::watchdog_arm();
        let st = pass.draw(&list, &mut decode, true).and_then(|st| Ok((st, pass.read_rgba()?)));
        (list.frame, pass.id_mode) = (2 * n as u64 + 2, true);
        let ids = pass.draw(&list, &mut decode, true).and_then(|_| pass.read_rgba());
        pass.id_mode = false;
        crate::replay::watchdog_disarm();
        let ((st, bits), ids) = (st?, ids?);
        // The id draw writes the list index + 1: this one-row list's draw is 1.
        let cover: Cover = ids.chunks_exact(4).zip(bits.chunks_exact(4)).enumerate().filter(|(_, (id, _))| id[..3] == [1, 0, 0]).map(|(i, (_, z))| (i as u32, f32::from_bits(le32(z, 0)))).collect();
        lines.push(format!("{k}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", class(&all[k]), all[k].stage, st.draws, st.triangles, st.instances, st.skinned, cover.len()));
        sides.push((k as u32, cover));
    }
    write_side(&out.join("ours.bin"), &sides)?;
    tsv(&out.join("ours.tsv"), &lines)?;
    println!("bridge audit (ours): {} rows in {:.1} s ({:.1} ms a row) -> {}", rows.len(), t0.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64() * 1e3 / rows.len().max(1) as f64, out.join("ours.bin").display());
    Ok(())
}

/// `--bridge-audit theirs`: every audited row through its own mesh program
/// into a depth target, the covered pixels to `<out>\theirs.bin`.
fn audit_theirs(args: &[String], dir: &Path, out: &Path, host: &xnw_gpu::host::Host, size: (u32, u32)) -> Result<(), String> {

    let (list, geoms) = load_list(dir)?;
    let pack = Pack::open(dir)?.ok_or("--bridge-audit needs a v2 capture (pack\\manifest.tsv)")?;
    let rows = audit_rows(args, &list, &pack)?;
    let mut programs: HashMap<u128, Result<(bridge::Programs, HashMap<(u16, u32), u32>), String>> = HashMap::new();
    let mut texels: Provider = Box::new(|_| None);
    let (mut sides, mut lines) = (Vec::new(), vec!["row\tdraw\tsink\tmesh\tgroups\tseconds\tpixels\tnote".to_string()]);
    // `--bridge-resume`: the rows an earlier run left are kept and not run
    // again; a program that removed the device then is not run at all (a
    // removed device ends a run: the rows after it wait for the next one).
    // With `--bridge-retry` the rows that failed then are run again.
    let retry = args.iter().any(|a| a == "--bridge-retry");
    let (mut removes, mut lost): (Vec<u128>, bool) = (Vec::new(), false);
    if args.iter().any(|a| a == "--bridge-resume") {
        let kept = read_side(&out.join("theirs.bin")).unwrap_or_default();
        for l in std::fs::read_to_string(out.join("theirs.tsv")).unwrap_or_default().lines().skip(1) {
            let c: Vec<&str> = l.split('\t').collect();
            let (Some(row), Some(hash)) = (c.first().and_then(|r| r.parse::<u32>().ok()), c.get(3).and_then(|h| u128::from_str_radix(h, 16).ok())) else { continue };
            if retry && c.get(7).is_some_and(|n| !n.is_empty()) {
                continue;
            }
            if c.get(7).is_some_and(|n| n.contains("device removed")) && !removes.contains(&hash) {
                removes.push(hash);
            }
            if let Some(cover) = kept.get(&row) {
                sides.push((row, cover.clone()));
            } else if c.get(7).is_some_and(|n| n.is_empty()) {
                continue;
            }
            lines.push(l.to_owned());
        }
    }
    let done: Vec<u32> = lines.iter().skip(1).filter_map(|l| l.split('\t').next()?.parse().ok()).collect();
    let t0 = std::time::Instant::now();
    for &(k, draw, sink) in rows.iter().filter(|r| !done.contains(&(r.0 as u32))) {
        let row = &list.draws[k];
        let mut one = |programs: &mut HashMap<u128, _>| -> Result<(u128, u32, f64, Cover), (u128, String)> {
            let pipeline = pack.data.sinks.iter().find(|s| s.index == sink).map(|s| s.pipeline).ok_or((0, "no such sink".to_string()))?;
            let hash = pack.data.programs.iter().find(|p| p.pipeline == pipeline && p.stage == "mesh").map(|p| p.container).ok_or((0, "the pipeline has no mesh program".to_string()))?;
            if removes.contains(&hash) {
                return Err((hash, "not run: this program removed the device on an earlier row".into()));
            }
            if lost {
                return Err((hash, String::new()));
            }
            let (p, used) = programs.entry(hash).or_insert_with(|| mesh_program(&pack, hash)).as_ref().map_err(|e| (hash, e.clone()))?;
            if let Some(gap) = bridge::gap(&host.caps, &p.needs()) {
                return Err((hash, gap));
            }
            let mut loaded = load(&pack, sink, &Row { handles: Vec::new(), ..Row::with_palette(row, &list, &pack, draw) }, geoms.get(&row.geom).map(|g| &g.0[..]), used, &mut texels);
            if !args.iter().any(|a| a == "--bridge-visible") {
                loaded.direct_meshlets();
            }
            // `--bridge-lacks`: what each row's record lacked, and what stood in.
            if args.iter().any(|a| a == "--bridge-lacks") {
                for l in &loaded.lacks {
                    println!("  row {k}: lacks {l}");
                }
            }
            if loaded.groups == 0 {
                return Err((hash, "no group count".into()));
            }
            let b = bridge::bind(&p.ms, &p.ps, &loaded.draw()).map_err(|e| (hash, e))?;
            let target = bridge::Target { width: size.0, height: size.1, depth: Some((true, 5)), limit: if host.caps.software { 120.0 } else { 1.0 }, depth_only: true, records: args.iter().any(|a| a == "--bridge-records") };
            crate::replay::watchdog_arm();
            let frame = bridge::run(host, p, &b, &target);
            crate::replay::watchdog_disarm();
            let frame = frame.map_err(|e| (hash, e))?;
            if !frame.records.is_empty() {
                println!("  row {k}: {}", records_summary(&frame.records, &p.layout));
            }
            Ok((hash, loaded.groups, frame.seconds, frame.depth.iter().enumerate().filter(|(_, z)| **z != frame.far).map(|(i, z)| (i as u32, *z)).collect()))
        };
        match one(&mut programs) {
            Ok((hash, groups, seconds, cover)) => {
                lines.push(format!("{k}\t{draw}\t{sink}\t{hash:032x}\t{groups}\t{seconds:.3}\t{}\t", cover.len()));
                sides.push((k as u32, cover));
            }
            Err((_, e)) if e.is_empty() => {}
            Err((hash, e)) => {
                lines.push(format!("{k}\t{draw}\t{sink}\t{hash:032x}\t0\t0\t-\t{}", e.replace(['\t', '\n'], " ")));
                // SAFETY: the device call takes no arguments.
                if unsafe { host.device.GetDeviceRemovedReason() }.is_err() {
                    lost = true;
                }
            }
        }
    }
    write_side(&out.join("theirs.bin"), &sides)?;
    tsv(&out.join("theirs.tsv"), &lines)?;
    println!("bridge audit (theirs): {} rows, {} drawn, {} mesh programs, in {:.1} s on {} -> {}", rows.len(), sides.len(), programs.len(), t0.elapsed().as_secs_f64(), host.caps.adapter, out.join("theirs.bin").display());
    Ok(())
}

/// `--bridge-audit table`: the two sides joined per row to
/// `<out>\audit.tsv`, with the summary by class and by mesh program.
fn audit_table(out: &Path, width: u32) -> Result<(), String> {
    let (theirs, ours) = (read_side(&out.join("theirs.bin"))?, read_side(&out.join("ours.bin"))?);
    let read = |name: &str| -> Result<Vec<Vec<String>>, String> { Ok(std::fs::read_to_string(out.join(name)).map_err(|e| format!("{name}: {e}"))?.lines().skip(1).map(|l| l.split('\t').map(str::to_owned).collect()).collect()) };
    let ours_rows: HashMap<u32, Vec<String>> = read("ours.tsv")?.into_iter().filter_map(|c| Some((c.first()?.parse().ok()?, c))).collect();
    let mut lines = vec!["row\tdraw\tsink\tclass\tstage\tmesh\tgroups\ttriangles\tinstances\ttheirs\tours\tboth\tmedian_dz\tmedian_rel\tmax_rel\tdx\tdy\tverdict\tnote".to_string()];
    // What the two projections differ by in depth everywhere: the median
    // over the draws of each draw's median z theirs - z ours.
    let mut offsets: Vec<f32> = theirs.iter().filter_map(|(row, t)| Some(agreement(t, ours.get(row)?, 0.0)).filter(|a| a.both > 0).map(|a| a.median_dz)).collect();
    offsets.sort_by(f32::total_cmp);
    let offset = offsets.get(offsets.len() / 2).copied().unwrap_or(0.0);
    println!("bridge audit: depth offset theirs - ours {offset:.4e} (median over {} draws; smallest {:.4e}, largest {:.4e}), taken out of the depth comparison", offsets.len(), offsets.first().copied().unwrap_or(0.0), offsets.last().copied().unwrap_or(0.0));
    // Rows per verdict and the first row that does not agree, by class and by class and program.
    let mut sums: [std::collections::BTreeMap<String, (std::collections::BTreeMap<&str, u32>, String)>; 2] = Default::default();
    let mut shifts: Vec<(f64, f64)> = Vec::new();
    for t in read("theirs.tsv")? {
        let Some(row) = t.first().and_then(|r| r.parse::<u32>().ok()) else { continue };
        let Some(o) = ours_rows.get(&row) else { continue };
        let field = |c: &[String], i: usize| c.get(i).cloned().unwrap_or_default();
        let (class, mesh) = (field(o, 1), field(&t, 3));
        let a = theirs.get(&row).zip(ours.get(&row)).map(|(t, o)| agreement(t, o, offset));
        // Our pass left the row out (a stage or class it does not draw): no comparison.
        let skipped = field(o, 3) == "0";
        let verdict = match a.map(|a| a.verdict()) {
            None => "not run",
            Some(_) if skipped => "ours skips",
            Some(None) => "empty",
            Some(Some(v)) => v,
        };
        let a = a.unwrap_or(Agreement { ours: ours.get(&row).map_or(0, Vec::len), ..Agreement::default() });
        // Where theirs lies from ours on the screen: the difference of the two covers' mean pixels.
        let shift = theirs.get(&row).and_then(|t| centroid(t, width)).zip(ours.get(&row).and_then(|o| centroid(o, width))).map(|(t, o)| (t.0 - o.0, t.1 - o.1));
        if let Some(sh) = shift.filter(|_| a.both >= 100 && a.both * 10 >= (a.theirs + a.ours - a.both) * 8) {
            shifts.push(sh);
        }
        lines.push(format!("{row}\t{}\t{}\t{class}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3e}\t{:.3e}\t{:.3e}\t{}\t{}\t{verdict}\t{}", field(&t, 1), field(&t, 2), field(o, 2), &mesh[..mesh.len().min(8)], field(&t, 4), field(o, 4), field(o, 5), a.theirs, a.ours, a.both, a.median_dz, a.median_rel, a.max_rel, shift.map_or("-".into(), |s| format!("{:.2}", s.0)), shift.map_or("-".into(), |s| format!("{:.2}", s.1)), field(&t, 7)));
        for (sum, key) in sums.iter_mut().zip([class.clone(), format!("{class} {}", &mesh[..mesh.len().min(8)])]) {
            let e = sum.entry(key).or_default();
            *e.0.entry(verdict).or_default() += 1;
            if e.1.is_empty() && !matches!(verdict, "agree" | "empty" | "ours skips") {
                e.1 = if verdict == "not run" { format!("row {row}: {}", field(&t, 7)) } else { format!("row {row} ({verdict}): theirs {} ours {} both {} px, depth median rel {:.2e}, max {:.2e}", a.theirs, a.ours, a.both, a.median_rel, a.max_rel) };
            }
        }
    }
    tsv(&out.join("audit.tsv"), &lines)?;
    if !shifts.is_empty() {
        let median = |pick: fn(&(f64, f64)) -> f64| {
            let mut v: Vec<f64> = shifts.iter().map(pick).collect();
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[v.len() / 10], v[v.len() - 1 - v.len() / 10])
        };
        let (x, y) = (median(|s| s.0), median(|s| s.1));
        println!("bridge audit: screen shift theirs - ours over {} draws of 100 px or more that overlap by 80 %: x median {:.2} px (p10 {:.2}, p90 {:.2}), y median {:.2} px (p10 {:.2}, p90 {:.2})", shifts.len(), x.0, x.1, x.2, y.0, y.1, y.2);
    }
    for (name, sum) in ["class", "class and mesh program"].iter().zip(&sums) {
        for (key, (counts, first)) in sum {
            println!("bridge audit by {name}: {key}: {}{}", counts.iter().map(|(v, n)| format!("{v} {n}")).collect::<Vec<_>>().join(", "), if first.is_empty() { String::new() } else { format!("; first: {first}") });
        }
    }
    println!("bridge audit: {} rows -> {}", lines.len() - 1, out.join("audit.tsv").display());
    Ok(())
}

/// `--bridge-audit ours|theirs|table --bridge-dir <dir>` of `xnw
/// world-render`: per audited row, the pixels and depth of the title's own
/// mesh program against our pass's. The two sides may come from two
/// processes (two adapters); `table` joins what both left in the directory.
pub fn audit_cli(args: &[String], dir: &Path, side: &str, host: &xnw_gpu::host::Host, pass: &mut Pass) -> Result<(), String> {
    let out = args.iter().position(|a| a == "--bridge-dir").and_then(|i| args.get(i + 1)).map(Path::new).ok_or("--bridge-audit needs --bridge-dir <dir>")?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    match side {
        "ours" => audit_ours(args, dir, out, pass),
        "theirs" => audit_theirs(args, dir, out, host, (pass.width, pass.height)),
        "table" => audit_table(out, pass.width),
        "pixels" => crate::bridge_ps::audit_pixels(args, dir, out, host, pass),
        _ => Err("--bridge-audit ours|theirs|table|pixels".into()),
    }
}

/// `--bridge <list index>` of `xnw world-render`.
pub fn cli(args: &[String], dir: &Path, index: &str, host: &xnw_gpu::host::Host, pass: &mut Pass) -> Result<(), String> {
    let k: usize = index.parse().map_err(|_| "--bridge <list index>: a row of list.tsv")?;
    let (mut list, geoms) = load_list(dir)?;
    let pack = Pack::open(dir)?.ok_or("--bridge needs a v2 capture (pack\\manifest.tsv)")?;
    let row = list.draws.get(k).cloned().ok_or_else(|| format!("list row {k} of {}", list.draws.len()))?;
    let draw = *pack.list_draws().ok_or("no list-draws.tsv")?.get(k).ok_or("list-draws.tsv is shorter than list.tsv")?;
    let sink = pack.sink_of(draw).ok_or_else(|| format!("hook draw {draw} is not paired 1:1 with a sink"))?;
    let (p, used, hashes) = programs(&pack, sink)?;
    if let Some(why) = pixel_refused(&pack, &hashes, host.caps.software) {
        return Err(format!("bridge: {why}"));
    }
    let triangles = geoms.get(&row.geom).map_or(0, |g| g.1.len() / 3);
    println!("bridge: list row {k} = hook draw {draw} = sink {sink}: stage {}, {triangles} triangles; mesh {:032x} as compute (records of {} B: {} vertices x {} B, {} primitives), pixel {:032x}", row.stage, hashes[0], p.layout.record_bytes(), p.layout.max_vertices, p.layout.vertex_stride(), p.layout.max_prims, hashes[1]);
    for (name, c) in [("mesh (compute form)", &p.ms.dxil), ("pass-through vertex", &p.vs), ("pixel", &p.ps.dxil)] {
        let valid = emit::validator::Validator::load(emit::validator::Validator::default_path().as_deref()).and_then(|mut v| v.validate(&mut c.clone()));
        let needs = bridge::needs(c, (name.starts_with("mesh")).then_some(p.wave)).map_or("not a DXIL container".into(), |n| n.text());
        println!("  {name}: {} B, {}; needs {needs}", c.len(), valid.map_or_else(|e| format!("NOT valid: {e}"), |()| "valid".into()));
    }
    for (name, s) in [("mesh", &p.ms), ("pixel", &p.ps)] {
        println!("  {name} binds: user data {:?}; root CBVs {:?}, root SRVs {:?}; stride table {}, heap bounds {}, static samplers {}", s.user_data.iter().map(|u| format!("ud{}=root{}.{}{}", u.index, u.slot, u.dword, if u.root == xnw_shader::RootType::Table { "t" } else { "c" })).collect::<Vec<_>>(), s.root_cbvs, s.root_srvs, s.reads_stride_table(), s.reads_heap_bounds(), s.static_samplers.len());
    }
    if let Some(d) = args.iter().position(|a| a == "--bridge-dump").and_then(|i| args.get(i + 1)) {
        for (name, c) in [("ms.dxil", &p.ms.dxil), ("vs.dxil", &p.vs), ("ps.dxil", &p.ps.dxil)] {
            std::fs::write(Path::new(d).join(name), c).map_err(|e| format!("{d}: {e}"))?;
        }
    }
    let counts = Arc::new([const { AtomicU64::new(0) }; 3]);
    let mut texels = pack.texel_provider(counts.clone());
    let mut loaded = load(&pack, sink, &Row::with_palette(&row, &list, &pack, draw), geoms.get(&row.geom).map(|g| &g.0[..]), &used, &mut texels);
    if !args.iter().any(|a| a == "--bridge-visible") {
        loaded.direct_meshlets();
    }
    for l in &loaded.lacks {
        println!("  the record lacks {l}");
    }
    let d = loaded.draw();
    let b = bridge::bind(&p.ms, &p.ps, &d)?;
    println!("  bound: {} groups; view heap {} slots ({} buffer views, {} textures at {:?}, {} B of buffers); {} sampler slots; mesh constants {:x?}; pixel constants {:x?}", b.groups, b.views.heap_slots, b.views.slots.len(), b.textures.len(), b.textures.iter().map(|t| t.0).collect::<Vec<_>>(), b.views.slots.iter().map(|s| s.1.bytes.len()).sum::<usize>(), b.sampler_slots, b.mesh.constants, b.pixel.constants);

    // Our pass's albedo for the same draw, and the pixels it covers.
    // `--bridge-only`: without our pass (the title's programs alone; target 0 to `--out`).
    let only = args.iter().any(|a| a == "--bridge-only");
    let (mut ours, mut mask): (Vec<u8>, Vec<bool>) = (Vec::new(), Vec::new());
    list.draws = vec![row.clone()];
    if !only {
    pass.set_raw_shader(&format!("#define LAB_VIEW 6\n{RAW_SHADER}"))?;
    (pass.sky, pass.water, pass.terrain.opts.on) = (false, false, false);
    pass.tex.opts.mode = TexMode::Provided;
    pass.tex.set_provider(Some(pack.texel_provider(counts)));
    let mut decode = |key: &GeomKey| geoms.get(key).map(|(vb, idx)| HostGeometry { vertices: Vec::new(), indices: idx.clone(), raw_vb: Some(vb.clone()), vb_place: 0 });
    list.frame = 1;
    pass.draw(&list, &mut decode, true)?;
    ours = pass.read_rgba()?;
    (list.frame, pass.id_mode) = (2, true);
    let ids = pass.draw(&list, &mut decode, true).and_then(|_| pass.read_rgba());
    pass.id_mode = false;
    // The id pass writes the list index + 1: this one-row list's draw is 1.
    mask = ids?.chunks_exact(4).map(|px| px[..3] == [1, 0, 0]).collect();
    let covered = mask.iter().filter(|m| **m).count();
    let mean: [f64; 3] = std::array::from_fn(|c| mask.iter().enumerate().filter(|m| *m.1).map(|(i, _)| f64::from(ours[4 * i + c]) / 255.0).sum::<f64>() / covered.max(1) as f64);
    println!("  our pass: the draw covers {covered} pixels of {}x{}; mean albedo ({:.3} {:.3} {:.3})", pass.width, pass.height, mean[0], mean[1], mean[2]);
    }

    if let Some(gap) = bridge::gap(&host.caps, &p.needs()) {
        println!("bridge: not run: {gap}");
        return Ok(());
    }
    let depth = row.state.filter(|s| s.depth_enable).map(|s| (s.depth_write, i32::from(s.depth_func)));
    let target = bridge::Target { width: pass.width, height: pass.height, depth, limit: if host.caps.software { 120.0 } else { 1.0 }, depth_only: false, records: false };
    crate::replay::watchdog_arm();
    let frame = bridge::run(host, &p, &b, &target);
    crate::replay::watchdog_disarm();
    let frame = frame?;
    let theirs = frame.rt0.iter().filter(|px| **px != bridge::CLEAR).count();
    if only {
        let lit: Vec<&[f32; 4]> = frame.rt0.iter().filter(|px| **px != bridge::CLEAR).collect();
        let mean: [f32; 4] = std::array::from_fn(|c| lit.iter().map(|px| px[c]).sum::<f32>() / lit.len().max(1) as f32);
        let deep = frame.depth.iter().filter(|z| **z != frame.far).count();
        println!("bridge: drawn on {} in {:.3} s into {} targets; target 0 covers {theirs} pixels of {}x{}, the depth target {deep}; mean of those ({:.3} {:.3} {:.3} {:.3})", host.caps.adapter, frame.seconds, frame.targets, pass.width, pass.height, mean[0], mean[1], mean[2], mean[3]);
        if let Some(out) = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)) {
            let rgba: Vec<u8> = frame.rt0.iter().flat_map(|px| px.map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)).collect();
            crate::replay::write_png(Path::new(out), pass.width, pass.height, &rgba).map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    let (n, diff) = bridge::compare(&ours, &mask, &frame.rt0);
    println!("bridge: drawn on {} in {:.3} s into {} targets; target 0 covers {theirs} pixels, {n} of them ours too; mean absolute difference of target 0 and our albedo over those: {diff:.4} (0..1)", host.caps.adapter, frame.seconds, frame.targets);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fp::{FramePack, Manifest, Ref, Sink, SlotView};

    /// Two covers of a made-up draw: three shared pixels (one a depth step
    /// apart), one pixel each side alone; the file form keeps them.
    #[test]
    fn covers_against_each_other() {
        let theirs: Cover = vec![(3, 0.5), (4, 0.25), (9, 0.125), (20, 0.1)];
        let ours: Cover = vec![(1, 0.5), (3, 0.5), (4, 0.26), (9, 0.125)];
        let a = agreement(&theirs, &ours, 0.0);
        assert_eq!((a.theirs, a.ours, a.both), (4, 4, 3));
        assert!(a.median_dz == 0.0 && a.median_rel == 0.0 && (a.max_rel - 0.01 / 0.26).abs() < 1e-6, "{a:?}");
        assert_eq!(a.agrees(), Some(true), "two single pixels of five are within 8");
        assert_eq!(agreement(&Vec::new(), &Vec::new(), 0.0).agrees(), None);
        let far: Cover = (0..100).map(|p| (p, 0.5)).collect();
        let shifted: Cover = (50..150).map(|p| (p, 0.5)).collect();
        assert_eq!(agreement(&far, &shifted, 0.0).verdict(), Some("cover"));
        assert_eq!(agreement(&far, &Vec::new(), 0.0).verdict(), Some("ours none"));
        let deeper: Cover = (0..100).map(|p| (p, 0.4)).collect();
        assert_eq!(agreement(&far, &deeper, 0.0).verdict(), Some("depth"));
        // A difference the two projections have everywhere is the caller's to take out.
        assert!((agreement(&far, &deeper, 0.0).median_dz - 0.1).abs() < 1e-6);
        assert_eq!(agreement(&far, &deeper, 0.1).agrees(), Some(true));
        assert_eq!((centroid(&vec![(0, 0.0), (10 + 2 * 8, 0.0)], 8), centroid(&Vec::new(), 8)), (Some((1.0, 1.5)), None));
        let path = std::env::temp_dir().join(format!("xnw-bridge-side-{}.bin", std::process::id()));
        write_side(&path, &[(7, theirs.clone()), (2263, Vec::new()), (9, ours.clone())]).unwrap();
        let back = read_side(&path).unwrap();
        assert_eq!((back.len(), &back[&7], &back[&9], back[&2263].len()), (3, &theirs, &ours, 0));
        let _ = std::fs::remove_file(path);
    }

    /// The depth view goes into the world pass shader's pixel entry.
    #[test]
    fn depth_view_in_the_pass_shader() {
        let s = depth_shader().unwrap();
        assert!(s.len() > RAW_SHADER.len() && s.matches("uint zb = asuint(i.pos.z)").count() == 1);
    }

    /// A made-up v2 frame: sink 0 with two constants, a root view the
    /// capture could not read, a frame table (a small buffer, a cut
    /// bindless buffer at slot 86 among 70 views) and a per-draw table (the
    /// meshlet buffer by its GPU address, a constant buffer, a large view
    /// nobody names). The record comes back as a binding input, with the
    /// stand-ins named. ALIAS1: by the default pair and by a moved one.
    #[test]
    fn a_record_as_a_binding_input() {
        record_as_binding_input(Pair::DEFAULT, "default");
        record_as_binding_input(Pair { cpu: 0x1_0000_0000, gpu: 0x6_6720_0000, len: 0x4_0000_0000 }, "moved");
    }

    /// The made-up frame of [`a_record_as_a_binding_input`] at the
    /// addresses of `pair`'s views.
    fn record_as_binding_input(pair: Pair, tag: &str) {
        let (mesh_at, vb_at) = (pair.cpu + 0x5000, pair.cpu + 0x8000);
        let root = std::env::temp_dir().join(format!("xnw-frame-bridge-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("frame-0001");
        let blobs = fp::blob_dir(&dir);
        let put = |h: u128, b: &[u8]| fp::write_blob(&blobs, h, b).unwrap();
        let buf = |slot, addr, bytes| SlotView { slot, kind: view::PLACED_BUFFER, addr, bytes, ..SlotView::default() };
        let mut frame: Vec<SlotView> = (0..70).map(|s| buf(s, 0x1000 + u64::from(s), 8)).collect();
        frame.push(buf(BINDLESS_SLOT, 0x2c89_0000, 1 << 20));
        put(1, &[0x96, 0x60, 0, 0x10, 0x40, 0, 0, 0]);
        put(2, &fp::encode_slots(&frame));
        put(3, &[9; 8]);
        put(4, &fp::encode_slots(&[buf(2, pair.to_gpu(mesh_at), 16), SlotView { slot: 0, kind: view::CONSTANT, addr: 0x6000, bytes: 32, ..SlotView::default() }, buf(5, 0x7000, 1 << 20)]));
        put(5, &[3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        put(6, &[7; 32]);
        put(7, &vec![1; SMALL + 4]);
        put(8, &fp::encode_slots(&[SlotView { slot: 300, kind: view::PLACED, addr: 0x9000, bytes: 64, ..SlotView::default() }]));
        let r = |kind, slot, hash| Ref { owner: 0, kind, slot, addr: 0, len: 0, reason: 0, hash };
        let mut refs = vec![r(kind::ROOTS, 4, 1), Ref { addr: pair.to_gpu(vb_at), reason: reason::IMPLAUSIBLE, ..r(kind::ROOT_VIEW, 3, 0) }, Ref { addr: 0x10_0000, ..r(kind::TABLE, 0, 2) }, Ref { addr: 0x20_0000, ..r(kind::TABLE, 1, 4) }, r(kind::BUFFER, 96 + 2, 5), r(kind::CBV, 96, 6), r(kind::BUFFER, 96 + 5, 7), Ref { owner: fp::FRAME_SCOPE, addr: 0x4_0000, ..r(kind::HEAP_SLOT, 0, 8) }];
        refs.extend((0..70).map(|s| r(kind::BUFFER, s, 3)));
        refs.push(Ref { reason: reason::CUT, ..r(kind::BUFFER, BINDLESS_SLOT as u16, 3) });
        let mut p = FramePack { manifest: Manifest { version: fp::VERSION, frame: 1, ..Manifest::default() }, refs, sinks: vec![Sink { index: 0, thread: 1, hook_draw: Some(0), op: fp::op::INDIRECT_MESH, count: 1, pipeline: 1 }], ..FramePack::default() };
        p.write(&dir).unwrap();
        let pack = Pack::open(&dir).unwrap().unwrap();
        let row = Row { mesh_at, vb_at, handles: vec![0x4_0000 + 32 * 300, 0], palette: None, window: None, pair };
        let used: HashMap<(u16, u32), u32> = [((1, 2), 0), ((1, 0), 0), ((0, BINDLESS_SLOT), 0)].into();
        let mut texels: Provider = Box::new(|h| (h == 0x4_0000 + 32 * 300).then(|| Provided { width: 1, height: 1, format: 28, rows: vec![0; 4], row_bytes: 4, row_count: 1, dim: xnw_world::textures::TEX_2D, depth: 1, mips: 1 }));
        let l = load(&pack, 0, &row, Some(&[5; 24]), &used, &mut texels);
        let d = l.draw();
        assert_eq!(d.groups, 3);
        assert!(matches!(&d.roots[3], Root::View(b) if b.len() == 24), "the list's vertex buffer at the root view");
        assert!(matches!(&d.roots[4], Root::Constants(c) if c == &[0x1000_6096, 0x40]));
        let Root::Table(per_draw) = &d.roots[1] else { panic!("root 1") };
        assert_eq!(per_draw.iter().map(|s| s.0).collect::<Vec<_>>(), [0, 2], "the unnamed large view is left out");
        assert!(matches!(per_draw[0].1, Slot::Buffer(View { kind: Kind::Cbv, bytes }) if bytes.len() == 32));
        let Root::Table(frame) = &d.roots[0] else { panic!("root 0") };
        let Some((_, Slot::Buffer(words))) = frame.iter().find(|s| s.0 == BINDLESS_SLOT) else { panic!("slot 86") };
        assert_eq!((words.bytes.len(), le32(words.bytes, 0x40), le32(words.bytes, 0x40 + 60), le32(words.bytes, 0x3c)), (0x40 + 64, 300, 300, 0));
        assert!(matches!(d.heap[..], [(300, Slot::Texture(_))]));
        for part in ["root view 3", "the group count", "the bindless words at slot 86 + 0x40", "sampler heap entries", "the targets' formats"] {
            assert!(l.lacks.iter().any(|x| x.contains(part)), "{part}: {:?}", l.lacks);
        }
        // The views fold onto the CPU view by the row's pair.
        assert!(l.addrs().iter().any(|(_, v)| v.addr == mesh_at), "{tag}: the meshlet view's title address");
        // Folded by another pair, the meshlet view and the root view are not the row's.
        if pair != Pair::DEFAULT {
            let other = Row { pair: Pair::DEFAULT, ..row };
            let l = load(&pack, 0, &other, Some(&[5; 24]), &used, &mut texels);
            assert_eq!(l.draw().groups, 0, "{tag} by the default pair");
            assert!(l.lacks.iter().any(|x| x.contains("root view 3") && x.contains("nothing stands in")), "{:?}", l.lacks);
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
