//! The frame pack v2 of a captured frame for the
//! replay and the audits: one `load_any(dir)` gives the v1 frame and, when
//! the frame has `pack\manifest.tsv`, its pack. Accessors hand out raw
//! bytes by v1 draw index; decoding stays with the caller (geo's skin
//! maths, the lab's textures).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use xnw_formats::framepack::{self as fp, FramePack, Ref, SlotView, kind, reason};

use crate::load::{LoadedFrame, load_frame};

/// A frame's pack, indexed by v1 draw.
pub struct Pack {
    pub dir: PathBuf,
    pub data: FramePack,
    /// v1 draw -> its sink draws.
    sinks_of: HashMap<u32, Vec<u32>>,
}

/// The v1 frame and its pack (None for a v1 capture).
pub fn load_any(dir: &Path) -> Result<(LoadedFrame, Option<Pack>), String> {
    Ok((load_frame(dir)?, Pack::open(dir)?))
}

impl Pack {
    pub fn open(dir: &Path) -> Result<Option<Pack>, String> {
        let Some(data) = FramePack::open(dir)? else { return Ok(None) };
        let mut sinks_of: HashMap<u32, Vec<u32>> = HashMap::new();
        for s in &data.sinks {
            if let Some(d) = s.hook_draw {
                sinks_of.entry(d).or_default().push(s.index);
            }
        }
        Ok(Some(Pack { dir: dir.to_path_buf(), data, sinks_of }))
    }

    /// A blob's bytes (None: not stored or not found).
    pub fn blob(&self, hash: u128) -> Option<Vec<u8>> {
        (hash != 0).then(|| FramePack::blob(&self.dir, hash)).flatten()
    }

    fn bytes_of(&self, r: &Ref) -> Option<Vec<u8>> {
        reason::has_bytes(r.reason).then(|| self.blob(r.hash)).flatten()
    }

    fn hook_ref(&self, draw: u32, k: u16, slot: u16) -> Option<&Ref> {
        self.data.refs_of(fp::HOOK | draw).find(|r| r.kind == k && r.slot == slot)
    }

    /// The palette spans the draw read at the hook (48 B per bone):
    /// (current, previous), each with its title address.
    pub fn hook_palettes(&self, draw: u32) -> [Option<(u64, Vec<u8>)>; 2] {
        [0u16, 1].map(|slot| self.hook_ref(draw, kind::PALETTE, slot).and_then(|r| Some((r.addr, self.bytes_of(r)?))))
    }

    /// Each list.tsv row's v1 draw (`list-draws.tsv`, written with the list
    /// dump since the pack's slot-bytes fix); None for older frames.
    pub fn list_draws(&self) -> Option<Vec<u32>> {
        let t = std::fs::read_to_string(self.dir.join("list-draws.tsv")).ok()?;
        t.lines().skip(1).map(|l| l.split('\t').nth(1)?.parse().ok()).collect()
    }

    /// The draw's model head (0x40 B) at the hook, with its address.
    pub fn model_head(&self, draw: u32) -> Option<(u64, Vec<u8>)> {
        let r = self.hook_ref(draw, kind::MODEL, 0)?;
        Some((r.addr, self.bytes_of(r)?))
    }

    /// The skinned draw's handler head at the hook (0x400 B from entity+0x28,
    /// v2.1), with its address.
    pub fn handler_head(&self, draw: u32) -> Option<(u64, Vec<u8>)> {
        let r = self.hook_ref(draw, kind::HANDLER, 0)?;
        Some((r.addr, self.bytes_of(r)?))
    }

    /// The draw's sink when it has exactly one (the pairing the exit trusts).
    pub fn sink_of(&self, draw: u32) -> Option<u32> {
        match self.sinks_of.get(&draw).map(Vec::as_slice) {
            Some([s]) => Some(*s),
            _ => None,
        }
    }

    /// A sink's table roots: (root index, handle, resolved slots).
    pub fn tables(&self, sink: u32) -> Vec<(u16, u64, Vec<SlotView>)> {
        self.data.refs_of(sink).filter(|r| r.kind == kind::TABLE).map(|r| (r.slot, r.addr, self.bytes_of(r).map(|b| fp::decode_slots(&b)).unwrap_or_default())).collect()
    }

    /// The bytes behind table slot `slot` (byte offset / 32) of the sink's
    /// root `root`: a buffer view's range (kind 7) or a constant buffer
    /// (kind 3), with the slot's view. `Err` names why there are none.
    pub fn slot_bytes(&self, sink: u32, root: u16, slot: u32) -> Result<(SlotView, Vec<u8>), String> {
        let view = self.tables(sink).into_iter().find(|t| t.0 == root).ok_or_else(|| format!("sink {sink}: no table at root {root}"))?.2.into_iter().find(|v| v.slot == slot).ok_or_else(|| format!("sink {sink} root {root}: slot {slot} empty"))?;
        let at = (u32::from(root) * 96 + slot) as u16;
        let r = self.data.refs_of(sink).find(|r| (r.kind == kind::BUFFER || r.kind == kind::CBV) && r.slot == at).ok_or_else(|| format!("sink {sink} root {root} slot {slot}: no bytes record (view kind {})", view.kind))?;
        let b = self.bytes_of(r).ok_or_else(|| format!("sink {sink} root {root} slot {slot}: reason {}", r.reason))?;
        Ok((view, b))
    }

    /// Every descriptor slot the frame resolved, by its handle (the address
    /// v1 `albedo_desc` and layer columns hold): table slots at handle +
    /// 32 * slot, bindless slots at heap start + 32 * index, v2.1 handles
    /// the hook draws named (HANDLE_VIEW: the handle itself, slot 0).
    pub fn views_by_handle(&self) -> HashMap<u64, SlotView> {
        let mut out = HashMap::new();
        // v2.4: the views a draw's words name, at heap start + 32 * index (DRAW_WORDS slot 1).
        for r in self.data.refs.iter().filter(|r| r.kind == kind::TABLE || r.kind == kind::HANDLE_VIEW || (r.kind == kind::HEAP_SLOT && r.owner == fp::FRAME_SCOPE && r.addr != 0) || (r.kind == kind::DRAW_WORDS && r.slot == 1 && r.addr != 0)) {
            for v in self.bytes_of(r).map(|b| fp::decode_slots(&b)).unwrap_or_default() {
                out.insert(r.addr + 32 * u64::from(v.slot), v);
            }
        }
        out
    }

    /// The frame's texture row for a view (by texel address and bytes).
    pub fn texture(&self, v: &SlotView) -> Option<&fp::Texture> {
        self.data.textures.iter().find(|t| t.at == v.addr && t.len == v.bytes)
    }

    /// The draw's T9-relative slot bytes: the first table root that has
    /// slot `slot` with bytes (the T9 root is not named by the capture;
    /// tbl9+0x20 = slot 1, tbl9+0 = slot 0).
    pub fn draw_slot_bytes(&self, draw: u32, slot: u32) -> Result<(u16, SlotView, Vec<u8>), String> {
        let sink = self.sink_of(draw).ok_or_else(|| format!("draw {draw}: not paired 1:1 with a sink"))?;
        let mut why = Vec::new();
        for (root, _, _) in self.tables(sink) {
            match self.slot_bytes(sink, root, slot) {
                Ok((v, b)) => return Ok((root, v, b)),
                Err(e) => why.push(e),
            }
        }
        Err(why.join("; "))
    }
}

/// One recorded ped pre-skin dispatch (a sink with op `dispatch`).
/// Its inputs are the sink's records: `Pack::tables(sink)`,
/// `Pack::slot_bytes(sink, root, slot)`, `Pack::heap_views(sink)`.
#[derive(Clone, Debug)]
pub struct PreskinDispatch {
    pub sink: u32,
    /// The compute program's container hash (the shader index hash).
    pub program: u128,
    /// The title's 144-B item (0: not found).
    pub item_at: u64,
    pub rec: fp::Dispatch,
}

/// A view the dispatch's constant buffer names by heap index.
#[derive(Clone, Debug)]
pub struct HeapView {
    /// 0, 1, 2: the index at b7+0x40, +0x44, +0x48.
    pub k: u16,
    /// The descriptor handle (the heap index when it did not resolve).
    pub handle: u64,
    pub view: Option<SlotView>,
    /// The view's range (`Err`: the reason code; 0 = no bytes record).
    pub bytes: Result<Vec<u8>, u32>,
}

/// A draw of a pre-skinned ped program and the dispatch that wrote
/// the range it reads.
#[derive(Clone, Debug)]
pub struct PreskinDraw {
    pub sink: u32,
    pub draw: Option<u32>,
    /// rec+0x54 (None: the sink has no 128-B record at a table's slot 0).
    pub word: Option<u32>,
    /// The dispatch sink whose range word names the same range.
    pub dispatch: Option<u32>,
    /// The two output views as the draw's T8 table holds them: T8+0x420
    /// (slot 33; the word's bits 24-25 == 0) and T8+0xa80 (slot 84).
    pub views: [Option<SlotView>; 2],
}

impl Pack {
    /// The frame's recorded dispatches, in sink order.
    pub fn dispatches(&self) -> Vec<PreskinDispatch> {
        let mut out = Vec::new();
        for s in self.data.sinks.iter().filter(|s| s.op == fp::op::DISPATCH) {
            let Some(r) = self.data.refs_of(s.index).find(|r| r.kind == kind::DISPATCH) else { continue };
            let Some(rec) = self.bytes_of(r).and_then(|b| fp::Dispatch::decode(&b)) else { continue };
            let program = self.data.programs.iter().find(|p| p.pipeline == s.pipeline && p.stage == "compute").map_or(0, |p| p.container);
            out.push(PreskinDispatch { sink: s.index, program, item_at: r.addr, rec });
        }
        out
    }

    /// The views behind the heap indices of dispatch sink `sink`.
    pub fn heap_views(&self, sink: u32) -> Vec<HeapView> {
        self.data
            .refs_of(sink)
            .filter(|r| r.kind == kind::HEAP_VIEW)
            .map(|r| {
                let view = self.bytes_of(r).and_then(|b| fp::decode_slots(&b).into_iter().next());
                let bytes = match self.data.refs_of(sink).find(|b| b.kind == kind::HEAP_BYTES && b.slot == r.slot) {
                    Some(b) => self.bytes_of(b).ok_or(b.reason),
                    None => Err(if view.is_some() { 0 } else { r.reason }),
                };
                HeapView { k: r.slot, handle: r.addr, view, bytes }
            })
            .collect()
    }

    /// A draw sink's range word: +0x54 of the 128-B per-draw record
    /// at slot 0 of one of its tables (the T9 record, frame pack kind 43).
    pub fn range_word(&self, sink: u32) -> Option<u32> {
        let b = self.record(sink)?;
        Some(u32::from_le_bytes(b.get(0x54..0x58)?.try_into().ok()?))
    }

    /// A draw sink's own 128-B per-draw record (slot 0 of a table).
    pub fn record(&self, sink: u32) -> Option<Vec<u8>> {
        self.tables(sink).into_iter().filter(|(_, _, views)| views.iter().any(|v| v.slot == 0 && v.bytes == 128)).find_map(|(root, _, _)| self.slot_bytes(sink, root, 0).ok().map(|(_, b)| b))
    }

    /// Every draw sink whose mesh (or vertex) program `is_preskin`
    /// names, joined to the frame's dispatch with the same range: the
    /// draw's rec+0x54 and the dispatch's range word (b7+0x34 = item+136)
    /// agree in `fp::RANGE_MASK` (offset >> 4 and the buffer bits; ranges
    /// are per frame).
    pub fn preskin_join(&self, is_preskin: &dyn Fn(u128) -> bool) -> Vec<PreskinDraw> {
        let by_word: HashMap<u32, u32> = self.dispatches().iter().filter(|d| d.rec.flags & fp::B7_FOUND != 0).map(|d| (d.rec.word & fp::RANGE_MASK, d.sink)).collect();
        let mut pipes: HashMap<u64, bool> = HashMap::new();
        let mut out = Vec::new();
        for s in self.data.sinks.iter().filter(|s| s.op != fp::op::DISPATCH) {
            let ped = *pipes.entry(s.pipeline).or_insert_with(|| self.data.programs.iter().any(|p| p.pipeline == s.pipeline && (p.stage == "mesh" || p.stage == "vertex") && is_preskin(p.container)));
            if !ped {
                continue;
            }
            let word = self.range_word(s.index);
            let tables = self.tables(s.index);
            let t8 = tables.iter().map(|t| &t.2).find(|views| views.iter().any(|v| v.slot == 84));
            let view = |slot: u32| t8.and_then(|views| views.iter().find(|v| v.slot == slot).copied());
            out.push(PreskinDraw { sink: s.index, draw: s.hook_draw, word, dispatch: word.and_then(|w| by_word.get(&(w & fp::RANGE_MASK)).copied()), views: [view(33), view(84)] });
        }
        out
    }

    /// Whether dispatch sink `sink` holds all its inputs: b7 and the
    /// item, bytes behind every constant buffer and every buffer view the
    /// GPU does not write, and each heap view with its bytes. `Err` names
    /// the first thing missing.
    pub fn dispatch_complete(&self, d: &PreskinDispatch) -> Result<(), String> {
        if d.rec.flags & fp::B7_FOUND == 0 {
            return Err("no b7".into());
        }
        if d.rec.flags & fp::ITEM_FOUND == 0 {
            return Err("no item".into());
        }
        if let Some(r) = self.data.refs_of(d.sink).find(|r| matches!(r.kind, kind::TABLE | kind::CBV | kind::BUFFER | kind::ROOT_VIEW | kind::HEAP_BYTES) && !reason::has_bytes(r.reason) && r.reason != reason::GPU_OUTPUT) {
            return Err(format!("{} slot {} reason {}", kind::name(r.kind), r.slot, r.reason));
        }
        let heap = self.heap_views(d.sink);
        if heap.len() < 3 {
            return Err(format!("{} of 3 heap views", heap.len()));
        }
        match heap.iter().find(|h| h.view.is_none() || h.bytes.is_err()) {
            Some(h) => Err(format!("heap view {} reason {}", h.k, h.bytes.as_ref().err().copied().unwrap_or(0))),
            None => Ok(()),
        }
    }

    /// v2.3: each imposter call's instances (slot = the call, as its
    /// IMPOSTER record's), in call order. `Err` is a reason: the record's
    /// own, `reason::NOT_CAPTURED` when its blob is missing, or
    /// `reason::IMPLAUSIBLE` when the blob does not decode.
    pub fn imposter_instances(&self) -> Vec<(u16, Result<fp::ImpInstances, u32>)> {
        let mut out: Vec<(u16, Result<fp::ImpInstances, u32>)> = self
            .data
            .refs_of(fp::FRAME_SCOPE)
            .filter(|r| r.kind == kind::IMP_INST)
            .map(|r| {
                let got = if reason::has_bytes(r.reason) { self.blob(r.hash).ok_or(reason::NOT_CAPTURED).and_then(|b| fp::ImpInstances::decode(&b).ok_or(reason::IMPLAUSIBLE)) } else { Err(r.reason) };
                (r.slot, got)
            })
            .collect();
        out.sort_by_key(|o| o.0);
        out
    }

    /// v2.3: an instanced wheel draw's palette windows: (base, 256 x 48 B
    /// of root 0 slot 5 at bone `base`, or a reason), in the order the
    /// draw's instances name them. The base is `u32::MAX` when the record
    /// failed before one was read (no view, no instance count).
    pub fn palette_windows(&self, sink: u32) -> Vec<(u32, Result<Vec<u8>, u32>)> {
        let pal = self.tables(sink).into_iter().find(|t| t.0 == 0).and_then(|t| t.2.into_iter().find(|v| v.slot == 5));
        self.data
            .refs_of(sink)
            .filter(|r| r.kind == kind::PAL_WINDOW)
            .map(|r| {
                let base = match pal {
                    Some(p) if r.addr >= p.addr && r.addr != 0 && (r.addr - p.addr) % 48 == 0 => u32::try_from((r.addr - p.addr) / 48).unwrap_or(u32::MAX),
                    _ => u32::MAX,
                };
                let got = if reason::has_bytes(r.reason) { self.blob(r.hash).ok_or(reason::NOT_CAPTURED) } else { Err(r.reason) };
                (base, got)
            })
            .collect()
    }
}

/// v2.4: a draw sink's draw words (kind 49).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawWords {
    /// Where the 64 bytes were read (root 0 slot 86's view + root 4 dword 1; 0 when not known).
    pub at: u64,
    /// The 16 words, or a reason.
    pub words: Result<[u32; 16], u32>,
    /// The view heap's first slot (0 when no word lay in it).
    pub heap_start: u64,
    /// The views at the words' heap indices (slot = heap index), or a reason.
    pub views: Result<Vec<SlotView>, u32>,
}

impl DrawWords {
    /// The view word `k` names, if it names one.
    pub fn view(&self, k: usize) -> Option<SlotView> {
        let w = *self.words.as_ref().ok()?.get(k)?;
        self.views.as_ref().ok()?.iter().find(|v| v.slot == w).copied()
    }
}

impl Pack {
    fn bytes_or_reason(&self, r: &Ref) -> Result<Vec<u8>, u32> {
        if reason::has_bytes(r.reason) { self.blob(r.hash).ok_or(reason::NOT_CAPTURED) } else { Err(r.reason) }
    }

    /// v2.4: the draw sink's draw words and their views; None when the sink
    /// has no such record (a capture before v2.4, or a dispatch sink).
    pub fn draw_words(&self, sink: u32) -> Option<DrawWords> {
        let r0 = self.data.refs_of(sink).find(|r| r.kind == kind::DRAW_WORDS && r.slot == 0)?;
        let words = self.bytes_or_reason(r0).and_then(|b| {
            let b = b.get(..fp::DRAW_WORDS_BYTES).ok_or(reason::IMPLAUSIBLE)?;
            Ok(std::array::from_fn(|k| u32::from_le_bytes([b[4 * k], b[4 * k + 1], b[4 * k + 2], b[4 * k + 3]])))
        });
        let r1 = self.data.refs_of(sink).find(|r| r.kind == kind::DRAW_WORDS && r.slot == 1);
        let views = r1.map_or(Err(reason::NULL), |r| self.bytes_or_reason(r).map(|b| fp::decode_slots(&b)));
        Some(DrawWords { at: r0.addr, words, heap_start: r1.map_or(0, |r| r.addr), views })
    }

    /// v2.4: a displacing pre-skin dispatch's maps, by element of its root
    /// 0 slot 8 buffer: (element, the heap index, the view or a reason).
    /// The index is 0 when the record failed before one was read.
    pub fn displacement_maps(&self, sink: u32) -> Vec<(u16, u32, Result<SlotView, u32>)> {
        self.data
            .refs_of(sink)
            .filter(|r| r.kind == kind::DISP_MAP)
            .map(|r| {
                let view = self.bytes_or_reason(r).and_then(|b| fp::decode_slots(&b).into_iter().next().ok_or(reason::IMPLAUSIBLE));
                let index = match (&view, r.reason) {
                    (Ok(v), _) => v.slot,
                    (Err(_), reason::IMPLAUSIBLE) => u32::try_from(r.addr).unwrap_or(0),
                    _ => 0,
                };
                (r.slot, index, view)
            })
            .collect()
    }

    /// v2.4: the sink's bound sampler heap: (its first slot, the slots with
    /// non-zero words, or a reason); None when the sink has no such record.
    pub fn samplers(&self, sink: u32) -> Option<(u64, Result<Vec<fp::SamplerEntry>, u32>)> {
        let r = self.data.refs_of(sink).find(|r| r.kind == kind::SAMPLERS)?;
        Some((r.addr, self.bytes_or_reason(r).and_then(|b| fp::decode_samplers(&b).ok_or(reason::IMPLAUSIBLE))))
    }
}

/// Mip 0 wider or taller than this gives a smaller mip to the replay.
const PROVIDE_MAX: u32 = 1024;

/// One mip of a tiled 2D chain as packed linear rows: (width, height,
/// rows, row bytes, row count), the finest mip not over [`PROVIDE_MAX`].
pub fn linear_mip(layout: &xnw_formats::texture::Layout, tiled: &[u8], width: u32, height: u32, format: u32) -> Option<(u32, u32, Vec<u8>, u32, u32)> {
    let f = xnw_formats::texture::format(format)?;
    let mips = layout.mips.len() as u32;
    let m = (0..mips).find(|m| (width >> m).max(1) <= PROVIDE_MAX && (height >> m).max(1) <= PROVIDE_MAX).unwrap_or(mips.saturating_sub(1));
    let (w, h) = ((width >> m).max(1), (height >> m).max(1));
    let row_bytes = w.div_ceil(f.block) * f.bytes;
    let row_count = h.div_ceil(f.block);
    let mut rows = vec![0u8; (row_bytes * row_count) as usize];
    layout.detile(tiled, m, 0, &mut rows, row_bytes as usize, (row_bytes * row_count) as usize).ok()?;
    Some((w, h, rows, row_bytes, row_count))
}

/// The game's texture files by name (`XNW_GAME_DRIVE`, else `game\`), found once.
fn game_file(name: &str) -> Option<PathBuf> {
    static FILES: std::sync::OnceLock<HashMap<String, PathBuf>> = std::sync::OnceLock::new();
    let files = FILES.get_or_init(|| {
        let mut out = HashMap::new();
        let root = std::env::var_os("XNW_GAME_DRIVE").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("game"));
        let mut stack = vec![root];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("ttpf")) {
                    out.entry(e.file_name().to_string_lossy().to_ascii_lowercase()).or_insert(p);
                }
            }
        }
        out
    });
    files.get(&name.to_ascii_lowercase()).cloned()
}

/// A texture's finest mip (capped) from its pack origin `<file>#<index>`.
fn from_origin(origin: &str) -> Option<xnw_world::textures::Provided> {
    use xnw_formats::ttpf;
    let (name, idx) = origin.rsplit_once('#')?;
    let path = game_file(name)?;
    let mut f = std::fs::File::open(&path).ok()?;
    let len = f.metadata().ok()?.len();
    let t = ttpf::Ttpf::read(&mut f, len).ok()?;
    let tex = *t.textures.get(idx.parse::<usize>().ok()?)?;
    if tex.kind != ttpf::KIND_2D {
        return None;
    }
    let tables = xnw_formats::bcpack::Tables::builtin().ok();
    let span = t.read_span(&mut f, tex.addr, tex.span, tables.as_ref()).ok()?;
    let chain = ttpf::chain(&tex, tex.mips).ok()?;
    let (width, height, rows, row_bytes, row_count) = linear_mip(&chain, &span, tex.width, tex.height, tex.format)?;
    Some(xnw_world::textures::Provided { width, height, format: tex.format, rows, row_bytes, row_count })
}

/// A texture's finest mip (capped) from its stored resident bytes.
fn from_texels(v: &SlotView, tiled: &[u8]) -> Option<xnw_world::textures::Provided> {
    use xnw_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
    let swizzle = match v.layout {
        1 => Swizzle::LINEAR,
        l if l & !0xff == 0x100 => Swizzle(l & 0xff),
        _ => Swizzle::SW_4KB_S,
    };
    let height = v.height.max(1);
    let t = Texture::new(&TextureDesc { dimension: Dimension::Tex2d, width: u32::try_from(v.width).ok()?, height, depth_or_array: u32::from(v.depth.max(1)), mips: u32::from(v.mips), format: v.format, samples: 1, swizzle }).ok()?;
    let layout = &t.planes.first()?.layout;
    // Short of the layout's size (a cut copy): the rest reads as zeros.
    let mut padded;
    let tiled = if (tiled.len() as u64) < layout.size {
        padded = tiled.to_vec();
        padded.resize(layout.size as usize, 0);
        &padded[..]
    } else {
        tiled
    };
    let (width, height, rows, row_bytes, row_count) = linear_mip(layout, tiled, v.width as u32, height, v.format)?;
    Some(xnw_world::textures::Provided { width, height, format: v.format, rows, row_bytes, row_count })
}

impl Pack {
    /// The texels of one texture view (a draw word's view): its pack
    /// origin, else its stored texels; Err says which is missing.
    pub fn view_texels(&self, v: &SlotView) -> Result<xnw_world::textures::Provided, &'static str> {
        let t = self.texture(v).ok_or("no texture row")?;
        if let Some(p) = t.origin.as_deref().and_then(from_origin) {
            return Ok(p);
        }
        let tiled = (t.texels != 0).then(|| FramePack::blob(&self.dir, t.texels)).flatten().ok_or("no texels")?;
        from_texels(v, &tiled).ok_or("texels not decoded")
    }

    /// The replay's texel source ([`TexMode::Provided`]): a slot handle ->
    /// its view -> the texture's pack origin, else its stored texels.
    /// Counts (asked, from origin, from texels) go to `counts`.
    pub fn texel_provider(&self, counts: std::sync::Arc<[std::sync::atomic::AtomicU64; 3]>) -> xnw_world::textures::Provider {
        let views = self.views_by_handle();
        let rows: HashMap<(u64, u64), fp::Texture> = self.data.textures.iter().map(|t| ((t.at, t.len), t.clone())).collect();
        let dir = self.dir.clone();
        Box::new(move |slot| {
            use std::sync::atomic::Ordering::Relaxed;
            counts[0].fetch_add(1, Relaxed);
            let v = views.get(&slot)?;
            let t = rows.get(&(v.addr, v.bytes))?;
            if let Some(p) = t.origin.as_deref().and_then(from_origin) {
                counts[1].fetch_add(1, Relaxed);
                return Some(p);
            }
            let tiled = (t.texels != 0).then(|| FramePack::blob(&dir, t.texels)).flatten()?;
            let p = from_texels(v, &tiled)?;
            counts[2].fetch_add(1, Relaxed);
            Some(p)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fp::{Manifest, Sink, view};

    #[test]
    fn hook_palettes_and_slot_bytes_by_draw() {
        let root = std::env::temp_dir().join(format!("xnw-frame-pack-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("frame-3");
        let blobs = fp::blob_dir(&dir);
        let put = |h: u128, b: &[u8]| fp::write_blob(&blobs, h, b).unwrap();
        let remap: Vec<u8> = (0..16u32).flat_map(|i| (i * 3).to_le_bytes()).collect();
        let slots = fp::encode_slots(&[SlotView { slot: 1, kind: view::BUFFER, addr: 0x5000, bytes: 64, width: 4, ..SlotView::default() }]);
        put(1, &[9; 96]);
        put(2, &[7; 0x40]);
        put(3, &slots);
        put(4, &remap);
        put(5, &fp::encode_slots(&[SlotView { slot: 0, kind: view::PLACED, addr: 0x9000, bytes: 256, ..SlotView::default() }]));
        let r = |owner, kind, slot, hash| Ref { owner, kind, slot, addr: 0x10, len: 0, reason: 0, hash };
        let mut p = FramePack {
            manifest: Manifest { version: fp::VERSION, frame: 3, ..Manifest::default() },
            refs: vec![r(fp::HOOK | 4, kind::PALETTE, 0, 1), Ref { reason: reason::NULL, ..r(fp::HOOK | 4, kind::PALETTE, 1, 0) }, r(fp::HOOK | 4, kind::MODEL, 0, 2), r(0, kind::TABLE, 2, 3), r(0, kind::BUFFER, 2 * 96 + 1, 4), r(fp::FRAME_SCOPE, kind::HANDLE_VIEW, 0, 5)],
            sinks: vec![Sink { index: 0, thread: 1, hook_draw: Some(4), op: fp::op::MESH, count: 1, pipeline: 0 }],
            ..FramePack::default()
        };
        p.write(&dir).unwrap();
        let pack = Pack::open(&dir).unwrap().unwrap();
        let [cur, prev] = pack.hook_palettes(4);
        assert_eq!(cur.unwrap().1, vec![9; 96]);
        assert!(prev.is_none());
        assert_eq!(pack.model_head(4).unwrap().1, vec![7; 0x40]);
        let (root_at, v, b) = pack.draw_slot_bytes(4, 1).unwrap();
        assert_eq!((root_at, v.width, b), (2, 4, remap));
        assert!(pack.draw_slot_bytes(4, 0).is_err());
        assert!(pack.draw_slot_bytes(5, 1).is_err());
        assert_eq!(pack.views_by_handle().get(&(0x10 + 32)).map(|v| v.addr), Some(0x5000));
        // v2.1: a handle a hook draw named resolves as itself.
        assert_eq!(pack.views_by_handle().get(&0x10).map(|v| v.addr), Some(0x9000));
        // A linear 8x8 RGBA8 texture's texels come back as its rows.
        let texels: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
        let v = SlotView { kind: view::PLACED, width: 8, height: 8, depth: 1, mips: 1, format: 28, layout: 1, ..SlotView::default() };
        let p = from_texels(&v, &texels).unwrap();
        assert_eq!((p.width, p.height, p.row_bytes, p.row_count), (8, 8, 32, 8));
        assert_eq!(&p.rows[..32], &texels[..32]);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A dispatch and the draws that read its range are joined by the
    /// range word: two dispatches (buffer A offset 0x1230, buffer B offset
    /// 0x1230), three pre-skinned draws (one per range, with other format
    /// bits; one whose range no dispatch wrote) and a draw of another
    /// program; the dispatch's inputs read back; an incomplete one says why.
    #[test]
    fn dispatch_and_draws_join_by_the_range_word() {
        let root = std::env::temp_dir().join(format!("xnw-frame-ped1-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("frame-0007");
        let blobs = fp::blob_dir(&dir);
        let put = |h: u128, b: &[u8]| fp::write_blob(&blobs, h, b).unwrap();
        let rec = |word: u32| {
            let mut b = vec![0u8; 128];
            b[0x54..0x58].copy_from_slice(&word.to_le_bytes());
            b
        };
        let b7 = |word: u32| {
            let mut b = vec![0u8; 96];
            b[0x34..0x38].copy_from_slice(&word.to_le_bytes());
            b[0x38..0x3C].copy_from_slice(&1000u32.to_le_bytes());
            b
        };
        let (wa, wb) = (0x6000_0123u32, 0x6100_0123u32);
        let disp = |word: u32, flags: u32| fp::Dispatch { groups: [32, 1, 1], flags, b7_slot: 5 * 96 + 8, word, vertices: 1000, b7: b7(word), item: vec![3; 144] }.encode();
        put(10, &disp(wa, fp::B7_FOUND | fp::ITEM_FOUND));
        put(11, &disp(wb, fp::B7_FOUND));
        // Draw tables: T9 (root 2: the 128-B record at slot 0) and T8 (root 1: slots 33 and 84).
        put(20, &fp::encode_slots(&[SlotView { slot: 0, kind: view::BUFFER, addr: 0x7000, bytes: 128, ..SlotView::default() }]));
        put(21, &fp::encode_slots(&[SlotView { slot: 33, kind: view::BUFFER, bytes: 26_214_400, resource: 0xA, ..SlotView::default() }, SlotView { slot: 84, kind: view::BUFFER, bytes: 52_428_800, resource: 0xB, ..SlotView::default() }]));
        put(30, &rec(wa & !0x4000_0000));
        put(31, &rec(wb));
        put(32, &rec(0x6000_0999));
        // Dispatch 4's inputs: a table (palette view, output view), the palette bytes, a heap view with bytes.
        let palette = SlotView { slot: 1, kind: view::BUFFER, addr: 0x8000, bytes: 48, ..SlotView::default() };
        let output = SlotView { slot: 24, kind: view::BUFFER, addr: 0x9000, bytes: 26_214_400, layout: view::UNORDERED, ..SlotView::default() };
        put(40, &fp::encode_slots(&[palette, output]));
        put(41, &[5; 48]);
        put(42, &fp::encode_slots(&[SlotView { slot: 777, kind: view::BUFFER, addr: 0xA000, bytes: 64, ..SlotView::default() }]));
        put(43, &[6; 64]);
        let r = |owner, kind, slot, hash| Ref { owner, kind, slot, addr: 0x10, len: 0, reason: 0, hash };
        let draw = |owner: u32, rec_hash: u128| [r(owner, kind::TABLE, 1, 21), r(owner, kind::TABLE, 2, 20), r(owner, kind::BUFFER, 2 * 96, rec_hash)];
        let mut refs = Vec::new();
        refs.extend(draw(0, 30));
        refs.extend(draw(1, 31));
        refs.extend(draw(2, 32));
        refs.extend(draw(3, 30));
        refs.extend([Ref { addr: 0x1_4000, ..r(4, kind::DISPATCH, 0, 10) }, r(4, kind::TABLE, 4, 40), r(4, kind::BUFFER, 4 * 96 + 1, 41), Ref { reason: reason::GPU_OUTPUT, ..r(4, kind::BUFFER, 4 * 96 + 24, 0) }]);
        refs.extend((0..3u16).flat_map(|k| [r(4, kind::HEAP_VIEW, k, 42), r(4, kind::HEAP_BYTES, k, 43)]));
        refs.extend([Ref { addr: 0, ..r(5, kind::DISPATCH, 0, 11) }, Ref { reason: reason::UNREADABLE, ..r(5, kind::CBV, 5 * 96 + 8, 0) }]);
        let sink = |index: u32, op: u8, pipeline: u64| Sink { index, thread: 0, hook_draw: (op != fp::op::DISPATCH).then_some(index + 100), op, count: 32, pipeline };
        let program = |pipeline: u64, stage: &str, container: u128| fp::Program { pipeline, stage: stage.into(), form: 1, container, ..fp::Program::default() };
        let mut p = FramePack {
            manifest: Manifest { version: fp::VERSION, frame: 7, ..Manifest::default() },
            refs,
            sinks: vec![sink(0, fp::op::MESH, 1), sink(1, fp::op::MESH, 1), sink(2, fp::op::MESH, 1), sink(3, fp::op::MESH, 2), sink(4, fp::op::DISPATCH, 3), sink(5, fp::op::DISPATCH, 3)],
            programs: vec![program(1, "mesh", 0x781), program(1, "pixel", 0x99), program(2, "mesh", 0x555), program(3, "compute", 0x0832)],
            ..FramePack::default()
        };
        p.write(&dir).unwrap();
        let pack = Pack::open(&dir).unwrap().unwrap();
        let ds = pack.dispatches();
        assert_eq!(ds.iter().map(|d| (d.sink, d.program, d.item_at, d.rec.word, d.rec.vertices)).collect::<Vec<_>>(), [(4, 0x0832, 0x1_4000, wa, 1000), (5, 0x0832, 0, wb, 1000)]);
        let joined = pack.preskin_join(&|h| h == 0x781);
        assert_eq!(joined.iter().map(|j| (j.sink, j.draw, j.word, j.dispatch)).collect::<Vec<_>>(), [(0, Some(100), Some(wa & !0x4000_0000), Some(4)), (1, Some(101), Some(wb), Some(5)), (2, Some(102), Some(0x6000_0999), None)]);
        assert_eq!(joined[0].views.map(|v| v.map(|v| (v.bytes, v.resource))), [Some((26_214_400, 0xA)), Some((52_428_800, 0xB))]);
        // The dispatch's inputs: the palette bytes by table slot, the output view without bytes, the heap views.
        assert_eq!(pack.slot_bytes(4, 4, 1).unwrap().1, vec![5; 48]);
        assert!(pack.slot_bytes(4, 4, 24).unwrap_err().contains("reason 10"));
        let heap = pack.heap_views(4);
        assert_eq!(heap.iter().map(|h| (h.k, h.view.map(|v| v.slot), h.bytes.clone())).collect::<Vec<_>>(), (0..3).map(|k| (k, Some(777), Ok(vec![6; 64]))).collect::<Vec<_>>());
        assert_eq!(pack.dispatch_complete(&ds[0]), Ok(()));
        assert_eq!(pack.dispatch_complete(&ds[1]), Err("no item".into()));
        let _ = std::fs::remove_dir_all(root);
    }

    /// v2.3: imposter instances come back per call in call order; a record
    /// without bytes, a missing blob and a short blob each give a reason.
    #[test]
    fn imposter_instances_by_call() {
        let root = std::env::temp_dir().join(format!("xnw-frame-imp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("frame-0009");
        let blobs = fp::blob_dir(&dir);
        let inst = fp::ImpInstances { list: 0x3000, count: 1, n: 1, visible: vec![0], records: vec![4; 64], ..fp::ImpInstances::default() };
        let bytes = inst.encode();
        fp::write_blob(&blobs, 50, &bytes).unwrap();
        fp::write_blob(&blobs, 51, &bytes[..bytes.len() - 4]).unwrap();
        let r = |slot, reason, hash| Ref { owner: fp::FRAME_SCOPE, kind: kind::IMP_INST, slot, addr: 0x3000, len: 0, reason, hash };
        let mut p = FramePack {
            manifest: Manifest { version: fp::VERSION, frame: 9, ..Manifest::default() },
            refs: vec![r(3, 0, 51), r(0, 0, 50), r(1, reason::UNREADABLE, 0), r(2, 0, 52)],
            ..FramePack::default()
        };
        p.write(&dir).unwrap();
        let pack = Pack::open(&dir).unwrap().unwrap();
        let got = pack.imposter_instances();
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), [0, 1, 2, 3]);
        assert_eq!(got[0].1.as_ref().map(|i| i.record(0).map(<[u8]>::to_vec)), Ok(Some(vec![4; 64])));
        assert_eq!((got[1].1.as_ref().err(), got[2].1.as_ref().err(), got[3].1.as_ref().err()), (Some(&reason::UNREADABLE), Some(&reason::NOT_CAPTURED), Some(&reason::IMPLAUSIBLE)));
        let _ = std::fs::remove_dir_all(root);
    }

    /// v2.3: a wheel draw's palette windows read back with their bases
    /// (from the root 0 slot 5 view's address); failed ones keep a reason.
    #[test]
    fn palette_windows_name_their_base() {
        let root = std::env::temp_dir().join(format!("xnw-frame-palwin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("frame-0011");
        let blobs = fp::blob_dir(&dir);
        let pal = 0x2_0000_0000u64;
        fp::write_blob(&blobs, 60, &fp::encode_slots(&[SlotView { slot: 5, kind: view::BUFFER, addr: pal, bytes: 48 << 20, ..SlotView::default() }])).unwrap();
        fp::write_blob(&blobs, 61, &[7; 48 * 256]).unwrap();
        let r = |kind, slot, addr, reason, hash| Ref { owner: 2, kind, slot, addr, len: 0, reason, hash };
        let mut p = FramePack {
            manifest: Manifest { version: fp::VERSION, frame: 11, ..Manifest::default() },
            refs: vec![r(kind::TABLE, 0, 0x10, 0, 60), r(kind::PAL_WINDOW, 0, pal + 48 * 438_333, 0, 61), r(kind::PAL_WINDOW, 1, pal + 48 * 9, reason::CUT, 62), r(kind::PAL_WINDOW, 2, 0, reason::NULL, 0)],
            sinks: vec![Sink { index: 2, thread: 0, hook_draw: Some(1), op: fp::op::INDIRECT_MESH, count: 4, pipeline: 0 }],
            ..FramePack::default()
        };
        p.write(&dir).unwrap();
        let pack = Pack::open(&dir).unwrap().unwrap();
        let w = pack.palette_windows(2);
        assert_eq!(w.iter().map(|w| w.0).collect::<Vec<_>>(), [438_333, 9, u32::MAX]);
        assert_eq!(w[0].1.as_ref().map(Vec::len), Ok(48 * 256));
        assert_eq!((w[1].1.clone().err(), w[2].1.clone().err()), (Some(reason::NOT_CAPTURED), Some(reason::NULL)));
        assert!(pack.palette_windows(3).is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    /// v2.4: draw words and their views by word, the sampler heap's entries,
    /// and the reasons of records without bytes, a missing blob and a
    /// capture without the records.
    #[test]
    fn draw_words_and_samplers_by_sink() {
        let root = std::env::temp_dir().join(format!("xnw-frame-pk6-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("frame-0013");
        let blobs = fp::blob_dir(&dir);
        let words: Vec<u8> = [9u32, 12, 9, 40].iter().chain([0u32; 12].iter()).flat_map(|w| w.to_le_bytes()).collect();
        fp::write_blob(&blobs, 70, &words).unwrap();
        fp::write_blob(&blobs, 71, &fp::encode_slots(&[SlotView { slot: 9, kind: view::PLACED, addr: 0x9000, bytes: 4096, ..SlotView::default() }, SlotView { slot: 12, kind: view::CONSTANT, addr: 0xC000, bytes: 256, ..SlotView::default() }])).unwrap();
        let entry = fp::SamplerEntry { index: 5, form: fp::SAMPLER_CONSOLE, raw: [1, 2, 3, 4], desc: [7; 27] };
        fp::write_blob(&blobs, 72, &fp::encode_samplers(&[entry])).unwrap();
        let r = |owner, kind, slot, addr, reason, hash| Ref { owner, kind, slot, addr, len: 0, reason, hash };
        let mut p = FramePack {
            manifest: Manifest { version: fp::VERSION, frame: 13, ..Manifest::default() },
            refs: vec![
                r(0, kind::DRAW_WORDS, 0, 0x2c89_0100, 0, 70),
                r(0, kind::DRAW_WORDS, 1, 0x7_0000, 0, 71),
                r(0, kind::SAMPLERS, 0, 0x8_0000, 0, 72),
                r(1, kind::DRAW_WORDS, 0, 0, reason::IMPLAUSIBLE, 0),
                r(1, kind::SAMPLERS, 0, 0, reason::NULL, 0),
                r(2, kind::DRAW_WORDS, 0, 0x2c89_0100, 0, 73),
            ],
            ..FramePack::default()
        };
        p.write(&dir).unwrap();
        let pack = Pack::open(&dir).unwrap().unwrap();
        let w = pack.draw_words(0).unwrap();
        assert_eq!((w.at, w.heap_start, w.words.map(|w| w[..4].to_vec())), (0x2c89_0100, 0x7_0000, Ok(vec![9, 12, 9, 40])));
        assert_eq!([0, 1, 2, 3, 4].map(|k| w.view(k).map(|v| v.addr)), [Some(0x9000), Some(0xC000), Some(0x9000), None, None]);
        assert_eq!(pack.samplers(0), Some((0x8_0000, Ok(vec![entry]))));
        let w = pack.draw_words(1).unwrap();
        assert_eq!((w.words, w.views), (Err(reason::IMPLAUSIBLE), Err(reason::NULL)));
        assert_eq!(pack.samplers(1), Some((0, Err(reason::NULL))));
        assert_eq!(pack.draw_words(2).unwrap().words, Err(reason::NOT_CAPTURED));
        assert_eq!((pack.draw_words(3), pack.samplers(3)), (None, None));
        assert!(pack.displacement_maps(0).is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    /// v2.4: a displacing dispatch's maps by element: the view with its
    /// heap index, an index outside the heap, an empty slot.
    #[test]
    fn displacement_maps_by_element() {
        let root = std::env::temp_dir().join(format!("xnw-frame-disp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("frame-0017");
        fp::write_blob(&fp::blob_dir(&dir), 80, &fp::encode_slots(&[SlotView { slot: 9812, kind: view::TEXTURE, addr: 0xA000, bytes: 4096, ..SlotView::default() }])).unwrap();
        let r = |slot, addr, reason, hash| Ref { owner: 4, kind: kind::DISP_MAP, slot, addr, len: 0, reason, hash };
        let mut p = FramePack {
            manifest: Manifest { version: fp::VERSION, frame: 17, ..Manifest::default() },
            refs: vec![r(0, 0x5_0000, 0, 80), r(1, 77, reason::IMPLAUSIBLE, 0), r(2, 0x5_0020, reason::NOT_RESIDENT, 0)],
            ..FramePack::default()
        };
        p.write(&dir).unwrap();
        let pack = Pack::open(&dir).unwrap().unwrap();
        let m = pack.displacement_maps(4);
        assert_eq!(m.iter().map(|m| (m.0, m.1, m.2.as_ref().map(|v| v.addr).map_err(|e| *e))).collect::<Vec<_>>(), [(0, 9812, Ok(0xA000)), (1, 77, Err(reason::IMPLAUSIBLE)), (2, 0, Err(reason::NOT_RESIDENT))]);
        let _ = std::fs::remove_dir_all(root);
    }

    /// The origin path on the real game files (`XNW_GAME_DRIVE`; ignored in
    /// CI): the first 2D texture of the first `.ttpf` found decodes to rows
    /// that are not all zero.
    #[test]
    #[ignore]
    fn origin_texels_from_the_game_files() {
        let root = std::env::var_os("XNW_GAME_DRIVE").map(PathBuf::from).expect("XNW_GAME_DRIVE");
        let mut stack = vec![root];
        let mut found = None;
        while let (Some(d), None) = (stack.pop(), &found) {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("ttpf")) {
                    found = Some(p);
                    break;
                }
            }
        }
        let path = found.expect("a .ttpf");
        let mut f = std::fs::File::open(&path).unwrap();
        let len = f.metadata().unwrap().len();
        let t = xnw_formats::ttpf::Ttpf::read(&mut f, len).unwrap();
        let i = t.textures.iter().position(|t| t.kind == xnw_formats::ttpf::KIND_2D).expect("a 2D texture");
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let p = from_origin(&format!("{name}#{i}")).expect("decoded");
        eprintln!("{name}#{i}: {}x{} format {} {} rows of {} B", p.width, p.height, p.format, p.row_count, p.row_bytes);
        assert!(p.rows.iter().any(|b| *b != 0));
    }
}
