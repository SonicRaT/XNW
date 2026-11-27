//! The title's own pixel programs in our pass (`XNW_WORLD_RENDER=...,
//! bridgeps=on|<hash>+<hash>`, default off). The rows of the named
//! pixel programs are drawn a second time, after our own pass and inside
//! its command list, with the title's mesh program (compute form) and
//! pixel program into our colour target, depth tested GREATER_EQUAL
//! against our depth without writing it (`bridge::Into`): where our pass
//! shows the surface, the title's program gives its colour.
//!
//! One path for the live title and the replay:
//! - [`take`] (pure over a [`Source`]): one draw's binding inputs as owned
//!   bytes ([`Owned`]): root constants, root views, the table views the two
//!   programs name, its 16 draw words and the textures they name (by
//!   descriptor handle), the sampler heap. The live source is the list
//!   sink's tap (apps/xnw `worldprobe/bridgeps.rs`), the replay's a frame
//!   pack sink (xnw-frame `bridge_ps.rs`).
//! - [`Cache`]: the translated pair and its two pipelines, made once per
//!   (mesh, pixel) container pair, on a thread of its own (live: a row
//!   whose pipelines are not ready is drawn by our programs alone).
//! - [`Gpu::record`]: every row of a [`Frame`] on our pass's open list. A
//!   texture is the host copy our texture table holds for its handle this
//!   frame (a 1x1 grey stands in when it has none). Its device objects are
//!   kept across frames: one view heap for all rows (each row's slots
//!   at its own base, [`bridge::bind_at`]), sampler heaps by content, the
//!   inputs a source hands over again as the same bytes in one default
//!   buffer ([`bridge::Kept`]: the live tap's kept views, vertex streams),
//!   one mapped upload buffer for the rest (frame tables, constants).
//!
//! The draw words are rewritten: word k naming a texture becomes the
//! row's base + k (that texture at view heap slot base + k; rows of one
//! words offset share the first one's), every other word [`NO_VIEW`] (past
//! the heap bounds: the guard). A row's slots are the 16 word slots and
//! the tables, not the title's heap up to its largest index.
//!
//! Confinement (BND1): the pair is translated with
//! `emit::Options::confine`, so no index of the title's reaches a
//! descriptor that is not the row's own, whatever data it comes from (a
//! draw word, a constant block, a per-pixel record: nothing is patched by
//! hand any more):
//! - a view picked by a computed heap index is read only at a slot of the
//!   row's draw-word block that holds a view the instruction's dimension
//!   reads (BND2, `bridge::dim_reads`: a 1D, 1D array, 2D or 2D array view
//!   for any of those four, a cube or 3D view for its own); the row's confinement words
//!   (`bridge::Binding::confine`) carry that block's first slot and, per
//!   dimension, the mask of its slots, made from the descriptors as they
//!   were written ([`block_masks`]). Every other index reads the heap's
//!   null of the instruction's kind and dimension (ten guards: the five
//!   kinds, then the null images that are not 2D);
//! - a table slot is read at a fixed slot the programs name; the table's
//!   block reaches past those, and a texture goes there only as a view
//!   the dimension the programs read the slot in reads, else the null
//!   image of that dimension ([`cut`], [`table_slot`]);
//! - the sampler heap is the row's own list, whole (bounded above);
//! - a pixel program leaves after `emit::BRIDGED_PIXEL_LOOP_LIMIT` back
//!   edges (16; `XNW_LOOP_LIMIT_PIXEL` set: that budget instead). The frame
//! table's slot 86 (the words' buffer, past 32 MiB in a captured frame) is one buffer
//! our pass keeps across frames: each row's rewritten words at the
//! row's own offset (root 4 dword 1, which the mesh programs read too:
//! not rewritten); the bytes between are never read.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use xnw_formats::framepack::view;
use xnw_shader::emit::{self, HostShader};
use xnw_shader::HwStage;

use super::bridge::{self, Arena, At, Draw, KEEP_MIN, KEPT_LEAST, Kept, Params, Programs, Root, SamplerDesc, Slot, unit};
use super::preskin::{Kind, View};

/// In [`Used`]: a slot read as a constant buffer; [`NAMED_IMAGE`] a slot
/// read through an image descriptor (8 dwords); other values: the stride a
/// buffer is read at.
pub const NAMED_CBV: u32 = u32::MAX;
pub const NAMED_IMAGE: u32 = u32::MAX - 1;
/// The table slots the two programs name, (root, slot) -> stride or [`NAMED_CBV`].
pub type Used = HashMap<(u16, u32), u32>;
/// The draw words: the frame table's (root 0) slot 86, at the byte offset
/// in root 4's constant dword 1 (`worldprobe/pack.rs` `draw_words`).
pub const WORDS_SLOT: u32 = 86;
pub const WORDS_ROOT: usize = 4;
pub const WORDS_DWORD: usize = 1;
pub const WORDS: usize = 16;
/// CLOTH2: the ped clothing pixel programs `PS_DeferredPedTexturedNC` of
/// effect 6a40bffd (341cfdae, 033897bc) and their mesh programs
/// (`MS_VS_PedTransformPreSkinD` 5f8ad1d5, 78161419) take their bindless
/// texture indices not from the draw words but from the frame table's
/// slot 20 (T8+0x280, a 4.9 MB raw buffer shared by the frame's rows), in
/// the row's block at b4 + 0x1c (root 1 slot 7, dword 7): the diffuse at
/// +8 (341cfdae 0xd08..0xe14: `s_lshl_b32 s20, s53, 5`, rgb sample), more
/// at +0, +4, +0xc, +0x790 + 16 x layer. Every read of slot 20 in those
/// programs is at b4 + 0x1c plus at most 0xf0c (`s_add_i32 sN, s19, lit`).
/// The title puts the block's +4/+8/+0xc indices among the draw words too
/// (pk8a f1027-1032: words 1/0/2), and the words are what we rewrite; a
/// block's index left as the title's falls outside the row's confined
/// block and reads the null image: black albedo under the title's shading.
/// So such a row gets its own copy of the block ([`Region`]), b4 + 0x1c
/// := 0, and the copy's dwords equal to one of its words' heap indices
/// take that word's rewritten value.
pub const REGION_SLOT: u32 = 20;
/// The block's root, slot and byte of its offset (b4 + 0x1c).
pub const REGION_AT: (usize, u32, usize) = (1, 7, 0x1c);
/// Bytes of a block copied (every read above lies within 0xf10).
pub const REGION_SPAN: usize = 0x1000;

/// LAMP1: the vehicle mesh programs (MSR_VS_TransformSkinD 7b142167 and
/// a555da7e, the lamps; _NoDeform f046740d, the body) read each bone's
/// mask|rank word from the frame table's slot 86 too: T8+0xac0 at the
/// draw record's +0x74 (root 1 slot 0, T9+0x0, 128 B) plus 4 x (bone >> 4)
/// (7b142167 0x6a0-0x744, f046740d 0x7ac-0x844; -1: no mask). Our slot 86
/// holds only the rows' words, so a bridged vehicle row read zeros there
/// (bone -> the compacted palette's zero entry) or another row's words
/// (pk7d f1067: the lamps of car0 as shards on the ground). Such a row
/// carries the title's bytes at that offset ([`Owned::mask`]), written
/// into our words buffer at the same offset, before the rows' words.
/// The arena slice is 64 words (1024 bones, the pack's ARENA record).
pub const MASK_AT: (usize, u32, usize) = (1, 0, 0x74);
pub const MASK_BYTES: usize = 256;
/// The draw record's size (root 1 slot 0 of the vehicle mesh programs).
const MASK_RECORD: usize = 128;

/// `XNW_BRIDGE_MASK=off` (A/B): no mask words in our slot 86, as before LAMP1.
pub fn masks_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("XNW_BRIDGE_MASK").as_deref(), Ok("off" | "0")))
}

/// LAMP1: the title's mask words a row's mesh program reads in slot 86
/// ([`MASK_AT`]): (byte offset, bytes), or None (no 128-B record, no mask,
/// or the bytes not readable).
pub fn mask_words(src: &dyn Source, out: &Owned) -> Option<(u32, Arc<Vec<u8>>)> {
    let (root, slot, byte) = MASK_AT;
    let Some(ORoot::Table(t)) = out.roots.get(root) else { return None };
    let rec = t.iter().find_map(|s| match s {
        (k, OSlot::Buffer(_, b)) if *k == slot && b.len() == MASK_RECORD => Some(b),
        _ => None,
    })?;
    let at = u32::from_le_bytes(rec.get(byte..byte + 4)?.try_into().ok()?);
    if at == u32::MAX {
        return None;
    }
    let b = src.title_words(u64::from(at), MASK_BYTES)?;
    (b.len() == MASK_BYTES).then(|| (at, Arc::new(b)))
}

/// CLOTH2: one row's copy of its slot-20 block ([`REGION_SLOT`]): placed
/// in our words buffer at `at` ([`Frame::new`]), written there each frame
/// with each patch (byte offset, word k) set to word k's rewritten value.
#[derive(Clone, Debug, Default)]
pub struct Region {
    pub bytes: Arc<Vec<u8>>,
    pub patches: Vec<(u32, u32)>,
    pub at: u64,
    /// CLOTH3: dwords equal to a word whose title view is not plain 2D,
    /// left as the title's index (they read the null image, as before CLOTH2).
    pub left: u32,
}

/// CLOTH3: `XNW_GPU_DEBUG=dred`: each bridged row's draw is preceded by a
/// marker naming its pixel and mesh programs (a PIX marker, UTF-16), which
/// DRED keeps as a breadcrumb context, so a removal in our pass names the
/// program it stopped in (idea: another runtime's per-list ring of bound shaders).
pub fn dred_marks() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNW_GPU_DEBUG").is_ok_and(|v| v.contains("dred")))
}

/// `XNW_BRIDGE_REGION=off` (A/B): slot 20 stays the title's shared buffer, as before CLOTH2.
pub fn regions_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("XNW_BRIDGE_REGION").as_deref(), Ok("off" | "0")))
}

/// CLOTH2: the row's slot-20 block as its own [`Region`] when its tables
/// hold slot 20 (raw) and b4 (a constant buffer) and the block holds an
/// index one of its draw words names (`title`: the words as the title
/// wrote them); the row's slot 20 then views the copy and b4 + 0x1c is 0.
/// None leaves the row as it was.
pub fn cut_region(out: &mut Owned, title: &[u32; WORDS]) -> Option<Region> {
    // Index 0 is left out: the block's zeros are its empty fields.
    let ws: Vec<(u32, u32, bool)> = out.heap.iter().map(|&(k, h)| (title[k as usize], k, region_word_2d(&out.dims, h))).filter(|n| n.0 != 0).collect();
    let named: Vec<(u32, u32)> = ws.iter().filter(|n| n.2).map(|n| (n.0, n.1)).collect();
    if named.is_empty() {
        return None;
    }
    let (root, slot, byte) = REGION_AT;
    let Some(ORoot::Table(t0)) = out.roots.first() else { return None };
    let Some(ORoot::Table(t1)) = out.roots.get(root) else { return None };
    let i0 = t0.iter().position(|s| s.0 == REGION_SLOT)?;
    let i1 = t1.iter().position(|s| s.0 == slot)?;
    let (OSlot::Buffer(Kind::Raw, all), OSlot::Buffer(Kind::Cbv, cb)) = (&t0[i0].1, &t1[i1].1) else { return None };
    let base = u32::from_le_bytes(cb.get(byte..byte + 4)?.try_into().ok()?) as usize;
    let end = base.checked_add(REGION_SPAN)?.min(all.len());
    if base >= end {
        return None;
    }
    let mut bytes = all[base..end].to_vec();
    bytes.resize(REGION_SPAN, 0);
    let mut left = 0;
    let patches: Vec<(u32, u32)> = bytes
        .chunks_exact(4)
        .enumerate()
        .filter_map(|(i, c)| {
            let w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let p = named.iter().find(|n| n.0 == w).map(|n| (4 * i as u32, n.1));
            left += u32::from(p.is_none() && ws.iter().any(|n| n.0 == w));
            p
        })
        .collect();
    if patches.is_empty() {
        return None;
    }
    let mut cb = cb.to_vec();
    cb[byte..byte + 4].fill(0);
    let bytes = Arc::new(bytes);
    if let Some(ORoot::Table(t1)) = out.roots.get_mut(root) {
        t1[i1].1 = OSlot::Buffer(Kind::Cbv, Arc::new(cb));
    }
    if let Some(ORoot::Table(t0)) = out.roots.first_mut() {
        t0[i0].1 = OSlot::Buffer(Kind::Raw, bytes.clone());
    }
    Some(Region { bytes, patches, at: 0, left })
}

/// CLOTH3: whether a slot-20 index may be rewritten to the row's word
/// naming texture `h`: its title view is plain 2D ([`Owned::dims`] holds
/// no other dimension for it, nor [`ZERO`]: a target the title's GPU
/// writes). Every slot-20 index of the clothing programs is read by a 2D
/// instruction (341cfdae/033897bc disasm: +8 at 0xe14, +4 at 0x2840, +0xc
/// at 0x32ac, all `image_sample ... dim:SQ_RSRC_IMG_2D`; their 2D-array
/// reads take indices from b4 + 0x24..0x2c). Under BND2's flat class a 2D
/// array view would serve them; D3D12 leaves that read undefined (HANG2
/// cause 1). Any other view keeps the title's index and reads the null.
pub fn region_word_2d(dims: &[(u64, u8)], h: u64) -> bool {
    dims.iter().find(|d| d.0 == h).is_none_or(|d| d.1 == DIM_2D)
}

/// CLOTH3: the census of [`Stats::region_views`] for the bridged line:
/// "; read as 2D 40, stand-in 8; left 0; formats 71 x30, 98 x10" (empty without region rows).
fn region_text(st: &Stats) -> String {
    if st.regions[0] == 0 {
        return String::new();
    }
    let v: Vec<String> = REGION_CLASSES.iter().zip(st.region_views).filter(|x| x.1 > 0).map(|(w, n)| format!("{w} {n}")).collect();
    let mut fs = st.region_formats.clone();
    fs.sort_unstable();
    let fs: Vec<String> = fs.iter().map(|(f, n)| format!("{f} x{n}")).collect();
    format!("; read as {}; left {}; formats {}", if v.is_empty() { "none".to_string() } else { v.join(", ") }, st.region_left, if fs.is_empty() { "none".to_string() } else { fs.join(", ") })
}

/// Adds `n` to format `f`'s count.
fn region_format(to: &mut Vec<(u32, u32)>, f: u32, n: u32) {
    match to.iter_mut().find(|x| x.0 == f) {
        Some(x) => x.1 += n,
        None => to.push((f, n)),
    }
}

/// CLOTH3: the classes of [`region_view_class`], for the census.
pub const REGION_CLASSES: [&str; 8] = ["2D", "stand-in", "not 2D", "typeless", "integer", "B5G5R5A1/B4G4R4A4", "MSAA", "no mips or slices"];

/// CLOTH3: how the view a region row's rewritten word is written with
/// (`view`: our table's view where it is written, None: the one-texel
/// stand-in) stands under a 2D sample instruction: an index into
/// [`REGION_CLASSES`]. 0 is the only proper one; 1 a stand-in (a 2D
/// RGBA8 texel); the others name an undefined read (a view dimension that
/// is not 2D, a format a filtered sample cannot read: typeless, depth or
/// integer, the 16-bit packed formats the host may refuse (VIctory),
/// a multisampled view, a view with zero mips or array slices (another runtime)).
pub fn region_view_class(view: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> usize {
    let Some(v) = view else { return 1 };
    let f = v.Format.0 as u32;
    // SAFETY: the union member read is the one ViewDimension names.
    let (mips, slices) = unsafe {
        match v.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE2D => (v.Anonymous.Texture2D.MipLevels, 1),
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => (v.Anonymous.Texture2DArray.MipLevels, v.Anonymous.Texture2DArray.ArraySize),
            _ => (1, 1),
        }
    };
    match v.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2DMS | D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => 6,
        _ if mips == 0 || slices == 0 => 7,
        D3D12_SRV_DIMENSION_TEXTURE2D => match f {
            0 => 3,
            _ if xnw_gpu::d3d12x::resource::is_typeless(f) || matches!(f, 20 | 40 | 45 | 55) => 3,
            3 | 4 | 7 | 8 | 12 | 14 | 17 | 18 | 22 | 25 | 30 | 32 | 36 | 38 | 42 | 43 | 47 | 50 | 52 | 57 | 59 | 62 | 64 => 4,
            86 | 115 => 5,
            _ => 0,
        },
        _ => 2,
    }
}

/// A rewritten draw word that names no texture: past every heap bound
/// (our view heap holds at most 1,000,000 slots; times 32 still fits 32
/// bits). Small enough that the stride table covers it: the translated
/// programs read the table at a slot before the slot is bounded
/// (`emit::lower::address` `stride_entry`), and a root view has no bounds
/// on the hardware adapter (WARP reads zeros past the buffer).
pub const NO_VIEW: u32 = 0x000F_FFFF;
/// One table view's bytes at most.
pub const VIEW_MAX: u64 = 16 << 20;
/// Bit of a mesh program's first root constant: set, group k draws the
/// meshlet the title's visible-meshlet list names at k (a GPU output our
/// pass does not have); cleared, meshlet k.
const VISIBLE_LIST: u32 = 1 << 28;

/// The table slots a console container's programs name, with the widest
/// stride each is read at, into `used`; `wave` becomes 64 for a mesh
/// program of 64-lane waves.
pub fn named(container: &[u8], used: &mut Used, wave: &mut u32) -> Result<(), String> {
    let parsed = xnw_shader::Container::parse(container).map_err(|e| e.to_string())?;
    let part = parsed.part(xnw_shader::FourCC::DXIL).ok_or("no DXIL part")?;
    for p in xnw_shader::native_programs(part.data).map_err(|e| e.to_string())? {
        if p.stage == HwStage::Ms && p.wave64 {
            *wave = 64;
        }
        for u in p.used.iter().filter(|u| u.root == xnw_shader::RootType::Table) {
            let e = used.entry((u16::from(u.slot), u.table_offset_dwords / 8)).or_insert(0);
            *e = (*e).max(match u.kind {
                xnw_shader::ResourceType::Cbv => NAMED_CBV,
                _ if u.dwords == 8 => NAMED_IMAGE,
                _ => u.stride,
            });
        }
    }
    Ok(())
}

/// The translated pair of a mesh and a pixel container, and the table
/// slots they name.
pub fn translate(msc: &[u8], psc: &[u8]) -> Result<(Programs, Used), String> {
    // `XNW_HEAP_NONUNIFORM=<hash prefix>,..|all` reaches the bridged programs too.
    // BND1: both programs confined to the row's own descriptors (`emit::Options::confine`).
    let opts = |c: &[u8]| emit::Options { nonuniform_heap: emit::nonuniform_heap_for(xnw_core::hash128(&[c])), confine: true, ..bridge::options() };
    // SKIN2 `XNW_REDUCE=..,code=ms-rfl`: the reduced row's mesh program with its first-lane reads the lane's own.
    let own = crate::reduce::ms_first_lane_own(psc);
    emit::first_lane_own(own);
    let ms = emit::mesh::translate_fallback(msc, opts(msc));
    if own {
        emit::first_lane_own(false);
    }
    let (ms, layout) = ms.map_err(|e| format!("mesh program: {e}"))?;
    // KILL1: the pixel program's returns joined in one block: WARP 1.0.22 ends the device on a pixel shader
    // with two `ret` blocks under our depth test (the 8 families with an early kill exit; `emit::Options::one_return`).
    // BND1: the pixel program's loop budget is 16 back edges an invocation, not the stage's 256
    // (`XNW_LOOP_LIMIT_PIXEL` set: that one, for comparing).
    let loop_limit = std::env::var("XNW_LOOP_LIMIT_PIXEL").is_err().then_some(emit::BRIDGED_PIXEL_LOOP_LIMIT);
    let ps_opts = emit::Options { one_return: crate::reduce::one_return(psc), loop_limit, ..opts(psc) };
    // `XNW_REDUCE` (KILL1): the reduced program with one feature left out.
    let ps = match crate::reduce::stub_for(psc) {
        Some(stub) => {
            let (r, sites) = emit::translate_stubbed(psc, ps_opts, stub);
            println!("{}", crate::reduce::sites_line(psc, &sites));
            r
        }
        None => emit::translate_with(psc, ps_opts),
    }
    .map_err(|e| format!("pixel program: {e}"))?;
    // CLOTH4: a pixel program reads its root views raw (the root-length bound is the compute form's,
    // `emit::shader` ROOT_LEN_SPACE), and a raw read past our 256-byte copy page-faults on the card (no
    // robustness for root descriptors). None of the 23 bridged families has one (V); a new one is refused.
    if !(ps.root_cbvs.is_empty() && ps.root_srvs.is_empty() && ps.root_uavs.is_empty()) {
        return Err(format!("pixel program: root views (CBV {:?}, SRV {:?}, UAV {:?}) unbounded on the card", ps.root_cbvs, ps.root_srvs, ps.root_uavs));
    }
    let vs =emit::mesh::passthrough_vs(&layout, Some(&ps.dxil)).map_err(|e| format!("pass-through vertex shader: {e}"))?;
    crate::reduce::dump(psc, &ps.dxil, &vs, &ms.dxil);
    let mut used = HashMap::new();
    let mut wave = 32;
    for c in [msc, psc] {
        named(c, &mut used, &mut wave)?;
    }
    Ok((Programs { ms, layout, vs, ps, wave, vertex: None }, used))
}

/// A descriptor slot as a source describes it (`xnw_formats::framepack::view` kinds).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SrcView {
    pub kind: u32,
    pub addr: u64,
    pub bytes: u64,
    pub format: u32,
    /// A buffer view's element size (0: none given).
    pub width: u64,
    /// A texture view's dimension as the title described it (the D3D12
    /// shader-resource dimension: 2 1D, 3 1D array, 4 2D, 5 2D array, 8 3D,
    /// 9 cube, 10 cube array); 0: not given (a resource view's own).
    pub dim: u32,
}

/// A 2D view, in [`view_dim`]'s numbers.
pub const DIM_2D: u8 = 4;
/// Bit of a stand-in's dimension ([`Owned::dims`], [`Lay::stand_in_dim`]):
/// the stand-in is zero, not grey. Set for a view of a resource object
/// (`view::TEXTURE`): a target or volume the title's GPU writes (rain
/// patterns, wind volumes; T8MAP), which our null run never fills. Zero is
/// what the title's own memory holds there; grey would be half wetness
/// and a constant wind. A placed texture whose texels are not here yet
/// stays grey. 2D, 2D array and 3D only (a cube or 1D one is grey).
pub const ZERO: u8 = 0x80;

/// The stand-in dimension of a title view: [`view_dim`], with [`ZERO`]
/// for a view of a resource object.
fn stand_dim(v: &SrcView) -> u8 {
    view_dim(v.dim) | if v.kind == view::TEXTURE { ZERO } else { 0 }
}

/// The dimension the bridge's view at a texture slot takes for a title
/// view of dimension `title` ([`SrcView::dim`]): the title's own, among the
/// kinds translated programs declare. A cube array is read as a cube
/// (emit's `cube_direction`), multisampled images are refused by the
/// translator, and a view that names no dimension is taken as 2D.
/// Why: D3D12 leaves a read through a view of another dimension undefined
/// and the software adapter does not mind it; pixel program 38110e02
/// sampled two 256x256x6 cube views through the bridge's 1x1 2D stand-in.
/// (That program still lost the RX 5700 XT with cube stand-ins bound,
/// CARD5 c1: this is a rule kept, not that hang's cause.) VIctory 8c26061
/// (`null_like`): a null view has the dimension of the title's description.
pub fn view_dim(title: u32) -> u8 {
    match title {
        2 | 3 | 5 | 8 | 9 => title as u8,
        10 => 9,
        _ => DIM_2D,
    }
}

/// The title's view dimension of a texture view ([`SrcView::dim`]) from
/// its captured words (`TapView`/`SlotView`: the resource description's
/// 14 words, then the view's): the view's own (`words[15]`); without a
/// view description the resource's default view, a 2D array when a 2D
/// resource has slices, 3D for a 3D one. 0: not given. Resource views
/// carry their words since BUG3 (`tap::view`); an older capture's are
/// zeros and such a view is taken as 2D.
pub fn title_dim(kind: u32, words: &[u32; 30]) -> u32 {
    if kind != view::PLACED && kind != view::TEXTURE {
        return 0;
    }
    match (words[15], words[0]) {
        (0, 4) => 8,
        (0, 3) if words[7] & 0xffff > 1 => 5,
        (d, _) => d,
    }
}

/// Whether a texture's view from our table can stand at a slot the
/// programs read in dimension `dim` ([`view_dim`]): a cube, a 2D array or
/// a 3D view only where that is the dimension read, and a 2D view not
/// where one of those is (D3D12 leaves a read through a view of another
/// dimension undefined; the stand-in of `dim` goes there instead). A cube
/// array view fits nothing: the translator declares a cube there.
pub fn view_fits(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC, dim: u8) -> bool {
    let dim = dim & !ZERO;
    match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => !matches!(dim, 5 | 8 | 9),
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => dim == 5,
        D3D12_SRV_DIMENSION_TEXTURE3D => dim == 8,
        D3D12_SRV_DIMENSION_TEXTURECUBE => dim == 9,
        _ => false,
    }
}

/// The texture our table holds for `h` this frame, if its view fits a slot
/// read in dimension `dim` ([`view_fits`]).
fn real(texs: &Texs, h: u64, dim: u8) -> Option<&(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)> {
    texs.get(&h)?.as_ref().filter(|t| view_fits(&t.1, dim))
}

/// [`view_dim`]'s number for an image instruction's dimension (GFX10 MIMG
/// `dim`: 0 1D, 1 2D, 2 3D, 3 cube, 4 1D array, 5 2D array, 6 and 7
/// multisampled), as emit's lowering declares the handle.
pub fn read_dim(mimg: u8) -> u8 {
    match mimg & 7 {
        0 => 2,
        2 => 8,
        3 => 9,
        4 => 3,
        5 => 5,
        _ => DIM_2D,
    }
}

/// The names of [`view_dim`]'s dimensions, for the statistics.
pub const DIMS: [(u8, &str); 6] = [(4, "2D"), (5, "2D array"), (9, "cube"), (8, "3D"), (2, "1D"), (3, "1D array")];

/// The description of a one-texel RGBA8 view of dimension `dim`
/// ([`view_dim`]): a stand-in over the texture of that shape, or a null.
pub fn one_texel_view(dim: u8) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let (d, a) = match dim & !ZERO {
        2 => (D3D12_SRV_DIMENSION_TEXTURE1D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture1D: D3D12_TEX1D_SRV { MipLevels: 1, ..Default::default() } }),
        3 => (D3D12_SRV_DIMENSION_TEXTURE1DARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture1DArray: D3D12_TEX1D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() } }),
        5 => (D3D12_SRV_DIMENSION_TEXTURE2DARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() } }),
        8 => (D3D12_SRV_DIMENSION_TEXTURE3D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_SRV { MipLevels: 1, ..Default::default() } }),
        9 => (D3D12_SRV_DIMENSION_TEXTURECUBE, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { TextureCube: D3D12_TEXCUBE_SRV { MipLevels: 1, ..Default::default() } }),
        _ => (D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } }),
    };
    D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: d, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: a }
}

/// The table slots the pair's programs read as images at a fixed place,
/// with the dimension each is declared in ([`read_dim`]): (root, slot,
/// dimension). A slot read in two dimensions keeps the one that is not 2D.
/// Reads through a computed heap index are not here: those views take the
/// title's own dimension ([`Owned::dims`]).
pub fn image_reads(containers: &[&[u8]]) -> Vec<(u16, u32, u8)> {
    let mut out: Vec<(u16, u32, u8)> = Vec::new();
    for c in containers {
        for u in super::compute::uses(c).unwrap_or_default() {
            let Some(i) = u.image else { continue };
            let (root, dim) = (u16::from(u.root), read_dim(i.dim));
            match out.iter_mut().find(|o| o.0 == root && o.1 == u.slot) {
                Some(o) if o.2 == DIM_2D => o.2 = dim,
                Some(_) => {}
                None => out.push((root, u.slot, dim)),
            }
        }
    }
    out
}

/// The dimensions the containers' programs read images in anywhere, table
/// slots and computed heap indices alike: bit [`read_dim`] of every image
/// instruction (0: none decoded). A title view of a dimension none of the
/// pair's programs reads is not read in that dimension, so its stand-in
/// stays 2D ([`Lay::stand_in_dim`]): pk6 f1205 row 1401's word 1 is a 1D
/// view of the title's (1024x1) that 38110e02 has no 1D instruction for.
pub fn dims_read(containers: &[&[u8]]) -> u16 {
    let mut mask = 0u16;
    for c in containers {
        let Ok(parsed) = xnw_shader::Container::parse(c) else { continue };
        let Some(part) = parsed.part(xnw_shader::FourCC::DXIL) else { continue };
        for p in xnw_shader::native_programs(part.data).unwrap_or_default() {
            for inst in xnw_shader::gfx10::decode(&p.patched_code()).unwrap_or_default() {
                if let xnw_shader::gfx10::Fields::Image { dim, .. } = inst.fields(p.wave64) {
                    mask |= 1 << read_dim(dim);
                }
            }
        }
    }
    mask
}

/// [`dims_read`]'s mask in words ("2D, 2D array, cube").
pub fn dims_text(mask: u16) -> String {
    let d: Vec<&str> = DIMS.iter().filter(|d| mask & 1 << d.0 != 0).map(|d| d.1).collect();
    if d.is_empty() { "no images".into() } else { d.join(", ") }
}

/// A root argument as a source has it.
#[derive(Clone, Debug, PartialEq)]
pub enum SrcRoot {
    None,
    Constants(Vec<u32>),
    /// A descriptor table: its first slot's handle.
    Table(u64),
    /// A root view's GPU address.
    View(u64),
}

/// Where a draw's binding inputs come from (the title at its list sink,
/// or a frame pack's sink).
pub trait Source {
    fn roots(&self) -> Vec<SrcRoot>;
    /// The view at table root `root`, slot `slot` (None: no view there).
    fn table(&self, root: u16, slot: u32) -> Option<SrcView>;
    /// A buffer or constant view's bytes (at most [`VIEW_MAX`]).
    fn bytes(&self, root: u16, slot: u32, v: &SrcView) -> Option<Arc<Vec<u8>>>;
    /// A root view's bytes from its address.
    fn root_view(&self, root: u16, addr: u64) -> Option<Arc<Vec<u8>>>;
    /// The draw's 16 words (None: not readable).
    fn words(&self) -> Option<[u32; WORDS]>;
    /// LAMP1: `len` bytes of the title's slot 86 at byte offset `at` (None: not readable).
    fn title_words(&self, _at: u64, _len: usize) -> Option<Vec<u8>> {
        None
    }
    /// The view at heap index `index` of the bound view heap, and its handle.
    fn heap(&self, index: u32) -> Option<(u64, SrcView)>;
    /// The bound sampler heap's entries, as host descriptions.
    fn samplers(&self) -> Option<Arc<Vec<(u32, SamplerDesc)>>>;
}

/// A slot's input as owned bytes or a texture's handle.
#[derive(Clone, Debug)]
pub enum OSlot {
    Buffer(Kind, Arc<Vec<u8>>),
    Texture(u64),
}

#[derive(Clone, Debug, Default)]
pub enum ORoot {
    #[default]
    None,
    Constants(Vec<u32>),
    Table(Vec<(u32, OSlot)>),
    View(Arc<Vec<u8>>),
}

/// The (mesh, pixel) container hashes of a pipeline.
pub type Key = (u128, u128);

/// The title's pixel output our colour target takes: 0 (its first
/// target, the albedo of a G-buffer program); `XNW_BRIDGE_TARGET=<1-7>`
/// shows another one instead (a lab view: the normal, material and
/// emissive targets, where array and cube texels land, BUG3). The
/// targets before it are bound null; a program without that target is
/// refused.
pub fn shown_target() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XNW_BRIDGE_TARGET").ok().and_then(|v| v.trim().parse::<u32>().ok()).map_or(0, |n| n.min(7)))
}

/// `ourlight=metal` with target 0 shown: the bridged rows write their
/// target 2 to the lighting pass's metal target (LITE2).
pub fn metal_on() -> bool {
    crate::ourlight::metal() && shown_target() == 0
}

/// `ournormals=title` too: their target 1 (the title's RT1: 1 - roughness
/// and the normal word) to the lighting pass's title normal target (NORM1).
pub fn normals_on() -> bool {
    metal_on() && crate::ourlight::title_normals()
}

/// `emissive=on` too (LGT1): their target 5 (the title's RT5, the packed
/// emissive word) to the lighting pass's emissive target.
pub fn emissive_on() -> bool {
    metal_on() && crate::ourlight::emissive()
}

/// What a bridged pipeline writes into the lighting pass's targets
/// (`ourlight=metal`): its target 2 into the metal target; with
/// `ournormals=title` (`Some`) whether its target 1 goes to the title normal
/// target; with `emissive=on` whether its target 5 goes to the emissive target.
pub type Lit = (bool, Option<bool>, bool);

/// One draw's binding inputs ([`take`]).
#[derive(Clone, Debug, Default)]
pub struct Owned {
    pub key: Key,
    pub roots: Vec<ORoot>,
    /// Textures the draw words name: (word k, handle).
    pub heap: Vec<(u32, u64)>,
    /// Textures (table slots and draw words) whose title view is not 2D:
    /// (handle, [`view_dim`]). The stand-in at such a slot has that
    /// dimension.
    pub dims: Vec<(u64, u8)>,
    /// The rewritten words and their byte offset in the title's slot 86.
    pub words: [u32; WORDS],
    pub words_at: u32,
    pub samplers: Arc<Vec<(u32, SamplerDesc)>>,
    /// Mesh groups (the meshlet buffer's count; 0: not known yet).
    pub groups: u32,
    /// Buffer views by title address and their first word (the meshlet
    /// buffer is the one at the row's `mesh_at`).
    pub firsts: Vec<(u64, u32)>,
    /// What stood in, by [`LACKS`] index.
    pub lacks: u32,
    /// Title address of each buffer input, by its bytes' address (the
    /// `Arc`'s data): `XNW_ZEROCOPY=on` binds it there in place (ZC1).
    pub addrs: Vec<(usize, u64)>,
    /// The draw's identity across frames ([`ident`]; 0: none, the replay).
    pub ident: u64,
    /// The title's blend of target 0 for this draw (`RenderState::blend`;
    /// zeros: none), applied as our own draws apply it.
    pub blend: [u8; 8],
    /// CLOTH2: its own copy of its slot-20 block ([`cut_region`]).
    pub region: Option<Region>,
    /// LAMP1: the title's mask words at their slot-86 offset ([`mask_words`]).
    pub mask: Option<(u32, Arc<Vec<u8>>)>,
}

/// A draw's identity across frames from its list row's own record: its
/// geometry (meshlet data, vertex buffer, the geometry's hash word) and
/// its pass stage. [`Gpu::record`] adds the pipeline and the draw's
/// occurrence among the frame's draws of one identity (instances of one
/// geometry in one pass, in list order); never 0.
pub fn ident(g: &super::GeomKey, stage: u32) -> u64 {
    let mut h = Fx::default();
    for w in [g.mesh_at, g.mesh_len, g.vb_at, g.vb_len, g.ib, g.hash, u64::from(stage)] {
        std::hash::Hasher::write_u64(&mut h, w);
    }
    std::hash::Hasher::finish(&h) | 1
}

/// What [`Owned::lacks`] counts, by bit.
pub const LACKS: [&str; 10] = ["no draw words", "a word below the heap bound naming no view", "a named constant buffer without bytes (zeros)", "a named view without bytes", "no samplers (defaults)", "a root view without bytes", "no group count", "a texture read as a buffer (dropped)", "a buffer read as an image (dropped)", "a texture's slot read as a constant buffer"];

impl Owned {
    /// The group count from the meshlet buffer at `mesh_at` (CPU or GPU address).
    pub fn set_groups(&mut self, mesh_at: u64) {
        self.set_groups_with(mesh_at, xnw_core::pair::pair());
    }

    /// [`Owned::set_groups`] with the paired reservation `pair` (ALIAS1):
    /// a view at `mesh_at` or, from the CPU view, at its GPU address.
    pub fn set_groups_with(&mut self, mesh_at: u64, pair: xnw_core::pair::Pair) {
        if let Some(&(_, w)) = self.firsts.iter().find(|(a, _)| *a >= mesh_at && pair.same(*a, mesh_at)) {
            self.groups = w & 0x1FFFF;
        }
        if self.groups == 0 {
            self.lacks |= 1 << 6;
        }
    }

    /// The textures this draw samples, by handle.
    pub fn handles(&self) -> impl Iterator<Item = u64> + '_ {
        self.heap.iter().map(|h| h.1).chain(self.roots.iter().flat_map(|r| match r {
            ORoot::Table(v) => v.iter().filter_map(|s| if let OSlot::Texture(h) = s.1 { Some(h) } else { None }).collect(),
            _ => Vec::new(),
        }))
    }
}

/// The binding inputs of one draw of pipeline `key` whose programs name
/// `used` (pure over `src`).
pub fn take(src: &dyn Source, key: Key, used: &Used) -> Owned {
    let mut out = Owned { key, ..Owned::default() };
    let roots = src.roots();
    // PERF5: read before the roots move into the row (their constants are not copied again).
    out.words_at = match roots.get(WORDS_ROOT) {
        Some(SrcRoot::Constants(c)) => c.get(WORDS_DWORD).copied().unwrap_or(0),
        _ => 0,
    };
    out.roots = vec![ORoot::None; roots.len()];
    for (i, r) in roots.into_iter().enumerate() {
        let root = i as u16;
        out.roots[i] = match r {
            SrcRoot::None => ORoot::None,
            SrcRoot::Constants(c) => ORoot::Constants(c),
            SrcRoot::View(a) => match src.root_view(root, a) {
                Some(b) => {
                    out.addrs.push((b[..].as_ptr() as usize, a));
                    ORoot::View(b)
                }
                None => {
                    out.lacks |= 1 << 5;
                    ORoot::None
                }
            },
            SrcRoot::Table(handle) => {
                let mut slots = Vec::new();
                let mut named: Vec<(u32, u32)> = used.iter().filter(|(k, _)| k.0 == root).map(|(k, s)| (k.1, *s)).collect();
                named.sort_unstable();
                for (slot, stride) in named {
                    // Slot 86 of the frame table is the frame's sparse words buffer.
                    if root == 0 && slot == WORDS_SLOT {
                        continue;
                    }
                    let Some(v) = src.table(root, slot) else {
                        if stride == NAMED_CBV {
                            // A program loading a constant buffer where the heap holds no view removes the device.
                            out.lacks |= 1 << 2;
                            slots.push((slot, OSlot::Buffer(Kind::Cbv, Arc::new(vec![0; 256]))));
                        }
                        continue;
                    };
                    match v.kind {
                        view::CONSTANT | view::BUFFER | view::PLACED_BUFFER if stride == NAMED_IMAGE => out.lacks |= 1 << 8,
                        view::CONSTANT | view::BUFFER | view::PLACED_BUFFER => {
                            let Some(b) = src.bytes(root, slot, &v) else {
                                if stride == NAMED_CBV {
                                    out.lacks |= 1 << 2;
                                    slots.push((slot, OSlot::Buffer(Kind::Cbv, Arc::new(vec![0; 256]))));
                                } else {
                                    out.lacks |= 1 << 3;
                                }
                                continue;
                            };
                            if b.len() >= 4 {
                                out.firsts.push((v.addr, u32::from_le_bytes([b[0], b[1], b[2], b[3]])));
                            }
                            // R32_TYPELESS (39) is a raw view.
                            let kind = match (v.kind, v.format, stride) {
                                (view::CONSTANT, ..) => Kind::Cbv,
                                (view::BUFFER, f, _) if f != 0 && f != 39 => Kind::Typed { format: f as i32, bytes: v.width.max(1) as u32 },
                                (_, _, NAMED_CBV) => Kind::Cbv,
                                (_, _, s) if s > 4 => Kind::Structured(s),
                                _ => Kind::Raw,
                            };
                            out.addrs.push((b[..].as_ptr() as usize, v.addr));
                            slots.push((slot, OSlot::Buffer(kind, b)));
                        }
                        // A texture read through a buffer descriptor gives the
                        // hardware a wild address (victory-fork 3c81420): the row is dropped.
                        view::TEXTURE | view::PLACED if stride != NAMED_IMAGE && stride != NAMED_CBV => out.lacks |= 1 << 7,
                        // HANG2: a slot one program reads as a constant buffer
                        // (`NAMED_CBV` wins the merge in `named`) holding a texture:
                        // the CBV loads read the texture's descriptor (counted).
                        view::TEXTURE | view::PLACED => {
                            if stride == NAMED_CBV {
                                out.lacks |= 1 << 9;
                            }
                            if stand_dim(&v) != DIM_2D {
                                out.dims.push((handle + 32 * u64::from(slot), stand_dim(&v)));
                            }
                            slots.push((slot, OSlot::Texture(handle + 32 * u64::from(slot))));
                        }
                        _ => {}
                    }
                }
                ORoot::Table(slots)
            }
        };
    }
    for r in &mut out.roots {
        if let ORoot::Constants(c) = r
            && let Some(w) = c.first_mut()
        {
            *w &= !VISIBLE_LIST;
        }
    }
    let title = src.words();
    match title {
        Some(words) => {
            for (k, w) in words.iter().enumerate() {
                out.words[k] = NO_VIEW;
                match src.heap(*w) {
                    Some((h, v)) if v.kind == view::TEXTURE || v.kind == view::PLACED => {
                        out.words[k] = k as u32;
                        out.heap.push((k as u32, h));
                        if stand_dim(&v) != DIM_2D {
                            out.dims.push((h, stand_dim(&v)));
                        }
                    }
                    Some(_) => {}
                    // A value past every heap bound is the material's data, not
                    // a view (floats 1.0, -1.0, 0.6... in pk6 f1205: words 4-15,
                    // `--bridgeps-inputs`): kept as it is (the sheet is unchanged).
                    None if *w > NO_VIEW => out.words[k] = *w,
                    None => out.lacks |= 1 << 1,
                }
            }
        }
        None => {
            out.words = [NO_VIEW; WORDS];
            out.lacks |= 1;
        }
    }
    if regions_on()
        && let Some(title) = title
    {
        out.region = cut_region(&mut out, &title);
    }
    if masks_on() {
        out.mask = mask_words(src, &out);
    }
    out.samplers = src.samplers().unwrap_or_else(|| {
        out.lacks |= 1 << 4;
        Arc::new(Vec::new())
    });
    out
}

/// A frame's bridged rows: (list row, inputs), and the end of slot 86's
/// words (the largest offset + 64).
#[derive(Debug, Default)]
pub struct Frame {
    pub rows: Vec<(u32, Owned)>,
    pub words_end: u64,
    /// Draws of the wanted programs whose pipelines were not ready, and
    /// draws not read (tap counters, live).
    pub not_ready: u32,
    pub unread: u32,
    /// CLOTH2: each row's [`Region`] by its bytes' address, and where it
    /// lies in our words buffer (after the words; `words_end` covers them).
    pub regions: Vec<(usize, u64)>,
}

impl Frame {
    pub fn new(mut rows: Vec<(u32, Owned)>) -> Frame {
        let mut words_end = rows.iter().map(|r| u64::from(r.1.words_at) + 4 * WORDS as u64).max().unwrap_or(0);
        // LAMP1: the mask words lie at the title's own offsets in slot 86.
        words_end = rows.iter().filter_map(|r| r.1.mask.as_ref()).map(|m| u64::from(m.0) + m.1.len() as u64).fold(words_end, u64::max);
        let mut regions = Vec::new();
        for (_, o) in &mut rows {
            if let Some(g) = &mut o.region {
                // This frame's buffer: never held or replayed (as the earlier bound rows).
                g.at = words_end.next_multiple_of(256);
                words_end = g.at + g.bytes.len() as u64;
                regions.push((g.bytes.as_ptr() as usize, g.at));
                o.ident = 0;
            }
        }
        Frame { rows, words_end, not_ready: 0, unread: 0, regions }
    }

    /// CLOTH2: writes row `o`'s [`Region`] into our words buffer `to`, its
    /// patches as the row's words are written (word k at `words_base` + k).
    pub fn write_region(o: &Owned, words_base: u32, to: &mut [u8]) {
        if let Some(g) = &o.region
            && let Some(d) = to.get_mut(g.at as usize..g.at as usize + g.bytes.len())
        {
            d.copy_from_slice(&g.bytes);
            for &(off, k) in &g.patches {
                d[off as usize..off as usize + 4].copy_from_slice(&(k + words_base).to_le_bytes());
            }
        }
    }

    /// LAMP1: writes row `o`'s mask words ([`Owned::mask`]) at their offset
    /// into `to` (before any row's words: those win where they meet).
    pub fn write_mask(o: &Owned, to: &mut [u8]) -> bool {
        let Some((at, b)) = &o.mask else { return false };
        let Some(d) = to.get_mut(*at as usize..*at as usize + b.len()) else { return false };
        d.copy_from_slice(b);
        true
    }

    /// Writes every row's mask words, then its words at its offset into `to`
    /// (`len` bytes; rows past it left out).
    pub fn write_words(&self, to: &mut [u8]) {
        for (_, o) in &self.rows {
            Frame::write_mask(o, to);
        }
        for (_, o) in &self.rows {
            if let Some(d) = to.get_mut(o.words_at as usize..o.words_at as usize + 4 * WORDS) {
                for (k, w) in o.words.iter().enumerate() {
                    d[4 * k..4 * k + 4].copy_from_slice(&w.to_le_bytes());
                }
            }
        }
    }

    /// Row `i` as the binding takes it (textures by handle).
    pub fn draw(&self, i: usize, sampler_slots: u32) -> Draw<'_> {
        let o = &self.rows[i].1;
        fn slot(s: &OSlot) -> Slot<'_> {
            match s {
                OSlot::Buffer(kind, bytes) => Slot::Buffer(View { kind: *kind, bytes }),
                OSlot::Texture(h) => Slot::Handle(*h),
            }
        }
        let roots = o
            .roots
            .iter()
            .enumerate()
            .map(|(r, x)| match x {
                ORoot::None => Root::None,
                ORoot::Constants(c) => Root::Constants(c.clone()),
                ORoot::Table(v) => {
                    let mut t: Vec<(u32, Slot)> = v.iter().map(|(k, s)| (*k, slot(s))).collect();
                    // Slot 86 holds its place; [`record`] views our words buffer there.
                    if r == 0 {
                        t.push((WORDS_SLOT, Slot::Buffer(View { kind: Kind::Raw, bytes: &WORDS_PLACE })));
                        t.sort_by_key(|s| s.0);
                    }
                    Root::Table(t)
                }
                ORoot::View(b) => Root::View(b),
            })
            .collect();
        Draw { roots, heap: o.heap.iter().map(|&(k, h)| (k, Slot::Handle(h))).collect(), samplers: o.samplers.to_vec(), sampler_slots, groups: o.groups }
    }
}

static WORDS_PLACE: [u8; 4 * WORDS] = [0; 4 * WORDS];

/// A translated pair with its pipelines for drawing into our pass.
pub struct Built {
    pub programs: Programs,
    pub used: Used,
    croot: ID3D12RootSignature,
    cpso: ID3D12PipelineState,
    groot: ID3D12RootSignature,
    gpso: ID3D12PipelineState,
    cp: Params,
    gp: Params,
    /// The pixel pipeline per title blend ([`blend_ok`]), made at its first row.
    blended: std::sync::Mutex<HashMap<([u8; 8], bool, (i32, u32)), ID3D12PipelineState>>,
    /// The pass-through vertex program (for those pipelines).
    vs: Vec<u8>,
    /// The pair's table layout ([`Lay`]).
    pub lay: Lay,
    /// `ourlight=metal` (LITE2): the pipelines have three targets (our
    /// colour, none or (`ournormals=title`) the title normal target, our
    /// metal target) and what this one writes there ([`Lit`]).
    pub metal: Option<Lit>,
    /// Milliseconds the translation and the pipelines took.
    pub ms: f64,
    /// The pixel program's hash (`xnw_core::hash128` of its record): [`crate::reduce::pipeline`] changes only its pipelines.
    pub ps_hash: u128,
}

// SAFETY: D3D12 objects are free-threaded; the programs are plain data.
unsafe impl Send for Built {}
// SAFETY: as above.
unsafe impl Sync for Built {}

/// The pair's pipelines on `dev`: the mesh program's compute form, and the
/// pass-through vertex shader with the pixel program into our colour
/// target (RGBA8; its other targets dropped) and depth (GREATER_EQUAL, no write).
pub fn build(dev: &ID3D12Device, caps: &xnw_gpu::host::Caps, msc: &[u8], psc: &[u8], title: &[u32]) -> Result<Built, String> {
    let t0 = std::time::Instant::now();
    // Card safety first: a program that hung the card gets no pipeline there.
    if let Some(why) = bridge::denied(xnw_core::hash128(&[psc]), caps.software) {
        return Err(why);
    }
    let (p, used) = translate(msc, psc)?;
    if let Some(gap) = bridge::gap(caps, &p.needs()) {
        return Err(gap);
    }
    let target = shown_target();
    if !bridge::targets(&p.ps.dxil).contains(&(target, DXGI_FORMAT_R32G32B32A32_FLOAT.0)) {
        return Err(format!("the pixel program's target {target} is not a float target"));
    }
    let eh = emit::host::Host::from_device(dev.clone());
    use emit::mesh::{INDEX_REGISTER, RECORD_REGISTER};
    let croot = eh.root_signature_records(&[(&p.ms, D3D12_SHADER_VISIBILITY_ALL)], &[(true, RECORD_REGISTER), (true, INDEX_REGISTER), (false, RECORD_REGISTER)])?;
    let cpso = eh.compute_pipeline_with(&croot, &p.ms.dxil).map_err(|x| format!("compute pipeline: {x}"))?;
    let vs = HostShader { stage: HwStage::Vs, dxil: p.vs.clone(), user_data: Vec::new(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false };
    let gstages = [(&vs, D3D12_SHADER_VISIBILITY_VERTEX), (&p.ps, D3D12_SHADER_VISIBILITY_PIXEL)];
    let groot = eh.root_signature_records(&gstages, &[(false, RECORD_REGISTER)])?;
    let (cp, gp) = (bridge::params(&[&p.ms], 3), bridge::params(&[&vs, &p.ps], 1));
    // `ourlight=metal`: target 2 (the title's F0, `PS_DeferredTextured` 3ed68103 0x10f0..0x11ac) into our metal target.
    // `ournormals=title`: target 1 (1 - roughness, the normal word: 3ed68103 0xf0c..0xfb0) into the title normal target.
    // Which of its targets those are is the title pipeline's target set's to say ([`lit_targets`]), not the index's.
    let writes = |k: Option<u32>| k.is_some_and(|k| bridge::targets(&p.ps.dxil).contains(&(k, DXGI_FORMAT_R32G32B32A32_FLOAT.0)));
    // A capture without the pipeline's targets: a program with fewer than five targets is not the opaque set's.
    let (normal, f0) = if title.is_empty() && bridge::targets(&p.ps.dxil).len() < 5 { (None, None) } else { lit_targets(title) };
    // `emissive=on` (LGT1): target 5 of the opaque set (the title's RT5), as the program writes it. Not the decal set's:
    // the title only scales RT5's alpha under a decal (DECAL_TARGETS), and their rows drawn without that blend wrote
    // their raw target 5 as emissive (pk7d f1067: ground litter lit, LGT1).
    let emissive = title == LIT_SETS[0].0 || (title.is_empty() && bridge::targets(&p.ps.dxil).len() >= 6);
    let metal = metal_on().then(|| (writes(f0), normals_on().then(|| writes(normal)), emissive_on() && emissive && writes(Some(5))));
    let ps_hash = xnw_core::hash128(&[psc]);
    let gpso = graphics(dev, &groot, &p.vs, &p.ps.dxil, [0; 8], false, metal, (0, 0), ps_hash)?;
    let lay = Lay::new(&p, &used, &image_reads(&[msc, psc]), dims_read(&[msc, psc]));
    let vs = p.vs.clone();
    Ok(Built { programs: p, used, croot, cpso, groot, gpso, cp, gp, blended: Default::default(), vs, lay, metal, ms: t0.elapsed().as_secs_f64() * 1e3, ps_hash })
}

/// The title's target sets (its pipelines' render-target formats, DXGI) and
/// which target holds what our lighting pass takes: (formats, the packed
/// normal with 1 - roughness, F0). A set that is not here gives neither (its
/// rows keep our own normal and no metal); no formats (a replay without the
/// pipeline's state): the opaque set, as before WIDE1.
/// Evidence: `out\cache\pipelines\pipeline-list.txt` (3ed68103, d064f214, d5f6eb33).
const LIT_SETS: &[(&[u32], Option<u32>, Option<u32>)] = &[
    // Opaque G-buffer: albedo sRGB, normal, F0, occlusion/ids, velocity R16G16, RT5 (meaning unknown).
    (&[29, 28, 28, 28, 34, 28], Some(1), Some(2)),
    // Decals: target 4 is the normal as n * 0.5 + 0.5 (materials.md); targets 1 and 2 as before.
    (&[29, 28, 28, 28, 28, 28], Some(1), Some(2)),
    // Foliage (trees, imposters): albedo, a normal target whose packing is not read yet, velocity at 2.
    (&[29, 28, 34], None, None),
];

/// (normal target, F0 target) of the title target set `title` ([`LIT_SETS`]).
pub fn lit_targets(title: &[u32]) -> (Option<u32>, Option<u32>) {
    if title.is_empty() {
        return (Some(1), Some(2));
    }
    LIT_SETS.iter().find(|s| s.0 == title).map_or((None, None), |s| (s.1, s.2))
}

/// A title blend of target 0 our pipelines take: enabled, factors and
/// operations in D3D12's ranges (the rule of our own draws' state pipelines).
pub fn blend_ok(b: [u8; 8]) -> bool {
    b[0] != 0 && [b[1], b[2], b[4], b[5]].iter().all(|v| (1..=19).contains(v)) && [b[3], b[6]].iter().all(|v| (1..=5).contains(v))
}

/// The title's blend of a stage-74 decal pipeline's six targets beyond
/// target 0's factors: (write mask, source alpha, destination alpha), D3D12
/// numbering. Every decal pipeline of `standard_decal`,
/// `standard_pxm_decal_ground` and `standard_decal_blend_ground`
/// (PS d064f214, 46bfb5fd, e01e4d6f) in the warm pipeline cache
/// blends
/// all six with SRC_ALPHA / INV_SRC_ALPHA / ADD on colour and these masks:
/// target 1 R only (1 - roughness; the normal word stays), target 2 RGB
/// target 3 one or two of R/G by pipeline (3 here), target 4 RGBA
/// (n x 0.5 + 0.5, alpha x (1 - a)), target 5 alpha only. The capture's
/// `RenderState` keeps target 0 alone (lib.rs `blend`), and the pack's
/// `gstate` covers Draw/DrawIndexed sinks only, so the rule stands in
/// for the per-target state of mesh decal rows.
pub const DECAL_TARGETS: [(u8, u8, u8); 6] = [(0x7, 2, 1), (0x1, 2, 1), (0x7, 2, 6), (0x3, 2, 1), (0xF, 1, 6), (0x8, 1, 6)];

/// A title blend of target 0 that is a decal's ([`DECAL_TARGETS`]):
/// SRC_ALPHA / INV_SRC_ALPHA / ADD into RGB.
/// `XNW_DECAL_TARGETS=off` (A/B): decal rows write target 0 alone, as before SHD1.
/// Live (SW1 `decal`): the pipeline cache keys on it ([`Built::pixel_pipeline`]).
pub fn decal_blend(b: [u8; 8]) -> bool {
    xnw_core::live::on(xnw_core::live::DECAL, || u32::from(!std::env::var("XNW_DECAL_TARGETS").is_ok_and(|v| v.trim() == "off"))) && blend_ok(b) && b[1] == 5 && b[2] == 6 && b[3] == 1 && b[7] & 0xF == 0x7
}

/// Target `k`'s blend for a decal row whose target 0 blends `b` ([`DECAL_TARGETS`]).
pub fn decal_target(b: [u8; 8], k: usize) -> D3D12_RENDER_TARGET_BLEND_DESC {
    let (mask, src_a, dst_a) = DECAL_TARGETS[k.min(5)];
    D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: true.into(),
        SrcBlend: D3D12_BLEND(i32::from(b[1])),
        DestBlend: D3D12_BLEND(i32::from(b[2])),
        BlendOp: D3D12_BLEND_OP(i32::from(b[3])),
        SrcBlendAlpha: D3D12_BLEND(i32::from(src_a)),
        DestBlendAlpha: D3D12_BLEND(i32::from(dst_a)),
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        RenderTargetWriteMask: mask,
        ..Default::default()
    }
}

/// FLK1 `bridgeulps`, `bridgeslope` (live; `XNW_BRIDGE_ULPS=n`,
/// `XNW_BRIDGE_SLOPE=n` start them, default 0): the bridged pipelines'
/// depth bias towards the viewer (reverse Z, test GREATER_EQUAL against
/// our depth). A row's depth differs from ours by its own vertex path's
/// rounding and by the depth slope times a part of a pixel, both in
/// proportion to the depth itself (z = near / d): D3D12's bias on a float
/// depth is that shape, `ulps` last places of the primitive's largest
/// depth plus `slope / 4` times its largest depth slope a pixel. The
/// viewport's constant bias (`bridgebias`) is the same number at every
/// distance: 1e-5 forgives 0.7 mm at 1 m and 6.7 m at 100 m.
pub fn rel_bias() -> (i32, u32) {
    let start = |name: &'static str| move || std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0);
    (xnw_core::live::value(xnw_core::live::BRIDGEULPS, start("XNW_BRIDGE_ULPS")) as i32, xnw_core::live::value(xnw_core::live::BRIDGESLOPE, start("XNW_BRIDGE_SLOPE")))
}

impl Built {
    /// The pixel pipeline for a row with title blend `b` (made once per
    /// blend and [`rel_bias`]).
    fn pixel_pipeline(&self, dev: &ID3D12Device, b: [u8; 8]) -> Result<ID3D12PipelineState, String> {
        let rel = rel_bias();
        if !blend_ok(b) && rel == (0, 0) {
            return Ok(self.gpso.clone());
        }
        let b = if blend_ok(b) { b } else { [0; 8] };
        let mut m = self.blended.lock().unwrap_or_else(|e| e.into_inner());
        let key = (b, decal_blend(b), rel);
        if let Some(p) = m.get(&key) {
            return Ok(p.clone());
        }
        let p = graphics(dev, &self.groot, &self.vs, &self.programs.ps.dxil, b, key.1, self.metal, rel, self.ps_hash)?;
        m.insert(key, p.clone());
        Ok(p)
    }
}

/// The pass-through vertex program and pixel program `ps` into our colour
/// target and depth (GREATER_EQUAL, no write); with a title blend
/// ([`blend_ok`]) its factors, operations and write mask on target 0, as
/// `pipeline_state_with` applies them to our own draws. `metal`
/// (`ourlight=metal`): three targets, target 2 into our metal target
/// ([`crate::ourlight::METAL`]) where the program writes it and the row
/// does not blend: its colour as written, its alpha 0 whatever the
/// program says (blend factors zero over a target cleared to alpha 1:
/// alpha 0 marks the pixels a bridged row wrote). With `ournormals=title`
/// target 1 into the title normal target as written, on the same rows.
/// With `emissive=on` six targets: target 5 into our emissive target
/// ([`crate::ourlight::EMISSIVE`]) where an opaque-set program writes it and
/// the row does not blend, as written; targets 3 and 4 none. `rel` ([`rel_bias`]): the depth bias towards the viewer. `ps_hash`: the
/// pixel program's ([`crate::reduce::pipeline`] reduces only its pipelines).
#[allow(clippy::too_many_arguments)]
fn graphics(dev: &ID3D12Device, groot: &ID3D12RootSignature, vs: &[u8], ps: &[u8], b: [u8; 8], decal: bool, metal: Option<Lit>, rel: (i32, u32), ps_hash: u128) -> Result<ID3D12PipelineState, String> {
    // SAFETY: a description whose pointers outlive the call.
    unsafe {
        let mut g = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(groot),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), DepthBias: rel.0, SlopeScaledDepthBias: rel.1 as f32 * 0.25, ..Default::default() },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC { DepthEnable: true.into(), DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO, DepthFunc: D3D12_COMPARISON_FUNC_GREATER_EQUAL, ..Default::default() },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: shown_target() + 1,
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let rt = &mut g.BlendState.RenderTarget[0];
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        if blend_ok(b) {
            rt.RenderTargetWriteMask = b[7] & 0xF;
            rt.BlendEnable = true.into();
            (rt.SrcBlend, rt.DestBlend, rt.BlendOp) = (D3D12_BLEND(i32::from(b[1])), D3D12_BLEND(i32::from(b[2])), D3D12_BLEND_OP(i32::from(b[3])));
            (rt.SrcBlendAlpha, rt.DestBlendAlpha, rt.BlendOpAlpha) = (D3D12_BLEND(i32::from(b[4])), D3D12_BLEND(i32::from(b[5])), D3D12_BLEND_OP(i32::from(b[6])));
        }
        g.RTVFormats[shown_target() as usize] = bridge::INTO_COLOUR;
        // A decal row shown at target k > 0 (the lab view): that target's own title blend.
        let k = shown_target() as usize;
        if k > 0 && decal {
            g.BlendState.IndependentBlendEnable = true.into();
            g.BlendState.RenderTarget[k] = decal_target(b, k);
        }
        if let Some((writes, normals, emits)) = metal {
            g.NumRenderTargets = 3;
            g.RTVFormats[2] = crate::ourlight::METAL;
            g.BlendState.IndependentBlendEnable = true.into();
            // `emissive=on`: every bridged pipeline has the emissive target at 5 (one binding for the batch); only writers write it.
            if emissive_on() {
                g.NumRenderTargets = 6;
                g.RTVFormats[5] = crate::ourlight::EMISSIVE;
                if emits && !blend_ok(b) {
                    g.BlendState.RenderTarget[5].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
                }
            }
            if let Some(n) = normals {
                g.RTVFormats[1] = crate::ourlight::TITLE_NORMAL;
                if n && !blend_ok(b) {
                    g.BlendState.RenderTarget[1].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
                } else if n && decal {
                    // The decal's 1 - roughness (R) over the row below; its normal word stays (SHD1).
                    g.BlendState.RenderTarget[1] = decal_target(b, 1);
                }
            }
            if writes && !blend_ok(b) {
                g.BlendState.RenderTarget[2] = D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: true.into(),
                    SrcBlend: D3D12_BLEND_ONE,
                    DestBlend: D3D12_BLEND_ZERO,
                    BlendOp: D3D12_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D12_BLEND_ZERO,
                    DestBlendAlpha: D3D12_BLEND_ZERO,
                    BlendOpAlpha: D3D12_BLEND_OP_ADD,
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
            } else if writes && decal {
                // The decal's F0 (RGB) over the row below; alpha (the bridged mark) stays.
                g.BlendState.RenderTarget[2] = decal_target(b, 2);
            }
        }
        crate::reduce::pipeline(&mut g, ps_hash);
        dev.CreateGraphicsPipelineState::<ID3D12PipelineState>(&g).map_err(|x| format!("graphics pipeline: {}", x.message()))
    }
}

enum Entry {
    Building,
    Ready(Arc<Built>),
    Failed,
}

/// The pipelines of the bridged pairs, made once each.
pub struct Cache {
    map: Mutex<HashMap<Key, Entry>>,
}

struct Device(ID3D12Device, xnw_gpu::host::Caps);
// SAFETY: D3D12 devices are free-threaded.
unsafe impl Send for Device {}

/// The process's cache (live and replay).
pub fn cache() -> &'static Cache {
    static C: OnceLock<Cache> = OnceLock::new();
    C.get_or_init(|| Cache { map: Mutex::new(HashMap::new()) })
}

impl Cache {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Key, Entry>> {
        self.map.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The pair's pipelines when made (None: building, failed or never asked).
    pub fn get(&self, key: Key) -> Option<Arc<Built>> {
        match self.lock().get(&key) {
            Some(Entry::Ready(b)) => Some(b.clone()),
            _ => None,
        }
    }

    /// Whether `key` was asked for already (made, building or failed).
    pub fn asked(&self, key: Key) -> bool {
        self.lock().contains_key(&key)
    }

    /// Makes the pair's pipelines once: on a thread of its own (`wait`
    /// false: the live pass never stalls on it) or here.
    /// `title`: the title pipeline's target formats ([`lit_targets`]; empty: not known).
    pub fn ask(&'static self, key: Key, msc: Vec<u8>, psc: Vec<u8>, title: &[u32], dev: &ID3D12Device, caps: &xnw_gpu::host::Caps, wait: bool) {
        {
            let mut m = self.lock();
            if m.contains_key(&key) {
                return;
            }
            m.insert(key, Entry::Building);
        }
        let d = Device(dev.clone(), caps.clone());
        let title = title.to_vec();
        let work = move || {
            let d = d;
            let e = match build(&d.0, &d.1, &msc, &psc, &title) {
                Ok(b) => {
                    xnw_core::info!("bridgeps: pipelines of pixel program {:08x} (mesh {:08x}) made in {:.0} ms; the pair reads images as {}; title targets {:?}: writes metal, normal, emissive {:?}", (key.1 >> 96) as u32, (key.0 >> 96) as u32, b.ms, dims_text(b.lay.dims_read), title, b.metal);
                    Entry::Ready(Arc::new(b))
                }
                Err(e) => {
                    xnw_core::warn!("bridgeps: pixel program {:08x} (mesh {:08x}): {e}: its rows stay ours", (key.1 >> 96) as u32, (key.0 >> 96) as u32);
                    Entry::Failed
                }
            };
            self.lock().insert(key, e);
        };
        if wait {
            work();
        } else if std::thread::Builder::new().name("xnw-bridgeps".into()).spawn(move || {
            xnw_core::win::helper_thread();
            work()
        }).is_err() {
            self.lock().insert(key, Entry::Failed);
        }
    }

    /// `made M, building B, failed F`.
    pub fn counts(&self) -> [usize; 3] {
        let m = self.lock();
        let mut n = [0; 3];
        for e in m.values() {
            n[match e {
                Entry::Ready(_) => 0,
                Entry::Building => 1,
                Entry::Failed => 2,
            }] += 1;
        }
        n
    }
}

/// What one [`Gpu::record`] did.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// The first refusal's reason.
    pub why: String,
    pub drawn: u32,
    /// Rows whose pipelines are not made (yet, or failed).
    pub no_pipeline: u32,
    /// Rows the binding refused, or without a group count.
    pub refused: u32,
    /// Textures from our table, and grey stand-ins.
    pub textures: u32,
    pub grey: u32,
    /// The grey ones by our texture table's reason ([`super::textures::FLAT_REASONS`]).
    pub grey_why: [u32; 7],
    /// The grey ones by the dimension of their view ([`DIMS`]' order).
    pub grey_dims: [u32; 6],
    /// Of `grey`, the zero stand-ins ([`ZERO`]).
    pub zero: u32,
    /// Of `textures`, those bound through a cube, a 2D array or a 3D view.
    pub shaped: u32,
    /// CLOTH2: rows drawn with their own slot-20 block ([`Region`]), and the indices rewritten in them.
    pub regions: [u32; 2],
    /// LAMP1: rows drawn with their mask words in our slot 86 ([`Owned::mask`]).
    pub masks: u32,
    /// CLOTH3: the census of the rewritten words of those rows: by the
    /// class of the view written ([`REGION_CLASSES`]), the block indices
    /// left unrewritten (title view not plain 2D), and the formats of the
    /// real views (format, count).
    pub region_views: [u32; 8],
    pub region_left: u32,
    pub region_formats: Vec<(u32, u32)>,
    /// Buffers uploaded this frame and their bytes (each distinct buffer once).
    pub buffers: u32,
    pub bytes: u64,
    /// Inputs kept on the device: bytes in use, and bytes copied there this frame.
    pub kept: u64,
    pub kept_new: u64,
    /// `XNW_ZEROCOPY=on`: inputs bound in place in title memory (views,
    /// bytes) instead of copied (ZC1).
    pub in_place: u32,
    pub in_place_bytes: u64,
    /// View heap slots this frame.
    pub slots: u32,
    pub ms: f64,
    /// Milliseconds: bindings, uploads, descriptors, list calls.
    pub parts: [f64; 4],
    /// Descriptor writes by [`WRITES`] kind.
    pub writes: [u32; WRITES.len()],
    /// The descriptors part by [`VIEW_PARTS`], milliseconds.
    pub view_ms: [f64; VIEW_PARTS.len()],
    /// Table blocks ([`tables_on`]): the rows' blocks, distinct this frame,
    /// kept from an earlier frame, made this frame; their slots: all rows',
    /// made; flushes of the block cache.
    pub blocks: [u32; 4],
    pub block_slots: [u32; 2],
    pub flushes: u32,
    /// Blocks by root (-1: the draw words' textures): rows', distinct, kept from an earlier frame.
    pub by_root: Vec<(i32, [u32; 3])>,
    /// Table blocks taken from an earlier row of the frame without cutting.
    pub uncut: u32,
    /// [`small_on`]: small inputs placed, made this frame, bytes made.
    pub small: [u32; 2],
    pub small_bytes: u64,
    /// [`steady_on`]: inputs at their draw's own slot, of them rewritten, bytes written.
    pub steady: [u32; 2],
    pub steady_bytes: u64,
    /// The small inputs' buffer: its end and size.
    pub small_end: [u64; 2],
    /// The bindings part: [`bridge::bind_at`], placing the inputs, of
    /// that the in-place lookups, the small inputs, the kept or frame ones (milliseconds).
    pub bind_ms: [f64; 5],
    /// Rows drawn with their title blend ([`blend_ok`]).
    pub blended: u32,
    /// Rows whose title blend is on but not one our pipelines take (drawn
    /// without it), and the frame's distinct title blends with their rows (8 at most).
    pub blend_refused: u32,
    pub blends: Vec<([u8; 8], u32)>,
    /// [`hold_mode`]: rows bound from a held binding, bindings held
    /// this frame, held bindings unlike a fresh one (`check`).
    pub held: [u32; 3],
    /// [`replay_mode`]: rows replayed from last frame (`check`: that
    /// would have been), of them unlike the fresh binding (`check`).
    pub replayed: [u32; 2],
    /// Replays left for the next frame, held rows that left none
    /// (an input without a version), those inputs; held rows not replayed:
    /// no replay, not last frame's or its blocks flushed, shape, samplers,
    /// handles or groups changed, textures changed, an input moved, a
    /// constant missing.
    pub replay_why: [u32; 9],
    /// PERF1 (the row cache's hit rate, by our key: [`ident`], pipeline,
    /// occurrence): rows with a key, of them bound from a binding held on
    /// an earlier frame or replayed, of those the rows that wrote no
    /// small input and staged no kept copy (every input's bytes are last
    /// frame's; a frame table's first row carries its writes).
    pub row_key: [u32; 3],
}

impl Stats {
    /// Adds a later [`Gpu::record`]'s counts of the same frame (the pass's
    /// `chunk`: the rows in several records): counts and times sum;
    /// the view heap slots and kept bytes are the largest; the first
    /// refusal's reason stays.
    pub fn add(&mut self, o: &Stats) {
        if self.why.is_empty() {
            self.why.clone_from(&o.why);
        }
        self.drawn += o.drawn;
        self.blended += o.blended;
        self.blend_refused += o.blend_refused;
        for (b, k) in &o.blends {
            match self.blends.iter().position(|x| x.0 == *b) {
                Some(i) => self.blends[i].1 += k,
                None if self.blends.len() < 8 => self.blends.push((*b, *k)),
                None => {}
            }
        }
        self.no_pipeline += o.no_pipeline;
        self.refused += o.refused;
        self.textures += o.textures;
        self.grey += o.grey;
        self.zero += o.zero;
        self.regions[0] += o.regions[0];
        self.regions[1] += o.regions[1];
        self.masks += o.masks;
        self.region_left += o.region_left;
        for (a, b) in self.region_views.iter_mut().zip(o.region_views) {
            *a += b;
        }
        for &(f, n) in &o.region_formats {
            region_format(&mut self.region_formats, f, n);
        }
        self.shaped += o.shaped;
        for (a, b) in self.grey_dims.iter_mut().zip(o.grey_dims) {
            *a += b;
        }
        self.buffers += o.buffers;
        self.bytes += o.bytes;
        self.kept = self.kept.max(o.kept);
        self.kept_new += o.kept_new;
        self.in_place += o.in_place;
        self.in_place_bytes += o.in_place_bytes;
        self.slots = self.slots.max(o.slots);
        self.ms += o.ms;
        self.flushes += o.flushes;
        self.uncut += o.uncut;
        self.small_bytes += o.small_bytes;
        let sum = |a: &mut [u32], b: &[u32]| a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        sum(&mut self.grey_why, &o.grey_why);
        sum(&mut self.writes, &o.writes);
        sum(&mut self.blocks, &o.blocks);
        sum(&mut self.block_slots, &o.block_slots);
        sum(&mut self.small, &o.small);
        sum(&mut self.held, &o.held);
        sum(&mut self.replayed, &o.replayed);
        sum(&mut self.replay_why, &o.replay_why);
        let sumf = |a: &mut [f64], b: &[f64]| a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        sumf(&mut self.parts, &o.parts);
        sumf(&mut self.view_ms, &o.view_ms);
        sumf(&mut self.bind_ms, &o.bind_ms);
        for (r, c) in &o.by_root {
            match self.by_root.iter_mut().find(|x| x.0 == *r) {
                Some(x) => sum(&mut x.1, c),
                None => self.by_root.push((*r, *c)),
            }
        }
    }
}

/// [`Stats::writes`]: typed nulls copied (slots), guards, null images,
/// raw, structured, typed, constant buffer views, the words buffer view,
/// textures, grey stand-ins.
pub const WRITES: [&str; 10] = ["nulls", "guards", "null images", "raw", "structured", "typed", "cbv", "words", "textures", "grey"];
/// [`Stats::view_ms`]: typed nulls, guards and null images, buffer views,
/// words and textures, samplers, blocks looked up in the cache, rows cut
/// into blocks with their keys, our texture table's lookups.
pub const VIEW_PARTS: [&str; 8] = ["nulls", "guards", "buffers", "textures", "samplers", "lookups", "cut", "texture lookups"];

/// `XNW_BRIDGE_TABLES` (default on; `off`: every row writes its whole
/// view block every frame, as did): each table root of a row, and its
/// draw-word textures, is one block of the view heap keyed by the exact
/// descriptors it holds; a block already in the heap (this frame's or an
/// earlier frame's) is bound where it is, with no write (after
/// victory-fork `State::tables`).
pub fn tables_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("XNW_BRIDGE_TABLES").as_deref(), Ok("off" | "0")))
}

/// What a [`Gpu::record`] needs of our pass.
pub struct Target<'a> {
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub width: u32,
    pub height: u32,
    /// The rows' viewport origin in pixels (FLK1, `unjitter_px`: the
    /// title's TAA jitter taken out, so they land on our depth).
    pub shift: [f32; 2],
    /// The rows' viewport MinDepth (FLK1 `bridgebias`; 0: none).
    pub bias: f32,
    /// List rows left out this frame, sorted (FLK1 `bridgecam=1`: their
    /// view constants are another camera's than our depth's).
    pub skip: &'a [u32],
    /// A texture by handle: its host copy and view this frame.
    pub texture: &'a dyn Fn(u64) -> Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)>,
    /// Why our table has no copy of a texture this frame (a [`super::textures::Flat`]).
    pub flat: &'a dyn Fn(u64) -> usize,
    /// A 1x1 texture in a shader-resource state (the grey stand-in).
    pub grey: &'a ID3D12Resource,
    /// The grey stand-ins of the other shapes, each one texel a slice, in a
    /// shader-resource state: six slices (cube views), a 3D and a 1D
    /// texture. A 2D array view is a view of `grey`. Then the zero ones
    /// ([`ZERO`]): a 2D and a 3D texel.
    pub grey_shapes: [&'a ID3D12Resource; 5],
    /// The words buffer (upload memory), its bytes and its mapping: each
    /// drawn row's words are written there at the row's offset.
    pub words: (&'a ID3D12Resource, u64, *mut u8),
    /// `ourlight=metal`: the lighting pass's metal target (target 2).
    pub metal: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    /// `ournormals=title`: its title normal target (target 1).
    pub normal: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    /// `emissive=on`: its emissive target (target 5).
    pub emissive: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    /// `preskinbind=on`: our pre-skin output buffer and the addresses
    /// of the stand-in bytes rows hand over for it (`preskin_gpu::bind_rows`):
    /// a view of such bytes is a view of that buffer from its first byte.
    pub preskin: Option<(&'a ID3D12Resource, &'a [usize])>,
}

/// A grey texture (128, 128, 128, 255) of one texel a slice in the shape a
/// stand-in of dimension `dim` ([`view_dim`]) views: six slices for a cube
/// (9), a 3D texture (8), a 1D texture (2, 3), else one 2D texel; with
/// [`ZERO`] the texel is (0, 0, 0, 0). Filled
/// on the open list `l` and left in a shader-resource state; the second
/// resource is its upload, which the caller keeps until the list has run.
pub fn grey_texture(dev: &ID3D12Device, l: &ID3D12GraphicsCommandList, dim: u8) -> Result<(ID3D12Resource, ID3D12Resource), String> {
    let e = |x: windows::core::Error| x.message().to_string();
    let texel: [u8; 4] = if dim & ZERO != 0 { [0; 4] } else { [128, 128, 128, 255] };
    let (dimension, slices) = match dim & !ZERO {
        9 => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 6u16),
        8 => (D3D12_RESOURCE_DIMENSION_TEXTURE3D, 1),
        2 | 3 => (D3D12_RESOURCE_DIMENSION_TEXTURE1D, 1),
        _ => (D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1),
    };
    let desc = D3D12_RESOURCE_DESC { Dimension: dimension, Width: 1, Height: 1, DepthOrArraySize: slices, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
    let bdesc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 256, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
    // SAFETY: our device and open list; the descriptions outlive the calls,
    // four bytes are written into the 256-byte upload's mapping.
    unsafe {
        let mut t: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut t).map_err(e)?;
        let t = t.ok_or("no grey texture")?;
        let mut up: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up).map_err(e)?;
        let up = up.ok_or("no grey upload")?;
        let mut at = std::ptr::null_mut();
        up.Map(0, None, Some(&mut at)).map_err(e)?;
        std::ptr::copy_nonoverlapping(texel.as_ptr(), at.cast::<u8>(), 4);
        up.Unmap(0, None);
        let fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: 1, Height: 1, Depth: 1, RowPitch: 256 } };
        for slice in 0..u32::from(slices) {
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::ManuallyDrop::new(Some(t.clone())), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: slice } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::ManuallyDrop::new(Some(up.clone())), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
            l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            drop(std::mem::ManuallyDrop::into_inner(dst.pResource));
            drop(std::mem::ManuallyDrop::into_inner(src.pResource));
        }
        l.ResourceBarrier(&[transition(&t, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
        Ok((t, up))
    }
}

fn transition(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            // SAFETY: a borrowed pointer the barrier never releases.
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: unsafe { std::mem::transmute_copy(r) }, Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: before, StateAfter: after }),
        },
    }
}

/// Sampler heap slots of a row at least (the title's indices reach 105 in a captured frame).
const SAMPLER_SLOTS: u32 = 128;
/// View heap slots: least, and most (resource binding tier 2 and up).
const HEAP_LEAST: u32 = 1 << 16;
const HEAP_MOST: u32 = 1_000_000;
/// Stride table entries: every slot up to [`NO_VIEW`].
const STRIDE_ENTRIES: u64 = NO_VIEW as u64 + 1;
/// Sampler heaps kept by content at most.
const SAMPLER_HEAPS: usize = 64;
/// The state the kept buffer is read in.
const KEPT_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0);

/// FxHash's multiply-rotate step, eight bytes at a time: the keys below
/// are compared whole on a hit, so the hash need only spread them
/// (SipHash over them took 3-8 ms a frame).
#[derive(Default)]
pub(crate) struct Fx(u64);

impl std::hash::Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut c = bytes.chunks_exact(8);
        for w in &mut c {
            self.write_u64(u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]));
        }
        for &b in c.remainder() {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

/// A map by whole content key.
type ByContent<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<Fx>>;
/// [`ByContent`] for the bridge's per-input maps (SipHash there cost a
/// lookup's worth per input).
pub(crate) type FxMap<K, V> = ByContent<K, V>;

/// A hasher that takes a key hashed once already.
#[derive(Default)]
struct Hashed(u64);

impl std::hash::Hasher for Hashed {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}


/// The table blocks in the view heap by key: heads by hash, entries
/// chained, key words in one list; no allocation a block.
#[derive(Default)]
struct BlockCache {
    head: HashMap<u64, u32, std::hash::BuildHasherDefault<Hashed>>,
    /// Key words, (first slot, slots, last frame), next entry of the hash (`u32::MAX`: none).
    ents: Vec<(std::ops::Range<usize>, (u32, u32, u64), u32)>,
    words: Vec<u64>,
}

impl BlockCache {
    fn find(&self, h: u64, k: &[u64]) -> Option<usize> {
        let mut i = *self.head.get(&h)?;
        while i != u32::MAX {
            let e = &self.ents[i as usize];
            if self.words[e.0.clone()] == *k {
                return Some(i as usize);
            }
            i = e.2;
        }
        None
    }

    fn get_mut(&mut self, h: u64, k: &[u64]) -> Option<&mut (u32, u32, u64)> {
        let i = self.find(h, k)?;
        Some(&mut self.ents[i].1)
    }

    fn insert(&mut self, h: u64, k: &[u64], v: (u32, u32, u64)) {
        let w = self.words.len();
        self.words.extend_from_slice(k);
        let next = self.head.insert(h, self.ents.len() as u32).unwrap_or(u32::MAX);
        self.ents.push((w..self.words.len(), v, next));
    }

    fn is_empty(&self) -> bool {
        self.ents.is_empty()
    }

    fn clear(&mut self) {
        self.head.clear();
        self.ents.clear();
        self.words.clear();
    }
}

/// A sampler entry list's content key.
fn sampler_key(s: &[(u32, SamplerDesc)], slots: u32) -> Vec<u32> {
    let mut w: Vec<u32> = vec![slots];
    for (k, d) in s.iter().filter(|s| s.0 < slots) {
        w.extend([*k, d.Filter.0 as u32, d.AddressU.0 as u32, d.AddressV.0 as u32, d.AddressW.0 as u32, d.MipLODBias.to_bits(), d.MaxAnisotropy, d.ComparisonFunc.0 as u32]);
        w.extend(d.BorderColor.map(f32::to_bits));
        w.extend([d.MinLOD.to_bits(), d.MaxLOD.to_bits()]);
    }
    w
}

/// `XNW_BRIDGE_SMALL` (default on with [`tables_on`]; `off`: in the
/// frame's upload memory by order, as): inputs under [`KEEP_MIN`] in
/// [`Small`].
pub fn small_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| tables_on() && !matches!(std::env::var("XNW_BRIDGE_SMALL").as_deref(), Ok("off" | "0")))
}

/// Inputs under [`KEEP_MIN`] (constant buffers and the like) by content in
/// one upload buffer kept across frames: equal bytes lie at one
/// offset on every frame that uses them, so the table blocks naming them
/// stay equal across frames (in the frame's upload memory their offsets
/// moved with the rows' order: pk8a f1030 -> f1031 kept 0 of 164 mesh-table
/// blocks, all 164 equal but for where their inputs lay). Bytes are written
/// once; an entry no row used for [`SMALL_FRAMES`] frames is freed at a
/// frame's start (the pass waited for the last frame's fence), its place
/// reused by size and alignment. Pure bookkeeping: the caller writes.
#[derive(Default)]
pub struct Small {
    pub size: u64,
    pub end: u64,
    /// Entries by content hash: bytes, size, unit; offset, capacity, alignment, last frame used.
    at: HashMap<u64, Vec<(Box<[u8]>, u64, u64, [u64; 4])>, std::hash::BuildHasherDefault<Hashed>>,
    free: HashMap<(u64, u64), Vec<u64>>,
    frame: u64,
    /// This frame: entries placed, made, bytes made.
    pub placed: u32,
    pub made: u32,
    pub made_bytes: u64,
    /// [`steady_on`]: inputs by draw identity ([`IdKey`]).
    ids: HashMap<IdKey, IdEnt, std::hash::BuildHasherDefault<Fx>>,
    /// This frame: inputs at their draw's slot, of them rewritten, bytes written there.
    pub id_placed: u32,
    pub id_written: u32,
    pub id_bytes: u64,
    /// PERF1: the identities to look at on frame `f`, at `f % DUE`: each
    /// entry is listed once, for the first frame it can be freed on
    /// (`last + ID_FRAMES + 1`), and listed again from there while it is
    /// used: a frame looks at about one entry in [`ID_FRAMES`] (walking
    /// every entry each frame took 2.5 ms of a test run's frames).
    due: Vec<Vec<IdKey>>,
}

/// One input of a draw known across frames: the draw's identity
/// ([`Owned::ident`]), its pipeline, its occurrence in the frame (draws
/// of one identity in list order), the input (a table view's slot in the
/// row; root views `1 << 24 | parameter`, the pixel stage's `2 << 24 | ..`).
pub type IdKey = (u64, Key, u32, u32);

/// [`Small::place_steady`]'s entry: the bytes last placed (a CPU copy:
/// the upload buffer is write-combined), their padded size and unit, the
/// draw's own slot ([offset, capacity, alignment]), the last frame used.
struct IdEnt {
    shadow: Box<[u8]>,
    size: u64,
    unit: u64,
    slot: [u64; 3],
    last: u64,
    /// The frame it is looked at next ([`Small::due`]).
    due: u64,
}

/// Frames an input by draw identity stays unused before its slot is freed.
const ID_FRAMES: u64 = 8;
/// [`Small::due`]'s lists.
const DUE: usize = ID_FRAMES as usize + 2;

/// `XNW_BRIDGE_STEADY` (default on with [`small_on`]; `off`: by content
/// only, as): the small inputs of a draw known across frames
/// ([`Owned::ident`]) lie in slots of their own in [`Small`]'s buffer,
/// kept while the draw lasts; each frame only the bytes that changed are
/// written there and the descriptors stay the same, so the draw's table
/// blocks are the same blocks on every frame (its per-draw constants
/// change bytes and title address every frame: by content or address
/// they made a new mesh-table block a frame). Rows handing over the same
/// bytes (one frame table) share the first row's place.
pub fn steady_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| small_on() && !matches!(std::env::var("XNW_BRIDGE_STEADY").as_deref(), Ok("off" | "0")))
}

/// `XNW_BRIDGE_PLACE_SPLIT=on` (PERF1; default off): the bridgeps line's
/// `place: in place, small, kept or frame` split is timed (three clock
/// reads an input, about 27,000 a frame of a test run); off, it reads 0.0.
pub fn place_split_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("XNW_BRIDGE_PLACE_SPLIT").as_deref(), Ok("on" | "1")))
}

/// `XNW_BRIDGE_SHARED` (PERF1; default on with [`steady_on`] and
/// [`hold_mode`]; `off`: under the first row's identity, as before): small
/// inputs several rows hand over lie in slots of their own ([`small_id`]).
pub fn shared_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("XNW_BRIDGE_SHARED").as_deref(), Ok("off" | "0")))
}

/// `XNW_ZEROCOPY_CBV=on` (default off; with `XNW_ZEROCOPY=on`): constant
/// buffers in title memory are bound in place too (ZC2 kept them as
/// copies: hot data, every lane reads them).
pub fn zc_cbv_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("XNW_ZEROCOPY_CBV").as_deref(), Ok("on" | "1")))
}

/// The first and one past the last 64-byte piece where `a` and `b` (one length) differ.
fn changed(a: &[u8], b: &[u8]) -> Option<std::ops::Range<usize>> {
    if a == b {
        return None;
    }
    let n = a.len();
    let piece = |i: usize| a[i..(i + 64).min(n)] != b[i..(i + 64).min(n)];
    let first = (0..n).step_by(64).find(|&i| piece(i))?;
    let last = (0..n).step_by(64).rev().find(|&i| piece(i))?;
    Some(first..(last + 64).min(n))
}

impl Small {
    /// A new frame: entries unused for [`SMALL_FRAMES`] frames are freed.
    pub fn next_frame(&mut self) {
        self.frame += 1;
        let (f, free) = (self.frame, &mut self.free);
        self.at.retain(|_, v| {
            v.retain(|e| {
                let keep = e.3[3] + SMALL_FRAMES >= f;
                if !keep {
                    free.entry((e.3[1], e.3[2])).or_default().push(e.3[0]);
                }
                keep
            });
            !v.is_empty()
        });
        if self.due.len() != DUE {
            self.due = vec![Vec::new(); DUE];
        }
        let now = std::mem::take(&mut self.due[f as usize % DUE]);
        for id in &now {
            // An entry made again under its identity is listed twice: only its own frame counts.
            let Some(e) = self.ids.get_mut(id).filter(|e| e.due == f) else { continue };
            if e.last + ID_FRAMES >= f {
                e.due = e.last + ID_FRAMES + 1;
                self.due[e.due as usize % DUE].push(*id);
            } else {
                free.entry((e.slot[1], e.slot[2])).or_default().push(e.slot[0]);
                self.ids.remove(id);
            }
        }
        let mut now = now;
        now.clear();
        if self.due[f as usize % DUE].is_empty() {
            self.due[f as usize % DUE] = now;
        }
        (self.placed, self.made, self.made_bytes) = (0, 0, 0);
        (self.id_placed, self.id_written, self.id_bytes) = (0, 0, 0);
    }

    /// Starts again empty with `size` bytes (the caller makes the new buffer).
    pub fn start(&mut self, size: u64) {
        (self.size, self.end) = (size, 0);
        self.at.clear();
        self.free.clear();
        self.ids.clear();
        self.due.clear();
    }

    /// A free place of `cap` bytes at alignment `al` (None: the buffer is full).
    fn alloc(&mut self, cap: u64, al: u64) -> Option<u64> {
        if let Some(o) = self.free.get_mut(&(cap, al)).and_then(|v| v.pop()) {
            return Some(o);
        }
        let o = self.end.next_multiple_of(al);
        if o + cap > self.size {
            return None;
        }
        self.end = o + cap;
        Some(o)
    }

    /// Input `id` of a draw known across frames ([`steady_on`]): where it
    /// lies, the part of `bytes` the caller writes there now, and whether
    /// the place is new (the caller zeros its capacity past the bytes);
    /// None when the buffer has no room (the caller places it by content).
    /// The draw's input has a slot of its own from its first frame, kept
    /// while it lasts; each frame only the changed 64-byte pieces are
    /// written there (one compare with the last frame's bytes). An input
    /// placed twice in one frame (an identity that is not one) gets None.
    pub fn place_steady(&mut self, id: IdKey, bytes: &[u8], size: u64, unit: u64) -> Option<(u64, std::ops::Range<usize>, bool)> {
        let (f, n) = (self.frame, bytes.len().min(size as usize));
        let bytes = &bytes[..n];
        let (cap, al) = (Small::cap(size), Small::align(unit));
        let (o, w, fresh) = match self.ids.get_mut(&id) {
            Some(e) if e.last == f => return None,
            Some(e) if e.size == size && e.unit == unit && e.shadow.len() == n => {
                e.last = f;
                let w = changed(&e.shadow, bytes).unwrap_or(0..0);
                e.shadow[w.clone()].copy_from_slice(&bytes[w.clone()]);
                (e.slot[0], w, false)
            }
            old => {
                // New, or another input under this identity (its old slot freed).
                if let Some(e) = old {
                    let s = e.slot;
                    self.free.entry((s[1], s[2])).or_default().push(s[0]);
                    self.ids.remove(&id);
                }
                let o = self.alloc(cap, al)?;
                if self.due.len() != DUE {
                    self.due = vec![Vec::new(); DUE];
                }
                let due = f + ID_FRAMES + 1;
                self.due[due as usize % DUE].push(id);
                self.ids.insert(id, IdEnt { shadow: bytes.into(), size, unit, slot: [o, cap, al], last: f, due });
                (o, 0..n, true)
            }
        };
        self.id_placed += 1;
        if !w.is_empty() {
            self.id_written += 1;
            self.id_bytes += w.len() as u64;
        }
        Some((o, w, fresh))
    }

    /// Input `id`, whose bytes are the ones placed at `at` when it was
    /// last placed ([`Small::place_steady`]), is used this frame as it
    /// lies: false when that is not its slot now, or it is used already.
    pub fn touch(&mut self, id: IdKey, at: u64, size: u64, unit: u64, n: usize) -> bool {
        let f = self.frame;
        match self.ids.get_mut(&id) {
            Some(e) if e.last != f && e.slot[0] == at && e.size == size && e.unit == unit && e.shadow.len() == n => {
                e.last = f;
                self.id_placed += 1;
                true
            }
            _ => false,
        }
    }

    fn hash(bytes: &[u8], size: u64, unit: u64) -> u64 {
        let mut h = Fx::default();
        std::hash::Hasher::write(&mut h, bytes);
        std::hash::Hasher::write_u64(&mut h, size);
        std::hash::Hasher::write_u64(&mut h, unit);
        std::hash::Hasher::finish(&h)
    }

    /// Where `bytes` (`size` padded, element size `unit`) lie, and whether
    /// the caller writes them there now (`true`: their first frame there);
    /// None when the buffer has no room (the frame's upload memory then).
    pub fn place(&mut self, bytes: &[u8], size: u64, unit: u64) -> Option<(u64, bool)> {
        let h = Small::hash(bytes, size, unit);
        self.placed += 1;
        if let Some(e) = self.at.get_mut(&h).and_then(|v| v.iter_mut().find(|e| e.1 == size && e.2 == unit && *e.0 == *bytes)) {
            e.3[3] = self.frame;
            return Some((e.3[0], false));
        }
        let (cap, al) = (Small::cap(size), Small::align(unit));
        let Some(o) = self.alloc(cap, al) else {
            self.placed -= 1;
            return None;
        };
        self.at.entry(h).or_default().push((bytes.into(), size, unit, [o, cap, al, self.frame]));
        self.made += 1;
        self.made_bytes += cap;
        Some((o, true))
    }

    /// The capacity of the entry at `o` (its bytes past the input are zeros).
    fn cap(size: u64) -> u64 {
        size.max(16).next_power_of_two()
    }

    /// The start an input of element size `unit` needs here: a constant
    /// buffer (and a root view, unit 256) 256, a buffer view a multiple of
    /// its element and of 16 (a view's first element is in elements;
    /// [`bridge::align`]'s 256 made a 76-byte stride start at 4,864 B,
    /// and the draws' own slots of pk7a3 f961 then filled the buffer).
    fn align(unit: u64) -> u64 {
        let unit = unit.max(4);
        if unit % 256 == 0 { unit } else { 16 * unit / gcd(16, unit) }
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Frames a small input stays placed unused (the title alternates some
/// inputs frame to frame: pk7a3 f962 found 393 of f960's mesh-table blocks).
const SMALL_FRAMES: u64 = 4;
/// The frame's upload copy in pieces of this many bytes (threads past [`copy_min`]).
const PIECE: usize = 4 << 20;

/// `XNW_BRIDGE_COPY_MIN=<MiB>` (PROF1; default 32): the frame's upload copy
/// goes on up to 4 threads from this many MiB. Live street frames copy 3
/// buffers of 16.1 MiB a frame (a test run f5220: uploads 1.7 ms, f6180 3.3 ms),
/// under the 32 MiB rule: one thread into write-combined memory.
pub fn copy_min() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("XNW_BRIDGE_COPY_MIN").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(8 * PIECE, |m| m << 20))
}
/// The small inputs' buffer: least size.
const SMALL_LEAST: u64 = 32 << 20;

/// The bridged rows' device objects, kept by our pass across frames (it
/// waits for each frame's fence before the next is recorded): one
/// shader-visible view heap (each row at its own base:
/// [`bridge::bind_at`]), sampler heaps by content, the inputs handed over
/// again on a later frame in one default-heap buffer ([`Kept`]), one
/// mapped upload buffer for the rest, the mesh programs' records and indices.
#[derive(Default)]
pub struct Gpu {
    views: Option<(ID3D12DescriptorHeap, u32)>,
    /// A null render target (`XNW_BRIDGE_TARGET`: the targets before the one shown).
    null_rtv: Option<ID3D12DescriptorHeap>,
    /// A CPU heap of typed nulls: a row's slots are filled from it in one copy.
    nulls: Option<(ID3D12DescriptorHeap, u32)>,
    samplers: ByContent<Vec<u32>, (ID3D12DescriptorHeap, u64)>,
    up: Option<(ID3D12Resource, *mut u8, u64)>,
    /// The stride table of the whole view heap (mapped upload memory):
    /// [`STRIDE_ENTRIES`] entries, so that a read at any slot up to
    /// [`NO_VIEW`] stays inside it; the entries last written.
    strides: Option<(ID3D12Resource, *mut u8)>,
    strides_len: usize,
    out: Option<(ID3D12Resource, ID3D12Resource, u64, u64)>,
    /// The kept buffer and whether it is in its read state.
    kept_buf: Option<(ID3D12Resource, bool)>,
    pub kept: Kept,
    frame: u64,
    /// [`tables_on`]: the block heap and its slots, the blocks in it by
    /// key, the next free slot, the resources their descriptors name (held
    /// so that a pointer names one resource while its blocks live), and
    /// the buffers every block may name (upload, kept, words, grey: a
    /// change of any flushes the blocks).
    bheap: Option<(ID3D12DescriptorHeap, u32)>,
    blocks: BlockCache,
    bump: u32,
    /// The rows cut into blocks, lists kept across frames.
    cuts: Cuts,
    /// [`small_on`]: the small inputs and their mapped upload buffer.
    pub small: Small,
    small_buf: Option<(ID3D12Resource, *mut u8)>,
    held: HashMap<usize, windows::core::IUnknown>,
    bases: Vec<windows::core::IUnknown>,
    /// [`hold_mode`]: rows' bindings by draw identity.
    holds: HashMap<HoldKey, (Arc<Built>, Held), std::hash::BuildHasherDefault<Fx>>,
    /// [`hold_mode`]: in-place ranges by (title address, bytes), kept
    /// until their window is released or their pages remapped ([`Gpu::zc_changed`]).
    zc_memo: std::collections::BTreeMap<(u64, u64), xnw_gpu::shared::SharedRange>,
    /// Moves on whenever `zc_memo` drops a range ([`Ver::Shared`]).
    zc_gen: u64,
    /// Moves on at every flush of the block cache (a [`Replay`]'s blocks are gone).
    epoch: u64,
}

/// The block heap's first slots: the guards every row's bounds name
/// (null 2D texture, raw buffer SRV, raw buffer UAV, 2D texture UAV, CBV),
/// then the null images that are not 2D (BND1: `bridge::CONFINE_DIMS`
/// after the first; a row's confinement word 7 names the first of them).
const GUARDS: u32 = 5 + bridge::NULL_IMAGES;

/// One descriptor of a table block: the typed null of the slot, a
/// buffer view, a texture by handle, our words buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Desc {
    Null,
    /// A null image of this dimension ([`view_dim`]).
    NullImage(u8),
    Buf(Kind, At, u32),
    /// A texture by handle; the dimension its stand-in takes.
    Tex(u64, u8),
    Words,
}

/// One slot of a block that is not a typed null with a zero stride entry
/// Its slot in the block, its stride entry (a raw twin by its slot
/// in the block, plus one), its descriptor.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ent {
    j: u32,
    stride: [u32; 2],
    desc: Desc,
}

/// One block of a row: its root (-1: the draw words' textures), its slots,
/// its entries ([`Cuts::ents`]), its key's hash and words ([`Cuts::keys`];
/// None: not keyable, made afresh), its first slot.
struct Block {
    root: i32,
    len: u32,
    ents: std::ops::Range<usize>,
    key: Option<(u64, std::ops::Range<usize>)>,
    first: u32,
    /// The earlier block of the frame it was taken from uncut.
    from: Option<usize>,
    /// The draw words' block: its slots by the dimension of the view each
    /// holds ([`block_masks`]), set where the block is placed.
    masks: [u32; 6],
}

/// A frame's rows cut into blocks, in flat lists kept across frames (no
/// allocation a row): the blocks, their entries and key words, each
/// row's blocks; the rest is scratch.
#[derive(Default)]
struct Cuts {
    blocks: Vec<Block>,
    ents: Vec<Ent>,
    keys: Vec<u64>,
    rows: Vec<std::ops::Range<usize>>,
    /// The row's slots, block after block (descriptor and stride entry).
    dense: Vec<([u32; 2], Desc)>,
    /// Each block's first slot in `dense`.
    at: Vec<u32>,
    /// Raw twins: (twin's local slot, block, slot in it), by twin.
    twins: Vec<(u32, usize, u32)>,
    /// Regions of the binding: (start, span, block).
    regions: Vec<(u32, u32, usize)>,
}

impl Cuts {
    fn clear(&mut self) {
        self.blocks.clear();
        self.ents.clear();
        self.keys.clear();
        self.rows.clear();
    }

    fn key(&self, blk: &Block) -> Option<(u64, &[u64])> {
        blk.key.as_ref().map(|(h, r)| (*h, &self.keys[r.clone()]))
    }
}

/// A pipeline's table layout, made once with its pipelines: the
/// table roots the two programs read, in order; per root one past the
/// highest slot the programs name, and the slots they read through an
/// image descriptor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lay {
    pub roots: Vec<u8>,
    pub named: Vec<u32>,
    /// Per root: the slots read as images, each with the dimension the
    /// programs declare there ([`image_reads`]; 2D where not known).
    pub images: Vec<Vec<(u32, u8)>>,
    /// The dimensions the pair's programs read images in ([`dims_read`]).
    pub dims_read: u16,
}

impl Lay {
    /// The dimension of the stand-in for a texture whose title view has
    /// dimension `title` ([`view_dim`]; None: 2D): the title's where the
    /// programs read images of that dimension, else 2D.
    pub fn stand_in_dim(&self, title: Option<u8>) -> u8 {
        let zero = title.map_or(0, |d| d & ZERO);
        title.map(|d| d & !ZERO).filter(|d| self.dims_read & 1 << d != 0).unwrap_or(DIM_2D) | zero
    }

    /// The dimension the programs read table slot (`root`, `slot`) in.
    pub fn image_dim(&self, root: u16, slot: u32) -> u8 {
        let Some(k) = self.roots.iter().position(|&r| u16::from(r) == root) else { return DIM_2D };
        self.images[k].iter().find(|i| i.0 == slot).map_or(DIM_2D, |i| i.1)
    }

    pub fn new(p: &Programs, used: &Used, reads: &[(u16, u32, u8)], dims_read: u16) -> Lay {
        let roots = table_roots(&p.ms, &p.ps);
        let named = roots.iter().map(|&r| used.keys().filter(|u| u.0 == u16::from(r)).map(|u| u.1 + 1).max().unwrap_or(0)).collect();
        let images = roots
            .iter()
            .map(|&r| {
                let mut v: Vec<(u32, u8)> = used
                    .iter()
                    .filter(|u| *u.1 == NAMED_IMAGE && u.0.0 == u16::from(r))
                    .map(|u| (u.0.1, reads.iter().find(|x| x.0 == u16::from(r) && x.1 == u.0.1).map_or(DIM_2D, |x| x.2)))
                    .collect();
                v.sort_unstable();
                v
            })
            .collect();
        Lay { roots, named, images, dims_read }
    }
}

/// A stopwatch of the descriptors part's pieces.
struct Lap(std::time::Instant);

impl Lap {
    fn to(&mut self, ms: &mut f64) {
        let now = std::time::Instant::now();
        *ms += (now - self.0).as_secs_f64() * 1e3;
        self.0 = now;
    }
}

/// A buffer view's [`WRITES`] index.
fn write_kind(kind: Kind) -> usize {
    match kind {
        Kind::Raw | Kind::RawUav { .. } => 3,
        Kind::Structured(_) => 4,
        Kind::Typed { .. } => 5,
        Kind::Cbv => 6,
    }
}

/// A texture by handle as our texture table has it this frame.
type Texs = ByContent<u64, Option<(ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC)>>;

/// What a block's key resolves: textures as our table has them this frame,
/// the grey stand-in, the in-place ranges.
struct KeyCx<'a> {
    texs: &'a Texs,
    grey: u64,
    /// Handles whose title view is not 2D, with its dimension ([`Owned::dims`]).
    dims: &'a HashMap<u64, u8>,
    zc: &'a [xnw_gpu::shared::SharedRange],
}

/// The key words of one entry: its slot and stride entry, and the exact
/// descriptor (buffers by where they lie, textures by resource and view,
/// the grey stand-in); false when a texture's view is not one this key
/// spells out (2D, 2D array, cube, cube array, 3D), or an in-place
/// view's range is unknown: the block is then made afresh.
fn key_words(e: &Ent, cx: &KeyCx, w: &mut Vec<u64>) -> bool {
    w.push(u64::from(e.j) | u64::from(e.stride[1]) << 32);
    let tag = |t: u64| u64::from(e.stride[0]) | t << 32;
    match e.desc {
        Desc::Null => w.push(tag(0)),
        Desc::NullImage(dim) => w.push(tag(1 | u64::from(dim) << 8)),
        Desc::Words => w.push(tag(2)),
        Desc::Buf(kind, at, n) => {
            let k = match kind {
                Kind::Raw => 0,
                Kind::Structured(s) => 1 | u64::from(s) << 8,
                Kind::Typed { format, bytes } => 2 | u64::from(format as u32) << 8 | u64::from(bytes) << 40,
                Kind::Cbv => 3,
                Kind::RawUav { len } => 4 | u64::from(len) << 8,
            };
            match at {
                At::Frame(o) => w.extend([tag(3), k, u64::from(n), o]),
                At::Kept(o) => w.extend([tag(4), k, u64::from(n), o]),
                At::Small(o) => w.extend([tag(8), k, u64::from(n), o]),
                At::Shared(i) => {
                    let Some(r) = cx.zc.get(i as usize) else { return false };
                    w.extend([tag(5), k, u64::from(n), windows::core::Interface::as_raw(&r.buffer) as u64, r.offset]);
                }
            }
        }
        Desc::Tex(h, dim) => match real(cx.texs, h, dim) {
            Some((tr, desc)) => {
                // 2D, 2D array, cube, cube array and 3D views (the
                // texture table's views of the title's arrays, cubes and volumes).
                let pair = |a: u32, b: u32| u64::from(a) | u64::from(b) << 32;
                // SAFETY: each arm reads the member its view dimension names.
                let u: [u64; 3] = unsafe {
                    match desc.ViewDimension {
                        D3D12_SRV_DIMENSION_TEXTURE2D => {
                            let t = desc.Anonymous.Texture2D;
                            [pair(t.MostDetailedMip, t.MipLevels), pair(t.PlaneSlice, t.ResourceMinLODClamp.to_bits()), 0]
                        }
                        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                            let t = desc.Anonymous.Texture2DArray;
                            [pair(t.MostDetailedMip, t.MipLevels), pair(t.FirstArraySlice, t.ArraySize), pair(t.PlaneSlice, t.ResourceMinLODClamp.to_bits())]
                        }
                        D3D12_SRV_DIMENSION_TEXTURECUBE => {
                            let t = desc.Anonymous.TextureCube;
                            [pair(t.MostDetailedMip, t.MipLevels), u64::from(t.ResourceMinLODClamp.to_bits()), 0]
                        }
                        D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                            let t = desc.Anonymous.TextureCubeArray;
                            [pair(t.MostDetailedMip, t.MipLevels), pair(t.First2DArrayFace, t.NumCubes), u64::from(t.ResourceMinLODClamp.to_bits())]
                        }
                        D3D12_SRV_DIMENSION_TEXTURE3D => {
                            let t = desc.Anonymous.Texture3D;
                            [pair(t.MostDetailedMip, t.MipLevels), u64::from(t.ResourceMinLODClamp.to_bits()), 0]
                        }
                        _ => return false,
                    }
                };
                w.extend([tag(6 | (desc.ViewDimension.0 as u64) << 8), windows::core::Interface::as_raw(tr) as u64, u64::from(desc.Format.0 as u32) | u64::from(desc.Shader4ComponentMapping) << 32, u[0], u[1], u[2]]);
            }
            None => w.extend([tag(7 | u64::from(dim) << 8), cx.grey]),
        },
    }
    true
}

/// A table's inputs' identity hash: slots, kinds' places and bytes by pointer, textures by handle.
fn table_hash(t: &[(u32, OSlot)]) -> u64 {
    let mut h = Fx::default();
    for (k, s) in t {
        std::hash::Hasher::write_u64(&mut h, u64::from(*k));
        std::hash::Hasher::write_u64(&mut h, match s {
            OSlot::Buffer(_, b) => b[..].as_ptr() as u64,
            OSlot::Texture(x) => *x,
        });
    }
    std::hash::Hasher::finish(&h)
}

/// Two tables of the same inputs: equal slots, kinds and handles, the same bytes by pointer.
fn same_table(a: &[(u32, OSlot)], b: &[(u32, OSlot)]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.0 == y.0
                && match (&x.1, &y.1) {
                    (OSlot::Buffer(ka, ba), OSlot::Buffer(kb, bb)) => ka == kb && Arc::ptr_eq(ba, bb),
                    (OSlot::Texture(p), OSlot::Texture(q)) => p == q,
                    _ => false,
                }
        })
}

/// The table slot heap slot `s` of a binding is: (root, slot in the
/// table), from the tables' first slots (`constants`, times 32); None
/// below the first table (a draw word's slot).
fn table_slot(lay: &Lay, constants: &[u32], s: u32) -> Option<(u16, u32)> {
    lay.roots.iter().filter_map(|&r| constants.get(usize::from(r)).map(|c| (r, c / 32))).filter(|x| x.1 <= s).max_by_key(|x| x.1).map(|(r, first)| (u16::from(r), s - first))
}

/// The D3D12 view dimension of what the bridge writes for a texture at a
/// slot of stand-in dimension `dim`: our table's view when it fits
/// ([`view_fits`]), else the stand-in's ([`one_texel_view`]).
fn written_dim(view: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>, dim: u8) -> i32 {
    view.filter(|v| view_fits(v, dim)).map_or(one_texel_view(dim).ViewDimension.0, |v| v.ViewDimension.0)
}

/// BND1: a draw-word block's confinement masks (`bridge::Binding::confine`
/// words 1-6) from its entries: slot j in the mask of the dimension of the
/// view written there ([`written_dim`]). A slot without a texture (a typed
/// null), past the 32 a mask holds or of a dimension without a mask is in
/// none: an index naming it reads the null of the reading instruction.
fn block_masks(ents: &[Ent], texs: &Texs) -> [u32; 6] {
    let mut c = [0u32; 8];
    for e in ents {
        if let Desc::Tex(h, dim) = e.desc {
            bridge::confine_mark(&mut c, e.j, written_dim(texs.get(&h).and_then(|t| t.as_ref()).map(|t| &t.1), dim));
        }
    }
    [c[1], c[2], c[3], c[4], c[5], c[6]]
}

/// The table roots the two programs read, in order.
fn table_roots(ms: &HostShader, ps: &HostShader) -> Vec<u8> {
    let mut t: Vec<u8> = [ms, ps].iter().flat_map(|s| s.user_data.iter().filter(|u| u.root == xnw_shader::RootType::Table).map(|u| u.slot)).collect();
    t.sort_unstable();
    t.dedup();
    t
}

/// A row's binding (made at heap slot `base`) cut into blocks, appended to
/// `out` with their keys: the draw words' textures, then one per table
/// root of `lay` with its raw twins after its slots. A table block reaches
/// at least past every slot the programs name in it (a named slot without
/// a view holds its typed null in the block, never a neighbour's view).
/// Only the slots that are not a typed null with a zero stride entry are
/// listed; a table root with `reuse[k]` takes that earlier block's entries
/// and key without being cut.
fn cut(b: &bridge::Binding, lay: &Lay, views: &[(u32, Kind, At, u32)], base: u32, cx: &KeyCx, reuse: &[Option<usize>], out: &mut Cuts) {
    use xnw_shader::emit::StrideEntry;
    let strides = &b.views.strides;
    let pre = strides.len() as u32;
    let b0 = out.blocks.len();
    out.twins.clear();
    out.regions.clear();
    let twins = strides.iter().filter(|s| s[1] & StrideEntry::RAW_SLOTS != 0).count() as u32;
    let t0 = pre - twins;
    let first = |k: usize| (b.views.constants[usize::from(lay.roots[k])] / 32).saturating_sub(base);
    let wspan = if lay.roots.is_empty() { t0 } else { first(0) };
    if wspan > 0 {
        out.regions.push((0, wspan, out.blocks.len()));
        out.blocks.push(Block { root: -1, len: wspan, ents: 0..0, key: None, first: 0, from: None, masks: [0; 6] });
    }
    let tb = out.blocks.len();
    for k in 0..lay.roots.len() {
        let start = first(k);
        let span = if k + 1 < lay.roots.len() { first(k + 1) } else { t0 }.saturating_sub(start);
        out.regions.push((start, span, out.blocks.len()));
        out.blocks.push(Block { root: i32::from(lay.roots[k]), len: span.max(lay.named[k]).max(1), ents: 0..0, key: None, first: 0, from: None, masks: [0; 6] });
    }
    let find = |regions: &[(u32, u32, usize)], j: u32| regions.iter().find(|g| j >= g.0 && j < g.0 + g.1).map(|g| (g.2, j - g.0));
    // A block that equals an earlier row's (same pipeline, same table inputs) is that block.
    let reused = |bk: usize| if bk >= tb { reuse.get(bk - tb).copied().flatten() } else { None };
    // Twins go after their owner's slots, in twin order: each block's length
    // first (a block taken uncut has its earlier block's length).
    for &(start, span, bk) in &out.regions {
        if let Some(from) = reused(bk) {
            out.blocks[bk].len = out.blocks[from].len;
            continue;
        }
        for j in start..start + span {
            let raw = strides[j as usize][1] & StrideEntry::RAW_SLOTS;
            if raw != 0 {
                let blk = &mut out.blocks[bk];
                out.twins.push((raw - 1 - base, bk, blk.len));
                blk.len += 1;
            }
        }
    }
    out.twins.sort_unstable_by_key(|x| x.0);
    out.at.clear();
    let mut total = 0u32;
    for bk in b0..out.blocks.len() {
        out.at.push(total);
        if reused(bk).is_none() {
            total += out.blocks[bk].len;
        }
    }
    out.dense.clear();
    out.dense.resize(total as usize, ([0, 0], Desc::Null));
    let at = |bk: usize, j: u32| (out.at[bk - b0] + j) as usize;
    // The regions' stride entries, the owners' twins, named images, views, textures, words: a later one wins.
    for &(start, span, bk) in &out.regions {
        if reused(bk).is_some() {
            continue;
        }
        let a = at(bk, 0);
        for j in 0..span as usize {
            out.dense[a + j].0 = [strides[start as usize + j][0], 0];
        }
    }
    for &(start, span, bk) in &out.regions {
        if reused(bk).is_some() {
            continue;
        }
        for local in 0..span {
            let s = strides[(start + local) as usize];
            let raw = s[1] & StrideEntry::RAW_SLOTS;
            let Some(i) = (raw != 0).then(|| out.twins.binary_search_by_key(&(raw - 1 - base), |x| x.0).ok()).flatten() else { continue };
            let tw = out.twins[i].2;
            out.dense[at(bk, local)].0 = [s[0], tw + 1];
            out.dense[at(bk, tw)].0 = strides.get((raw - 1 - base) as usize).copied().unwrap_or([4, 0]);
        }
    }
    for (k, images) in lay.images.iter().enumerate() {
        if reused(tb + k).is_none() {
            for &(j, dim) in images {
                out.dense[at(tb + k, j)].1 = Desc::NullImage(dim);
            }
        }
    }
    let place = |out: &Cuts, s: u32| -> Option<usize> {
        let j = s.checked_sub(base)?;
        let (bk, l) = if j >= t0 && j < pre { out.twins.binary_search_by_key(&j, |x| x.0).ok().map(|i| (out.twins[i].1, out.twins[i].2))? } else { find(&out.regions, j)? };
        reused(bk).is_none().then(|| at(bk, l))
    };
    for &(s, kind, where_, n) in views {
        if let Some(i) = place(out, s) {
            out.dense[i].1 = Desc::Buf(kind, where_, n);
        }
    }
    for &(s, h) in &b.handles {
        if let Some(i) = place(out, s) {
            let dim = lay.stand_in_dim(cx.dims.get(&h).copied());
            // BND1: a table slot takes the texture only as a view of the dimension the
            // programs read the slot in; else the null image of that dimension.
            out.dense[i].1 = match table_slot(lay, &b.views.constants, s).map(|(r, k)| lay.image_dim(r, k)) {
                Some(want) if !bridge::dim_reads(written_dim(cx.texs.get(&h).and_then(|t| t.as_ref()).map(|t| &t.1), dim), want) => Desc::NullImage(want),
                _ => Desc::Tex(h, dim),
            };
        }
    }
    if let Some(k) = lay.roots.iter().position(|&r| r == 0)
        && reused(tb + k).is_none()
        && WORDS_SLOT < out.blocks[tb + k].len
        && first(k) + WORDS_SLOT < t0
    {
        out.dense[at(tb + k, WORDS_SLOT)].1 = Desc::Words;
    }
    // Each block's entries and key.
    for bk in b0..out.blocks.len() {
        if let Some(from) = reused(bk) {
            let (ents, key) = (out.blocks[from].ents.clone(), out.blocks[from].key.clone());
            let blk = &mut out.blocks[bk];
            (blk.ents, blk.key, blk.from) = (ents, key, Some(from));
            continue;
        }
        let e0 = out.ents.len();
        let (a, len) = (at(bk, 0), out.blocks[bk].len as usize);
        for (j, &(stride, desc)) in out.dense[a..a + len].iter().enumerate() {
            if desc != Desc::Null || stride != [0, 0] {
                out.ents.push(Ent { j: j as u32, stride, desc });
            }
        }
        let k0 = out.keys.len();
        out.keys.push(len as u64);
        let mut ok = true;
        for e in &out.ents[e0..] {
            if !key_words(e, cx, &mut out.keys) {
                ok = false;
                break;
            }
        }
        let key = if ok {
            let mut h = Fx::default();
            for &w in &out.keys[k0..] {
                std::hash::Hasher::write_u64(&mut h, w);
            }
            Some((std::hash::Hasher::finish(&h), k0..out.keys.len()))
        } else {
            out.keys.truncate(k0);
            None
        };
        let blk = &mut out.blocks[bk];
        blk.ents = e0..out.ents.len();
        blk.key = key;
    }
    out.rows.push(b0..out.blocks.len());
}

/// One row as [`Gpu::record`] binds it.
struct Row<'a> {
    i: usize,
    built: Arc<Built>,
    b: bridge::Binding<'a>,
    views: Vec<(u32, Kind, At, u32)>,
    mroots: Vec<(u32, bool, At)>,
    proots: Vec<(u32, bool, At)>,
    rec: u64,
    idx: u64,
    /// The view heap slot of word 0's texture.
    words_base: u32,
    /// The slot the binding was made at.
    base: u32,
    /// Its held binding ([`hold_mode`]).
    hk: Option<HoldKey>,
    /// [`replay_mode`]: replayed from last frame (its blocks, table
    /// offsets and bounds are last frame's); its inputs' versions; with
    /// `check`, the replay that would have been made.
    replay: bool,
    ins: Vec<In>,
    cand: Option<Box<Cand<'a>>>,
}

/// A row as its [`Replay`] binds it this frame.
struct Cand<'a> {
    b: bridge::Binding<'a>,
    views: Vec<(u32, Kind, At, u32)>,
    mroots: Vec<(u32, bool, At)>,
    proots: Vec<(u32, bool, At)>,
    base: u32,
    words_base: u32,
}

/// `XNW_BRIDGE_HOLD` (default off): `on`: a row of a draw known
/// across frames ([`Owned::ident`]) keeps its binding ([`Held`]) while its
/// inputs keep their shape; each frame the views are pointed at that
/// frame's bytes and moved to the row's base, without [`bridge::bind_at`].
/// `check`: as `on`, and every held binding is compared with a fresh
/// `bind_at` (counted on the bridgeps line).
/// Live (SW1 `hold`, taken once a frame; 0 lets every held binding go).
pub fn hold_mode() -> u8 {
    xnw_core::live::value(xnw_core::live::HOLD, || match std::env::var("XNW_BRIDGE_HOLD").as_deref() {
        Ok("on" | "1") => 1,
        Ok("check") => 2,
        _ => 0,
    }) as u8
}

/// Where a byte slice or handle of a held binding comes from in the row's inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Src {
    /// Entry `i` of table root `r` ([`ORoot::Table`]).
    Table(u8, u32),
    /// Draw word texture `i` ([`Owned::heap`]).
    Heap(u32),
    /// Slot 86's place in the frame table ([`Frame::draw`]).
    Words,
}

/// A row's binding kept by its draw identity: [`bridge::bind_at`]'s
/// result at base 0 without its bytes (they are the frame's), where each
/// view's bytes and each texture's handle come from, and the inputs' shape
/// it holds for ([`shape`]).
struct Held {
    shape: Vec<u64>,
    /// Bytes empty; `vsrc` names each slot's.
    views: super::preskin::Binding<'static>,
    vsrc: Vec<Src>,
    handles: Vec<(u32, Src)>,
    /// The table roots (their constants move with the base).
    tables: Vec<u8>,
    /// Root views' bytes empty (each one's root names them); constants
    /// at base 0.
    mesh: bridge::Stage<'static>,
    pixel: bridge::Stage<'static>,
    /// Per stage: (user-data index, root, dword) of root constants, and
    /// (user-data index, table root) of tables.
    consts: [Vec<(u32, u8, u32)>; 2],
    tabs: [Vec<(u32, u8)>; 2],
    last: u64,
    /// Per slot: the bytes placed there last frame at the draw's own
    /// small slot ([`Small::place_steady`]), held so that the same
    /// pointer is the same bytes, and that slot.
    prev: Vec<Option<(Arc<Vec<u8>>, u64)>>,
    /// The sampler list last bound (held: the same address is the same
    /// list), its slots, its heap (held alive here).
    sheap: Option<(Arc<Vec<(u32, SamplerDesc)>>, u32, ID3D12DescriptorHeap)>,
    /// [`replay_mode`]: what the row bound last frame.
    replay: Option<Box<Replay>>,
}

/// A held binding's identity: the draw's identity, its pipeline, its occurrence.
type HoldKey = (u64, Key, u32);

/// `XNW_BRIDGE_REPLAY` (default off; with [`hold_mode`], [`tables_on`]
/// and [`steady_on`]): `on`: a held row whose inputs' versions all equal
/// last frame's ([`Replay`]) is not bound, placed or cut again: its views,
/// root arguments and blocks are last frame's (UE 5.9's clean-slot rule,
/// 1); only its changed small inputs are
/// written at their slots. `check`: every row is bound afresh, and each
/// replay that would have been made is compared with it (counted on the
/// bridgeps line: `replayed N (unlike U)`).
pub fn replay_mode() -> u8 {
    static ON: OnceLock<u8> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("XNW_BRIDGE_REPLAY").as_deref() {
        Ok("on" | "1") => 1,
        Ok("check") => 2,
        _ => 0,
    })
}

/// An input's version, as the place it got: in place (entry `ai` of
/// the row's title addresses, at `title`, while the in-place memo was at
/// generation `generation`: [`Gpu::zc_changed`] moves it on whenever it drops a
/// range), at an offset of the small inputs' buffer (the draw's own slot,
/// or an earlier row's of the same bytes; the bytes themselves are
/// compared and written there each frame), or kept on the device.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ver {
    Shared { ai: u32, title: u64, generation: u64 },
    Small(u64),
    Kept(u64),
    /// At its own place in the kept buffer by title address ([`Kept::place_steady`]).
    Steady { ai: u32, title: u64, o: u64 },
}

/// One placed input of a row: where its bytes come from (`slot`: index
/// into the held binding's `vsrc`; else root view `root`), the place's
/// arguments, its version (None: a place no version names, e.g. the
/// frame's upload memory: the row is not replayed).
#[derive(Clone, Copy, Debug)]
struct In {
    slot: Option<u32>,
    root: u8,
    size: u64,
    unit: u64,
    input: u32,
    ver: Option<Ver>,
}

/// What a held row bound last frame ([`replay_mode`]): its inputs'
/// versions (views, then mesh root views, then pixel root views), the
/// views and root views it placed, its binding with the table offsets and
/// bounds of its blocks (bytes empty), its base and words' base, its
/// textures' descriptors as block keys spell them, the frame and the
/// block cache's epoch (its blocks are where they were until a flush).
struct Replay {
    frame: u64,
    epoch: u64,
    ins: Vec<In>,
    views: Vec<(u32, Kind, At, u32)>,
    mroots: Vec<(u32, bool, At)>,
    proots: Vec<(u32, bool, At)>,
    b: bridge::Binding<'static>,
    base: u32,
    words_base: u32,
    texs: Vec<u64>,
}

/// A binding without its bytes (a [`Replay`]'s).
fn stripped(b: &bridge::Binding) -> bridge::Binding<'static> {
    let mut views = super::preskin::Binding { constants: b.views.constants.clone(), slots: Vec::with_capacity(b.views.slots.len()), heap_slots: b.views.heap_slots, strides: b.views.strides.clone(), bounds: b.views.bounds, texels: b.views.texels.clone() };
    views.slots.extend(b.views.slots.iter().map(|&(s, v)| (s, View { kind: v.kind, bytes: &[] })));
    let stage = |s: &bridge::Stage| bridge::Stage { constants: s.constants.clone(), views: s.views.iter().map(|v| (v.0, v.1, &[][..])).collect(), lengths: s.lengths.clone() };
    bridge::Binding { views, textures: Vec::new(), handles: b.handles.clone(), samplers: Vec::new(), sampler_slots: b.sampler_slots, mesh: stage(&b.mesh), pixel: stage(&b.pixel), groups: b.groups, confine: b.confine }
}

/// The key words of a row's textures (handle by handle, as [`key_words`]
/// spells a block's texture entry; the dimension from the row's own views).
fn tex_words(b: &bridge::Binding, o: &Owned, lay: &Lay, cx: &KeyCx, w: &mut Vec<u64>) {
    w.clear();
    for &(_, h) in &b.handles {
        let dim = lay.stand_in_dim(o.dims.iter().find(|d| d.0 == h).map(|d| d.1));
        if !key_words(&Ent { j: 0, stride: [0, 0], desc: Desc::Tex(h, dim) }, cx, w) {
            w.push(u64::MAX);
        }
    }
}

/// Two places alike (in-place ones by their buffer and offset).
fn same_at(a: At, b: At, zc: &[xnw_gpu::shared::SharedRange]) -> bool {
    match (a, b) {
        (At::Shared(i), At::Shared(j)) => zc.get(i as usize).zip(zc.get(j as usize)).is_some_and(|(x, y)| x.buffer == y.buffer && x.offset == y.offset && x.len == y.len),
        _ => a == b,
    }
}

/// A buffer kind as one key word.
fn kind_word(kind: Kind) -> u64 {
    match kind {
        Kind::Raw => 0,
        Kind::Structured(s) => 1 | u64::from(s) << 8,
        Kind::Typed { format, bytes } => 2 | u64::from(format as u32) << 8 | u64::from(bytes) << 40,
        Kind::Cbv => 3,
        Kind::RawUav { len } => 4 | u64::from(len) << 8,
    }
}

/// The shape of a row's inputs: everything [`bridge::bind_at`]'s result
/// depends on but the bytes' addresses, handles and constants' values
/// (roots' kinds, table slots with each one's kind and byte length, root
/// views' and constants' lengths, the draw words' texture positions).
fn shape(o: &Owned, w: &mut Vec<u64>) {
    w.clear();
    w.push(o.roots.len() as u64 | (o.heap.len() as u64) << 32);
    for r in &o.roots {
        match r {
            ORoot::None => w.push(0),
            ORoot::Constants(c) => w.push(1 | (c.len() as u64) << 8),
            ORoot::View(b) => w.push(2 | (b.len() as u64) << 8),
            ORoot::Table(t) => {
                w.push(3 | (t.len() as u64) << 8);
                for (k, s) in t {
                    match s {
                        OSlot::Texture(_) => w.push(u64::from(*k) | 1 << 32),
                        OSlot::Buffer(kind, b) => w.extend([u64::from(*k) | 2 << 32, kind_word(*kind), b.len() as u64]),
                    }
                }
            }
        }
    }
    w.extend(o.heap.iter().map(|h| u64::from(h.0)));
}

/// The bytes of input `s` of row `o`.
fn src_bytes(o: &Owned, s: Src) -> Option<&[u8]> {
    match s {
        Src::Table(r, i) => match o.roots.get(usize::from(r))? {
            ORoot::Table(t) => match &t.get(i as usize)?.1 {
                OSlot::Buffer(_, b) => Some(&b[..]),
                OSlot::Texture(_) => None,
            },
            _ => None,
        },
        Src::Words => Some(&WORDS_PLACE[..]),
        Src::Heap(_) => None,
    }
}

/// The bytes of buffer input `s` of row `o` as held ([`ORoot::Table`]'s).
fn src_arc(o: &Owned, s: Src) -> Option<&Arc<Vec<u8>>> {
    match s {
        Src::Table(r, i) => match o.roots.get(usize::from(r))? {
            ORoot::Table(t) => match &t.get(i as usize)?.1 {
                OSlot::Buffer(_, b) => Some(b),
                OSlot::Texture(_) => None,
            },
            _ => None,
        },
        _ => None,
    }
}

/// The handle of texture input `s` of row `o`.
fn src_handle(o: &Owned, s: Src) -> Option<u64> {
    match s {
        Src::Table(r, i) => match o.roots.get(usize::from(r))? {
            ORoot::Table(t) => match t.get(i as usize)?.1 {
                OSlot::Texture(h) => Some(h),
                OSlot::Buffer(..) => None,
            },
            _ => None,
        },
        Src::Heap(i) => o.heap.get(i as usize).map(|h| h.1),
        Src::Words => None,
    }
}

/// PERF1: the identity of a small input several rows of a frame hand over
/// as the same bytes (a frame table's views): ident, pipeline key.
const SHARED_IDENT: u64 = 0x5348_4152_0000_0000;
const SHARED_KEY: Key = (u128::MAX, 0);

/// PERF1: where a shared small input sits in the title's own numbering,
/// whatever row hands it over first: a table view by its root and the
/// title's slot in that table (a raw twin has its owner's; the unit tells
/// them apart), our words' place, a root view by its root.
fn shared_code(o: &Owned, s: Src) -> Option<u32> {
    match s {
        Src::Table(r, i) => match o.roots.get(usize::from(r))? {
            ORoot::Table(t) => Some(0x8000_0000 | u32::from(r) << 20 | t.get(i as usize)?.0 & 0xf_ffff),
            _ => None,
        },
        Src::Words => Some(0x8000_0000 | 0xf_ffff),
        Src::Heap(_) => None,
    }
}

/// A root view's [`shared_code`].
fn shared_root(r: u8) -> u32 {
    0x4000_0000 | u32::from(r)
}

/// PERF1: the identity a small input is placed under in [`Small`]
/// (`place_steady`, `touch`). Bytes that several rows of the frame hand
/// over (`multi`) and whose place in the title's numbering is known
/// (`shared`) go by that place, their unit and their occurrence among
/// the frame's distinct inputs of that place (`occ`: counted here), so
/// the slot does not depend on which row the list has first (it changed
/// every other frame of pk6 f1205-1210: every frame-table block was made
/// again and no row could be replayed). Anything else by the draw:
/// (`rid`, `input`); None without a draw identity.
fn small_id(shared: Option<u32>, multi: bool, unit: u64, rid: Option<(u64, Key, u32)>, input: u32, occ: &mut ByContent<(u32, u64), u32>) -> Option<IdKey> {
    rid?;
    match shared.filter(|_| multi) {
        Some(code) => {
            let n = occ.entry((code, unit)).or_insert(0);
            *n += 1;
            Some((SHARED_IDENT | unit & 0xffff_ffff, SHARED_KEY, *n - 1, code))
        }
        None => rid.map(|(ident, key, n)| (ident, key, n, input)),
    }
}

/// PERF1: the place of bytes a steady slot holds already this frame
/// (`at`: bytes' address and length -> offset, capacity) for a view of
/// them of another unit (`pk`: address, length, padded size, unit): a
/// raw twin reads its owner's slot when that offset is one its own view
/// can start at and the slot holds its size (each was placed, compared
/// and written a second time: 3 of a pk6 row's 9 own inputs).
fn twin_at(at: &ByContent<(usize, usize), (u64, u64)>, pk: (usize, usize, u64, u64)) -> Option<u64> {
    at.get(&(pk.0, pk.1)).filter(|x| x.0.is_multiple_of(Small::align(pk.3)) && Small::cap(pk.2) <= x.1).map(|x| x.0)
}

impl Held {
    /// The binding `b` (made by [`bridge::bind_at`] at base 0 for row `o`)
    /// without its bytes; None when a slot's bytes or a handle cannot be
    /// named by an input of `o` (the row is then bound afresh each frame).
    fn new(b: &bridge::Binding, o: &Owned, ms: &HostShader, ps: &HostShader, shape: Vec<u64>, frame: u64) -> Option<Held> {
        use xnw_shader::emit::StrideEntry;
        let tables = table_roots(ms, ps);
        // The table slot `s` falls in: (root, its slot in the table).
        let in_table = |s: u32| tables.iter().filter(|&&r| b.views.constants.get(usize::from(r)).is_some_and(|c| c / 32 <= s)).max_by_key(|&&r| b.views.constants[usize::from(r)]).map(|&r| (r, s - b.views.constants[usize::from(r)] / 32));
        let table_src = |r: u8, k: u32| -> Option<Src> {
            match o.roots.get(usize::from(r))? {
                ORoot::Table(t) => match t.iter().position(|x| x.0 == k) {
                    Some(i) => Some(Src::Table(r, i as u32)),
                    None if r == 0 && k == WORDS_SLOT => Some(Src::Words),
                    None => None,
                },
                _ => None,
            }
        };
        let same = |s: Src, v: &[u8]| src_bytes(o, s).is_some_and(|x| std::ptr::eq(x.as_ptr(), v.as_ptr()) && x.len() == v.len());
        let mut vsrc: Vec<Option<Src>> = b.views.slots.iter().map(|&(s, v)| in_table(s).and_then(|(r, k)| table_src(r, k)).filter(|&x| same(x, v.bytes))).collect();
        // Raw twins take their owner's bytes.
        for i in 0..vsrc.len() {
            if vsrc[i].is_some() {
                continue;
            }
            let s = b.views.slots[i].0;
            let owner = b.views.slots.iter().zip(&vsrc).find(|(o, src)| src.is_some() && b.views.strides.get(o.0 as usize).is_some_and(|w| w[1] & StrideEntry::RAW_SLOTS != 0 && (w[1] & StrideEntry::RAW_SLOTS) - 1 == s));
            vsrc[i] = owner.and_then(|x| *x.1).filter(|&x| same(x, b.views.slots[i].1.bytes));
        }
        let vsrc: Vec<Src> = vsrc.into_iter().collect::<Option<_>>()?;
        let handles: Vec<(u32, Src)> = b
            .handles
            .iter()
            .map(|&(s, h)| {
                let src = match in_table(s) {
                    Some((r, k)) => table_src(r, k)?,
                    None => Src::Heap(o.heap.iter().position(|x| x.0 == s)? as u32),
                };
                (src_handle(o, src) == Some(h)).then_some((s, src))
            })
            .collect::<Option<_>>()?;
        let mut views = super::preskin::Binding { constants: b.views.constants.clone(), slots: Vec::with_capacity(b.views.slots.len()), heap_slots: b.views.heap_slots, strides: b.views.strides.clone(), bounds: b.views.bounds, texels: b.views.texels.clone() };
        views.slots.extend(b.views.slots.iter().map(|&(s, v)| (s, View { kind: v.kind, bytes: &[] })));
        let mut consts = [Vec::new(), Vec::new()];
        let mut tabs = [Vec::new(), Vec::new()];
        let mut stages = [bridge::Stage::default(), bridge::Stage::default()];
        for (k, (s, st)) in [(ms, &b.mesh), (ps, &b.pixel)].into_iter().enumerate() {
            for u in &s.user_data {
                match u.root {
                    xnw_shader::RootType::Table if u.dword == 0 => tabs[k].push((u.index, u.slot)),
                    xnw_shader::RootType::Constants => consts[k].push((u.index, u.slot, u.dword)),
                    _ => {}
                }
            }
            // Root views by their root: the bytes must be that root's.
            for v in &st.views {
                match o.roots.get(usize::from(v.0)) {
                    Some(ORoot::View(x)) if std::ptr::eq(x.as_ptr(), v.2.as_ptr()) && x.len() == v.2.len() => {}
                    _ => return None,
                }
            }
            stages[k] = bridge::Stage { constants: st.constants.clone(), views: st.views.iter().map(|v| (v.0, v.1, &[][..])).collect(), lengths: st.lengths.clone() };
        }
        let [mesh, pixel] = stages;
        let prev = vec![None; vsrc.len()];
        Some(Held { shape, views, vsrc, handles, tables, mesh, pixel, consts, tabs, last: frame, prev, sheap: None, replay: None })
    }

    /// The binding of row `o` at heap slot `base` (as [`bridge::bind_at`]
    /// makes it at `base`, samplers left to the row's own list); None when
    /// an input is not where the held binding names it.
    fn bind<'f>(&self, o: &'f Owned, base: u32, sampler_slots: u32) -> Option<bridge::Binding<'f>> {
        use xnw_shader::emit::StrideEntry;
        let mut views: super::preskin::Binding<'f> = self.views.clone();
        for (v, &s) in views.slots.iter_mut().zip(&self.vsrc) {
            v.0 += base;
            v.1.bytes = src_bytes(o, s)?;
        }
        if base > 0 {
            for &r in &self.tables {
                views.constants[usize::from(r)] += 32 * base;
            }
            for w in &mut views.strides {
                if w[1] & StrideEntry::RAW_SLOTS != 0 {
                    w[1] += base;
                }
            }
            for g in &mut views.bounds[..6] {
                *g += base;
            }
        }
        views.bounds[6] = sampler_slots;
        let handles = self.handles.iter().map(|&(s, src)| Some((s + base, src_handle(o, src)?))).collect::<Option<Vec<_>>>()?;
        let mut stages = [self.mesh.clone(), self.pixel.clone()];
        for (k, st) in stages.iter_mut().enumerate() {
            for &(i, r) in &self.tabs[k] {
                st.constants[i as usize] = views.constants[usize::from(r)];
            }
            for &(i, r, d) in &self.consts[k] {
                let ORoot::Constants(c) = o.roots.get(usize::from(r))? else { return None };
                st.constants[i as usize] = *c.get(d as usize)?;
            }
            for v in &mut st.views {
                let ORoot::View(x) = o.roots.get(usize::from(v.0))? else { return None };
                v.2 = &x[..];
            }
        }
        let [mesh, pixel] = stages;
        let confine = bridge::confine_at(&views, base);
        Some(bridge::Binding { views, textures: Vec::new(), handles, samplers: Vec::new(), sampler_slots, mesh, pixel, groups: o.groups, confine })
    }
}

/// The sampler slots [`bridge::bind_at`] makes for a row with `samplers`.
fn sampler_slots(samplers: &[(u32, SamplerDesc)]) -> u32 {
    samplers.iter().map(|s| s.0 + 1).max().unwrap_or(0).max(SAMPLER_SLOTS).clamp(1, bridge::SAMPLER_HEAP_MAX)
}

/// Two bindings alike but for their samplers' list (`XNW_BRIDGE_HOLD=check`).
fn same_binding(a: &bridge::Binding, b: &bridge::Binding) -> bool {
    let ptrs = |x: &bridge::Binding| x.views.slots.iter().map(|v| (v.0, v.1.kind, v.1.bytes.as_ptr() as usize, v.1.bytes.len())).collect::<Vec<_>>();
    let roots = |s: &bridge::Stage| s.views.iter().map(|v| (v.0, v.1, v.2.as_ptr() as usize, v.2.len())).collect::<Vec<_>>();
    a.views.constants == b.views.constants
        && ptrs(a) == ptrs(b)
        && a.views.heap_slots == b.views.heap_slots
        && a.views.strides == b.views.strides
        && a.views.bounds == b.views.bounds
        && a.handles == b.handles
        && a.textures.is_empty()
        && b.textures.is_empty()
        && a.sampler_slots == b.sampler_slots
        && a.groups == b.groups
        && [(&a.mesh, &b.mesh), (&a.pixel, &b.pixel)].iter().all(|(x, y)| x.constants == y.constants && x.lengths == y.lengths && roots(x) == roots(y))
}

impl Gpu {
    /// The in-place ranges kept across frames ([`hold_mode`]) whose
    /// window was released or whose title pages were remapped since the
    /// last frame go (all of them when the changes overflowed). The pass
    /// calls it at a frame's start with what it took from
    /// [`xnw_gpu::shared::world_changes`].
    pub fn zc_changed(&mut self, c: &xnw_gpu::shared::WorldChanges) {
        if self.zc_memo.is_empty() {
            return;
        }
        // A bound on what is kept: the frame finds the ones it uses again.
        let before = self.zc_memo.len();
        if self.zc_memo.len() > 1 << 16 {
            self.zc_memo.clear();
            self.zc_gen += 1;
        }
        if c.overflow || !c.gone.is_empty() {
            // SAFETY: a plain query on a live buffer.
            self.zc_memo.retain(|_, r| !c.overflow && {
                let g = unsafe { r.buffer.GetGPUVirtualAddress() } + r.offset;
                !c.gone.iter().any(|&(at, n)| g < at + n && g + r.len as u64 > at)
            });
        }
        for &(a, n) in &c.remapped {
            let hit: Vec<(u64, u64)> = self.zc_memo.range((a.saturating_sub(VIEW_MAX + 4096), 0)..(a + n, 0)).filter(|((t, len), _)| t + len > a).map(|(k, _)| *k).collect();
            for k in hit {
                self.zc_memo.remove(&k);
            }
        }
        if self.zc_memo.len() != before {
            self.zc_gen += 1;
        }
    }

    /// Records every row of `f` whose pipelines are made on the open list
    /// `l` (our pass's, after its own draws: colour and depth in their
    /// target states). Every mesh program first (their records and index
    /// triples in one buffer each), one barrier, then every pixel program.
    /// Objects replaced this frame and the textures go to `keep`, for the
    /// caller to hold until the list's fence. The caller binds its own
    /// heaps again after it if it draws more.
    pub fn record(&mut self, dev: &ID3D12Device, l: &ID3D12GraphicsCommandList, f: &Frame, t: &Target, keep: &mut Vec<windows::core::IUnknown>) -> Result<Stats, String> {
        self.record_with(dev, &mut || Ok(l.clone()), f, t, keep)
    }

    /// PERF3 [`Gpu::record`] with the open list asked of `list` only once
    /// every row is prepared (bindings, places, uploads, views, pipelines):
    /// the part before it needs no list, so `bridgepar` runs it on a thread
    /// of its own while the pass records its draws, and hands the list over
    /// at the rows' place in the pass. The list's commands and their order
    /// are the serial record's. The time waited in `list` is not counted.
    pub fn record_with<'f>(&mut self, dev: &ID3D12Device, list: &mut dyn FnMut() -> Result<ID3D12GraphicsCommandList, String>, f: &'f Frame, t: &Target, keep: &mut Vec<windows::core::IUnknown>) -> Result<Stats, String> {
        let t0 = std::time::Instant::now();
        // PROF1 (`XNW_RECORD_PROF`): the bindings part by section.
        let mut lap = super::recprof::Lap::new();
        let mut st = Stats::default();
        let tables = tables_on();
        let e = |e: windows::core::Error| e.message().to_string();
        self.frame += 1;
        if let Some(size) = self.kept.restart {
            super::recprof::event(0);
            if let Some((old, _)) = self.kept_buf.take() {
                keep.push(windows::core::Interface::cast(&old).map_err(e)?);
            }
            self.kept.start(size);
            self.epoch += 1;
        }
        if self.kept.size == 0 {
            self.kept.start(KEPT_LEAST);
        }
        self.kept.next_frame();
        let small = small_on();
        if small {
            // A new buffer when there is none or the last one is three quarters used.
            if self.small_buf.is_none() || self.small.end > self.small.size / 4 * 3 {
                if let Some((old, _)) = self.small_buf.take() {
                    keep.push(windows::core::Interface::cast(&old).map_err(e)?);
                }
                let size = (2 * self.small.size).max(SMALL_LEAST).min(1 << 30);
                super::recprof::event(1);
                // SAFETY: a description that outlives the call; the buffer is mapped once and lives in `self`.
                unsafe {
                    let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
                    let mut r: Option<ID3D12Resource> = None;
                    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut r).map_err(e)?;
                    let r = r.ok_or("no small inputs' buffer")?;
                    let mut to = std::ptr::null_mut();
                    r.Map(0, None, Some(&mut to)).map_err(e)?;
                    self.small_buf = Some((r, to.cast::<u8>()));
                }
                self.small.start(size);
                self.epoch += 1;
            }
            self.small.next_frame();
        }
        let small_to = self.small_buf.as_ref().map_or(std::ptr::null_mut(), |b| b.1);
        lap.mark(super::recprof::B_SETUP);
        // The frame's keepable inputs by their bytes' address.
        let mut arcs: ByContent<usize, &Arc<Vec<u8>>> = ByContent::default();
        // PERF1: the small inputs by their bytes' address: the first row
        // handing them over, and whether another row does too ([`small_id`]).
        let share = steady_on() && hold_mode() > 0 && shared_on();
        let mut uses: ByContent<usize, (u32, bool)> = ByContent::default();
        let mut used = |p: usize, i: usize| {
            let e = uses.entry(p).or_insert((i as u32, false));
            e.1 |= e.0 != i as u32;
        };
        for (i, (_, o)) in f.rows.iter().enumerate() {
            for r in &o.roots {
                match r {
                    ORoot::Table(v) => {
                        for (_, s) in v {
                            if let OSlot::Buffer(_, b) = s {
                                if b.len() >= KEEP_MIN {
                                    arcs.insert(b[..].as_ptr() as usize, b);
                                } else if share {
                                    used(b[..].as_ptr() as usize, i);
                                }
                            }
                        }
                    }
                    ORoot::View(b) if b.len() >= KEEP_MIN => {
                        arcs.insert(b[..].as_ptr() as usize, b);
                    }
                    ORoot::View(b) if share => used(b[..].as_ptr() as usize, i),
                    _ => {}
                }
            }
        }
        if share {
            uses.insert(WORDS_PLACE.as_ptr() as usize, (0, true));
        }
        let multi = |p: usize| uses.get(&p).is_some_and(|u| u.1);
        let mut shared_occ: ByContent<(u32, u64), u32> = ByContent::default();
        // PERF1 [`twin_at`]: the steady slots written this frame by their bytes.
        let mut by_bytes: ByContent<(usize, usize), (u64, u64)> = ByContent::default();
        lap.mark(super::recprof::B_ARCS);
        // The bindings, each row at its base in the view heap, and where every input and output goes.
        let mut rows: Vec<Row> = Vec::new();
        let mut arena = Arena::default();
        let mut copies: Vec<(usize, usize, u64, u64)> = Vec::new();
        let (mut rec_end, mut idx_end) = (0u64, 0u64);
        let (mut next, mut most) = (0u32, 0u32);
        let mut strides: Vec<[u32; 2]> = Vec::new();
        // Words offset -> (its rows' word base, the words, the textures they name).
        let mut words_at: ByContent<u32, (u32, [u32; WORDS], &[(u32, u64)])> = ByContent::default();
        // With blocks: inputs shorter than [`KEEP_MIN`] (constant buffers and
        // the like) by content, so that rows handing over equal bytes share
        // one place in the frame's upload memory, one view, one block.
        let mut canon: ByContent<&'f [u8], &'f [u8]> = ByContent::default();
        // `XNW_ZEROCOPY=on` (ZC1): inputs bound in place in title memory.
        // Constant buffers stay copies: every lane reads them, and over the
        // bus in-place reads that depend on each other ran ~30x slower
        // (an idea from upstream; a copy path stays for hot data).
        let zc_on = xnw_gpu::shared::world_part(xnw_gpu::shared::WORLD_BRIDGE);
        let zc_cbv = zc_on && zc_cbv_on();
        let mut zc: Vec<xnw_gpu::shared::SharedRange> = Vec::new();
        let mut zc_bytes = 0u64;
        // Place's time in the in-place lookups, and for small inputs.
        let mut place_ms = [0f64; 3];
        let split = place_split_on();
        // [`steady_on`]: each row's identity and occurrence among the frame's rows of it.
        let steady = steady_on();
        let mut occurs: ByContent<(u64, Key), u32> = ByContent::default();
        // Their small inputs placed this frame by the bytes' address: rows handing over the same bytes share one place.
        let mut by_ptr: ByContent<(usize, usize, u64, u64), u64> = ByContent::default();
        // [`hold_mode`]: held bindings unused for [`ID_FRAMES`] frames go.
        let hold = hold_mode();
        let frame = self.frame;
        self.holds.retain(|_, h| hold > 0 && h.1.last + ID_FRAMES >= frame);
        let mut scratch: Vec<u64> = Vec::new();
        // [`replay_mode`]: held rows replayed from last frame; the
        // frame's textures by handle (looked up once, here or below).
        let replay = if hold > 0 && tables && steady { replay_mode() } else { 0 };
        let mut texs = Texs::default();
        let mut tw: Vec<u64> = Vec::new();
        let no_dims: HashMap<u64, u8> = HashMap::new();
        let zc_gen = self.zc_gen;
        for i in 0..f.rows.len() {
            let o = &f.rows[i].1;
            if t.skip.binary_search(&f.rows[i].0).is_ok() {
                continue;
            }
            let Some(built) = cache().get(o.key) else {
                st.no_pipeline += 1;
                continue;
            };
            let occ = (o.ident != 0 && (steady || hold > 0)).then(|| {
                let n = occurs.entry((o.ident, o.key)).or_insert(0);
                *n += 1;
                (o.ident, o.key, *n - 1)
            });
            let rid = occ.filter(|_| steady);
            let hk = occ.filter(|_| hold > 0);
            let refuse = |st: &mut Stats, why: String| {
                st.refused += 1;
                if st.why.is_empty() {
                    st.why = format!("row {}: {why}", f.rows[i].0);
                }
            };
            // A texture read as a buffer, or a buffer as an image: the RX 5700 XT
            // reads a wild address (victory-fork 3c81420). Not drawn.
            if o.lacks & (1 << 7 | 1 << 8) != 0 {
                refuse(&mut st, "a view read through a descriptor of the other kind".into());
                continue;
            }
            if u64::from(o.words_at) + 4 * WORDS as u64 > t.words.1 {
                refuse(&mut st, format!("its words at {:#x} lie past our words buffer ({} B)", o.words_at, t.words.1));
                continue;
            }
            // Rows of one words offset share its textures: the first row's base names them.
            let words_base = match words_at.get(&o.words_at) {
                Some(w) if w.1 == o.words && w.2 == &o.heap[..] => w.0,
                Some(_) => {
                    refuse(&mut st, format!("its words at {:#x} differ from an earlier row's", o.words_at));
                    continue;
                }
                None => next,
            };
            let p = &built.programs;
            lap.mark(super::recprof::B_CHECKS);
            let tb = std::time::Instant::now();
            let wrote = (self.small.id_written, self.small.made, copies.len());
            st.row_key[0] += u32::from(hk.is_some());
            if hk.is_some() {
                shape(o, &mut scratch);
            }
            // A held row whose inputs' versions are last frame's: its
            // views and root views where they were (small inputs' changed
            // bytes written at their slots), its binding as last frame's.
            let mut cand: Option<Box<Cand>> = None;
            if replay > 0
                && let Some(k) = hk
                && let Some((hb, h)) = self.holds.get_mut(&k)
            {
                let why = match &h.replay {
                    None => 3,
                    Some(rp) if rp.frame + 1 != self.frame || rp.epoch != self.epoch => 4,
                    Some(rp) if !Arc::ptr_eq(hb, &built) || h.shape != scratch || o.groups == 0 || sampler_slots(&o.samplers) != rp.b.sampler_slots || h.handles.len() != rp.b.handles.len() || !h.handles.iter().all(|&(_, src)| src_handle(o, src).is_some()) => 5,
                    _ => 0,
                };
                if why > 0 {
                    st.replay_why[why] += 1;
                }
            }
            if replay > 0
                && let Some(k) = hk
                && let Some((hb, h)) = self.holds.get_mut(&k)
                && Arc::ptr_eq(hb, &built)
                && h.shape == scratch
                && let Some(rp) = h.replay.as_mut()
                && rp.frame + 1 == self.frame
                && rp.epoch == self.epoch
                && o.groups > 0
                && sampler_slots(&o.samplers) == rp.b.sampler_slots
                && h.handles.len() == rp.b.handles.len()
                && h.handles.iter().all(|&(_, src)| src_handle(o, src).is_some())
            {
                // PERF1: the handles are this frame's (pk8a's rows name their
                // textures by another descriptor every frame: no row was
                // replayed); what they resolve to is compared below (`rp.texs`).
                for (to, &(_, src)) in rp.b.handles.iter_mut().zip(&h.handles) {
                    if let Some(x) = src_handle(o, src) {
                        to.1 = x;
                    }
                }
                for &(_, x) in &rp.b.handles {
                    texs.entry(x).or_insert_with(|| (t.texture)(x));
                }
                let cx = KeyCx { texs: &texs, grey: windows::core::Interface::as_raw(t.grey) as u64, dims: &no_dims, zc: &[] };
                tex_words(&rp.b, o, &built.lay, &cx, &mut tw);
                let mut ok = tw == rp.texs;
                st.replay_why[6] += u32::from(!ok);
                let mut ats: Vec<At> = Vec::with_capacity(rp.ins.len());
                for inp in &rp.ins {
                    if !ok {
                        break;
                    }
                    let bytes = match inp.slot {
                        Some(j) => h.vsrc.get(j as usize).and_then(|&s| src_bytes(o, s)),
                        None => match o.roots.get(usize::from(inp.root)) {
                            Some(ORoot::View(x)) => Some(&x[..]),
                            _ => None,
                        },
                    };
                    let at = match (bytes, inp.ver) {
                        (Some(bytes), Some(Ver::Shared { ai, title, generation })) if generation == zc_gen && o.addrs.get(ai as usize) == Some(&(bytes.as_ptr() as usize, title)) => self.zc_memo.get(&(title, inp.size)).map(|r| {
                            zc_bytes += inp.size;
                            zc.push(xnw_gpu::shared::SharedRange { buffer: r.buffer.clone(), offset: r.offset, len: r.len });
                            At::Shared(zc.len() as u32 - 1)
                        }),
                        (Some(bytes), Some(Ver::Small(off))) => {
                            let pk = (bytes.as_ptr() as usize, bytes.len(), inp.size, inp.unit);
                            let x = match by_ptr.get(&pk).copied().or_else(|| twin_at(&by_bytes, pk).filter(|_| share && rid.is_some())) {
                                Some(x) => {
                                    by_ptr.insert(pk, x);
                                    Some(x)
                                }
                                None => {
                                    let code = match inp.slot {
                                        Some(j) => h.vsrc.get(j as usize).and_then(|&s| shared_code(o, s)),
                                        None => Some(shared_root(inp.root)),
                                    };
                                    small_id(code, multi(pk.0), inp.unit, rid, inp.input, &mut shared_occ)
                                }
                                .and_then(|id| self.small.place_steady(id, bytes, inp.size, inp.unit)).map(|(x, w, fresh)| {
                                    // SAFETY: as in `place` below: the draw's own slot lies inside
                                    // the mapped buffer, and no list in flight reads it.
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(bytes.as_ptr().add(w.start), small_to.add(x as usize + w.start), w.len());
                                        if fresh {
                                            let n = bytes.len().min(inp.size as usize);
                                            std::ptr::write_bytes(small_to.add(x as usize + n), 0, Small::cap(inp.size) as usize - n);
                                        }
                                    }
                                    by_ptr.insert(pk, x);
                                    by_bytes.insert((pk.0, pk.1), (x, Small::cap(inp.size)));
                                    if fresh { u64::MAX } else { x }
                                }),
                            };
                            (x == Some(off)).then_some(At::Small(off))
                        }
                        (Some(bytes), Some(Ver::Steady { ai, title, o: off })) if o.addrs.get(ai as usize) == Some(&(bytes.as_ptr() as usize, title)) => match self.kept.place_steady(title, bytes, inp.size, inp.unit) {
                            Some((x, fresh)) => {
                                if fresh {
                                    copies.push((bytes.as_ptr() as usize, bytes.len().min(inp.size as usize), inp.size, x));
                                }
                                (x == off).then_some(At::Kept(off))
                            }
                            None => None,
                        },
                        (Some(bytes), Some(Ver::Kept(off))) => {
                            let (a, fresh) = self.kept.place(&mut arena, bytes, inp.size, inp.unit, arcs.get(&(bytes.as_ptr() as usize)).copied());
                            if let (At::Kept(x), true) = (a, fresh) {
                                copies.push((bytes.as_ptr() as usize, bytes.len().min(inp.size as usize), inp.size, x));
                            }
                            (a == At::Kept(off) && !fresh).then_some(a)
                        }
                        _ => None,
                    };
                    match at {
                        Some(a) => ats.push(a),
                        None => {
                            ok = false;
                            st.replay_why[7] += 1;
                        }
                    }
                }
                if ok {
                    let (nv, nm) = (rp.views.len(), rp.mroots.len());
                    let mut b: bridge::Binding<'f> = rp.b.clone();
                    b.groups = o.groups;
                    for (k, stage) in [&mut b.mesh, &mut b.pixel].into_iter().enumerate() {
                        for &(i, r, d) in &h.consts[k] {
                            match o.roots.get(usize::from(r)) {
                                Some(ORoot::Constants(c)) if (d as usize) < c.len() && (i as usize) < stage.constants.len() => stage.constants[i as usize] = c[d as usize],
                                _ => ok = false,
                            }
                        }
                    }
                    st.replay_why[8] += u32::from(!ok);
                    if ok {
                        let views = rp.views.iter().zip(&ats[..nv]).map(|(v, &a)| (v.0, v.1, a, v.3)).collect();
                        let mroots = rp.mroots.iter().zip(&ats[nv..nv + nm]).map(|(v, &a)| (v.0, v.1, a)).collect();
                        let proots = rp.proots.iter().zip(&ats[nv + nm..]).map(|(v, &a)| (v.0, v.1, a)).collect();
                        cand = Some(Box::new(Cand { b, views, mroots, proots, base: rp.base, words_base: rp.words_base }));
                        if replay == 1 {
                            rp.frame = self.frame;
                            h.last = self.frame;
                        }
                    }
                }
            }
            if replay == 1
                && let Some(c) = cand.take()
            {
                if next + c.b.views.heap_slots > HEAP_MOST {
                    refuse(&mut st, format!("the view heap is full ({next} slots)"));
                    continue;
                }
                st.replayed[0] += 1;
                st.row_key[1] += 1;
                st.row_key[2] += u32::from(wrote == (self.small.id_written, self.small.made, copies.len()));
                words_at.entry(o.words_at).or_insert((words_base, o.words, &o.heap[..]));
                next += c.b.views.heap_slots;
                most = most.max(c.b.views.heap_slots);
                let lay = &p.layout;
                let groups = u64::from(c.b.groups);
                let (rec, idx) = (rec_end, idx_end);
                rec_end += (groups * u64::from(lay.record_bytes())).next_multiple_of(256);
                idx_end += (groups * u64::from(lay.index_bytes())).next_multiple_of(256);
                st.bind_ms[1] += tb.elapsed().as_secs_f64() * 1e3;
                let Cand { b, views, mroots, proots, base, words_base } = *c;
                rows.push(Row { i, built: built.clone(), b, views, mroots, proots, rec, idx, words_base, base, hk, replay: true, ins: Vec::new(), cand: None });
                continue;
            }
            // A held binding whose inputs keep their shape, moved to `next`.
            let mut held = None;
            let mut by_held = false;
            if let Some(k) = hk {
                if let Some((hb, h)) = self.holds.get_mut(&k)
                    && Arc::ptr_eq(hb, &built)
                    && h.shape == scratch
                {
                    h.last = self.frame;
                    held = h.bind(o, next, sampler_slots(&o.samplers));
                }
            }
            let was_held = held.is_some();
            let b = match held {
                Some(b) if b.groups > 0 => {
                    st.held[0] += 1;
                    by_held = true;
                    if hold == 2 && !bridge::bind_at(&p.ms, &p.ps, &f.draw(i, SAMPLER_SLOTS), next).is_ok_and(|x| same_binding(&x, &b)) {
                        st.held[2] += 1;
                    }
                    b
                }
                Some(_) => {
                    refuse(&mut st, "no group count".into());
                    continue;
                }
                None => {
                    let d = f.draw(i, SAMPLER_SLOTS);
                    let b = match bridge::bind_at(&p.ms, &p.ps, &d, if hk.is_some() { 0 } else { next }) {
                        Ok(b) if b.groups > 0 => b,
                        r => {
                            refuse(&mut st, r.err().unwrap_or_else(|| "no group count".into()));
                            continue;
                        }
                    };
                    let made = hk.and_then(|k| Some((k, Held::new(&b, o, &p.ms, &p.ps, scratch.clone(), self.frame)?)));
                    match made {
                        Some((k, h)) => {
                            st.held[1] += 1;
                            by_held = true;
                            let b = h.bind(o, next, sampler_slots(&o.samplers));
                            self.holds.insert(k, (built.clone(), h));
                            match b {
                                Some(b) => b,
                                None => {
                                    refuse(&mut st, "a held binding names no input".into());
                                    continue;
                                }
                            }
                        }
                        None if hk.is_some() => match bridge::bind_at(&p.ms, &p.ps, &d, next) {
                            Ok(b) => b,
                            Err(x) => {
                                refuse(&mut st, x);
                                continue;
                            }
                        },
                        None => b,
                    }
                }
            };
            if next + b.views.heap_slots > HEAP_MOST {
                refuse(&mut st, format!("the view heap is full ({next} slots)"));
                continue;
            }
            let kept = &mut self.kept;
            let addrs = &o.addrs;
            // `align`: the offset an in-place view of these bytes needs.
            let canon = &mut canon;
            let small_in = &mut self.small;
            // A held row's slots placed last frame at its own small
            // slots (`same`: that slot, when the bytes are the held ones),
            // and where this frame's went (`own`).
            let mut hl = hk.and_then(|k| self.holds.get_mut(&k)).filter(|h| h.1.prev.len() == b.views.slots.len()).map(|h| (std::mem::take(&mut h.1.prev), std::mem::take(&mut h.1.vsrc)));
            let own = std::cell::Cell::new(None);
            // The version of each place made ([`Ver`]; None: none names it).
            let route: std::cell::Cell<Option<Ver>> = std::cell::Cell::new(None);
            let ins_on = replay > 0 && by_held;
            let mut row_ins: Vec<In> = Vec::new();
            let zc_memo = &mut self.zc_memo;
            let mut place = |a: &mut Arena, c: &mut Vec<(usize, usize, u64, u64)>, bytes: &'f [u8], size: u64, unit: u64, align: u64, cbv: bool, input: u32, same: Option<u64>, shared: Option<u32>| -> At {
                own.set(None);
                route.set(None);
                // Our pre-skin output where the row's range slots stand (no copy).
                if let Some((buf, ptrs)) = t.preskin
                    && ptrs.contains(&(bytes.as_ptr() as usize))
                {
                    zc.push(xnw_gpu::shared::SharedRange { buffer: buf.clone(), offset: 0, len: bytes.len() });
                    return At::Shared(zc.len() as u32 - 1);
                }
                // CLOTH2: a row's slot-20 block copy, where it lies in our words buffer (written with the words).
                if let Some(&(_, at)) = f.regions.iter().find(|g| g.0 == bytes.as_ptr() as usize) {
                    zc.push(xnw_gpu::shared::SharedRange { buffer: t.words.0.clone(), offset: at, len: bytes.len() });
                    return At::Shared(zc.len() as u32 - 1);
                }
                // PERF1: the split's clock reads (three an input) only with `XNW_BRIDGE_PLACE_SPLIT=on`.
                let clock = || split.then(std::time::Instant::now);
                let since = |t: Option<std::time::Instant>| t.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
                let tz = clock();
                // hold: a range found on an earlier frame whose window and pages stay.
                let mut range = |title: u64| -> Option<xnw_gpu::shared::SharedRange> {
                    if hold == 0 {
                        return xnw_gpu::shared::world_range(title, size);
                    }
                    let copy = |r: &xnw_gpu::shared::SharedRange| xnw_gpu::shared::SharedRange { buffer: r.buffer.clone(), offset: r.offset, len: r.len };
                    if let Some(r) = zc_memo.get(&(title, size)) {
                        return Some(copy(r));
                    }
                    let r = xnw_gpu::shared::world_range(title, size)?;
                    zc_memo.insert((title, size), copy(&r));
                    Some(r)
                };
                if zc_on
                    && (!cbv || zc_cbv)
                    && let Some(&(_, title)) = addrs.iter().find(|x| x.0 == bytes.as_ptr() as usize)
                    && let Some(r) = range(title)
                    && r.offset.is_multiple_of(if cbv { 256 } else { align })
                {
                    zc_bytes += size;
                    zc.push(r);
                    if ins_on {
                        let ai = addrs.iter().position(|x| x.0 == bytes.as_ptr() as usize).unwrap_or(0) as u32;
                        route.set(Some(Ver::Shared { ai, title, generation: zc_gen }));
                    }
                    place_ms[0] += since(tz);
                    return At::Shared(zc.len() as u32 - 1);
                }
                place_ms[0] += since(tz);
                let ts = clock();
                let pk = (bytes.as_ptr() as usize, bytes.len(), size, unit);
                if small
                    && bytes.len() < KEEP_MIN
                    && rid.is_some()
                    && let Some(&o) = by_ptr.get(&pk)
                {
                    route.set(Some(Ver::Small(o)));
                    place_ms[1] += since(ts);
                    return At::Small(o);
                }
                if share
                    && small
                    && bytes.len() < KEEP_MIN
                    && rid.is_some()
                    && let Some(o) = twin_at(&by_bytes, pk)
                {
                    by_ptr.insert(pk, o);
                    route.set(Some(Ver::Small(o)));
                    place_ms[1] += since(ts);
                    return At::Small(o);
                }
                let id = if small && bytes.len() < KEEP_MIN { small_id(shared, multi(pk.0), unit, rid, input, &mut shared_occ) } else { None };
                if let Some(id) = id
                    && let Some(o) = same
                    && small_in.touch(id, o, size, unit, bytes.len().min(size as usize))
                {
                    // The held bytes at the draw's own slot: written there already.
                    by_ptr.insert(pk, o);
                    by_bytes.insert((pk.0, pk.1), (o, Small::cap(size)));
                    own.set(Some(o));
                    route.set(Some(Ver::Small(o)));
                    place_ms[1] += since(ts);
                    return At::Small(o);
                }
                if let Some(id) = id
                    && let Some((o, w, fresh)) = small_in.place_steady(id, bytes, size, unit)
                {
                    by_ptr.insert(pk, o);
                    by_bytes.insert((pk.0, pk.1), (o, Small::cap(size)));
                    own.set(Some(o));
                    route.set(Some(Ver::Small(o)));
                    // SAFETY: as below: the draw's own slot lies inside the
                    // mapped buffer, and no list in flight reads it (the pass
                    // waited for the last frame's fence; one row a frame writes it).
                    unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr().add(w.start), small_to.add(o as usize + w.start), w.len());
                        if fresh {
                            let n = bytes.len().min(size as usize);
                            std::ptr::write_bytes(small_to.add(o as usize + n), 0, Small::cap(size) as usize - n);
                        }
                    }
                    place_ms[1] += since(ts);
                    return At::Small(o);
                }
                if small
                    && bytes.len() < KEEP_MIN
                    && let Some((o, fresh)) = small_in.place(bytes, size, unit)
                {
                    if fresh {
                        let n = bytes.len().min(size as usize);
                        // SAFETY: `o` and its capacity lie inside the mapped buffer
                        // (`Small::place`); no list in flight reads it (the pass
                        // waited for the last frame's fence; the entry is new).
                        unsafe {
                            std::ptr::copy_nonoverlapping(bytes.as_ptr(), small_to.add(o as usize), n);
                            std::ptr::write_bytes(small_to.add(o as usize + n), 0, Small::cap(size) as usize - n);
                        }
                    }
                    place_ms[1] += since(ts);
                    return At::Small(o);
                }
                let tk = clock();
                // A copied input of a title address at its own place (`XNW_BRIDGE_REPLAY`).
                if replay > 0
                    && bytes.len() >= KEEP_MIN
                    && let Some(ai) = addrs.iter().position(|x| x.0 == bytes.as_ptr() as usize)
                    && let Some((o, fresh)) = kept.place_steady(addrs[ai].1, bytes, size, unit)
                {
                    if fresh {
                        c.push((bytes.as_ptr() as usize, bytes.len().min(size as usize), size, o));
                    }
                    route.set(Some(Ver::Steady { ai: ai as u32, title: addrs[ai].1, o }));
                    place_ms[2] += since(tk);
                    return At::Kept(o);
                }
                let bytes = if tables && bytes.len() < KEEP_MIN { *canon.entry(bytes).or_insert(bytes) } else { bytes };
                let (at, fresh) = kept.place(a, bytes, size, unit, arcs.get(&(bytes.as_ptr() as usize)).copied());
                if let (At::Kept(o), true) = (at, fresh) {
                    c.push((bytes.as_ptr() as usize, bytes.len().min(size as usize), size, o));
                }
                if let At::Kept(o) = at {
                    route.set(Some(Ver::Kept(o)));
                }
                place_ms[2] += since(tk);
                at
            };
            let tp = std::time::Instant::now();
            st.bind_ms[0] += (tp - tb).as_secs_f64() * 1e3;
            lap.mark(super::recprof::B_BIND);
            let mut views = Vec::new();
            for (j, &(s, v)) in b.views.slots.iter().enumerate() {
                if let Kind::RawUav { .. } = v.kind {
                    continue;
                }
                if let Some((size, n)) = v.kind.extent(v.bytes.len() as u64) {
                    let same = hl.as_ref().and_then(|h| h.0[j].as_ref()).filter(|x| std::ptr::eq(x.0.as_ptr(), v.bytes.as_ptr()) && x.0.len() == v.bytes.len()).map(|x| x.1);
                    views.push((s, v.kind, place(&mut arena, &mut copies, v.bytes, size, unit(v.kind), unit(v.kind), v.kind == Kind::Cbv, s - next, same, hl.as_ref().and_then(|h| shared_code(o, h.1[j]))), n));
                    if ins_on {
                        row_ins.push(In { slot: Some(j as u32), root: 0, size, unit: unit(v.kind), input: s - next, ver: route.get() });
                    }
                    if let Some(h) = &mut hl {
                        h.0[j] = own.get().and_then(|at| Some((src_arc(o, h.1[j])?.clone(), at)));
                    }
                }
            }
            if let (Some(k), Some(h)) = (hk, hl.take())
                && let Some(e) = self.holds.get_mut(&k)
            {
                (e.1.prev, e.1.vsrc) = h;
            }
            let mut roots = |vs: &[(u8, bool, &'f [u8])], params: &[(u8, bool, u32)], stage: u32| -> Result<Vec<(u32, bool, At)>, String> {
                params
                    .iter()
                    .map(|&(r, is_srv, param)| {
                        let v = vs.iter().find(|v| v.0 == r && v.1 == is_srv).ok_or_else(|| format!("root view {r} not bound"))?;
                        let size = (v.2.len() as u64).next_multiple_of(256);
                        let at = place(&mut arena, &mut copies, v.2, size, 256, 4, !is_srv, stage << 24 | param, None, Some(shared_root(r)));
                        if ins_on {
                            row_ins.push(In { slot: None, root: r, size, unit: 256, input: stage << 24 | param, ver: route.get() });
                        }
                        Ok((param, is_srv, at))
                    })
                    .collect()
            };
            let (mroots, proots) = match (roots(&b.mesh.views, &built.cp.views, 1), roots(&b.pixel.views, &built.gp.views, 2)) {
                (Ok(m), Ok(p)) => (m, p),
                (Err(x), _) | (_, Err(x)) => {
                    refuse(&mut st, x);
                    continue;
                }
            };
            st.bind_ms[1] += tp.elapsed().as_secs_f64() * 1e3;
            lap.mark(super::recprof::B_PLACE);
            words_at.entry(o.words_at).or_insert((words_base, o.words, &o.heap[..]));
            // The row's stride entries at its slots in the frame's one table.
            let base = next as usize;
            if !tables {
                strides.resize(base + b.views.heap_slots as usize, [0, 0]);
                strides[base..base + b.views.strides.len()].copy_from_slice(&b.views.strides);
            }
            next += b.views.heap_slots;
            most = most.max(b.views.heap_slots);
            let lay = &p.layout;
            let groups = u64::from(b.groups);
            let (rec, idx) = (rec_end, idx_end);
            rec_end += (groups * u64::from(lay.record_bytes())).next_multiple_of(256);
            idx_end += (groups * u64::from(lay.index_bytes())).next_multiple_of(256);
            let hk = hk.filter(|k| self.holds.contains_key(k));
            if was_held {
                st.row_key[1] += 1;
                st.row_key[2] += u32::from(wrote == (self.small.id_written, self.small.made, copies.len()));
            }
            rows.push(Row { i, built: built.clone(), b, views, mroots, proots, rec, idx, words_base, base: base as u32, hk, replay: false, ins: row_ins, cand });
            lap.mark(super::recprof::B_PUSH);
        }
        st.slots = next;
        st.bind_ms[2..].copy_from_slice(&place_ms);
        (st.small, st.small_bytes) = ([self.small.placed, self.small.made], self.small.made_bytes);
        (st.steady, st.steady_bytes) = ([self.small.id_placed, self.small.id_written], self.small.id_bytes);
        st.small_end = [self.small.end, self.small.size];
        let base_at = arena.place(&[0; 16], 16, 16);
        let stride_bytes: Vec<u8> = strides.iter().flatten().flat_map(|w| w.to_le_bytes()).collect();
        // The kept inputs' copies go through the upload memory too.
        let staged: Vec<u64> = copies
            .iter()
            .map(|&(p, n, size, _)| {
                // SAFETY: `p` and `n` name bytes of an input of `f` (alive here).
                let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, n) };
                arena.place(bytes, size, 16)
            })
            .collect();
        st.parts[0] = t0.elapsed().as_secs_f64() * 1e3;
        lap.mark(super::recprof::B_STAGED);
        lap.flush();
        if rows.is_empty() {
            st.ms = st.parts[0];
            return Ok(st);
        }
        let t1 = std::time::Instant::now();
        // PERF3: the ms waited for the list (`bridgepar`: the pass reaching the rows).
        let (waited, tex_wait): (f64, f64);
        // SAFETY: D3D12 calls with descriptions that outlive them; the
        // arena's parts point into `f` (the rows' bytes) and this
        // function's own tables, alive here; our objects live in `self`
        // (the caller waits for the frame's fence before the next record),
        // and those replaced go to `keep`.
        unsafe {
            let committed = |size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES| -> Result<ID3D12Resource, String> {
                let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: size.max(256), Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: flags, ..Default::default() };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).map_err(e)?;
                r.ok_or_else(|| "no resource".to_string())
            };
            // The upload buffer: mapped once, written each frame.
            if self.up.as_ref().is_none_or(|u| u.2 < arena.end) {
                if let Some((old, _, _)) = self.up.take() {
                    old.Unmap(0, None);
                    keep.push(windows::core::Interface::cast(&old).map_err(e)?);
                }
                let size = arena.end.next_power_of_two().max(16 << 20);
                let r = committed(size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
                let mut to = std::ptr::null_mut();
                r.Map(0, None, Some(&mut to)).map_err(e)?;
                self.up = Some((r, to.cast::<u8>(), size));
            }
            let (up, to, _) = self.up.clone().ok_or("no upload buffer")?;
            // Pieces of at most 4 MiB on up to 4 threads (a test run-br12:
            // two 16 MiB views a frame, 3-9 ms of copying into write-combined memory on one thread).
            let mut pieces: Vec<(usize, usize, usize, usize)> = Vec::new();
            for &(o, ptr, n, size) in &arena.parts {
                let (o, all) = (o as usize, size.max(16) as usize);
                for at in (0..all).step_by(PIECE) {
                    let len = PIECE.min(all - at);
                    pieces.push((o + at, ptr + at.min(n), n.saturating_sub(at).min(len), len));
                }
            }
            let to_at = to as usize;
            let copy = |p: &[(usize, usize, usize, usize)]| {
                // `from..from + n` lies in an input of `f`, `o..o + len` in the mapped upload buffer (the block's SAFETY).
                for &(o, from, n, len) in p {
                    std::ptr::copy_nonoverlapping(from as *const u8, (to_at + o) as *mut u8, n);
                    std::ptr::write_bytes((to_at + o + n) as *mut u8, 0, len - n);
                }
            };
            let bytes: usize = pieces.iter().map(|p| p.3).sum();
            if bytes < copy_min() || pieces.len() < 2 {
                copy(&pieces);
            } else {
                let per = pieces.len().div_ceil(4);
                std::thread::scope(|sc| {
                    for part in pieces.chunks(per) {
                        sc.spawn(|| {
                            xnw_core::win::helper_thread();
                            copy(part)
                        });
                    }
                });
            }
            // The stride table: this frame's entries, zeros past them.
            if self.strides.is_none() {
                let r = committed(STRIDE_ENTRIES * 8, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
                let mut to = std::ptr::null_mut();
                r.Map(0, None, Some(&mut to)).map_err(e)?;
                std::ptr::write_bytes(to.cast::<u8>(), 0, (STRIDE_ENTRIES * 8) as usize);
                self.strides = Some((r, to.cast::<u8>()));
                self.strides_len = 0;
            }
            let (stable, sp) = self.strides.clone().ok_or("no stride table")?;
            // Blocks write their own entries when they are made (below).
            if !tables {
                std::ptr::copy_nonoverlapping(stride_bytes.as_ptr(), sp, stride_bytes.len());
                if self.strides_len > stride_bytes.len() {
                    std::ptr::write_bytes(sp.add(stride_bytes.len()), 0, self.strides_len - stride_bytes.len());
                }
                self.strides_len = stride_bytes.len();
            }
            let sva = stable.GetGPUVirtualAddress();
            st.buffers = arena.parts.len() as u32;
            st.bytes = arena.end;
            st.kept = self.kept.used;
            st.kept_new = copies.iter().map(|c| c.2).sum();
            st.in_place = zc.len() as u32;
            st.in_place_bytes = zc_bytes;
            let uva = up.GetGPUVirtualAddress();
            // The mesh programs' outputs.
            let have = self.out.as_ref().map_or((0, 0), |o| (o.2, o.3));
            if have.0 < rec_end || have.1 < idx_end {
                if let Some((r, i, _, _)) = self.out.take() {
                    keep.push(windows::core::Interface::cast(&r).map_err(e)?);
                    keep.push(windows::core::Interface::cast(&i).map_err(e)?);
                }
                let (rn, inn) = (rec_end.next_power_of_two().max(have.0).max(1 << 20), idx_end.next_power_of_two().max(have.1).max(1 << 20));
                // Buffers start in (and decay to) COMMON; the first dispatch
                // promotes them to UAV (the layer ignores another initial state, id 1328).
                let common = D3D12_RESOURCE_STATE_COMMON;
                self.out = Some((committed(rn, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, common)?, committed(inn, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, common)?, rn, inn));
            }
            let (records, indices, _, _) = self.out.clone().ok_or("no outputs")?;
            let (rva, iva) = (records.GetGPUVirtualAddress(), indices.GetGPUVirtualAddress());
            // The kept buffer, and this frame's copies into it.
            if self.kept_buf.is_none() {
                self.kept_buf = Some((committed(self.kept.size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?, false));
            }
            let (kept, read) = self.kept_buf.clone().ok_or("no kept buffer")?;
            let before = if read { KEPT_READ } else { D3D12_RESOURCE_STATE_COMMON };
            // The copies into it are recorded with the list (below).
            self.kept_buf = Some((kept.clone(), true));
            let kva = kept.GetGPUVirtualAddress();
            let sva_small = self.small_buf.as_ref().map_or(0, |b| b.0.GetGPUVirtualAddress());
            let addr = |a: At| match a {
                At::Small(o) => sva_small + o,
                At::Frame(o) => uva + o,
                At::Kept(o) => kva + o,
                At::Shared(i) => zc[i as usize].buffer.GetGPUVirtualAddress() + zc[i as usize].offset,
            };
            // The drawn rows' words, each naming its textures at the row's base
            // (with blocks: once their places are known, below).
            let words = std::slice::from_raw_parts_mut(t.words.2, t.words.1 as usize);
            // LAMP1: the rows' mask words first; the words below win where they meet.
            for r in rows.iter().filter(|_| !tables) {
                st.masks += u32::from(Frame::write_mask(&f.rows[r.i].1, words));
            }
            for r in rows.iter().filter(|_| !tables) {
                let o = &f.rows[r.i].1;
                if let Some(d) = words.get_mut(o.words_at as usize..o.words_at as usize + 4 * WORDS) {
                    for (k, w) in o.words.iter().enumerate() {
                        let w = if o.heap.iter().any(|h| h.0 == k as u32) { w + r.words_base } else { *w };
                        d[4 * k..4 * k + 4].copy_from_slice(&w.to_le_bytes());
                    }
                }
                Frame::write_region(o, r.words_base, words);
                if let Some(g) = &o.region {
                    st.regions[0] += 1;
                    st.regions[1] += g.patches.len() as u32;
                }
            }
            st.parts[1] = t1.elapsed().as_secs_f64() * 1e3;
            // PERF3 `bridgepar`: the frame's textures are here from now (the
            // preparing thread waits for the pass's table; serial: at once).
            let tw = std::time::Instant::now();
            let _ = (t.texture)(0);
            tex_wait = tw.elapsed().as_secs_f64() * 1e3;
            let t2 = std::time::Instant::now();
            let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let sstep = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
            let srv = |format, first: u64, n: u32, stride: u32, flags| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_BUFFER, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: first, NumElements: n, StructureByteStride: stride, Flags: flags } } };
            let tex2d = |format| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } } };
            let raw_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R32_TYPELESS, ViewDimension: D3D12_UAV_DIMENSION_BUFFER, Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() } } };
            let tex_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() };
            let mut lap = Lap(std::time::Instant::now());
            // With blocks: textures by handle, once a frame (held to the
            // fence), then each row cut into its blocks with their keys
            //
            let mut cuts = std::mem::take(&mut self.cuts);
            cuts.clear();
            // The title's view dimension of every texture that is not 2D,
            // and the stand-in of a dimension: its texture and view.
            let mut dims: HashMap<u64, u8> = HashMap::new();
            for r in &rows {
                dims.extend(f.rows[r.i].1.dims.iter().copied());
            }
            let stand_in = |dim: u8| -> (&ID3D12Resource, D3D12_SHADER_RESOURCE_VIEW_DESC) {
                let tex = match (dim & !ZERO, dim & ZERO != 0) {
                    (9, _) => t.grey_shapes[0],
                    (2 | 3, _) => t.grey_shapes[2],
                    (8, false) => t.grey_shapes[1],
                    (8, true) => t.grey_shapes[4],
                    (_, true) => t.grey_shapes[3],
                    _ => t.grey,
                };
                (tex, one_texel_view(dim))
            };
            let dim_at = |dim: u8| DIMS.iter().position(|d| d.0 == dim & !ZERO).unwrap_or(0);
            // CLOTH3: what each slot-20 index rewritten to a word reads: the view written for
            // that word's texture ([`real`]'s rule), classed under the 2D sample that reads it.
            for r in &rows {
                let o = &f.rows[r.i].1;
                let Some(g) = &o.region else { continue };
                st.region_left += g.left;
                for &(_, k) in &g.patches {
                    let Some(&(_, h)) = o.heap.iter().find(|x| x.0 == k) else { continue };
                    let dim = r.built.lay.stand_in_dim(dims.get(&h).copied());
                    let ours = (t.texture)(h).filter(|x| view_fits(&x.1, dim));
                    let view = ours.as_ref().map(|x| x.1).unwrap_or_else(|| one_texel_view(dim));
                    let class = if ours.is_some() { region_view_class(Some(&view)) } else if view.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D { 1 } else { 2 };
                    st.region_views[class] += 1;
                    if ours.is_some() {
                        region_format(&mut st.region_formats, view.Format.0 as u32, 1);
                    }
                }
            }
            if tables {
                for r in &rows {
                    for &(_, h) in &r.b.handles {
                        texs.entry(h).or_insert_with(|| (t.texture)(h));
                    }
                }
                for (tr, _) in texs.values().flatten() {
                    keep.push(windows::core::Interface::cast(tr).map_err(e)?);
                }
                lap.to(&mut st.view_ms[7]);
                let cx = KeyCx { texs: &texs, grey: windows::core::Interface::as_raw(t.grey) as u64, dims: &dims, zc: &zc };
                // A table whose inputs are an earlier row's (same pipeline and
                // root, the same bytes by pointer) takes that row's block uncut:
                // the frame table of most rows.
                let mut memo: ByContent<(usize, u8, u64), Vec<(&'f [(u32, OSlot)], usize)>> = ByContent::default();
                let mut reuse: Vec<Option<usize>> = Vec::new();
                let mut look: Vec<(usize, u64, &'f [(u32, OSlot)])> = Vec::new();
                for r in &rows {
                    if r.replay {
                        // Its blocks are last frame's.
                        cuts.rows.push(cuts.blocks.len()..cuts.blocks.len());
                        continue;
                    }
                    let o = &f.rows[r.i].1;
                    let (lay, id) = (&r.built.lay, Arc::as_ptr(&r.built) as usize);
                    reuse.clear();
                    look.clear();
                    for (k, &root) in lay.roots.iter().enumerate() {
                        let mut from = None;
                        if let Some(ORoot::Table(t)) = o.roots.get(usize::from(root)) {
                            let h = table_hash(t);
                            from = memo.get(&(id, root, h)).and_then(|v| v.iter().find(|x| same_table(x.0, t)).map(|x| x.1));
                            look.push((k, h, &t[..]));
                        }
                        reuse.push(from);
                    }
                    cut(&r.b, lay, &r.views, r.base, &cx, &reuse, &mut cuts);
                    let tb = cuts.blocks.len() - lay.roots.len();
                    for &(k, h, t) in &look {
                        if reuse[k].is_none() {
                            memo.entry((id, lay.roots[k], h)).or_default().push((t, tb + k));
                        } else {
                            st.uncut += 1;
                        }
                    }
                }
                lap.to(&mut st.view_ms[6]);
            }
            let longest = cuts.blocks.iter().map(|b| b.len).max().unwrap_or(0);
            // The nulls' template: every row's (or block's) slots in one copy.
            let want_nulls = if tables { most.max(longest).max(GUARDS) } else { most };
            if self.nulls.as_ref().is_none_or(|n| n.1 < want_nulls) {
                let n = want_nulls.next_power_of_two().max(256);
                let h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: n, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, ..Default::default() }).map_err(e)?;
                let start = h.GetCPUDescriptorHandleForHeapStart().ptr;
                for s in 0..n as usize {
                    dev.CreateShaderResourceView(None, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, D3D12_BUFFER_SRV_FLAG_RAW)), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start + s * step });
                }
                self.nulls = Some((h, n));
            }
            let (nh, _) = self.nulls.clone().ok_or("no nulls")?;
            let nstart = nh.GetCPUDescriptorHandleForHeapStart();
            lap.to(&mut st.view_ms[0]);
            let vh: ID3D12DescriptorHeap = if tables {
                // The buffers blocks name: another one flushes them.
                let mut ids: Vec<windows::core::IUnknown> = vec![windows::core::Interface::cast(&up).map_err(e)?, windows::core::Interface::cast(&kept).map_err(e)?, windows::core::Interface::cast(t.words.0).map_err(e)?, windows::core::Interface::cast(t.grey).map_err(e)?];
                for g in t.grey_shapes {
                    ids.push(windows::core::Interface::cast(g).map_err(e)?);
                }
                if let Some((b, _)) = &self.small_buf {
                    ids.push(windows::core::Interface::cast(b).map_err(e)?);
                }
                let mut flush = self.bases.len() != ids.len() || self.bases.iter().zip(&ids).any(|(a, b)| a != b);
                if flush {
                    self.bases = ids;
                }
                // Slots the frame makes: blocks neither in `have` nor made earlier this frame.
                // Distinct blocks by their key's hash (blocks taken uncut are
                // their earlier block): a slot count that two keys of one hash
                // would short, so a block is made only below the heap's end
                // (checked again where it is made).
                let need = |cuts: &Cuts, have: Option<&BlockCache>| -> u64 {
                    let mut seen: std::collections::HashSet<u64, std::hash::BuildHasherDefault<Hashed>> = Default::default();
                    let mut n = 0u64;
                    for blk in cuts.blocks.iter().filter(|b| b.from.is_none()) {
                        if let Some((h, k)) = cuts.key(blk) {
                            if have.is_some_and(|m| m.find(h, k).is_some()) || !seen.insert(h) {
                                continue;
                            }
                        }
                        n += u64::from(blk.len);
                    }
                    n
                };
                let cap = self.bheap.as_ref().map_or(0, |h| h.1);
                let mut fresh = need(&cuts, (!flush).then_some(&self.blocks));
                if flush || u64::from(self.bump) + fresh > u64::from(cap) {
                    flush = true;
                    fresh = need(&cuts, None);
                }
                // A flush takes the replayed rows' blocks too: they are cut now.
                if flush && rows.iter().any(|r| r.replay) {
                    let cx = KeyCx { texs: &texs, grey: windows::core::Interface::as_raw(t.grey) as u64, dims: &dims, zc: &zc };
                    for (ri, r) in rows.iter_mut().enumerate().filter(|x| x.1.replay) {
                        cut(&r.b, &r.built.lay, &r.views, r.base, &cx, &[], &mut cuts);
                        cuts.rows[ri] = cuts.rows.pop().unwrap_or_default();
                        r.replay = false;
                        st.replayed[0] -= 1;
                    }
                    fresh = need(&cuts, None);
                }
                lap.to(&mut st.view_ms[5]);
                // A new heap's time goes with the nulls (the earlier split).
                let mut whole = false;
                if u64::from(GUARDS) + fresh > u64::from(cap) {
                    if u64::from(GUARDS) + fresh > u64::from(HEAP_MOST) {
                        self.cuts = cuts;
                        return Err(format!("the view heap cannot hold the frame's {fresh} block slots"));
                    }
                    if let Some((old, _)) = self.bheap.take() {
                        keep.push(windows::core::Interface::cast(&old).map_err(e)?);
                    }
                    let n = (4 * (u64::from(GUARDS) + fresh)).next_power_of_two().clamp(u64::from(HEAP_LEAST), u64::from(HEAP_MOST)) as u32;
                    let h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: n, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).map_err(e)?;
                    self.bheap = Some((h, n));
                    (whole, flush) = (true, true);
                }
                let (vh, cap_now) = self.bheap.clone().ok_or("no view heap")?;
                let vstart = vh.GetCPUDescriptorHandleForHeapStart().ptr;
                let at = |slot: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: vstart + slot as usize * step };
                if flush {
                    // Only the guards: every slot below a row's bound lies in a
                    // block made (typed nulls first) since the last flush, so
                    // the slots past the next free one are never read; their
                    // stride entries go back to zeros.
                    if whole {
                        dev.CopyDescriptorsSimple(GUARDS, at(0), nstart, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                        dev.CreateShaderResourceView(None, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(0));
                        dev.CreateUnorderedAccessView(None, None, Some(&raw_uav), at(2));
                        dev.CreateUnorderedAccessView(None, None, Some(&tex_uav), at(3));
                        dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at(4));
                        for (k, dim) in bridge::CONFINE_DIMS[1..].iter().enumerate() {
                            dev.CreateShaderResourceView(None, Some(&one_texel_view(*dim)), at(5 + k as u32));
                        }
                        st.writes[1] += 4 + bridge::NULL_IMAGES;
                    }
                    let end = self.bump.max(self.strides_len as u32 / 8).max(GUARDS) as usize;
                    std::ptr::write_bytes(sp.add(GUARDS as usize * 8), 0, (end - GUARDS as usize) * 8);
                    self.strides_len = 0;
                    if !self.blocks.is_empty() {
                        st.flushes += 1;
                    }
                    self.blocks.clear();
                    self.epoch += 1;
                    // The resources the old blocks named go with this frame's list
                    // (released after its fence), never before (VIctory: a
                    // texture's release must not take a slot a recorded list reads).
                    keep.extend(self.held.drain().map(|x| x.1));
                    self.bump = GUARDS;
                }
                lap.to(&mut st.view_ms[0]);
                // Each block bound where it is, or made at the next free slots.
                let mut words_first: ByContent<u32, (u32, [u32; 6])> = ByContent::default();
                // By root + 1 (0: the words' textures).
                let mut by_root = [[0u32; 3]; 257];
                for (r, rb) in rows.iter().zip(cuts.rows.clone()) {
                    let o = &f.rows[r.i].1;
                    for bi in rb {
                        let blk = &cuts.blocks[bi];
                        let len = blk.len;
                        st.blocks[0] += 1;
                        st.block_slots[0] += len;
                        let rc = &mut by_root[(blk.root + 1) as usize];
                        rc[0] += 1;
                        // Rows of one words offset share the first one's textures.
                        if blk.root < 0
                            && let Some(&w) = words_first.get(&o.words_at)
                        {
                            (cuts.blocks[bi].first, cuts.blocks[bi].masks) = w;
                            continue;
                        }
                        let key = cuts.key(blk);
                        let first = if let Some(from) = blk.from {
                            cuts.blocks[from].first
                        } else if let Some((h, k)) = key
                            && let Some(x) = self.blocks.get_mut(h, k)
                        {
                            // First use this frame: one distinct block.
                            if x.2 != self.frame {
                                st.blocks[1] += 1;
                                rc[1] += 1;
                                if x.2 < self.frame {
                                    st.blocks[2] += 1;
                                    rc[2] += 1;
                                }
                            }
                            x.2 = self.frame;
                            x.0
                        } else {
                            let first = self.bump;
                            if first + len > cap_now {
                                self.cuts = cuts;
                                return Err(format!("the view heap's {cap_now} slots cannot hold the frame's blocks"));
                            }
                            self.bump += len;
                            if let Some((h, k)) = key {
                                self.blocks.insert(h, k, (first, len, self.frame));
                            }
                            st.blocks[1] += 1;
                            st.blocks[3] += 1;
                            st.block_slots[1] += len;
                            rc[1] += 1;
                            lap.to(&mut st.view_ms[5]);
                            dev.CopyDescriptorsSimple(len, at(first), nstart, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                            std::ptr::write_bytes(sp.add(first as usize * 8), 0, len as usize * 8);
                            st.writes[0] += len;
                            lap.to(&mut st.view_ms[0]);
                            for en in &cuts.ents[blk.ents.clone()] {
                                let s = first + en.j;
                                match en.desc {
                                    Desc::Null => {}
                                    Desc::NullImage(dim) => {
                                        dev.CreateShaderResourceView(None, Some(&one_texel_view(dim)), at(s));
                                        st.writes[2] += 1;
                                    }
                                    Desc::Buf(kind, where_, n) => {
                                        let (res, off) = match where_ {
                                            At::Frame(o) => (&up, o),
                                            At::Kept(o) => (&kept, o),
                                            At::Small(o) => (&self.small_buf.as_ref().ok_or("no small inputs' buffer")?.0, o),
                                            At::Shared(i) => {
                                                let b = &zc[i as usize].buffer;
                                                self.held.entry(windows::core::Interface::as_raw(b) as usize).or_insert(windows::core::Interface::cast(b).map_err(e)?);
                                                (b, zc[i as usize].offset)
                                            }
                                        };
                                        let first = off / unit(kind);
                                        match kind {
                                            Kind::Raw => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT_R32_TYPELESS, off / 4, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s)),
                                            Kind::Structured(stride) => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT_UNKNOWN, first, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                                            Kind::Typed { format, .. } => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT(format), first, n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                                            Kind::Cbv => dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: addr(where_), SizeInBytes: n }), at(s)),
                                            Kind::RawUav { .. } => {}
                                        }
                                        st.writes[write_kind(kind)] += 1;
                                    }
                                    Desc::Words => {
                                        dev.CreateShaderResourceView(t.words.0, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, (t.words.1 / 4) as u32, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s));
                                        st.writes[7] += 1;
                                    }
                                    Desc::Tex(h, dim) => match real(&texs, h, dim) {
                                        Some((tr, desc)) => {
                                            dev.CreateShaderResourceView(tr, Some(desc), at(s));
                                            self.held.entry(windows::core::Interface::as_raw(tr) as usize).or_insert(windows::core::Interface::cast(tr).map_err(e)?);
                                            st.writes[8] += 1;
                                        }
                                        None => {
                                            let (tex, view) = stand_in(dim);
                                            dev.CreateShaderResourceView(tex, Some(&view), at(s));
                                            st.writes[9] += 1;
                                        }
                                    },
                                }
                                // Its stride entry (a raw twin by its slot in the heap).
                                if en.stride != [0, 0] {
                                    let words = [en.stride[0], if en.stride[1] != 0 { first + en.stride[1] } else { 0 }];
                                    std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), sp.add(s as usize * 8), 8);
                                }
                            }
                            lap.to(&mut st.view_ms[2]);
                            first
                        };
                        cuts.blocks[bi].first = first;
                        if cuts.blocks[bi].root < 0 {
                            // BND1: the block's slots by the dimension of the view each holds (the
                            // key spells the same descriptors, so a block found is as this one).
                            let masks = block_masks(&cuts.ents[cuts.blocks[bi].ents.clone()], &texs);
                            cuts.blocks[bi].masks = masks;
                            words_first.insert(o.words_at, (first, masks));
                        }
                    }
                }
                lap.to(&mut st.view_ms[5]);
                st.by_root = by_root.iter().enumerate().filter(|x| x.1[0] > 0).map(|(r, c)| (r as i32 - 1, *c)).collect();
                st.slots = self.bump;
                // The rows' table offsets, bounds and word base from their blocks.
                for (r, rb) in rows.iter_mut().zip(cuts.rows.clone()) {
                    if r.replay {
                        continue;
                    }
                    let plan = &cuts.blocks[rb];
                    let p = &r.built.programs;
                    for (s, c) in [(&p.ms, &mut r.b.mesh.constants), (&p.ps, &mut r.b.pixel.constants)] {
                        for u in s.user_data.iter().filter(|u| u.root == xnw_shader::RootType::Table && u.dword == 0) {
                            if let Some(blk) = plan.iter().find(|b| b.root == i32::from(u.slot))
                                && let Some(w) = c.get_mut(u.index as usize)
                            {
                                *w = blk.first * 32;
                            }
                        }
                    }
                    let end = plan.iter().map(|b| b.first + b.len).max().unwrap_or(0).max(GUARDS);
                    r.b.views.bounds[..6].copy_from_slice(&[end, 0, 1, 2, 3, 4]);
                    let wb = plan.iter().find(|b| b.root < 0);
                    r.words_base = wb.map_or(0, |b| b.first);
                    // BND1: the views the row picks by index are its draw words' block, each slot
                    // in the dimension of the view it holds; a row without one picks nulls only.
                    let m = wb.map_or([0; 6], |b| b.masks);
                    r.b.confine = [r.words_base, m[0], m[1], m[2], m[3], m[4], m[5], GUARDS - bridge::NULL_IMAGES];
                }
                let words = std::slice::from_raw_parts_mut(t.words.2, t.words.1 as usize);
                for r in &rows {
                    st.masks += u32::from(Frame::write_mask(&f.rows[r.i].1, words));
                }
                for r in &rows {
                    let o = &f.rows[r.i].1;
                    if let Some(d) = words.get_mut(o.words_at as usize..o.words_at as usize + 4 * WORDS) {
                        for (k, w) in o.words.iter().enumerate() {
                            let w = if o.heap.iter().any(|h| h.0 == k as u32) { w + r.words_base } else { *w };
                            d[4 * k..4 * k + 4].copy_from_slice(&w.to_le_bytes());
                        }
                    }
                    Frame::write_region(o, r.words_base, words);
                    if let Some(g) = &o.region {
                        st.regions[0] += 1;
                        st.regions[1] += g.patches.len() as u32;
                    }
                }
                lap.to(&mut st.view_ms[3]);
                // Each held row bound afresh leaves its replay for the
                // next frame (`check`: compared with the one that would have been made).
                if replay > 0 {
                    let cx = KeyCx { texs: &texs, grey: windows::core::Interface::as_raw(t.grey) as u64, dims: &dims, zc: &zc };
                    for r in rows.iter_mut().filter(|r| !r.replay) {
                        let Some(k) = r.hk else { continue };
                        let Some((_, h)) = self.holds.get_mut(&k) else { continue };
                        let o = &f.rows[r.i].1;
                        let mut tw = Vec::new();
                        tex_words(&r.b, o, &r.built.lay, &cx, &mut tw);
                        if let Some(c) = r.cand.take() {
                            let rel = |s: u32, base: u32| s.wrapping_sub(base);
                            let roots = |a: &[(u32, bool, At)], b: &[(u32, bool, At)]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.0 == y.0 && x.1 == y.1 && same_at(x.2, y.2, &zc));
                            let like = c.views.len() == r.views.len()
                                && c.views.iter().zip(&r.views).all(|(x, y)| rel(x.0, c.base) == rel(y.0, r.base) && x.1 == y.1 && x.3 == y.3 && same_at(x.2, y.2, &zc))
                                && roots(&c.mroots, &r.mroots)
                                && roots(&c.proots, &r.proots)
                                && c.b.mesh.constants == r.b.mesh.constants
                                && c.b.pixel.constants == r.b.pixel.constants
                                && c.b.mesh.lengths == r.b.mesh.lengths
                                && c.b.views.bounds == r.b.views.bounds
                                && c.b.confine == r.b.confine
                                && c.b.groups == r.b.groups
                                && c.b.sampler_slots == r.b.sampler_slots
                                && c.words_base == r.words_base
                                && c.b.handles.len() == r.b.handles.len()
                                && c.b.handles.iter().zip(&r.b.handles).all(|(x, y)| rel(x.0, c.base) == rel(y.0, r.base) && x.1 == y.1);
                            st.replayed[0] += 1;
                            st.replayed[1] += u32::from(!like);
                            if !like && st.replayed[1] == 1 {
                                xnw_core::warn!("bridgeps: frame {}: row {}: its replay is unlike its fresh binding", self.frame, f.rows[r.i].0);
                            }
                        }
                        let ok = r.ins.len() == r.views.len() + r.mroots.len() + r.proots.len() && r.ins.iter().all(|x| x.ver.is_some());
                        st.replay_why[usize::from(!ok)] += 1;
                        st.replay_why[2] += r.ins.iter().filter(|x| x.ver.is_none()).count() as u32;
                        h.replay = ok.then(|| Box::new(Replay { frame: self.frame, epoch: self.epoch, ins: std::mem::take(&mut r.ins), views: r.views.clone(), mroots: r.mroots.clone(), proots: r.proots.clone(), b: stripped(&r.b), base: r.base, words_base: r.words_base, texs: tw }));
                    }
                }
                vh
            } else {
                // The view heap: every row's slots, rewritten each frame.
                if self.views.as_ref().is_none_or(|v| v.1 < next) {
                    if let Some((old, _)) = self.views.take() {
                        keep.push(windows::core::Interface::cast(&old).map_err(e)?);
                    }
                    let n = next.next_power_of_two().clamp(HEAP_LEAST, HEAP_MOST);
                    let h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: n, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).map_err(e)?;
                    self.views = Some((h, n));
                }
                self.views.clone().ok_or("no view heap")?.0
            };
            self.cuts = cuts;
            let vstart = vh.GetCPUDescriptorHandleForHeapStart().ptr;
            let at = |slot: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: vstart + slot as usize * step };
            let mut sheaps: Vec<ID3D12DescriptorHeap> = Vec::with_capacity(rows.len());
            // Each row list's sampler heap once a frame, by the list's address (hashing
            // the whole key every row took 1.5 ms a frame of pk7a3 f962's 229 rows).
            let mut sfound: ByContent<(usize, u32), ID3D12DescriptorHeap> = ByContent::default();
            // BND1, without blocks: each row's picked slots by the dimension written there, by the row's base.
            let mut picked: HashMap<u32, [u32; 8]> = HashMap::new();
            for r in &rows {
                let (o, b) = (&f.rows[r.i].1, &r.b);
                if tables {
                    for &(_, h) in &b.handles {
                        let dim = r.built.lay.stand_in_dim(dims.get(&h).copied());
                        if let Some((_, desc)) = real(&texs, h, dim) {
                            st.textures += 1;
                            st.shaped += u32::from(desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D);
                        } else {
                            st.grey += 1;
                            st.zero += u32::from(dim & ZERO != 0);
                            st.grey_dims[dim_at(dim)] += 1;
                            if let Some(n) = st.grey_why.get_mut((t.flat)(h)) {
                                *n += 1;
                            }
                        }
                    }
                } else {
                    // Typed nulls at every slot, then the guards, the buffers, the textures.
                    let g = &b.views.bounds;
                    dev.CopyDescriptorsSimple(b.views.heap_slots, at(g[0] - b.views.heap_slots), nstart, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                    st.writes[0] += b.views.heap_slots;
                    lap.to(&mut st.view_ms[0]);
                    dev.CreateShaderResourceView(None, Some(&tex2d(DXGI_FORMAT_R8G8B8A8_UNORM)), at(g[1]));
                    dev.CreateUnorderedAccessView(None, None, Some(&raw_uav), at(g[3]));
                    dev.CreateUnorderedAccessView(None, None, Some(&tex_uav), at(g[4]));
                    dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC::default()), at(g[5]));
                    // BND1: the null images that are not 2D (`bridge::bind_at` put them last).
                    for (k, dim) in bridge::CONFINE_DIMS[1..].iter().enumerate() {
                        dev.CreateShaderResourceView(None, Some(&one_texel_view(*dim)), at(b.confine[7] + k as u32));
                    }
                    st.writes[1] += 4 + bridge::NULL_IMAGES;
                    // Slots the programs read through an image descriptor hold a null
                    // 2D texture, not the raw buffer null (a texture below overwrites it).
                    for (&(root, slot), _) in r.built.used.iter().filter(|u| *u.1 == NAMED_IMAGE) {
                        if matches!(o.roots.get(usize::from(root)), Some(ORoot::Table(_)))
                            && let Some(c) = b.views.constants.get(usize::from(root))
                            && c / 32 + slot < g[0]
                        {
                            dev.CreateShaderResourceView(None, Some(&one_texel_view(r.built.lay.image_dim(root, slot))), at(c / 32 + slot));
                            st.writes[2] += 1;
                        }
                    }
                    lap.to(&mut st.view_ms[1]);
                    for &(s, kind, where_, n) in &r.views {
                        let (res, off) = match where_ {
                            At::Frame(o) => (&up, o),
                            At::Kept(o) => (&kept, o),
                            At::Small(o) => (&self.small_buf.as_ref().ok_or("no small inputs' buffer")?.0, o),
                            At::Shared(i) => (&zc[i as usize].buffer, zc[i as usize].offset),
                        };
                        let first = off / unit(kind);
                        match kind {
                            Kind::Raw => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT_R32_TYPELESS, off / 4, n, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s)),
                            Kind::Structured(stride) => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT_UNKNOWN, first, n, stride.max(4), D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                            Kind::Typed { format, .. } => dev.CreateShaderResourceView(res, Some(&srv(DXGI_FORMAT(format), first, n, 0, D3D12_BUFFER_SRV_FLAG_NONE)), at(s)),
                            Kind::Cbv => dev.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: addr(where_), SizeInBytes: n }), at(s)),
                            Kind::RawUav { .. } => {}
                        }
                        st.writes[write_kind(kind)] += 1;
                    }
                    lap.to(&mut st.view_ms[2]);
                    // Slot 86 of the frame table: our words buffer, whole.
                    if let Some(s86) = b.views.constants.first().map(|c| c / 32 + WORDS_SLOT).filter(|s| b.views.slots.iter().any(|v| v.0 == *s)) {
                        dev.CreateShaderResourceView(t.words.0, Some(&srv(DXGI_FORMAT_R32_TYPELESS, 0, (t.words.1 / 4) as u32, 0, D3D12_BUFFER_SRV_FLAG_RAW)), at(s86));
                        st.writes[7] += 1;
                    }
                    let marks = picked.entry(r.base).or_insert([0; 8]);
                    for &(s, h) in &b.handles {
                        let dim = r.built.lay.stand_in_dim(dims.get(&h).copied());
                        let ours = (t.texture)(h);
                        let wrote = written_dim(ours.as_ref().map(|x| &x.1), dim);
                        // BND1: a table slot takes the texture as a view of the dimension the programs
                        // read it in or keeps the null image of it; a draw word's slot is marked with
                        // the dimension written.
                        match table_slot(&r.built.lay, &b.views.constants, s).map(|(root, k)| r.built.lay.image_dim(root, k)) {
                            Some(want) if !bridge::dim_reads(wrote, want) => {
                                dev.CreateShaderResourceView(None, Some(&one_texel_view(want)), at(s));
                                st.writes[2] += 1;
                                continue;
                            }
                            Some(_) => {}
                            None => bridge::confine_mark(marks, s - r.base, wrote),
                        }
                        match ours.filter(|x| view_fits(&x.1, dim)) {
                            Some((tr, desc)) => {
                                dev.CreateShaderResourceView(&tr, Some(&desc), at(s));
                                keep.push(windows::core::Interface::cast(&tr).map_err(e)?);
                                st.textures += 1;
                                st.shaped += u32::from(desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D);
                                st.writes[8] += 1;
                            }
                            None => {
                                let (tex, view) = stand_in(dim);
                                dev.CreateShaderResourceView(tex, Some(&view), at(s));
                                st.grey += 1;
                                st.zero += u32::from(dim & ZERO != 0);
                                st.grey_dims[dim_at(dim)] += 1;
                                st.writes[9] += 1;
                                if let Some(n) = st.grey_why.get_mut((t.flat)(h)) {
                                    *n += 1;
                                }
                            }
                        }
                    }
                }
                lap.to(&mut st.view_ms[3]);
                // Samplers: one heap per entry list, kept by content: one
                // default (linear, wrap) at every slot, the title's entries at their indices.
                let sid = (Arc::as_ptr(&o.samplers) as usize, b.sampler_slots);
                // A held row's list as last frame: its heap.
                if let Some(k) = r.hk
                    && let Some((_, h)) = self.holds.get(&k)
                    && let Some((list, slots, heap)) = &h.sheap
                    && Arc::ptr_eq(list, &o.samplers)
                    && *slots == b.sampler_slots
                {
                    sheaps.push(heap.clone());
                    lap.to(&mut st.view_ms[4]);
                    continue;
                }
                if let Some(h) = sfound.get(&sid) {
                    if let Some(k) = r.hk
                        && let Some(e) = self.holds.get_mut(&k)
                    {
                        e.1.sheap = Some((o.samplers.clone(), b.sampler_slots, h.clone()));
                    }
                    sheaps.push(h.clone());
                    lap.to(&mut st.view_ms[4]);
                    continue;
                }
                let sk = &sampler_key(&o.samplers, b.sampler_slots);
                let sh = match self.samplers.get_mut(sk) {
                    Some(h) => {
                        h.1 = self.frame;
                        h.0.clone()
                    }
                    None => {
                        let h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, NumDescriptors: b.sampler_slots.max(1), Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, ..Default::default() }).map_err(e)?;
                        let sat = |s: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: h.GetCPUDescriptorHandleForHeapStart().ptr + s * sstep };
                        let wrap = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
                        for s in 0..b.sampler_slots as usize {
                            dev.CreateSampler(&D3D12_SAMPLER_DESC { Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR, AddressU: wrap, AddressV: wrap, AddressW: wrap, MaxLOD: f32::MAX, ..Default::default() }, sat(s));
                        }
                        for (s, desc) in o.samplers.iter().filter(|s| s.0 < b.sampler_slots) {
                            dev.CreateSampler(desc, sat(*s as usize));
                        }
                        if self.samplers.len() >= SAMPLER_HEAPS
                            && let Some(old) = self.samplers.iter().filter(|x| x.1.1 != self.frame).min_by_key(|x| x.1.1).map(|x| x.0.clone())
                            && let Some((oh, _)) = self.samplers.remove(&old)
                        {
                            keep.push(windows::core::Interface::cast(&oh).map_err(e)?);
                        }
                        self.samplers.insert(sk.clone(), (h.clone(), self.frame));
                        h
                    }
                };
                sfound.insert(sid, sh.clone());
                if let Some(k) = r.hk
                    && let Some(e) = self.holds.get_mut(&k)
                {
                    e.1.sheap = Some((o.samplers.clone(), b.sampler_slots, sh.clone()));
                }
                sheaps.push(sh);
                lap.to(&mut st.view_ms[4]);
            }
            // BND1, without blocks: a row picks from the slots its words name (the first row's of its words offset).
            for r in rows.iter_mut().filter(|_| !tables) {
                let m = picked.get(&r.words_base).copied().unwrap_or_default();
                r.b.confine = [r.words_base, m[1], m[2], m[3], m[4], m[5], m[6], r.b.confine[7]];
            }
            st.parts[2] = t2.elapsed().as_secs_f64() * 1e3;
            let t3 = std::time::Instant::now();
            // Each row's pixel pipeline: the pair's, or its blend's.
            let mut psos = Vec::with_capacity(rows.len());
            for r in &rows {
                let b = f.rows[r.i].1.blend;
                st.blended += u32::from(blend_ok(b));
                st.blend_refused += u32::from(b[0] != 0 && !blend_ok(b));
                match st.blends.iter().position(|x| x.0 == b) {
                    Some(i) => st.blends[i].1 += 1,
                    None if st.blends.len() < 8 => st.blends.push((b, 1)),
                    None => {}
                }
                psos.push(r.built.pixel_pipeline(dev, b)?);
            }
            // The list from here on (PERF3: the first command of the rows is the kept copy).
            let tl = std::time::Instant::now();
            let l = &list()?;
            waited = tl.elapsed().as_secs_f64() * 1e3;
            if copies.is_empty() {
                if !read {
                    l.ResourceBarrier(&[transition(&kept, before, KEPT_READ)]);
                }
            } else {
                l.ResourceBarrier(&[transition(&kept, before, D3D12_RESOURCE_STATE_COPY_DEST)]);
                for (c, from) in copies.iter().zip(&staged) {
                    l.CopyBufferRegion(&kept, c.3, &up, *from, c.2);
                }
                l.ResourceBarrier(&[transition(&kept, D3D12_RESOURCE_STATE_COPY_DEST, KEPT_READ)]);
            }
            // Heaps and pipelines are set only when they change. A root
            // signature that indexes the heaps directly is set again after
            // new heaps (the driver takes the heaps with it: on the RX 5700 XT
            // a frame with rows took over the 1 s watchdog without it, WARP did not mind).
            let mut bound: Option<*mut std::ffi::c_void> = None;
            let set_heaps = |sh: &ID3D12DescriptorHeap, bound: &mut Option<*mut std::ffi::c_void>| -> bool {
                let k = windows::core::Interface::as_raw(sh);
                if *bound != Some(k) {
                    l.SetDescriptorHeaps(&[Some(vh.clone()), Some(sh.clone())]);
                    *bound = Some(k);
                    return true;
                }
                false
            };
            // The mesh programs.
            let mut pipe: *mut std::ffi::c_void = std::ptr::null_mut();
            for (r, sh) in rows.iter().zip(&sheaps) {
                let (built, b) = (&r.built, &r.b);
                if set_heaps(sh, &mut bound) {
                    pipe = std::ptr::null_mut();
                }
                if windows::core::Interface::as_raw(&built.cpso) != pipe {
                    l.SetComputeRootSignature(&built.croot);
                    l.SetPipelineState(&built.cpso);
                    pipe = windows::core::Interface::as_raw(&built.cpso);
                }
                l.SetComputeRoot32BitConstants(0, b.mesh.constants.len() as u32, b.mesh.constants.as_ptr().cast(), 0);
                for &(param, is_srv, a) in &r.mroots {
                    if is_srv { l.SetComputeRootShaderResourceView(param, addr(a)) } else { l.SetComputeRootConstantBufferView(param, addr(a)) }
                }
                if let Some(param) = built.cp.stride {
                    l.SetComputeRootShaderResourceView(param, sva);
                }
                l.SetComputeRootUnorderedAccessView(built.cp.records, rva + r.rec);
                l.SetComputeRootUnorderedAccessView(built.cp.records + 1, iva + r.idx);
                l.SetComputeRootShaderResourceView(built.cp.records + 2, uva + base_at);
                if let Some(param) = built.cp.lengths {
                    let mut lengths = b.mesh.lengths.clone();
                    lengths.resize(lengths.len().next_multiple_of(4).max(4), 0);
                    l.SetComputeRoot32BitConstants(param, lengths.len() as u32, lengths.as_ptr().cast(), 0);
                }
                if let Some(param) = built.cp.bounds {
                    l.SetComputeRoot32BitConstants(param, 8, b.views.bounds.as_ptr().cast(), 0);
                }
                if let Some(param) = built.cp.confine {
                    l.SetComputeRoot32BitConstants(param, 8, b.confine.as_ptr().cast(), 0);
                }
                l.Dispatch(b.groups, 1, 1);
            }
            let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            l.ResourceBarrier(&[transition(&records, uav, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE), transition(&indices, uav, D3D12_RESOURCE_STATE_INDEX_BUFFER)]);
            // The pixel programs into our targets.
            let target = shown_target() as usize;
            if target == 0 && t.metal.is_none() {
                l.OMSetRenderTargets(1, Some(&t.rtv), false, Some(&t.dsv));
            } else {
                // `XNW_BRIDGE_TARGET`: null targets before the one shown;
                // `ourlight=metal`: our colour, none (`ournormals=title`: the title normal target), the metal target.
                if self.null_rtv.is_none() {
                    let h: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).map_err(e)?;
                    dev.CreateRenderTargetView(None, Some(&D3D12_RENDER_TARGET_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D, ..Default::default() }), h.GetCPUDescriptorHandleForHeapStart());
                    self.null_rtv = Some(h);
                }
                let mut rtvs = [self.null_rtv.as_ref().ok_or("no null target")?.GetCPUDescriptorHandleForHeapStart(); 8];
                rtvs[target] = t.rtv;
                let n = match t.metal.filter(|_| target == 0) {
                    Some(m) => {
                        rtvs[2] = m;
                        if let Some(n) = t.normal {
                            rtvs[1] = n;
                        }
                        // `emissive=on`: the emissive target at 5 (3 and 4 none).
                        match t.emissive {
                            Some(e) => {
                                rtvs[5] = e;
                                6
                            }
                            None => 3,
                        }
                    }
                    None => target as u32 + 1,
                };
                l.OMSetRenderTargets(n, Some(rtvs.as_ptr()), false, Some(&t.dsv));
            }
            l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: t.shift[0], TopLeftY: t.shift[1], Width: t.width as f32, Height: t.height as f32, MinDepth: t.bias, MaxDepth: 1.0 }]);
            l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: t.width as i32, bottom: t.height as i32 }]);
            l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            pipe = std::ptr::null_mut();
            bound = None;
            for (k, (r, sh)) in rows.iter().zip(&sheaps).enumerate() {
                let (built, b) = (&r.built, &r.b);
                if set_heaps(sh, &mut bound) {
                    pipe = std::ptr::null_mut();
                }
                let lay = &built.programs.layout;
                let gpso = &psos[k];
                if windows::core::Interface::as_raw(gpso) != pipe {
                    l.SetPipelineState(gpso);
                    l.SetGraphicsRootSignature(&built.groot);
                    l.SetGraphicsRoot32BitConstants(0, 1, [0u32].as_ptr().cast(), 0);
                    pipe = windows::core::Interface::as_raw(gpso);
                }
                l.SetGraphicsRoot32BitConstants(1, b.pixel.constants.len() as u32, b.pixel.constants.as_ptr().cast(), 0);
                for &(param, is_srv, a) in &r.proots {
                    if is_srv { l.SetGraphicsRootShaderResourceView(param, addr(a)) } else { l.SetGraphicsRootConstantBufferView(param, addr(a)) }
                }
                if let Some(param) = built.gp.stride {
                    l.SetGraphicsRootShaderResourceView(param, sva);
                }
                l.SetGraphicsRootShaderResourceView(built.gp.records, rva + r.rec);
                if let Some(param) = built.gp.bounds {
                    l.SetGraphicsRoot32BitConstants(param, 8, b.views.bounds.as_ptr().cast(), 0);
                }
                if let Some(param) = built.gp.confine {
                    l.SetGraphicsRoot32BitConstants(param, 8, b.confine.as_ptr().cast(), 0);
                }
                let n = u64::from(b.groups) * u64::from(lay.index_bytes());
                l.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW { BufferLocation: iva + r.idx, SizeInBytes: n as u32, Format: DXGI_FORMAT_R32_UINT }));
                if dred_marks() {
                    // CLOTH3: a DRED breadcrumb context naming the row's programs (`d3d12x::host::dred_report`).
                    let o = &f.rows[r.i].1;
                    let m: Vec<u16> = format!("bridgeps ps {:08x} ms {:08x} row {} slot20 {}", (o.key.1 >> 96) as u32, (o.key.0 >> 96) as u32, r.i, u8::from(o.region.is_some())).encode_utf16().chain([0]).collect();
                    l.SetMarker(0, Some(m.as_ptr().cast()), (2 * m.len()) as u32);
                }
                // SKIN2 `XNW_REDUCE=..,mesh=nodraw`: the row dispatched, not drawn.
                // `mesh=scissor`: under a 1x1 scissor.
                let ps_hash = f.rows[r.i].1.key.1;
                let cut = crate::reduce::scissor(ps_hash);
                if cut {
                    l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: 1, bottom: 1 }]);
                }
                if !crate::reduce::no_draw(ps_hash) {
                    l.DrawIndexedInstanced(b.groups * lay.max_prims * 3, 1, 0, 0, 0);
                }
                if cut {
                    l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: t.width as i32, bottom: t.height as i32 }]);
                }
                st.drawn += 1;
            }
            l.ResourceBarrier(&[transition(&records, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, uav), transition(&indices, D3D12_RESOURCE_STATE_INDEX_BUFFER, uav)]);
            st.parts[3] = t3.elapsed().as_secs_f64() * 1e3 - waited;
        }
        st.ms = t0.elapsed().as_secs_f64() * 1e3 - waited - tex_wait;
        super::recprof::add(super::recprof::B_TEXWAIT, tex_wait);
        for (k, i) in [super::recprof::B_UPLOADS, super::recprof::B_VIEWS, super::recprof::B_LIST].into_iter().enumerate() {
            super::recprof::add(i, st.parts[k + 1]);
        }
        super::recprof::add(super::recprof::B_OTHER, st.ms - st.parts.iter().sum::<f64>());
        Ok(st)
    }
}

/// `bridgeps: frame F: rows R (drawn D, no pipelines N, refused X; not
/// ready T, unread U), textures from our table K (grey G), buffers B
/// (MiB), M ms; pipelines made/building/failed; lacking ...; record
/// parts ...; view heap S slots, kept on the device K MiB (N MiB new)`.
pub fn line(frame: u64, f: &Frame, st: &Stats) -> String {
    let c = cache().counts();
    let mut lacks = [0u32; LACKS.len()];
    for (_, o) in &f.rows {
        for (k, n) in lacks.iter_mut().enumerate() {
            *n += o.lacks >> k & 1;
        }
    }
    let lacks: Vec<String> = LACKS.iter().zip(lacks).filter(|(_, n)| *n > 0).map(|(w, n)| format!("{w} {n}")).collect();
    format!(
        "world render: frame {frame}: bridgeps: rows {} (drawn {}, no pipelines {}, refused {}; draws not ready {}, unread {}), textures from our table {} (grey {}), buffers {} ({:.1} MiB), record {:.1} ms; pipelines made {}, building {}, failed {}; rows with the title's blend {} (on but not taken {}; blends {:?}); lacking: {}{}; record parts: bindings {:.1} ms (bind {:.1}, place {:.1}: in place {:.1}, small {:.1}, kept or frame {:.1}), uploads {:.1} ms, views {:.1} ms, list {:.1} ms; view heap {} slots, kept on the device {:.1} MiB ({:.1} MiB new); in place {} ({:.1} MiB); small inputs {} (made {}, {:.0} KiB; buffer {:.1} of {:.0} MiB); steady {} (rewritten {}, {:.0} KiB); held {} (made {}, unlike {}); replayed {} (unlike {}; left {}, none for {} rows: {} inputs without a version; not replayed: none {}, frame or epoch {}, shape {}, textures {}, inputs {}, constants {}); grey by reason: {}; descriptor writes: {}; views by part: {}; {}",
        f.rows.len(),
        st.drawn,
        st.no_pipeline,
        st.refused,
        f.not_ready,
        f.unread,
        st.textures,
        st.grey,
        st.buffers,
        st.bytes as f64 / f64::from(1 << 20),
        st.ms,
        c[0],
        c[1],
        c[2],
        st.blended,
        st.blend_refused,
        st.blends,
        if lacks.is_empty() { "nothing".into() } else { lacks.join(", ") },
        if st.why.is_empty() { String::new() } else { format!("; first refused: {}", st.why) },
        st.parts[0],
        st.bind_ms[0],
        st.bind_ms[1],
        st.bind_ms[2],
        st.bind_ms[3],
        st.bind_ms[4],
        st.parts[1],
        st.parts[2],
        st.parts[3],
        st.slots,
        st.kept as f64 / f64::from(1 << 20),
        st.kept_new as f64 / f64::from(1 << 20),
        st.in_place,
        st.in_place_bytes as f64 / f64::from(1 << 20),
        st.small[0],
        st.small[1],
        st.small_bytes as f64 / 1024.0,
        st.small_end[0] as f64 / f64::from(1 << 20),
        st.small_end[1] as f64 / f64::from(1 << 20),
        st.steady[0],
        st.steady[1],
        st.steady_bytes as f64 / 1024.0,
        st.held[0],
        st.held[1],
        st.held[2],
        st.replayed[0],
        st.replayed[1],
        st.replay_why[0],
        st.replay_why[1],
        st.replay_why[2],
        st.replay_why[3],
        st.replay_why[4],
        st.replay_why[5],
        st.replay_why[6],
        st.replay_why[7],
        st.replay_why[8],
        {
            let g: Vec<String> = super::textures::FLAT_REASONS.iter().zip(st.grey_why).filter(|x| x.1 > 0).map(|(w, n)| format!("{w} {n}")).collect();
            let d: Vec<String> = DIMS.iter().zip(st.grey_dims).filter(|x| x.1 > 0).map(|(w, n)| format!("{} {n}", w.1)).collect();
            format!("{}; grey by view dimension: {}; of the grey, zero (views the title's GPU writes) {}; cubes, arrays and 3D bound {}; rows with their own slot-20 block {} (indices rewritten {}{}); rows with their mask words {}", if g.is_empty() { "none".to_string() } else { g.join(", ") }, if d.is_empty() { "none".to_string() } else { d.join(", ") }, st.zero, st.shaped, st.regions[0], st.regions[1], region_text(st), st.masks)
        },
        WRITES.iter().zip(st.writes).map(|(w, n)| format!("{w} {n}")).collect::<Vec<_>>().join(", "),
        VIEW_PARTS.iter().zip(st.view_ms).map(|(w, n)| format!("{w} {n:.2} ms")).collect::<Vec<_>>().join(", "),
        blocks_text(st),
    )
}

/// `table blocks: B (distinct D, kept from earlier frames K, made M), slots
/// S of T, flushes F, uncut U; by root r: B/D/K ...` ([`tables_on`]; `off` says so).
pub fn blocks_text(st: &Stats) -> String {
    if !tables_on() {
        return "table blocks off".into();
    }
    let roots: Vec<String> = st.by_root.iter().map(|(r, c)| format!("{} {}/{}/{}", if *r < 0 { "words".to_string() } else { format!("r{r}") }, c[0], c[1], c[2])).collect();
    format!("table blocks {} (distinct {}, kept from earlier frames {}, made {}), slots made {} of {}, flushes {}, uncut {}; by root (rows'/distinct/kept): {}; row key: rows with a key {}, bound from an earlier frame's binding {}, of those with no input rewritten {}", st.blocks[0], st.blocks[1], st.blocks[2], st.blocks[3], st.block_slots[1], st.block_slots[0], st.flushes, st.uncut, roots.join(", "), st.row_key[0], st.row_key[1], st.row_key[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decal rows' title blend (list.tsv rstate `1,5,6,1,2,1,1,7`) is a
    /// decal's; its target 1 writes R only, target 2 RGB, both with
    /// target 0's colour factors (warm cache, SHD1); opaque rows are not.
    #[test]
    fn decal_rows_blend_each_target_the_title_way() {
        let b = [1, 5, 6, 1, 2, 1, 1, 7];
        assert!(decal_blend(b));
        assert!(!decal_blend([0, 2, 1, 1, 2, 1, 1, 15]));
        assert!(!decal_blend([1, 2, 16, 1, 2, 18, 1, 15]));
        let (t1, t2) = (decal_target(b, 1), decal_target(b, 2));
        assert_eq!((t1.RenderTargetWriteMask, t2.RenderTargetWriteMask), (0x1, 0x7));
        assert_eq!((t1.SrcBlend, t1.DestBlend, t2.SrcBlend, t2.DestBlend), (D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA));
        assert_eq!(decal_target(b, 5).RenderTargetWriteMask, 0x8);
    }

    /// A made-up source: root 0 a table (slot 3 a constant buffer, slot 5
    /// a texture), root 4 constants (dword 1 the words' offset), heap
    /// index 7 a texture, 9 a buffer.
    struct Made;
    impl Source for Made {
        fn roots(&self) -> Vec<SrcRoot> {
            vec![SrcRoot::Table(0x1000), SrcRoot::None, SrcRoot::None, SrcRoot::None, SrcRoot::Constants(vec![VISIBLE_LIST | 5, 0x40])]
        }
        fn table(&self, root: u16, slot: u32) -> Option<SrcView> {
            match (root, slot) {
                (0, 3) => Some(SrcView { kind: view::CONSTANT, addr: 0x9000, bytes: 64, ..SrcView::default() }),
                (0, 4) => Some(SrcView { kind: view::BUFFER, addr: 0x5_0000_8000, bytes: 64, ..SrcView::default() }),
                (0, 5) => Some(SrcView { kind: view::TEXTURE, addr: 0xA000, bytes: 4096, ..SrcView::default() }),
                _ => None,
            }
        }
        fn bytes(&self, _: u16, slot: u32, v: &SrcView) -> Option<Arc<Vec<u8>>> {
            let mut b = vec![slot as u8; v.bytes as usize];
            b[..4].copy_from_slice(&0x2_0007u32.to_le_bytes());
            Some(Arc::new(b))
        }
        fn root_view(&self, _: u16, _: u64) -> Option<Arc<Vec<u8>>> {
            None
        }
        fn words(&self) -> Option<[u32; WORDS]> {
            let mut w = [0; WORDS];
            w[0] = 7;
            w[1] = 9;
            w[2] = 7;
            w[4] = 0x3f80_0000;
            w[5] = 0x2_0000;
            Some(w)
        }
        fn heap(&self, index: u32) -> Option<(u64, SrcView)> {
            let h = 0x10_0000 + 32 * u64::from(index);
            match index {
                _ if index >= 0x1_0000 => None,
                7 => Some((h, SrcView { kind: view::TEXTURE, ..SrcView::default() })),
                9 => Some((h, SrcView { kind: view::BUFFER, ..SrcView::default() })),
                _ => Some((h, SrcView::default())),
            }
        }
        fn samplers(&self) -> Option<Arc<Vec<(u32, SamplerDesc)>>> {
            None
        }
    }

    #[test]
    fn take_reads_the_named_slots_and_rewrites_the_words() {
        let used: Used = [((0, 3), NAMED_CBV), ((0, 4), 16), ((0, 5), NAMED_IMAGE), ((0, 6), NAMED_CBV), ((0, WORDS_SLOT), 4)].into_iter().collect();
        let mut o = take(&Made, (1, 2), &used);
        let ORoot::Table(t) = &o.roots[0] else { panic!("no table") };
        let kinds: Vec<(u32, String)> = t.iter().map(|(k, s)| (*k, format!("{s:?}").chars().take(12).collect())).collect();
        // Slot 3 a CBV, 4 structured, 5 a texture by handle, 6 zeros (named, no view), 86 left to the frame.
        assert_eq!(kinds.iter().map(|k| k.0).collect::<Vec<_>>(), vec![3, 4, 5, 6]);
        assert!(matches!(t[0].1, OSlot::Buffer(Kind::Cbv, _)) && matches!(t[1].1, OSlot::Buffer(Kind::Structured(16), _)));
        assert!(matches!(t[2].1, OSlot::Texture(0x10A0)));
        assert!(matches!(&t[3].1, OSlot::Buffer(Kind::Cbv, b) if b.iter().all(|x| *x == 0)));
        assert_eq!(o.lacks, 1 << 1 | 1 << 2 | 1 << 4);
        // Words: 7 a texture (word 0 and 2), 9 a buffer, 0 an empty slot.
        assert_eq!(&o.words[..4], &[0, NO_VIEW, 2, NO_VIEW]);
        // A word past every heap bound is data (1.0) and stays; one below it naming no view goes to the guard.
        assert_eq!(&o.words[4..6], &[0x3f80_0000, NO_VIEW]);
        assert_eq!(o.heap, vec![(0, 0x10_0000 + 32 * 7), (2, 0x10_0000 + 32 * 7)]);
        assert_eq!(o.words_at, 0x40);
        // The visible-meshlet bit is cleared.
        assert!(matches!(&o.roots[4], ORoot::Constants(c) if c[0] == 5));
        // The meshlet buffer at its GPU address names the group count.
        o.set_groups(0x1_0000_8000);
        assert_eq!(o.groups, 7);
        assert_eq!(o.handles().count(), 3);
    }

    /// ALIAS1 (GRP1): the group count follows the pair published, not a
    /// fixed alias: a test run's GPU view at 0x6_6720_0000.
    #[test]
    fn group_count_follows_the_pair() {
        use xnw_core::pair::Pair;
        let moved = Pair { cpu: 0x1_0000_0000, gpu: 0x6_6720_0000, len: 0x4_0000_0000 };
        let owned = |firsts: Vec<(u64, u32)>| Owned { firsts, ..Owned::default() };
        let mut o = owned(vec![(0x5_0000_8000, 9), (0x6_6720_8000, 7)]);
        o.set_groups_with(0x1_0000_8000, moved);
        assert_eq!((o.groups, o.lacks), (7, 0));
        // The fixed alias takes the view at 0x5_0000_8000, which is not the meshlet buffer here.
        let mut o = owned(vec![(0x6_6720_8000, 7)]);
        o.set_groups_with(0x1_0000_8000, Pair::DEFAULT);
        assert_eq!((o.groups, o.lacks), (0, 1 << 6));
        // A GPU-view mesh address and a CPU-view one each find their own view.
        let mut o = owned(vec![(0x6_6720_8000, 7)]);
        o.set_groups_with(0x6_6720_8000, moved);
        assert_eq!(o.groups, 7);
        let mut o = owned(vec![(0x1_0000_8000, 5)]);
        o.set_groups_with(0x1_0000_8000, moved);
        assert_eq!(o.groups, 5);
    }

    /// WIDE1 fault 1: the lighting pass's inputs come from the title target
    /// set, not the index: the foliage set's target 2 is velocity.
    #[test]
    fn lit_targets_follow_the_title_target_set() {
        assert_eq!(lit_targets(&[29, 28, 28, 28, 34, 28]), (Some(1), Some(2)));
        assert_eq!(lit_targets(&[29, 28, 28, 28, 28, 28]), (Some(1), Some(2)));
        assert_eq!(lit_targets(&[29, 28, 34]), (None, None));
        assert_eq!(lit_targets(&[10]), (None, None), "a set that is not in the table");
        assert_eq!(lit_targets(&[]), (Some(1), Some(2)), "no state captured: as before");
    }

    /// CLOTH2: a ped clothing row's slot-20 block (at b4 + 0x1c) copied
    /// with the indices its words name rewritten as the words are; a
    /// value no word names (0x20, 0x8c6e) is kept; the copy lies after the
    /// words; a row whose block names none of its words is left as it was.
    #[test]
    fn slot20_block_copied_with_its_word_indices() {
        let base = 0x200usize;
        let mut s20 = vec![0u8; base + 0x2000];
        for (off, w) in [(0usize, 0x8c6eu32), (4, 0x13f82), (8, 0x13f7c), (0xc, 0x13f73), (0x798, 0x20), (0x1000, 0x13f7c)] {
            s20[base + off..base + off + 4].copy_from_slice(&w.to_le_bytes());
        }
        let mut b4 = vec![0u8; 48];
        b4[0x1c..0x20].copy_from_slice(&(base as u32).to_le_bytes());
        let mut title = [0u32; WORDS];
        title[..3].copy_from_slice(&[0x13f7c, 0x13f82, 0x13f73]);
        title[5] = 0x2_0000;
        let row = |words_at: u32, heap: Vec<(u32, u64)>| Owned {
            roots: vec![ORoot::Table(vec![(20, OSlot::Buffer(Kind::Raw, Arc::new(s20.clone())))]), ORoot::Table(vec![(7, OSlot::Buffer(Kind::Cbv, Arc::new(b4.clone())))])],
            heap,
            words: [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            words_at,
            ident: 7,
            ..Owned::default()
        };
        let mut o = row(0x40, vec![(0, 0x1000), (1, 0x2000), (2, 0x3000)]);
        let g = cut_region(&mut o, &title).expect("a block naming the words");
        assert_eq!(g.patches, vec![(4, 1), (8, 0), (0xc, 2)], "+0x1000 lies past the span; 0x8c6e and 0x20 name no word");
        assert_eq!(g.bytes.len(), REGION_SPAN);
        let (ORoot::Table(t0), ORoot::Table(t1)) = (&o.roots[0], &o.roots[1]) else { panic!() };
        assert!(matches!(&t0[0].1, OSlot::Buffer(Kind::Raw, b) if Arc::ptr_eq(b, &g.bytes)));
        assert!(matches!(&t1[0].1, OSlot::Buffer(Kind::Cbv, b) if b[0x1c..0x20] == [0; 4]), "b4 + 0x1c := 0");
        o.region = Some(g);
        let mut none = row(0x80, vec![(5, 0x5000)]);
        assert!(cut_region(&mut none, &title).is_none(), "word 5 names nothing in the block");
        assert!(matches!(&none.roots[0], ORoot::Table(t) if matches!(&t[0].1, OSlot::Buffer(_, b) if b.len() == s20.len())));
        let f = Frame::new(vec![(0, o), (1, none)]);
        let at = f.rows[0].1.region.as_ref().unwrap().at;
        assert_eq!((at, f.words_end), (0x100, 0x100 + REGION_SPAN as u64), "after the words (0x80 + 64), 256-aligned");
        assert_eq!((f.regions.len(), f.rows[0].1.ident, f.rows[1].1.ident), (1, 0, 7));
        let mut buf = vec![0u8; f.words_end as usize];
        Frame::write_region(&f.rows[0].1, 1000, &mut buf);
        let dw = |o: usize| u32::from_le_bytes(buf[at as usize + o..at as usize + o + 4].try_into().unwrap());
        assert_eq!([dw(0), dw(4), dw(8), dw(0xc), dw(0x798)], [0x8c6e, 1001, 1000, 1002, 0x20]);
        assert_eq!(f.rows[0].1.region.as_ref().unwrap().left, 0);
        // CLOTH3: a word whose title view is a 2D array (or a target the title's GPU writes)
        // keeps the title's index in the block (read as the null); a block naming only such words is not cut.
        let mut arr = row(0x40, vec![(0, 0x1000), (1, 0x2000), (2, 0x3000)]);
        arr.dims = vec![(0x2000, 5), (0x3000, DIM_2D | ZERO)];
        let g = cut_region(&mut arr, &title).expect("word 0 is plain 2D");
        assert_eq!((g.patches, g.left), (vec![(8, 0)], 2), "+4 (2D array) and +0xc (target) left");
        let mut only = row(0x40, vec![(1, 0x2000)]);
        only.dims = vec![(0x2000, 9)];
        assert!(cut_region(&mut only, &title).is_none(), "a cube word alone: the row keeps the title's slot 20");
        assert!(region_word_2d(&[(1, 5)], 2) && region_word_2d(&[(2, DIM_2D)], 2) && !region_word_2d(&[(2, 3)], 2));
    }

    /// CLOTH3: the census classes of a region word's written view under a 2D sample.
    #[test]
    fn region_view_classes() {
        let tex2d = |f: i32, mips: u32| D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT(f),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: mips, ..Default::default() } },
        };
        let c = |v: &D3D12_SHADER_RESOURCE_VIEW_DESC| region_view_class(Some(v));
        // BC1/BC3/BC5/BC7 (sRGB too), RGBA8, R8: sampled.
        for f in [71, 72, 77, 83, 98, 99, 28, 29, 61, 10] {
            assert_eq!(c(&tex2d(f, 9)), 0, "format {f}");
        }
        assert_eq!(region_view_class(None), 1);
        assert_eq!([c(&one_texel_view(5)), c(&one_texel_view(9)), c(&one_texel_view(2))], [2, 2, 2]);
        // Typeless (BC7, RGBA8, R32), depth, unknown.
        for f in [97, 27, 39, 40, 45, 0] {
            assert_eq!(c(&tex2d(f, 1)), 3, "format {f}");
        }
        for f in [30, 42, 57, 62, 64] {
            assert_eq!(c(&tex2d(f, 1)), 4, "format {f}");
        }
        assert_eq!([c(&tex2d(86, 1)), c(&tex2d(115, 1)), c(&tex2d(85, 1))], [5, 5, 0]);
        let ms = D3D12_SHADER_RESOURCE_VIEW_DESC { ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DMS, ..tex2d(28, 1) };
        assert_eq!(c(&ms), 6);
        assert_eq!(c(&tex2d(71, 0)), 7);
        let mut arr = one_texel_view(5);
        arr.Anonymous.Texture2DArray.ArraySize = 0;
        assert_eq!(c(&arr), 7);
        assert_eq!(REGION_CLASSES.len(), 8);
    }

    #[test]
    fn frame_words_at_their_offsets() {
        let o = |at: u32, k: u32| Owned { words_at: at, words: [k; WORDS], ..Owned::default() };
        let f = Frame::new(vec![(0, o(0x40, 1)), (1, o(0, 2)), (2, o(0x400, 3))]);
        assert_eq!(f.words_end, 0x440);
        let mut b = vec![0u8; 0x80];
        f.write_words(&mut b);
        assert_eq!((&b[0..4], &b[0x40..0x44]), (&2u32.to_le_bytes()[..], &1u32.to_le_bytes()[..]));
        // The frame table of a row's draw holds slot 86's place; root 4 is kept.
        let row = Owned { roots: vec![ORoot::Table(Vec::new()), ORoot::None, ORoot::None, ORoot::None, ORoot::Constants(vec![9, 0x40, 7])], groups: 1, ..o(0x40, 2) };
        let f = Frame::new(vec![(0, row)]);
        let d = f.draw(0, 128);
        assert!(matches!(&d.roots[0], Root::Table(t) if t.len() == 1 && t[0].0 == WORDS_SLOT));
        assert!(matches!(&d.roots[4], Root::Constants(c) if c == &vec![9, 0x40, 7]));
    }

    /// LAMP1: a vehicle row's mask words (rec+0x74 into slot 86) go with
    /// the row and into our words buffer at the title's offset, under the
    /// rows' words; no record, a 64-B one, or +0x74 = -1: none.
    #[test]
    fn mask_words_at_their_offset() {
        struct Arena;
        impl Source for Arena {
            fn roots(&self) -> Vec<SrcRoot> {
                Vec::new()
            }
            fn table(&self, _: u16, _: u32) -> Option<SrcView> {
                None
            }
            fn bytes(&self, _: u16, _: u32, _: &SrcView) -> Option<Arc<Vec<u8>>> {
                None
            }
            fn root_view(&self, _: u16, _: u64) -> Option<Arc<Vec<u8>>> {
                None
            }
            fn words(&self) -> Option<[u32; WORDS]> {
                None
            }
            fn heap(&self, _: u32) -> Option<(u64, SrcView)> {
                None
            }
            fn samplers(&self) -> Option<Arc<Vec<(u32, SamplerDesc)>>> {
                None
            }
            fn title_words(&self, at: u64, len: usize) -> Option<Vec<u8>> {
                (at + len as u64 <= 0x2000).then(|| (0..len).map(|i| (at as usize + i) as u8).collect())
            }
        }
        let rec = |len: usize, at: u32| {
            let mut b = vec![0u8; len];
            if len >= 0x78 {
                b[0x74..0x78].copy_from_slice(&at.to_le_bytes());
            }
            Owned { roots: vec![ORoot::None, ORoot::Table(vec![(0, OSlot::Buffer(Kind::Cbv, Arc::new(b)))])], words_at: 0x40, words: [5; WORDS], ..Owned::default() }
        };
        let o = rec(128, 0x1000);
        let m = mask_words(&Arena, &o).expect("mask");
        assert_eq!((m.0, m.1.len(), m.1[0], m.1[255]), (0x1000, MASK_BYTES, 0x00, 0xff));
        assert!(mask_words(&Arena, &rec(128, u32::MAX)).is_none(), "-1: no mask");
        assert!(mask_words(&Arena, &rec(64, 0x1000)).is_none(), "not the 128-B draw record");
        assert!(mask_words(&Arena, &rec(128, 0x1F80)).is_none(), "past the slot");
        // A second row's words inside the first row's mask win.
        let a = Owned { mask: Some(m), ..o };
        let b = Owned { words_at: 0x1010, words: [9; WORDS], ..Owned::default() };
        let f = Frame::new(vec![(0, a), (1, b)]);
        assert_eq!(f.words_end, 0x1100);
        let mut buf = vec![0u8; f.words_end as usize];
        f.write_words(&mut buf);
        assert_eq!((buf[0x1000], buf[0x100F], &buf[0x1010..0x1014], buf[0x1050], &buf[0x40..0x44]), (0x00, 0x0F, &9u32.to_le_bytes()[..], 0x50, &5u32.to_le_bytes()[..]));
    }
    /// A row's binding cut into blocks: the draw words' textures,
    /// each table with its raw twins after its slots, a named image slot
    /// past a table's views inside its block, slot 86 our words buffer.
    #[test]
    fn a_binding_cut_into_blocks() {
        use xnw_shader::RootType::{Constants, Table};
        let sh = |stage, user: &[(u32, u8, xnw_shader::RootType, u32)]| HostShader { stage, dxil: Vec::new(), user_data: user.iter().map(|&(index, slot, root, dword)| emit::UserData { index, slot, root, dword }).collect(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false };
        let ms = sh(HwStage::Ms, &[(2, 0, Table, 0), (3, 1, Table, 0), (5, 4, Constants, 0)]);
        let ps = sh(HwStage::Ps, &[(0, 0, Table, 0), (1, 2, Table, 0)]);
        let p = Programs { ms, layout: emit::mesh::Layout { max_vertices: 3, max_prims: 1, rows: 1, elements: Vec::new() }, vs: Vec::new(), ps, wave: 32, vertex: None };
        let used: Used = [((0, 1), NAMED_CBV), ((1, 2), 8), ((2, 0), NAMED_CBV), ((2, 3), NAMED_IMAGE), ((2, 5), NAMED_IMAGE)].into_iter().collect();
        let (cb, words, rec, st8, mat) = ([1u8; 256], [0u8; 64], [3u8; 128], [4u8; 64], [5u8; 256]);
        let v = |kind, bytes| Slot::Buffer(View { kind, bytes });
        let d = Draw {
            roots: vec![
                Root::Table(vec![(1, v(Kind::Cbv, &cb)), (86, v(Kind::Raw, &words))]),
                Root::Table(vec![(0, v(Kind::Raw, &rec)), (2, v(Kind::Structured(8), &st8))]),
                Root::Table(vec![(0, v(Kind::Cbv, &mat)), (3, Slot::Handle(0x5000))]),
                Root::None,
                Root::Constants(vec![7, 0x40]),
            ],
            heap: vec![(9, Slot::Handle(0x7000))],
            samplers: Vec::new(),
            sampler_slots: 1,
            groups: 1,
        };
        let b = bridge::bind_at(&p.ms, &p.ps, &d, 500).unwrap();
        let views: Vec<(u32, Kind, At, u32)> = b.views.slots.iter().map(|&(s, v)| (s, v.kind, At::Frame(u64::from(s)), 1)).collect();
        // The programs read root 2 slot 5 as a cube; the draw words' texture
        // 0x7000 is a cube view of the title's (DIMS: 9).
        let lay = Lay::new(&p, &used, &[(2, 5, 9)], 1 << DIM_2D | 1 << 9);
        assert_eq!(lay, Lay { roots: vec![0, 1, 2], named: vec![2, 3, 6], images: vec![vec![], vec![], vec![(3, DIM_2D), (5, 9)]], dims_read: 0x210 });
        // A title dimension the programs never read stays a 2D stand-in.
        assert_eq!((lay.stand_in_dim(Some(9)), lay.stand_in_dim(Some(2)), lay.stand_in_dim(None)), (9, DIM_2D, DIM_2D));
        // A zero stand-in stays zero (WIDE2); a 3D one the programs do not read is a zero 2D one.
        assert_eq!((lay.stand_in_dim(Some(DIM_2D | ZERO)), lay.stand_in_dim(Some(8 | ZERO)), lay.stand_in_dim(Some(9 | ZERO))), (DIM_2D | ZERO, DIM_2D | ZERO, 9 | ZERO));
        assert_eq!((dims_text(lay.dims_read), dims_text(0)), ("2D, cube".to_string(), "no images".to_string()));
        assert_eq!((lay.image_dim(2, 5), lay.image_dim(2, 3), lay.image_dim(7, 0)), (9, DIM_2D, DIM_2D));
        let texs = Texs::default();
        let dims: HashMap<u64, u8> = [(0x7000, 9)].into_iter().collect();
        let cx = KeyCx { texs: &texs, grey: 1, dims: &dims, zc: &[] };
        let mut c = Cuts::default();
        cut(&b, &lay, &views, 500, &cx, &[], &mut c);
        // A block's slots written out: descriptors and stride entries.
        let dense = |c: &Cuts, blk: &Block| -> (Vec<Desc>, Vec<[u32; 2]>) {
            let (mut d, mut s) = (vec![Desc::Null; blk.len as usize], vec![[0, 0]; blk.len as usize]);
            for e in &c.ents[blk.ents.clone()] {
                (d[e.j as usize], s[e.j as usize]) = (e.desc, e.stride);
            }
            (d, s)
        };
        let blocks: Vec<(Vec<Desc>, Vec<[u32; 2]>)> = c.blocks.iter().map(|x| dense(&c, x)).collect();
        assert_eq!(c.blocks.iter().map(|x| (x.root, x.len)).collect::<Vec<_>>(), vec![(-1, 10), (0, 87), (1, 4), (2, 6)]);
        assert_eq!(c.rows, vec![0..4]);
        assert_eq!(blocks[0].0[9], Desc::Tex(0x7000, 9), "the stand-in there is a cube");
        assert!(blocks[0].0[..9].iter().all(|d| *d == Desc::Null));
        assert!(matches!(blocks[1].0[1], Desc::Buf(Kind::Cbv, ..)) && blocks[1].0[86] == Desc::Words);
        // Root 1: the structured view at 2 names its raw twin, the block's slot 3.
        assert!(matches!(blocks[2].0[2], Desc::Buf(Kind::Structured(8), ..)) && matches!(blocks[2].0[3], Desc::Buf(Kind::Raw, ..)));
        assert_eq!(blocks[2].1, vec![[4, 0], [0, 0], [8, 4], [4, 0]]);
        // Root 2: the texture at 3, the named image at 5 past its views a null of the dimension read there (a cube).
        assert_eq!((blocks[3].0[3], blocks[3].0[5], blocks[3].0[4]), (Desc::Tex(0x5000, DIM_2D), Desc::NullImage(9), Desc::Null));
        // Only the slots that are not a plain null are listed.
        assert_eq!(c.ents[c.blocks[2].ents.clone()].len(), 3);
        // The same draw at another base cuts into the same blocks and keys, appended.
        let b2 = bridge::bind_at(&p.ms, &p.ps, &d, 20).unwrap();
        let views2: Vec<(u32, Kind, At, u32)> = b2.views.slots.iter().map(|&(s, v)| (s, v.kind, At::Frame(u64::from(s) + 480), 1)).collect();
        cut(&b2, &lay, &views2, 20, &cx, &[], &mut c);
        assert_eq!(c.rows, vec![0..4, 4..8]);
        for k in 0..4 {
            assert_eq!(blocks[k], dense(&c, &c.blocks[4 + k]));
            assert!(c.key(&c.blocks[k]).is_some());
            assert_eq!(c.key(&c.blocks[k]), c.key(&c.blocks[4 + k]));
        }
        assert_ne!(c.key(&c.blocks[2]), c.key(&c.blocks[3]));
        // A table taken from an earlier row's block: its entries and key, not cut again.
        cut(&b2, &lay, &views2, 20, &cx, &[Some(1), None, None], &mut c);
        assert_eq!((c.blocks[9].ents.clone(), c.blocks[9].len), (c.blocks[1].ents.clone(), 87));
        assert_eq!(c.key(&c.blocks[9]), c.key(&c.blocks[1]));
        assert_eq!(dense(&c, &c.blocks[10]), blocks[2]);
        // The same words' texture as a 2D view of the title's: another key
        // (a kept block made for the cube stand-in is not taken for it).
        let none = HashMap::new();
        let mut c2 = Cuts::default();
        cut(&b, &lay, &views, 500, &KeyCx { texs: &texs, grey: 1, dims: &none, zc: &[] }, &[], &mut c2);
        assert_ne!(c2.key(&c2.blocks[0]), c.key(&c.blocks[0]));
        assert_eq!(c2.key(&c2.blocks[3]), c.key(&c.blocks[3]));
        // BND1: the row's confinement. Its draw words' block by the dimension written: slot 9 holds a cube.
        let m = block_masks(&c.ents[c.blocks[0].ents.clone()], &texs);
        assert_eq!(m, [0, 0, 1 << 9, 0, 0, 0]);
        // The block at slot 40 of a shared heap whose rows reach slot 30,000 (22,000-29,000); guards 0-9.
        let (bounds, confine) = ([30_000, 0, 1, 2, 3, 4, 1, 0], [40, m[0], m[1], m[2], m[3], m[4], m[5], 5]);
        let read = |index: u32, dim: u8| bridge::confined_slot(&bounds, &confine, index, dim);
        // Its own cube, by the word that names it, through a cube instruction.
        assert_eq!(read(49, 9), 49);
        // The title's own heap numbers (9812, 9816, 11683 lay inside our heap, on other rows'
        // descriptors), an index below the block, past it, past the heap: the null of the instruction's dimension.
        for wild in [9812, 9816, 11683, 0, 39, 50, 72, 29_999, NO_VIEW, u32::MAX] {
            assert_eq!((read(wild, DIM_2D), read(wild, 9), read(wild, 8), read(wild, 5)), (0, 6, 7, 5), "{wild}");
        }
        // An own slot through an instruction of another dimension (the cube read as 2D or 3D), and an own
        // slot that holds no view (word 3): the null of the instruction's dimension too.
        assert_eq!((read(49, DIM_2D), read(49, 8), read(43, 9), read(43, DIM_2D)), (0, 7, 6, 0));
        // BND2: a flat view (1D, 1D array, 2D, 2D array) serves an instruction of any flat dimension;
        // a cube or a 3D view only its own, and no flat instruction reads them.
        assert!(bridge::dim_reads(4, 5) && bridge::dim_reads(5, 4) && bridge::dim_reads(4, 2) && bridge::dim_reads(3, 5) && bridge::dim_reads(9, 9));
        assert!(!bridge::dim_reads(9, 4) && !bridge::dim_reads(4, 9) && !bridge::dim_reads(4, 8) && !bridge::dim_reads(8, 2) && !bridge::dim_reads(1, 4));
        let mut w = [0u32; 8];
        bridge::confine_mark(&mut w, 3, 4);
        bridge::confine_mark(&mut w, 5, 9);
        bridge::confine_mark(&mut w, 32, 4);
        assert_eq!(w, [0, 1 << 3, 1 << 3, 1 << 5, 0, 1 << 3, 1 << 3, 0]);
        // A table slot: root 2 slot 3 is read as 2D. The title's view there a cube (which the pair
        // reads elsewhere): the null 2D image goes there, not the cube.
        let cube: HashMap<u64, u8> = [(0x5000, 9)].into_iter().collect();
        let mut c3 = Cuts::default();
        cut(&b, &lay, &views, 500, &KeyCx { texs: &texs, grey: 1, dims: &cube, zc: &[] }, &[], &mut c3);
        assert_eq!(dense(&c3, &c3.blocks[3]).0[3], Desc::NullImage(DIM_2D));
        assert_eq!(table_slot(&lay, &b.views.constants, 500 + 103), Some((2, 3)));
        assert_eq!(table_slot(&lay, &b.views.constants, 500 + 9), None);
    }

    /// The view at a texture slot takes the title's dimension among the
    /// kinds translated programs declare, and a one-texel view of each is
    /// one D3D12 accepts over the stand-in of that shape.
    #[test]
    fn stand_in_views_take_the_titles_dimension() {
        // Title view dimensions (D3D12): 4 2D, 5 2D array, 8 3D, 9 cube, 10 cube array, 2 1D, 3 1D array; 0 none; 6, 7 multisampled.
        assert_eq!([4, 5, 8, 9, 10, 2, 3, 0, 6, 7, 99].map(view_dim), [4, 5, 8, 9, 9, 2, 3, 4, 4, 4, 4]);
        // The title's dimension from a view's words: the view's own, a
        // resource view's default (slices: an array; a 3D resource), none
        // for a buffer or a capture without words.
        let words = |res: u32, depth: u32, dim: u32| {
            let mut w = [0u32; 30];
            (w[0], w[7], w[15]) = (res, depth, dim);
            w
        };
        assert_eq!([title_dim(view::PLACED, &words(3, 6, 9)), title_dim(view::TEXTURE, &words(3, 6, 10)), title_dim(view::TEXTURE, &words(3, 6, 0)), title_dim(view::TEXTURE, &words(3, 1, 0)), title_dim(view::TEXTURE, &words(4, 8, 0)), title_dim(view::TEXTURE, &[0; 30]), title_dim(view::BUFFER, &words(3, 6, 9))], [9, 10, 5, 0, 8, 0, 0]);
        // A view from our table stands only where its dimension is read.
        let v = |d| D3D12_SHADER_RESOURCE_VIEW_DESC { ViewDimension: d, ..Default::default() };
        let fits = |d, dims: [u8; 6]| dims.map(|x| view_fits(&v(d), x));
        assert_eq!(fits(D3D12_SRV_DIMENSION_TEXTURE2D, [4, 2, 3, 5, 8, 9]), [true, true, true, false, false, false]);
        assert_eq!(fits(D3D12_SRV_DIMENSION_TEXTURECUBE, [4, 2, 3, 5, 8, 9]), [false, false, false, false, false, true]);
        assert_eq!(fits(D3D12_SRV_DIMENSION_TEXTURE2DARRAY, [4, 2, 3, 5, 8, 9]), [false, false, false, true, false, false]);
        assert_eq!(fits(D3D12_SRV_DIMENSION_TEXTURE3D, [4, 2, 3, 5, 8, 9]), [false, false, false, false, true, false]);
        assert_eq!(fits(D3D12_SRV_DIMENSION_TEXTURECUBEARRAY, [4, 2, 3, 5, 8, 9]), [false; 6]);
        // The zero bit changes the stand-in, not which view fits or its shape.
        assert_eq!(fits(D3D12_SRV_DIMENSION_TEXTURE3D, [4 | ZERO, 2, 3, 5, 8 | ZERO, 9]), [false, false, false, false, true, false]);
        assert_eq!(one_texel_view(8 | ZERO).ViewDimension, D3D12_SRV_DIMENSION_TEXTURE3D);
        // GFX10 image instruction dimensions 0..7.
        assert_eq!([0, 1, 2, 3, 4, 5, 6, 7].map(read_dim), [2, 4, 8, 9, 3, 5, 4, 4]);
        for (dim, d3d) in [(4, D3D12_SRV_DIMENSION_TEXTURE2D), (5, D3D12_SRV_DIMENSION_TEXTURE2DARRAY), (9, D3D12_SRV_DIMENSION_TEXTURECUBE), (8, D3D12_SRV_DIMENSION_TEXTURE3D), (2, D3D12_SRV_DIMENSION_TEXTURE1D), (3, D3D12_SRV_DIMENSION_TEXTURE1DARRAY)] {
            let v = one_texel_view(dim);
            assert_eq!((v.ViewDimension, v.Format), (d3d, DXGI_FORMAT_R8G8B8A8_UNORM), "{dim}");
            assert!(DIMS.iter().any(|d| d.0 == dim));
        }
        // SAFETY: each description's own member.
        unsafe {
            assert_eq!((one_texel_view(5).Anonymous.Texture2DArray.ArraySize, one_texel_view(5).Anonymous.Texture2DArray.MipLevels), (1, 1));
            assert_eq!(one_texel_view(9).Anonymous.TextureCube.MipLevels, 1);
        }
    }

    /// A draw's textures whose title view is not 2D are listed with their
    /// dimension, table slots and draw words alike (pk6 f1205 row 1401:
    /// words 4 and 11 are 256x256x6 cube views, word 3 a 32-slice array).
    #[test]
    fn take_lists_the_textures_that_are_not_2d() {
        struct S;
        impl Source for S {
            fn roots(&self) -> Vec<SrcRoot> {
                vec![SrcRoot::Table(0x4000)]
            }
            fn table(&self, _root: u16, slot: u32) -> Option<SrcView> {
                match slot {
                    2 => Some(SrcView { kind: view::PLACED, dim: 5, ..SrcView::default() }),
                    3 => Some(SrcView { kind: view::PLACED, dim: 4, ..SrcView::default() }),
                    4 => Some(SrcView { kind: view::TEXTURE, ..SrcView::default() }),
                    _ => None,
                }
            }
            fn bytes(&self, _root: u16, _slot: u32, _v: &SrcView) -> Option<Arc<Vec<u8>>> {
                None
            }
            fn root_view(&self, _root: u16, _addr: u64) -> Option<Arc<Vec<u8>>> {
                None
            }
            fn words(&self) -> Option<[u32; WORDS]> {
                let mut w = [NO_VIEW + 1; WORDS];
                (w[4], w[11], w[12]) = (43499, 43442, 26992);
                Some(w)
            }
            fn heap(&self, index: u32) -> Option<(u64, SrcView)> {
                let dim = match index {
                    43499 => 9,
                    43442 => 10,
                    _ => 4,
                };
                Some((0x10_0000 + 32 * u64::from(index), SrcView { kind: view::PLACED, dim, ..SrcView::default() }))
            }
            fn samplers(&self) -> Option<Arc<Vec<(u32, SamplerDesc)>>> {
                None
            }
        }
        let used: Used = [((0, 2), NAMED_IMAGE), ((0, 3), NAMED_IMAGE), ((0, 4), NAMED_IMAGE)].into_iter().collect();
        let o = take(&S, (1, 2), &used);
        let mut dims = o.dims.clone();
        dims.sort_unstable();
        // Slot 4 is a view of a resource object (the title's GPU writes it): its stand-in is zero.
        assert_eq!(dims, vec![(0x4000 + 32 * 2, 5), (0x4000 + 32 * 4, DIM_2D | ZERO), (0x10_0000 + 32 * 43442, 9), (0x10_0000 + 32 * 43499, 9)]);
    }

    /// Small inputs by content: equal bytes at one offset on every
    /// frame that uses them, written once; freed after SMALL_FRAMES unused, their
    /// place taken by the next of that size; none past the buffer's end.
    #[test]
    fn small_inputs_stay_where_they_lie() {
        let mut m = Small::default();
        m.start(1024);
        m.next_frame();
        let (a, b, c) = ([1u8; 100], [2u8; 100], [3u8; 300]);
        assert_eq!(m.place(&a, 256, 256), Some((0, true)));
        assert_eq!(m.place(&b, 256, 256), Some((256, true)));
        assert_eq!(m.place(&a, 256, 256), Some((0, false)));
        // Another size or unit is another entry.
        assert_eq!(m.place(&a, 512, 256), Some((512, true)));
        for _ in 0..SMALL_FRAMES {
            m.next_frame();
            assert_eq!((m.place(&b, 256, 256), m.placed, m.made), (Some((256, false)), 1, 0));
        }
        m.next_frame();
        // Both `a` entries went unused for SMALL_FRAMES frames: freed; `c` takes the 512-byte place, a new 256-byte input the other.
        assert_eq!(m.place(&c, 512, 256), Some((512, true)));
        assert_eq!(m.place(&[9u8; 16], 256, 256), Some((0, true)));
        assert_eq!(m.place(&[8u8; 16], 256, 256), None);
        assert_eq!(m.place(&b, 256, 256), Some((256, false)));
    }

    #[test]
    fn a_held_binding_is_the_fresh_one_at_any_base() {
        use xnw_shader::RootType::{Constants, Table};
        let sh = |stage, user: &[(u32, u8, xnw_shader::RootType, u32)], cbvs: Vec<u8>| HostShader { stage, dxil: Vec::new(), user_data: user.iter().map(|&(index, slot, root, dword)| emit::UserData { index, slot, root, dword }).collect(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: cbvs, root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false };
        let ms = sh(HwStage::Ms, &[(2, 0, Table, 0), (3, 1, Table, 0), (5, 4, Constants, 1)], Vec::new());
        let ps = sh(HwStage::Ps, &[(0, 0, Table, 0), (1, 2, Table, 0), (4, 4, Constants, 0)], vec![3]);
        let typed = Kind::Typed { format: 2, bytes: 16 };
        let row = |seed: u8, h: u64| Owned {
            ident: 9,
            roots: vec![
                ORoot::Table(vec![(1, OSlot::Buffer(Kind::Cbv, Arc::new(vec![seed; 256])))]),
                ORoot::Table(vec![(0, OSlot::Buffer(Kind::Raw, Arc::new(vec![seed; 128]))), (2, OSlot::Buffer(Kind::Structured(8), Arc::new(vec![seed; 64]))), (4, OSlot::Buffer(typed, Arc::new(vec![seed; 96])))]),
                ORoot::Table(vec![(0, OSlot::Buffer(Kind::Cbv, Arc::new(vec![seed; 256]))), (3, OSlot::Texture(h))]),
                ORoot::View(Arc::new(vec![seed; 512])),
                ORoot::Constants(vec![u32::from(seed), 0x40 + u32::from(seed)]),
            ],
            heap: vec![(9, h + 0x100), (2, h + 0x200)],
            samplers: Arc::new(vec![(3, SamplerDesc::default())]),
            groups: 4 + u32::from(seed),
            ..Owned::default()
        };
        let f = Frame::new(vec![(0, row(1, 0x5000)), (1, row(2, 0x9000))]);
        let fresh = |i: usize, base: u32| bridge::bind_at(&ms, &ps, &f.draw(i, SAMPLER_SLOTS), base).unwrap();
        let (mut w0, mut w1) = (Vec::new(), Vec::new());
        shape(&f.rows[0].1, &mut w0);
        shape(&f.rows[1].1, &mut w1);
        assert_eq!(w0, w1);
        let h = Held::new(&fresh(0, 0), &f.rows[0].1, &ms, &ps, w0, 1).expect("held");
        // Raw twins (structured, typed) take their owner's bytes.
        assert_eq!(h.vsrc.iter().filter(|s| matches!(s, Src::Table(1, 1 | 2))).count(), 4);
        for (i, base) in [(0, 0), (0, 700), (1, 0), (1, 1234)] {
            let o = &f.rows[i].1;
            let held = h.bind(o, base, sampler_slots(&o.samplers)).expect("bound");
            let new = fresh(i, base);
            assert!(same_binding(&held, &new), "row {i} at {base}");
            assert_eq!(held.handles.iter().map(|x| x.1).collect::<Vec<_>>(), new.handles.iter().map(|x| x.1).collect::<Vec<_>>());
        }
        // Another shape is another binding.
        let mut other = row(1, 0x5000);
        if let ORoot::Table(t) = &mut other.roots[1] {
            t.pop();
        }
        let mut w2 = Vec::new();
        shape(&other, &mut w2);
        assert_ne!(w2, h.shape);
    }

    #[test]
    fn a_draws_changing_constants_keep_one_place() {
        let mut m = Small::default();
        m.start(4096);
        m.next_frame();
        let id = |occ| (7u64, (1u128, 2u128), occ, 3u32);
        let (a, mut b) = ([1u8; 200], [1u8; 200]);
        // Frame 1: each draw's input in a slot of its own, written whole.
        assert_eq!(m.place_steady(id(0), &a, 256, 256), Some((0, 0..200, true)));
        assert_eq!(m.place_steady(id(1), &a, 256, 256), Some((256, 0..200, true)));
        // A second place of one id in a frame: refused (the caller goes by content).
        assert_eq!(m.place_steady(id(0), &a, 256, 256), None);
        // Frame 2: the same bytes, the same place, nothing written.
        m.next_frame();
        assert_eq!(m.place_steady(id(0), &a, 256, 256), Some((0, 0..0, false)));
        // Frame 3: one byte changes: the same slot, only its 64-byte piece written.
        m.next_frame();
        b[130] = 5;
        assert_eq!(m.place_steady(id(0), &b, 256, 256), Some((0, 128..192, false)));
        m.next_frame();
        assert_eq!(m.place_steady(id(0), &b, 256, 256), Some((0, 0..0, false)));
        assert_eq!((m.id_placed, m.id_written), (1, 0));
        // Another size under the id: a new slot, the old one freed for the next input of its size.
        m.next_frame();
        assert_eq!(m.place_steady(id(0), &a, 512, 256), Some((512, 0..200, true)));
        assert_eq!(m.place_steady(id(2), &a, 256, 256), Some((0, 0..200, true)));
        // Unused for ID_FRAMES frames: the entries go, their slots free.
        for _ in 0..=ID_FRAMES {
            m.next_frame();
        }
        assert!(m.ids.is_empty());
        assert_eq!(m.free.values().map(Vec::len).sum::<usize>(), 3);
        assert_eq!(changed(&[0; 130], &[0; 130]), None);
        assert_eq!(changed(&[0; 130], &[[0; 129].as_slice(), &[1]].concat()), Some(128..130));
        assert_eq!(Small::align(76), 304);
        assert_eq!(Small::align(256), 256);
    }

    /// PERF1: a small input several rows hand over is placed under its
    /// place in the title's numbering, whatever row has it first; any
    /// changed part of that place is another identity.
    #[test]
    fn shared_inputs_keep_one_identity_whatever_row_is_first() {
        let table = |k: u32| ORoot::Table(vec![(k, OSlot::Buffer(Kind::Raw, Arc::new(vec![0u8; 64])))]);
        let row = |k: u32| Owned { roots: vec![table(k), ORoot::None, table(7)], ..Default::default() };
        let (a, b) = (row(56), row(56));
        let code = shared_code(&a, Src::Table(0, 0));
        assert_eq!(code, Some(0x8000_0000 | 56));
        assert_eq!(shared_code(&b, Src::Table(0, 0)), code);
        // Another slot of the table, another root, our words, a root view: other codes; no table there: none.
        assert_ne!(shared_code(&row(57), Src::Table(0, 0)), code);
        assert_eq!(shared_code(&a, Src::Table(2, 0)), Some(0x8000_0000 | 2 << 20 | 7));
        assert_eq!(shared_code(&a, Src::Words), Some(0x8000_0000 | 0xf_ffff));
        assert_eq!((shared_code(&a, Src::Table(1, 0)), shared_code(&a, Src::Table(0, 3)), shared_code(&a, Src::Heap(0))), (None, None, None));
        assert_ne!(shared_root(0), 0x8000_0000);
        let (r1, r2) = (Some((11u64, (1u128, 2u128), 0u32)), Some((12u64, (3u128, 4u128), 5u32)));
        let mut occ = ByContent::default();
        // Two frames with another first row: the same identity.
        let first = small_id(code, true, 2044, r1, 56, &mut occ);
        occ.clear();
        assert_eq!(small_id(code, true, 2044, r2, 51, &mut occ), first);
        assert!(first.is_some_and(|k| k.1 == SHARED_KEY && k.2 == 0));
        // Other bytes at the same place in one frame: the next occurrence.
        let second = small_id(code, true, 2044, r2, 51, &mut occ);
        assert!(second != first && second.is_some_and(|k| k.2 == 1));
        // Another unit (the raw twin) or another place: another identity, counted apart.
        assert!(small_id(code, true, 16, r1, 56, &mut occ).is_some_and(|k| Some(k) != first && k.2 == 0));
        assert!(small_id(Some(shared_root(3)), true, 2044, r1, 56, &mut occ).is_some_and(|k| Some(k) != first && k.2 == 0));
        // One row's alone, or no known place: the draw's own identity and input; no draw identity: none.
        assert_eq!(small_id(code, false, 2044, r1, 56, &mut occ), Some((11, (1, 2), 0, 56)));
        assert_eq!(small_id(None, true, 2044, r2, 51, &mut occ), Some((12, (3, 4), 5, 51)));
        assert_eq!(small_id(code, true, 2044, None, 56, &mut occ), None);
        // A raw twin takes its owner's slot when the offset suits its unit and the slot holds its size.
        let mut at: ByContent<(usize, usize), (u64, u64)> = ByContent::default();
        at.insert((0x1000, 68), (272, 128));
        assert_eq!(twin_at(&at, (0x1000, 68, 68, 16)), Some(272));
        // Other bytes or another length, a larger view, an offset its unit cannot start at: placed on its own.
        assert_eq!((twin_at(&at, (0x2000, 68, 68, 16)), twin_at(&at, (0x1000, 64, 64, 16))), (None, None));
        assert_eq!((twin_at(&at, (0x1000, 68, 256, 16)), twin_at(&at, (0x1000, 68, 68, 256))), (None, None));
    }

    /// PERF1: the identities' sweep by due frame frees an unused entry on
    /// the frame the whole walk did, and keeps a used one.
    #[test]
    fn steady_slots_go_on_their_due_frame() {
        let mut m = Small::default();
        m.start(4096);
        m.next_frame();
        let id = |n| (7u64, (1u128, 2u128), n, 3u32);
        let a = [1u8; 100];
        assert!(m.place_steady(id(0), &a, 256, 256).is_some() && m.place_steady(id(1), &a, 256, 256).is_some());
        // id(1) is used every frame, id(0) never again: it stays ID_FRAMES frames, then goes.
        for _ in 0..ID_FRAMES {
            m.next_frame();
            assert_eq!(m.place_steady(id(1), &a, 256, 256), Some((256, 0..0, false)));
            assert_eq!(m.ids.len(), 2);
        }
        m.next_frame();
        assert_eq!((m.ids.len(), m.ids.contains_key(&id(1)), m.free.values().map(Vec::len).sum::<usize>()), (1, true, 1));
        // Used again late, made again under its identity (another size): each still goes ID_FRAMES after its last use.
        for _ in 0..3 * ID_FRAMES {
            m.next_frame();
            assert!(m.place_steady(id(1), &a, 256, 256).is_some());
        }
        assert!(m.place_steady(id(2), &a, 256, 256).is_some());
        m.next_frame();
        assert!(m.place_steady(id(2), &a, 512, 256).is_some_and(|x| x.2));
        for k in 0..ID_FRAMES {
            assert_eq!(m.ids.len(), 2, "frame {k}");
            m.next_frame();
        }
        // id(1) was last used a frame before id(2).
        assert_eq!((m.ids.len(), m.ids.contains_key(&id(2))), (1, true));
        m.next_frame();
        assert!(m.ids.is_empty() && m.due.iter().all(Vec::is_empty));
    }

    /// Cut cost on a frame shaped like pk7c2 f1400 (722 rows; a frame
    /// table of 160 slots, 40 views; a mesh table of 12; a material table
    /// of 24 with 10 textures; 8 word textures): `cargo test -p xnw-world
    /// --lib cut_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn cut_cost() {
        use xnw_shader::RootType::Table;
        let sh = |stage, user: &[(u32, u8)]| HostShader { stage, dxil: Vec::new(), user_data: user.iter().map(|&(index, slot)| emit::UserData { index, slot, root: Table, dword: 0 }).collect(), samplers: Vec::new(), static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(), half_views: Vec::new(), address_map: false, descriptor_words: false };
        let p = Programs { ms: sh(HwStage::Ms, &[(0, 0), (1, 1)]), layout: emit::mesh::Layout { max_vertices: 3, max_prims: 1, rows: 1, elements: Vec::new() }, vs: Vec::new(), ps: sh(HwStage::Ps, &[(0, 0), (1, 2)]), wave: 32, vertex: None };
        let used: Used = (0..160).map(|k| ((0u16, k), NAMED_CBV)).chain((0..12).map(|k| ((1, k), 4))).chain((0..24).map(|k| ((2, k), NAMED_IMAGE))).collect();
        let lay = Lay::new(&p, &used, &[], 0);
        let bytes = vec![0u8; 4096];
        let v = |kind| Slot::Buffer(View { kind, bytes: &bytes });
        let d = Draw {
            roots: vec![
                Root::Table((0..160).step_by(4).map(|k| (k, v(if k % 8 == 0 { Kind::Structured(16) } else { Kind::Cbv }))).collect()),
                Root::Table((0..12).map(|k| (k, v(if k % 3 == 0 { Kind::Typed { format: 2, bytes: 16 } } else { Kind::Raw }))).collect()),
                Root::Table((0..24).map(|k| if k < 10 { (k, Slot::Handle(0x5000 + u64::from(k))) } else { (k, v(Kind::Cbv)) }).collect()),
            ],
            heap: (0..8).map(|k| (k, Slot::Handle(0x9000 + u64::from(k)))).collect(),
            samplers: Vec::new(),
            sampler_slots: 1,
            groups: 1,
        };
        let rows: Vec<_> = (0..722u32).map(|i| bridge::bind_at(&p.ms, &p.ps, &d, i * 300).unwrap()).collect();
        let views: Vec<Vec<(u32, Kind, At, u32)>> = rows.iter().map(|b| b.views.slots.iter().map(|&(s, v)| (s, v.kind, At::Frame(u64::from(s)), 1)).collect()).collect();
        let texs = Texs::default();
        let none = HashMap::new();
        let cx = KeyCx { texs: &texs, grey: 1, dims: &none, zc: &[] };
        let mut c = Cuts::default();
        for round in 0..3 {
            for reuse in [&[][..], &[Some(1), None, None][..]] {
                c.clear();
                let t = std::time::Instant::now();
                for (i, b) in rows.iter().enumerate() {
                    cut(b, &lay, &views[i], i as u32 * 300, &cx, if i == 0 { &[] } else { reuse }, &mut c);
                }
                println!("round {round} reuse {}: {} rows {:.2} ms; blocks {}, ents {}, key words {}", !reuse.is_empty(), rows.len(), t.elapsed().as_secs_f64() * 1e3, c.blocks.len(), c.ents.len(), c.keys.len());
            }
        }
    }
}
