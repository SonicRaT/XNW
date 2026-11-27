//! INT1: the title's per-material vehicle colour (`vehmat=on`).
//!
//! The vehicle pixel programs (PS_VehicleLiveryOffOnlyDeferred_*, the
//! bodies and the interiors alike) colour each vertex's material from a
//! GPU record, not from the texture alone:
//! - the mesh program reads the vertex stream at mesh word 0x04 (RGBA8,
//!   format nibble 6) and takes `slot = T9+0x100[u8(alpha)]` (a 256-entry
//!   u32 table, the vehicle's material remap) [V MS 42c87283 0x2b4-0x2dc,
//!   0xd38-0xdc0];
//! - the pixel program reads `T8+0x540` (76-byte elements) at element index
//!   `3 * (T9+0x100[0x400] + slot)` [V PS a8000c5c 0x7fc-0x818; 0e9eff9d
//!   0x3420-0x3430]: element +16 RGB8 (the colour when any byte is
//!   non-zero), +20 byte 0 palette row, byte 1 palette layer (else a palette
//!   sample) [V].
//! The CPU writer is the function (slot a4; T8+0xAA0 rows a7 + 100 * slot,
//! T8+0x540 records a8 + 228 * slot: elements at +0, +76, +152) [V pseudo].
//!
//! PAL1: element 0's colour, as a8000c5c makes it [V disasm 0x29c4-0x2a20,
//! 0x2a44-0x2ab8, 0x2c48-0x2c84]: `c = rgb / 255` when any of +16's RGB
//! bytes is non-zero, else `c = Pal.SampleLevel(u, (row + 0.5) / 256,
//! layer)` (the palette: draw word 8, `s14 + 32` of the T8+0xAC0 words, a
//! 2D array; pk8a: heap 35700, 128 x 256 x 8, R8G8B8A8_UNORM_SRGB, origin
//! ttdb_vehicles_loose_0_1.ttpf#5051); then `c += s18 * (Tex0.rgb - c)`
//! (s18: a constant at +0xd0 of the s[4:7] buffer [U], taken as 0: the
//! colour replaces the word-0 texture). u = saturate(v100 + Tex0.g) (v100
//! a per-pixel term from s12 [U], taken as 0); with +48 bit 27 u is a
//! position gradient from the T8+0xA00 record [V 0x2964-0x29b8; not
//! ported: Tex0.g too]. Elements 1 and 2 (the second layers, weights from
//! other textures) are the same rule [V 0x2698-0x28a0], not ported.
//!
//! Here: the decode, and a block per draw: 256 entries (index = the
//! vertex's alpha byte) then the palette rows the entries name, 128 texels
//! each. Pure: no title, no device.

/// Bytes of one T8+0x540 element (the view's stride) [V pk8a slot 42 width 76].
pub const ELEMENT: u64 = 76;
/// Elements per material slot (228-byte records at the writer).
pub const SLOT_ELEMENTS: u64 = 3;
/// The remap buffer (T9+0x100): 256 u32 slots, the vehicle's base at +0x400.
pub const REMAP_BYTES: usize = 0x404;
/// Dwords a table takes in the frame's instance buffer: 256, padded to
/// whole instances (12 floats) so later instance indices stay aligned.
pub const LUT_DWORDS: usize = 264;
/// A table entry's kind (bits 24-31; 0: none, the texture is kept):
/// the colour, which replaces the texture [V a8000c5c 0x2c58-0x2c84, s18 0 I];
pub const HAS_COLOUR: u32 = 1 << 24;
/// a palette row of sRGB texels (bits 0-15: its row block after the entries);
pub const PALETTE_SRGB: u32 = 2 << 24;
/// a palette row of UNORM texels;
pub const PALETTE_UNORM: u32 = 3 << 24;
/// the colour times the texture (`vehmat=mul`: INT1/INT2's look).
pub const COLOUR_MUL: u32 = 4 << 24;
/// The kind's bits of an entry (24-26); bits 27-31 are CAR1's ([`COAT`], [`COAT_ROUGH_SHIFT`]).
pub const KIND_MASK: u32 = 7 << 24;
/// CAR1B (`carshade=on`): the slot is exterior paint: element 0's +32 RGB8
/// (a second colour the program applies when any byte is non-zero, weighted
/// by a mask [V a8000c5c 0x2d30-0x2d9c]) is set. pk8a f1029 (base 23909):
/// set on the body's paint slots (+16 and +32 both 220202) only; the trim,
/// pedal, seat and dash slots (palette rows, RGB 1-10 greys) hold 0 [V
/// the capture's records].
pub const COAT: u32 = 1 << 31;
/// CAR1B: bits 27-30 hold the high nibble of element 0's +24 byte 3: the
/// upper end of a range the program lerps from byte 2 to byte 3 by a
/// per-pixel term [V a8000c5c 0x26f8-0x2764]; a roughness [I: 0x1e on the
/// paint slots, 0x7f-0xff on trims]. Only with [`COAT`].
pub const COAT_ROUGH_SHIFT: u32 = 27;

/// CAR1B: the coat bits of a slot from element 0's +24 and +32 bytes.
pub fn coat_bits(at24: &[u8; 8], at32: &[u8; 8]) -> u32 {
    if at32[..3] == [0; 3] { 0 } else { COAT | u32::from(at24[3] >> 4) << COAT_ROUGH_SHIFT }
}

/// Texels of one palette row in the block (the pk8a palette's width; other
/// widths are resampled, nearest).
pub const PAL_TEXELS: usize = 128;
/// Distinct palette rows one block holds at most (more: entry 0).
pub const PAL_ROWS_MOST: usize = 64;

/// One T8+0x540 element's colour fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Element {
    /// +16: RGB8 (byte 3 is 0xFF in the captures, unread by the program).
    pub rgb: [u8; 3],
    /// +20 byte 0, byte 1: the palette row and layer (used when `rgb` is 0).
    pub row: u8,
    pub layer: u8,
}

impl Element {
    /// From the element's bytes +16..+24 (`at16`: 8 bytes).
    pub fn decode(at16: &[u8]) -> Option<Element> {
        let b = at16.get(..8)?;
        Some(Element { rgb: [b[0], b[1], b[2]], row: b[4], layer: b[5] })
    }

    /// The table entry: the colour (`mul`: [`COLOUR_MUL`]), or 0 when the
    /// element names a palette row ([`table`] makes those).
    pub fn entry(&self, mul: bool) -> u32 {
        if self.rgb == [0; 3] { 0 } else { (if mul { COLOUR_MUL } else { HAS_COLOUR }) | u32::from(self.rgb[0]) | u32::from(self.rgb[1]) << 8 | u32::from(self.rgb[2]) << 16 }
    }
}

/// A vehicle palette's texels (mip 0 of each layer), RGBA8 as stored
/// (`srgb`: the view decodes them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Palette {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub srgb: bool,
    /// Layer after layer, row after row, `width` texels (R | G << 8 | B << 16 | A << 24).
    pub texels: Vec<u32>,
}

impl Palette {
    /// From provided texels (a 2D array or 2D, every slice's mips from the
    /// finest, `xnw_world::textures::Provided`'s order) of an 8-bit RGBA or
    /// BGRA format (DXGI 28/29, 87/91); None otherwise.
    pub fn from_provided(p: &crate::textures::Provided) -> Option<Palette> {
        let (srgb, bgra) = match p.format {
            28 => (false, false),
            29 => (true, false),
            87 => (false, true),
            91 => (true, true),
            _ => return None,
        };
        let (w, h) = (p.width as usize, p.height as usize);
        let layers = if p.dim == crate::textures::TEX_2D_ARRAY { p.depth.max(1) } else { 1 } as usize;
        // One slice: its mips from the finest, 4 bytes a texel.
        let slice: usize = (0..p.mips.max(1)).map(|m| (w >> m).max(1) * (h >> m).max(1) * 4).sum();
        if w == 0 || h == 0 || p.row_bytes as usize != w * 4 || p.rows.len() < slice * layers {
            return None;
        }
        let mut texels = Vec::with_capacity(w * h * layers);
        for l in 0..layers {
            let b = &p.rows[l * slice..l * slice + w * h * 4];
            texels.extend(b.chunks_exact(4).map(|c| if bgra { u32::from_le_bytes([c[2], c[1], c[0], c[3]]) } else { u32::from_le_bytes([c[0], c[1], c[2], c[3]]) }));
        }
        Some(Palette { width: w as u32, height: h as u32, layers: layers as u32, srgb, texels })
    }

    /// Row `row` of layer `layer` ([`PAL_TEXELS`] texels): v = (row + 0.5)
    /// / 256 as the program takes it [V 0x2a00-0x2a0c] at this palette's
    /// height, the layer clamped as a sampler clamps an array index.
    pub fn row(&self, row: u8, layer: u8) -> [u32; PAL_TEXELS] {
        let (w, h) = (self.width as usize, self.height as usize);
        let y = (((f32::from(row) + 0.5) / 256.0 * h as f32) as usize).min(h - 1);
        let l = usize::from(layer).min(self.layers as usize - 1);
        let at = (l * h + y) * w;
        std::array::from_fn(|i| self.texels[at + (i * w / PAL_TEXELS).min(w - 1)])
    }
}

/// The remap table and the vehicle's base from T9+0x100's bytes.
pub fn remap(b: &[u8]) -> Option<([u32; 256], u32)> {
    let b = b.get(..REMAP_BYTES)?;
    let w = |i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
    Some((std::array::from_fn(w), w(256)))
}

/// Byte offset in T8+0x540 of element 0's +16 for a material slot.
pub fn colour_offset(base: u32, slot: u32) -> u64 {
    (SLOT_ELEMENTS * (u64::from(base) + u64::from(slot))) * ELEMENT + 16
}

/// What a block holds: entries with a colour, entries from a palette row,
/// entries naming a palette row that got none (no palette, or more than
/// [`PAL_ROWS_MOST`] rows).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub coloured: u32,
    pub palette: u32,
    pub no_palette: u32,
}

/// The per-draw block: entry `a` (of 256) is element 0 of slot `remap[a]`:
/// its colour, or (`palette` given) its palette row's block `k`, those rows
/// following the entries ([`PAL_TEXELS`] dwords each, the block padded to
/// whole instances of 12). `read(offset)` returns the 8 bytes at a T8+0x540
/// byte offset (None: unreadable: entry 0). Each distinct slot is read once.
/// The per-draw block: entry `a` (of 256) is element 0 of slot `remap[a]`:
/// its colour, or (`palette` given) its palette row's block `k`, those rows
/// following the entries ([`PAL_TEXELS`] dwords each, the block padded to
/// whole instances of 12). `read(offset)` returns the 8 bytes at a T8+0x540
/// byte offset (None: unreadable: entry 0). Each distinct slot is read once.
/// CAR1B: `coat` (`carshade=on`) also reads element 0's +24 and +32 and
/// puts [`coat_bits`] into the entry's bits 27-31 (the kind stays bits 24-26).
pub fn table(remap: &[u32; 256], base: u32, palette: Option<&Palette>, mul: bool, coat: bool, read: impl FnMut(u64) -> Option<[u8; 8]>) -> (Vec<u32>, Counts) {
    let (t, n, _) = table_with(remap, base, palette, mul, coat, read, None);
    (t, n)
}

/// Makes table entries, the palette rows they name shared ([`table`]).
struct Entries<'a, R> {
    seen: std::collections::HashMap<u64, Option<Element>>,
    coats: std::collections::HashMap<u64, u32>,
    coat: bool,
    rows: Vec<(u8, u8)>,
    n: Counts,
    palette: Option<&'a Palette>,
    mul: bool,
    read: R,
}

impl<R: FnMut(u64) -> Option<[u8; 8]>> Entries<'_, R> {
    /// The entry of the element whose +16 is at T8+0x540 byte `off` (0: unreadable).
    fn at(&mut self, off: u64) -> u32 {
        let read = &mut self.read;
        let Some(e) = *self.seen.entry(off).or_insert_with(|| read(off).and_then(|b| Element::decode(&b))) else { return 0 };
        let c = if self.coat { *self.coats.entry(off).or_insert_with(|| read(off + 8).zip(read(off + 16)).map_or(0, |(p, q)| coat_bits(&p, &q))) } else { 0 };
        let entry = e.entry(self.mul) | c;
        if entry & KIND_MASK != 0 {
            self.n.coloured += 1;
            return entry;
        }
        let k = self.rows.iter().position(|r| *r == (e.row, e.layer)).or_else(|| (self.palette.is_some() && self.rows.len() < PAL_ROWS_MOST).then(|| {
            self.rows.push((e.row, e.layer));
            self.rows.len() - 1
        }));
        match (self.palette, k) {
            (Some(p), Some(k)) => {
                self.n.palette += 1;
                (if p.srgb { PALETTE_SRGB } else { PALETTE_UNORM }) | k as u32 | c
            }
            _ => {
                self.n.no_palette += 1;
                0
            }
        }
    }
}

/// [`table`], and (POL2, `livery` given) the livery section after the
/// palette rows: the block is 256 entries, the palette rows, the section
/// ([`Livery::section`]), padded to 12. Returns the section's dword index
/// in the block (None: no livery, or none built).
pub fn table_with(remap: &[u32; 256], base: u32, palette: Option<&Palette>, mul: bool, coat: bool, read: impl FnMut(u64) -> Option<[u8; 8]>, livery: Option<&mut Livery<'_>>) -> (Vec<u32>, Counts, Option<u32>) {
    let mut e = Entries { seen: std::collections::HashMap::new(), coats: std::collections::HashMap::new(), coat, rows: Vec::new(), n: Counts::default(), palette, mul, read };
    let mut out: Vec<u32> = remap.iter().map(|slot| e.at(colour_offset(base, *slot))).collect();
    // The livery slots' colours: element 0 of material `rec+0x20 + n`
    // (absolute: the PS's `- s85` and `+ s85` cancel [V 05483074 0x2694, 0x2e28]).
    let section = livery.and_then(|l| l.section(|m| e.at(colour_offset(0, m))));
    if let Some(p) = palette {
        for (row, layer) in &e.rows {
            out.extend_from_slice(&p.row(*row, *layer));
        }
    }
    let at = section.map(|s| {
        let at = out.len() as u32;
        out.extend_from_slice(&s);
        at
    });
    out.resize(out.len().max(LUT_DWORDS).div_ceil(12) * 12, 0);
    (out, e.n, at)
}

// POL2: the livery (PS_VehicleLiveryOn*, e.g. 05483074
// PS_VehicleLiveryOnDetailMapsOffDeferred_NoDeform with MS 42c87283).
// Per vertex [V MS 42c87283 0xd38-0xe6c]: a = u8(alpha) of mesh word 0x04,
// u = u8(255 x alpha) of mesh word 0x30's RGBA8 stream [V 0x32c]; attr1.x =
// remap[a] | u << 8 | B << 16 | A << 24, where A = L.byte0 - 1, B = L.byte1 - 1
// (0xFF: none) of L = T9+0x60 (root 1 slot 3) element u, dword +12, unless
// bit a of the 256-bit word at T8+0xA00 record +0x36c is set (then both
// 0xFF). Per pixel [V PS 05483074 0x2300-0x2b68]: record index = T8+0xA00
// record +0x3a8 + byte (that word 0: index 0); index 0 is no layer.

/// The draw's T8+0xA00 record: index `k = T8+0xB20[48 j + 36] >> 16`, `j =
/// T9+0x40 [0x14]`, at `k x 0x62c` [V PS 05483074 0x0b4-0x154 s29, 0x1d4 s83,
/// 0xe60 `s84 = s83 x 0x62c`; MS 42c87283 0x0a4-0x0bc, 0x158 s77, 0xd80 the
/// same index into T8+0xA00]. (Draw word 9, the s84 the PS loads at 0x2ec
/// first, is replaced before the livery: on pk8nort f1027 its +0x36c holds
/// floats [V].)
pub const LIVERY_J_OFF: u64 = 0x14;
pub const LIVERY_K_STRIDE: u64 = 48;
pub const LIVERY_K_OFF: u64 = 36;
pub const LIVERY_DRAW_STRIDE: u64 = 0x62c;
/// T8+0xA00 record: the 256-bit no-livery mask by material byte `a`
/// [V MS 0xd80 offset 876, `(a >> 3) & ~3`] and the livery record base
/// [V PS 0x2318 `s84 + 0x3a8`].
pub const LIVERY_MASK_OFF: u64 = 0x36c;
pub const LIVERY_BASE_OFF: u64 = 0x3a8;
/// T9+0x60 element +12: the layer bytes [V MS 0xddc, 0xe0c].
pub const LIVERY_LAYERS_OFF: u64 = 12;
/// Bytes of a T8+0x5c0 record read (+0 .. +0x3c) [V PS reads +0..+56].
pub const LIVERY_RECORD: usize = 64;
/// Distinct records a section holds at most (record 0 first; more: none).
pub const LIVERY_RECORDS_MOST: usize = 64;
/// A section: 8 mask dwords, 256 layer words (by u), the draw's UV
/// transform ([`LIVERY_UVX`]), then records of [`LIVERY_REC_DWORDS`]: 16
/// header dwords, 16 slot entries.
pub const LIVERY_HEAD: usize = 8 + 256 + 8;
pub const LIVERY_REC_DWORDS: usize = 32;
/// POL3: the section's UV transform: dwords +264..+269 the u row (a, b, c)
/// and the v row (d, e, f) as f32 bits (u' = a u + b v + c, v' = d u + e v
/// + f), +270 1 when set (0: identity). RAW_SHADER reads these offsets.
pub const LIVERY_UVX: usize = 8 + 256;

/// POL3: the mesh program's UV transform, applied to every UV set it
/// exports (UV0, mesh words 0x10 and 0x38: attr2.xy, attr2.zw, attr3.xy)
/// [V MS 42c87283 0xf98-0x1268, exports 0x1818/0x1820]: CB b4 (root 2
/// slot 0) +0xb4 = s44 (a float index), +0xb8 = s45, +0xbc = s46; s44 < 0:
/// identity [V 0xff4 `v_cmp_lt_f32 s44, 0`]; else the u row is the three
/// floats at T8+0x600 (root 0 slot 48) byte `r * 0x60 + 12 s44`, the v row
/// 48 bytes further, `r = T8+0xB20[48 j + 4] >> 16` (j = T9+0x40 [0x14], the
/// T8+0xA00 index's neighbour) [V 0x0b8 `s_or_b32 s6, s4, 4`, 0x164 s78];
/// u' = row.x u + row.y v + row.z s45, v' = vrow.x u + vrow.y v + vrow.z s46.
pub const LIVERY_UVX_CB_SLOT: u32 = 0;
pub const LIVERY_UVX_CB_ROOT: usize = 2;
pub const LIVERY_UVX_CB_OFF: u64 = 0xb4;
pub const LIVERY_UVX_ROW_OFF: u64 = 4;
pub const LIVERY_UVX_TABLE_SLOT: u32 = 0x600 / 32;
pub const LIVERY_UVX_STRIDE: u64 = 0x60;
pub const LIVERY_UVX_ENTRY: u64 = 12;
pub const LIVERY_UVX_V: u64 = 48;

/// The transform's entry from CB b4 +0xb4's bits (None: identity, s44 < 0
/// or not a number).
pub fn uvx_index(s44: u32) -> Option<u64> {
    let f = f32::from_bits(s44);
    (f >= 0.0).then(|| f as u64)
}

/// The transform from its two rows and s45, s46 (bits), as the section holds it.
pub fn uvx(u_row: [f32; 3], v_row: [f32; 3], s45: u32, s46: u32) -> [f32; 6] {
    [u_row[0], u_row[1], u_row[2] * f32::from_bits(s45), v_row[0], v_row[1], v_row[2] * f32::from_bits(s46)]
}

/// The transform applied to a UV (mirrors RAW_SHADER `lvX`).
pub fn uvx_apply(x: &[f32; 6], q: [f32; 2]) -> [f32; 2] {
    [x[0] * q[0] + x[1] * q[1] + x[2], x[3] * q[0] + x[4] * q[1] + x[5]]
}

/// POL4: whether a transform as read can be the title's: finite, and its
/// 2x2 part not collapsed (|a e - b d| > 1e-6). The body program samples
/// its albedo, normal and the rest at the same transformed UV0 / word 0x10
/// [V PS 05483074 0x200-0x28c: v[8:9] = v50/v44 or v29/v28, attr2 = MS
/// param2 = the transformed sets]: a collapsed transform would paint the
/// whole body one texel in the title, so one here is a misread (sinks take
/// it as identity and count it).
pub fn uvx_ok(x: &[f32; 6]) -> bool {
    x.iter().all(|v| v.is_finite()) && (x[0] * x[4] - x[1] * x[3]).abs() > 1e-6
}
/// Layer word bits: A's record (0-7), A present (8), B's record (16-23), B present (24).
pub const LV_A: u32 = 1 << 8;
pub const LV_B: u32 = 1 << 24;
/// RAW_SHADER layerFlags bit 9: the draw's vehmat block has a livery
/// section (layerTex: mask entries in bits 0-7, 8-15, the section's dword
/// index in the block in 16-31).
pub const LIVERY_FLAG: u32 = 0x200;
/// RAW_SHADER layerFlags bit 10 (`vehlivery=debug`): livery rows show
/// which step stops the livery (vmLivery's debug colours).
pub const LIVERY_DEBUG: u32 = 0x400;
/// A record's mask texture field: none.
pub const LV_NO_MASK: u32 = 0xFF;

/// The layer bytes (A, B) of a T9+0x60 +12 dword [V MS 0xde8
/// `(L << 24) + 0xff000000`, 0xe18-0xe2c `((L << 8) & 0xff0000) + 0xff0000`].
pub fn layer_bytes(l: u32) -> (u8, u8) {
    ((l as u8).wrapping_sub(1), ((l >> 8) as u8).wrapping_sub(1))
}

/// What a livery section reads, per draw.
pub struct Livery<'a> {
    /// The T8+0xA00 record's +0x3a8 (0: no records).
    pub base: u32,
    /// The no-livery mask (bit a).
    pub off: [u32; 8],
    /// T9+0x60 element u's +12 dword, u = 0.. (missing: 0, both bytes none).
    pub layers: &'a [u32],
    /// A T8+0x5c0 record's bytes by index (None: unreadable).
    pub record: &'a mut dyn FnMut(u32) -> Option<[u8; LIVERY_RECORD]>,
    /// A mask texture's heap index -> its field (0 or 1: the draw's
    /// [`crate::Material::livery`] entry; None: more than two masks).
    pub mask: &'a mut dyn FnMut(u32) -> Option<u32>,
    /// POL3: the draw's UV transform ([`uvx`]; None: identity).
    pub uvx: Option<[f32; 6]>,
    pub n: LiveryCounts,
}

/// What a section holds: records, layer words with A, with B, masks the
/// two fields could not take, unreadable records.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveryCounts {
    pub records: u32,
    pub with_a: u32,
    pub with_b: u32,
    pub masks_lost: u32,
    pub unreadable: u32,
}

impl Livery<'_> {
    /// The section (None: no base, or no layer word names a record).
    /// `entry(m)`: the table entry of absolute material `m`.
    pub fn section(&mut self, mut entry: impl FnMut(u32) -> u32) -> Option<Vec<u32>> {
        if self.base == 0 {
            return None;
        }
        let mut recs: Vec<u32> = vec![0];
        let mut out = vec![0u32; LIVERY_HEAD];
        out[..8].copy_from_slice(&self.off);
        if let Some(x) = self.uvx {
            for (k, v) in x.iter().enumerate() {
                out[LIVERY_UVX + k] = v.to_bits();
            }
            out[LIVERY_UVX + 6] = 1;
        }
        let idx_of = |recs: &mut Vec<u32>, byte: u8| -> Option<u32> {
            (byte != 0xFF).then_some(())?;
            let i = self.base.wrapping_add(u32::from(byte));
            (i != 0).then_some(())?;
            let k = recs.iter().position(|r| *r == i).or_else(|| (recs.len() < LIVERY_RECORDS_MOST).then(|| {
                recs.push(i);
                recs.len() - 1
            }))?;
            Some(k as u32)
        };
        for u in 0..256 {
            let (a, b) = layer_bytes(self.layers.get(u).copied().unwrap_or(0));
            let mut w = 0;
            if let Some(k) = idx_of(&mut recs, a) {
                w |= k | LV_A;
                self.n.with_a += 1;
            }
            if let Some(k) = idx_of(&mut recs, b) {
                w |= k << 16 | LV_B;
                self.n.with_b += 1;
            }
            out[8 + u] = w;
        }
        if self.n.with_a + self.n.with_b == 0 {
            return None;
        }
        for &i in &recs {
            let mut h = [0u32; LIVERY_REC_DWORDS];
            match (self.record)(i) {
                Some(b) => {
                    let d = |k: usize| u32::from_le_bytes([b[4 * k], b[4 * k + 1], b[4 * k + 2], b[4 * k + 3]]);
                    // +0 .. +36 as read; +40 / +44 the A / B mask's heap index
                    // [V 0x2890, 0x2580] as its field; +48 / +52 / +56 the X / A / B
                    // UV scale [V 0x24ec, 0x28cc, 0x25d0].
                    for k in 0..10 {
                        h[k] = d(k);
                    }
                    for (k, f) in [(10, 10), (11, 11)] {
                        h[k] = match d(f) {
                            0 => LV_NO_MASK,
                            t => (self.mask)(t).unwrap_or_else(|| {
                                self.n.masks_lost += 1;
                                LV_NO_MASK
                            }),
                        };
                    }
                    h[12] = d(12);
                    h[13] = d(13);
                    h[14] = d(14);
                    // The slot colours: material +0x20 + n, n = 0..15 (n = int(16 b)).
                    if i != 0 {
                        for n in 0..16 {
                            h[16 + n] = entry(h[8].wrapping_add(n as u32));
                        }
                    }
                }
                None => {
                    self.n.unreadable += 1;
                    (h[10], h[11]) = (LV_NO_MASK, LV_NO_MASK);
                }
            }
            out.extend_from_slice(&h);
        }
        self.n.records = recs.len() as u32;
        Some(out)
    }
}

/// POL4: layerFlags bit 11 (`vehlivery=mask`, with [`LIVERY_DEBUG`]):
/// inside A's rect the livery mask's raw sample (r g b as read) in place
/// of the debug colours; bit 12 (`vehlivery=uv`): the mask's UV (set x
/// scale) as frac(u), frac(v), 0.25. Outside the rect: the debug colours.
pub const LIVERY_SHOW_MASK: u32 = 0x800;
pub const LIVERY_SHOW_UV: u32 = 0x1000;

/// `XNW_WORLD_RENDER=...,vehlivery=on` (POL2; default off); `vehlivery=debug`,
/// `=mask`, `=uv` are on too, with their views ([`parse_livery_view`]).
pub fn parse_livery(v: &str) -> bool {
    v.split(',').any(|p| matches!(p.trim(), "vehlivery=on" | "vehlivery=debug" | "vehlivery=mask" | "vehlivery=uv"))
}

/// The layerFlags view bits of `vehlivery=debug` (POL3: the step colours),
/// `=mask`, `=uv` (POL4); 0: none.
pub fn parse_livery_view(v: &str) -> u32 {
    v.split(',').fold(0, |a, p| {
        a | match p.trim() {
            "vehlivery=debug" => LIVERY_DEBUG,
            "vehlivery=mask" => LIVERY_DEBUG | LIVERY_SHOW_MASK,
            "vehlivery=uv" => LIVERY_DEBUG | LIVERY_SHOW_UV,
            _ => 0,
        }
    })
}

/// `vehlivery=debug` (POL3): the livery rows' debug colours.
pub fn parse_livery_debug(v: &str) -> bool {
    parse_livery_view(v) != 0
}

/// [`parse_livery_view`] of `XNW_WORLD_RENDER` (read once).
pub fn livery_view() -> u32 {
    static ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XNW_WORLD_RENDER").map_or(0, |v| parse_livery_view(&v)))
}

/// The shader's livery weights (mirrors RAW_SHADER `vmLivery`, for the
/// tests): curve 1 `saturate((r - 0.52) x 2.087)` [V 0x268c, 0x26a4], curve
/// 2 `saturate((r - 0.1) x 2.5)` [V 0x26e8, 0x26f8]; the slot n = int(16 b)
/// [V 0x2684, 0x26a0].
pub fn mask_decode(r: f32, b: f32) -> (u32, f32, f32) {
    let n = ((b * 16.0) as u32).min(15);
    (n, ((r + f32::from_bits(0xbf05_1eb8)) * f32::from_bits(0x4005_6042)).clamp(0.0, 1.0), ((r + f32::from_bits(0xbdcc_cccd)) * 2.5).clamp(0.0, 1.0))
}

/// Layer B's weight, and the two terms it leaves for layer A
/// (v59, v60) [V 0x26b4-0x2798]: `rec` the B record's header dwords,
/// `x_tex` whether the A record's +36 texture exists.
pub fn weight_b(rec: &[u32], n: u32, c1: f32, c2: f32, x_tex: bool) -> (f32, f32, f32) {
    let bit = |w: u32, k: u32| (w >> (k & 31)) & 1 != 0;
    let v100 = if x_tex && bit(rec[4], n) { c1 } else { 0.0 };
    let w = if bit(rec[5], n) { c2 } else if bit(rec[7], 16 + n) { c1 } else { v100 };
    (w, if bit(rec[6], 16 + n) { c2 } else { 0.0 }, if bit(rec[6], n) { c2 } else { 0.0 })
}

/// Layer A's weight [V 0x29a0-0x2a1c]: from its record, B's (v59, v60).
pub fn weight_a(rec: &[u32], n: u32, c1: f32, c2: f32, x_tex: bool, v59: f32, v60: f32) -> f32 {
    let bit = |w: u32, k: u32| (w >> (k & 31)) & 1 != 0;
    let v50 = if x_tex && bit(rec[4], n) { c1 - v60 } else { 0.0 };
    (if bit(rec[5], n) { c2 - v59 } else { v50 }).clamp(0.0, 1.0)
}

/// The two kept of base, B, A: (weight, which: 0 base, 1 B, 2 A) for the
/// primary and the secondary [V 0x2b1c-0x2b60]: A's weight is taken from
/// what B left the base; A replaces the secondary when the base leads, else
/// the primary when it beats it.
pub fn keep_two(wb: f32, wa: f32) -> [(f32, u8); 2] {
    let (w1, w2) = (1.0 - wb, wb);
    let (a, w1) = (wa * w1, w1 * (1.0 - wa));
    if w1 > w2 {
        [(w1, 0), if a > w2 { (a, 2) } else { (w2, 1) }]
    } else {
        [if a > w1 { (a, 2) } else { (w1, 0) }, (w2, 1)]
    }
}

/// `XNW_WORLD_RENDER=...,vehmat=off` (default on).
pub fn parse_switch(v: &str) -> bool {
    !v.split(',').any(|p| p.trim() == "vehmat=off")
}

// POL5: the deciding values for a livery whose layer paints its whole rect
// One mask 4x4 BC7 with one mip, the other the 2048x2048 lettering;
// the port's layer loop, weights and keep-two re-read against PS 05483074
// 0x2300-0x2b60 and the two-candidate use after it, 0x2dc8-0x2f88 and
// 0x3b68-0x3d48: faithful [V]). A mask of one colour gives one slot and one
// weight over the whole rect, so whether the title paints that box depends
// only on its texels and the record bits: these print both.

/// BC7 block `b` (16 bytes) as 16 RGBA8 texels, row-major (BC7 spec,
/// D3D11.3 19.5.2): modes 4, 5, 6 (one subset; the modes an encoder takes for
/// a flat block) and the reserved mode (byte 0 = 0: all zero, given as mode
/// 8); the partitioned modes 0-3 and 7: None. Returns (mode, texels).
pub fn bc7_texels(b: &[u8; 16]) -> (u32, Option<[[u8; 4]; 16]>) {
    let bits = u128::from_le_bytes(*b);
    let mut at = 0u32;
    let mut take = |n: u32| {
        let v = ((bits >> at) & ((1u128 << n) - 1)) as u32;
        at += n;
        v
    };
    if b[0] == 0 {
        return (8, Some([[0; 4]; 16]));
    }
    let mode = b[0].trailing_zeros();
    take(mode + 1);
    const W2: [u32; 4] = [0, 21, 43, 64];
    const W3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
    const W4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
    let lerp = |e0: u32, e1: u32, w: u32| ((64 - w) * e0 + w * e1 + 32) >> 6;
    let expand = |v: u32, n: u32| (v << (8 - n)) | (v >> (2 * n - 8));
    // Index runs: `n` bits each, texel 0 (the anchor) one bit less.
    let indices = |take: &mut dyn FnMut(u32) -> u32, n: u32| -> [u32; 16] { std::array::from_fn(|i| take(if i == 0 { n - 1 } else { n })) };
    let (rot, isel) = match mode {
        4 => (take(2), take(1)),
        5 => (take(2), 0),
        6 => (0, 0),
        _ => return (mode, None),
    };
    let (cb, ab) = match mode {
        4 => (5, 6),
        5 => (7, 8),
        _ => (7, 7),
    };
    let mut e = [[0u32; 2]; 4];
    for c in e.iter_mut().take(3) {
        *c = [take(cb), take(cb)];
    }
    e[3] = [take(ab), take(ab)];
    let (c, a) = if mode == 6 {
        let p = [take(1), take(1)];
        for ch in &mut e {
            *ch = [ch[0] << 1 | p[0], ch[1] << 1 | p[1]];
        }
        let i = indices(&mut take, 4).map(|k| W4[k as usize]);
        (i, i)
    } else {
        for ch in e.iter_mut().take(3) {
            *ch = [expand(ch[0], cb), expand(ch[1], cb)];
        }
        e[3] = [expand(e[3][0], ab), expand(e[3][1], ab)];
        let i2 = indices(&mut take, 2).map(|k| W2[k as usize]);
        let other = if mode == 4 { indices(&mut take, 3).map(|k| W3[k as usize]) } else { indices(&mut take, 2).map(|k| W2[k as usize]) };
        if mode == 4 && isel == 1 { (other, i2) } else { (i2, other) }
    };
    let t = std::array::from_fn(|i| {
        let mut p = [0u8; 4];
        for (k, ch) in e.iter().enumerate() {
            p[k] = lerp(ch[0], ch[1], if k == 3 { a[i] } else { c[i] }) as u8;
        }
        match rot {
            1 => p.swap(3, 0),
            2 => p.swap(3, 1),
            3 => p.swap(3, 2),
            _ => {}
        }
        p
    });
    (mode, Some(t))
}

/// What one mask texel gives a layer alone (no other layer, no X texture):
/// the slot n and the weight as B ([`weight_b`]) and as A ([`weight_a`]).
pub fn texel_weights(rec: &[u32], r: f32, b: f32) -> (u32, f32, f32) {
    let (n, c1, c2) = mask_decode(r, b);
    (n, weight_b(rec, n, c1, c2, false).0, weight_a(rec, n, c1, c2, false, 0.0, 0.0))
}

/// A section record's 16 header dwords (None: past the section).
fn section_record(s: &[u32], k: usize) -> Option<&[u32]> {
    s.get(LIVERY_HEAD + LIVERY_REC_DWORDS * k..LIVERY_HEAD + LIVERY_REC_DWORDS * k + 16)
}

/// The records of a section ([`Livery::section`]; `records` of them) as the
/// frame line's text: per record its index in the section, the layer words
/// that use it as A and as B, +0's mask / rect UV sets, the f16 rect, the
/// +16 .. +28 bit masks, the slot base, the X texture, the A / B mask
/// fields (0 / 1: `mask k` of the line, 255 none), the X / A / B scales.
pub fn section_text(s: &[u32], records: u32) -> String {
    let lo = |w: u32| crate::hair::f16(w as u16);
    let hi = |w: u32| crate::hair::f16((w >> 16) as u16);
    let layers = s.get(8..8 + 256).unwrap_or(&[]);
    (0..records as usize)
        .filter_map(|k| {
            let r = section_record(s, k)?;
            let uses = |bit: u32, shift: u32| layers.iter().filter(|w| *w & bit != 0 && (*w >> shift) & 0xFF == k as u32).count();
            Some(format!(
                "rec {k} (A {}, B {}): sets mask {} rect {}; rect u' = {} u + {} v + {}, v' = {} u + {} v + {}; bits {:08x} {:08x} {:08x} {:08x}; slots {}; X {}; mask A {} B {}; scale {} {} {}",
                uses(LV_A, 0),
                uses(LV_B, 16),
                r[0] & 0xFF,
                (r[0] >> 8) & 0xFF,
                lo(r[2]),
                hi(r[2]),
                lo(r[1]),
                lo(r[3]),
                hi(r[3]),
                hi(r[1]),
                r[4],
                r[5],
                r[6],
                r[7],
                r[8],
                r[9],
                r[10],
                r[11],
                f32::from_bits(r[12]),
                f32::from_bits(r[13]),
                f32::from_bits(r[14])
            ))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// A mask's bytes in title memory as text: 16-byte blocks, those not zero,
/// the first of them (offset, hex) and, for BC7 (`bc7`), its mode, the r
/// and b ranges of its texels and what its lowest r and b give each record
/// of the section whose A or B mask is this one (`field`: 0 / 1).
pub fn mask_bytes_text(bytes: &[u8], bc7: bool, section: &[u32], records: u32, field: u32) -> String {
    let nz: Vec<(usize, [u8; 16])> = bytes.chunks_exact(16).enumerate().filter_map(|(i, c)| <[u8; 16]>::try_from(c).ok().filter(|b| b.iter().any(|x| *x != 0)).map(|b| (16 * i, b))).collect();
    let mut t = format!("bytes {} blocks {} not zero {}", bytes.len(), bytes.len() / 16, nz.len());
    let Some((off, b)) = nz.first() else { return t };
    t += &format!(" first at {off}: {}", b.iter().map(|x| format!("{x:02x}")).collect::<String>());
    if !bc7 {
        return t;
    }
    let (mode, texels) = bc7_texels(b);
    let Some(px) = texels else { return t + &format!(" (BC7 mode {mode}, not decoded)") };
    let range = |k: usize| (px.iter().map(|p| p[k]).min().unwrap_or(0), px.iter().map(|p| p[k]).max().unwrap_or(0));
    let ((r0, r1), (b0, b1)) = (range(0), range(2));
    t += &format!(" (BC7 mode {mode}: r {r0}-{r1} b {b0}-{b1}");
    let (r, bl) = (f32::from(r0) / 255.0, f32::from(b0) / 255.0);
    for k in 0..records as usize {
        let Some(rec) = section_record(section, k) else { continue };
        let (n, wb, wa) = texel_weights(rec, r, bl);
        if rec[10] == field {
            t += &format!("; rec {k} as A: slot {n} weight {wa:.2}");
        }
        if rec[11] == field {
            t += &format!("; rec {k} as B: slot {n} weight {wb:.2}");
        }
    }
    t + ")"
}

/// `vehmat=mul`: the record colour times the texture (INT1/INT2), not
/// instead of it (palette rows replace it either way).
pub fn parse_mul(v: &str) -> bool {
    v.split(',').any(|p| p.trim() == "vehmat=mul")
}

/// NPC1: the far peds (`ped_slod_tint`) take their colour per vertex from a
/// per-ped block of the frame table's T8+0x60 bytes, as mesh program
/// MSNC_VS_TransformSkinD_dc61c767 (132ea7e6) makes attr0, which pixel
/// program d8337d46 stores as the albedo (RT0 = attr0.xyz, no image) [V
/// disasm 132ea7e6 0x86c-0x9f8, d8337d46 exp mrt0]:
/// - s21 = root 4 dword 0; block = (s21 << 9) & 0x1FFFE00 bytes;
/// - a = the vertex stream at mesh word 0x04 (RGBA8, format 6: the MS header
///   load s[4:7] at +0, s5 its stream); s21 bit 16 set: k = round(255 a.x)
///   (the x byte); clear: k = r(21.25 x) + (21.25 y >= 0.5: r(21.25 y) + 21)
///   + (21.25 z >= 0.5: r(21.25 z) + 42), r rounding to nearest even;
/// - e = byte[block + 0x100 + k]; colour = bytes[block + 4e + 3, + 2, + 1] as
///   R, G, B, sRGB-decoded (the 0.04045 / 12.92 / 2.4 branch).
/// pk8a f1027 rows 527-529 (s21 0x10035, block 53): every vertex's k lands on
/// entries 3/13, 4/16 and 5/6/8/16, none on entry 0 (black) [V refs.bin,
/// out\npc1]. The pass gets `lut[k]` ([`SLOD_SRGB`] | RGB8) as the draw's
/// vehmat block; the vertex program picks k ([`SLOD_FLAG`], [`SLOD_RGB`]),
/// the pixel program decodes.
pub const SLOD_SRGB: u32 = 5 << 24;
/// layerFlags bit 11: a slod row (its vehmat block is [`slod_lut`]'s, k as
/// above); bit 12: s21 bit 16 clear (the three-channel k).
pub const SLOD_FLAG: u32 = 0x800;
pub const SLOD_RGB: u32 = 0x1000;
/// A block's bytes, and its index table's offset in it.
pub const SLOD_BLOCK: usize = 512;
pub const SLOD_TABLE: usize = 0x100;

/// The block's byte offset in T8+0x60 for root 4 dword 0 `s21`.
pub fn slod_block_at(s21: u32) -> u64 {
    u64::from((s21 << 9) & 0x01FF_FE00)
}

/// The row's slod mode for `s21` (1: k = the x byte; 2: the three-channel k).
pub fn slod_mode(s21: u32) -> u8 {
    if s21 & 0x1_0000 != 0 { 1 } else { 2 }
}

/// The 256-entry table the vertex program reads at k (padded to
/// [`LUT_DWORDS`]): [`SLOD_SRGB`] | R | G << 8 | B << 16 of entry
/// `table[k]`; an entry past the block reads 0 (black, as entry 0's zeros).
/// `bytes` from the block's start; None when shorter than its table.
pub fn slod_lut(bytes: &[u8]) -> Option<Vec<u32>> {
    let table = bytes.get(SLOD_TABLE..SLOD_TABLE + 256)?;
    let mut out = vec![0u32; LUT_DWORDS];
    for (k, e) in table.iter().enumerate() {
        let c = usize::from(*e) * 4;
        let (r, g, b) = bytes.get(c..c + 4).map_or((0, 0, 0), |x| (x[3], x[2], x[1]));
        out[k] = SLOD_SRGB | u32::from(r) | u32::from(g) << 8 | u32::from(b) << 16;
    }
    Some(out)
}

/// The title's k of an RGBA8 vertex word (the vertex program's `slodK`;
/// `rgb`: s21 bit 16 clear).
pub fn slod_k(word: u32, rgb: bool) -> usize {
    if !rgb {
        return (word & 0xFF) as usize;
    }
    let q = |s: u32| ((word >> s) & 0xFF) as f32 * (21.25 / 255.0);
    let (x, y, z) = (q(0), q(8), q(16));
    let mut k = x.round_ties_even() as usize;
    if y >= 0.5 {
        k += y.round_ties_even() as usize + 21;
    }
    if z >= 0.5 {
        k += z.round_ties_even() as usize + 42;
    }
    k.min(255)
}

/// The title's sRGB decode of a colour byte (the mesh program's; the pixel
/// program's `slodDecode`).
pub fn slod_decode(b: u8) -> f32 {
    let c = f32::from(b) / 255.0;
    if c > 0.04045 { ((c + 0.055) / 1.055).powf(2.4) } else { c / 12.92 }
}

/// `XNW_WORLD_RENDER=...,slodtint=off` (default on).
pub fn parse_slod(v: &str) -> bool {
    !v.split(',').any(|p| p.trim() == "slodtint=off")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NPC1: pk8a f1027 block 53's entries and index runs (the values of
    /// refs.bin's T8+0x60 bytes at 0x6A00, sinks 112/119/124), and vertex
    /// x bytes of its three sections.
    #[test]
    fn slod_lookup_of_the_captured_block() {
        let mut b = vec![0u8; SLOD_BLOCK];
        for (e, c) in [(3usize, 0x005c_70a2u32), (4, 0x0063_79a8), (5, 0x0064_76a2), (6, 0x0045_4545), (8, 0x003b_3e6a), (13, 0x00e5_e5e5), (16, 0x0080_80dd)] {
            b[4 * e..4 * e + 4].copy_from_slice(&c.to_be_bytes());
        }
        for (lo, hi, e) in [(64, 68, 3u8), (69, 69, 13), (70, 71, 3), (72, 77, 13), (78, 81, 3), (128, 128, 4), (129, 130, 16), (171, 180, 4), (203, 203, 6), (204, 212, 5), (217, 217, 8)] {
            for k in lo..=hi {
                b[SLOD_TABLE + k] = e;
            }
        }
        assert_eq!((slod_block_at(0x0001_0035), slod_mode(0x0001_0035), slod_mode(0x35)), (53 * 512, 1, 2));
        let t = slod_lut(&b).unwrap();
        assert_eq!(t.len(), LUT_DWORDS);
        // Bytes 3, 2, 1 are R, G, B: entry 3 (bytes 00 5c 70 a2) is (162, 112, 92).
        let rgb = |w: u32| (w & KIND_MASK, w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF);
        assert_eq!(rgb(t[64]), (SLOD_SRGB, 162, 112, 92));
        assert_eq!(rgb(t[69]), (SLOD_SRGB, 229, 229, 229));
        assert_eq!(rgb(t[128]), (SLOD_SRGB, 168, 121, 99));
        assert_eq!(rgb(t[129]), (SLOD_SRGB, 221, 128, 128));
        assert_eq!(rgb(t[204]), (SLOD_SRGB, 162, 118, 100));
        assert_eq!(rgb(t[203]), (SLOD_SRGB, 69, 69, 69));
        assert_eq!(rgb(t[217]), (SLOD_SRGB, 106, 62, 59));
        assert_eq!(rgb(t[0]), (SLOD_SRGB, 0, 0, 0), "entry 0: black, as the title's");
        // Mode 1: k is the x byte (sink 124's 204 -> entry 5).
        assert_eq!(slod_k(0xFFFF_00CC, false), 204);
        // Mode 2: x, y, z quantised by 21.25 / 255 (12 -> 1, 24 -> 2 + 21, 0 -> none).
        assert_eq!(slod_k(0x0000_180C, true), 1 + 2 + 21);
        assert_eq!(slod_k(0x00FF_FFFF, true), 21 + 21 + 21 + 21 + 42);
        // The decode: 0 -> 0, 255 -> 1, 162 -> 0.3613 (the title's branch above 0.04045).
        assert_eq!((slod_decode(0), slod_decode(255)), (0.0, 1.0));
        assert!((slod_decode(162) - 0.3613).abs() < 1e-3 && (slod_decode(10) - 10.0 / 255.0 / 12.92).abs() < 1e-7);
        // A block too short for its table: no lookup.
        assert!(slod_lut(&b[..0x1FF]).is_none());
        assert!(parse_slod("screen,vehmat=on") && !parse_slod("a, slodtint=off"));
    }

    /// pk8a f1032 sink 597 (row 717, the steering wheel), T8+0x540 at
    /// 0x986bedf00: element 0 of slots 37 (+16 080808ff, +20 ffff0000) and
    /// 14 (+16 000000ff, +20 f2000000), and the body's slot 0 (220202ff).
    #[test]
    fn element_decode_from_the_capture() {
        let e = Element::decode(&[0x08, 0x08, 0x08, 0xFF, 0xFF, 0xFF, 0, 0]).unwrap();
        assert_eq!(e, Element { rgb: [8, 8, 8], row: 0xFF, layer: 0xFF });
        assert_eq!(e.entry(false), HAS_COLOUR | 0x08_0808);
        assert_eq!(e.entry(true), COLOUR_MUL | 0x08_0808);
        let p = Element::decode(&[0, 0, 0, 0xFF, 0xF2, 0, 0, 0]).unwrap();
        assert_eq!((p.row, p.layer, p.entry(false)), (0xF2, 0, 0), "a palette row: no colour");
        assert_eq!(Element::decode(&[0x22, 0x02, 0x02, 0xFF, 0xFF, 0xFF, 0, 0]).unwrap().entry(false), HAS_COLOUR | 0x02_0222);
        assert_eq!(Element::decode(&[0; 7]), None);
    }

    /// CAR1B: pk8a f1029 base 23909: the body's paint slot 0 (+24
    /// 0000001e.., +32 220202) is coat with roughness nibble 1; a trim slot
    /// (+24 00004cb2, +32 0) is not; the table puts the bits above the kind.
    #[test]
    fn coat_bits_from_the_capture() {
        assert_eq!(coat_bits(&[0, 0, 0, 0x1e, 0, 0xbf, 0, 0], &[0x22, 2, 2, 0, 0x97, 6, 0x84, 0x7f]), COAT | 1 << COAT_ROUGH_SHIFT);
        assert_eq!(coat_bits(&[0, 0, 0x4c, 0xb2, 0, 0xbf, 0x22, 0x22], &[0; 8]), 0);
        let mut b = vec![0u8; REMAP_BYTES];
        b[4 * 256..4 * 256 + 4].copy_from_slice(&23909u32.to_le_bytes());
        b[4..8].copy_from_slice(&2u32.to_le_bytes());
        let (m, base) = remap(&b).unwrap();
        let rec = |off: u64| match off {
            o if o == colour_offset(base, 0) => Some([0x22, 2, 2, 0xFF, 0xFF, 0xFF, 0, 0]),
            o if o == colour_offset(base, 0) + 8 => Some([0, 0, 0, 0x1e, 0, 0xbf, 0, 0]),
            o if o == colour_offset(base, 0) + 16 => Some([0x22, 2, 2, 0, 0, 0, 0, 0]),
            o if o == colour_offset(base, 2) => Some([1, 1, 1, 0xFF, 0xFF, 0xFF, 0, 0]),
            o if o == colour_offset(base, 2) + 8 => Some([0, 0, 0, 0x1e, 0, 0xbf, 0, 0]),
            o if o == colour_offset(base, 2) + 16 => Some([0; 8]),
            _ => None,
        };
        let (t, _) = table(&m, base, None, false, true, rec);
        assert_eq!((t[0], t[1]), (HAS_COLOUR | 0x02_0222 | COAT | 1 << COAT_ROUGH_SHIFT, HAS_COLOUR | 0x01_0101));
        assert_eq!(table(&m, base, None, false, false, rec).0[0], HAS_COLOUR | 0x02_0222, "carshade off: no coat bits");
    }

    /// A 4 x 256 x 2 palette whose texel is (x, row, layer, 0xFF).
    fn test_palette(format: u32) -> crate::textures::Provided {
        let (w, h, l) = (4u32, 256u32, 2u32);
        let mut rows = Vec::new();
        for z in 0..l {
            for y in 0..h {
                for x in 0..w {
                    rows.extend_from_slice(&[x as u8, y as u8, z as u8, 0xFF]);
                }
            }
        }
        crate::textures::Provided { width: w, height: h, format, rows, row_bytes: 4 * w, row_count: h * l, dim: crate::textures::TEX_2D_ARRAY, depth: l, mips: 1 }
    }

    /// The record index 3 * (base + slot) at 76 bytes (pk8a: base 23909,
    /// remap[42] = 37), the block reading each slot once, palette rows
    /// after the entries (PAL1).
    #[test]
    fn table_from_remap_and_records() {
        assert_eq!(colour_offset(23909, 37), 3 * (23909 + 37) * 76 + 16);
        let mut b = vec![0u8; REMAP_BYTES];
        let mut put = |i: usize, v: u32| b[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        put(42, 37);
        put(18, 14);
        put(19, 15);
        put(256, 23909);
        let (m, base) = remap(&b).unwrap();
        assert_eq!((m[42], m[18], m[0], base), (37, 14, 0, 23909));
        assert_eq!(remap(&b[..0x400]), None);
        let rec = |off: u64| match off {
            o if o == colour_offset(base, 37) => Some([8, 8, 8, 0xFF, 0xFF, 0xFF, 0, 0]),
            o if o == colour_offset(base, 14) => Some([0, 0, 0, 0xFF, 0xF2, 0, 0, 0]),
            o if o == colour_offset(base, 15) => Some([0, 0, 0, 0xFF, 0x76, 1, 0, 0]),
            o if o == colour_offset(base, 0) => Some([0x22, 2, 2, 0xFF, 0xFF, 0xFF, 0, 0]),
            _ => None,
        };
        // No palette: palette rows keep the texture (INT1/INT2's table).
        let mut reads = 0;
        let (t, n) = table(&m, base, None, false, false, |off| {
            reads += 1;
            rec(off)
        });
        assert_eq!(reads, 4, "slots 0, 14, 15, 37 once each");
        assert_eq!(t.len(), LUT_DWORDS);
        assert_eq!((t[42], t[18], t[19], t[0], t[255]), (HAS_COLOUR | 0x08_0808, 0, 0, HAS_COLOUR | 0x02_0222, HAS_COLOUR | 0x02_0222));
        assert_eq!(n, Counts { coloured: 254, palette: 0, no_palette: 2 });
        // With the palette: rows 0xF2 (layer 0) and 0x76 (layer 1) follow the entries.
        let p = Palette::from_provided(&test_palette(29)).unwrap();
        assert_eq!((p.width, p.height, p.layers, p.srgb, p.texels.len()), (4, 256, 2, true, 4 * 256 * 2));
        let (t, n) = table(&m, base, Some(&p), true, false, rec);
        assert_eq!(n, Counts { coloured: 254, palette: 2, no_palette: 0 });
        assert_eq!((t[18], t[19], t[42]), (PALETTE_SRGB, PALETTE_SRGB | 1, COLOUR_MUL | 0x08_0808));
        assert_eq!(t.len(), (256 + 2 * PAL_TEXELS).div_ceil(12) * 12);
        assert_eq!(t[256], 0xFF00_F200, "row 0xF2 layer 0, x 0");
        assert_eq!(t[256 + PAL_TEXELS - 1], 0xFF00_F203, "x = 127 * 4 / 128 = 3");
        assert_eq!(t[256 + PAL_TEXELS], 0xFF01_7600, "row 0x76 layer 1");
        // BGRA swaps to RGBA; a layer past the last clamps; other formats give none.
        let q = Palette::from_provided(&test_palette(87)).unwrap();
        assert_eq!((q.srgb, q.row(5, 0)[64]), (false, 0xFF02_0500), "B=x G=row R=layer: x = 64 * 4 / 128 = 2");
        assert_eq!(p.row(5, 9)[0], 0xFF01_0500);
        assert_eq!(Palette::from_provided(&test_palette(71)), None);
        assert!(parse_switch("screen,paint=off") && !parse_switch("screen, vehmat=off"));
        assert!(parse_mul("screen,vehmat=mul") && !parse_mul("screen"));
    }

    /// POL2: layer bytes as the mesh program makes them, the section's
    /// layer words and records, the slot colours read at material +0x20 + n.
    #[test]
    fn livery_section_from_records() {
        assert_eq!(layer_bytes(0), (0xFF, 0xFF), "0: none, both");
        assert_eq!(layer_bytes(0x0000_0302), (1, 2));
        assert_eq!(layer_bytes(0xAB00_0001), (0, 0xFF), "upper bytes unread");
        // u = 5: A = 1, B none; u = 7: A none, B = 3; u = 9: both = 1 (one record).
        let mut layers = vec![0u32; 256];
        layers[5] = 0x02;
        layers[7] = 0x0400;
        layers[9] = 0x0202;
        let rec = |i: u32| -> Option<[u8; LIVERY_RECORD]> {
            let mut b = [0u8; LIVERY_RECORD];
            let mut put = |o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
            match i {
                0 => {}
                101 => {
                    put(0, 0x0201);
                    put(16, 0x5);
                    put(32, 900);
                    put(40, 7000);
                    put(44, 7001);
                    put(56, 4.0f32.to_bits());
                }
                103 => {
                    put(32, 950);
                    put(44, 7002);
                }
                _ => return None,
            }
            Some(b)
        };
        let mut rec = rec;
        let mut masks: Vec<u32> = Vec::new();
        let mut mask = |t: u32| masks.iter().position(|m| *m == t).or_else(|| (masks.len() < 2).then(|| {
            masks.push(t);
            masks.len() - 1
        })).map(|k| k as u32);
        let mut l = Livery { base: 100, off: [0x20, 0, 0, 0, 0, 0, 0, 1 << 31], layers: &layers, record: &mut rec, mask: &mut mask, uvx: None, n: LiveryCounts::default() };
        let s = l.section(|m| 0x0100_0000 | m as u32).unwrap();
        assert_eq!(l.n, LiveryCounts { records: 3, with_a: 2, with_b: 2, masks_lost: 1, unreadable: 0 });
        assert_eq!(s.len(), LIVERY_HEAD + 3 * LIVERY_REC_DWORDS);
        assert_eq!((s[0], s[7]), (0x20, 1 << 31), "the mask by a");
        assert_eq!((s[8 + 5], s[8 + 7], s[8 + 9], s[8]), (1 | LV_A, 2 << 16 | LV_B, 1 | LV_A | 1 << 16 | LV_B, 0));
        let r = |k: usize| &s[LIVERY_HEAD + k * LIVERY_REC_DWORDS..][..LIVERY_REC_DWORDS];
        assert_eq!((r(0)[10], r(0)[11], r(0)[16]), (LV_NO_MASK, LV_NO_MASK, 0), "record 0: no masks, no colours");
        assert_eq!((r(1)[0], r(1)[4], r(1)[8], r(1)[10], r(1)[11], f32::from_bits(r(1)[14])), (0x0201, 5, 900, 0, 1, 4.0));
        assert_eq!((r(1)[16], r(1)[31]), (0x0100_0000 | 900, 0x0100_0000 | 915));
        assert_eq!((r(2)[10], r(2)[11], r(2)[16]), (LV_NO_MASK, LV_NO_MASK, 0x0100_0000 | 950), "a third mask: lost");
        // No base, or no layer word: no section.
        let mut l = Livery { base: 0, off: [0; 8], layers: &layers, record: &mut rec, mask: &mut |_| Some(0), uvx: None, n: LiveryCounts::default() };
        assert_eq!(l.section(|_| 0), None);
        let none = vec![0u32; 256];
        let mut l = Livery { base: 100, off: [0; 8], layers: &none, record: &mut rec, mask: &mut |_| Some(0), uvx: None, n: LiveryCounts::default() };
        assert_eq!(l.section(|_| 0), None);
        assert!(parse_livery("screen,vehlivery=on") && !parse_livery("screen,vehlivery=off") && !parse_livery("screen"));
    }

    /// The section follows the palette rows; its slot colours share them.
    #[test]
    fn livery_section_in_the_block() {
        let remap = [0u32; 256];
        let rd = |off: u64| if off == colour_offset(0, 902) { Some([0, 0, 0, 0xFF, 0x30, 0, 0, 0]) } else { Some([1, 2, 3, 0xFF, 0xFF, 0xFF, 0, 0]) };
        let mut layers = vec![0u32; 256];
        layers[0] = 0x0100;
        let mut rec = |i: u32| {
            let mut b = [0u8; LIVERY_RECORD];
            if i == 40 {
                b[32..36].copy_from_slice(&900u32.to_le_bytes());
                b[44..48].copy_from_slice(&1u32.to_le_bytes());
            }
            Some(b)
        };
        let mut l = Livery { base: 40, off: [0; 8], layers: &layers, record: &mut rec, mask: &mut |_| Some(0), uvx: None, n: LiveryCounts::default() };
        let p = Palette::from_provided(&test_palette(28)).unwrap();
        let (t, n, at) = table_with(&remap, 7, Some(&p), false, false, rd, Some(&mut l));
        assert_eq!(at, Some(256 + PAL_TEXELS as u32), "after the one palette row (material 902's)");
        let at = at.unwrap() as usize;
        assert_eq!(t.len(), (at + LIVERY_HEAD + 2 * LIVERY_REC_DWORDS).div_ceil(12) * 12);
        let r1 = &t[at + LIVERY_HEAD + LIVERY_REC_DWORDS..];
        assert_eq!((r1[16], r1[18], r1[11]), (HAS_COLOUR | 0x03_0201, PALETTE_UNORM, 0));
        assert_eq!(n.palette, 1);
        // Without a livery: the INT1/PAL1 block unchanged.
        assert_eq!(table_with(&remap, 7, None, false, true, rd, None).0, table(&remap, 7, None, false, true, rd).0);
    }

    /// The weights [V 05483074 0x2684-0x2b60].
    #[test]
    fn livery_weights() {
        let (n, c1, c2) = mask_decode(1.0, 0.5);
        assert_eq!(n, 8);
        assert!((c1 - 1.0).abs() < 1e-6 && (c2 - 1.0).abs() < 1e-6);
        let (n, c1, c2) = mask_decode(0.3, 1.0);
        assert_eq!((n, c1), (15, 0.0), "b = 1: n 16 taken as 15");
        assert!((c2 - 0.5).abs() < 1e-6);
        // B: +20 bit n -> curve 2; else +28 bit 16 + n -> curve 1; else (+36, +16 bit n) curve 1; else 0.
        let mut rec = [0u32; 16];
        rec[5] = 1 << 3;
        assert_eq!(weight_b(&rec, 3, 0.9, 0.4, false).0, 0.4);
        rec[5] = 0;
        rec[7] = 1 << 19;
        assert_eq!(weight_b(&rec, 3, 0.9, 0.4, false).0, 0.9);
        rec[7] = 0;
        rec[4] = 1 << 3;
        assert_eq!((weight_b(&rec, 3, 0.9, 0.4, false).0, weight_b(&rec, 3, 0.9, 0.4, true).0), (0.0, 0.9));
        rec[6] = 1 << 3 | 1 << 19;
        assert_eq!(weight_b(&rec, 3, 0.9, 0.4, true), (0.9, 0.4, 0.4));
        // A: +20 bit n -> curve 2 - v59; else (+36, +16 bit n) curve 1 - v60; clamped.
        let mut a = [0u32; 16];
        a[5] = 1 << 2;
        assert_eq!(weight_a(&a, 2, 0.9, 0.75, false, 0.25, 0.0), 0.5);
        a[5] = 0;
        a[4] = 1 << 2;
        assert_eq!(weight_a(&a, 2, 0.9, 0.7, true, 0.0, 1.5), 0.0);
        // Keep two of three.
        assert_eq!(keep_two(0.0, 0.0), [(1.0, 0), (0.0, 1)]);
        assert_eq!(keep_two(1.0, 0.0), [(0.0, 0), (1.0, 1)]);
        assert_eq!(keep_two(0.25, 0.5), [(0.375, 0), (0.375, 2)]);
        assert_eq!(keep_two(0.75, 1.0), [(0.25, 2), (0.75, 1)]);
    }

    /// POL3: the mesh program's UV transform [V MS 42c87283 0xf98-0x1268]:
    /// s44 < 0 is identity; else u' = row.x u + row.y v + row.z s45 (v' from
    /// the row 48 bytes on, s46); the section carries it at LIVERY_UVX.
    #[test]
    fn livery_uv_transform() {
        assert_eq!((uvx_index((-1.0f32).to_bits()), uvx_index(f32::NAN.to_bits())), (None, None), "s44 < 0 (or NaN): identity");
        assert_eq!((uvx_index(0.0f32.to_bits()), uvx_index(2.0f32.to_bits())), (Some(0), Some(2)));
        // An atlas cell: u' = 0.5 u + 0.25 (row.z 0.5 x s45 0.5), v' = 0.25 v + 0.75 (vrow.z 1.5 x s46 0.5).
        let x = uvx([0.5, 0.0, 0.5], [0.0, 0.25, 1.5], 0.5f32.to_bits(), 0.5f32.to_bits());
        assert_eq!(x, [0.5, 0.0, 0.25, 0.0, 0.25, 0.75]);
        assert_eq!(uvx_apply(&x, [1.0, 1.0]), [0.75, 1.0]);
        // The record's rect test ([0, 1]^2 of A's UV set) sees the transformed UV: a raw UV of 1.5 falls
        // outside, its transform (1.0) inside; identity leaves it.
        let id = uvx([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0, 0);
        assert_eq!(uvx_apply(&id, [1.5, 0.5]), [1.5, 0.5]);
        assert_eq!(uvx_apply(&x, [1.5, 0.5]), [1.0, 0.875]);
        // In the section: six f32 rows at LIVERY_UVX, the flag after; records after LIVERY_HEAD.
        let mut layers = vec![0u32; 256];
        layers[3] = 0x01;
        let mut rec = |_: u32| Some([0u8; LIVERY_RECORD]);
        let mut l = Livery { base: 9, off: [0; 8], layers: &layers, record: &mut rec, mask: &mut |_| Some(0), uvx: Some(x), n: LiveryCounts::default() };
        let s = l.section(|_| 0).unwrap();
        assert_eq!(s[LIVERY_UVX..LIVERY_UVX + 7].iter().map(|w| f32::from_bits(*w)).take(6).collect::<Vec<_>>(), x.to_vec());
        assert_eq!((s[LIVERY_UVX + 6], s[8 + 3]), (1, 1 | LV_A), "flag set; u = 3 names record 1 (index 9)");
        assert_eq!((LIVERY_UVX, LIVERY_HEAD, s.len()), (264, 272, LIVERY_HEAD + 2 * LIVERY_REC_DWORDS));
        let mut l = Livery { base: 9, off: [0; 8], layers: &layers, record: &mut rec, mask: &mut |_| Some(0), uvx: None, n: LiveryCounts::default() };
        assert_eq!(l.section(|_| 0).unwrap()[LIVERY_UVX..LIVERY_HEAD], [0; 8], "identity: all zero");
        assert!(parse_livery("screen,vehlivery=debug") && parse_livery_debug("vehlivery=debug,screen") && !parse_livery_debug("vehlivery=on"));
        // RAW_SHADER reads the same layout (records at LIVERY_HEAD, the transform at LIVERY_UVX..+6),
        // the flags, and samples a livery slot's palette row at u = 0.5.
        let sh = crate::RAW_SHADER;
        for want in [format!("(layerTex >> 16) + {LIVERY_HEAD};"), format!("vmLoad(s + {}) != 0", LIVERY_UVX + 6), format!("vmLoad(s + {})", LIVERY_UVX), format!("vmLoad(s + {})))", LIVERY_UVX + 5)] {
            assert!(sh.contains(&want), "RAW_SHADER: {want}");
        }
        assert!(sh.contains(&format!("(layerFlags & 0x{LIVERY_DEBUG:x}) != 0")) && sh.contains(&format!("(layerFlags & 0x{LIVERY_FLAG:x}) != 0")));
        assert!(sh.contains("c2 = vmSlot(eb, 0.5, base, enc), ca = vmSlot(ea, 0.5, base, enc)"));
        assert!(sh.contains("sel == 1 ? i.uvl :"), "UV set 1 is the transformed UV0");
        // Both vertex entries compile (FXC; no device): the VS output grew by the transformed UV0.
        for e in [c"vs", c"vs_rec"] {
            crate::compile_source(sh, e, c"vs_5_1").unwrap_or_else(|err| panic!("{e:?}: {err:?}"));
        }
    }

    /// POL4: the transform and the mask decode against values worked from
    /// the disasm, the collapsed-transform guard, the view switches.
    #[test]
    fn livery_mask_and_transform_as_the_title() {
        // MS 42c87283 0x1058-0x1080: s[20:22] = the u row at r*0x60 + 12 s44; v41 = s21 v25 (v), += s20 v24
        // (u), += s45 s22. Rows at r = 2, s44 = 1 of a 0x60-byte table (u at 2*0x60+12, v 48 on).
        let mut t = vec![0f32; 3 * 24];
        let (r, e) = (2u64, 1u64);
        let at = |o: u64| (o / 4) as usize;
        t[at(r * LIVERY_UVX_STRIDE + e * LIVERY_UVX_ENTRY)..][..3].copy_from_slice(&[2.0, 0.5, 3.0]);
        t[at(r * LIVERY_UVX_STRIDE + e * LIVERY_UVX_ENTRY + LIVERY_UVX_V)..][..3].copy_from_slice(&[-1.0, 4.0, 0.25]);
        let row = |o: u64| [t[at(o)], t[at(o) + 1], t[at(o) + 2]];
        let base = r * LIVERY_UVX_STRIDE + uvx_index(1.0f32.to_bits()).unwrap() * LIVERY_UVX_ENTRY;
        let x = uvx(row(base), row(base + LIVERY_UVX_V), 0.5f32.to_bits(), 2.0f32.to_bits());
        // u = 0.25, v = 0.5: u' = 2 x 0.25 + 0.5 x 0.5 + 3 x 0.5 = 2.25; v' = -0.25 + 2 + 0.25 x 2 = 2.25.
        assert_eq!(uvx_apply(&x, [0.25, 0.5]), [2.25, 2.25]);
        assert!(uvx_ok(&x));
        // A collapsed or non-finite read is refused (the title's albedo shares the transform).
        assert!(!uvx_ok(&uvx([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.5f32.to_bits(), 0.5f32.to_bits())), "both rows constant");
        assert!(!uvx_ok(&uvx([1.0, 2.0, 0.0], [2.0, 4.0, 0.0], 0, 0)), "rank 1");
        assert!(!uvx_ok(&uvx([f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], 0, 0)));
        assert!(uvx_ok(&uvx([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0, 0)) && uvx_ok(&uvx([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0, 0)), "identity, a swap");
        // PS 05483074 0x2684-0x26f8: n = u32(16 x v103) (the sample's third channel, b), curve 1 =
        // sat((v101 + 0xbf051eb8) x 0x40056042), curve 2 = sat((v101 + 0xbdcccccd) x 0x40200000) (first, r).
        let (b, rr) = (0.5f32, 0.75f32);
        let want = ((16.0 * b) as u32, ((rr + f32::from_bits(0xbf05_1eb8)) * f32::from_bits(0x4005_6042)).clamp(0.0, 1.0), ((rr + f32::from_bits(0xbdcc_cccd)) * f32::from_bits(0x4020_0000)).clamp(0.0, 1.0));
        assert_eq!(mask_decode(rr, b), want);
        assert_eq!(want.0, 8);
        assert!((want.1 - 0.479_32).abs() < 1e-4 && want.2 == 1.0);
        // A grey texel (0.5, 0.5, 0.5) picks slot 8 at full curve-2 weight: one slot over the whole rect.
        assert_eq!(mask_decode(0.5, 0.5), (8, 0.0, 1.0));
        let sh = crate::RAW_SHADER;
        for want in ["uint n = min((uint)(mb.b * 16), 15);", "c2 = saturate((mb.r - 0.1) * 2.5)", "uint n = min((uint)(ma.b * 16), 15);", "c2 = saturate((ma.r - 0.1) * 2.5)", "g0 = float4(ddx(i.uvl), ddy(i.uvl))"] {
            assert!(sh.contains(want), "RAW_SHADER: {want}");
        }
        assert!(sh.contains(&format!("(layerFlags & 0x{:x}) != 0 && inside", LIVERY_SHOW_MASK | LIVERY_SHOW_UV)));
        assert!(sh.contains(&format!("(layerFlags & 0x{LIVERY_SHOW_MASK:x}) != 0 && m.x >= 0")) && sh.contains(&format!("(layerFlags & 0x{LIVERY_SHOW_UV:x}) != 0) return float3(frac(mq)")));
        // The switches.
        assert_eq!(parse_livery_view("screen,vehlivery=on"), 0);
        assert_eq!(parse_livery_view("vehlivery=debug"), LIVERY_DEBUG);
        assert_eq!(parse_livery_view("vehlivery=mask,screen"), LIVERY_DEBUG | LIVERY_SHOW_MASK);
        assert_eq!(parse_livery_view("vehlivery=uv"), LIVERY_DEBUG | LIVERY_SHOW_UV);
        assert!(parse_livery("vehlivery=mask") && parse_livery("vehlivery=uv") && !parse_livery("vehlivery=off"));
        crate::compile_source(sh, c"ps", c"ps_5_1").unwrap_or_else(|err| panic!("ps: {err:?}"));
    }

    /// A BC7 block from (value, bits) runs, LSB first.
    fn bc7_block(runs: &[(u32, u32)]) -> [u8; 16] {
        let (mut bits, mut at) = (0u128, 0u32);
        for &(v, n) in runs {
            bits |= u128::from(v) << at;
            at += n;
        }
        assert!(at <= 128);
        bits.to_le_bytes()
    }

    /// POL5: the BC7 single-subset decode (modes 6 and 5, rotation, the
    /// anchor's short index), the reserved mode, a partitioned mode refused.
    #[test]
    fn bc7_flat_blocks_decode() {
        // Mode 6: R 127/0, G 0/0, B 64/64, A 127/127, p 0/0; texel 1 index 15 (e1), the rest 0.
        let b = bc7_block(&[(1 << 6, 7), (127, 7), (0, 7), (0, 7), (0, 7), (64, 7), (64, 7), (127, 7), (127, 7), (0, 1), (0, 1), (0, 3), (15, 4)]);
        let (mode, t) = bc7_texels(&b);
        let t = t.unwrap();
        assert_eq!((mode, t[0], t[1], t[2]), (6, [254, 0, 128, 254], [0, 0, 128, 254], [254, 0, 128, 254]));
        // Mode 5, rotation 1 (A and R swap): R 127/127 -> 255, A 10/10, indices 0.
        let b = bc7_block(&[(1 << 5, 6), (1, 2), (127, 7), (127, 7), (0, 7), (0, 7), (0, 7), (0, 7), (10, 8), (10, 8)]);
        assert_eq!(bc7_texels(&b), (5, Some([[10, 0, 0, 255]; 16])));
        assert_eq!(bc7_texels(&[0; 16]), (8, Some([[0; 4]; 16])), "reserved mode: zeros");
        assert_eq!(bc7_texels(&bc7_block(&[(2, 2)])).0, 1);
        assert!(bc7_texels(&bc7_block(&[(2, 2)])).1.is_none(), "partitioned: not decoded");
    }

    /// POL5: a test run's police livery (layer masks 4x4 one-mip BC7 and the
    /// 2048x2048 lettering) through the title's weights and keep-two
    /// [V 05483074 0x2580-0x2b60]: a one-colour mask paints its whole rect
    /// (the box) unless its red is below 0.1 (curve 2) or the record's bits
    /// give its slot no curve; with the box as A the lettering (B) still
    /// shows over it; with the box as B nothing of A's lettering is left.
    #[test]
    fn livery_box_from_a_flat_mask() {
        // Every slot curve 2 (+20 all bits), nothing else.
        let rec = [0x0202, 0, 0x3c00, 0x3c00_0000, 0, u32::MAX, 0, 0, 1000, 0, 0, LV_NO_MASK, 0, 4.5f32.to_bits(), 1.0f32.to_bits(), 0];
        let (n, wb, wa) = texel_weights(&rec, 1.0, 0.5);
        assert_eq!((n, wb, wa), (8, 1.0, 1.0), "flat mask, red 1: full weight over the rect");
        assert_eq!(texel_weights(&rec, 0.05, 0.5), (8, 0.0, 0.0), "red below 0.1: no layer");
        let none = [0, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, LV_NO_MASK, 0, 0, 0, 0];
        assert_eq!(texel_weights(&none, 1.0, 0.5), (8, 0.0, 0.0), "no bit for the slot: no layer");
        // The lettering mask: red 1 on a letter, 0 between.
        let (_, letter_b, _) = texel_weights(&rec, 1.0, 0.25);
        let (_, gap_b, _) = texel_weights(&rec, 0.0, 0.25);
        // Box as A, lettering as B: letters (B) on the box (A), no base left.
        assert_eq!(keep_two(letter_b, wa), [(0.0, 0), (1.0, 1)]);
        assert_eq!(keep_two(gap_b, wa), [(1.0, 2), (0.0, 1)]);
        // Lettering as A, box as B: B everywhere, the letters never win.
        for a in [letter_b, 0.0] {
            assert_eq!(keep_two(wb, a)[1], (1.0, 1));
            assert!(keep_two(wb, a)[0].0 == 0.0);
        }
    }

    /// POL5: the `livery detail` texts over a section: each record's use
    /// and fields; a mask's memory (one non-zero block after 64 zero bytes,
    /// a flat mode 6 block: red 254, blue 128 -> slot 8, curve 2 = 1).
    #[test]
    fn livery_detail_texts() {
        let mut s = vec![0u32; LIVERY_HEAD + 2 * LIVERY_REC_DWORDS];
        (s[8], s[9]) = (1 | LV_A, 1 | LV_A);
        let r0 = LIVERY_HEAD;
        (s[r0 + 10], s[r0 + 11]) = (LV_NO_MASK, LV_NO_MASK);
        let r1 = LIVERY_HEAD + LIVERY_REC_DWORDS;
        s[r1..r1 + 15].copy_from_slice(&[0x0202, 0, 0x3c00, 0x3c00_0000, 0, u32::MAX, 0, 0, 1000, 0, 0, LV_NO_MASK, 0, 4.5f32.to_bits(), 1.0f32.to_bits()]);
        let t = section_text(&s, 2);
        assert!(t.contains("rec 0 (A 0, B 0)"), "{t}");
        assert!(t.contains("rec 1 (A 2, B 0): sets mask 2 rect 2; rect u' = 1 u + 0 v + 0, v' = 0 u + 1 v + 0; bits 00000000 ffffffff 00000000 00000000; slots 1000; X 0; mask A 0 B 255; scale 0 4.5 1"), "{t}");
        let flat = bc7_block(&[(1 << 6, 7), (127, 7), (127, 7), (0, 7), (0, 7), (64, 7), (64, 7), (127, 7), (127, 7)]);
        let mut mem = vec![0u8; 64];
        mem.extend_from_slice(&flat);
        let m = mask_bytes_text(&mem, true, &s, 2, 0);
        assert!(m.starts_with("bytes 80 blocks 5 not zero 1 first at 64: "), "{m}");
        assert!(m.ends_with("(BC7 mode 6: r 254-254 b 128-128; rec 1 as A: slot 8 weight 1.00)"), "{m}");
        assert_eq!(mask_bytes_text(&[0; 32], true, &s, 2, 0), "bytes 32 blocks 2 not zero 0");
        assert!(!mask_bytes_text(&mem, false, &s, 2, 0).contains("BC7"));
    }
}
