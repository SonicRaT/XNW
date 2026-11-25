//! `XNW_WORLD_PROBE=<dir>[,frames=<n>][,max=<k>][,at=<f>[+<f>..]][,raw=<n>]`:
//! read a frame's world geometry out at the engine's geometry draw
//! (title 0x140055E50, `rcx` = grmGeometry*)
//! and write it as OBJ. Off (no hook) when unset.
//!
//! The hook (an entry detour from watch.rs: registers preserved, the
//! original runs after it) does guarded reads only (ReadProcessMemory on
//! ourselves: a bad pointer makes that field unreadable and is counted) and
//! only during a probed frame; a background thread decodes after the frame
//! and writes `<dir>\frame-<n>\`: `scene.obj` (one object per draw, world
//! matrix = first half of the 128-byte cache entry), `scene-alt.obj` (second
//! half), `scene-local.obj` (one object per distinct geometry,
//! untransformed), `draws.tsv`, `effects.tsv`, `camera.txt`, and with
//! `raw=<n>` (first n probed frames) the raw blobs under `blobs\`.
//!
//! Probed frames: presented frames base+3, base+30, base+120 (then every
//! 120; `frames=<n>` how many), base = the world start when
//! `XNW_GPU=null@world` (the frame null mode begins), else the frame of the
//! first geometry draw; `at=` gives absolute presented frames instead.
//! `max=<k>` caps the captured draws per frame (default 20,000).
//!
//! Layouts (V = read in the title's code, I = inferred):
//! - TLS block = `gs:[58h][index]` (the title reads slot 0; the loader
//!   redirects it to its index: read from the thunk the prologue calls) [V].
//!   +0x11D8 drawable, +0x1208 model, +0x1210 geometry index, +0x1220 shader,
//!   +0x12A8 context, +0x12B8 viewport, +0x12C0 technique byte [V].
//! - geometry: +8 vertex buffer, +0x10 index buffer, +0x40 meshlet header
//!   (+0 grcBuffer*, +8 u16 & 0x7FFF meshlets, +0xA & 0x7FFFFF triangles) [V].
//! - grc buffer (as grcTexture): +8 u32 count, +0xC u16 & 0xFFF element
//!   size, +0x18 CPU address, +0x28 view (view +0x18 = descriptor record) [V
//!   0x140033CC0, 0x14325B4F0].
//! - the meshlet data and the vertex header: as `MS_VS_TransformD_aa739e09`
//!   reads them (decoders below) [V for that program, I for all].
//! - world: the CB the matrix setter binds, ctx+0x40+8*(word 0x1475EACB8 >> 2),
//!   equal to 0x1475E8F80 + 128 * i32 ctx+0x12190 [V 0x144E3D960]; the program
//!   multiplies its first 64 bytes into pos0 and the second into the
//!   previous position [V], so first = current [I].

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

// The instance rule (LOD slot, frustum, sane record) is shared with the device cull (PERF3).
use xnw_world::gpucull::{MAX_BATCH, frustum_planes, in_frustum, lod_slot, plausible_record};
use xnw_frame::{Bounds, Mesh, OBJ_MAX_VERTICES, Streams, WaterMap, WorldDraw, WorldSrc, WorldStats, camera_position, decode_indexed, decode_meshlets, expand_meshlets, fnv, light_of, plausible, rigid_world, rstate_text, sky_of, sun_of, vb_ready, water_grid, water_tint, write_world};

const IMAGE_BASE: usize = 0x1_4000_0000;
const GEOMETRY_DRAW: usize = 0x1_4005_5E50;
const MATRIX_CACHE: usize = 0x1_475E_8F80;
const MATRIX_SLOT: usize = 0x1_475E_ACB8;
const DIFFUSE_TEX: u32 = 0xC8E8_C282;
/// Albedo names in 0x145429040's order: 6994ef3f, DiffuseTex, 7d4f12e2, InteriorDiffuseMap [V].
const ALBEDO_NAMES: [u32; 4] = [0x6994_EF3F, DIFFUSE_TEX, 0x7D4F_12E2, 0x17D3_B5C2];

const TLS_DRAWABLE: u64 = 0x11D8;
const TLS_MODEL: u64 = 0x1208;
const TLS_GEOM_INDEX: u64 = 0x1210;
const TLS_SHADER: u64 = 0x1220;
const TLS_CTX: u64 = 0x12A8;
const TLS_VIEWPORT: u64 = 0x12B8;
/// The stage id of the running pass (0x14325D2A0 stores it).
const TLS_STAGE: u64 = 0x950;
const TLS_TECHNIQUE: u64 = 0x12C0;
const CTX_SLOTS: u64 = 0x40;
const CTX_MATRIX_INDEX: u64 = 0x1_2190;
const CTX_PROGRAMS: u64 = 0x1_2328;
const CTX_SRV_LAZY: u64 = 0x6200;
const TLS_PALETTE_N: u64 = 0x11E4;
const TLS_PALETTE: u64 = 0x11E8;
const TEX_HANDLES: usize = 0x1_49C3_E108;

const SLOTS: usize = 16;
/// Distinct geometries per frame whose raw objects go to `hex\`.
const DUMPS: usize = 20;
const MESH_CAP: u64 = 4 << 20;
const BUFFER_CAP: u64 = 16 << 20;
/// Raw bytes copied per frame at most.
const FRAME_BUDGET: u64 = 1 << 30;

/// Kinds of unreadable fields.
const KINDS: [&str; 11] = ["tls", "geometry", "meshlets", "mesh-data", "vertex", "index", "matrix", "shader", "effect", "slots", "viewport"];
#[derive(Clone, Copy)]
enum Kind {
    Tls,
    Geometry,
    Meshlets,
    MeshData,
    Vertex,
    Index,
    Matrix,
    Shader,
    Effect,
    Slots,
    Viewport,
}

macro_rules! miss {
    ($c:expr, $k:ident) => {
        $c.unread[Kind::$k as usize] += 1
    };
}

/// The frame pack v2 writer.
mod pack;
/// Draw-time bones for the live pass (`drawbones=on`).
mod skinhook;
/// The ped pre-skin's live reads (`preskin=on`).
mod preskin;
/// The imposter submits' live reads (`imposters=on`).
mod imposter;
/// The bridged rows' live reads (`bridgeps=on|<hashes>`).
mod bridgeps;
/// The list's geometry reads (`geomcopy=direct`, `geomasync=on`).
mod geomread;
/// `xnw burst`: a captured burst through the live list and pass functions.
pub(crate) mod burst;
/// The burst's title memory as the run had it: the census and `XNW_ZEROCOPY=on` (ZC1).
mod burst_mem;

// ---- configuration ----

#[derive(Debug, PartialEq)]
struct Config {
    dir: PathBuf,
    frames: usize,
    max: usize,
    at: Vec<u64>,
    raw: usize,
    /// `XNW_WORLD_RENDER=1`: every frame, no files; lists for our pass.
    render: bool,
    /// A `seq=` capture (light files, completion markers).
    seq: bool,
    /// Proof capture: `seq=auto:peds>=N:draws>=D:frames=F`: the first
    /// frame with at least N ped draws (remap from a skin bind) and D draws,
    /// and the F-1 frames after it, are dumped (screen or list mode).
    auto: Option<(u32, u32, u32)>,
    /// `seq=auto:...:vehicles>=V`: a second window, the first
    /// frame with at least V vehicle-effect draws (`standard_vehicle*`,
    /// `vehicle_*`) and the F-1 after it (0: none).
    auto_veh: u32,
    /// `cut=present`: the old frame cut at our present (default: the
    /// title's drain when both drain hooks are in).
    cut_present: bool,
    /// `census=1`: per distinct (effect, program set, stage,
    /// albedo outcome) a draw count, written to census-programs.tsv.
    census: bool,
    /// `skinlog=<n>`: in render mode, every 60th frame, log the first n
    /// skinned draws (palette, hash at hook and at our draw, bones used).
    skinlog: usize,
    /// `pack=2`: the frame pack v2 in seq frames.
    pack: bool,
    /// `hookin=1`: the hook's inputs per hook draw in the pack (v2.8).
    hookin: bool,
    /// `scan=exit`: scan.txt from the scene scan's records at the exit of
    /// the compaction 0x14547BE70 (its wrapper 0x140B5F960 calls it from
    /// our hook), not at the drain's entry, which can race the next scan.
    scan_exit: bool,
}

fn parse(v: &str) -> Result<Config, String> {
    let mut parts = v.split(',').map(str::trim);
    let dir = parts.next().filter(|d| !d.is_empty()).ok_or("no directory")?;
    let mut c = Config { dir: PathBuf::from(dir), frames: 3, max: 20_000, at: Vec::new(), raw: 0, render: false, seq: false, skinlog: 0, census: false, auto: None, auto_veh: 0, cut_present: false, pack: false, scan_exit: false, hookin: false };
    for p in parts {
        let (k, val) = p.split_once('=').ok_or_else(|| format!("`{p}`: key=value"))?;
        let n = |s: &str| s.trim().parse::<u64>().map_err(|_| format!("`{p}`: bad number"));
        match k.trim() {
            "frames" => c.frames = n(val)?.max(1) as usize,
            "max" => c.max = n(val)? as usize,
            "raw" => c.raw = n(val)? as usize,
            "skinlog" => c.skinlog = n(val)? as usize,
            "census" => c.census = n(val)? != 0,
            "cut" => c.cut_present = val.trim() == "present",
            "hookin" => c.hookin = n(val)? != 0,
            "scan" => match val.trim() {
                "exit" => c.scan_exit = true,
                "drain" => c.scan_exit = false,
                _ => return Err(format!("`{p}`: scan=exit or scan=drain")),
            },
            "pack" => match n(val)? {
                0 => c.pack = false,
                2 => c.pack = true,
                _ => return Err(format!("`{p}`: pack=2 (frame pack v2) or pack=0")),
            },
            "seq" if val.starts_with("auto") => {
                // seq=auto:peds>=N:draws>=D:frames=F (colons: commas split keys).
                let (mut peds, mut draws, mut frames) = (1u32, 2000u32, 6u32);
                for part in val.split(':').skip(1) {
                    let (k, v) = part.split_once(">=").or_else(|| part.split_once('=')).ok_or_else(|| format!("`{p}`: seq=auto:peds>=N:draws>=D:frames=F"))?;
                    let v = n(v)? as u32;
                    match k {
                        "peds" => peds = v,
                        "draws" => draws = v,
                        "frames" => frames = v.max(1),
                        "vehicles" => c.auto_veh = v,
                        _ => return Err(format!("`{p}`: seq=auto:peds>=N:draws>=D:frames=F")),
                    }
                }
                c.auto = Some((peds, draws, frames));
                c.seq = true;
            }
            "seq" => {
                // seq=<start>+<count>[/<step>]: count frames from start, every step.
                let (range, step) = val.split_once('/').map_or((val, "1"), |(r, s)| (r, s));
                let (start, count) = range.split_once('+').ok_or_else(|| format!("`{p}`: seq=<start>+<count>[/<step>]"))?;
                let (start, count, step) = (n(start)?, n(count)?, n(step)?.max(1));
                c.at = (0..count).map(|k| start + k * step).collect();
                c.seq = true;
            }
            "at" => {
                c.at = val.split('+').map(n).collect::<Result<_, _>>()?;
                c.at.sort_unstable();
            }
            _ => return Err(format!("`{p}`: unknown key (frames, max, at, seq, raw, skinlog, census, cut, pack, scan, hookin)")),
        }
    }
    Ok(c)
}

/// The probe's configuration from XNW_WORLD_PROBE and XNW_WORLD_RENDER
/// (`1` or `screen..`: the light capture feeding our pass). None: off.
fn config(probe: Option<&str>, render: Option<&str>) -> Result<Option<Config>, String> {
    // `capture`: the recorder, hooks and seq writer only; nothing of ours
    // touches D3D12 (screen_mode() is None for it: the pass is never made).
    let render = render.is_some_and(|v| v.trim() == "1" || v.trim() == "capture" || v.trim().starts_with("screen"));
    let v = match probe.filter(|v| !v.trim().is_empty()) {
        Some(v) => v,
        None if render => "-",
        None => return Ok(None),
    };
    let mut c = parse(v)?;
    if render {
        (c.render, c.max) = (true, 100_000);
        // Seq frames in screen mode need the dump (raw=0 wrote nothing).
        if c.seq {
            c.raw = c.raw.max(1);
        }
    }
    Ok(Some(c))
}

/// Which writer a probed frame goes to.
#[derive(Debug, PartialEq, Clone, Copy)]
enum Writer {
    /// Probe mode: draws.tsv, geometry blobs, OBJ.
    Heavy,
    /// Probe mode, `seq=`: draws.tsv, blobs, palbuf, complete; no OBJ.
    SeqLight,
    /// Screen mode: the live list (list.tsv, blobs) at `at=` frames.
    ScreenList,
    /// Screen mode, `seq=`: the light seq files, then the live list, then complete.
    ScreenSeq,
}

fn writer(c: &Config) -> Writer {
    match (c.render, c.seq) {
        (false, false) => Writer::Heavy,
        (false, true) => Writer::SeqLight,
        (true, false) => Writer::ScreenList,
        (true, true) => Writer::ScreenSeq,
    }
}

/// Offsets of the probed frames from the base.
fn offsets(frames: usize) -> Vec<u64> {
    (0..frames).map(|i| [3, 30, 120].get(i).copied().unwrap_or_else(|| 120 * (i as u64 - 1))).collect()
}

// ---- guarded reads ----

fn bytes(addr: u64, len: usize) -> Option<Vec<u8>> {
    if burst::serving() {
        return burst::read(addr, len, |to, from| crate::watch::guarded_read(from as usize, to));
    }
    if addr < 0x1_0000 || addr.checked_add(len as u64).is_none_or(|end| end > 0x8000_0000_0000) {
        return None;
    }
    let mut b = vec![0u8; len];
    crate::watch::guarded_read(addr as usize, &mut b).then_some(b)
}

fn arr<const N: usize>(addr: u64) -> Option<[u8; N]> {
    if addr < 0x1_0000 || addr >= 0x8000_0000_0000 {
        return None;
    }
    let mut b = [0u8; N];
    crate::watch::guarded_read(addr as usize, &mut b).then_some(b)
}

fn q(addr: u64) -> Option<u64> {
    arr::<8>(addr).map(u64::from_le_bytes)
}

fn le32(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4).map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
}

fn le16(b: &[u8], at: usize) -> u16 {
    b.get(at..at + 2).map_or(0, |s| u16::from_le_bytes(s.try_into().unwrap()))
}

fn le64(b: &[u8], at: usize) -> u64 {
    b.get(at..at + 8).map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap()))
}

fn f32s<const N: usize>(b: &[u8]) -> [f32; N] {
    std::array::from_fn(|i| f32::from_bits(le32(b, 4 * i)))
}

/// A grc resource header: (CPU address, bytes = count * (element & 0xFFF)).
fn grc(obj: &[u8]) -> (u64, u64) {
    (le64(obj, 0x18), u64::from(le32(obj, 8)) * u64::from(le16(obj, 0xC) & 0xFFF))
}

// ---- capture ----

#[derive(Default, Clone)]
struct Draw {
    /// The frame pack's hook sequence number (0: pack off).
    pack_seq: u32,
    /// A rigid part's bone, bones[model+0x33] (3 float4 rows,
    /// object space), read at the hook from TLS+0x11E8 when the model is not
    /// skinned and its index is under TLS+0x11E4 (0x1401D77D0).
    rigid: Option<[f32; 12]>,
    geom: u64,
    ret: u64,
    thread: u32,
    drawable: u64,
    model: u64,
    geom_index: u32,
    shader: u64,
    effect: u64,
    program: u64,
    program_flags: u64,
    technique: u8,
    viewport: u64,
    /// The meshlet header: count, triangles, the variant bit (39).
    meshlets: u32,
    triangles: u32,
    variant: bool,
    mesh_at: u64,
    mesh_len: u64,
    /// GPU base of the meshlet buffer's descriptor (V#), for comparison.
    mesh_vsharp: u64,
    vb_at: u64,
    vb_len: u64,
    vb_stride: u32,
    ib_at: u64,
    ib_len: u64,
    ib_elem: u32,
    /// The transform (current, previous 64 B), where from and which source
    /// ([`WorldSrc`]).
    world: Option<[f32; 32]>,
    world_at: u64,
    world_src: WorldSrc,
    /// The transform SRV slot (word 0x1475EACB8 >> 2), its lazy size
    /// (ctx+0x6200+2*slot: 0x80 when srv[] holds CPU matrices) and pointer.
    srv_slot: u16,
    srv_lazy: u16,
    bound: u64,
    /// When the bound pointer is a grc object: its buffer (address, bytes).
    bound_buffer: (u64, u64),
    /// The cache entry by index (0x1475E8F80 + 128 * ctx[+0x12190]).
    cache_at: u64,
    /// The skin palette at hook time: TLS+0x11E8 matrices (48 B), TLS+0x11E4 count.
    palette: (u64, u32),
    /// The second palette TLS+0x11F0 (previous frame? [I]).
    palette2: u64,
    /// The model says skinned (model+0x35 bit 0 [V 0x1443C6650]).
    skinned: bool,
    /// The geometry's hash +0x30; this draw's palette copy in the frame.
    ghash: u64,
    pal_index: u32,
    pal2_index: u32,
    /// Instanced draw (0x14006B210).
    inst: Option<InstInfo>,
    /// The last CPU-pointer view writer of srv[slot]
    /// (return address, r8, r9) and 64 B at r8; the viewport's world vp+0x000.
    view_writer: [u64; 3],
    view_elem0: Option<[f32; 16]>,
    vp_world: Option<[f32; 16]>,
    /// The draw's skin constants {curIndex, prevIndex, curCount, prevCount}.
    skin_cb: [u32; 4],
    /// This draw's bones copied at the hook (first bone in the
    /// capture's `hook_bones`, bones); None: not copied (`drawbones=on` only).
    hook_bones: Option<(u32, u32)>,
    /// Vehicle rules (`skinrules` bits 6/7): the draw's mask words (first in
    /// the capture's `hook_bones.masks`) and record +0x54; None: not a
    /// vehicle draw or not copied.
    veh: Option<(u32, u32)>,
    /// `skinrules` bit 8: an instanced vehicle draw's instances and the
    /// bones at its palette base, read at the list sink.
    wheel: Option<Box<skinhook::WheelInst>>,
    /// The capture's hook count when this draw was recorded (a sink pairs
    /// with the thread's last draw only while no other hook came between).
    hook_n: u64,
    /// Record +0x54 at the hook (TLS index): a PreSkinD draw's range word
    /// as the hook saw it (None: not read). Not the join key: the hook's
    /// record can be another draw's (the wheels read the body's).
    range: Option<u32>,
    /// The draw's own record +0x54, read at the list sink from slot 0 of
    /// its T9 table (`preskin=on`, PreSkinD programs): the pre-skin join key.
    sink_range: Option<u32>,
    /// `bridgeps`: the draw's binding inputs, read at its list sink when its
    /// pixel program is a named one (`worldprobe/bridgeps.rs`).
    bridged: Option<Box<xnw_world::bridged::Owned>>,
    /// The ped remap bases (record +0x74 current, +0x78 previous; -1: none).
    remap: [u32; 2],
    /// The ped skin record this draw's remap base came from (0x144E351B0
    /// on this thread, its current base equal to rec+0x74): (skin, buffer
    /// object at skin+0x20, comp, lod); skin 0: none (no remap applied).
    ped_skin: (u64, u64, u32, u32),
    /// `skinlog`: FNV-1a of the palette bytes this draw indexes, read at
    /// the hook (0: not logged), and the palette buffer base then.
    skin_hash: u64,
    pal_base: u64,
    /// The render state indices at the draw: blend (ctx+0x12378 u16),
    /// depth-stencil (+0x1236E u16), raster (+0x1236C u8), stencil ref (+0x12368 u8).
    state: [u16; 4],
    /// The stage id at TLS+0x950 at the draw (0: unread).
    stage_tls: u32,
    /// Microseconds since this thread's capture of the frame began,
    /// and the draw's index in it.
    hook_us: u32,
    seq_no: u32,
    /// The title's frame id (dword_14ABCE2A0) at the draw, and the time
    /// since the probe started (same clock as the drain markers).
    title_frame: u32,
    abs_us: u64,
    /// The camera position of the draw's viewport at hook time (seq only).
    hook_cam: [f32; 3],
    /// The camera position in the draw's own view block b0 (seq only).
    b0_cam: [f32; 3],
    /// The value at ctx+0x12360 at the draw (seq only; u32::MAX unread).
    sample_mask: u32,
    /// The map entity this draw belongs to (the draw-list entry
    /// 0x140139BB0 was handing to its handler on this thread; 0: none).
    ent: Entity,
    /// The per-draw record's +0x00 fade, seq only.
    fade: f32,
    /// The effect (*(TLS+0x1220)) read at hook time (seq or
    /// census): the shader object is per draw and gone when the writer runs
    /// (without it every row's effect name was "-").
    fx: u64,
    /// The pass this draw belongs to: (phase, stage) of the last pass begin
    /// 0x140369A80 on this thread and how many passes began here.
    pass: (u64, u32, u64),
    /// DiffuseTex: (var record index, texVars value, the texture).
    diffuse: Option<(u32, u16, u64, Tex)>,
    /// Per bound slot ctx+0x40+8*i: object pointer, and the resource when
    /// it reads as a grc object. Empty in the light capture (they made
    /// a Draw 1.5 KB: growth copies and fresh pages inside the hook).
    slots: Vec<u64>,
    slot_mem: Vec<Tex>,
    /// The albedo view's descriptor slot (view+0x18) and the
    /// effect's alpha-clip flag ([`crate::worldtex::albedo`]).
    tex_desc: u64,
    alpha_clip: bool,
    /// Layered effects: the layer diffuse descriptor slots.
    layer_desc: [u64; 4],
    /// The 2x2-quad sample mask at ctx+0x12360 (u64::MAX: solid).
    stipple: u64,
}

/// A grc resource object: memory address, bytes, format byte +0x3F, mips
/// +0x42, and the words at +0x38 / +0x3A (width, height? [U]).
#[derive(Default, Clone, Copy)]
struct Tex {
    obj: u64,
    at: u64,
    len: u64,
    fmt: u8,
    mips: u8,
    w: u16,
    h: u16,
    /// The name key (type-5 attachment) and how it was found (1 link tag, 2 node tag).
    key: u64,
    key_via: u8,
}



#[derive(Default)]
struct GeomData {
    mesh: Option<Vec<u8>>,
    vb: Option<Vec<u8>>,
    ib: Option<Vec<u8>>,
    ib_elem: u32,
    /// For the first 20 distinct geometries: named raw dumps (geometry,
    /// vertex buffer object, declaration at geom+0x20).
    dump: Vec<(&'static str, u64, Vec<u8>)>,
}

/// sh->fx (techniques, passes) and its template *fx (vars at +0x50,
/// count at +4).
struct Effect {
    head: Vec<u8>,
    template: u64,
    template_head: Vec<u8>,
    name: String,
    vars: u8,
}

#[derive(Clone, Copy)]
struct Camera {
    view: [f32; 16],
    proj: [f32; 16],
    origin: [f32; 4],
    draws: u64,
    /// The view and projection read at the LAST draw that used this
    /// viewport in the frame (a change mid-frame shows here), and the TAA
    /// jitter at viewport +0x708 at the first draw.
    last: Option<([f32; 16], [f32; 16])>,
    jitter: [f32; 2],
}

struct Capture {
    frame: u64,
    ordinal: usize,
    /// When this capture began (its first draw).
    t0: Instant,
    seen: u64,
    draws: Vec<Draw>,
    geoms: HashMap<u64, GeomData>,
    effects: HashMap<u64, Effect>,
    cameras: Vec<(u64, Camera)>,
    unread: [u64; KINDS.len()],
    copied: u64,
    /// Instance calls this frame: [accepted cpu-list, accepted gpu-buffer
    /// list, refused null, refused count, refused unreadable, via the buffer
    /// fallback, records accepted].
    inst_calls: [u64; 7],
    over_budget: u64,
    hook_ns: u64,
    threads: Vec<u32>,
    /// Raw dumps (first few each): texture objects (0x80 B), transform
    /// sources that are not finite matrices, bound buffers (4 KiB), palettes.
    dumps: Vec<(String, u64, Vec<u8>)>,
    dumped_tex: Vec<u64>,
    /// Light capture: palette copies by (pointer, drawable, bones),
    /// and how often one pointer served another drawable in the frame.
    pal_keys: HashMap<(u64, u64, u32), u32>,
    pal_list: Vec<Vec<u8>>,
    pal_shared: u64,
    /// Instance record blocks by (records pointer, count).
    inst_keys: HashMap<(u64, u32), u32>,
    /// Shared blocks: the static map records of kind-S batches are
    /// kept across frames (`STATIC_INST`), not copied every frame.
    inst_list: Vec<std::sync::Arc<Vec<u8>>>,
    /// Each block's source (0 cpu-list, 1 gpu-buffer list, 2 buffer view: grass or instfallback).
    inst_src: Vec<u8>,
    /// The frame's LOD records (140 B at *(rec+0x10)), once per pointer.
    lod_keys: HashMap<u64, u32>,
    lod_list: Vec<Vec<u8>>,
    /// Skinned draws (by the skin constants), how many agree with the TLS
    /// palette pointer, and the palette buffer extent used (bones).
    skin_draws: u64,
    skin_agree: u64,
    skin_extent: u32,
    /// The bones skinned draws read at their hook (`drawbones=on`).
    hook_bones: skinhook::HookBones,
    /// Slow frames: geometries first seen this frame (guarded reads in the hook).
    geom_misses: u32,
    /// Sampled hook parts (ns, scaled by the sampling rate): see PART_NAMES.
    parts: [u64; PARTS],
    /// Viewports whose first camera differed between recording threads.
    cam_conflicts: u32,
    /// The frame's hide bitsets, once per bits pointer.
    mask_keys: HashMap<u64, u32>,
    mask_list: Vec<Vec<u32>>,
    /// Draws per (effect, program set, stage, albedo outcome).
    census: HashMap<(u64, u64, u32, u8), u32>,
    /// The writer dumps this frame (decided when it is sent).
    dump: bool,
    /// The frame pack's data, moved in when the frame is sent.
    pack: Mutex<Option<pack::Taken>>,
}

/// The hook's parts, sampled every PART_SAMPLE-th call per thread.
const PARTS: usize = 7;
const PART_NAMES: [&str; PARTS] = ["lock wait", "read", "first-seen geometry", "transform+state", "model+skin", "instances", "other"];
const PART_SAMPLE: u64 = 16;

impl Capture {
    /// Hands the draw vector back for the next frame (a Draw is about
    /// 1.5 KB: a fresh vector per frame cost growth copies and page faults
    /// inside the hook).
    fn recycle(self) {
        spare(self.draws);
    }

    /// One frame from the per-thread captures: draws appended (block,
    /// LOD record and palette indices rebased), cameras united, counts summed.
    fn merge(mut self, o: Capture) -> Capture {
        let (blocks, lods, pals, masks) = (self.inst_list.len() as u32, self.lod_list.len() as u32, self.pal_list.len() as u32, self.mask_list.len() as u32);
        let shift = |i: u32, by: u32| if i == NO_PAL { NO_PAL } else { i + by };
        let mut draws = o.draws;
        for d in &mut draws {
            if let Some(i) = d.inst.as_mut() {
                i.block = shift(i.block, blocks);
                i.lodrec = shift(i.lodrec, lods);
                i.mask = shift(i.mask, masks);
            }
            d.pal_index = shift(d.pal_index, pals);
            d.pal2_index = shift(d.pal2_index, pals);
        }
        // The other thread's draw-time bones follow ours.
        let (bones, words) = self.hook_bones.merge(o.hook_bones);
        for d in &mut draws {
            if let Some(h) = d.hook_bones.as_mut() {
                h.0 += bones;
            }
            if let Some(v) = d.veh.as_mut() {
                v.0 += words;
            }
        }
        self.draws.extend(draws.drain(..));
        spare(draws);
        self.inst_list.extend(o.inst_list);
        self.inst_src.extend(o.inst_src);
        for k in 0..7 {
            self.inst_calls[k] += o.inst_calls[k];
        }
        self.lod_list.extend(o.lod_list);
        self.mask_list.extend(o.mask_list);
        self.pal_list.extend(o.pal_list);
        self.geoms.extend(o.geoms);
        self.effects.extend(o.effects);
        for (v, k) in o.cameras {
            match self.cameras.iter_mut().find(|(w, _)| *w == v) {
                // Another thread saw this viewport too: a different
                // first camera means the camera changed while recording.
                Some((_, mine)) => {
                    self.cam_conflicts += u32::from(mine.view != k.view || mine.proj != k.proj);
                    mine.draws += k.draws;
                    if k.last.is_some() {
                        mine.last = k.last;
                    }
                }
                None => self.cameras.push((v, k)),
            }
        }
        self.cam_conflicts += o.cam_conflicts;
        for (a, b) in self.unread.iter_mut().zip(o.unread) {
            *a += b;
        }
        for (a, b) in self.parts.iter_mut().zip(o.parts) {
            *a += b;
        }
        self.threads.extend(o.threads);
        self.dumps.extend(o.dumps);
        self.dumped_tex.extend(o.dumped_tex);
        (self.seen, self.copied, self.over_budget, self.hook_ns) = (self.seen + o.seen, self.copied + o.copied, self.over_budget + o.over_budget, self.hook_ns + o.hook_ns);
        (self.pal_shared, self.skin_draws, self.skin_agree, self.geom_misses) = (self.pal_shared + o.pal_shared, self.skin_draws + o.skin_draws, self.skin_agree + o.skin_agree, self.geom_misses + o.geom_misses);
        self.skin_extent = self.skin_extent.max(o.skin_extent);
        for (k, n) in o.census {
            *self.census.entry(k).or_default() += n;
        }
        self.ordinal = self.ordinal.min(o.ordinal);
        self
    }

    fn new(frame: u64, ordinal: usize) -> Capture {
        Capture { frame, ordinal, t0: Instant::now(), seen: 0, draws: SPARE.lock().ok().and_then(|mut v| v.pop()).unwrap_or_else(|| Vec::with_capacity(4096)), geoms: HashMap::new(), effects: HashMap::new(), cameras: Vec::new(), unread: [0; KINDS.len()], copied: 0, inst_calls: [0; 7], over_budget: 0, hook_ns: 0, threads: Vec::new(), dumps: Vec::new(), dumped_tex: Vec::new(), pal_keys: HashMap::new(), pal_list: Vec::new(), pal_shared: 0, inst_keys: HashMap::new(), inst_list: Vec::new(), inst_src: Vec::new(), lod_keys: HashMap::new(), lod_list: Vec::new(), skin_draws: 0, skin_agree: 0, skin_extent: 0, hook_bones: skinhook::HookBones::default(), geom_misses: 0, parts: [0; PARTS], census: HashMap::new(), mask_keys: HashMap::new(), mask_list: Vec::new(), cam_conflicts: 0, dump: false, pack: Mutex::new(None) }
    }

    fn dump(&mut self, kind: &str, at: u64, len: usize, max: usize) {
        if self.dumps.iter().filter(|d| d.0 == kind).count() < max
            && let Some(b) = bytes(at, len)
        {
            self.dumps.push((kind.to_string(), at, b));
        }
    }
}

/// Addresses of the title globals the capture reads.
#[derive(Clone, Copy, Default)]
struct Globals {
    /// word 0x1475EACB8: the transform SRV slot code.
    matrix_slot: u64,
    /// 0x1475E8F80: the transform cache (128 B per context index).
    matrix_cache: u64,
    /// 0x148FCA220: per (TLS+0x1110 + TLS+0x1294) 128-B draw constants;
    /// +0x30 = {curIndex, prevIndex, curCount, prevCount} (0x144E35520).
    skin_cb: u64,
    /// qword 0x1477AE218: base of the one palette buffer (stride 48).
    pal_base: u64,
    /// dword 0x1477AE2E4: its size (taken as bones, the larger reading).
    pal_size: u64,
    /// Qword 0x1475DF140: the per-frame arena base (offsets = pointer - base,
    /// 0x144DA8E10 V); the ped remap tables live there [I: t19 view over it].
    arena: u64,
}

impl Globals {
    fn of(title: usize) -> Globals {
        Globals { matrix_slot: real(title, MATRIX_SLOT), matrix_cache: real(title, MATRIX_CACHE), skin_cb: real(title, SKIN_CB), pal_base: real(title, PAL_BASE), pal_size: real(title, PAL_SIZE), arena: real(title, ARENA_BASE) }
    }
}

const SKIN_CB: usize = 0x1_48FC_A220;
const PAL_BASE: usize = 0x1_477A_E218;
const PAL_SIZE: usize = 0x1_477A_E2E4;
const ARENA_BASE: usize = 0x1_475D_F140;
/// Remap entries copied per base (bone indices are 8-bit in most layouts).
const REMAP_SPAN: u32 = 256;
const REMAP_MAX: u32 = 262_144;
const TLS_CB_A: u64 = 0x1110;
const TLS_CB_B: u64 = 0x1294;

struct State {
    cfg: Config,
    title: usize,
    tls_index: u64,
    /// Title globals the light capture reads (real addresses; tests point them at fakes).
    g: Globals,
    /// Base presented frame, `u64::MAX` until known.
    base: AtomicU64,
    wait_world: bool,
    done: AtomicBool,
    /// The frame being captured, `u64::MAX` when none (spares the locks).
    pending: AtomicU64,
    /// One capture slot per recording thread (five Subrender
    /// threads serialised on one capture lock, hook 118-142 ms a frame).
    slots: Mutex<Vec<Slot>>,
    /// The last frame handed on (a late slot of an older frame is dropped).
    sent: AtomicU64,
    /// Draws and captures recorded for a frame already handed on (after
    /// their frame's cut), dropped at send; the coherence line reports and
    /// resets them.
    late: [AtomicU64; 2],
    tx: Mutex<mpsc::Sender<Capture>>,
}

type Slot = std::sync::Arc<Mutex<Option<Capture>>>;

thread_local! {
    /// This thread's capture slot (keyed by its State: tests make several).
    static MY_SLOT: std::cell::RefCell<Option<(usize, Slot)>> = const { std::cell::RefCell::new(None) };
    /// Calls on this thread (part timers sample every PART_SAMPLE-th).
    static CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Per-thread caches of geometry keys (by pointer, hash) and model skinning.
    static GEOMS: std::cell::RefCell<HashMap<(u64, u64), GeomKey>> = std::cell::RefCell::new(HashMap::new());
    static MODELS: std::cell::RefCell<HashMap<u64, bool>> = std::cell::RefCell::new(HashMap::new());
}

static STATE: OnceLock<State> = OnceLock::new();

fn real(base: usize, va: usize) -> u64 {
    base.wrapping_add(va.wrapping_sub(IMAGE_BASE)) as u64
}

impl State {
    fn targets(&self) -> Option<Vec<u64>> {
        if !self.cfg.at.is_empty() {
            return Some(self.cfg.at.clone());
        }
        let b = self.base.load(Ordering::Relaxed);
        (b != u64::MAX).then(|| offsets(self.cfg.frames).into_iter().map(|o| b + o).collect())
    }

    /// The ordinal of `now` among the probed frames, fixing the base first.
    fn probed(&self, now: u64) -> Option<usize> {
        if self.cfg.render {
            // `at=` frames are dumped (heavy capture, files, raw blobs) while shown.
            return Some(self.cfg.at.iter().position(|f| *f == now).unwrap_or(usize::MAX));
        }
        if self.cfg.at.is_empty() && self.base.load(Ordering::Relaxed) == u64::MAX {
            let base = if self.wait_world { xnw_gpu::d3d12x::world_start()? } else { now };
            if self.base.compare_exchange(u64::MAX, base, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                let t = self.targets().unwrap_or_default();
                xnw_core::info!("world probe: base frame {base}, probing presented frames {t:?}");
            }
        }
        let t = self.targets()?;
        if t.last().is_some_and(|l| now > *l) {
            self.done.store(true, Ordering::Relaxed);
        }
        t.iter().position(|f| *f == now)
    }

    /// This thread's slot.
    fn slot(&self) -> Slot {
        let me = self as *const State as usize;
        MY_SLOT.with(|m| {
            let mut m = m.borrow_mut();
            if let Some((p, s)) = m.as_ref()
                && *p == me
            {
                return s.clone();
            }
            let s: Slot = std::sync::Arc::new(Mutex::new(None));
            if let Ok(mut v) = self.slots.lock() {
                v.push(s.clone());
            }
            *m = Some((me, s.clone()));
            s
        })
    }

    /// Whether every slot is empty.
    fn idle(&self) -> bool {
        self.slots.lock().is_ok_and(|v| v.iter().all(|s| s.lock().is_ok_and(|c| c.is_none())))
    }

    /// Hands the finished frame(s) to the writer: every slot's capture of a
    /// frame other than `now`, merged per frame.
    fn flush(&self, now: u64) {
        let p = self.pending.load(Ordering::Relaxed);
        if p == u64::MAX || p == now {
            return;
        }
        if self.pending.compare_exchange(p, u64::MAX, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            return;
        }
        self.send_where(|f| f != now);
    }

    /// Frame `n`'s drain returned: every recording thread is done with it
    /// (workers record only inside the drain): hand on frames up to `n`.
    fn flush_drain(&self, n: u64) {
        self.send_where(|f| f <= n);
    }

    /// Takes every slot's capture whose frame passes `take`, merges per frame
    /// and hands each frame on once, in order.
    fn send_where(&self, take: impl Fn(u64) -> bool) {
        let slots: Vec<Slot> = self.slots.lock().map(|v| v.clone()).unwrap_or_default();
        let mut done: Vec<Capture> = Vec::new();
        for s in &slots {
            let Ok(mut g) = s.lock() else { continue };
            if g.as_ref().is_some_and(|c| take(c.frame)) {
                done.push(g.take().unwrap());
            }
        }
        done.sort_by_key(|c| (c.frame, std::cmp::Reverse(c.draws.len())));
        let mut frames: Vec<Capture> = Vec::new();
        for c in done {
            match frames.last_mut() {
                Some(f) if f.frame == c.frame => {
                    let f0 = frames.pop().unwrap();
                    frames.push(f0.merge(c));
                }
                _ => frames.push(c),
            }
        }
        for mut c in frames {
            let sent = self.sent.load(Ordering::Relaxed);
            if sent != u64::MAX && c.frame <= sent {
                self.late[0].fetch_add(c.draws.len() as u64, Ordering::Relaxed);
                self.late[1].fetch_add(1, Ordering::Relaxed);
                c.recycle();
                continue;
            }
            self.sent.store(c.frame, Ordering::Relaxed);
            // Whether the writer dumps it is decided here, in send order
            // (it was decided by the writer, far behind the title), and the
            // frame's pack data moves into the capture (fault A).
            c.dump = !self.cfg.render || auto_dump(&self.cfg, &c) || (c.ordinal != usize::MAX && self.cfg.raw > 0);
            if !c.dump {
                // Its scan snapshot is not wanted: the queue keeps only the
                // dumped frames' (a 120-frame window kept 25 of 120 while it
                // held every frame).
                let f = c.frame;
                if let Ok(mut q) = SCANS.lock() {
                    q.retain(|(g, _)| u64::from(*g) != f);
                }
            }
            if self.cfg.pack {
                c.pack = Mutex::new(pack::take(c.frame, c.dump));
            }
            if let Ok(tx) = self.tx.lock() {
                let _ = tx.send(c);
            }
        }
    }
}

/// The imposter submit (ecx type, edx stage, r8d, r9 h+0x308, a5
/// h+0x228, a7 quad/strip, a8 IcRec) and its first 14 bytes (push rsi; push
/// rbx; sub rsp,78h; mov r10,[rsp+0C8h]; read from the exe).
const IMPOSTER_SUBMIT: usize = 0x1_451D_0AC0;
const IMPOSTER_SUBMIT_ENTRY: [u8; 14] = [0x56, 0x53, 0x48, 0x83, 0xEC, 0x78, 0x4C, 0x8B, 0x94, 0x24, 0xC8, 0x00, 0x00, 0x00];

/// The imposter submit's entry: registers at frame-24 (rcx), -32 (rdx),
/// -40 (r8), -48 (r9); stack arguments 5, 7, 8 at frame+0x30/+0x40/+0x48.
extern "system" fn on_imposter(frame: usize, _id: usize) {
    // SAFETY: the stub's frame and the caller's argument slots.
    let r = |o: isize| unsafe { ((frame as isize + o) as *const u64).read() };
    let args = [r(-24) & 0xFFFF_FFFF, r(-32) & 0xFFFF_FFFF, r(-40) & 0xFFFF_FFFF, r(-48), r(0x30), r(0x40) & 0xFF];
    let frame = frame_key();
    pack::imposter(frame, args, r(0x48));
    if imposter::on() {
        imposter::on_call(frame, args, r(0x48));
    }
}

const PASS_BEGIN: usize = 0x1_4036_9A80;
const PASS_BEGIN_ENTRY: [u8; 15] = [0x56, 0x57, 0x53, 0x48, 0x83, 0xEC, 0x20, 0x48, 0x8B, 0x01, 0x44, 0x89, 0xC3, 0x89, 0xD6];

thread_local! {
    /// (phase, stage, passes begun) of this thread's current pass.
    static PASS: std::cell::Cell<(u64, u32, u64)> = const { std::cell::Cell::new((0, 0, 0)) };
}

/// Pass begin 0x140369A80 (rcx = phase, edx = stage; first call of every
/// pass body): remembered per thread for the draws that follow.
extern "system" fn on_pass(frame: usize, _id: usize) {
    // SAFETY: the stub's frame.
    let (phase, stage) = unsafe { (((frame - 24) as *const u64).read(), ((frame - 32) as *const u64).read() as u32) };
    PASS.with(|p| {
        let n = p.get().2 + 1;
        p.set((phase, stage, n));
    });
}

/// The other geometry entries: the instanced geometry draw
/// 0x1402EC470 and the InstanceCulling consumers 0x14006B210 / 0x14013EFB0,
/// counted (calls since the start) to tell whether static world goes past
/// our hook. Index 0 is our hook 0x140055E50.
const ENTRIES: [(usize, &str); 4] = [(GEOMETRY_DRAW, "geometry 0x140055E50"), (0x1_402E_C470, "instanced geometry 0x1402EC470"), (0x1_4006_B210, "instance batch 0x14006B210"), (0x1_4013_EFB0, "instance batch 0x14013EFB0")];
static ENTRY_CALLS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

extern "system" fn on_entry(frame: usize, id: usize) {
    if let Some(c) = ENTRY_CALLS.get(id) {
        c.fetch_add(1, Ordering::Relaxed);
    }
    // 0x14006B210 (id 2) is captured as an instanced draw.
    if id == 2 {
        // SAFETY: the stub's frame.
        let (geom, draw, ret) = unsafe { (((frame - 24) as *const u64).read(), ((frame - 32) as *const u64).read(), ((frame + 8) as *const u64).read()) };
        hook_draw(geom, ret, Some(draw));
        return;
    }
    // 0x1402EC470 (id 1) has geometry_draw's signature (rcx = geometry).
    if id == 1 {
        // SAFETY: the stub's frame.
        let (geom, ret) = unsafe { (((frame - 24) as *const u64).read(), ((frame + 8) as *const u64).read()) };
        hook_draw(geom, ret, None);
    }
    // The inputs of 0x1402EC470 / 0x14013EFB0 in fully dumped frames (first 20 each).
    let Some(s) = STATE.get() else { return };
    let now = frame_key();
    let slot = s.slot();
    if s.probed(now).is_some_and(|o| o != usize::MAX || !s.cfg.render)
        && let Ok(mut g) = slot.lock()
        && let Some(c) = g.as_mut()
        && c.frame == now
    {
        // SAFETY: the stub's frame.
        let regs: [u64; 5] = unsafe { [frame - 24, frame - 32, frame - 40, frame - 48, frame + 8].map(|a| (a as *const u64).read()) };
        let name = if id == 1 { "entry-1402ec470" } else { "entry-14013efb0" };
        if c.dumps.iter().filter(|d| d.0 == name).count() < 20 {
            let mut b: Vec<u8> = regs.iter().flat_map(|r| r.to_le_bytes()).collect();
            b.extend(bytes(regs[1], 64).unwrap_or_default());
            c.dumps.push((name.to_string(), regs[0], b));
        }
    }
}

/// Per (context sub, SRV slot) the last caller of the CPU-pointer
/// view maker 0x144D9DCE0 (rcx = ctx+0x12780, dx = code, r8/r9 as passed).
static VIEW_WRITERS: Mutex<Option<HashMap<(u64, u16), [u64; 3]>>> = Mutex::new(None);
const VIEW_MAKER: usize = 0x1_44D9_DCE0;
const VIEW_MAKER_ENTRY: [u8; 14] = [0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x53, 0x48, 0x81, 0xEC];

extern "system" fn on_view(frame: usize, _id: usize) {
    // SAFETY: the stub's frame.
    let r: [u64; 5] = unsafe { [frame - 24, frame - 32, frame - 40, frame - 48, frame + 8].map(|a| (a as *const u64).read()) };
    if let Ok(mut m) = VIEW_WRITERS.lock() {
        m.get_or_insert_with(HashMap::new).insert((r[0], (r[1] as u16) >> 2), [r[4], r[2], r[3]]);
    }
}

/// The entry counts since the last call: "name n, ...".
fn entry_line() -> String {
    static LAST: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    ENTRIES.iter().enumerate().map(|(i, (_, n))| {
        let now = ENTRY_CALLS[i].load(Ordering::Relaxed);
        format!("{n} {}", now - LAST[i].swap(now, Ordering::Relaxed))
    }).collect::<Vec<_>>().join(", ")
}

fn tls_block(index: u64) -> Option<u64> {
    let array: u64;
    // SAFETY: reads TEB.ThreadLocalStoragePointer of the calling thread.
    unsafe { std::arch::asm!("mov {}, gs:[0x58]", out(reg) array, options(nostack, readonly, preserves_flags)) };
    q(array.checked_add(8 * index)?)
}

/// The hook: `[frame+8]` the return address, rcx at `frame-24`.
extern "system" fn on_draw(frame: usize, _id: usize) {
    ENTRY_CALLS[0].fetch_add(1, Ordering::Relaxed);
    // SAFETY: the stub's own frame, readable.
    let (geom, ret) = unsafe { (((frame - 24) as *const u64).read(), ((frame + 8) as *const u64).read()) };
    hook_draw(geom, ret, None);
}

/// A geometry draw (`inst`: the IcDraw* of an instanced one, 0x14006B210).
fn hook_draw(geom: u64, ret: u64, inst: Option<u64>) {
    let Some(s) = STATE.get() else { return };
    if s.done.load(Ordering::Relaxed) {
        return;
    }
    hook_with(s, frame_key(), geom, ret, inst, title_tls(s.tls_index));
}

/// The hook's body for presented frame `now` on the calling thread (tests
/// drive it with fake title memory): this thread's own slot, so recording
/// threads never wait for each other.
fn hook_with(s: &State, now: u64, geom: u64, ret: u64, inst: Option<u64>, tls: u64) {
    let t0 = Instant::now();
    let sampled = CALLS.with(|n| {
        let k = n.get() + 1;
        n.set(k);
        k % PART_SAMPLE == 0
    });
    if !DRAIN_CUT.load(Ordering::Relaxed) {
        s.flush(now);
    }
    let Some(ordinal) = s.probed(now) else { return };
    let slot = s.slot();
    let tl = sampled.then(Instant::now);
    let Ok(mut g) = slot.lock() else { return };
    if g.as_ref().is_some_and(|c| c.frame != now) {
        // An older frame no flush took yet: hand it on first.
        drop(g);
        if DRAIN_CUT.load(Ordering::Relaxed) {
            s.flush_drain(now.wrapping_sub(1));
        } else {
            s.pending.store(now.wrapping_sub(1), Ordering::Relaxed);
            s.flush(now);
        }
        g = match slot.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
    }
    let c = g.get_or_insert_with(|| Capture::new(now, ordinal));
    if let Some(tl) = tl {
        c.parts[0] += PART_SAMPLE * tl.elapsed().as_nanos() as u64;
    }
    s.pending.store(now, Ordering::Relaxed);
    c.seen += 1;
    let pack_seq = pack::hook_entry(now);
    if c.draws.len() < s.cfg.max {
        let before = c.draws.len();
        if s.cfg.render {
            // Screen mode records the live path in every frame; `at=`/`seq=`
            // frames dump that very list (dump_list), so offline = live.
            capture_light(s, c, geom, ret, tls, sampled);
        } else {
            capture_draw(s, c, geom, ret);
        }
        // The entity whose draw-list entry is being drawn.
        if c.draws.len() > before
            && let Some(d) = c.draws.last_mut()
        {
            d.ent = current_entity();
            d.pack_seq = pack_seq;
            d.hook_n = c.seen;
            if pack_seq != 0 && d.palette.1 != 0 {
                // SAFETY: the calling thread's TLS block (as the light capture reads it).
                let prev = if tls != 0 { unsafe { pread::<u64>(tls + TLS_PALETTE + 8) }.unwrap_or(0) } else { d.palette2 };
                pack::hook_objects(now, pack_seq, d.palette, prev, d.model, d.ent.at);
            }
            if pack_seq != 0 {
                pack::hook_inputs(now, pack_seq, tls, geom);
                pack::hook_views(now, &[d.tex_desc, d.layer_desc[0], d.layer_desc[1], d.layer_desc[2], d.layer_desc[3]], &[(d.vb_at, d.vb_len), (d.mesh_at, d.mesh_len)]);
            }
        }
        // Instanced calls whose geometry draw was not recorded, by reason.
        if inst.is_some() && c.draws.len() == before {
            INST_LOST[usize::from(tls != 0)].fetch_add(1, Ordering::Relaxed);
        }
        if let Some(draw) = inst
            && c.draws.len() > before
        {
            let ti = sampled.then(Instant::now);
            // SAFETY: the calling thread's TLS block.
            let ctx = if tls != 0 { unsafe { pread::<u64>(tls + TLS_CTX) }.unwrap_or(0) } else { 0 };
            capture_instances(c, draw, ctx);
            if let Some(ti) = ti {
                c.parts[5] += PART_SAMPLE * ti.elapsed().as_nanos() as u64;
            }
        }
    }
    let total = t0.elapsed().as_nanos() as u64;
    c.hook_ns += total;
    if sampled {
        // The sampled total; "other" = total - the parts (render_frame).
        c.parts[6] += PART_SAMPLE * total;
    }
}

/// The instances of a 0x14006B210 draw: rec = *draw,
/// desc = *rec, count u32 desc+8, CPU records desc+0x78 (desc+0x68 == 0:
/// no GPU buffer), lod u32 draw+8. The records (64 B each, built with the
/// map) are copied once per (records, count) per frame, at most 16,384.
/// The batch's hide bitset [V 0x144EBE530 / 0x1435A4D50]: P = ctx+0xA10
/// (set by the instance handler just before its draws when the mask applies,
/// 0 otherwise); P = {u32* bits, u32 words +8, u32 bits +0xC}. Copied once
/// per bits pointer per frame (at most 2,048 words = 65,536 instances).
fn hide_mask(c: &mut Capture, ctx: u64) -> u32 {
    if ctx == 0 {
        return NO_PAL;
    }
    // SAFETY: the draw's render context and the bitset it binds (fault-fixed).
    let Some(p) = (unsafe { dread::<u64>(ctx + 0xA10) }).filter(|p| *p != 0) else { return NO_PAL };
    let Some(h) = (unsafe { dread::<[u8; 16]>(p) }) else { return NO_PAL };
    let (bits, words) = (le64(&h, 0), le32(&h, 8).min(2048));
    if bits == 0 || words == 0 {
        return NO_PAL;
    }
    if let Some(i) = c.mask_keys.get(&bits) {
        return *i;
    }
    let Some(b) = dcopy(bits, 4 * words as usize) else { return NO_PAL };
    let i = c.mask_list.len() as u32;
    c.mask_list.push(b.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect());
    c.mask_keys.insert(bits, i);
    i
}

fn capture_instances(c: &mut Capture, draw: u64, ctx: u64) {
    if c.draws.is_empty() {
        return;
    }
    // Direct reads: the IcDraw, its IcRec, desc, LOD record and CPU
    // records are what this very call reads (a guarded read is a
    // syscall, five of them per instanced draw).
    // SAFETY: the call's own arguments and what it reads.
    let rec = unsafe { dread::<u64>(draw) }.unwrap_or(0);
    // IcRec +0 desc, +0x10 the LOD record T, +0xA0 bits 15-18 the LOD mask.
    // SAFETY: as above.
    let Some(r) = (unsafe { dread::<[u8; 0xA4]>(rec) }) else { return };
    let desc = le64(&r, 0);
    // SAFETY: as above.
    let Some(h) = (unsafe { dread::<[u8; 0x90]>(desc) }) else { return };
    let n = le32(&h, 8).min(MAX_BATCH);
    let (gpu, cpu) = (le64(&h, 0x68), le64(&h, 0x78));
    let mask = le32(&r, 0xA0) >> 15 & 0xF;
    let first_lod = if mask == 0 { NO_LOD } else { mask.trailing_zeros() as u8 };
    let tp = le64(&r, 0x10);
    let lodrec = match c.lod_keys.get(&tp) {
        Some(i) => *i,
        // SAFETY: the LOD record the call hands to the culling shader (V).
        None => match unsafe { dread::<[u8; LODREC_BYTES]>(tp) }.map(|b| b.to_vec()) {
            Some(b) => {
                let i = c.lod_list.len() as u32;
                c.lod_list.push(b);
                c.lod_keys.insert(tp, i);
                i
            }
            None => NO_PAL,
        },
    };
    // SAFETY: the IcDraw.
    let lod = unsafe { dread::<u32>(draw + 8) }.unwrap_or(0);
    let mask = hide_mask(c, ctx);
    // The handler and the fields 0x1435A4D50 / 0x144E54970 test.
    let handler = le64(&r, 8).wrapping_sub(0x2C4);
    // SAFETY: the batch's handler (fault-fixed reads).
    let (h228, h33c) = unsafe { (dread::<u32>(handler.wrapping_add(0x228)).unwrap_or(u32::MAX), dread::<u16>(handler.wrapping_add(0x33C)).unwrap_or(u16::MAX)) };
    if let Some(d) = c.draws.last_mut() {
        d.inst = Some(InstInfo { desc, count: le32(&h, 8), cpu, gpu, lod, block: NO_PAL, first_lod, lodrec, mask, handler, h228, h33c, rec_a0: le32(&r, 0xA0) });
    }
    // Kind S (desc+0x68 set) [layout V, CPU list I]: the batch's
    // static map records at L+0x90 (L = desc+0, count u16 L+0x98 == n),
    // copied once per (L, records, n) and shared by later frames.
    if gpu != 0 && n != 0 {
        let l = le64(&h, 0);
        // SAFETY: the instance list the title's culling reads for this call.
        let lh = unsafe { dread::<[u8; 0xA0]>(l) };
        let (recs, count) = lh.map_or((0, 0), |lh| (le64(&lh, 0x90), u32::from(le16(&lh, 0x98))));
        // Refusal reasons: 2 null list pointer, 3 count differs, 4 list unreadable.
        let refused = if lh.is_none() { 4 } else if recs == 0 { 2 } else if count != le32(&h, 8) { 3 } else { 0 };
        let (src, block) = if refused == 0 {
            c.inst_calls[1] += 1;
            let key = (l, recs, n);
            match c.inst_keys.get(&(recs, n)) {
                Some(b) => (recs, Some(*b)),
                None => match static_records(key, c.frame) {
                    Some(b) => {
                        let i = c.inst_list.len() as u32;
                        c.inst_list.push(b);
                        c.inst_src.push(1);
                        c.inst_keys.insert((recs, n), i);
                        (recs, Some(i))
                    }
                    None => (recs, None),
                },
            }
        } else {
            c.inst_calls[refused] += 1;
            if REFUSAL_LOGS.fetch_add(1, Ordering::Relaxed) < 8 {
                // L's first qwords (a vtable names its type) and the batch desc's.
                let q = |b: &[u8], at: usize| b.get(at..at + 8).map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap()));
                let lq = lh.map(|lh| [q(&lh, 0), q(&lh, 8), q(&lh, 0x10), q(&lh, 0x18)]).unwrap_or_default();
                let title = STATE.get().map_or(0, |s| s.title as u64);
                let rel = |v: u64| if title != 0 && v >= title && v - title < 0x1000_0000 { format!("title+{:#x}", v - title) } else { format!("{v:#x}") };
                xnw_core::info!("world probe: instance batch refused ({}): L {l:#x} (qwords {} {:#x} {:#x} {:#x}), records {recs:#x}, u16 count {count}, batch count {}, buffer {gpu:#x}, desc qwords {:#x} {:#x} {:#x} {:#x}", ["", "", "null list", "count differs", "list unreadable"][refused], rel(lq[0]), lq[1], lq[2], lq[3], le32(&h, 8), q(&h, 0x10), q(&h, 0x18), q(&h, 0x20), q(&h, 0x28));
                // Grass (vtable 0x1461CE3D0, ctor 0x145BAAF30 [V]): +0x80/+0x90 are
                // its AABB (FLT_MAX-initialised Vec4s, so "records" read two floats),
                // atArrays {ptr, u16 count, u16 capacity} at +0x158/+0x168/+0x178/+0x188
                // [V dtor 0x1457B1D00]; desc+0x68 = **(L+0x158). Which holds the
                // instances, and their stride: these lines, plus the buffer view's bytes.
                // SAFETY: the same list object (fault-fixed read).
                if let Some(lx) = unsafe { dread::<[u8; 0x1A0]>(l) } {
                    let arr: Vec<String> = [0x158usize, 0x168, 0x178, 0x188].iter().map(|&o| {
                        let (p, n) = (q(&lx, o), u16::from_le_bytes([lx[o + 8], lx[o + 9]]));
                        // SAFETY: the array's first bytes (fault-fixed read).
                        let first = unsafe { dread::<[u8; 32]>(p) }.map_or("-".into(), |b| b.iter().map(|x| format!("{x:02x}")).collect::<String>());
                        format!("+{o:#x} {p:#x} n {n} [{first}]")
                    }).collect();
                    let view = q(&lx, 0x158);
                    // SAFETY: as above: element 0 of the +0x158 array and the buffer view behind desc+0x68.
                    let e0 = unsafe { dread::<[u64; 4]>(q(&lx, 0x158)) }.unwrap_or_default();
                    let range = crate::worldprobe::buffer_view_of(gpu);
                    xnw_core::info!("world probe: refused L {l:#x}: aabb {:?} {:?}; arrays {}; +0x158 elements {e0:x?} (array {view:#x}); buffer view {range:x?} (bytes / batch count = {})", f32s::<4>(&lx[0x80..]), f32s::<4>(&lx[0x90..]), arr.join("; "), range.map_or(0, |(_, len)| len / u64::from(le32(&h, 8).max(1))));
                }
            }
            // Fallback: the buffer view behind desc+0x68 (its CPU-visible bytes).
            // Off by default (`instfallback=on`): in one capture the stage 76/69
            // instances read this way have rotations of 1e29-1e37 and a zero
            // translation (not IcInstance records): screen-sized black polygons.
            // Grass batch lists (vtable 0x1461CE3D0; effects grass_batch_wind /
            // grass_batch_lod_wind, stage 77): 16-B packed instances through the
            // same buffer view, decoded as CS_Grass_BatchCull_ValidateInst does.
            // SAFETY: L's vtable (fault-fixed read).
            let is_grass = STATE.get().is_some_and(|s| unsafe { dread::<u64>(l) } == Some((s.title as u64).wrapping_add((GRASS_LIST_VT - IMAGE_BASE) as u64)));
            let read = if is_grass && render_switch("grass", true) {
                grass_records(l, gpu, n, c.frame)
            } else if render_switch("instfallback", false) {
                buffer_records(gpu, n, c.frame)
            } else {
                None
            };
            match read {
                Some((addr, b)) => {
                    c.inst_calls[5] += 1;
                    match c.inst_keys.get(&(addr, n)) {
                        Some(i) => (addr, Some(*i)),
                        None => {
                            let i = c.inst_list.len() as u32;
                            c.inst_list.push(b);
                            c.inst_src.push(2);
                            c.inst_keys.insert((addr, n), i);
                            (addr, Some(i))
                        }
                    }
                }
                None => (0, None),
            }
        };
        let _ = src;
        let Some(block) = block else { return };
        c.inst_calls[6] += u64::from(n);
        if let Some(d) = c.draws.last_mut()
            && let Some(info) = d.inst.as_mut()
        {
            info.block = block;
        }
        return;
    }
    if gpu != 0 || cpu == 0 || n == 0 {
        return;
    }
    let block = match c.inst_keys.get(&(cpu, n)) {
        Some(b) => *b,
        None => {
            if !(0x1_0000..0x8000_0000_0000).contains(&cpu) {
                return;
            }
            // The CPU records this call draws (desc+0x78 with no GPU set).
            let Some(b) = dcopy(cpu, 64 * n as usize) else { return };
            c.copied += b.len() as u64;
            c.inst_calls[0] += 1;
            c.inst_calls[6] += u64::from(n);
            let i = c.inst_list.len() as u32;
            c.inst_list.push(std::sync::Arc::new(b));
            c.inst_src.push(0);
            c.inst_keys.insert((cpu, n), i);
            i
        }
    };
    if let Some(d) = c.draws.last_mut()
        && let Some(info) = d.inst.as_mut()
    {
        info.block = block;
    }
}

/// Kind-S static records by (L, records pointer, count): the bytes
/// and the frame they were last used in; entries unused for 600 frames go.
type StaticInst = HashMap<(u64, u64, u32), (std::sync::Arc<Vec<u8>>, u64)>;
static STATIC_INST: Mutex<Option<StaticInst>> = Mutex::new(None);
/// Instanced calls with no recorded draw: [no TLS block, the
/// geometry or its header not readable].
static INST_LOST: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Refusals logged so far (the first 8 are).
static REFUSAL_LOGS: AtomicU64 = AtomicU64::new(0);

/// Fallback [I]: the records of a kind-S batch through its GPU
/// buffer object: view *(gbuf+0x28), its descriptor slot view+0x18, resolved
/// by our layer to the title memory the buffer view reads; at most `n`
/// records of 64 B and never past the view's bytes; cached like the lists.
/// The grass batch list's vtable (ctor 0x145BAAF30, dtor 0x1457B1D00 [V]).
const GRASS_LIST_VT: usize = 0x1_461C_E3D0;

/// One packed grass instance (16 B) as CS_Grass_BatchCull_ValidateInst
/// (shader 1c621f63..., 0x160-0x218) reads it [V]: `v_lshl_add_u32 v1, v0, 4`
/// (stride 16), dwordx3; x = u16 0 / 65535, y = u16 1 / 65535, z = u16 2 /
/// 65535, each `fma(q, extent, min)` against the batch box; scale = u8 11 /
/// 255. Bytes 6-7 (normal), 8-10 (colour), 12-15: not used here [I an earlier
/// title's layout]. Returns (position, scale fraction).
fn grass_instance(b: &[u8], min: [f32; 3], max: [f32; 3]) -> ([f32; 3], f32) {
    let u = |k: usize| f32::from(u16::from_le_bytes([b[2 * k], b[2 * k + 1]])) / 65535.0;
    (std::array::from_fn(|k| min[k] + u(k) * (max[k] - min[k])), f32::from(b[11]) / 255.0)
}

/// A grass batch's instances as 64-B IcInstance-shaped records for the
/// common path: a uniform scale from the list's scale range (+0xA0 x..y,
/// lerped by the instance's fraction [I]), no rotation (blades face the
/// camera / terrain in the title's shader [I]), absolute translation; the
/// LOD fields rec+0x30/+0x38 = 1/scale. Box: L+0x80 min, L+0x90 max [V ctor].
fn grass_block(raw: &[u8], n: usize, min: [f32; 3], max: [f32; 3], range: [f32; 2]) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 * n);
    for b in raw.chunks_exact(16).take(n) {
        let (p, f) = grass_instance(b, min, max);
        let s = if range[0] > 0.0 && range[1] >= range[0] && range[1] < 100.0 { range[0] + f * (range[1] - range[0]) } else { 1.0 };
        let r: [f32; 16] = [s, 0.0, 0.0, p[0], 0.0, s, 0.0, p[1], 0.0, 0.0, s, p[2], 1.0 / s, 0.0, 1.0 / s, 0.0];
        for v in r {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// A grass batch read through its buffer view (16 B per instance),
/// cached by (list, address, count) like the static records.
fn grass_records(l: u64, gbuf: u64, n: u32, frame: u64) -> Option<(u64, std::sync::Arc<Vec<u8>>)> {
    let (addr, len) = buffer_view_of(gbuf)?;
    let n = n.min((len / 16) as u32);
    if n == 0 {
        return None;
    }
    let key = (l, addr | 1 << 63, n);
    {
        let mut g = STATIC_INST.lock().ok()?;
        if let Some((b, used)) = g.get_or_insert_with(HashMap::new).get_mut(&key) {
            *used = frame;
            return Some((addr, b.clone()));
        }
    }
    // SAFETY: the list object (fault-fixed read).
    let lx = unsafe { dread::<[u8; 0xB0]>(l) }?;
    let (lo, hi, sr) = (f32s::<4>(&lx[0x80..]), f32s::<4>(&lx[0x90..]), f32s::<4>(&lx[0xA0..]));
    if !(0..3).all(|k| lo[k].is_finite() && hi[k].is_finite() && hi[k] >= lo[k] && hi[k] - lo[k] < 1e5) {
        return None;
    }
    let raw = dcopy(addr, 16 * n as usize)?;
    let b = std::sync::Arc::new(grass_block(&raw, n as usize, [lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]], [sr[0], sr[1]]));
    if let Ok(mut g) = STATIC_INST.lock() {
        g.get_or_insert_with(HashMap::new).insert(key, (b.clone(), frame));
    }
    Some((addr, b))
}

/// The title memory (address, bytes) behind a batch's GPU buffer desc+0x68.
fn buffer_view_of(gbuf: u64) -> Option<(u64, u64)> {
    let view = q(gbuf.wrapping_add(0x28))?;
    let slot = q(view.wrapping_add(0x18))?;
    xnw_gpu::d3d12x::buffer_view_range(slot)
}

fn buffer_records(gbuf: u64, n: u32, frame: u64) -> Option<(u64, std::sync::Arc<Vec<u8>>)> {
    let view = q(gbuf.wrapping_add(0x28))?;
    let slot = q(view.wrapping_add(0x18))?;
    let (addr, len) = xnw_gpu::d3d12x::buffer_view_range(slot)?;
    let n = n.min((len / 64) as u32);
    if n == 0 {
        return None;
    }
    static_records((gbuf, addr, n), frame).map(|b| (addr, b))
}

fn static_records(key: (u64, u64, u32), frame: u64) -> Option<std::sync::Arc<Vec<u8>>> {
    let mut g = STATIC_INST.lock().ok()?;
    let m = g.get_or_insert_with(HashMap::new);
    if let Some((b, used)) = m.get_mut(&key) {
        *used = frame;
        return Some(b.clone());
    }
    let b = std::sync::Arc::new(dcopy(key.1, 64 * key.2 as usize)?);
    if m.len() > 8192 {
        m.retain(|_, (_, used)| frame.saturating_sub(*used) <= 600);
    }
    m.insert(key, (b.clone(), frame));
    Some(b)
}

#[derive(Clone, Copy, Debug, Default)]
struct InstInfo {
    desc: u64,
    count: u32,
    cpu: u64,
    gpu: u64,
    lod: u32,
    /// Index into the frame's copied record blocks (NO_PAL: none).
    block: u32,
    /// The batch's first LOD (lowest bit of rec+0xA0 b15-18;
    /// NO_LOD: none) and its LOD record in `Capture::lod_list` (NO_PAL: none).
    first_lod: u8,
    lodrec: u32,
    /// The batch's hide bitset in `Capture::mask_list` (NO_PAL: none).
    mask: u32,
    /// The batch's handler h = *(rec+8) - 0x2C4, its h+0x228 (dword),
    /// h+0x33C/+0x33D (two bytes) and rec+0xA0 (the call's LOD mask word).
    handler: u64,
    h228: u32,
    h33c: u16,
    rec_a0: u32,
}

const NO_LOD: u8 = 0xFF;
/// A list row drawn from the CPU's instances (not a device-culled batch).
const NO_CULL: u32 = u32::MAX;
const LODREC_BYTES: usize = 140;

/// One instanced call's records filtered by the LOD rule into
/// camera-relative worlds (12 floats each, appended to `out`): (tested, drawn).
/// `lod`: (record T, this call's slot); None draws every record.
/// `hide`: the title's batchdisabledbitfield (u32 words, LSB first):
/// instance i is dropped when bit i&31 of word i>>5 is set. Returns
/// (tested, drawn, hidden by the mask).
#[cfg(test)]
fn prefilter(block: &[u8], lod: Option<(&[f32; 35], usize)>, hide: Option<&[u32]>, cam: [f32; 3], out: &mut Vec<f32>) -> (u32, u32, u32) {
    prefilter_why(block, lod, hide, cam, out, None, None, &mut [0; 3])
}

/// [`prefilter`] that can also note, per instance, (index, outcome, distance,
/// slot): outcome 0 drawn, 1 the title's hide mask, 2 scale field <= 0, 3 past
/// the far fade, 4 another LOD slot.
/// A batch's absolute bounding sphere: the instances' LOD-record spheres
/// (centre M_i * T+0x08, radius T+0x14 / min scale) inside one sphere around
/// their bounding box. None without a usable instance.
fn batch_bounds(block: &[u8], t: &[f32; 35]) -> Option<([f32; 3], f32)> {
    let (mut lo, mut hi, mut rmax, mut any) = ([f32::MAX; 3], [f32::MIN; 3], 0f32, false);
    let c = [t[2], t[3], t[4]];
    for rb in block.chunks_exact(64).take(MAX_BATCH as usize) {
        let r = f32s::<16>(rb);
        let s = r[12].min(r[14]);
        if !(s > 0.0) {
            continue;
        }
        let p: [f32; 3] = std::array::from_fn(|k| r[4 * k] * c[0] + r[4 * k + 1] * c[1] + r[4 * k + 2] * c[2] + r[4 * k + 3]);
        if !p.iter().all(|v| v.is_finite()) {
            continue;
        }
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
        rmax = rmax.max(t[5] / s);
        any = true;
    }
    any.then(|| {
        let ctr: [f32; 3] = std::array::from_fn(|k| (lo[k] + hi[k]) * 0.5);
        let half = ((0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f32>()).sqrt() * 0.5;
        (ctr, half + rmax)
    })
}

/// Batch bounds by (records, length, the LOD record's sphere): the
/// records of kind-S batches are static, so this is computed once.
type BoundsCache = HashMap<(usize, usize, [u32; 4]), Option<([f32; 3], f32)>>;
static BOUNDS: Mutex<Option<BoundsCache>> = Mutex::new(None);

fn cached_bounds(block: &std::sync::Arc<Vec<u8>>, t: &[f32; 35]) -> Option<([f32; 3], f32)> {
    let key = (std::sync::Arc::as_ptr(block) as usize, block.len(), [t[2].to_bits(), t[3].to_bits(), t[4].to_bits(), t[5].to_bits()]);
    let mut g = BOUNDS.lock().ok()?;
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() > 65_536 {
        m.clear();
    }
    *m.entry(key).or_insert_with(|| batch_bounds(block, t))
}

/// The batch first, then its instances: a batch sphere outside a plane
/// is counted outside whole; one inside every plane skips the per-instance
/// frustum test; a batch whose distance band [d - R, d + R] maps wholly to
/// another LOD slot (or past the far fade) is counted wrong-LOD whole (the slot
/// grows with distance). Returns (tested, drawn, hidden) as
/// [`prefilter_why`], with `counts` [outside, wrong LOD].
#[allow(clippy::too_many_arguments)]
fn prefilter_batch(block: &std::sync::Arc<Vec<u8>>, len: usize, lod: Option<(&[f32; 35], usize)>, hide: Option<&[u32]>, cam: [f32; 3], out: &mut Vec<f32>, why: Option<&mut Vec<(u32, u8, f32, u8)>>, planes: &[[f32; 4]; 5], counts: &mut [u32; 3]) -> (u32, u32, u32) {
    let n = (len / 64).min(MAX_BATCH as usize) as u32;
    match batch_test(block, n, lod, cam, planes, counts) {
        None => (n, 0, 0),
        Some(need_planes) => prefilter_why(&block[..len], lod, hide, cam, out, why, need_planes.then_some(planes), counts),
    }
}

/// The batch-level half of [`prefilter_batch`]: None when the whole batch
/// is outside or at another LOD slot (counted in `counts`), else whether its
/// instances still need the frustum test (false: the batch sphere is inside
/// every plane). The device cull (`gpucull=on`) keeps this on the CPU.
fn batch_test(block: &std::sync::Arc<Vec<u8>>, n: u32, lod: Option<(&[f32; 35], usize)>, cam: [f32; 3], planes: &[[f32; 4]; 5], counts: &mut [u32; 3]) -> Option<bool> {
    let mut need_planes = true;
    if let Some((t, slot)) = lod
        && let Some((c, rad)) = cached_bounds(block, t)
    {
        let p = [c[0] - cam[0], c[1] - cam[1], c[2] - cam[2]];
        let dist: Vec<f32> = planes.iter().map(|pl| pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3]).collect();
        if dist.iter().any(|d| *d < -rad) {
            counts[0] += n;
            return None;
        }
        if dist.iter().all(|d| *d >= rad) {
            need_planes = false;
        }
        let d = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let unit = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let (near, far) = (lod_slot(t, &unit, (d - rad).max(0.0)), lod_slot(t, &unit, d + rad));
        let skip = match (near, far) {
            (None, _) => true,
            (Some(a), Some(b)) => a > slot || b < slot,
            (Some(a), None) => a > slot,
        };
        if skip {
            counts[1] += n;
            return None;
        }
    }
    Some(need_planes)
}

/// With `planes`, outcome 5 = outside the frustum (tested before the
/// LOD); `counts` gains [outside the frustum, another LOD slot].
#[allow(clippy::too_many_arguments)]
fn prefilter_why(block: &[u8], lod: Option<(&[f32; 35], usize)>, hide: Option<&[u32]>, cam: [f32; 3], out: &mut Vec<f32>, mut why: Option<&mut Vec<(u32, u8, f32, u8)>>, planes: Option<&[[f32; 4]; 5]>, counts: &mut [u32; 3]) -> (u32, u32, u32) {
    let (mut tested, mut drawn, mut hidden) = (0, 0, 0);
    for (i, rb) in block.chunks_exact(64).take(MAX_BATCH as usize).enumerate() {
        let r = f32s::<16>(rb);
        tested += 1;
        let v = [r[3] - cam[0], r[7] - cam[1], r[11] - cam[2]];
        let d = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let mut note = |code: u8, k: u8| {
            if let Some(w) = why.as_deref_mut() {
                w.push((i as u32, code, d, k));
            }
        };
        if hide.is_some_and(|h| h.get(i >> 5).is_some_and(|w| w & (1 << (i & 31)) != 0)) {
            hidden += 1;
            note(1, 0);
            continue;
        }
        if !plausible_record(&r) {
            counts[2] += 1;
            note(6, 0);
            continue;
        }
        // The frustum before the LOD test (BatchCull's order).
        if let (Some((t, _)), Some(pl)) = (lod, planes)
            && !in_frustum(t, &r, cam, pl)
        {
            counts[0] += 1;
            note(5, 0);
            continue;
        }
        if let Some((t, slot)) = lod {
            // Translations (dwords 3, 7, 11) are absolute [I strong].
            match lod_slot(t, &r, d) {
                Some(k) if k == slot => {}
                Some(k) => {
                    counts[1] += 1;
                    note(4, k as u8);
                    continue;
                }
                None => {
                    note(if r[12].min(r[14]) <= 0.0 { 2 } else { 3 }, 0);
                    continue;
                }
            }
        }
        note(0, lod.map_or(0, |l| l.1 as u8));
        out.extend(xnw_world::instance_world(&r, cam));
        drawn += 1;
    }
    (tested, drawn, hidden)
}

/// What the light capture keeps per geometry (by pointer and the hash
/// at geom+0x30): its buffers' addresses and sizes.
/// Per model pointer, whether it is skinned (model+0x35 bit 0).

/// The static maps' sizes (GEOMS, MODELS, BONES, DUMPED), each cleared
/// once past [`MAP_CAP`] entries (they only grew).
const MAP_CAP: usize = 1 << 18;
fn bound_maps() -> [usize; 4] {
    fn cap<K, V>(m: &Mutex<Option<HashMap<K, V>>>) -> usize {
        let Ok(mut g) = m.lock() else { return 0 };
        let n = g.as_ref().map_or(0, HashMap::len);
        if n > MAP_CAP {
            *g = None;
        }
        n
    }
    let dumped = DUMPED.lock().map_or(0, |g| g.as_ref().map_or(0, |s| s.len()));
    // GEOMS and MODELS are per recording thread now, each cleared past MAP_CAP
    // on its own thread (capture_light); their sizes are not visible here.
    [0, 0, cap(&BONES), dumped]
}

/// A read of memory the title is using for this very draw (its TLS block,
/// render context, the geometry it was called with, the transform it is
/// about to bind, its own image): no syscall (the guarded reads cost
/// about 1 us each, 10-12 a draw). Only the shape of the address
/// is checked.
///
/// # Safety
/// `addr` is readable for `T`'s size: memory of the draw being made.
unsafe fn dread<T: Copy>(addr: u64) -> Option<T> {
    if !(0x1_0000..0x8000_0000_0000).contains(&addr) {
        return None;
    }
    let mut v = std::mem::MaybeUninit::<T>::uninit();
    // SAFETY: a fault-fixed copy into our own local (fcopy); T is plain data.
    (unsafe { fcopy(v.as_mut_ptr().cast(), addr as *const u8, std::mem::size_of::<T>()) }).then(|| unsafe { v.assume_init() })
}

/// A plain read, only for memory that is always there: the calling
/// thread's TLS block (our loader's) and the title image's globals.
///
/// # Safety
/// `addr` is in the calling thread's TLS block or the title image.
unsafe fn pread<T: Copy>(addr: u64) -> Option<T> {
    // SAFETY: the caller's.
    (0x1_0000..0x8000_0000_0000).contains(&addr).then(|| unsafe { (addr as *const T).read_unaligned() })
}

/// `len` bytes at `addr` by the fault-fixed copy (None: out of range or faulted).
fn dcopy(addr: u64, len: usize) -> Option<Vec<u8>> {
    if burst::serving() {
        // SAFETY: into the table's own buffer of `to.len()` bytes.
        return burst::read(addr, len, |to, from| unsafe { fcopy(to.as_mut_ptr(), from, to.len()) });
    }
    if !(0x1_0000..0x8000_0000_0000).contains(&addr) || addr.checked_add(len as u64)? > 0x8000_0000_0000 {
        return None;
    }
    // Not zeroed first: the copy writes every byte (a second pass over
    // 47 MiB of a cut frame's geometry otherwise).
    let mut v = Vec::<u8>::with_capacity(len);
    // SAFETY: into our own buffer's `len` bytes of capacity; the length is
    // set only after the copy wrote them all.
    unsafe { fcopy(v.as_mut_ptr(), addr as *const u8, len) }.then(|| {
        // SAFETY: as above.
        unsafe { v.set_len(len) };
        v
    })
}

// The hook's reads of title objects (direct, no syscall) with a fault
// fixup: `xnw_wp_copy(dst, src, len)` is one `rep movsb`; an access violation
// at that instruction is turned by our vectored handler into a return of 0
// (`xnw_wp_copy_fault`), counted in HOOK_FAULTS, and never reaches the
// title's handlers. Other faults pass on untouched.
std::arch::global_asm!(
    ".globl xnw_wp_copy",
    ".globl xnw_wp_copy_insn",
    ".globl xnw_wp_copy_fault",
    "xnw_wp_copy:",
    "push rdi",
    "push rsi",
    "mov rdi, rcx",
    "mov rsi, rdx",
    "mov rcx, r8",
    "xnw_wp_copy_insn:",
    "rep movsb",
    "mov eax, 1",
    "2:",
    "pop rsi",
    "pop rdi",
    "ret",
    "xnw_wp_copy_fault:",
    "xor eax, eax",
    "jmp 2b",
);

unsafe extern "system" {
    fn xnw_wp_copy(dst: *mut u8, src: *const u8, len: usize) -> u32;
    fn xnw_wp_copy_insn();
    fn xnw_wp_copy_fault();
}

/// 0x144E351B0(skin, comp, lod): binds the remap table (the buffer object
/// at skin+0x20) and writes rec+0x74/+0x78 = u16 at *(skin+0x08) / *(skin+0x18)
/// + 8*comp + 2*lod [V pseudo]. Hooked to learn, per thread, which skin the
/// following ped draws use.
const PED_SKIN_BIND: usize = 0x1_44E3_51B0;
const PED_SKIN_BIND_ENTRY: [u8; 14] = [0x41, 0x56, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83, 0xEC, 0x20, 0x48, 0x8B, 0x59, 0x20];

thread_local! {
    /// This thread's last ped skin bind: (skin, table object, comp, lod, cur base, prev base).
    static PED_SKIN: std::cell::Cell<(u64, u64, u32, u32, u32, u32)> = const { std::cell::Cell::new((0, 0, 0, 0, u32::MAX, u32::MAX)) };
}

/// The render thread's draw-list walk [V]:
/// drawlist_walk_handlers 0x140139BB0: 32-B entries at list+0x28 (+0 entity,
/// +8 handler, +0x18 f32, +0x1E u16 LOD bits); per entry
/// `handler->vt[+0x30](handler, params)` draws at once on this thread. The
/// entry is in r15/r14 only (`[r15+28h] + r14`): the call-site hook's
/// prelude hands it over in r11.
const LIST_WALK: usize = 0x1_4013_9BB0;
const LIST_WALK_ENTRY: [u8; 14] = [0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x81];
/// `mov rdx,[rsp+78h]; mov [rsi+34h],eax; mov rax,[r13+0]; call [rax+30h]`
/// (no branch lands inside; r11 dead).
const LIST_CALL: usize = 0x1_4013_9ECA;
const LIST_CALL_BYTES: [u8; 15] = [0x48, 0x8B, 0x54, 0x24, 0x78, 0x89, 0x46, 0x34, 0x49, 0x8B, 0x45, 0x00, 0xFF, 0x50, 0x30];
/// `mov r11,[r15+28h]; add r11,r14`: the entry's address.
const LIST_CALL_PRELUDE: [u8; 7] = [0x4D, 0x8B, 0x5F, 0x28, 0x4D, 0x01, 0xF3];

thread_local! {
    /// The draw-list entry being drawn on this thread (0: none).
    static LIST_ENTRY: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

extern "system" fn on_list_walk(_frame: usize, _id: usize) {
    LIST_ENTRY.with(|c| c.set(0));
}

extern "system" fn on_list_call(frame: usize, _id: usize) {
    // SAFETY: the stub's frame: r11 (the prelude's entry address).
    let entry = unsafe { ((frame - 64) as *const u64).read() };
    LIST_ENTRY.with(|c| c.set(entry));
}

/// A draw's entity, from its draw-list entry and the entity
/// [V fields, I names].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Entity {
    at: u64,
    /// u8 +0x30, (u8 +0xB4) & 0x3E (8 = kind D), u8 +0x90 fade, u16
    /// +0x96 & 0xF lodType, u8 +0x95 (bit0 ready), u32 +0xA4 map ref.
    kind: u8,
    etype: u8,
    fade: u8,
    lod_type: u8,
    flags: u8,
    map_ref: u32,
    /// The entry's +0x18 (f32, passed to the list's callback) and +0x1E (LOD bits).
    entry_f: f32,
    entry_lod: u16,
}

const ENTITY_BYTES: usize = 0xB8;

fn entity_of(entry: &[u8; 32], e: &[u8]) -> Entity {
    let at = le64(entry, 0);
    let b = |k: usize| e.get(k).copied().unwrap_or(0);
    Entity {
        at,
        kind: b(0x30),
        etype: b(0xB4) & 0x3E,
        fade: b(0x90),
        lod_type: b(0x96) & 0xF,
        flags: b(0x95),
        map_ref: if e.len() >= 0xA8 { le32(e, 0xA4) } else { 0 },
        entry_f: f32::from_bits(le32(entry, 0x18)),
        entry_lod: le16(entry, 0x1E),
    }
}

/// The entity of the entry this thread is drawing (fault-fixed reads).
fn current_entity() -> Entity {
    let entry = LIST_ENTRY.with(std::cell::Cell::get);
    if entry == 0 {
        return Entity::default();
    }
    // SAFETY: the title's draw-list entry and entity (fault-fixed reads).
    let Some(en) = (unsafe { dread::<[u8; 32]>(entry) }) else { return Entity::default() };
    let e = unsafe { dread::<[u8; ENTITY_BYTES]>(le64(&en, 0)) };
    entity_of(&en, e.as_ref().map_or(&[][..], |e| &e[..]))
}

extern "system" fn on_ped_skin(frame: usize, _id: usize) {
    // SAFETY: the stub's frame: rcx, rdx, r8.
    let (skin, comp, lod) = unsafe { (((frame - 24) as *const u64).read(), ((frame - 32) as *const u64).read() as u32, ((frame - 40) as *const u64).read() as u8 as u32) };
    PED_SKIN.with(|c| c.set(ped_skin_read(skin, comp, lod)));
}

/// The skin record's fields as 0x144E351B0 reads them (fault-fixed reads).
fn ped_skin_read(skin: u64, comp: u32, lod: u32) -> (u64, u64, u32, u32, u32, u32) {
    // SAFETY: the skin object the title is binding (fault-fixed reads).
    unsafe {
        let Some(h) = dread::<[u64; 5]>(skin) else { return (skin, 0, comp, lod, u32::MAX, u32::MAX) };
        let at = |table: u64| dread::<u16>(table + 8 * u64::from(comp) + 2 * u64::from(lod)).map_or(u32::MAX, u32::from);
        (skin, h[4], comp, lod, at(h[1]), at(h[3]))
    }
}

/// [V] The title's frame id (dword_14ABCE2A0, snapshot taken when frame
/// N's queue opens; constant for the whole drain), the ring drain on the
/// render thread (0x1443C38A0: one call drains exactly one frame) and the
/// render-side flip right after it (0x1423CF3E0, return 0x1423CF504 from
/// the drain's caller): the frame boundary our lists must follow.
const TITLE_FRAME: usize = 0x1_4ABC_E2A0;
/// Per-context view blocks (b0), 480 B each [V].
const VIEW_BLOCKS: usize = 0x1_475E_9580;
const DRAIN: usize = 0x1_443C_38A0;
const DRAIN_ENTRY: [u8; 14] = [0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x83];
const DRAIN_FLIP: usize = 0x1_423C_F3E0;
const DRAIN_FLIP_ENTRY: [u8; 14] = [0xB8, 0x01, 0x00, 0x00, 0x00, 0x2B, 0x41, 0x5C, 0x89, 0x41, 0x5C, 0xC3, 0xCC, 0xCC];

/// Frame markers since the last dump: (enter 1 / flip 0, title frame id,
/// presented(), thread, return address, microseconds since start).
static DRAINS: Mutex<Vec<(u8, u32, u64, u32, u64, u64, [f32; 4], [u64; 3])>> = Mutex::new(Vec::new());
/// Caps: scan record cursor qword_14957F170 (array full at 47,500),
/// occluder count dword_147531228 (at most 8), aux records dword_14940BFE0
/// (at most 5,500), read at each drain mark.
const SCAN_RECORDS: usize = 0x1_4957_F170;
/// Records in the scan array: qword_14957F170 is the append cursor, a
/// pointer into the 32-B record array 0x14940BFF0: `(cursor -
/// 0x14940BFF0) >> 5` [V], not a count. Until then drains.tsv printed the
/// pointer and scan.txt always took 47,500 records (stale ones included).
fn scan_count(title: usize, cursor: u64) -> u64 {
    let start = real(title, SCAN_ARRAY);
    if cursor < start || cursor > start + 32 * 47_500 { 0 } else { (cursor - start) / 32 }
}
const SCAN_OCCLUDERS: usize = 0x1_4753_1228;
const SCAN_AUX: usize = 0x1_4940_BFE0;
/// The game clock block (timer update 0x1422264D0 [V]): +0 step, +4
/// 1/step, +8 / +0xC game milliseconds * 0.001 (two clocks).
const GAME_CLOCK: usize = 0x1_4ABC_E278;
static T_START: OnceLock<Instant> = OnceLock::new();

fn title_frame() -> u32 {
    let Some(s) = STATE.get() else { return 0 };
    // SAFETY: a title image global.
    unsafe { pread::<u32>(real(s.title, TITLE_FRAME)) }.unwrap_or(0)
}

fn drain_mark(kind: u8, frame: usize) {
    // SAFETY: the stub's frame: the return address.
    let ret = unsafe { ((frame + 8) as *const u64).read() };
    let us = T_START.get_or_init(Instant::now).elapsed().as_micros() as u64;
    // SAFETY: a plain query of the calling thread.
    let tid = unsafe { xnw_core::win::GetCurrentThreadId() };
    if let Ok(mut d) = DRAINS.lock()
        && d.len() < 100_000
    {
        // SAFETY: a title image global.
        let clock = STATE.get().and_then(|s| unsafe { pread::<[f32; 4]>(real(s.title, GAME_CLOCK)) }).unwrap_or([0.0; 4]);
        // SAFETY: title image globals.
        let counts = STATE.get().map_or([0; 3], |s| unsafe { [scan_count(s.title, pread::<u64>(real(s.title, SCAN_RECORDS)).unwrap_or(0)), u64::from(pread::<u32>(real(s.title, SCAN_OCCLUDERS)).unwrap_or(0)), u64::from(pread::<u32>(real(s.title, SCAN_AUX)).unwrap_or(0))] });
        d.push((kind, title_frame(), xnw_gpu::d3d12x::presented(), tid, ret, us, clock, counts));
    }
}

extern "system" fn on_drain(frame: usize, _id: usize) {
    drain_mark(1, frame);
    if !COMPACT_HOOKED.load(Ordering::Relaxed) {
        scan_snapshot();
    }
}

/// `scan=exit`: the compaction's wrapper (sub rsp,28h; call 0x14547BE70;
/// mov al,1; add rsp,28h; ret [V exe bytes]) and the compaction.
const COMPACT_WRAPPER: usize = 0x1_40B5_F960;
const COMPACT_WRAPPER_BYTES: [u8; 16] = [0x48, 0x83, 0xEC, 0x28, 0xE8, 0x07, 0xC5, 0x91, 0x04, 0xB0, 0x01, 0x48, 0x83, 0xC4, 0x28, 0xC3];
const COMPACT: usize = 0x1_4547_BE70;
static COMPACT_AT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static COMPACT_HOOKED: AtomicBool = AtomicBool::new(false);

/// In place of the wrapper: the compaction, then the scan snapshot of
/// the records it left; returns the wrapper's 1 (its caller gets it in al).
extern "system" fn on_compaction(_frame: usize, _id: usize) -> u64 {
    let at = COMPACT_AT.load(Ordering::Relaxed);
    // SAFETY: the title's compaction, which takes no arguments (the
    // wrapper sets none [V exe bytes]; the caller takes none [I]).
    let compact = unsafe { std::mem::transmute::<usize, extern "system" fn()>(at) };
    compact();
    scan_snapshot();
    1
}

/// The scene scan's visible records (32 B each at 0x14940BFF0, count
/// qword_14957F170 [V]) as entity pointers, taken at drain entry of
/// each title frame the seq writer will dump (it runs far behind: kept until
/// written, at most 512 frames) for scan.txt.
const SCAN_ARRAY: usize = 0x1_4940_BFF0;
static SCANS: Mutex<std::collections::VecDeque<(u32, Vec<u64>)>> = Mutex::new(std::collections::VecDeque::new());

fn scan_snapshot() {
    let Some(s) = STATE.get().filter(|s| s.cfg.seq) else { return };
    // Only frames the seq writer will dump (it runs far behind the title).
    let f = title_frame();
    if !s.cfg.at.contains(&u64::from(f)) && s.cfg.auto.is_none() {
        return;
    }
    // SAFETY: title image globals.
    let n = scan_count(s.title, unsafe { pread::<u64>(real(s.title, SCAN_RECORDS)) }.unwrap_or(0)) as usize;
    let Some(b) = dcopy(real(s.title, SCAN_ARRAY), 32 * n) else { return };
    // (entity, its draw handler entity+0x28 [V]) per visible record.
    let ents: Vec<u64> = b
        .chunks_exact(32)
        .flat_map(|r| {
            let e = le64(r, 0);
            // SAFETY: a scan record's entity (fault-fixed).
            [e, unsafe { dread::<u64>(e.wrapping_add(0x28)) }.unwrap_or(0)]
        })
        .collect();
    if let Ok(mut q) = SCANS.lock() {
        // Several scans of one frame id: the last one stands.
        q.retain(|(g, _)| *g != f);
        if q.len() >= 512 {
            q.pop_front();
        }
        q.push_back((f, ents));
    }
}

extern "system" fn on_drain_flip(frame: usize, _id: usize) {
    drain_mark(0, frame);
    // The flip right after the drain (return 0x1423CF504; the other caller,
    // the loading path, has no drain) closes the drained frame's list.
    let Some(s) = STATE.get() else { return };
    // SAFETY: the stub's frame: the return address.
    let ret = unsafe { ((frame + 8) as *const u64).read() };
    if DRAIN_CUT.load(Ordering::Relaxed) && ret == real(s.title, DRAIN_RETURN) {
        s.flush_drain(u64::from(title_frame()));
    }
}

/// The drain's caller returns here (the flip call after the drain).
const DRAIN_RETURN: usize = 0x1_423C_F504;

/// Drain cut on: lists are keyed by the title's frame id and closed when its
/// drain returns (both drain hooks in, `cut=present` not asked).
static DRAIN_CUT: AtomicBool = AtomicBool::new(false);

/// The frame a draw belongs to: the title's frame id under the drain cut,
/// else our presented frame (the old cut).
fn frame_key() -> u64 {
    if burst::serving() {
        return burst::frame();
    }
    if DRAIN_CUT.load(Ordering::Relaxed) { u64::from(title_frame()) } else { xnw_gpu::d3d12x::presented() }
}

/// `skinrules` bit 8: gives an instanced draw's reads to this thread's
/// last recorded draw, when that is the draw the sink belongs to (this
/// frame's, and no hook since). False: no such draw.
fn attach_wheel(w: skinhook::WheelInst) -> bool {
    if burst::serving() {
        return burst::attach(|d| d.skinned.then(|| d.wheel = Some(w)).is_some());
    }
    let Some(s) = STATE.get() else { return false };
    let slot = s.slot();
    let Ok(mut g) = slot.lock() else { return false };
    let now = frame_key();
    match g.as_mut() {
        Some(c) if c.frame == now => match c.draws.last_mut() {
            Some(d) if d.hook_n == c.seen && d.skinned => {
                d.wheel = Some(Box::new(w));
                true
            }
            _ => false,
        },
        _ => false,
    }
}

/// `bridgeps`: this thread's last recorded draw's vertex stream and
/// meshlet buffer (vb_at, vb_len, mesh_at), when it is the draw a sink
/// belongs to (this frame's, and no hook since).
fn last_draw_streams() -> Option<(u64, u64, u64)> {
    if burst::serving() {
        return burst::current().map(|d| d.streams);
    }
    let s = STATE.get()?;
    let slot = s.slot();
    let g = slot.lock().ok()?;
    let c = g.as_ref().filter(|c| c.frame == frame_key())?;
    c.draws.last().filter(|d| d.hook_n == c.seen).map(|d| (d.vb_at, d.vb_len, d.mesh_at))
}

/// `bridgeps`: gives a draw's binding inputs to this thread's last
/// recorded draw (as [`attach_sink_range`]). False: no such draw.
fn attach_bridged(o: Box<xnw_world::bridged::Owned>) -> bool {
    if burst::serving() {
        return burst::attach(|d| {
            d.bridged = Some(o);
            true
        });
    }
    let Some(s) = STATE.get() else { return false };
    let slot = s.slot();
    let Ok(mut g) = slot.lock() else { return false };
    let now = frame_key();
    match g.as_mut() {
        Some(c) if c.frame == now => match c.draws.last_mut() {
            Some(d) if d.hook_n == c.seen => {
                d.bridged = Some(o);
                true
            }
            _ => false,
        },
        _ => false,
    }
}

/// `preskin=on`: gives a PreSkinD draw's own range word (its sink record
/// +0x54) to this thread's last recorded draw, when that is the draw the
/// sink belongs to (this frame's, and no hook since). False: no such draw.
fn attach_sink_range(word: u32) -> bool {
    if burst::serving() {
        return burst::attach(|d| {
            d.sink_range = Some(word);
            true
        });
    }
    let Some(s) = STATE.get() else { return false };
    let slot = s.slot();
    let Ok(mut g) = slot.lock() else { return false };
    let now = frame_key();
    match g.as_mut() {
        Some(c) if c.frame == now => match c.draws.last_mut() {
            Some(d) if d.hook_n == c.seen => {
                d.sink_range = Some(word);
                true
            }
            _ => false,
        },
        _ => false,
    }
}

/// The list sink's draw consumers (`tap::set_draw_tap` takes one): the
/// instanced wheels (`skinrules` bit 8), the PreSkinD records (`preskin=on`)
/// and the bridged rows (`bridgeps`).
static SINK_WHEELS: AtomicBool = AtomicBool::new(false);
static SINK_PRESKIN: AtomicBool = AtomicBool::new(false);
static SINK_BRIDGE: AtomicBool = AtomicBool::new(false);

/// Registers the list sink's consumers the switches ask for (the live
/// probe, and `xnw burst`, which hands them a capture's sinks).
pub(crate) fn install_sinks(render: bool) {
    if preskin::on() {
        preskin::install();
        SINK_PRESKIN.store(true, Ordering::Relaxed);
        xnw_core::info!("world render: pre-skin: PreSkinD draws joined by their own record (+0x54 at T9 slot 0, read at the list sink)");
    }
    if skinhook::rules() & skinhook::VEH_INST != 0 && render {
        SINK_WHEELS.store(true, Ordering::Relaxed);
        xnw_core::info!("world render: vehicle wheels: instanced wheel and tyre draws read at the list sink (skinrules bit 8)");
    }
    if bridgeps::on() && render {
        SINK_BRIDGE.store(true, Ordering::Relaxed);
        xnw_core::info!("world render: bridgeps: draws of pixel programs {:08x?} read at the list sink for the title's own programs in our pass", bridgeps::wanted());
    }
    if SINK_PRESKIN.load(Ordering::Relaxed) || SINK_WHEELS.load(Ordering::Relaxed) || SINK_BRIDGE.load(Ordering::Relaxed) {
        xnw_gpu::d3d12x::tap::set_draw_tap(Some((sink_want, on_sink)));
    }
}

/// Which consumers are on: (wheels, pre-skin, bridged rows).
pub(crate) fn sinks_on() -> (bool, bool, bool) {
    (SINK_WHEELS.load(Ordering::Relaxed), SINK_PRESKIN.load(Ordering::Relaxed), SINK_BRIDGE.load(Ordering::Relaxed))
}

fn sink_want(pipeline: usize) -> bool {
    (SINK_WHEELS.load(Ordering::Relaxed) && skinhook::wheel_want(pipeline)) || (SINK_PRESKIN.load(Ordering::Relaxed) && preskin::draw_want(pipeline)) || (SINK_BRIDGE.load(Ordering::Relaxed) && bridgeps::draw_want(pipeline))
}

fn on_sink(d: &xnw_gpu::d3d12x::tap::TapDraw) {
    if SINK_WHEELS.load(Ordering::Relaxed) && skinhook::wheel_want(d.pipeline) {
        skinhook::on_wheel_sink(d);
    } else if SINK_PRESKIN.load(Ordering::Relaxed) && preskin::draw_want(d.pipeline) {
        preskin::on_draw_sink(d);
    }
    if SINK_BRIDGE.load(Ordering::Relaxed) && bridgeps::draw_want(d.pipeline) {
        bridgeps::on_draw_sink(d);
    }
}

/// A row's placement fields as the wheel placement reads and sets them.
pub(crate) struct WheelRow {
    pub world: xnw_world::Mat,
    pub palette: u32,
    pub bone_first: u32,
    pub bone_count: u32,
    pub remap: u32,
    pub inst_first: u32,
    pub inst_count: u32,
}

/// Instanced vehicle draws (skinrules bit 8): one instance per record read
/// at the sink, the bones at the draw's palette base after the palette,
/// reached through a remap relative to the row's first bone. `left`: rows
/// with reads, of them left as they were (a remap already, no palette or
/// instances, palette too far); `rows`: rows drawn per instance. The live
/// list build and `xnw burst` both place a row's wheel here.
pub(crate) fn place_wheel(list: &mut xnw_world::FrameList, wheel_pal: &mut Option<u32>, w: &skinhook::WheelInst, r: &mut WheelRow, left: &mut [u32; 4], rows: &mut u32) {
    left[0] += 1;
    if r.remap != xnw_world::NO_REMAP {
        left[1] += 1;
    } else if w.worlds.is_empty() || (r.palette as usize) >= list.palettes.len() {
        left[2] += 1;
    }
    // The wheel bones go into the wheels' own palette and the row's bones
    // start at them (remap entry b = b), so `palette too far` (left[3])
    // stays 0. Appended to the row's palette they lay past the 12-bit
    // remap reach (0x1000 bones from the row's first bone): frame 2760 of a capture
    // `palette too far 8` of 8, behind the frame's 3215 skin bones and 11
    // vehicle index tables (352 KiB, about 7500 bones); the replay's
    // palette holds no vehicle tables, so its wheels fit.
    if r.remap == xnw_world::NO_REMAP && !w.worlds.is_empty() && (r.palette as usize) < list.palettes.len() {
        let p = *wheel_pal.get_or_insert_with(|| {
            list.palette_at_draw.resize(list.palettes.len(), false);
            list.palettes.push(Vec::new());
            list.palette_at_draw.push(true);
            (list.palettes.len() - 1) as u32
        });
        let (pal, n) = (&mut list.palettes[p as usize], skinhook::WHEEL_BONES as u32);
        let e = (pal.len() / 12) as u32;
        pal.extend_from_slice(&w.bones);
        r.remap = pal.len() as u32;
        pal.extend((0..n).map(f32::from_bits));
        pal.resize(pal.len().next_multiple_of(12), 0.0);
        (r.palette, r.bone_first, r.bone_count) = (p, e, n);
        r.inst_first = (list.instances.len() / 12) as u32;
        for x in &w.worlds {
            list.instances.extend_from_slice(x);
        }
        r.inst_count = w.worlds.len() as u32;
        r.world = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        *rows += 1;
    }
}

/// Faults the copy turned into a skipped read.
static HOOK_FAULTS: AtomicU64 = AtomicU64::new(0);

/// The reader whose copy faulted (the thread's site at the fault), and the
/// last faulting address: says which reads fault and when.
pub(crate) const FAULT_SITES: [&str; 9] = ["draw hook", "bones and masks", "vehicle wheels", "imposters", "pre-skin streams", "pre-skin hide mask", "pre-skin delta", "pre-skin face", "bridged rows"];
pub(crate) const SITE_BONES: u8 = 1;
pub(crate) const SITE_WHEELS: u8 = 2;
pub(crate) const SITE_IMPOSTERS: u8 = 3;
pub(crate) const SITE_PRESKIN: u8 = 4;
/// The pre-skin's tolerant reads (a fault reads as 0): the mask pick, its record, the mask.
pub(crate) const SITE_PRESKIN_MASK: u8 = 5;
/// Variant 2's t76 delta table (`preskinbits` bit 2).
pub(crate) const SITE_PRESKIN_DELTA: u8 = 6;
/// Variant 11's face table header.
pub(crate) const SITE_PRESKIN_FACE: u8 = 7;
/// The bridged rows' reads (`bridgeps`).
pub(crate) const SITE_BRIDGE: u8 = 8;
static FAULTS_BY: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
static FAULT_LAST: AtomicU64 = AtomicU64::new(0);
thread_local! {
    static FAULT_SITE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// The thread's reads are `site`'s until the guard drops (the one before
/// comes back).
pub(crate) struct FaultSite(u8);

pub(crate) fn fault_site(site: u8) -> FaultSite {
    FaultSite(FAULT_SITE.with(|c| c.replace(site)))
}

/// [`fault_site`] unless a site is already set (a reader's general read
/// inside a named part keeps the part's name).
pub(crate) fn fault_site_or_kept(site: u8) -> FaultSite {
    FaultSite(FAULT_SITE.with(|c| {
        let was = c.get();
        if was == 0 {
            c.set(site);
        }
        was
    }))
}

impl Drop for FaultSite {
    fn drop(&mut self) {
        FAULT_SITE.with(|c| c.set(self.0));
    }
}

/// `hook faults N (draw hook a, ...; last at 0x..)`, readers with a count only.
fn faults_text() -> String {
    let n = HOOK_FAULTS.load(Ordering::Relaxed);
    let by: Vec<String> = FAULT_SITES.iter().zip(&FAULTS_BY).map(|(w, a)| (w, a.load(Ordering::Relaxed))).filter(|(_, v)| *v > 0).map(|(w, v)| format!("{w} {v}")).collect();
    if by.is_empty() { format!("hook faults {n}") } else { format!("hook faults {n} ({}; last at {:#x})", by.join(", "), FAULT_LAST.load(Ordering::Relaxed)) }
}

/// # Safety
/// `dst` holds `len` bytes.
unsafe fn fcopy(dst: *mut u8, src: *const u8, len: usize) -> bool {
    FAULT_FIXUP.call_once(install_fixup);
    // SAFETY: the caller's; a fault in the copy is fixed up (install_fixup).
    unsafe { xnw_wp_copy(dst, src, len) == 1 }
}

static FAULT_FIXUP: std::sync::Once = std::sync::Once::new();

fn install_fixup() {
    unsafe extern "system" fn on_fault(info: *mut xnw_core::win::ExceptionPointers) -> i32 {
        use xnw_core::win::{Context, EXCEPTION_ACCESS_VIOLATION, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH};
        // SAFETY: the system passes the exception being dispatched.
        let (r, ctx) = unsafe { (&*(*info).record, &mut *(*info).context) };
        if r.code != EXCEPTION_ACCESS_VIOLATION || ctx.rip() != xnw_wp_copy_insn as usize {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        ctx.0[Context::RIP..Context::RIP + 8].copy_from_slice(&(xnw_wp_copy_fault as usize as u64).to_le_bytes());
        HOOK_FAULTS.fetch_add(1, Ordering::Relaxed);
        // A const thread-local without a destructor: a plain read on the faulting thread.
        let site = FAULT_SITE.try_with(std::cell::Cell::get).unwrap_or(0);
        FAULTS_BY[usize::from(site).min(FAULTS_BY.len() - 1)].fetch_add(1, Ordering::Relaxed);
        FAULT_LAST.store(if r.param_count >= 2 { r.params[1] as u64 } else { 0 }, Ordering::Relaxed);
        EXCEPTION_CONTINUE_EXECUTION
    }
    // SAFETY: the handler touches the context of our own copy only.
    unsafe { xnw_core::win::AddVectoredExceptionHandler(1, on_fault) };
}

/// `XNW_WORLD_RENDER`: a draw recorded cheaply. Direct reads of the
/// draw's own state (TLS block, context, geometry header, transform); a
/// guarded read only for a geometry's buffer objects (once per geometry
/// and hash), the skinned bit (once per model), and the skin palette (once
/// per palette pointer, drawable and bone count in a frame). An earlier version copied
/// a palette once per pointer per frame, but one pointer served several
/// objects in a frame.
/// The title's TLS block of the calling thread (0 when unreadable).
fn title_tls(index: u64) -> u64 {
    let array: u64;
    // SAFETY: reads TEB.ThreadLocalStoragePointer of the calling thread.
    unsafe { std::arch::asm!("mov {}, gs:[0x58]", out(reg) array, options(nostack, readonly, preserves_flags)) };
    // SAFETY: the calling thread's TLS array and the title's block in it.
    unsafe { pread::<u64>(array + 8 * index) }.unwrap_or(0)
}

fn capture_light(s: &State, c: &mut Capture, geom: u64, ret: u64, tls: u64, sampled: bool) {
    let mut tp = sampled.then(Instant::now);
    let mut part = |c: &mut Capture, k: usize| {
        if let Some(t) = tp.as_mut() {
            let now = Instant::now();
            c.parts[k] += PART_SAMPLE * (now - *t).as_nanos() as u64;
            *t = now;
        }
    };
    let mut d = Draw { geom, ret, ..Default::default() };
    d.pass = PASS.with(std::cell::Cell::get);
    if tls == 0 {
        miss!(c, Tls);
        return;
    }
    // SAFETY: the title's TLS block of this thread (all fields read by the draw).
    let r64 = |o: u64| unsafe { pread::<u64>(tls + o) }.unwrap_or(0);
    d.palette = (r64(TLS_PALETTE), (r64(TLS_PALETTE_N) & 0xFFFF_FFFF) as u32);
    d.drawable = r64(TLS_DRAWABLE);
    d.model = r64(TLS_MODEL);
    // Rigid parts (fragments: bike racks, props): world' = bones[model+0x33] * world.
    if d.model != 0
        && d.palette.0 != 0
        // SAFETY: the model this draw binds (fault-fixed read).
        && let Some(mh) = unsafe { dread::<[u8; 0x38]>(d.model) }
        && le16(&mh, 0x14) == 0
        && mh[0x35] & 1 == 0
        && u32::from(mh[0x33]) < d.palette.1.min(4096)
    {
        // SAFETY: the bone the title composes this part with.
        d.rigid = unsafe { dread::<[f32; 12]>(d.palette.0 + 48 * u64::from(mh[0x33])) }.filter(|m| m.iter().all(|v| v.is_finite() && v.abs() < 1e6));
    }
    d.shader = r64(TLS_SHADER);
    d.viewport = r64(TLS_VIEWPORT);
    if d.shader != 0 && crate::worldtex::enabled() {
        (d.tex_desc, d.alpha_clip) = crate::worldtex::albedo(&crate::worldtex::globals(), d.shader);
        d.layer_desc = crate::worldtex::last_layers();
    }
    d.stage_tls = (r64(TLS_STAGE) & 0xFFFF_FFFF) as u32;
    let ctx = r64(TLS_CTX);
    if (s.cfg.seq || s.cfg.census) && d.shader != 0 {
        // SAFETY: the draw's shader object (fault-fixed).
        d.fx = unsafe { dread::<u64>(d.shader) }.unwrap_or(0);
    }
    if s.cfg.census && d.shader != 0 {
        // SAFETY: the draw's shader object and context (fault-fixed reads).
        let (fx, progs) = unsafe { (dread::<u64>(d.shader).unwrap_or(0), if ctx != 0 { dread::<u64>(ctx + CTX_PROGRAMS).unwrap_or(0) } else { 0 }) };
        let why = if crate::worldtex::enabled() { crate::worldtex::last_why() } else { 4 };
        *c.census.entry((fx, progs, d.stage_tls, why)).or_default() += 1;
    }
    // SAFETY: the geometry this draw was called with.
    let Some(g) = (unsafe { dread::<[u8; 0x48]>(geom) }) else {
        miss!(c, Geometry);
        return;
    };
    d.ghash = le64(&g, 0x30);
    let id = (geom, d.ghash);
    let cached = GEOMS.with(|m| m.borrow().get(&id).copied());
    part(c, 1);
    let key = match cached {
        Some(k) => k,
        None => {
            c.geom_misses += 1;
            let mut k = GeomKey { hash: d.ghash, ..Default::default() };
            // Direct reads: the buffer objects this draw binds (a
            // guarded read is a ReadProcessMemory syscall; 2,200 first-seen
            // geometries at world entry cost 3 of them each).
            // SAFETY: the geometry's own buffer objects, used by this draw.
            if let Some(vb) = unsafe { dread::<[u8; 0x20]>(le64(&g, 8)) } {
                (k.vb_at, k.vb_len) = grc(&vb);
            }
            // SAFETY: as above (the meshlet buffer object and its grc header).
            if let Some(buf) = unsafe { dread::<u64>(le64(&g, 0x40)).and_then(|m| dread::<[u8; 0x20]>(m)) } {
                (k.mesh_at, k.mesh_len) = grc(&buf);
            // SAFETY: as above.
            } else if let Some(ib) = unsafe { dread::<[u8; 0x20]>(le64(&g, 0x10)) } {
                // A classic geometry: its index buffer stands in for the meshlets (mesh_len 0 marks it).
                k.mesh_at = le64(&ib, 0x18);
                k.ib = u64::from(le32(&ib, 8)) << 8 | u64::from(le16(&ib, 0xC) & 0xFFF);
            }
            GEOMS.with(|m| {
                let mut m = m.borrow_mut();
                if m.len() > MAP_CAP {
                    m.clear();
                }
                m.insert(id, k);
            });
            part(c, 2);
            k
        }
    };
    (d.mesh_at, d.mesh_len, d.vb_at, d.vb_len) = (key.mesh_at, key.mesh_len, key.vb_at, key.vb_len);
    d.ib_len = key.ib;
    if ctx != 0 {
        transform_direct(s, c, &mut d, ctx);
        // SAFETY: the live render context of this draw.
        // One read of ctx+0x12360..+0x1237A: the sample mask (u64,
        // -1 solid), stencil ref, raster, depth-stencil, blend.
        (d.state, d.stipple) = match unsafe { dread::<[u8; 0x1A]>(ctx + 0x1_2360) } {
            Some(b) => ([le16(&b, 0x18), le16(&b, 0xE), u16::from(b[0xC]), u16::from(b[8])], le64(&b, 0)),
            None => ([0xFFFF; 4], u64::MAX),
        };
    }
    part(c, 3);
    if d.model != 0 {
        let known = MODELS.with(|m| m.borrow().get(&d.model).copied());
        d.skinned = known.unwrap_or_else(|| {
            // Skinned when model+0x14 (u16) != 0 or model+0x35 bit 0.
            // SAFETY: the draw's model (TLS+0x1208).
            let v = unsafe { dread::<u16>(d.model + 0x14).is_some_and(|b| b != 0) || dread::<u8>(d.model + 0x35).is_some_and(|b| b & 1 != 0) };
            MODELS.with(|m| {
                let mut m = m.borrow_mut();
                if m.len() > MAP_CAP {
                    m.clear();
                }
                m.insert(d.model, v);
            });
            v
        });
    }
    d.pal_index = NO_PAL;
    // The skin the GPU uses for this draw is the title's own draw
    // constant {curIndex, prevIndex, curCount, prevCount} at skin_cb +
    // 128 * (TLS+0x1110 + TLS+0x1294) + 0x30, indexing the one palette buffer
    // (base qword 0x1477AE218, 48 B per bone). curCount 0: not skinned (the
    // TLS palette pointer is stale then). Cross-check: (TLS pointer - base) / 48.
    // SAFETY: the TLS block and the title's draw-constant array.
    let cb = unsafe { pread::<u32>(tls + TLS_CB_A).unwrap_or(0) + u32::from(pread::<u8>(tls + TLS_CB_B).unwrap_or(0)) };
    // SAFETY: as above.
    if s.cfg.seq {
        // PerBatch +0x00 fade: the draw's fade value, to test
        // whether alternate-frame decals are a temporal dither of it.
        // SAFETY: the title's per-draw record array (image global).
        d.fade = unsafe { pread::<f32>(s.g.skin_cb + 128 * u64::from(cb.min(4095))) }.unwrap_or(f32::NAN);
    }
    if let Some(r) = unsafe { pread::<[u32; 20]>(s.g.skin_cb + 128 * u64::from(cb.min(4095)) + 0x30) } {
        let k = [r[0], r[1], r[2], r[3]];
        d.skin_cb = k;
        // +0x74 / +0x78 the ped remap bases.
        d.remap = [r[17], r[18]];
        d.range = Some(r[9]);
        // The remap is the ped's only when this thread's last skin bind
        // wrote this very base (vehicles write arena offsets here: no remap).
        if d.remap[0] != u32::MAX {
            let k = PED_SKIN.with(std::cell::Cell::get);
            if k.0 != 0 && k.1 != 0 && k.4 == d.remap[0] {
                d.ped_skin = (k.0, k.1, k.2, k.3);
            }
        }
        if d.skinned && k[2] > 0 && k[2] <= 4096 {
            c.skin_draws += 1;
            // SAFETY: a title global.
            let base = unsafe { pread::<u64>(s.g.pal_base) }.unwrap_or(0);
            if base != 0 && d.palette.0 >= base && (d.palette.0 - base) % 48 == 0 && (d.palette.0 - base) / 48 == u64::from(k[0]) {
                c.skin_agree += 1;
            }
            c.skin_extent = c.skin_extent.max(k[0].saturating_add(k[2]));
            // The bones as this draw reads them (TLS pointer and count), fault-fixed.
            let _site = fault_site(SITE_BONES);
            if skinhook::on() {
                d.hook_bones = c.hook_bones.copy(d.palette.0, d.palette.1, &mut |at, dst| {
                    // SAFETY: the title's palette span this draw binds; a fault in the copy is fixed up.
                    (0x1_0000..0x8000_0000_0000).contains(&at) && unsafe { fcopy(dst.as_mut_ptr().cast(), at as *const u8, 4 * dst.len()) }
                });
            }
            // Vehicle rules: a non-ped draw with a mask offset (+0x74) reads the arena's mask words.
            if skinhook::rules() & (skinhook::VEH_OFFSET | skinhook::VEH_MASK) != 0 && d.remap[0] != u32::MAX && d.ped_skin.0 == 0 {
                let arena = q(s.g.arena).unwrap_or(0);
                d.veh = c.hook_bones.masks.copy(arena, d.remap[0], &mut |at, dst| {
                    // SAFETY: the title's per-frame arena; a fault in the copy is fixed up.
                    (0x1_0000..0x8000_0000_0000).contains(&at) && unsafe { fcopy(dst.as_mut_ptr().cast(), at as *const u8, 4 * dst.len()) }
                }).map(|w| (w, r[9]));
            }
            if s.cfg.skinlog > 0 && c.frame % 60 == 0 && c.skin_draws <= s.cfg.skinlog as u64 {
                d.pal_base = base;
                d.skin_hash = bytes(base + 48 * u64::from(k[0]), 48 * k[2] as usize).map_or(1, |b| fnv(&b));
            }
        }
    }
    part(c, 4);
    if d.viewport != 0 && !c.cameras.iter().any(|(v, _)| *v == d.viewport) {
        // SAFETY: the viewport the draw renders with.
        match unsafe { dread::<[u8; 0xC0]>(d.viewport) } {
            Some(v) => {
                // SAFETY: the same viewport (fault-fixed).
                let jitter = unsafe { dread::<[f32; 2]>(d.viewport + 0x708) }.unwrap_or([0.0; 2]);
                c.cameras.push((d.viewport, Camera { view: f32s::<16>(&v[0x40..]), proj: f32s::<16>(&v[0x80..]), origin: [0.0; 4], draws: 0, last: None, jitter }));
            }
            None => miss!(c, Viewport),
        }
    } else if s.cfg.seq
        && d.viewport != 0
        && let Some((_, cam)) = c.cameras.iter_mut().find(|(v, _)| *v == d.viewport)
    {
        // Seq captures only: the camera as the last draw saw it.
        // SAFETY: the viewport this draw renders with (fault-fixed).
        if let Some(v) = unsafe { dread::<[f32; 32]>(d.viewport + 0x40) } {
            cam.last = Some((v[..16].try_into().unwrap(), v[16..].try_into().unwrap()));
        }
        cam.draws += 1;
    }
    // The recording thread, its sequence number and the time since
    // this capture began (per-thread frame boundaries).
    // SAFETY: a plain query of the calling thread.
    d.thread = unsafe { xnw_core::win::GetCurrentThreadId() };
    d.hook_us = c.t0.elapsed().as_micros() as u32;
    d.title_frame = title_frame();
    if s.cfg.seq && ctx != 0 {
        // The context's sample mask word (ctx+0x12360) at the draw.
        // SAFETY: the draw's context (fault-fixed).
        d.sample_mask = unsafe { dread::<u32>(ctx + 0x1_2360) }.unwrap_or(u32::MAX);
    }
    if s.cfg.seq && ctx != 0 {
        // The view block [V] the draw's program reads (b0, 480 B per
        // context: 0x1475E9580 + 480 * ctx[+0x12190]); +0xC0 = minus the
        // absolute camera position (camera-relative mode).
        // SAFETY: the draw's context and the title image's view blocks (fault-fixed).
        if let Some(i) = unsafe { dread::<i32>(ctx + CTX_MATRIX_INDEX) }.filter(|i| (0..4096).contains(i))
            && let Some(v) = unsafe { pread::<[f32; 4]>(real(s.title, VIEW_BLOCKS) + 480 * i as u64 + 0xC0) }
        {
            d.b0_cam = [-v[0], -v[1], -v[2]];
        }
    }
    if s.cfg.seq && d.viewport != 0 {
        // The camera this draw's viewport held at hook time (seq only).
        // SAFETY: the viewport this draw renders with (fault-fixed).
        if let Some(v) = unsafe { dread::<[f32; 16]>(d.viewport + 0x40) } {
            d.hook_cam = camera_position(&v);
        }
    }
    d.abs_us = T_START.get_or_init(Instant::now).elapsed().as_micros() as u64;
    d.seq_no = c.draws.len() as u32;
    c.draws.push(d);
}

const NO_PAL: u32 = u32::MAX;

/// The palette buffer is a large ring (curIndex near 689,220): copy
/// a window of it, never from its base. Window at most this many bones (12 MiB).
const SKIN_WINDOW_MAX: u32 = 262_144;

/// One frame's skin bones in one float array: per (curIndex,
/// curCount) the first bone of that span in `floats`.
#[derive(Default)]
struct SkinCopy {
    floats: Vec<f32>,
    at: HashMap<(u32, u32), u32>,
    /// The window copied (first bone, bones), or None: spans one by one.
    window: Option<(u32, u32)>,
    bones: u64,
    /// Spans not copied (past the buffer's size or unreadable).
    refused: u32,
    ms: f64,
    /// Per remap base, the float index in `floats` of remap[base] (u32
    /// entries appended after the bones); bases not copied are absent.
    remap_at: HashMap<(u64, u32), u32>,
    remap_entries: u64,
}

/// Copies the bones of `spans` (curIndex, curCount): one window
/// [min curIndex, max end) when it is at most SKIN_WINDOW_MAX bones and not
/// sparse (at most 4x the bones needed + 1,024), else each distinct span on
/// its own. `limit`: the buffer's size in bones (spans past it are refused);
/// `read(first, n)`: 48 * n bytes of bones first.. (guarded).
fn skin_copy(spans: &[(u32, u32)], limit: u64, read: &mut dyn FnMut(u32, u32) -> Option<Vec<u8>>) -> SkinCopy {
    let t0 = Instant::now();
    let mut out = SkinCopy::default();
    let mut u: Vec<(u32, u32)> = spans.iter().copied().filter(|(_, n)| (1..=4096).contains(n)).collect();
    u.sort_unstable();
    u.dedup();
    let before = u.len();
    u.retain(|(i, n)| u64::from(*i) + u64::from(*n) <= limit);
    out.refused = (before - u.len()) as u32;
    let Some(lo) = u.iter().map(|s| s.0).min() else { return out };
    let hi = u.iter().map(|(i, n)| i + n).max().unwrap_or(lo);
    let need: u64 = u.iter().map(|s| u64::from(s.1)).sum();
    let floats = |b: Vec<u8>| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>();
    if hi - lo <= SKIN_WINDOW_MAX
        && u64::from(hi - lo) <= 4 * need + 1024
        && let Some(b) = read(lo, hi - lo)
    {
        out.floats = floats(b);
        out.at = u.iter().map(|(i, n)| ((*i, *n), i - lo)).collect();
        out.window = Some((lo, hi - lo));
        out.bones = u64::from(hi - lo);
    } else {
        for (i, n) in u {
            match read(i, n) {
                Some(b) => {
                    out.at.insert((i, n), (out.floats.len() / 12) as u32);
                    out.floats.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
                    out.bones += u64::from(n);
                }
                None => out.refused += 1,
            }
        }
    }
    out.ms = t0.elapsed().as_secs_f64() * 1e3;
    out
}

/// A rigid part's world: its bone (3 rows: row k = (image of model axis
/// ... as columns, translation in .w), object space) then the object's world
/// (row vectors): p' = (p * Bt) * W, Bt rows i = (r0[i], r1[i], r2[i], 0),
/// row 3 = (r0.w, r1.w, r2.w, 1) [I order, as the skin shader's dot(r, (p,1))].
/// `XNW_WORLD_RENDER=...,rigid=on`: compose rigid parts with their bone live.
fn rigid_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|p| p.trim() == "rigid=on")));
    *ON
}


/// A draw's (palette, first bone, bones) in the frame's copy (NO_PALETTE: rigid).
fn skin_of(d: &Draw, skin: &SkinCopy) -> (u32, u32, u32) {
    match skin.at.get(&(d.skin_cb[0], d.skin_cb[2])) {
        Some(f) if d.skinned => (0, *f, d.skin_cb[2]),
        _ => (xnw_world::NO_PALETTE, 0, 0),
    }
}

/// The live copier: the frame's skinned spans from the title's buffer.
fn skin_copy_live(s: &State, draws: &[Draw]) -> SkinCopy {
    let spans: Vec<(u32, u32)> = draws.iter().filter(|d| d.skinned).map(|d| (d.skin_cb[0], d.skin_cb[2])).collect();
    if spans.is_empty() {
        return SkinCopy::default();
    }
    let Some(base) = q(s.g.pal_base).filter(|b| *b != 0) else {
        return SkinCopy { refused: spans.len() as u32, ..Default::default() };
    };
    let size = arr::<4>(s.g.pal_size).map_or(0, u32::from_le_bytes);
    let limit = if size == 0 { u64::MAX } else { u64::from(size) };
    let mut k = skin_copy(&spans, limit, &mut |first, n| bytes(base + 48 * u64::from(first), 48 * n as usize));
    // The per-frame arena is the VEHICLE remap's home (byte
    // offsets to mask|rank words), not the peds': read as a ped remap it fed
    // garbage bones to 87 draws a frame. No arena remap is applied; the ped
    // remap comes from the skin object's table (skin+0x20, a grc buffer:
    // CPU address +0x18, bytes = count +8 x element +0xC) only, u32 entries,
    // bounded by its size.
    let mut tables: HashMap<u64, Vec<u32>> = HashMap::new();
    for d in draws.iter().filter(|d| d.skinned && d.ped_skin.0 != 0) {
        tables.entry(d.ped_skin.1).or_default().push(d.remap[0]);
    }
    for (obj, bases) in tables {
        let Some((at, len)) = bytes(obj, 0x20).map(|h| grc(&h)) else { continue };
        let entries = (len / 4).min(1 << 20) as u32;
        remap_copy(&mut k, obj, &bases, &mut |first, n| {
            let n = n.min(entries.saturating_sub(first));
            (n > 0).then(|| bytes(at + 4 * u64::from(first), 4 * n as usize)).flatten()
        });
    }
    k
}

/// Appends the remap tables the frame's ped draws use: one window
/// [min base, max base + REMAP_SPAN) when it holds at most REMAP_MAX entries,
/// else each base's REMAP_SPAN entries. `read(first, n)`: n u32 entries from
/// entry `first` of the remap view (guarded).
fn remap_copy(k: &mut SkinCopy, table: u64, bases: &[u32], read: &mut dyn FnMut(u32, u32) -> Option<Vec<u8>>) {
    let mut u: Vec<u32> = bases.to_vec();
    u.sort_unstable();
    u.dedup();
    let (Some(lo), Some(hi)) = (u.first().copied(), u.last().map(|h| h.saturating_add(REMAP_SPAN))) else { return };
    let put = |k: &mut SkinCopy, b: Vec<u8>| -> u32 {
        let at = k.floats.len() as u32;
        k.floats.extend(b.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap()))));
        k.remap_entries += (b.len() / 4) as u64;
        at
    };
    if hi - lo <= REMAP_MAX
        && let Some(b) = read(lo, hi - lo)
    {
        let at = put(k, b);
        k.remap_at.extend(u.iter().map(|b| ((table, *b), at + (b - lo))));
        return;
    }
    for b in u {
        if let Some(bytes) = read(b, REMAP_SPAN) {
            let at = put(k, bytes);
            k.remap_at.insert((table, b), at);
        }
    }
}

/// Emptied draw vectors for the next captures (one per recording thread).
static SPARE: Mutex<Vec<Vec<Draw>>> = Mutex::new(Vec::new());

fn spare(mut v: Vec<Draw>) {
    v.clear();
    if v.capacity() > 0
        && let Ok(mut s) = SPARE.lock()
        && s.len() < 8
    {
        s.push(v);
    }
}


/// `skinlog`: per geometry, the vertex count used and the highest bone
/// index with a non-zero weight (None: no skin streams).
static BONES: Mutex<Option<HashMap<GeomKey, (usize, Option<u8>, bool, usize)>>> = Mutex::new(None);

fn bones_used(vb: &[u8], indices: &[u32]) -> (usize, Option<u8>, bool, usize) {
    let Some(s) = Streams::parse(vb) else { return (0, None, false, 0) };
    // An earlier title marks cloth vertices with blend index 255: count them.
    let mut cloth = 0;
    // A per-vertex 16-B delta stream at slot 0x44 (format 4) in ped layouts [U use].
    let delta = le32(vb, 0x44) & 15 == 4 && le32(vb, 0x44) >> 3 & 0x3C != 0;
    let mut seen = std::collections::HashSet::new();
    let mut max: Option<u8> = None;
    for v in indices {
        if seen.insert(*v)
            && let Some((idx, w)) = s.skin(vb, *v)
        {
            for k in 0..8 {
                if w[k] != 0 {
                    max = Some(max.map_or(idx[k], |m| m.max(idx[k])));
                }
            }
            cloth += usize::from((0..8).any(|k| w[k] != 0 && idx[k] == 255));
        }
    }
    (seen.len(), max, delta, cloth)
}

/// Geometry blobs already in a sequence's shared blobs dir (geometry, hash).
static DUMPED: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);

/// [`read_transform`] with direct reads of the context and of the matrices
/// the draw binds (title image cache or the CPU block srv[] points at); the
/// object-bound case reads its buffer guarded.
fn transform_direct(s: &State, c: &mut Capture, d: &mut Draw, ctx: u64) {
    // SAFETY: the title's image and its live render context.
    let rd16 = |a: u64| unsafe { dread::<u16>(a) }.unwrap_or(0);
    // SAFETY: a title image global.
    let slot = unsafe { pread::<u16>(s.g.matrix_slot) }.unwrap_or(0) >> 2;
    d.srv_slot = slot;
    // SAFETY: as above.
    d.bound = unsafe { dread::<u64>(ctx + CTX_SLOTS + 8 * u64::from(slot)) }.unwrap_or(0);
    d.srv_lazy = rd16(ctx + CTX_SRV_LAZY + 2 * u64::from(slot));
    // SAFETY: as above.
    let index = unsafe { dread::<i32>(ctx + CTX_MATRIX_INDEX) };
    d.cache_at = index.map_or(0, |i| s.g.matrix_cache.wrapping_add((i64::from(i) * 128) as u64));
    let finite = |m: [f32; 32]| m.iter().all(|v| v.is_finite()).then_some(m);
    // SAFETY: srv[slot] with lazy 0x80 is the CPU block the flush copies for this draw.
    let head = unsafe { dread::<u64>(d.bound) }.unwrap_or(0);
    let is_object = in_title(s.title, head);
    let pick = if d.srv_lazy != 0 && !is_object {
        // SAFETY: as above.
        unsafe { dread::<[f32; 32]>(d.bound) }.and_then(finite).map(|m| (m, d.bound, WorldSrc::Bound))
    } else {
        None
    };
    // SAFETY: the title image's transform cache.
    // SAFETY: the transform cache is a title image array (index bounded by its use).
    let pick = pick.or_else(|| unsafe { dread::<[f32; 32]>(d.cache_at) }.and_then(finite).map(|m| (m, d.cache_at, WorldSrc::Cache)));
    let pick = pick.or_else(|| {
        if !is_object {
            return None;
        }
        d.bound_buffer = bytes(d.bound, 0x40).map(|h| grc(&h)).unwrap_or_default();
        bytes(d.bound_buffer.0, 128).map(|m| f32s::<32>(&m)).and_then(finite).map(|m| (m, d.bound_buffer.0, WorldSrc::BoundBuffer))
    });
    match pick {
        Some((m, at, src)) => (d.world, d.world_at, d.world_src) = (Some(m), at, src),
        None => miss!(c, Matrix),
    }
}
fn capture_draw(s: &State, c: &mut Capture, geom: u64, ret: u64) {
    let mut d = Draw { geom, ret, ..Default::default() };
    // SAFETY: a plain query of the calling thread.
    d.thread = unsafe { xnw_core::win::GetCurrentThreadId() };
    d.pass = PASS.with(std::cell::Cell::get);
    let tls = tls_block(s.tls_index);
    let mut ctx = None;
    match tls.and_then(|t| bytes(t + TLS_DRAWABLE, (TLS_TECHNIQUE - TLS_DRAWABLE + 1) as usize)) {
        Some(b) => {
            let at = |o: u64| (o - TLS_DRAWABLE) as usize;
            d.palette = (le64(&b, at(TLS_PALETTE)), le32(&b, at(TLS_PALETTE_N)));
            d.palette2 = le64(&b, at(TLS_PALETTE + 8));
            d.drawable = le64(&b, at(TLS_DRAWABLE));
            d.model = le64(&b, at(TLS_MODEL));
            d.geom_index = le32(&b, at(TLS_GEOM_INDEX));
            d.shader = le64(&b, at(TLS_SHADER));
            d.viewport = le64(&b, at(TLS_VIEWPORT));
            d.technique = b[at(TLS_TECHNIQUE)];
            ctx = Some(le64(&b, at(TLS_CTX)));
        }
        None => miss!(c, Tls),
    }
    // Geometry and its buffers.
    let mut want = GeomData::default();
    let dump = !c.geoms.contains_key(&geom) && c.geoms.len() < DUMPS;
    match bytes(geom, 0x80) {
        Some(g) => {
            d.ghash = le64(&g, 0x30);
            if dump {
                want.dump.push(("geometry", geom, g.clone()));
                for (what, at) in [("vertex-buffer-object", le64(&g, 8)), ("declaration", le64(&g, 0x20))] {
                    if let Some(b) = bytes(at, 0x100) {
                        want.dump.push((what, at, b));
                    }
                }
            }
            let (vb, ib, ml) = (le64(&g, 8), le64(&g, 0x10), le64(&g, 0x40));
            if ml != 0 {
                match bytes(ml, 16) {
                    Some(m) => {
                        let w = le64(&m, 8);
                        d.meshlets = (w & 0x7FFF) as u32;
                        d.triangles = le32(&m, 0xA) & 0x7F_FFFF;
                        d.variant = w >> 39 & 1 != 0;
                        match bytes(le64(&m, 0), 0x30) {
                            Some(buf) => {
                                (d.mesh_at, d.mesh_len) = grc(&buf);
                                if let Some(v) = q(le64(&buf, 0x28) + 0x18).and_then(|r| arr::<8>(r)) {
                                    let v = u64::from_le_bytes(v);
                                    d.mesh_vsharp = v & 0xFFFF_FFFF | (v >> 32 & 0xFFFF) << 32;
                                }
                            }
                            None => miss!(c, Meshlets),
                        }
                    }
                    None => miss!(c, Meshlets),
                }
            }
            if vb != 0 {
                match bytes(vb, 0x20) {
                    Some(b) => {
                        (d.vb_at, d.vb_len) = grc(&b);
                        d.vb_stride = u32::from(le16(&b, 0xC) & 0xFFF);
                    }
                    None => miss!(c, Vertex),
                }
            }
            if ib != 0 {
                match bytes(ib, 0x20) {
                    Some(b) => {
                        (d.ib_at, d.ib_len) = grc(&b);
                        d.ib_elem = u32::from(le16(&b, 0xC) & 0xFFF);
                    }
                    None => miss!(c, Index),
                }
            }
        }
        None => miss!(c, Geometry),
    }
    // A sequence copies a geometry's bytes only until its blobs are on disk.
    if s.cfg.seq && !c.geoms.contains_key(&geom) && DUMPED.lock().is_ok_and(|g| g.as_ref().is_some_and(|g| g.contains(&(geom, d.ghash)))) {
        c.geoms.insert(geom, GeomData::default());
    }
    if !c.geoms.contains_key(&geom) {
        let mut copy = |at: u64, len: u64, cap: u64, k: Kind| -> Option<Vec<u8>> {
            if at == 0 || len == 0 {
                return None;
            }
            let len = len.min(cap);
            if c.copied + len > FRAME_BUDGET {
                c.over_budget += 1;
                return None;
            }
            let b = bytes(at, len as usize);
            match &b {
                Some(_) => c.copied += len,
                None => c.unread[k as usize] += 1,
            }
            b
        };
        want.mesh = copy(d.mesh_at, d.mesh_len, MESH_CAP, Kind::MeshData);
        want.vb = copy(d.vb_at, d.vb_len, BUFFER_CAP, Kind::Vertex);
        if want.mesh.is_none() {
            want.ib = copy(d.ib_at, d.ib_len, BUFFER_CAP, Kind::Index);
            want.ib_elem = d.ib_elem;
        }
        c.geoms.insert(geom, want);
    }
    if let Some(ctx) = ctx {
        read_transform(s, c, &mut d, ctx);
        match bytes(ctx + CTX_SLOTS, 8 * SLOTS) {
            Some(b) => {
                d.slots = (0..SLOTS).map(|i| le64(&b, 8 * i)).collect();
                d.slot_mem = d.slots.iter().map(|o| texture(s.title, *o).unwrap_or_default()).collect();
            }
            None => miss!(c, Slots),
        }
        if let Some(p) = q(ctx + CTX_PROGRAMS).and_then(|p| q(p + 8)) {
            d.program = p;
            d.program_flags = q(p + 0x50).unwrap_or(0);
        }
    }
    // Shader, effect, the diffuse texture.
    if d.shader != 0 {
        match bytes(d.shader, 0x18) {
            Some(sh) => {
                d.effect = le64(&sh, 0);
                let tex_vars = le64(&sh, 0x10);
                if !c.effects.contains_key(&d.effect) {
                    match effect(d.effect) {
                        Some(e) => {
                            c.effects.insert(d.effect, e);
                        }
                        None => miss!(c, Effect),
                    }
                }
                if let Some(e) = c.effects.get(&d.effect) {
                    d.diffuse = diffuse(e, tex_vars, s.title);
                }
                if let Some((_, _, _, t)) = d.diffuse
                    && t.obj != 0
                    && !c.dumped_tex.contains(&t.obj)
                    && c.dumped_tex.len() < 20
                {
                    c.dumped_tex.push(t.obj);
                    c.dump("texture", t.obj, 0x80, 20);
                }
            }
            None => miss!(c, Shader),
        }
    }
    if d.viewport != 0 && !c.cameras.iter().any(|(v, _)| *v == d.viewport) {
        match bytes(d.viewport, 0x410) {
            Some(v) => {
                c.cameras.push((d.viewport, Camera { view: f32s::<16>(&v[0x40..]), proj: f32s::<16>(&v[0x80..]), origin: f32s::<4>(&v[0x400..]), draws: 0, last: None, jitter: [0.0; 2] }));
                // The whole grcViewport (0x730 B: near/far/FOV/aspect at
                // +0x6D0.., forward/reverse-Z byte +0x6CC) for its size.
                c.dump("viewport", d.viewport, 0x730, 32);
            }
            None => miss!(c, Viewport),
        }
    }
    if let Some((_, cam)) = c.cameras.iter_mut().find(|(v, _)| *v == d.viewport) {
        cam.draws += 1;
    }
    if !c.threads.contains(&d.thread) {
        c.threads.push(d.thread);
    }
    c.draws.push(d);
}

/// The draw's transform. In one captured frame srv[slot] pointed at a
/// per-context object, not at matrices, in 1,220 of 1,782 draws (their 128
/// bytes start with a pointer into the title image: a vtable); those are
/// read as a grc buffer, and the cache entry by index is the fallback.
fn read_transform(s: &State, c: &mut Capture, d: &mut Draw, ctx: u64) {
    read_transform_with(s, c, d, ctx, true);
}

/// The last CPU-pointer view written to srv[slot] in this
/// context, element 0 (64 B) of what it points at, and the viewport's
/// world (vp+0x000, SetWorld) for this draw. Guarded reads.
fn view_info(d: &mut Draw, ctx: u64) {
    let w = VIEW_WRITERS.lock().ok().and_then(|m| m.as_ref().and_then(|m| m.get(&(ctx + 0x1_2780, d.srv_slot)).copied()));
    if let Some(w) = w {
        d.view_writer = w;
        d.view_elem0 = bytes(w[1], 64).map(|b| f32s::<16>(&b));
    }
    d.vp_world = bytes(d.viewport, 64).map(|b| f32s::<16>(&b));
}

fn read_transform_with(s: &State, c: &mut Capture, d: &mut Draw, ctx: u64, dumps: bool) {
    let slot = arr::<2>(real(s.title, MATRIX_SLOT)).map(u16::from_le_bytes).unwrap_or(0) >> 2;
    d.srv_slot = slot;
    d.bound = q(ctx + CTX_SLOTS + 8 * u64::from(slot)).unwrap_or(0);
    d.srv_lazy = arr::<2>(ctx + CTX_SRV_LAZY + 2 * u64::from(slot)).map(u16::from_le_bytes).unwrap_or(0);
    let index = arr::<4>(ctx + CTX_MATRIX_INDEX).map(i32::from_le_bytes);
    d.cache_at = index.map_or(0, |i| real(s.title, MATRIX_CACHE).wrapping_add((i64::from(i) * 128) as u64));
    let finite = |at: u64| bytes(at, 128).map(|m| f32s::<32>(&m)).filter(|m| m.iter().all(|v| v.is_finite()));
    let head = bytes(d.bound, 0x40);
    let is_object = head.as_ref().is_some_and(|h| in_title(s.title, le64(h, 0)));
    if is_object && let Some(h) = &head {
        d.bound_buffer = grc(h);
    }
    let pick = if d.srv_lazy != 0 && !is_object { finite(d.bound).map(|m| (m, d.bound, WorldSrc::Bound)) } else { None };
    let pick = pick.or_else(|| finite(d.cache_at).map(|m| (m, d.cache_at, WorldSrc::Cache)));
    let pick = pick.or_else(|| if is_object { finite(d.bound_buffer.0).map(|m| (m, d.bound_buffer.0, WorldSrc::BoundBuffer)) } else { None });
    match pick {
        Some((m, at, src)) => (d.world, d.world_at, d.world_src) = (Some(m), at, src),
        None => miss!(c, Matrix),
    }
    if !dumps {
        return;
    }
    view_info(d, ctx);
    if is_object {
        c.dump("transform-object", d.bound, 0x80, 20);
        if d.bound_buffer.0 != 0 {
            c.dump("transform-object-buffer", d.bound_buffer.0, d.bound_buffer.1.clamp(0x80, 4096) as usize, 20);
        }
    } else if d.world_src != WorldSrc::Bound {
        c.dump("transform-bound", d.bound, 0x80, 20);
    }
    // Both palettes copied per (pointer, drawable, bones) in the frame
    // (copied once per pointer per frame; one pointer serves several
    // objects), up to 512 bones.
    d.pal_index = pal_copy(c, d.palette.0, d.drawable, d.palette.1);
    // The draw's skin constants, as in capture_light (guarded here).
    if let Some(tls) = tls_block(s.tls_index) {
        let cb = arr::<4>(tls + TLS_CB_A).map_or(0, u32::from_le_bytes) + arr::<1>(tls + TLS_CB_B).map_or(0, |b| u32::from(b[0]));
        if let Some(k) = bytes(s.g.skin_cb + 128 * u64::from(cb.min(4095)) + 0x30, 16) {
            d.skin_cb = [le32(&k, 0), le32(&k, 4), le32(&k, 8), le32(&k, 12)];
            if d.skin_cb[2] > 0 && d.skin_cb[2] <= 4096 {
                c.skin_extent = c.skin_extent.max(d.skin_cb[0].saturating_add(d.skin_cb[2]));
            }
        }
    }
    d.pal2_index = pal_copy(c, d.palette2, d.drawable ^ 1, d.palette.1);
}

/// This frame's copy of a palette for (pointer, drawable, bones), made once.
fn pal_copy(c: &mut Capture, p: u64, drawable: u64, bones: u32) -> u32 {
    if p == 0 || bones == 0 {
        return NO_PAL;
    }
    let k = (p, drawable, bones);
    if let Some(i) = c.pal_keys.get(&k) {
        return *i;
    }
    let Some(b) = bytes(p, 48 * bones.min(512) as usize) else { return NO_PAL };
    if c.pal_keys.keys().any(|(ptr, drw, _)| *ptr == p && *drw != drawable) {
        c.pal_shared += 1;
    }
    let i = c.pal_list.len() as u32;
    c.pal_list.push(b);
    c.pal_keys.insert(k, i);
    i
}

/// A pointer into the title image (a vtable).
fn in_title(title: usize, p: u64) -> bool {
    p >= title as u64 && p - (title as u64) < 0x1000_0000
}

/// A bound texture pointer: it is a 32-byte view (vtable 0x1461ECDA0),
/// `+8` the grcTexture: +0x18 texels, +0x30 tagged attachment list,
/// +0x38 width, +0x3A height, +0x3F DXGI_FORMAT, +0x42 mips [V]. A
/// pointer that is a texture itself (no view) is taken as one.
fn texture(title: usize, view: u64) -> Option<Tex> {
    let v = bytes(view, 0x20)?;
    if !in_title(title, le64(&v, 0)) {
        return None;
    }
    let inner = le64(&v, 8);
    let (obj, b) = match bytes(inner, 0x48).filter(|b| in_title(title, le64(b, 0))) {
        Some(b) => (inner, b),
        None => (view, bytes(view, 0x48)?),
    };
    let (at, len) = grc(&b);
    let (key, via) = name_key(le64(&b, 0x30));
    Some(Tex { obj, at, len, fmt: b[0x3F], mips: b[0x42], w: le16(&b, 0x38), h: le16(&b, 0x3A), key, key_via: via })
}

/// The u64 name key in a texture's attachment list (+0x30): the head is a
/// plain pointer to a node {next | own type (low 3 bits), payload}; the
/// first type-5 node's payload is the key [V 0x14501F330, 0x14249AA50].
/// (key, 1) when found.
fn name_key(head: u64) -> (u64, u8) {
    let mut p = head;
    for _ in 0..16 {
        if p < 0x1_0000 {
            break;
        }
        let Some(n) = bytes(p, 16) else { break };
        if le64(&n, 0) & 7 == 5 {
            return (le64(&n, 8), 1);
        }
        p = le64(&n, 0) & !7;
    }
    (0, 0)
}

/// A texVars value: a grcTexture*, or with bit 63 a handle resolved as
/// 0x144EE12A0 does: record = qword_149C3E108 + 16 * u16[h+6]; resident
/// when min(u8 rec+14, u8 rec+12) != 0; texture = *(*rec + 0x28) [V].
fn resolve_texture(title: usize, v: u64) -> u64 {
    if v >> 63 == 0 {
        return v;
    }
    let h = v & !(1 << 63);
    let (Some(i), Some(table)) = (arr::<2>(h + 6).map(u16::from_le_bytes), q(real(title, TEX_HANDLES))) else { return 0 };
    let rec = table + 16 * u64::from(i);
    let Some(r) = bytes(rec, 16) else { return 0 };
    if r[14].min(r[12]) == 0 {
        return 0;
    }
    q(le64(&r, 0) + 0x28).unwrap_or(0)
}

/// Text at the first pointer among `head`'s qwords that reads as a name.
fn name_in(head: &[u8], what: &str) -> Option<String> {
    for i in 0..head.len() / 8 {
        let Some(t) = bytes(le64(head, 8 * i), 48) else { continue };
        let s: Vec<u8> = t.iter().copied().take_while(|b| *b != 0).collect();
        if s.len() >= 4 && s.len() < 48 && s.iter().all(|b| b.is_ascii_graphic()) {
            return Some(format!("{what}+{:#x}:{}", 8 * i, String::from_utf8_lossy(&s)));
        }
    }
    None
}

/// sh->fx and its template *fx (112 bytes).
fn effect(fx: u64) -> Option<Effect> {
    let head = bytes(fx, 0x60)?;
    let template = le64(&head, 0);
    let template_head = bytes(template, 0x70)?;
    let name = name_in(&head, "fx").or_else(|| name_in(&template_head, "t")).unwrap_or_default();
    Some(Effect { vars: template_head[4], head, template, template_head, name })
}

/// The albedo texture as 0x145429040 orders the names: 6994ef3f,
/// DiffuseTex, 7d4f12e2, InteriorDiffuseMap; each as 0x144F29850(t, hash,
/// kind 0) finds it: the first var record whose hash (+0, then +4:
/// 0x144F298A0 / 0x144F298D0) matches with kind byte (+12) 0;
/// texVars[record.index (+8)]. Returns (name hash, var index, value, texture).
fn diffuse(e: &Effect, tex_vars: u64, title: usize) -> Option<(u32, u16, u64, Tex)> {
    let n = usize::from(e.vars);
    let vars = bytes(le64(&e.template_head, 0x50), 16 * n)?;
    for name in ALBEDO_NAMES {
        let hit = |h: usize| (0..n).find(|i| le32(&vars, 16 * i + h) == name && vars[16 * i + 12] == 0);
        let Some(r) = hit(0).or_else(|| hit(4)) else { continue };
        let idx = le16(&vars, 16 * r + 8);
        let v = q(tex_vars + 8 * u64::from(idx)).unwrap_or(0);
        let tex = texture(title, resolve_texture(title, v)).unwrap_or_default();
        return Some((name, idx, v, tex));
    }
    None
}



/// Geometries not ready read at most this many times a frame.
const NOT_READY_RETRIES: u32 = 256;

// ---- decoders ----












// ---- world space ----








// ---- writing ----




fn where_(title: usize, ret: u64) -> String {
    if in_title(title, ret) { format!("title+{:#x}", ret - title as u64) } else { format!("{ret:#x}") }
}

/// The `.ttpf` texture hash of the texture whose texels lie at `addr` (the
/// origin registry): what the name key should equal [I].
fn pack_hash(addr: u64) -> Option<u64> {
    let o = (addr != 0).then(|| xnw_formats::origin::origin_of(addr, 0x1000)).flatten()?;
    Some(xnw_formats::origin::file(o.file)?.texture(o.texture)?.hash)
}

/// A texture's pack identity (the origin registry), else "-".
fn identity(addr: u64) -> String {
    if addr == 0 {
        return "-".into();
    }
    match xnw_formats::origin::origin_of(addr, 0x1000) {
        Some(o) => {
            let f = xnw_formats::origin::file(o.file).and_then(|f| f.path.file_name().map(|n| n.to_string_lossy().into_owned()));
            format!("{}#{}", f.unwrap_or_else(|| format!("file{}", o.file)), o.texture)
        }
        None => "-".into(),
    }
}

fn matrix_text(m: &[f32]) -> String {
    m.chunks(4).map(|r| format!("{:.5} {:.5} {:.5} {:.5}", r[0], r[1], r[2], r[3])).collect::<Vec<_>>().join(" | ")
}

/// A camera matrix as text that reads back bit for bit (shortest round-trip
/// decimals): with 5 decimals the replay's frustum planes were ~2 mm off at
/// 157 m and three records of pk5 f977 row 431 flipped at the top plane.
fn matrix_exact(m: &[f32]) -> String {
    m.chunks(4).map(|r| format!("{} {} {} {}", r[0], r[1], r[2], r[3])).collect::<Vec<_>>().join(" | ")
}

fn write_frame(s: &State, c: Capture) -> std::io::Result<()> {
    write_frame_with(s, &c, true)
}

/// `finish` false leaves the complete marker and the seq line to the
/// caller (screen mode writes its list after this).
fn write_frame_with(s: &State, c: &Capture, finish: bool) -> std::io::Result<()> {
    let t0 = Instant::now();
    let dir = s.cfg.dir.join(format!("frame-{}", c.frame));
    std::fs::create_dir_all(&dir)?;
    // Decode each distinct geometry once.
    let mut meshes: HashMap<u64, Result<Mesh, String>> = HashMap::new();
    let mut errors: Vec<String> = Vec::new();
    // Seq frames decode nothing and write no OBJ (the offline renderer
    // reads the blobs): draws.tsv, palettes and matrices first, `complete` last.
    let lite = s.cfg.seq;
    for (g, data) in c.geoms.iter().filter(|_| !lite) {
        let r = match (&data.mesh, &data.vb, &data.ib) {
            (Some(m), Some(v), _) => decode_meshlets(m, v),
            (None, Some(v), Some(i)) => decode_indexed(i, data.ib_elem, v),
            (_, None, _) => Err("no vertex buffer".into()),
            _ => Err("neither meshlets nor indices".into()),
        };
        if let Err(e) = &r
            && errors.len() < 8
        {
            errors.push(format!("{g:#x}: {e}"));
        }
        meshes.insert(*g, r);
    }
    // `raw=<n>` (n > 0) dumps every probed frame: geometry blobs in the
    // sequence's shared <dir>\blobs as <geometry>-<hash>.<ext> (written once
    // across the frames), palettes per frame in frame-N\blobs\pal-<i>.bin.
    if s.cfg.raw > 0 {
        let shared = s.cfg.dir.join("blobs");
        std::fs::create_dir_all(&shared)?;
        let hash_of: HashMap<u64, u64> = c.draws.iter().map(|d| (d.geom, d.ghash)).collect();
        for (g, data) in &c.geoms {
            let h = hash_of.get(g).copied().unwrap_or(0);
            if let Ok(mut d) = DUMPED.lock() {
                d.get_or_insert_with(std::collections::HashSet::new).insert((*g, h));
            }
            for (ext, b) in [("mesh", &data.mesh), ("vb", &data.vb), ("ib", &data.ib)] {
                let path = shared.join(format!("{g:x}-{h:x}.{ext}"));
                if let Some(b) = b
                    && !path.exists()
                {
                    std::fs::write(path, b)?;
                }
            }
        }
        let raw = dir.join("blobs");
        std::fs::create_dir_all(&raw)?;
        for (i, b) in c.pal_list.iter().enumerate() {
            std::fs::write(raw.join(format!("pal-{i}.bin")), b)?;
        }
        for (i, b) in c.inst_list.iter().enumerate() {
            std::fs::write(raw.join(format!("inst-{i}.bin")), b.as_slice())?;
        }
        // The timecycle render copy of this frame (sun, ambient).
        if let Some(tc) = bytes(real(s.title, TIMECYCLE), 0x16C0) {
            std::fs::write(dir.join("timecycle.bin"), tc)?;
        }
        // Per viewport the camera at the first and last draw, the
        // TAA jitter (+0x708) and the projection's offset row; per draw the
        // recording thread, its sequence number and hook time.
        let mut cams = String::from("viewport\tdraws\tcam_first\tcam_last\tmoved_m\tview_changed\tproj_changed\tjitter\tproj_row2\tthread_conflicts\n");
        for (vp, k) in &c.cameras {
            let pos = |v: &[f32; 16]| camera_position(v);
            let (a, b) = (pos(&k.view), k.last.map(|(v, _)| pos(&v)).unwrap_or(pos(&k.view)));
            let moved = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            let (vc, pc) = k.last.map_or((false, false), |(v, p)| (v != k.view, p != k.proj));
            cams += &format!("{vp:#x}\t{}\t{:.3} {:.3} {:.3}\t{:.3} {:.3} {:.3}\t{moved:.4}\t{}\t{}\t{:.6} {:.6}\t{:.6} {:.6} {:.6} {:.6}\t{}\n", k.draws + 1, a[0], a[1], a[2], b[0], b[1], b[2], u8::from(vc), u8::from(pc), k.jitter[0], k.jitter[1], k.proj[8], k.proj[9], k.proj[10], k.proj[11], c.cam_conflicts);
        }
        std::fs::write(dir.join("cameras.tsv"), cams)?;
        let mut times = String::from("draw\tthread\tseq\thook_us\tviewport\tstage\tghash\ttitle_frame\tabs_us\thook_cam\tb0_cam\tsample_mask\tfade\tentity\tent_kind\tent_type\tent_fade\tlod_type\tent_flags\tmap_ref\tentry_f\tentry_lod\n");
        for (i, d) in c.draws.iter().enumerate() {
            times += &format!("{i}\t{}\t{}\t{}\t{:#x}\t{}\t{:#x}\t{}\t{}\t{:.3} {:.3} {:.3}\t{:.3} {:.3} {:.3}\t{:#x}\t{:.4}\n", d.thread, d.seq_no, d.hook_us, d.viewport, stage_of(d), d.ghash, d.title_frame, d.abs_us, d.hook_cam[0], d.hook_cam[1], d.hook_cam[2], d.b0_cam[0], d.b0_cam[1], d.b0_cam[2], d.sample_mask, d.fade);
            let e = &d.ent;
            times.pop();
            times += &format!("\t{:#x}\t{}\t{}\t{}\t{}\t{:#x}\t{:#x}\t{:.3}\t{:#x}\n", e.at, e.kind, e.etype, e.fade, e.lod_type, e.flags, e.map_ref, e.entry_f, e.entry_lod);
        }
        std::fs::write(dir.join("draw-times.tsv"), times)?;
        // Per instanced call its handler and the fields the title's
        // batch path tests, and the distinct handlers seen this frame.
        let mut calls = String::from("draw\tghash\tstage\tlod\tfirst_lod\tcount\tkind\thandler\th228\th33c\th33d\trec_a0\n");
        let mut handlers = std::collections::BTreeSet::new();
        for (i, d) in c.draws.iter().enumerate() {
            if let Some(n) = d.inst {
                handlers.insert(n.handler);
                calls += &format!("{i}\t{:#x}\t{}\t{}\t{}\t{}\t{}\t{:#x}\t{}\t{:#04x}\t{:#04x}\t{:#x}\n", d.ghash, stage_of(d), n.lod, n.first_lod, n.count, if n.gpu != 0 { "gpu" } else { "cpu" }, n.handler, n.h228, n.h33c & 0xFF, n.h33c >> 8, n.rec_a0);
            }
        }
        std::fs::write(dir.join("inst-calls.tsv"), calls)?;
        std::fs::write(dir.join("handlers.txt"), handlers.iter().map(|h| format!("{h:#x}\n")).collect::<String>())?;
        // The drain markers since the last dump (drain enter / flip after it).
        let marks: Vec<_> = DRAINS.lock().map(|mut d| std::mem::take(&mut *d)).unwrap_or_default();
        let mut text = String::from("kind\ttitle_frame\tpresented\tthread\treturn\tabs_us\tstep\tgame_s\tgame_s2\tscan_records\toccluders\taux\n");
        for (k, f, p, tid, ret, us, clk, n) in marks {
            text += &format!("{}\t{f}\t{p}\t{tid}\t{ret:#x}\t{us}\t{:.6}\t{:.4}\t{:.4}\t{}\t{}\t{}\n", if k == 1 { "drain" } else { "flip" }, clk[0], clk[2], clk[3], n[0], n[1], n[2]);
        }
        std::fs::write(dir.join("drains.tsv"), text)?;
        // The scan's visible entities of this title frame, if kept.
        let taken = SCANS.lock().ok().and_then(|mut q| q.iter().position(|(f, _)| u64::from(*f) == c.frame).and_then(|i| q.remove(i)));
        if let Some((_, e)) = taken {
            std::fs::write(dir.join("scan.txt"), e.chunks_exact(2).map(|p| format!("{:#x}\t{:#x}\n", p[0], p[1])).collect::<String>())?;
        }
        // Per distinct ped skin table: the whole table (bounded 1 MiB)
        // once per run, and per frame the binds seen (skin, comp, lod, bases).
        let mut binds: HashMap<(u64, u64, u32, u32, u32, u32), u32> = HashMap::new();
        for d in c.draws.iter().filter(|d| d.ped_skin.0 != 0) {
            *binds.entry((d.ped_skin.0, d.ped_skin.1, d.ped_skin.2, d.ped_skin.3, d.remap[0], d.remap[1])).or_default() += 1;
        }
        if !binds.is_empty() {
            let mut text = String::from("skin\ttable\tcomp\tlod\tcur_base\tprev_base\ttable_at\ttable_bytes\tdraws\n");
            let mut rows: Vec<_> = binds.into_iter().collect();
            rows.sort();
            for ((skin, obj, comp, lod, cur, prev), n) in rows {
                let (at, len) = bytes(obj, 0x20).map(|h| grc(&h)).unwrap_or((0, 0));
                text += &format!("{skin:#x}\t{obj:#x}\t{comp}\t{lod}\t{cur}\t{}\t{at:#x}\t{len}\t{n}\n", prev as i32);
                let file = s.cfg.dir.join("blobs").join(format!("remap-{obj:x}.bin"));
                if at != 0
                    && !file.exists()
                    && let Some(b) = bytes(at, len.min(1 << 20) as usize)
                {
                    std::fs::create_dir_all(s.cfg.dir.join("blobs"))?;
                    std::fs::write(file, b)?;
                }
            }
            std::fs::write(dir.join("remap.tsv"), text)?;
        }
        // The water height map, once per map (header + u16 texels).
        if let Some(m) = water_map_read(s.title) {
            let wdir = s.cfg.dir.join("water");
            let name = format!("{:016x}", m.id());
            if !wdir.join(format!("{name}.txt")).exists()
                && let Some(texels) = water_texels(&m)
            {
                std::fs::create_dir_all(&wdir)?;
                std::fs::write(wdir.join(format!("{name}.bin")), texels.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                let zero = texels.iter().filter(|v| **v == 0).count();
                let (lo, hi) = texels.iter().filter(|v| **v != 0).fold((u16::MAX, 0), |(a, b), v| (a.min(*v), b.max(*v)));
                std::fs::write(wdir.join(format!("{name}.txt")), format!("map {:#x}\ndata {:#x}\nwidth {}\nheight {}\naabb {:?}\ntransform {:?}\nzero texels {zero} of {}\nnon-zero range {lo}..{hi} ({:.2} .. {:.2} m)\nframe {}\n", m.map, m.data, m.w, m.h, m.aabb, m.s, texels.len(), 1600.0 * f32::from(lo) / 65535.0 - 800.0, 1600.0 * f32::from(hi) / 65535.0 - 800.0, c.frame))?;
            }
        }
        // The terrain tile records of this frame (read on the
        // writer thread just after the frame: the array is rebuilt per frame).
        if let Some((cap, count, recs)) = terrain_tiles(s.title) {
            std::fs::write(raw.join("terrain.bin"), &recs)?;
            let (tiles, fields) = terrain_report(&recs);
            std::fs::write(dir.join("terrain-tiles.tsv"), tiles)?;
            std::fs::write(dir.join("terrain-fields.tsv"), fields)?;
            let lods = recs.chunks_exact(TERRAIN_REC).fold([0u32; 8], |mut a, r| {
                a[(le64(r, 0x16C) as usize).min(7)] += 1;
                a
            });
            xnw_core::info!("world probe: frame {}: terrain tiles {count} (cap {cap}), {} records read; per LOD {lods:?}", c.frame, recs.len() / TERRAIN_REC);
        }
        // The LOD records (IcLodRec, 140 B) the instanced draws name.
        for (i, b) in c.lod_list.iter().enumerate() {
            std::fs::write(raw.join(format!("lod-{i}.bin")), b)?;
        }
        // The frame's skinned bones (window or spans) and
        // where each (curIndex, curCount) starts in them.
        let skin = skin_copy_live(s, &c.draws);
        if !skin.at.is_empty() {
            std::fs::write(raw.join("palbuf.bin"), skin.floats.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
            let mut rows: Vec<_> = skin.at.iter().collect();
            rows.sort();
            let text: String = rows.iter().map(|((i, n), f)| format!("{i}\t{n}\t{f}\n")).collect();
            std::fs::write(raw.join("palbuf.tsv"), format!("cur_idx\tcur_n\tfirst\n{text}"))?;
        }
    }
    let empty = Mesh::default();
    let mesh_of = |g: u64| meshes.get(&g).and_then(|r| r.as_ref().ok()).unwrap_or(&empty);
    let ok_meshes: HashMap<u64, &Mesh> = meshes.iter().filter_map(|(g, r)| r.as_ref().ok().map(|m| (*g, m))).collect();
    let views: HashMap<u64, [f32; 16]> = c.cameras.iter().map(|(v, k)| (*v, k.view)).collect();
    let mut world: Vec<WorldDraw> = c.draws.iter().enumerate().map(|(i, d)| WorldDraw { index: i, geom: d.geom, effect: d.effect, viewport: d.viewport, world: d.world, src: d.world_src }).collect();
    let stats = if lite { WorldStats { sources: "-".into(), borrowed: 0, missing: 0, skipped: 0, vertices: 0, bounds: Bounds::new() } } else { write_world(&dir, c.frame, &mut world, &ok_meshes, &views)? };
    // draws.tsv
    // Light captures carry no effect record: the name from the shader (fx+0xD08), once per shader.
    let mut light_names: HashMap<u64, String> = HashMap::new();
    {
        let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("draws.tsv"))?);
        let mut head = String::from("draw\tgeometry\tcaller\tthread\tmeshlets\thdr_tris\tvariant\tvertices\ttriangles\tculled\tdecode\tmesh_data\tmesh_bytes\tmesh_vsharp\tvb\tvb_bytes\tvb_stride\tib\tib_bytes\tib_elem\tdrawable\tmodel\tgeom_index\tshader\teffect\teffect_name\tprogram\tprogram_flags\ttechnique\tviewport\tdiffuse_var\tdiffuse_tex\tdiffuse_mem\tdiffuse_wh\tdiffuse_origin\tworld_at\tworld\tworld_alt\tworld_src\tsrv_slot\tsrv_lazy\tbound\tbound_buffer\tcache_at\tpalette\tpalette_n\tdiffuse_obj\tdiffuse_bytes\tdiffuse_format\tdiffuse_mips\tphase\tstage\tpass_no\tdiffuse_key\tkey_via\tpack_hash\tkey_match\tpalette2\tghash\tpal_index\tpal2_index\tmodel_skinned\tinst_desc\tinst_count\tinst_cpu\tinst_gpu\tinst_lod\tinst_block\tview_writer\tview_ptr\tview_r9\tview_elem0\tvp_world\tcur_idx\tprev_idx\tcur_n\tprev_n\tinst_first_lod\tinst_lodrec");
        for i in 0..SLOTS {
            head += &format!("\tslot{i}");
        }
        // The range word of the hook's record and of the draw's own (sink) record.
        head += "\trange\town_range";
        writeln!(out, "{head}")?;
        let mut keys = (0u32, 0u32, 0u32);
        for (i, d) in c.draws.iter().enumerate() {
            let (verts, tris, culled, status) = match meshes.get(&d.geom) {
                Some(Ok(m)) => (m.pos.len(), m.tris.len(), m.culled, "ok".to_string()),
                Some(Err(e)) => (0, 0, 0, e.replace('\t', " ")),
                None => (0, 0, 0, "not copied".into()),
            };
            let fx = c.effects.get(&d.effect);
            let (dv, dval, dt) = d.diffuse.map_or(("-".into(), "-".into(), Tex::default()), |(n, k, v, t)| (format!("{n:08x}:{k}"), format!("{v:#x}"), t));
            let (dmem, dwh) = (dt.at, format!("{}x{}:f{}", dt.w, dt.h, dt.fmt));
            let (wa, wb) = d.world.map_or(("-".into(), "-".into()), |w| (matrix_text(&w[..16]), matrix_text(&w[16..])));
            let mut row = format!(
                "{i}\t{:#x}\t{}\t{}\t{}\t{}\t{}\t{verts}\t{tris}\t{culled}\t{status}\t{:#x}\t{}\t{:#x}\t{:#x}\t{}\t{}\t{:#x}\t{}\t{}\t{:#x}\t{:#x}\t{}\t{:#x}\t{:#x}\t{}\t{:#x}\t{:#x}\t{}\t{:#x}\t{dv}\t{dval}\t{dmem:#x}\t{dwh}\t{}\t{:#x}\t{wa}\t{wb}\t{}\t{}\t{}\t{:#x}\t{:#x}:{}\t{:#x}\t{:#x}\t{}\t{:#x}\t{}\t{}\t{}\t{:#x}\t{}\t{}",
                d.geom,
                where_(s.title, d.ret),
                d.thread,
                d.meshlets,
                d.triangles,
                u8::from(d.variant),
                d.mesh_at,
                d.mesh_len,
                d.mesh_vsharp,
                d.vb_at,
                d.vb_len,
                d.vb_stride,
                d.ib_at,
                d.ib_len,
                d.ib_elem,
                d.drawable,
                d.model,
                d.geom_index,
                d.shader,
                d.effect,
                fx.map_or_else(|| light_names.entry(d.fx).or_insert_with(|| if d.fx != 0 { light_effect_name(d.fx) } else { q(d.shader).map_or("-".into(), light_effect_name) }).as_str(), |f| if f.name.is_empty() { "-" } else { f.name.as_str() }),
                d.program,
                d.program_flags,
                d.technique,
                d.viewport,
                identity(dmem),
                d.world_at,
                world[i].src.name(),
                d.srv_slot,
                d.srv_lazy,
                d.bound,
                d.bound_buffer.0,
                d.bound_buffer.1,
                d.cache_at,
                d.palette.0,
                d.palette.1,
                dt.obj,
                dt.len,
                dt.fmt,
                dt.mips,
                d.pass.0,
                d.pass.1,
                d.pass.2,
            );
            let pack = pack_hash(dt.at);
            row += &format!("\t{:#018x}\t{}\t{}\t{}\t{:#x}", dt.key, dt.key_via, pack.map_or("-".into(), |h| format!("{h:#018x}")), match pack {
                Some(h) if dt.key != 0 => u8::from(h == dt.key).to_string(),
                _ => "-".into(),
            }, d.palette2);
            let ix = |i: u32| if i == NO_PAL { "-".to_string() } else { i.to_string() };
            row += &format!("\t{:#x}\t{}\t{}\t{}", d.ghash, ix(d.pal_index), ix(d.pal2_index), (arr::<2>(d.model + 0x14).is_some_and(|b| u16::from_le_bytes(b) != 0) || arr::<1>(d.model + 0x35).is_some_and(|b| b[0] & 1 != 0)).then_some("1").unwrap_or(if d.model == 0 { "-" } else { "0" }));
            match d.inst {
                Some(i) => row += &format!("\t{:#x}\t{}\t{:#x}\t{:#x}\t{}\t{}", i.desc, i.count, i.cpu, i.gpu, i.lod, ix(i.block)),
                None => row += "\t-\t0\t-\t-\t-\t-",
            }
            row += &format!("\t{}\t{:#x}\t{:#x}\t{}\t{}", where_(s.title, d.view_writer[0]), d.view_writer[1], d.view_writer[2], d.view_elem0.map_or("-".into(), |m| matrix_text(&m)), d.vp_world.map_or("-".into(), |m| matrix_text(&m)));
            row += &format!("\t{}\t{}\t{}\t{}", d.skin_cb[0], d.skin_cb[1], d.skin_cb[2], d.skin_cb[3]);
            match d.inst {
                Some(i) => row += &format!("\t{}\t{}", if i.first_lod == NO_LOD { "-".into() } else { i.first_lod.to_string() }, ix(i.lodrec)),
                None => row += "\t-\t-",
            }
            if dt.key != 0 {
                keys.0 += 1;
            }
            if let Some(h) = pack {
                keys.1 += 1;
                keys.2 += u32::from(h == dt.key);
            }
            for k in 0..SLOTS {
                let (slot, t) = (d.slots.get(k).copied().unwrap_or(0), d.slot_mem.get(k).copied().unwrap_or_default());
                if slot == 0 {
                    row += "\t-";
                } else if t.at == 0 {
                    row += &format!("\t{:#x}", slot);
                } else {
                    row += &format!("\t{:#x}:{:#x}:{}:f{}:{}x{}:{}", slot, t.at, t.len, t.fmt, t.w, t.h, identity(t.at));
                }
            }
            let word = |w: Option<u32>| w.map_or("-".into(), |w| format!("{w:#x}"));
            row += &format!("\t{}\t{}", word(d.range), word(d.sink_range));
            writeln!(out, "{row}")?;
        }
        out.flush()?;
        let albedo = c.draws.iter().filter(|d| d.diffuse.is_some()).count();
        xnw_core::info!("world probe: frame {}: albedo texture in {albedo} of {} draws; name key in {}, pack texture found in {}, key equal to the pack hash in {}", c.frame, c.draws.len(), keys.0, keys.1, keys.2);
        // Draws per pass (phase, stage of the last pass begin on the thread).
        let mut passes: Vec<((u64, u32), usize)> = Vec::new();
        for d in &c.draws {
            match passes.iter_mut().find(|(k, _)| *k == (d.pass.0, d.pass.1)) {
                Some((_, n)) => *n += 1,
                None => passes.push(((d.pass.0, d.pass.1), 1)),
            }
        }
        let list: Vec<String> = passes.iter().map(|((p, s), n)| format!("{p:#x}/{s}: {n}")).collect();
        xnw_core::info!("world probe: frame {}: draws per pass (phase/stage): {}; calls per entry since the last probed frame: {}", c.frame, list.join(", "), entry_line());
        // Are instance translations absolute, or relative to the 32 m cell?
        let inst = c.draws.iter().filter(|d| d.inst.is_some()).count();
        let first = c.draws.iter().find_map(|d| Some((d.inst?.block, d.viewport)).filter(|(b, _)| *b != NO_PAL));
        if let Some((b, vp)) = first
            && let Some(block) = c.inst_list.get(b as usize)
            && let Some((_, cam)) = c.cameras.iter().find(|(v, _)| *v == vp)
        {
            let r = f32s::<16>(block);
            let campos = camera_position(&cam.view);
            let t = [r[3], r[7], r[11]];
            let dist = |o: [f32; 3]| ((t[0] - o[0]).powi(2) + (t[1] - o[1]).powi(2) + (t[2] - o[2]).powi(2)).sqrt();
            xnw_core::info!("world probe: frame {}: {inst} instanced draws, {} record blocks; first record translation {:?}, camera {:?}: distance {:.1} m as absolute, {:.1} m as camera-relative, viewport origin (+0x400) {:?}", c.frame, c.inst_list.len(), t, campos, dist(campos), dist([0.0; 3]), cam.origin);
        }
    }
    // effects.tsv
    {
        let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("effects.tsv"))?);
        writeln!(out, "effect\ttemplate\tvars\tname\tfx_0x60\ttemplate_0x70")?;
        let hex = |b: &[u8]| b.iter().map(|b| format!("{b:02x}")).collect::<String>();
        for (fx, e) in &c.effects {
            writeln!(out, "{fx:#x}\t{:#x}\t{}\t{}\t{}\t{}", e.template, e.vars, if e.name.is_empty() { "-" } else { &e.name }, hex(&e.head), hex(&e.template_head))?;
        }
    }
    // hex\: raw objects of the first distinct geometries, the first 4 KiB of
    // their vertex and meshlet buffers.
    if !lite {
        let hexdir = dir.join("hex");
        std::fs::create_dir_all(&hexdir)?;
        for (g, data) in &c.geoms {
            if data.dump.is_empty() {
                continue;
            }
            let mut out = std::io::BufWriter::new(std::fs::File::create(hexdir.join(format!("{g:x}.txt")))?);
            let mut parts: Vec<(&str, u64, &[u8])> = data.dump.iter().map(|(w, a, b)| (*w, *a, b.as_slice())).collect();
            let d = c.draws.iter().find(|d| d.geom == *g);
            for (what, at, b) in [("vertex-buffer", d.map_or(0, |d| d.vb_at), &data.vb), ("meshlet-buffer", d.map_or(0, |d| d.mesh_at), &data.mesh), ("index-buffer", d.map_or(0, |d| d.ib_at), &data.ib)] {
                if let Some(b) = b {
                    parts.push((what, at, &b[..b.len().min(4096)]));
                }
            }
            for (what, at, b) in parts {
                writeln!(out, "{what} at {at:#x}, {} bytes", b.len())?;
                for (k, row) in b.chunks(16).enumerate() {
                    let words: Vec<String> = row.chunks(4).map(|w| w.iter().rev().map(|x| format!("{x:02x}")).collect()).collect();
                    writeln!(out, "  {:04x}: {}", 16 * k, words.join(" "))?;
                }
            }
        }
    }
    // dumps.txt: texture objects, transform sources, palettes.
    {
        let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("dumps.txt"))?);
        for (what, at, b) in &c.dumps {
            writeln!(out, "{what} at {at:#x}, {} bytes", b.len())?;
            for (k, row) in b.chunks(16).enumerate() {
                let words: Vec<String> = row.chunks(4).map(|w| w.iter().rev().map(|x| format!("{x:02x}")).collect()).collect();
                writeln!(out, "  {:04x}: {}", 16 * k, words.join(" "))?;
            }
        }
    }
    // camera.txt
    {
        let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("camera.txt"))?);
        writeln!(out, "# viewports seen in frame {} (grcViewport: +0x40 view, +0x80 projection, +0x400 camera origin), most draws first", c.frame)?;
        let mut cams = c.cameras.clone();
        cams.sort_by_key(|(_, k)| std::cmp::Reverse(k.draws));
        for (vp, k) in &cams {
            writeln!(out, "viewport {vp:#x} draws {}", k.draws)?;
            writeln!(out, "  view {}", matrix_exact(&k.view))?;
            writeln!(out, "  proj {}", matrix_exact(&k.proj))?;
            writeln!(out, "  origin {:.4} {:.4} {:.4} {:.4}", k.origin[0], k.origin[1], k.origin[2], k.origin[3])?;
        }
    }
    let ok = meshes.values().filter(|r| r.is_ok()).count();
    let decoded_meshlets: u64 = meshes.values().filter_map(|r| r.as_ref().ok()).map(|m| u64::from(m.meshlets)).sum();
    let tris: usize = c.draws.iter().map(|d| mesh_of(d.geom).tris.len()).sum();
    let unread: Vec<String> = KINDS.iter().zip(c.unread).filter(|(_, n)| *n > 0).map(|(k, n)| format!("{k} {n}")).collect();
    let hook_ms = c.hook_ns as f64 / 1e6;
    xnw_core::info!(
        "world probe: frame {}: {} draws seen, {} captured (max {}), {} distinct geometries ({ok} decoded, {decoded_meshlets} meshlets), {tris} triangles drawn; unreadable: {}; {} MiB copied{}; transforms: {} ({} borrowed, {} none), world bounds {}; {} viewports, threads {:?}; hook {hook_ms:.1} ms ({:.1} us/draw); written in {:.1} s to {}",
        c.frame,
        c.seen,
        c.draws.len(),
        s.cfg.max,
        c.geoms.len(),
        if unread.is_empty() { "none".into() } else { unread.join(", ") },
        c.copied >> 20,
        if c.over_budget > 0 { format!(" ({} copies over the 1 GiB budget)", c.over_budget) } else { String::new() },
        stats.sources,
        stats.borrowed,
        stats.missing,
        stats.bounds.show(),
        c.cameras.len(),
        c.threads,
        hook_ms * 1000.0 / c.seen.max(1) as f64,
        t0.elapsed().as_secs_f64(),
        dir.display()
    );
    if stats.skipped > 0 {
        xnw_core::warn!("world probe: frame {}: {} draws left out of the world OBJ (over {OBJ_MAX_VERTICES} vertices)", c.frame, stats.skipped);
    }
    for e in errors {
        xnw_core::warn!("world probe: frame {}: decode failed: {e}", c.frame);
    }
    // Wheel and tyre effects (`*_wheel_inst`, `*_tyre_inst`): rigid
    // parts whose world already carries palette[model+0x33]; never skinned again.
    let mut wheels: Vec<String> = Vec::new();
    for d in &c.draws {
        if let Some(e) = c.effects.get(&d.effect)
            && (e.name.contains("wheel") || e.name.contains("tyre"))
            && wheels.len() < 12
        {
            wheels.push(format!("{} geom {:#x} model {:#x} skinned {} cb n {} src {:?}", e.name, d.geom, d.model, d.skinned, d.skin_cb[2], d.world_src));
        }
    }
    if !wheels.is_empty() {
        xnw_core::info!("world probe: frame {}: wheel/tyre draws (first 12): {}", c.frame, wheels.join("; "));
    }
    // The frame pack v2 (pack=2), before `complete`.
    if s.cfg.pack && lite {
        let seqs: Vec<u32> = c.draws.iter().map(|d| d.pack_seq).collect();
        let taken = c.pack.lock().ok().and_then(|mut p| p.take());
        pack::write(&dir, c.frame, &seqs, taken)?;
    }
    if lite && finish {
        seq_complete(&dir, c.frame, t0, c.draws.len(), c.hook_ns)?;
    }
    Ok(())
}

/// Screen mode with `seq=`: a dumped frame gets the light seq
/// files (draws.tsv, palbuf, LOD records, timecycle) from the live capture,
/// then the live list (list.tsv, blobs) from render_frame, then `complete`.
fn render_seq_frame(s: &State, c: Capture, cache: &mut xnw_world::GeometryCache<()>, dump: Option<&std::path::Path>) {
    let seq = s.cfg.seq && dump.is_some();
    let (frame, draws, hook_ns, t0) = (c.frame, c.draws.len(), c.hook_ns, Instant::now());
    if seq && let Err(e) = write_frame_with(s, &c, false) {
        xnw_core::warn!("world probe: writing failed: {e}");
        render_frame(c, cache, dump);
        return;
    }
    render_frame(c, cache, dump);
    if seq && let Err(e) = seq_complete(&s.cfg.dir.join(format!("frame-{frame}")), frame, t0, draws, hook_ns) {
        xnw_core::warn!("world probe: writing failed: {e}");
    }
}

/// Marks a seq frame complete (everything else is on disk) and logs it.
fn seq_complete(dir: &std::path::Path, frame: u64, t0: Instant, draws: usize, hook_ns: u64) -> std::io::Result<()> {
    let size = |d: &std::path::Path| std::fs::read_dir(d).map(|r| r.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum::<u64>()).unwrap_or(0);
    let bytes_written = size(dir) + size(&dir.join("blobs"));
    std::fs::write(dir.join("complete"), format!("{frame}\n"))?;
    xnw_core::info!("world probe: seq frame {frame} complete, {:.1} MB, {} ms ({draws} draws, hook {:.1} ms)", bytes_written as f64 / 1e6, t0.elapsed().as_millis(), hook_ns as f64 / 1e6);
    Ok(())
}









// ---- lists for our own world pass ----

use xnw_world::{Albedo, Camera as RCamera, DrawRecord, FrameList, GeomKey};

/// The viewports sharing the main camera's view and projection
/// (the main view's stages each have their own viewport object).
fn main_viewports(cameras: &[xnw_world::Camera], main: Option<u64>) -> Vec<u64> {
    let Some(m) = main.and_then(|m| cameras.iter().find(|c| c.viewport == m)) else { return Vec::new() };
    let same = |a: &[f32; 16], b: &[f32; 16]| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + x.abs()));
    cameras.iter().filter(|c| same(&c.view, &m.view) && same(&c.proj, &m.proj)).map(|c| c.viewport).collect()
}

/// A `name=on|off` switch of XNW_WORLD_RENDER (read once).
fn render_switch(name: &str, default: bool) -> bool {
    static OPTS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let v = OPTS.get_or_init(|| std::env::var("XNW_WORLD_RENDER").unwrap_or_default());
    switch_in(v, name, default)
}

fn switch_in(v: &str, name: &str, default: bool) -> bool {
    v.split(',').map(str::trim).filter_map(|p| p.strip_prefix(name)?.strip_prefix('=')).last().map_or(default, |s| s != "off")
}

/// The share of the screen an instance's bounding sphere covers:
/// `w` the camera-relative 3x3 + translation the pass draws, `t` its LOD
/// record (sphere centre +0x08, radius +0x14), `f` the projection's focal
/// scale; 1 when the camera is inside the sphere.
fn instance_cover(w: &[f32], t: &[f32; 35], f: f32) -> f32 {
    let s = (0..3).map(|r| (w[3 * r] * w[3 * r] + w[3 * r + 1] * w[3 * r + 1] + w[3 * r + 2] * w[3 * r + 2]).sqrt()).fold(0f32, f32::max);
    let c: [f32; 3] = std::array::from_fn(|k| t[2] * w[k] + t[3] * w[3 + k] + t[4] * w[6 + k] + w[9 + k]);
    let (d, rad) = ((c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt(), t[5] * s);
    if !(d.is_finite() && rad.is_finite()) {
        return 1.0;
    }
    if d <= rad {
        return 1.0;
    }
    let r = rad * f / d;
    (std::f32::consts::PI * r * r / 4.0).min(1.0)
}

/// One prefilter outcome row of instances-why.bin, 40 bytes little
/// endian: ghash u64, lod u32, first_lod u8, outcome u8, slot u8, pad u8,
/// instance u32, position f32 x3, distance f32, pad u32.
const WHY_ROW: usize = 40;

#[allow(clippy::too_many_arguments)]
fn why_row(out: &mut Vec<u8>, ghash: u64, lod: u32, first_lod: u8, idx: u32, pos: [f32; 3], dist: f32, code: u8, slot: u8) {
    out.extend_from_slice(&ghash.to_le_bytes());
    out.extend_from_slice(&lod.to_le_bytes());
    out.extend_from_slice(&[first_lod, code, slot, 0]);
    out.extend_from_slice(&idx.to_le_bytes());
    for v in pos {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&dist.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
}

/// Why a hooked draw has no list row (render_frame), one code per draw
/// in skips.tsv and the frame line; SKIP_NAMES[code] is the name both use.
const SKIP_LISTED: u8 = 0;
const SKIP_NO_BUFFERS: u8 = 1;
const SKIP_VB_NOT_READY: u8 = 2;
const SKIP_RETRY_CAP: u8 = 3;
const SKIP_UNREADABLE: u8 = 4;
const SKIP_INDICES: u8 = 5;
const SKIP_NOT_CACHED: u8 = 6;
const SKIP_REPEAT: u8 = 7;
const SKIP_INST_REFUSED: u8 = 8;
const SKIP_INST_NO_CAMERA: u8 = 9;
const SKIP_INST_NO_LOD_RECORD: u8 = 10;
const SKIP_INST_EMPTY: u8 = 11;
const SKIP_INST_HIDDEN: u8 = 12;
const SKIP_INST_FRUSTUM: u8 = 13;
const SKIP_INST_WRONG_LOD: u8 = 14;
const SKIP_INST_LOD_NONE: u8 = 15;
const SKIP_INST_INSANE: u8 = 16;
const SKIP_NO_WORLD: u8 = 17;
const SKIP_IMPLAUSIBLE_WORLD: u8 = 18;
const SKIP_NAMES: [&str; 19] = [
    "listed",
    "no_buffers",        // geometry key with vb_at or mesh_at 0
    "vb_not_ready",      // vertex header not 0x4042/0x3063 yet (vb_ready)
    "retry_cap",         // NOT_READY_RETRIES reads this frame used up
    "vb_unreadable",     // vertex buffer bytes not readable
    "indices",           // meshlets or index buffer unreadable or undecodable
    "not_cached",        // not in the recorder cache, no reason this frame
    "repeat",            // same stage, geometry, placement, state (repeatdrop)
    "inst_refused",      // instance batch refused at the hook (no record block)
    "inst_no_camera",    // batch viewport has no captured camera
    "inst_no_lod_record",// no LOD record and not the first LOD's call
    "inst_empty",        // record block holds no records
    "inst_hidden",       // prefilter kept none: most hidden by the batch mask
    "inst_frustum",      // ... most outside the frustum
    "inst_wrong_lod",    // ... most another LOD's
    "inst_lod_none",     // ... most past every LOD (faded or out of range)
    "inst_insane",       // ... most failing the record sanity rule
    "no_world",          // no world matrix read
    "implausible_world", // world fails xnw_frame::world::plausible
];

/// The reason a prefiltered batch drew nothing: its most common
/// outcome (n tested, h hidden, counts = [outside, wrong LOD, insane]; the
/// rest found no LOD slot). Ties go to the earlier prefilter test.
fn inst_zero_reason(n: u32, h: u32, counts: [u32; 3]) -> u8 {
    if n == 0 {
        return SKIP_INST_EMPTY;
    }
    let none = n.saturating_sub(h + counts.iter().sum::<u32>());
    let c = [(h, SKIP_INST_HIDDEN), (counts[2], SKIP_INST_INSANE), (counts[0], SKIP_INST_FRUSTUM), (counts[1], SKIP_INST_WRONG_LOD), (none, SKIP_INST_LOD_NONE)];
    c.iter().fold((0, SKIP_INST_LOD_NONE), |a, &(v, k)| if v > a.0 { (v, k) } else { a }).1
}

/// skips.tsv: one row per hooked draw (the draws.tsv row), its code and name.
fn skip_rows(skip: &[u8]) -> String {
    let mut out = String::from("draw\tcode\treason\n");
    for (i, k) in skip.iter().enumerate() {
        let _ = writeln!(out, "{i}\t{k}\t{}", SKIP_NAMES.get(*k as usize).unwrap_or(&"?"));
    }
    out
}

/// The frame line: draws without a list row, then per reason (non-zero only).
fn skip_line(skip: &[u8]) -> String {
    let mut n = [0u32; SKIP_NAMES.len()];
    for k in skip {
        n[(*k as usize).min(SKIP_NAMES.len() - 1)] += 1;
    }
    let parts: Vec<String> = (1..n.len()).filter(|&k| n[k] > 0).map(|k| format!("{} {}", SKIP_NAMES[k], n[k])).collect();
    format!("{} of {} ({})", skip.len() as u32 - n[0], skip.len(), if parts.is_empty() { "none".into() } else { parts.join(", ") })
}

/// What makes a submission a repeat: geometry, viewport, stage,
/// state indices (blend, depth-stencil, raster, stencil ref), sample mask, and
/// the placement (world bits, or the instance block with its LOD).
type RepeatKey = (GeomKey, u64, u32, [u16; 4], u64, [u32; 16], (u32, u32, u32));

fn repeat_key(d: &Draw, key: GeomKey) -> RepeatKey {
    let world = d.world.map_or([0; 16], |w| std::array::from_fn(|i| w[i].to_bits()));
    let inst = d.inst.map_or((u32::MAX, 0, 0), |i| (i.block, i.lod, i.lodrec));
    (key, d.viewport, stage_of(d), d.state, d.stipple, world, inst)
}

fn key_of(d: &Draw) -> GeomKey {
    GeomKey { mesh_at: d.mesh_at, mesh_len: d.mesh_len, vb_at: d.vb_at, vb_len: d.vb_len, ib: if d.mesh_len == 0 { d.ib_len } else { 0 }, hash: d.ghash }
}

/// A texture's pack identity: its origin (file, texture) and the file's
/// texture-pack id; else a flat colour of the effect.
fn albedo(tex_at: u64, effect: u64) -> Albedo {
    let pack = (tex_at != 0).then(|| xnw_formats::origin::origin_of(tex_at, 0x1000)).flatten().and_then(|o| Some((xnw_formats::origin::file(o.file)?.pack_id?, o.texture)));
    match pack {
        Some((file, texture)) => Albedo::Pack { file, texture },
        None => Albedo::Flat(xnw_world::flat_colour(effect)),
    }
}

/// Turns a captured frame into the pass's list: each new geometry once (its
/// title vertex buffer and meshlets copied here, on the writer thread, and
/// the meshlets expanded to vertex-buffer indices; the vertex decode is the
/// pass's shader), one record per draw with a transform, the frame's
/// palettes; published for the presenter or anyone taking lists.
fn render_frame(c: Capture, cache: &mut xnw_world::GeometryCache<()>, dump: Option<&std::path::Path>) {
    use xnw_world::NO_PALETTE;
    let t0 = Instant::now();
    let mut list = FrameList { frame: c.frame, cameras: c.cameras.iter().map(|(v, k)| RCamera { viewport: *v, view: k.view, proj: k.proj }).collect(), ..Default::default() };
    // The timecycle block (render copy, 0x16C0 B): the sun.
    let tc = STATE.get().and_then(|s| bytes(real(s.title, TIMECYCLE), 0x16C0));
    list.sun = tc.as_deref().and_then(sun_of);
    list.sky = tc.as_deref().and_then(sky_of);
    list.light = tc.as_deref().map(|b| light_of(b, list.sky.as_ref(), true));
    if let Some(s) = STATE.get()
        && let Some(grid) = water_now(s.title)
        && let Some(cam) = list.main_camera()
    {
        list.water = Some(xnw_world::Water { grid, origin: xnw_world::camera_position(&cam.view), tint: water_tint(list.sun) });
    }
    // The ocean wave ring the CPU samples (worldwater.rs, XNW_WATER_SIM).
    if let Some(s) = STATE.get() {
        crate::worldwater::tick(s.title, c.frame);
    }
    // The heightmap terrain's tile records (worldterrain.rs).
    if crate::worldterrain::enabled()
        && let Some(s) = STATE.get()
    {
        list.terrain = crate::worldterrain::now(s.title);
    }
    // The imposter submits of this frame (`imposters=on`).
    if imposter::on() {
        list.imposters = imposter::take(c.frame);
    }
    let skinlog = STATE.get().is_some_and(|s| s.cfg.skinlog > 0);
    // The slow-frame line's parts before the draw loop: the frame state
    // reads above, the new geometries' reads, the drains of the ahead rings.
    let state_ms = t0.elapsed().as_secs_f64() * 1e3;
    let tg = Instant::now();
    // Why a hooked draw gets no list row, per draw (SKIP_NAMES; 0 listed),
    // and per geometry why it is not in the cache.
    let mut skip = vec![SKIP_LISTED; c.draws.len()];
    let mut geom_why: HashMap<GeomKey, u8> = HashMap::new();
    let taken = geomread::take(&c.draws, c.frame, cache, &mut list.new_geoms, &mut geom_why, skinlog, geomread::asynchronous() && dump.is_none(), geomread::direct());
    let (copied, failed, not_ready) = (taken.copied, taken.failed, taken.not_ready);
    // Drawables placed by the streamer since the last frame: their
    // geometries offered ahead of any draw (worldahead.rs).
    let geom_ms = tg.elapsed().as_secs_f64() * 1e3;
    let ta = Instant::now();
    if crate::worldahead::enabled() {
        crate::worldahead::drain(c.frame, &mut list, &mut |k| xnw_world::geometry::known().wants(k, c.frame));
        crate::worldahead::drain_events(c.frame);
        let cam = list.main_camera().map(|c| xnw_world::camera_position(&c.view));
        crate::worldahead::drain_attaches(c.frame, cam);
    }
    let ahead_ms = ta.elapsed().as_secs_f64() * 1e3;
    // The frame's skinned spans, copied as one window of the
    // title's ring buffer (or span by span); each draw's bones are rebased to it.
    let skin = STATE.get().map(|s| skin_copy_live(s, &c.draws)).unwrap_or_default();
    if !skin.at.is_empty() {
        list.palettes.push(skin.floats.clone());
    }
    // `drawbones=on`: the bones copied at each draw, one palette marked draw-time.
    let hook_pal = (!c.hook_bones.floats.is_empty()).then(|| {
        list.palette_at_draw.resize(list.palettes.len(), false);
        list.palettes.push(c.hook_bones.floats.clone());
        list.palette_at_draw.push(true);
        (list.palettes.len() - 1) as u32
    });
    // Vehicle rules: per (palette, mask words) one index array after the palette, +0x54 picking the start.
    let mut veh_tables = skinhook::VehTables::default();
    let mut veh_rows = 0u32;
    let mut wheel_rows = 0u32;
    // Rows with an instanced draw's reads; of them left as they were: a remap already, no palette or instances, palette too far.
    let mut wheel_left = [0u32; 4];
    // The instanced wheels' own palette (made at the first such row): their
    // bones read at the sink, each row's remap relative to its own bones.
    let mut wheel_pal: Option<u32> = None;
    let mut states = StateTables::default();
    // Instance blocks as camera-relative worlds, filtered per call by the
    // LOD rule, once per (block, viewport, lod, LOD record).
    // The third value is why a batch draws nothing (SKIP_*; 0 when it draws).
    // The fourth is the batch in `list.cull` (gpucull=on; NO_CULL: drawn from the CPU's instances).
    let mut inst_at: HashMap<(u32, u64, u32, u32), (u32, u32, u8, u32)> = HashMap::new();
    // `gpucull=on` (PERF3): the batch test stays here, the per-instance test and
    // the worlds are made by the pass's compute; a dumped frame keeps the CPU
    // path (its list.tsv and instances.bin are the replay's input).
    let gpucull = render_switch("gpucull", false) && dump.is_none();
    // `hair=on` (HAIR2, default off): the strand chain runs in the replay only; the live path is a stub.
    static HAIR_STUB: std::sync::Once = std::sync::Once::new();
    if render_switch("hair", false) {
        HAIR_STUB.call_once(|| xnw_core::info!("world: hair=on: the strand chain and strand draws run in the replay only (world-render); live path stubbed while the card is frozen"));
    }
    // The title's multi-pass techniques (0x1451AD540: one draw
    // per technique pass, passes[base+k] with its own program set) submit the
    // same geometry, placement, state and sample mask again; our pass draws
    // one material program, so a repeat is overdraw with equal results. Same
    // stage only; a repeat with any other state is drawn.
    let mut repeat_seen: std::collections::HashSet<RepeatKey> = std::collections::HashSet::new();
    let mut repeats = 0u32;
    let repeat_drop = render_switch("repeatdrop", true);
    let ti = Instant::now();
    let (mut batches, mut tested, mut drawn, mut no_lod, mut hidden) = (0u32, 0u32, 0u32, 0u32, 0u32);
    // Per instance the prefilter's outcome, on dumped frames
    // (in one capture 24-34 ms a frame as text at 40-58k instances): binary
    // rows of WHY_ROW bytes (see why_row), tools/why_tsv.py makes the old TSV.
    let mut inst_why: Vec<u8> = Vec::new();
    // Batches from the static map list (kind S, desc+0x68) among them.
    let mut gpu_batches = 0u32;
    let (mut outside, mut wrong_lod) = (0u32, 0u32);
    // Records failing the sanity rule, by block source (cpu-list, gpu-buffer list, buffer view).
    let mut insane = [0u32; 3];
    // (Instances covering > 25 % of their view, the first few named).
    let mut big_cover: (u32, Vec<String>) = (0, Vec::new());
    // Where the list-building time goes: [instance prefilter, outcome
    // dump formatting, everything else], in ns (10.46 ms "instances"
    // covered the whole loop, every draw).
    let mut split = [0u64; 3];
    // Distinct record blocks and their records (about 79,000 expected), over all
    // passes and in the main view's colour stages (66-82); prefilter tested
    // and drawn in the main view.
    // The main view is known only after the loop (list.draws picks it; one
    // viewport object per stage shares its camera): per viewport here.
    let mut seen_all = std::collections::HashSet::new();
    let mut recs_all = 0usize;
    let mut per_vp: HashMap<u64, (u32, u32)> = HashMap::new();
    // Albedo per (texture, effect) once a frame: origin_of walks the
    // texture registry (shared counters, per-file lookup) on every call.
    let mut albedos: HashMap<(u64, u64), Albedo> = HashMap::new();
    // Each list row's v1 draw (draws.tsv index), as list-draws.tsv.
    let mut list_v1: Vec<usize> = Vec::new();
    let v1_of = |d: &Draw| (d as *const Draw as usize - c.draws.as_ptr() as usize) / std::mem::size_of::<Draw>();
    let identity: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    for (di, d) in c.draws.iter().enumerate() {
        let key = key_of(d);
        if cache.get_or_make(key, c.frame, || None).is_none() {
            skip[di] = geom_why.get(&key).copied().unwrap_or(SKIP_NOT_CACHED);
            continue;
        }
        if repeat_drop && !repeat_seen.insert(repeat_key(d, key)) {
            repeats += 1;
            skip[di] = SKIP_REPEAT;
            continue;
        }
        if let Some(i) = d.inst {
            let Some(block) = c.inst_list.get(i.block as usize) else {
                skip[di] = SKIP_INST_REFUSED;
                continue;
            };
            if seen_all.insert(i.block) {
                recs_all += block.len() / 64;
            }
            let Some((_, cam)) = c.cameras.iter().find(|(v, _)| *v == d.viewport) else {
                skip[di] = SKIP_INST_NO_CAMERA;
                continue;
            };
            let (first, count, why_none, cull_batch) = *inst_at.entry((i.block, d.viewport, i.lod, i.lodrec)).or_insert_with(|| {
                let origin = camera_position(&cam.view);
                let first = (list.instances.len() / 12) as u32;
                let t = c.lod_list.get(i.lodrec as usize).map(|b| f32s::<35>(b));
                let slot = (i.first_lod != NO_LOD && i.lod >= u32::from(i.first_lod)).then(|| (i.lod - u32::from(i.first_lod)) as usize);
                let lod = match (&t, slot) {
                    (Some(t), Some(s)) => Some((t, s)),
                    _ => None,
                };
                if lod.is_none() {
                    no_lod += 1;
                    // Without a LOD record only the first LOD's call draws (no stacking).
                    if i.first_lod != NO_LOD && i.lod != u32::from(i.first_lod) {
                        return (first, 0, SKIP_INST_NO_LOD_RECORD, NO_CULL);
                    }
                }
                let room = (4 * xnw_world::MAX_INSTANCES as usize).saturating_sub(list.instances.len() / 12);
                let hide = c.mask_list.get(i.mask as usize).map(Vec::as_slice);
                let mut why = dump.map(|_| Vec::new());
                // Our camera's planes (the proj as captured; the pass drops the jitter for drawing).
                let planes = frustum_planes(&xnw_world::Camera { viewport: 0, view: cam.view, proj: cam.proj }.view_proj_relative());
                let mut counts = [0u32; 3];
                let tp = Instant::now();
                if gpucull {
                    let n = (block.len() / 64).min(MAX_BATCH as usize) as u32;
                    let verdict = batch_test(block, n, lod, origin, &planes, &mut counts);
                    let pushed = verdict.and_then(|need_planes| {
                        // Kind S and grass blocks are the same allocation across frames (kept on the device).
                        let keep = c.inst_src.get(i.block as usize).is_some_and(|s| *s != 0);
                        list.cull.push(xnw_world::gpucull::Batch { records: block.clone(), len: n, keep, lod: lod.map(|(t, s)| (*t, s as u32)), hide: hide.map(<[u32]>::to_vec).unwrap_or_default(), origin, planes: need_planes.then_some(planes), out_first: 0 })
                    });
                    // A full cull frame falls back to the CPU test below.
                    if verdict.is_none() || pushed.is_some() {
                        split[0] += tp.elapsed().as_nanos() as u64;
                        outside += counts[0];
                        wrong_lod += counts[1];
                        batches += 1;
                        gpu_batches += u32::from(i.gpu != 0);
                        tested += n;
                        per_vp.entry(d.viewport).or_default().0 += n;
                        return match pushed {
                            Some(b) => (list.cull.batches[b as usize].out_first, n, SKIP_LISTED, b),
                            None => (first, 0, inst_zero_reason(n, 0, counts), NO_CULL),
                        };
                    }
                }
                let (n, k, h) = prefilter_batch(block, block.len().min(64 * room), lod, hide, origin, &mut list.instances, why.as_mut(), &planes, &mut counts);
                split[0] += tp.elapsed().as_nanos() as u64;
                // Drawn instances whose bounding sphere (LOD record
                // radius x scale) covers more than a quarter of the screen.
                if let Some((t, _)) = lod {
                    let f = cam.proj[0].abs().max(cam.proj[5].abs());
                    for j in first as usize..first as usize + k as usize {
                        let Some(w) = list.instances.get(12 * j..12 * j + 12) else { break };
                        if instance_cover(w, t, f) > xnw_frame::audit::BIG_COVER {
                            big_cover.0 += 1;
                            if big_cover.1.len() < 4 {
                                big_cover.1.push(format!("stage {} {:#x} fx {:#x}", stage_of(d), d.ghash, d.shader.max(d.effect)));
                            }
                        }
                    }
                }
                outside += counts[0];
                wrong_lod += counts[1];
                insane[c.inst_src.get(i.block as usize).copied().unwrap_or(0).min(2) as usize] += counts[2];
                let tw = Instant::now();
                if let Some(w) = why {
                    inst_why.reserve(w.len() * WHY_ROW);
                    for (idx, code, dist, slot) in w {
                        let r = f32s::<16>(&block[64 * idx as usize..]);
                        why_row(&mut inst_why, d.ghash, i.lod, i.first_lod, idx, [r[3], r[7], r[11]], dist, code, slot);
                    }
                }
                split[1] += tw.elapsed().as_nanos() as u64;
                hidden += h;
                batches += 1;
                gpu_batches += u32::from(i.gpu != 0);
                tested += n;
                drawn += k;
                let v = per_vp.entry(d.viewport).or_default();
                v.0 += n;
                v.1 += k;
                (first, k, if k == 0 { inst_zero_reason(n, h, counts) } else { SKIP_LISTED }, NO_CULL)
            });
            if count == 0 {
                skip[di] = why_none;
                continue;
            }
            if cull_batch != NO_CULL {
                list.cull.rows.push((list.draws.len() as u32, cull_batch));
            }
            list.draws.push(DrawRecord { geom: key, world: identity, viewport: d.viewport, effect: d.shader, albedo: Albedo::Flat(xnw_world::flat_colour(d.shader)), palette: NO_PALETTE, stage: stage_of(d), inst_first: first, inst_count: count, bone_first: 0, bone_count: 0, material: xnw_world::Material { albedo_desc: d.tex_desc, alpha_clip: d.alpha_clip, layers: d.layer_desc, stipple: xnw_world::stipple_lanes(d.stipple), state_index: [d.state[0], d.state[1], d.state[2]], ..Default::default() }, state: states.get(d), remap: xnw_world::NO_REMAP, ped: false });
            list_v1.push(v1_of(d));
            continue;
        }
        let Some(w) = d.world else {
            skip[di] = SKIP_NO_WORLD;
            continue;
        };
        let world: [f32; 16] = w[..16].try_into().unwrap();
        if !plausible(&world) {
            skip[di] = SKIP_IMPLAUSIBLE_WORLD;
            continue;
        }
        // Rigid parts: the bone travels with the list (rigid.tsv); composed
        // only with `rigid=on` until a capture shows the hook's world lacks it
        // (wheel/tyre worlds were found that already carry it).
        if let Some(b) = d.rigid {
            list.rigid.push((list.draws.len() as u32, b));
        }
        let world = match d.rigid {
            Some(b) if rigid_on() => rigid_world(&b, &world),
            _ => world,
        };
        list.skin_hooked += u32::from(d.skinned && (1..=4096).contains(&d.skin_cb[2]));
        let (palette, bone_first, bone_count) = skin_of(d, &skin);
        let remap = if palette == 0 && d.ped_skin.0 != 0 { skin.remap_at.get(&(d.ped_skin.1, d.remap[0])).copied().unwrap_or(xnw_world::NO_REMAP) } else { xnw_world::NO_REMAP };
        // Draw-time bones replace the end-of-frame ones for a skinned draw
        // without a ped skin record (a remap's entries live in the end-of-frame palette);
        // the ped flag stays as the end-of-frame values set it.
        let ped = d.ped_skin.0 != 0 || (palette == 0 && bone_count > 8);
        let (palette, bone_first, bone_count) = match (hook_pal, d.hook_bones) {
            (Some(p), Some((f, n))) if d.skinned && d.ped_skin.0 == 0 => (p, f, n),
            _ => (palette, bone_first, bone_count),
        };
        // Vehicle rules: the bone index through rec+0x54 and the mask|rank words (a vehicle
        // program, so no ped weight normalisation).
        // An instanced draw read at its sink (skinrules bit 8) is not given the
        // vehicle rules: its hook record is the body's (pk5 f977 draws 223-227:
        // cur_idx 219899 as the body rows 90-93), so +0x74 there is the body's
        // mask, while the sink's own record has none (`wheel_reads`, mask on 0).
        let (remap, ped) = match d.veh {
            Some((w, off)) if d.wheel.is_none() && remap == xnw_world::NO_REMAP && (palette as usize) < list.palettes.len() => {
                let words = c.hook_bones.masks.words.get(w as usize..w as usize + skinhook::MASK_WORDS).unwrap_or(&[]);
                let rules = skinhook::rules();
                let at = veh_tables.remap(palette, &mut list.palettes[palette as usize], words, off, rules);
                veh_rows += 1;
                (at, false)
            }
            _ => (remap, ped),
        };
        // Instanced vehicle draws (skinrules bit 8): one instance per record read at the
        // sink, the bones at the draw's palette base after the palette, reached through a
        // remap relative to the row's first bone.
        let mut wr = WheelRow { world, palette, bone_first, bone_count, remap, inst_first: 0, inst_count: 0 };
        if let Some(w) = d.wheel.as_deref() {
            place_wheel(&mut list, &mut wheel_pal, w, &mut wr, &mut wheel_left, &mut wheel_rows);
        }
        let WheelRow { world, palette, bone_first, bone_count, remap, inst_first, inst_count } = wr;
        let tex = d.diffuse.map_or(0, |(_, _, _, t)| t.at);
        let ta = Instant::now();
        let alb = (tex != 0).then(|| *albedos.entry((tex, d.effect)).or_insert_with(|| albedo(tex, d.effect)));
        split[2] += ta.elapsed().as_nanos() as u64;
        list.draws.push(DrawRecord { geom: key, world, viewport: d.viewport, effect: d.shader.max(d.effect), albedo: if let Some(a) = alb { a } else { Albedo::Flat(xnw_world::flat_colour(d.shader)) }, palette, stage: stage_of(d), inst_first, inst_count, bone_first, bone_count, material: xnw_world::Material { albedo_desc: d.tex_desc, alpha_clip: d.alpha_clip, layers: d.layer_desc, stipple: xnw_world::stipple_lanes(d.stipple), state_index: [d.state[0], d.state[1], d.state[2]], ..Default::default() }, state: states.get(d), remap, ped });
        list_v1.push(v1_of(d));
    }
    if skinlog && c.frame % 60 == 0 {
        skin_log(&c, &skin);
    }
    if c.frame % 60 == 0 || (skin.refused > 0 && skin.at.is_empty()) {
        xnw_core::info!(
            "world render: frame {}: skin copy {} bones ({}), {} spans placed, {} refused, {:.2} ms; ped remap: {} bases, {} entries",
            c.frame,
            skin.bones,
            skin.window.map_or("span by span".to_string(), |(f, n)| format!("window {f}..{} ({n} bones)", f + n)),
            skin.at.len(),
            skin.refused,
            skin.ms,
            skin.remap_at.len(),
            skin.remap_entries
        );
    }
    if c.frame % 60 == 0 && skinhook::on() {
        xnw_core::info!("{}", skinhook::line(c.frame, &list, list.skin_hooked, &c.hook_bones));
    }
    let rules = skinhook::rules();
    if c.frame % 60 == 0 && rules & (skinhook::VEH_OFFSET | skinhook::VEH_MASK) != 0 {
        xnw_core::info!("{}", skinhook::vehicle_line(c.frame, rules, veh_rows, &c.hook_bones.masks, &veh_tables));
    }
    if c.frame % 60 == 0 && rules & skinhook::VEH_INST != 0 {
        xnw_core::info!("{}", skinhook::wheel_line(c.frame, wheel_rows, wheel_left));
    }
    let inst_ms = ti.elapsed().as_secs_f64() * 1e3;
    if c.frame % 60 == 0 {
        // The split of that time.
        let (pf, wd, al) = (split[0] as f64 / 1e6, split[1] as f64 / 1e6, split[2] as f64 / 1e6);
        xnw_core::info!("world render: frame {}: list build {inst_ms:.2} ms: instance prefilter {pf:.2} ms ({tested} tested), outcome dump {wd:.2} ms, albedo lookup {al:.2} ms ({} textures), other draws and records {:.2} ms ({} draws)", c.frame, albedos.len(), (inst_ms - pf - wd - al).max(0.0), c.draws.len());
        xnw_core::info!("world render: frame {}: big-cover instances (sphere over 25 % of the screen): {} {:?}", c.frame, big_cover.0, big_cover.1);
        xnw_core::info!("world render: frame {}: repeated submissions dropped (same stage, geometry, placement, state, sample mask): {repeats}", c.frame);
        // The main view by camera (view and projection), as main_draws.
        let mains = main_viewports(&list.cameras, list.main_camera().map(|m| m.viewport));
        let mut seen_main = std::collections::HashSet::new();
        let mut recs_main = 0usize;
        for d in &c.draws {
            if let Some(i) = d.inst
                && mains.contains(&d.viewport)
                && (66..=82).contains(&stage_of(d))
                && seen_main.insert(i.block)
            {
                recs_main += c.inst_list.get(i.block as usize).map_or(0, |b| b.len() / 64);
            }
        }
        let (main_tested, main_drawn) = per_vp.iter().filter(|(v, _)| mains.contains(v)).fold((0u32, 0u32), |a, (_, t)| (a.0 + t.0, a.1 + t.1));
        xnw_core::info!("world render: frame {}: distinct instance records: main view stages 66-82 {recs_main} ({} blocks, {} viewports), all passes {recs_all} ({} blocks); main view prefilter: tested {main_tested}, drawn {main_drawn}", c.frame, seen_main.len(), mains.len(), seen_all.len());
    }
    if let Some(dir) = dump {
        let fdir = dir.join(format!("frame-{}", c.frame));
        let _ = std::fs::create_dir_all(&fdir);
        let _ = std::fs::write(fdir.join("instances-why.bin"), &inst_why);
        // Per hooked draw (draws.tsv row) why it has no list row.
        let _ = std::fs::write(fdir.join("skips.tsv"), skip_rows(&skip));
    }
    if c.frame % 60 == 0 || dump.is_some() {
        xnw_core::info!("world render: frame {}: hooked draws without a list row: {}", c.frame, skip_line(&skip));
    }
    if !c.census.is_empty() {
        let write = c.frame % 600 == 0 || dump.is_some();
        census_add(&c.census, STATE.get().filter(|_| write).map(|s| s.cfg.dir.as_path()));
    }
    // The hook runs on the title's threads: a slow one is a slow frame.
    let (hook_ms, list_ms) = (c.hook_ns as f64 / 1e6, t0.elapsed().as_secs_f64() * 1e3);
    xnw_world::cost_hook(hook_ms);
    let parts = c.parts;
    if hook_ms > 50.0 || list_ms > 50.0 {
        xnw_core::warn!(
            "world render: slow frame {}: hook {hook_ms:.1} ms on the title's threads ({} draws, {} new geometries read), list {list_ms:.1} ms off them ({} new geometries copied, {} KiB; skin {:.1} ms; instances {inst_ms:.1} ms; frame state {state_ms:.1} ms, geometry reads {geom_ms:.1} ms ({}{}), ahead drains {ahead_ms:.1} ms)",
            c.frame,
            c.draws.len(),
            c.geom_misses,
            list.new_geoms.len(),
            copied >> 10,
            skin.ms,
            taken.parts.line(),
            if taken.sent + taken.arrived > 0 { format!("; reader: sent {}, arrived {}, waited {} frames at most", taken.sent, taken.arrived, taken.waited) } else { String::new() }
        );
    }
    if geomread::asynchronous() && c.frame % 60 == 0 {
        let (p, flight) = geomread::stats();
        xnw_core::info!("world render: frame {}: geometry reader (60 frames): {}; in flight {flight}", c.frame, p.line());
    }
    // Main-view keys hooked but given no row (ours vs the title's drops).
    let main_vps = list.main_viewports();
    let skipped: HashMap<(u64, u32), u8> = c.draws.iter().zip(&skip).filter(|(d, r)| **r != SKIP_LISTED && main_vps.contains(&d.viewport)).map(|(d, r)| ((key_of(d).hash, stage_of(d)), *r)).collect();
    coherence(&list, skipped);
    cache.evict(c.frame, 600);
    let evicted = cache.evicted;
    // The recorder's own maps, bounded (cleared past a cap: they are
    // caches of header reads, refilled on demand), counted every 600 frames.
    let sizes = bound_maps();
    if c.frame % 600 == 0 {
        xnw_core::info!("world render: memory (recorder): geometry headers {}, models {}, bone maps {}, dumped {}, recorder cache {}, hand-shake keys {}", sizes[0], sizes[1], sizes[2], sizes[3], cache.len(), xnw_world::geometry::known().len());
    }
    xnw_world::geometry::known().sweep(c.frame, 600);
    let (skin_draws, skin_agree, skin_extent) = (c.skin_draws, c.skin_agree, c.skin_extent);
    let (main, frame, draws, palettes) = (list.main_draws().count(), list.frame, list.draws.len(), list.palettes.len());
    let list_sun = list.sun;
    let new = list.new_geoms.len();
    // Census: draws per (stage, blend, depth-stencil, raster).
    let mut census: HashMap<(u32, [u16; 3]), usize> = HashMap::new();
    for d in &c.draws {
        *census.entry((d.pass.1, [d.state[0], d.state[1], d.state[2]])).or_default() += 1;
    }
    let mut census: Vec<_> = census.into_iter().collect();
    census.sort_by_key(|(k, n)| (std::cmp::Reverse(*n), *k));
    if frame_has_dump(dump, list.frame)
        && let Some(dir) = dump
    {
        let fdir = dir.join(format!("frame-{}", list.frame));
        let _ = std::fs::create_dir_all(&fdir);
        let text: String = census.iter().map(|((s, k), n)| format!("{s}\t{}\t{}\t{}\t{n}\n", k[0], k[1], k[2])).collect();
        let _ = std::fs::write(fdir.join("census.tsv"), format!("stage\tblend\tdepth_stencil\traster\tdraws\n{text}"));
        if let Some(b) = &tc {
            let _ = std::fs::write(fdir.join("timecycle.bin"), b);
            // The lighting values used for this frame (also derivable from timecycle.bin).
            if let Some(l) = list.light {
                let _ = std::fs::write(fdir.join("lighting.txt"), format!("{}\n", l.line()));
            }
        }
    }
    let td = Instant::now();
    if let Some(dir) = dump
        && let Err(e) = dump_list(dir, &list).and_then(|()| std::fs::write(dir.join(format!("frame-{}", list.frame)).join("list-draws.tsv"), list_v1.iter().enumerate().map(|(r, d)| format!("{r}\t{d}\n")).fold(String::from("row\tdraw\n"), |a, l| a + &l)))
    {
        xnw_core::warn!("world render: frame {frame}: list dump failed: {e}");
    }
    let dump_ms = td.elapsed().as_secs_f64() * 1e3;
    if dump.is_some() {
        // The probe's file cost apart from the always-on hook (hook = capture only).
        xnw_core::info!("world render: frame {frame}: probe write {dump_ms:.1} ms (hook {:.2} ms, list {:.2} ms)", c.hook_ns as f64 / 1e6, (td - t0).as_secs_f64() * 1e3);
    }
    // Only a pass on screen takes lists: in list-only mode (XNW_WORLD_RENDER=1)
    // a published list would pile up its new geometries forever.
    if xnw_world::screen_mode().is_some() {
        // The ped pre-skin (`preskin=on`): this frame's items and the rows that take them.
        if preskin::on() {
            // The join key is the draw's own record read at its sink, not the hook's.
            let draw_of = |row: usize| list_v1.get(row).and_then(|&i| c.draws.get(i));
            list.preskin = preskin::frame_for(list.frame, &list, |row| draw_of(row).and_then(|d| d.sink_range).unwrap_or(u32::MAX));
            if frame % 60 == 0 {
                let words: Vec<(Option<u32>, Option<u32>)> = (0..list.draws.len()).filter_map(draw_of).map(|d| (d.sink_range, d.range)).collect();
                xnw_core::info!("{}", preskin::join_line(frame, &words, list.preskin.draws.len()));
            }
        }
        // `bridgeps`: the rows whose sinks were read, for the title's own programs in our pass.
        if SINK_BRIDGE.load(Ordering::Relaxed) {
            let rows: Vec<(u32, xnw_world::bridged::Owned)> = list_v1.iter().enumerate().filter_map(|(row, &i)| c.draws.get(i)?.bridged.as_deref().map(|o| (row as u32, o.clone()))).collect();
            let (not_ready, unread) = bridgeps::take_counts();
            let mut f = xnw_world::bridged::Frame::new(rows);
            (f.not_ready, f.unread) = (not_ready, unread);
            if frame % 60 == 0 {
                xnw_core::info!("{}", bridgeps::reads_line(frame, f.rows.len()));
            }
            list.bridged = Some(std::sync::Arc::new(f));
        }
        // `preskinbits` bit 3: ped-flagged rows without a ped remap are props (after the join, which takes the flag).
        if preskin::bits() & xnw_world::preskin_gpu::BIT_PROPS != 0 {
            let n = xnw_world::preskin_gpu::props_unflag(&mut list);
            if frame % 60 == 0 {
                xnw_core::info!("world render: frame {frame}: props: {n} ped-flagged rows without a ped remap drawn as skinned props");
            }
        }
        xnw_world::lists().publish(list);
    }
    if frame % 60 == 0 {
        let (published, replaced) = xnw_world::lists().counts();
        let mut passes: Vec<((u64, u32), usize)> = Vec::new();
        for d in &c.draws {
            match passes.iter_mut().find(|(k, _)| *k == (d.pass.0, d.pass.1)) {
                Some((_, n)) => *n += 1,
                None => passes.push(((d.pass.0, d.pass.1), 1)),
            }
        }
        let passes: Vec<String> = passes.iter().map(|((p, s), n)| format!("{p:#x}/{s} {n}")).collect();
        xnw_core::info!(
            "world render: frame {frame}: {} draws seen, {draws} recorded ({main} main view), {palettes} palettes, skinned draws {skin_draws} ({skin_agree} agree with the TLS pointer, extent {skin_extent} bones); {new} new geometries ({} KiB copied, {failed} failed, geometry not ready {not_ready}), cache {} ({evicted} evicted); hook {:.2} ms ({} B read: palettes), list {:.2} ms; lists {published} published, {replaced} replaced; passes {}; entries (60 frames) {}; instanced batches {batches} (cpu-list {}, gpu-buffer {gpu_batches}; instanced calls not recorded: {} without TLS, {} geometry unreadable), ({no_lod} without a LOD record), instances: tested {tested}, outside frustum {outside}, wrong LOD {wrong_lod}, hidden {hidden}, drawn {drawn}; {inst_ms:.2} ms",
            c.seen,
            copied >> 10,
            cache.len(),
            c.hook_ns as f64 / 1e6,
            c.copied,
            t0.elapsed().as_secs_f64() * 1e3,
            passes.join(", "),
            entry_line(),
            batches - gpu_batches,
            INST_LOST[0].load(Ordering::Relaxed),
            INST_LOST[1].load(Ordering::Relaxed)
        );
        // This frame's instance calls and records (whether the expected ~79,000 are reached).
        let k = c.inst_calls;
        xnw_core::info!(
            "world render: frame {frame}: instance calls: accepted {} (cpu-list {}, gpu-buffer {}), refused {} (null {}, count {}, unreadable {}), via the buffer view (grass lists, or instfallback=on) {}; records: total {}; failing the sanity rule (not drawn): cpu-list {}, gpu-buffer {}, buffer view {}",
            k[0] + k[1] + k[5],
            k[0],
            k[1],
            k[2] + k[3] + k[4],
            k[2],
            k[3],
            k[4],
            k[5],
            k[6],
            insane[0],
            insane[1],
            insane[2]
        );
        // The albedo slot read at the hook, over the last 60 frames.
        let t = crate::worldtex::take_counts();
        let alpha = c.draws.iter().filter(|d| d.alpha_clip).count();
        xnw_core::info!("world render: frame {frame}: albedo at the hook (60 frames): slot found {}, no albedo variable {}, not resident {}, not a view {}, effect lookups {}; this frame {alpha} alpha-clip draws", t[0], t[1], t[2], t[3], t[4]);
        let decoded: Vec<String> = states.known.iter().take(12).map(|(k, v)| match v {
            Some(s) => format!("{}/{}/{}: depth {}{} f{} bias {} slope {:.2} blend {:?}", k[0], k[1], k[2], u8::from(s.depth_enable), if s.depth_write { "w" } else { "" }, s.depth_func, s.depth_bias, f32::from_bits(s.slope_bias), s.blend),
            None => format!("{}/{}/{}: unread", k[0], k[1], k[2]),
        }).collect();
        xnw_core::info!("world render: frame {frame}: {} render states decoded: {}; stage at TLS+0x950 differs from the pass begin in {} draws", states.known.len(), decoded.join("; "), c.draws.iter().filter(|d| d.stage_tls != 0 && d.stage_tls != d.pass.1).count());
        let p = parts;
        let known: u64 = p[..6].iter().sum();
        let faults = faults_text();
        let ms = |v: u64| v as f64 / 1e6;
        xnw_core::info!(
            "world render: hook parts (frame {frame}, sampled 1 in {PART_SAMPLE}): {}, other {:.2} ms; total {:.2} ms; {faults}",
            (0..6).map(|k| format!("{} {:.2} ms", PART_NAMES[k], ms(p[k]))).collect::<Vec<_>>().join(", "),
            ms(p[6].saturating_sub(known)),
            hook_ms
        );
        let top: Vec<String> = census.iter().take(10).map(|((s, k), n)| format!("{s}/{}/{}/{} {n}", k[0], k[1], k[2])).collect();
        xnw_core::info!("world render: frame {frame}: state census (stage/blend/depth-stencil/raster draws): {}; sun {:?}", top.join(", "), list_sun);
    }
    c.recycle();
}

/// `skinlog`: one line per logged skinned draw: the TLS palette (pointer,
/// count), the skin constants, the palette hash at the hook and in the copy
/// we draw from, the drawable, and the geometry's bone use (vertices, highest
/// weighted index) against curCount: an index at or past curCount reads
/// another draw's bones.
fn skin_log(c: &Capture, skin: &SkinCopy) {
    let bones = BONES.lock().ok();
    for d in c.draws.iter().filter(|d| d.skin_hash != 0) {
        let k = d.skin_cb;
        let ours = skin.at.get(&(k[0], k[2])).and_then(|f| skin.floats.get(12 * *f as usize..12 * (*f + k[2]) as usize)).map(|p| fnv(&p.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()));
        let (verts, max, delta, cloth) = bones.as_ref().and_then(|m| m.as_ref()?.get(&key_of(d)).copied()).unwrap_or((0, None, false, 0));
        let verdict = match max {
            Some(m) if u32::from(m) >= k[2] => "INDEX PAST curCount",
            Some(_) => "in range",
            None => "no bone streams",
        };
        xnw_core::info!(
            "world render: frame {}: skin draw geom {:#x} hash {:#x} drawable {:#x} model {:#x}: TLS palette {:#x} n {} (base {:#x}, as index {}); cb cur {} prev {} n {} prevn {}; palette hash at hook {:#018x}, at our draw {}; geometry {} vertices, max bone {}: {verdict}; {}",
            c.frame,
            d.geom,
            d.ghash,
            d.drawable,
            d.model,
            d.palette.0,
            d.palette.1,
            d.pal_base,
            if d.pal_base != 0 && d.palette.0 >= d.pal_base { format!("{}", (d.palette.0 - d.pal_base) as f64 / 48.0) } else { "-".into() },
            k[0],
            k[1],
            k[2],
            k[3],
            d.skin_hash,
            ours.map_or("- (no copy)".into(), |h| format!("{h:#018x}{}", if h == d.skin_hash { " (same)" } else { " (CHANGED)" })),
            verts,
            max.map_or("-".into(), |m| m.to_string()),
            format!("{}; delta stream 0x44 {}; cloth-index (255) vertices {cloth}; {}", remap_note(d, skin), if delta { "present" } else { "absent" }, bone_map_note(d)),
        );
    }
}




/// The water height map. Map object *(0x147EC8890 + 8)
/// (changes when another map streams in); texels {u64 data, u32 width, u32
/// height} at 0x147ED03D0 [V], read as u16 [I]; AABB {min x, min y, max x,
/// max y} at 0x14A91D73C; world to texel {sx, sy, ox, oy} at 0x14A91D74C:
/// u = x*sx + ox - 0.5 [V]; z = 1600 * texel / 65535 - 800 m [V]; texel 0 =
/// no water [I].
const WATER_MAP: usize = 0x1_47EC_8890;
const WATER_TEXELS: usize = 0x1_47ED_03D0;
const WATER_AABB: usize = 0x1_4A91_D73C;
/// Texels read at most (4096 x 4096).
const WATER_MAX_TEXELS: u64 = 1 << 24;


/// The title's water map now.
fn water_map_read(title: usize) -> Option<WaterMap> {
        let map = q(q(real(title, WATER_MAP))? + 8)?;
        let tx = bytes(real(title, WATER_TEXELS), 16)?;
        let geo = bytes(real(title, WATER_AABB), 32)?;
        let m = WaterMap { map, data: le64(&tx, 0), w: le32(&tx, 8), h: le32(&tx, 12), aabb: f32s::<4>(&geo), s: f32s::<4>(&geo[16..]) };
        let ok = m.map != 0 && m.data != 0 && m.w > 1 && m.h > 1 && u64::from(m.w) * u64::from(m.h) <= WATER_MAX_TEXELS && m.s[..2].iter().all(|v| v.is_finite() && *v != 0.0) && m.s[2..].iter().all(|v| v.is_finite());
        ok.then_some(m)
    }

/// The water map's texels from title memory.
fn water_texels(m: &WaterMap) -> Option<Vec<u16>> {
        let b = bytes(m.data, 2 * m.w as usize * m.h as usize)?;
        Some(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }



/// The heightmap terrain tile array {u32 cap, u32 count, rec*}
/// at 0x148420858, 372-B records rebuilt every frame by 0x143288320 [V].
const TERRAIN_TILES: usize = 0x1_4842_0858;
const TERRAIN_REC: usize = 372;
const TERRAIN_MAX: u32 = 4096;

/// The tile records now: (cap, count, bytes of min(count, TERRAIN_MAX) records).
fn terrain_tiles(title: usize) -> Option<(u32, u32, Vec<u8>)> {
    let h = bytes(real(title, TERRAIN_TILES), 16)?;
    let (cap, count, at) = (le32(&h, 0), le32(&h, 4), le64(&h, 8));
    if count == 0 || at == 0 {
        return Some((cap, count, Vec::new()));
    }
    Some((cap, count, bytes(at, TERRAIN_REC * count.min(TERRAIN_MAX) as usize)?))
}

/// Per tile its known fields (origin +0x30, nBaked
/// +0x168, lod +0x16C, bakedA..D[0] +0x40/+0x80/+0xC0/+0x100) and, for every
/// dword whose meaning is open (+0x00..+0x2C, +0x3C, +0x140..+0x164), its
/// value per tile (tiles.tsv) and over the tiles (fields.tsv: distinct
/// values, min, max, as float, small-index-like): the heightmap SRV index is
/// one of them.
fn terrain_report(recs: &[u8]) -> (String, String) {
    let open: Vec<usize> = (0..0x30).step_by(4).chain([0x3C]).chain((0x140..0x168).step_by(4)).collect();
    let mut tiles = String::from("tile\tox\toy\toz\tn_baked\tlod\tbakedA0\tbakedB0\tbakedC0\tbakedD0");
    for o in &open {
        tiles += &format!("\t+{o:x}");
    }
    tiles.push('\n');
    for (i, r) in recs.chunks_exact(TERRAIN_REC).enumerate() {
        let f = |o: usize| f32::from_bits(le32(r, o));
        tiles += &format!("{i}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", f(0x30), f(0x34), f(0x38), le32(r, 0x168), le64(r, 0x16C), le32(r, 0x40), le32(r, 0x80), le32(r, 0xC0), le32(r, 0x100));
        for o in &open {
            tiles += &format!("\t{:#x}", le32(r, *o));
        }
        tiles.push('\n');
    }
    let mut fields = String::from("offset\tdistinct\tmin\tmax\tfirst\tas_float\tlooks_like\n");
    for o in &open {
        let v: Vec<u32> = recs.chunks_exact(TERRAIN_REC).map(|r| le32(r, *o)).collect();
        let Some(first) = v.first().copied() else { break };
        let mut d = v.clone();
        d.sort_unstable();
        d.dedup();
        let (lo, hi) = (d[0], d[d.len() - 1]);
        let fl = f32::from_bits(first);
        let looks = if hi == 0 {
            "zero"
        } else if d.len() == v.len() && hi < 1 << 20 {
            "per-tile small index (heightmap SRV candidate)"
        } else if hi < 1 << 20 {
            "small index / count"
        } else if fl.is_finite() && fl.abs() > 1e-6 && fl.abs() < 1e7 {
            "float"
        } else {
            "pointer / bits"
        };
        fields += &format!("+{o:#x}\t{}\t{lo:#x}\t{hi:#x}\t{first:#x}\t{fl}\t{looks}\n", d.len());
    }
    (tiles, fields)
}


/// The current water grid (rebuilt when the map's key changes) and its key.
static WATER: Mutex<Option<std::sync::Arc<xnw_world::WaterGrid>>> = Mutex::new(None);

fn water_now(title: usize) -> Option<std::sync::Arc<xnw_world::WaterGrid>> {
    let m = water_map_read(title)?;
    let mut g = WATER.lock().ok()?;
    if g.as_ref().is_none_or(|w| w.id != m.id()) {
        let t0 = Instant::now();
        let texels = water_texels(&m)?;
        let grid = water_grid(&m, &texels);
        xnw_core::info!("world render: water map {:#x}: {}x{} texels, AABB {:?}, transform {:?}: grid {} vertices, {} triangles in {:.1} ms", m.map, m.w, m.h, m.aabb, m.s, grid.vertices.len(), grid.indices.len() / 3, t0.elapsed().as_secs_f64() * 1e3);
        *g = Some(std::sync::Arc::new(grid));
    }
    g.clone()
}


/// The run's census so far: per (effect, program set, stage,
/// albedo outcome) the draws; names resolved once per pointer.
static CENSUS: Mutex<Option<HashMap<(u64, u64, u32, u8), u64>>> = Mutex::new(None);

/// The effect's name (fx+0xD08: a pointer to the string [V]).
fn effect_name(fx: u64) -> String {
    // An fx near u64::MAX once overflowed the add and killed the writer.
    let Some(at) = fx.checked_add(0xD08) else { return "-".into() };
    bytes(at, 8).map(|p| le64(&p, 0)).and_then(|p| bytes(p, 64)).map_or("-".into(), |s| String::from_utf8_lossy(&s.iter().copied().take_while(|b| *b != 0).collect::<Vec<u8>>()).into_owned())
}

/// An effect's name for the light capture's rows: the name
/// pointer at fx+0xD08, else the heavy capture's reading (the effect head's
/// own name, then its template's: instanced effects carry it there).
fn light_effect_name(fx: u64) -> String {
    let n = effect_name(fx);
    if n != "-" && !n.is_empty() {
        return n;
    }
    effect(fx).map(|e| e.name).filter(|n| !n.is_empty()).unwrap_or_else(|| "-".into())
}

/// The program set's programs (+0, +8, +16, +24) as their name hashes
/// (Prog+24): the pixel program is the one at +16 [I].
fn program_hashes(set: u64) -> [u32; 4] {
    let Some(b) = bytes(set, 32) else { return [0; 4] };
    std::array::from_fn(|k| bytes(le64(&b, 8 * k), 0x1C).map_or(0, |p| le32(&p, 0x18)))
}

/// Adds a frame's census and, with `dir`, rewrites census-programs.tsv:
/// one row per distinct (effect name, program hashes, stage, albedo outcome).
fn census_add(frame: &HashMap<(u64, u64, u32, u8), u32>, dir: Option<&std::path::Path>) {
    let Ok(mut g) = CENSUS.lock() else { return };
    let all = g.get_or_insert_with(HashMap::new);
    for (k, n) in frame {
        *all.entry(*k).or_default() += u64::from(*n);
    }
    let Some(dir) = dir else { return };
    let mut names: HashMap<u64, String> = HashMap::new();
    let mut progs: HashMap<u64, [u32; 4]> = HashMap::new();
    let mut rows: HashMap<(String, [u32; 4], u32, u8), u64> = HashMap::new();
    for ((fx, set, stage, why), n) in all.iter() {
        let name = names.entry(*fx).or_insert_with(|| effect_name(*fx)).clone();
        let p = *progs.entry(*set).or_insert_with(|| program_hashes(*set));
        *rows.entry((name, p, *stage, *why)).or_default() += n;
    }
    let mut rows: Vec<_> = rows.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut text = String::from("effect\tprog0\tprog1\tprog2_ps\tprog3\tstage\talbedo\tdraws\n");
    for ((name, p, stage, why), n) in &rows {
        text += &format!("{name}\t{:08x}\t{:08x}\t{:08x}\t{:08x}\t{stage}\t{}\t{n}\n", p[0], p[1], p[2], p[3], crate::worldtex::WHY_NAMES.get(usize::from(*why)).unwrap_or(&"?"));
    }
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join("census-programs.tsv"), text);
    let miss: u64 = rows.iter().filter(|r| r.0.3 == 1).map(|r| r.1).sum();
    xnw_core::info!("world probe: program census: {} distinct (effect, programs, stage, albedo) rows, {} effects, {} program sets; {miss} draws without an albedo variable", rows.len(), names.len(), progs.len());
}

/// Whether `seq=auto:...` dumps this frame: the first frame meeting the
/// condition and the frames after it (consecutive: the consumer sees every
/// frame in order), once per run.
fn auto_dump(cfg: &Config, c: &Capture) -> bool {
    let ped = auto_ped(cfg, c);
    auto_vehicles(cfg, c) || ped
}

/// The vehicle window of `seq=auto` (see `Config::auto_veh`), once per run.
fn auto_vehicles(cfg: &Config, c: &Capture) -> bool {
    let Some((_, draws, frames)) = cfg.auto.filter(|_| cfg.auto_veh > 0) else { return false };
    if VEH_LEFT.load(Ordering::Relaxed) > 0 {
        VEH_LEFT.fetch_sub(1, Ordering::Relaxed);
        return true;
    }
    if VEH_DONE.load(Ordering::Relaxed) || (c.draws.len() as u32) < draws {
        return false;
    }
    let mut names: HashMap<u64, bool> = HashMap::new();
    let veh = c.draws.iter().filter(|d| d.fx != 0 && *names.entry(d.fx).or_insert_with(|| is_vehicle_effect(&effect_name(d.fx)))).count() as u32;
    if veh < cfg.auto_veh {
        return false;
    }
    VEH_DONE.store(true, Ordering::Relaxed);
    VEH_LEFT.store(u64::from(frames - 1), Ordering::Relaxed);
    xnw_core::info!("world probe: auto vehicle window starts at frame {} ({veh} vehicle-effect draws, {} draws), {frames} frames", c.frame, c.draws.len());
    true
}

fn is_vehicle_effect(name: &str) -> bool {
    name.starts_with("standard_vehicle") || name.starts_with("vehicle_")
}

static VEH_LEFT: AtomicU64 = AtomicU64::new(0);
static VEH_DONE: AtomicBool = AtomicBool::new(false);

fn auto_ped(cfg: &Config, c: &Capture) -> bool {
    let (left, done) = (&AUTO_LEFT, &AUTO_DONE);
    let Some((peds, draws, frames)) = cfg.auto else { return false };
    if left.load(Ordering::Relaxed) > 0 {
        left.fetch_sub(1, Ordering::Relaxed);
        return true;
    }
    let ped = c.draws.iter().filter(|d| d.ped_skin.0 != 0).count() as u32;
    if !done.load(Ordering::Relaxed) && ped >= peds && c.draws.len() as u32 >= draws {
        done.store(true, Ordering::Relaxed);
        left.store(u64::from(frames - 1), Ordering::Relaxed);
        xnw_core::info!("world probe: auto capture starts at frame {} ({ped} ped draws, {} draws), {frames} frames", c.frame, c.draws.len());
        // A reference run (frameskip over the loading screens) renders every frame from here.
        xnw_gpu::d3d12x::frameskip_render_all();
        return true;
    }
    false
}

static AUTO_LEFT: AtomicU64 = AtomicU64::new(0);
static AUTO_DONE: AtomicBool = AtomicBool::new(false);

/// Whether a frame's pack data will be written (the pack records no
/// other frame): `at=` frames, `seq=auto` until its window is over, every
/// probed frame of a plain seq capture; nothing once the probe is done.
fn want_pack(frame: u64) -> bool {
    let Some(s) = STATE.get() else { return true };
    if s.done.load(Ordering::Relaxed) {
        return false;
    }
    if !s.cfg.render {
        return s.targets().is_none_or(|t| t.contains(&frame));
    }
    let open = |done: &AtomicBool, left: &AtomicU64| !(done.load(Ordering::Relaxed) && left.load(Ordering::Relaxed) == 0);
    let auto_open = s.cfg.auto.is_some() && (open(&AUTO_DONE, &AUTO_LEFT) || (s.cfg.auto_veh > 0 && open(&VEH_DONE, &VEH_LEFT)));
    auto_open || (s.cfg.raw > 0 && s.cfg.at.contains(&frame))
}

/// A draw's stage: TLS+0x950 at the draw, else the last pass begin.
fn stage_of(d: &Draw) -> u32 {
    if d.stage_tls != 0 { d.stage_tls } else { d.pass.1 }
}

/// Record tables (filled at run time, indexed by the context values):
/// depth-stencil 24 B, raster 16 B, blend 68 B.
const DSS_TABLE: usize = 0x1_475A_657A;
const RASTER_TABLE: usize = 0x1_475A_53DC;
const BLEND_TABLE: usize = 0x1_475A_96E2;

/// The title's state triples of one frame, decoded once each.
#[derive(Default)]
struct StateTables {
    known: HashMap<[u16; 3], Option<xnw_world::RenderState>>,
}

impl StateTables {
    fn get(&mut self, d: &Draw) -> Option<xnw_world::RenderState> {
        let key = [d.state[0], d.state[1], d.state[2]];
        if key.contains(&0xFFFF) {
            return None;
        }
        *self.known.entry(key).or_insert_with(|| {
            let title = STATE.get()?.title;
            let dss = bytes(real(title, DSS_TABLE) + 24 * u64::from(key[1]), 24)?;
            let ras = bytes(real(title, RASTER_TABLE) + 16 * u64::from(key[2]), 16)?;
            let blend = bytes(real(title, BLEND_TABLE) + 68 * u64::from(key[0]), 68)?;
            Some(decode_state(&dss, &ras, &blend))
        })
    }
}

/// Records to our state: DSS {DepthEnable, DepthWrite, Func, ...},
/// raster {Fill, Cull, FrontCCW, i8 DepthBias, f32 clamp +4, f32 slope +8},
/// blend {A2C, Independent, -, -, 8 x {Enable, Src, Dst, Op, SrcA, DstA, OpA, Mask}}.
fn decode_state(dss: &[u8], ras: &[u8], blend: &[u8]) -> xnw_world::RenderState {
    let mut b = [0u8; 8];
    b.copy_from_slice(&blend[4..12]);
    xnw_world::RenderState { depth_enable: dss[0] != 0, depth_write: dss[1] != 0, depth_func: dss[2], depth_bias: i32::from(ras[3] as i8), bias_clamp: le32(ras, 4), slope_bias: le32(ras, 8), blend: b }
}

/// Skinlog (an earlier title's layout): the model's u16 bone-tag map candidate
/// (model+0x28 pointer, +0x30 count), whether the count equals geom+0x2C,
/// and its first 8 entries (compare with the remap's first entries).
fn bone_map_note(d: &Draw) -> String {
    let (Some(m), Some(g)) = (bytes(d.model, 0x38), bytes(d.geom, 0x30)) else { return "model map: unreadable".into() };
    let (ptr, count, gcount) = (le64(&m, 0x28), le32(&m, 0x30), le16(&g, 0x2C));
    let first: Vec<u16> = bytes(ptr, 16).map(|b| b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()).unwrap_or_default();
    format!("model+0x28 map {ptr:#x} count {count} (u16 {}), geom+0x2C {gcount} ({}), first {first:?}", count & 0xFFFF, if count & 0xFFFF == u32::from(gcount) { "equal" } else { "differs" })
}

/// Skinlog: the draw's remap bases, whether we placed it, the
/// first entries under both readings of the base (u32 index into the arena
/// view, or byte offset) and how many map inside curCount; whether the
/// palette looks like dual quaternions (row 0 of unit length).
fn remap_note(d: &Draw, skin: &SkinCopy) -> String {
    let n = d.skin_cb[2];
    let dq = skin.at.get(&(d.skin_cb[0], n)).map_or(0, |f| {
        (0..n as usize).filter(|b| {
            let r = skin.floats.get(*f as usize * 12 + 12 * b..*f as usize * 12 + 12 * b + 4).unwrap_or(&[0.0; 4]);
            ((r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt() - 1.0).abs() < 1e-3
        }).count()
    });
    let mut s = format!("remap base {} prev {}; dq-like row0 in {dq} of {n} bones", d.remap[0] as i32, d.remap[1] as i32);
    if d.remap[0] == u32::MAX {
        return s;
    }
    let (sk, obj, comp, lod) = d.ped_skin;
    s += &format!("; ped skin {sk:#x} table {obj:#x} comp {comp} lod {lod}; placed {}", skin.remap_at.contains_key(&(obj, d.remap[0])));
    if obj != 0
        && let Some((at, len)) = bytes(obj, 0x20).map(|h| grc(&h))
    {
        let v: Vec<u32> = bytes(at + 4 * u64::from(d.remap[0]), 32).map(|b| b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()).unwrap_or_default();
        let inside = v.iter().filter(|x| **x & 0xFFF < n).count();
        s += &format!("; table {at:#x} {len} B, entries at base: {v:?} ({inside}/8 inside curCount)");
        // (an earlier title: the model's tag map, n = u8 model+0x30, local indices
        // 0..n-1): the n table entries from the base: all < curCount and distinct?
        let tags = bytes(d.model, 0x38).map(|m| (le64(&m, 0x28), u32::from(m[0x30]), u32::from(le16(&bytes(d.geom, 0x30).unwrap_or_default(), 0x2C))));
        if let Some((map, nt, ng)) = tags {
            let count = nt.max(ng).min(4096);
            let run: Vec<u32> = bytes(at + 4 * u64::from(d.remap[0]), 4 * count as usize).map(|b| b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) & 0xFFF).collect()).unwrap_or_default();
            let max = run.iter().max().copied();
            let mut sorted = run.clone();
            sorted.sort_unstable();
            sorted.dedup();
            let tag: Vec<u16> = bytes(map, 2 * nt.min(16) as usize).map(|b| b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()).unwrap_or_default();
            s += &format!(
                "; n u8 model+0x30 {nt}, geom+0x2C {ng}; run of {count}: max {} ({}), {} distinct ({}); tags {tag:?} vs run {:?}",
                max.map_or("-".into(), |m| m.to_string()),
                if max.is_some_and(|m| m < n) { "in range" } else { "OUT OF RANGE" },
                sorted.len(),
                if sorted.len() == run.len() { "all distinct" } else { "REPEATS" },
                &run[..run.len().min(16)]
            );
        }
    }
    if let Some(st) = STATE.get()
        && let Some(arena) = q(st.g.arena)
    {
        for (name, at) in [("as index", arena + 4 * u64::from(d.remap[0])), ("as bytes", arena + u64::from(d.remap[0]))] {
            if let Some(b) = bytes(at, 64 * 4) {
                let v: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                let inside = v.iter().filter(|x| **x & 0xFFF < n).count();
                s += &format!("; {name}: {inside}/64 inside curCount, first {:?}", &v[..8]);
            }
        }
    }
    s
}

/// Scene stability, measured on the lists every second.
#[derive(Default)]
struct Coherence {
    /// The last two lists: per draw key (geometry, stage) its count, and per
    /// static key (geometry, rotation) its absolute position.
    prev: Option<(u64, HashMap<(u64, u32), u32>, HashMap<(u64, [i32; 3]), [f32; 3]>)>,
    prev2: Option<HashMap<(u64, u32), u32>>,
    start: Option<Instant>,
    frames: u32,
    gaps: u32,
    residual: Vec<f32>,
    missing_back: u32,
    there_once: u32,
    /// Missing-then-back by (stage, instanced): what the count is made of.
    by_class: HashMap<(u32, bool), u32>,
    /// The previous list's main-view keys hooked but given no row, with
    /// the reason (SKIP_NAMES).
    prev_skip: HashMap<(u64, u32), u8>,
    /// Missing-then-back draws whose key was hooked in the missing frame
    /// and skipped by us, per reason; the rest the title never hooked.
    ours: HashMap<u8, u32>,
    /// Of `ours`, those whose main camera turned less than
    /// [`STILL_TURN_DEG`] across the three lists: a prefilter drop that follows the
    /// camera is a correct cull (the pass draws with the camera it tested); one
    /// under a still camera is not (FLK3 replay: only LOD-pair swaps there).
    ours_still: u32,
    /// Of the title's drops (not hooked), those under a still main camera:
    /// all stages, and stages 65, 70, 71 (depth prepass, G-buffer). FLK4: in 120
    /// still-camera frames (pk7f) the title dropped no stage 65/70 draw, so a
    /// still-camera count there is a drop the camera does not explain.
    title_still: [u32; 2],
    /// The main camera's forward axis in the last two lists (newest first).
    fwd: [Option<[f32; 3]>; 2],
}

/// The largest main-camera turn, in degrees, that counts as a still camera.
const STILL_TURN_DEG: f32 = 0.5;

/// The angle in degrees between two unit axes.
fn turn_deg(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0).acos().to_degrees()
}

static COHERENCE: Mutex<Option<Coherence>> = Mutex::new(None);

/// Per list: frame-id gaps (lists we never made), the static residual (a
/// static draw's absolute position vs its position in the previous list:
/// 0 when camera and transforms pair), draws missing for one list between two
/// that have them, and draws there for one list only; once a second the line
/// `world render: coherence: ...`. `skipped`: this list's main-view keys
/// (geometry, stage) that were hooked but given no row, with the reason: a
/// missing draw is ours when its key is there, the title's when not.
fn coherence(list: &FrameList, skipped: HashMap<(u64, u32), u8>) {
    let Ok(mut g) = COHERENCE.lock() else { return };
    let k = g.get_or_insert_with(Coherence::default);
    let start = *k.start.get_or_insert_with(Instant::now);
    let Some(cam) = list.main_camera() else { return };
    let origin = xnw_world::camera_position(&cam.view);
    // The view's third column: the camera's depth axis in world space.
    let fwd = [cam.view[2], cam.view[6], cam.view[10]];
    let still = match k.fwd {
        [Some(a), Some(b)] => turn_deg(b, a).max(turn_deg(a, fwd)) < STILL_TURN_DEG,
        _ => false,
    };
    let mut keys: HashMap<(u64, u32), u32> = HashMap::new();
    let mut inst_keys: std::collections::HashSet<(u64, u32)> = std::collections::HashSet::new();
    let mut statics: HashMap<(u64, [i32; 3]), [f32; 3]> = HashMap::new();
    let mut seen: HashMap<(u64, [i32; 3]), u32> = HashMap::new();
    for r in list.main_draws() {
        *keys.entry((r.geom.hash, r.stage)).or_default() += 1;
        if r.inst_count > 0 {
            inst_keys.insert((r.geom.hash, r.stage));
        }
        if r.palette == xnw_world::NO_PALETTE && r.inst_count == 0 {
            let rot = [r.world[0], r.world[5], r.world[10]].map(|v| (v * 1000.0) as i32);
            let key = (r.geom.hash, rot);
            *seen.entry(key).or_default() += 1;
            statics.insert(key, [r.world[12] + origin[0], r.world[13] + origin[1], r.world[14] + origin[2]]);
        }
    }
    statics.retain(|key, _| seen.get(key) == Some(&1));
    if let Some((pf, pkeys, pstatic)) = &k.prev {
        k.gaps += (list.frame.saturating_sub(*pf + 1)) as u32;
        for (key, a) in &statics {
            if let Some(b) = pstatic.get(key) {
                k.residual.push(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt());
            }
        }
        if let Some(p2) = &k.prev2 {
            for (key, n) in p2 {
                if keys.get(key) == Some(n) && pkeys.get(key).copied().unwrap_or(0) < *n {
                    let m = n - pkeys.get(key).copied().unwrap_or(0);
                    k.missing_back += m;
                    *k.by_class.entry((key.1, inst_keys.contains(key))).or_default() += m;
                    if let Some(r) = k.prev_skip.get(key) {
                        *k.ours.entry(*r).or_default() += m;
                        k.ours_still += if still { m } else { 0 };
                    } else if still {
                        k.title_still[0] += m;
                        k.title_still[1] += if matches!(key.1, 65 | 70 | 71) { m } else { 0 };
                    }
                }
            }
            for (key, n) in pkeys {
                if p2.get(key).copied().unwrap_or(0) == keys.get(key).copied().unwrap_or(0) && *n > p2.get(key).copied().unwrap_or(0) && p2.contains_key(key) {
                    k.there_once += n - p2[key];
                }
            }
        }
    }
    let old = k.prev.take();
    k.prev2 = old.map(|o| o.1);
    k.prev = Some((list.frame, keys, statics));
    k.prev_skip = skipped;
    k.fwd = [Some(fwd), k.fwd[0]];
    k.frames += 1;
    if start.elapsed().as_secs_f64() >= 1.0 {
        k.residual.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = k.residual.get(k.residual.len() / 2).copied().unwrap_or(0.0);
        let p99 = k.residual.get(k.residual.len() * 99 / 100).copied().unwrap_or(0.0);
        let (published, replaced) = xnw_world::lists().counts();
        let late = STATE.get().map_or([0; 2], |s| [s.late[0].swap(0, Ordering::Relaxed), s.late[1].swap(0, Ordering::Relaxed)]);
        xnw_core::info!(
            "world render: coherence: {} lists (last frame {}), frame-id gaps {}, static residual median {:.4} m p99 {:.3} m over {} draws, missing-then-back {} ({}), there-once {}; lists published {published}, replaced before drawn {replaced}; {}",
            k.frames,
            list.frame,
            k.gaps,
            med,
            p99,
            k.residual.len(),
            k.missing_back,
            {
                let mut v: Vec<_> = k.by_class.iter().collect();
                v.sort_by(|a, b| b.1.cmp(a.1));
                v.iter().take(4).map(|((s, i), n)| format!("stage {s}{} {n}", if *i { " instanced" } else { "" })).collect::<Vec<_>>().join(", ")
            },
            k.there_once,
            ours_line(k.missing_back, &k.ours, k.ours_still, k.title_still, late)
        );
        *k = Coherence { prev: k.prev.take(), prev2: k.prev2.take(), prev_skip: std::mem::take(&mut k.prev_skip), fwd: k.fwd, start: Some(Instant::now()), ..Default::default() };
    }
}

/// The coherence line's tail: of the missing-then-back draws, those
/// hooked that frame but given no row by us (per reason), those the title
/// never hooked; and the draws recorded after their frame's cut (dropped).
/// `still`: of ours, those under a still main camera; `title_still`: of the
/// title's, those under a still main camera (all stages, stages 65/70/71).
fn ours_line(missing: u32, ours: &HashMap<u8, u32>, still: u32, title_still: [u32; 2], late: [u64; 2]) -> String {
    let n: u32 = ours.values().sum();
    let mut v: Vec<_> = ours.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let why = v.iter().take(4).map(|(r, c)| format!("{} {c}", SKIP_NAMES.get(**r as usize).unwrap_or(&"?"))).collect::<Vec<_>>().join(", ");
    format!("missing-then-back hooked but not listed {n} ({why}; camera still {still}), not hooked {} (camera still {}, at stages 65/70/71 {}); late draws {} in {} captures (recorded after their frame's cut)", missing.saturating_sub(n), title_still[0], title_still[1], late[0], late[1])
}

/// Whether this frame's list is dumped.
fn frame_has_dump(dump: Option<&std::path::Path>, _frame: u64) -> bool {
    dump.is_some()
}

const TIMECYCLE: usize = 0x1_4AB4_A400;

/// The live frame list as files, `<dir>\frame-N\`: list.tsv (one row
/// per record: key, world, viewport, stage, palette, instances), camera.txt
/// (all cameras), pal-<i>.bin, instances.bin; and per geometry key in the
/// shared `<dir>\blobs\k-<mesh>-<hash>.{vb,idx}` (raw vertex buffer and
/// expanded indices, read again from title memory, written once).
/// `xnw world-render <dir> --seq` draws these exactly as the live pass did.
fn dump_list(dir: &std::path::Path, list: &FrameList) -> std::io::Result<()> {
    let fdir = dir.join(format!("frame-{}", list.frame));
    let shared = dir.join("blobs");
    std::fs::create_dir_all(&fdir)?;
    std::fs::create_dir_all(&shared)?;
    // albedo_desc (descriptor slot, hex) and alpha_clip (0/1) at the end.
    let mut out = String::from("mesh_at\tmesh_len\tvb_at\tvb_len\tib\thash\tviewport\tstage\tpalette\tinst_first\tinst_count\teffect\tworld\tbone_first\tbone_count\talbedo_desc\talpha_clip\tremap\tped\tlayer0\tlayer1\tlayer2\tlayer3\tstipple\tstate\trstate\n");
    for r in &list.draws {
        let g = r.geom;
        let _ = writeln!(out, "{:#x}\t{}\t{:#x}\t{}\t{}\t{:#x}\t{:#x}\t{}\t{}\t{}\t{}\t{:#x}\t{}\t{}\t{}\t{:#x}\t{}\t{}\t{}\t{:#x}\t{:#x}\t{:#x}\t{:#x}\t{:#x}\t{}/{}/{}\t{}", g.mesh_at, g.mesh_len, g.vb_at, g.vb_len, g.ib, g.hash, r.viewport, r.stage, r.palette, r.inst_first, r.inst_count, r.effect, matrix_text(&r.world), r.bone_first, r.bone_count, r.material.albedo_desc, u8::from(r.material.alpha_clip), r.remap, u8::from(r.ped), r.material.layers[0], r.material.layers[1], r.material.layers[2], r.material.layers[3], r.material.stipple, r.material.state_index[0], r.material.state_index[1], r.material.state_index[2], rstate_text(r.state));
        let stem = shared.join(format!("k-{:x}-{:x}", g.mesh_at, g.hash));
        if !stem.with_extension("vb").exists()
            && let Some(vb) = bytes(g.vb_at, g.vb_len.min(64 << 20) as usize)
            && vb_ready(&vb)
        {
            let idx = if g.mesh_len != 0 {
                bytes(g.mesh_at, g.mesh_len.min(16 << 20) as usize).and_then(|m| expand_meshlets(&m).ok())
            } else {
                let (n, elem) = ((g.ib >> 8).min(3 << 20) as usize, (g.ib & 0xFF) as usize);
                bytes(g.mesh_at, n * elem).map(|ib| match elem {
                    2 => ib.chunks_exact(2).map(|c| u32::from(u16::from_le_bytes([c[0], c[1]]))).collect(),
                    _ => ib.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(),
                })
            };
            if let Some(idx) = idx {
                std::fs::write(stem.with_extension("vb"), &vb)?;
                std::fs::write(stem.with_extension("idx"), idx.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>())?;
            }
        }
    }
    std::fs::write(fdir.join("list.tsv"), out)?;
    let mut cams = String::from("# live frame list cameras (view, projection)\n");
    for c in &list.cameras {
        let _ = writeln!(cams, "viewport {:#x} draws {}\n  view {}\n  proj {}", c.viewport, list.draws.iter().filter(|d| d.viewport == c.viewport).count(), matrix_exact(&c.view), matrix_exact(&c.proj));
    }
    std::fs::write(fdir.join("camera.txt"), cams)?;
    for (i, p) in list.palettes.iter().enumerate() {
        std::fs::write(fdir.join(format!("pal-{i}.bin")), p.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    std::fs::write(fdir.join("instances.bin"), list.instances.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
    // Drawables placed since the last dump (place.tsv; empty without the hook).
    std::fs::write(fdir.join("place.tsv"), crate::worldahead::PLACE_HEADER.to_string() + &crate::worldahead::take_rows().join("\n") + "\n")?;
    // Scene events ticked since the last dump (cine.tsv).
    std::fs::write(fdir.join("cine.tsv"), crate::worldahead::CINE_HEADER.to_string() + &crate::worldahead::take_event_rows().join("\n") + "\n")?;
    // Capture mode: the geometry keys of placed drawables (which later draws they cover).
    // Rigid parts' bones (draw index in list.tsv, 12 floats).
    std::fs::write(fdir.join("rigid.tsv"), list.rigid.iter().map(|(i, b)| format!("{i}\t{}\n", b.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(" "))).collect::<String>())?;
    std::fs::write(fdir.join("attach.tsv"), crate::worldahead::ATTACH_HEADER.to_string() + &crate::worldahead::take_attach_rows().join("\n") + "\n")?;
    std::fs::write(fdir.join("ahead-keys.tsv"), crate::worldahead::KEYS_HEADER.to_string() + &crate::worldahead::take_key_rows().join("\n") + "\n")?;
    // The terrain tile records (terrain.bin, terrain.txt).
    if let Some(t) = &list.terrain {
        crate::worldterrain::dump(&fdir, t)?;
    }
    Ok(())
}





/// The title's TLS index: from the prologue at `entry` + 9, the title's
/// `mov rax, gs:[58h]` (index 0) or the loader's `call thunk` whose
/// `lea rax, [rax + index*8]` names it.
fn tls_index(entry: u64) -> Result<u64, String> {
    let b = arr::<9>(entry + 9).ok_or("prologue unreadable")?;
    if b == [0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0] {
        return Ok(0);
    }
    if b[0] != 0xE8 {
        return Err(format!("prologue +9 is not the TLS load: {b:02x?}"));
    }
    let thunk = (entry + 14).wrapping_add(i32::from_le_bytes(b[1..5].try_into().unwrap()) as i64 as u64);
    let t = arr::<16>(thunk).ok_or("TLS thunk unreadable")?;
    if t[..12] != [0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0, 0x48, 0x8D, 0x80] {
        return Err(format!("TLS thunk not recognised: {t:02x?}"));
    }
    Ok(u64::from(u32::from_le_bytes(t[12..16].try_into().unwrap())) / 8)
}

/// Starts what `XNW_WORLD_PROBE` asks for; nothing when unset.
///
/// # Safety
/// `base` is the mapped title and none of its code runs yet.
pub unsafe fn start(base: usize) {
    // `1`: lists only; `screen[,wire]`: lists drawn and shown by the presenter.
    let (probe, render) = (std::env::var("XNW_WORLD_PROBE").ok(), std::env::var("XNW_WORLD_RENDER").ok());
    let cfg = match config(probe.as_deref(), render.as_deref()) {
        Ok(Some(c)) => c,
        Ok(None) => return,
        Err(e) => {
            xnw_core::warn!("world probe: XNW_WORLD_PROBE={}: {e}; off", probe.unwrap_or_default());
            return;
        }
    };
    crate::worldtex::set_globals(base);
    let (cfg_cut_present, cfg_scan_exit, cfg_hookin) = (cfg.cut_present, cfg.scan_exit, cfg.hookin);
    // The residency and scene-event hooks in every recording mode (capture
    // too: place.tsv / cine.tsv); geometry bytes are offered only to a screen pass.
    let (install, offer) = crate::worldahead::modes(render.as_deref());
    if install {
        crate::worldahead::set_offer(offer);
        // The rows go only into frame dumps; a screen run dumps with `raw=` or `seq=auto` (c.dump).
        crate::worldahead::set_rows(!cfg.render || cfg.raw > 0 || cfg.auto.is_some());
        // SAFETY: forwarded (nothing runs the title yet).
        unsafe { crate::worldahead::install(base) };
    }
    xnw_core::info!("world probe: writer {:?} (render {}, seq {}, raw {}, {} probed frames)", writer(&cfg), cfg.render, cfg.seq, cfg.raw, cfg.at.len());
    let entry = real(base, GEOMETRY_DRAW);
    let index = match tls_index(entry) {
        Ok(i) => i,
        Err(e) => {
            xnw_core::warn!("world probe: {e}; off");
            return;
        }
    };
    let wait_world = std::env::var("XNW_GPU").is_ok_and(|g| g.trim() == "null@world");
    if cfg.pack {
        if cfg.seq {
            pack::set_want(want_pack);
            pack::set_arena(real(base, ARENA_BASE));
            pack::set_imposter_types(real(base, 0x1_4767_65A8));
            pack::set_preskin_lists(real(base, 0x1478A9A50));
            pack::set_lights(real(base, 0x1_487B_AF40), real(base, 0x1_48FD_C4D0), real(base, 0x1_488C_D81C));
            pack::set_hookin(cfg_hookin, real(base, MATRIX_SLOT), real(base, MATRIX_CACHE), real(base, PAL_BASE));
            pack::enable();
        } else {
            xnw_core::warn!("world probe: pack=2 writes seq frames only (add seq=...); pack off");
        }
    }
    install_sinks(cfg.render);
    // The imposter reads need the arena and the type table (the pack sets
    // them too; they are addresses only).
    if imposter::on() {
        pack::set_arena(real(base, ARENA_BASE));
        pack::set_imposter_types(real(base, 0x1_4767_65A8));
        xnw_core::info!("world probe: imposters on: each imposter submit call is read for our card pass");
    }
    let (tx, rx) = mpsc::channel::<Capture>();
    let dir = cfg.dir.clone();
    let state = State { cfg, title: base, tls_index: index, g: Globals::of(base), base: AtomicU64::new(u64::MAX), wait_world, done: AtomicBool::new(false), pending: AtomicU64::new(u64::MAX), slots: Mutex::new(Vec::new()), sent: AtomicU64::new(u64::MAX), late: [AtomicU64::new(0), AtomicU64::new(0)], tx: Mutex::new(tx) };
    if STATE.set(state).is_err() {
        return;
    }
    // SAFETY: nothing runs the title yet; `on_draw` is a plain extern fn.
    match unsafe { crate::watch::hook_entry_with(entry as usize as *mut u8, 0, on_draw as usize) } {
        Ok(_) => xnw_core::info!("world probe: geometry draw hooked at title+{:#x}, TLS index {index}, output {}{}", GEOMETRY_DRAW - IMAGE_BASE, dir.display(), if wait_world { ", base = world start" } else { ", base = first draw" }),
        Err(e) => {
            xnw_core::warn!("world probe: geometry draw not hooked: {e}");
            return;
        }
    }
    // Count the other geometry entries (ids 2.. of the hook stubs).
    for (k, (va, name)) in ENTRIES.iter().enumerate().skip(1) {
        let at = real(base, *va) as usize;
        // SAFETY: nothing runs the title yet.
        if let Err(e) = unsafe { crate::watch::hook_entry_with(at as *mut u8, k, on_entry as usize) } {
            xnw_core::warn!("world probe: {name} not counted: {e}");
        }
    }
    // The CPU-pointer view maker, last writer per (context, slot).
    let at = real(base, VIEW_MAKER) as usize;
    let mut got = [0u8; 14];
    if crate::watch::guarded_read(at, &mut got) && got == VIEW_MAKER_ENTRY {
        // SAFETY: nothing runs the title yet.
        if let Err(e) = unsafe { crate::watch::hook_entry_with(at as *mut u8, 5, on_view as usize) } {
            xnw_core::warn!("world probe: view maker not hooked: {e}");
        }
    }
    // The frame boundary: the ring drain and the flip after it.
    let mut hooked = 0;
    for (va, want, hook, name) in [(DRAIN, &DRAIN_ENTRY, on_drain as usize, "ring drain 0x1443C38A0"), (DRAIN_FLIP, &DRAIN_FLIP_ENTRY, on_drain_flip as usize, "render flip 0x1423CF3E0")] {
        let at = real(base, va) as usize;
        let mut got = [0u8; 14];
        if crate::watch::guarded_read(at, &mut got) && &got == want {
            // SAFETY: nothing runs the title yet.
            match unsafe { crate::watch::hook_entry_with(at as *mut u8, 7, hook) } {
                Ok(_) => {
                    hooked += 1;
                    xnw_core::info!("world probe: {name} hooked at {at:#x}");
                }
                Err(e) => xnw_core::warn!("world probe: {name} not hooked: {e}"),
            }
        } else {
            xnw_core::warn!("world probe: {name} entry {got:02x?} not the expected one; no frame boundary marks");
        }
    }
    if cfg_scan_exit {
        let at = real(base, COMPACT_WRAPPER) as usize;
        let mut got = [0u8; 16];
        if crate::watch::guarded_read(at, &mut got) && got == COMPACT_WRAPPER_BYTES {
            COMPACT_AT.store(real(base, COMPACT) as usize, Ordering::Relaxed);
            // SAFETY: nothing runs the title yet; the hook always returns 1, so the relocated wrapper never runs.
            match unsafe { crate::watch::hook_entry_early(at as *mut u8, 8, on_compaction as usize) } {
                Ok(_) => {
                    COMPACT_HOOKED.store(true, Ordering::Relaxed);
                    xnw_core::info!("world probe: scan=exit: compaction wrapper 0x140B5F960 hooked; scan.txt from the records at the compaction's exit");
                }
                Err(e) => xnw_core::warn!("world probe: scan=exit: compaction wrapper not hooked: {e}; scan.txt at drain entry"),
            }
        } else {
            xnw_core::warn!("world probe: scan=exit: wrapper bytes {got:02x?} not the expected ones; scan.txt at drain entry");
        }
    }
    let drain_cut = hooked == 2 && !cfg_cut_present;
    DRAIN_CUT.store(drain_cut, Ordering::Relaxed);
    xnw_core::info!("world probe: frame cut {}", if drain_cut { "at the title's drain (frame id dword_14ABCE2A0; lists close when the drain returns)" } else { "at our present (old)" });
    // The ped skin bind 0x144E351B0 (which remap table the next ped draws use).
    let at = real(base, PED_SKIN_BIND) as usize;
    let mut got = [0u8; 14];
    if crate::watch::guarded_read(at, &mut got) && got == PED_SKIN_BIND_ENTRY {
        // SAFETY: nothing runs the title yet.
        if let Err(e) = unsafe { crate::watch::hook_entry_with(at as *mut u8, 6, on_ped_skin as usize) } {
            xnw_core::warn!("world probe: ped skin bind not hooked: {e}; no ped remap");
        }
    } else {
        xnw_core::warn!("world probe: ped skin bind 0x144E351B0 entry {got:02x?} not the expected one; no ped remap");
    }
    // Each draw's entity (the draw-list walk's entry).
    let walk = real(base, LIST_WALK) as usize;
    let call = real(base, LIST_CALL) as usize;
    let (mut a, mut b) = ([0u8; 14], [0u8; 15]);
    if crate::watch::guarded_read(walk, &mut a) && a == LIST_WALK_ENTRY && crate::watch::guarded_read(call, &mut b) && b == LIST_CALL_BYTES {
        // SAFETY: nothing runs the title yet; LIST_CALL's 15 bytes hold no branch target and r11 is dead there.
        let r = unsafe { crate::watch::hook_entry_with(walk as *mut u8, 9, on_list_walk as usize).and_then(|_| crate::watch::hook_site_with(call as *mut u8, 10, on_list_call as usize, &LIST_CALL_PRELUDE)) };
        match r {
            Ok(_) => xnw_core::info!("world probe: draw-list walk 0x140139BB0 hooked (entry and handler call): draws carry their entity"),
            Err(e) => xnw_core::warn!("world probe: draw-list walk not hooked: {e}; no entity per draw"),
        }
    } else {
        xnw_core::warn!("world probe: draw-list walk 0x140139BB0 bytes {a:02x?} / {b:02x?} not the expected ones; no entity per draw");
    }
    // Tag draws with their pass (phase, stage).
    let pass = real(base, PASS_BEGIN) as usize;
    let mut got = [0u8; 15];
    if crate::watch::guarded_read(pass, &mut got) && got == PASS_BEGIN_ENTRY {
        // SAFETY: nothing runs the title yet.
        if let Err(e) = unsafe { crate::watch::hook_entry_with(pass as *mut u8, 1, on_pass as usize) } {
            xnw_core::warn!("world probe: pass begin not hooked: {e}");
        }
    } else {
        xnw_core::warn!("world probe: pass begin 0x140369A80 entry {got:02x?} not the expected one; draws carry no pass");
    }
    // The imposter submit, per call, into the frame pack.
    if pack::on() || imposter::on() {
        let at = real(base, IMPOSTER_SUBMIT) as usize;
        let mut got = [0u8; 14];
        // An `XNW_CALLS` hook already on this entry (`jmp [rip+0]`) stays: ours runs first.
        let read = crate::watch::guarded_read(at, &mut got);
        let chained = got[..6] == [0xFF, 0x25, 0, 0, 0, 0];
        if read && (got == IMPOSTER_SUBMIT_ENTRY || chained) {
            // SAFETY: nothing runs the title yet.
            match unsafe { crate::watch::hook_entry_with(at as *mut u8, 11, on_imposter as usize) } {
                Ok(_) => xnw_core::info!("world probe: imposter submit 0x1451D0AC0 hooked{} ({})", if chained { " in front of an earlier detour" } else { "" }, if pack::on() { "pack: one imposter record and its instances per call" } else { "live: batches for our card pass" }),
                Err(e) => xnw_core::warn!("world probe: imposter submit not hooked: {e}; no imposter records"),
            }
        } else {
            xnw_core::warn!("world probe: imposter submit 0x1451D0AC0 entry {got:02x?} not the expected one; no imposter records");
        }
    }
    std::thread::spawn(move || {
        let s = STATE.get().unwrap();
        let mut cache = xnw_world::GeometryCache::<()>::default();
        let mut panics = 0u32;
        for c in rx {
            let frame = c.frame;
            // A panic in one frame's writing once cost every later frame.
            // Each frame is now written under catch_unwind: a failing frame is
            // logged and skipped, the writer goes on.
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if s.cfg.render {
                    let dump = c.dump.then(|| s.cfg.dir.clone());
                    render_seq_frame(s, c, &mut cache, dump.as_deref());
                } else if let Err(e) = write_frame(s, c) {
                    xnw_core::warn!("world probe: writing failed: {e}");
                }
            }));
            if let Err(e) = r {
                panics += 1;
                let what = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                xnw_core::warn!("world probe: writer panic in frame {frame} ({panics} so far): {what}; frame skipped, writing goes on");
            }
        }
    });
    std::thread::spawn(|| {
        let s = STATE.get().unwrap();
        loop {
            std::thread::sleep(Duration::from_millis(200));
            // Under the drain cut the drain closes frames; this keeps the old cut only.
            if !DRAIN_CUT.load(Ordering::Relaxed) {
                s.flush(xnw_gpu::d3d12x::presented());
            }
            if s.done.load(Ordering::Relaxed) && s.idle() {
                break;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_frame::*;

    /// A stream word for (stride, offset, format).
    fn stream(stride: u32, offset: u32, format: u32) -> u32 {
        (stride / 4) << 5 | (offset / 4) << 9 | format
    }

    /// A vertex buffer: header, then per vertex 8 bytes position, 4 normal, 4 uv.
    fn vertex_buffer(raw: &[[u32; 3]], mode: u32) -> Vec<u8> {
        let mut vb = vec![0u8; 0x80];
        let n = raw.len() as u32;
        let (pos, nrm, uv) = (0x80, 0x80 + 8 * n, 0x80 + 12 * n);
        vb[0..4].copy_from_slice(&stream(8, pos, 0).to_le_bytes());
        vb[8..12].copy_from_slice(&stream(4, nrm, 0).to_le_bytes());
        vb[12..16].copy_from_slice(&stream(4, uv, 1).to_le_bytes());
        for (i, f) in [-100.0f32, 50.0, 2.0, mode as f32, 0.5, 0.25, 0.125, 2.0].iter().enumerate() {
            let b = if i == 3 { mode.to_le_bytes() } else { f.to_bits().to_le_bytes() };
            vb[0x60 + 4 * i..0x64 + 4 * i].copy_from_slice(&b);
        }
        for c in raw {
            let a = c[0] << 8 | (c[1] & 0xFF);
            let b = c[1] >> 8 | c[2] << 16;
            vb.extend(a.to_le_bytes());
            vb.extend(b.to_le_bytes());
        }
        for _ in raw {
            // +z: octahedral (0, 0) in bytes: 127.5 -> 128 ~ 0.0039
            vb.extend((0x8080u32).to_le_bytes());
        }
        for i in 0..n {
            let u = (i * 1000) as i16 as u16 as u32;
            vb.extend((u | 0x4000 << 16).to_le_bytes());
        }
        vb
    }

    fn expect(c: [u32; 3], mode: u32) -> [f32; 3] {
        let v = [c[0] as f32 * 0.5 - 100.0, c[1] as f32 * 0.25 + 50.0, c[2] as f32 * 0.125 + 2.0];
        match mode {
            0 => [v[2], v[1], v[0]],
            1 => [v[1], v[2], v[0]],
            _ => v,
        }
    }

    /// Meshlets: (vertex-buffer indices, local triangles) each.
    fn meshlet_blob(meshlets: &[(Vec<u16>, Vec<[u32; 3]>)]) -> Vec<u8> {
        let count = meshlets.len() as u32;
        let prims: u32 = meshlets.iter().map(|m| m.1.len() as u32).sum();
        let mut b = Vec::new();
        b.extend(count.to_le_bytes());
        b.extend(prims.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        let (mut fv, mut fp) = (0u32, 0u32);
        for (v, p) in meshlets {
            let h0 = (v.len() as u32 - 1) | (p.len() as u32 - 1) << 7 | (fv >> 16 & 1) << 14 | (fp >> 16 & 1) << 15;
            let h1 = fv & 0xFFFF | (fp & 0xFFFF) << 16;
            b.extend(h0.to_le_bytes());
            b.extend(h1.to_le_bytes());
            b.extend([0u8; 8]);
            fv += v.len() as u32;
            fp += p.len() as u32;
        }
        for (_, p) in meshlets {
            for t in p {
                b.extend((t[0] | t[1] << 7 | t[2] << 14).to_le_bytes());
            }
        }
        for (v, _) in meshlets {
            for i in v {
                b.extend(i.to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn stream_words() {
        let s = Stream::parse(stream(12, 0x1234 * 4, 2));
        assert_eq!((s.stride, s.offset, s.format), (12, 0x1234 * 4, 2));
    }

    #[test]
    fn meshlets_decode_to_shared_vertices() {
        let raw: Vec<[u32; 3]> = (0..6).map(|i| [i * 1000 + 7, 0xABCDEF - i * 3, 0xFFFF - i]).collect();
        for mode in [0, 1, 2] {
            let vb = vertex_buffer(&raw, mode);
            // Meshlet 0: vertices 0,1,2,3 (two triangles, one culled: index 4 >= 4).
            // Meshlet 1: vertices 2,3,4,5 (shares 2 and 3).
            let blob = meshlet_blob(&[(vec![0, 1, 2, 3], vec![[0, 1, 2], [2, 1, 3], [0, 4, 1]]), (vec![2, 3, 4, 5], vec![[0, 1, 2], [2, 3, 1]])]);
            let m = decode_meshlets(&blob, &vb).unwrap();
            assert_eq!((m.meshlets, m.culled, m.pos.len(), m.tris.len()), (2, 1, 6, 4));
            assert_eq!(m.tris, vec![[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 5, 3]]);
            for (i, r) in raw.iter().enumerate() {
                assert_eq!(m.pos[i], expect(*r, mode), "mode {mode} vertex {i}");
            }
            assert_eq!(m.normal.len(), 6);
            assert!(m.normal.iter().all(|n| n[2] > 0.99), "{:?}", m.normal[0]);
            assert_eq!(m.uv.len(), 6);
            assert!((m.uv[1][0] - 1000.0 / 32767.0 * 2.0).abs() < 1e-6 && (m.uv[1][1] - 0x4000 as f32 / 32767.0 * 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn meshlet_high_bits_and_errors() {
        // First primitive / vertex past 0xFFFF use the header's bits 15 / 14.
        let raw: Vec<[u32; 3]> = (0..3).map(|i| [i, i, i]).collect();
        let vb = vertex_buffer(&raw, 2);
        let mut big = vec![(vec![0u16, 1, 2], vec![[0, 1, 2]])];
        let mut blob = meshlet_blob(&big);
        let h0 = u32::from_le_bytes(blob[12..16].try_into().unwrap()) | 1 << 15;
        blob[12..16].copy_from_slice(&h0.to_le_bytes());
        let e = decode_meshlets(&blob, &vb).unwrap_err();
        assert!(e.contains("past"), "{e}");
        big[0].0[2] = 50; // vertex past the buffer
        let e = decode_meshlets(&meshlet_blob(&big), &vb).unwrap_err();
        assert!(e.contains("vertex 50"), "{e}");
        assert!(decode_meshlets(&[1, 0, 0, 0], &vb).is_err());
        assert!(decode_meshlets(&meshlet_blob(&big), &vb[..0x40]).is_err());
    }

    #[test]
    fn octahedral_ten_bit_normals() {
        let mut vb = vertex_buffer(&[[0, 0, 0]], 2);
        // 10:10, sign bit set: (511.5, 511.5) ~ (0, 0) -> +z; (0, 511.5) -> -x side.
        let at = 0x80 + 8;
        vb[at..at + 4].copy_from_slice(&(1u32 << 31 | 512 << 10 | 512).to_le_bytes());
        let s = Streams::parse(&vb).unwrap();
        let n = s.normal(&vb, 0).unwrap();
        assert!(n[2] > 0.99, "{n:?}");
        vb[at..at + 4].copy_from_slice(&(1u32 << 31 | 512 << 10).to_le_bytes());
        let n = s.normal(&vb, 0).unwrap();
        assert!(n[0] < -0.99, "{n:?}");
    }

    #[test]
    fn indexed_geometry() {
        let raw: Vec<[u32; 3]> = (0..4).map(|i| [i * 10, i, i]).collect();
        let vb = vertex_buffer(&raw, 2);
        let ib: Vec<u8> = [0u16, 1, 2, 2, 1, 3].iter().flat_map(|i| i.to_le_bytes()).collect();
        let m = decode_indexed(&ib, 2, &vb).unwrap();
        assert_eq!((m.pos.len(), m.tris.len()), (4, 2));
        assert_eq!(m.pos[3], expect(raw[3], 2));
        assert!(decode_indexed(&ib, 3, &vb).is_err());
    }

    #[test]
    fn transform_uses_rows() {
        let mut m = [0f32; 16];
        m[0] = 2.0;
        m[5] = 3.0;
        m[10] = 4.0;
        m[15] = 1.0;
        m[12..15].copy_from_slice(&[10.0, 20.0, 30.0]);
        assert_eq!(transform(&m, [1.0, 1.0, 1.0]), [12.0, 23.0, 34.0]);
    }

    #[test]
    fn obj_objects_number_globally() {
        let mut out = Vec::new();
        let mut base = 0;
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        write_obj_object(&mut out, "a", &pos, &[], &[], &[[0, 1, 2]], &mut base).unwrap();
        write_obj_object(&mut out, "b", &pos, &[[0.0, 0.0]; 3], &[[0.0, 0.0, 1.0]; 3], &[[2, 1, 0]], &mut base).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(base, 6);
        assert!(text.starts_with("o a\nv 0.0000 0.0000 0.0000\n"));
        assert!(text.contains("\nf 1 2 3\no b\n"));
        assert!(text.contains("\nvt 0.00000 0.00000\n"));
        assert!(text.ends_with("f 6/6/6 5/5/5 4/4/4\n"), "{text}");
        assert_eq!(text.lines().filter(|l| l.starts_with("v ")).count(), 6);
    }

    #[test]
    fn float_positions_format_3() {
        // Stream word: stride 12, offset 0x60, format 3: no scale/bias block.
        let mut vb = vec![0u8; 0x60];
        vb[0..4].copy_from_slice(&stream(12, 0x60, 3).to_le_bytes());
        for v in [[1.5f32, -2.0, 3.25], [4.0, 5.0, 6.0]] {
            for f in v {
                vb.extend(f.to_le_bytes());
            }
        }
        let s = Streams::parse(&[vb.clone(), vec![0; 0x40]].concat()).unwrap();
        assert_eq!(s.position(&vb, 1), Some([4.0, 5.0, 6.0]));
        assert_eq!(s.position(&vb, 0), Some([1.5, -2.0, 3.25]));
        assert_eq!(s.position(&vb, 2), None);
    }

    #[test]
    fn world_space_and_borrowing() {
        // Frame 1700's main view; a draw 2.85, 2.28, -1.02 from the camera.
        let view: [f32; 16] = [0.75284, -0.01068, -0.65811, 0.0, -0.65820, -0.01222, -0.75274, 0.0, 0.0, 0.99987, -0.01623, 0.0, 54.06376, -255.99721, -304.74072, 1.0];
        let c = camera_position(&view);
        assert!((c[0] + 243.99).abs() < 0.05 && (c[1] + 196.93).abs() < 0.05 && (c[2] - 251.02).abs() < 0.05, "{c:?}");
        let mut rel = [0f32; 32];
        for i in [0, 5, 10, 15] {
            rel[i] = 1.0;
        }
        rel[12..15].copy_from_slice(&[2.85, 2.28, -1.02]);
        let a = absolute(&rel, &view);
        assert!((a[12] + 241.14).abs() < 0.05 && (a[13] + 194.65).abs() < 0.05 && (a[14] - 250.0).abs() < 0.05, "{a:?}");
        assert!(plausible(&a));
        let mut bad = a;
        bad[12] = 6e8;
        assert!(!plausible(&bad));
        bad = a;
        bad[0] = 1e5;
        assert!(!plausible(&bad));
        // Two draws of one geometry, the second without a transform: it borrows.
        let raw: Vec<[u32; 3]> = (0..3).map(|i| [i, i, i]).collect();
        let mesh = decode_meshlets(&meshlet_blob(&[(vec![0, 1, 2], vec![[0, 1, 2]])]), &vertex_buffer(&raw, 2)).unwrap();
        let meshes: HashMap<u64, &Mesh> = [(7u64, &mesh)].into_iter().collect();
        let views: HashMap<u64, [f32; 16]> = [(1u64, view)].into_iter().collect();
        let mut draws = vec![
            WorldDraw { index: 0, geom: 7, effect: 1, viewport: 1, world: Some(rel), src: WorldSrc::Cache },
            WorldDraw { index: 1, geom: 7, effect: 1, viewport: 1, world: None, src: WorldSrc::None },
            WorldDraw { index: 2, geom: 8, effect: 1, viewport: 1, world: None, src: WorldSrc::None },
        ];
        let dir = std::env::temp_dir().join("xnw_w3_world");
        std::fs::create_dir_all(&dir).unwrap();
        let st = write_world(&dir, 1, &mut draws, &meshes, &views).unwrap();
        assert_eq!((st.borrowed, st.missing, st.vertices), (1, 0, 6));
        assert_eq!(draws[1].src, WorldSrc::Borrowed);
        let obj = std::fs::read_to_string(dir.join("scene.obj")).unwrap();
        assert!(obj.contains("o d0_v1_g7_e1_scache\n") && obj.contains("o d1_v1_g7_e1_sborrowed\n"), "{obj}");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[test]
    fn skin_decode_and_blend() {
        // Header: position packed (mode 2, as vertex_buffer), weights slot 6
        // at 0x18, indices slot 8 at 0x20, both 4-byte streams after the rest.
        let raw: Vec<[u32; 3]> = vec![[200, 0, 0], [0, 0, 0]];
        let mut vb = vertex_buffer(&raw, 2);
        let wat = vb.len() as u32;
        vb.extend([128u8, 127, 0, 0, 255, 0, 0, 0]); // vertex 0: half/half; vertex 1: bone 0 only
        let iat = vb.len() as u32;
        vb.extend([0u8, 1, 0, 0, 0, 0, 0, 0]);
        vb[0x18..0x1C].copy_from_slice(&stream(4, wat, 6).to_le_bytes());
        vb[0x20..0x24].copy_from_slice(&stream(4, iat, 5).to_le_bytes());
        let s = Streams::parse(&vb).unwrap();
        assert_eq!(s.skin(&vb, 0), Some(([0, 1, 0, 0, 0, 0, 0, 0], [128, 127, 0, 0, 0, 0, 0, 0])));
        let m = decode_meshlets(&meshlet_blob(&[(vec![0, 1, 1], vec![[0, 1, 2]])]), &vb).unwrap();
        assert_eq!(m.skin.len(), m.pos.len());
        // Bone 0: +10 in x; bone 1: +20 in y (3x4 rows: identity with translation in column 3).
        let bone = |t: [f32; 3]| [1.0, 0.0, 0.0, t[0], 0.0, 1.0, 0.0, t[1], 0.0, 0.0, 1.0, t[2]];
        let palette: Vec<f32> = [bone([10.0, 0.0, 0.0]), bone([0.0, 20.0, 0.0])].concat();
        let k = m.skinned(&palette).unwrap();
        let (p0, p1) = (m.pos[0], m.pos[1]);
        let (w0, w1) = (128.0 / 255.0, 127.0 / 255.0);
        let want0 = [p0[0] + 10.0 * w0, p0[1] + 20.0 * w1, p0[2]];
        assert!((0..3).all(|j| (k.pos[0][j] - want0[j]).abs() < 1e-3), "{:?} vs {want0:?}", k.pos[0]);
        assert!((k.pos[1][0] - (p1[0] + 10.0)).abs() < 1e-4 && (k.pos[1][1] - p1[1]).abs() < 1e-4);
        assert!(Mesh::default().skinned(&palette).is_some(), "no vertices: nothing to skin, still a mesh");
        assert!(m.skinned(&[]).is_none());
        // skinlog: vertex 0 weights bones 0 and 1, vertex 1 bone 0 only.
        assert_eq!(bones_used(&vb, &[0, 1, 1]), (2, Some(1), false, 0));
        assert_eq!(bones_used(&vb, &[1]), (1, Some(0), false, 0));
        assert_ne!(fnv(&[1, 2]), fnv(&[2, 1]));
    }

    #[test]
    fn name_key_walks_the_attachment_list() {
        // head -> node A (own type 2) -> node B (own type 5, the key).
        let b: Box<[u64; 2]> = Box::new([5, 0xDEAD_BEEF_1234_5678]);
        let a: Box<[u64; 2]> = Box::new([(&*b as *const _ as u64) | 2, 7]);
        let head = &*a as *const _ as u64;
        assert_eq!(name_key(head), (0xDEAD_BEEF_1234_5678, 1));
        assert_eq!(name_key(0), (0, 0));
    }

    #[test]
    fn png_writer() {
        let p = std::env::temp_dir().join("xnw_w4.png");
        write_png(&p, 3, 2, &[255u8; 3 * 2 * 4]).unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&b[12..16], b"IHDR");
        assert_eq!(&b[b.len() - 8..b.len() - 4], b"IEND");
        let _ = std::fs::remove_file(p);
    }
    /// A rigid part's world = its bone (object space) then the object's
    /// world: a part bone that turns +90 deg about z and moves (1, 2, 3).
    #[test]
    fn rigid_part_world() {
        // rows r0..r2 (x', y', z' as dot(r, (p, 1))): x' = -y + 1, y' = x + 2, z' = z + 3
        let b = [0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0];
        let mut w = [0f32; 16];
        (w[0], w[5], w[10], w[15], w[12]) = (1.0, 1.0, 1.0, 1.0, 10.0);
        let r = rigid_world(&b, &w);
        let p = |v: [f32; 3]| -> [f32; 3] { std::array::from_fn(|j| v[0] * r[j] + v[1] * r[4 + j] + v[2] * r[8 + j] + r[12 + j]) };
        assert_eq!(p([1.0, 0.0, 0.0]), [10.0 + 1.0, 3.0, 3.0], "(1,0,0) -> bone (1, 3, 3) -> world +10 x");
        assert_eq!(p([0.0, 1.0, 0.0]), [10.0 + 0.0, 2.0, 3.0]);
    }

    /// A live list dumped and read back is the same list (geometry
    /// bytes come from title memory and are absent here).
    #[test]
    fn live_list_round_trip() {
        use xnw_world::{Camera as C, DrawRecord, FrameList, GeomKey};
        let dir = std::env::temp_dir().join("xnw_w7_list");
        let _ = std::fs::remove_dir_all(&dir);
        let world: [f32; 16] = std::array::from_fn(|i| i as f32 * 0.5);
        let geom = GeomKey { mesh_at: 0x1234, mesh_len: 64, vb_at: 0x5678, vb_len: 128, ib: 0, hash: 0xABC };
        let list = FrameList {
            frame: 7,
            draws: vec![DrawRecord { geom, world, viewport: 0x99, effect: 0x42, albedo: Albedo::Flat([1.0; 3]), palette: 0, stage: 65, inst_first: 2, inst_count: 3, bone_first: 7, bone_count: 9, material: xnw_world::Material { albedo_desc: 0x2_4000_0040, alpha_clip: true, ..Default::default() }, state: Some(xnw_world::RenderState { depth_enable: true, depth_write: false, depth_func: 7, depth_bias: -3, bias_clamp: 0x3f80_0000, slope_bias: 0x4000_0000, blend: [1, 5, 6, 1, 2, 1, 1, 15] }), remap: xnw_world::NO_REMAP, ped: false }],
            cameras: vec![C { viewport: 0x99, view: world, proj: world }],
            palettes: vec![vec![1.0; 24]],
            instances: vec![0.25; 60],
            ..Default::default()
        };
        dump_list(&dir, &list).unwrap();
        let (back, geoms) = load_list(&dir.join("frame-7")).unwrap();
        assert_eq!(back.draws.len(), 1);
        let r = &back.draws[0];
        assert_eq!((r.geom, r.viewport, r.stage, r.palette, r.inst_first, r.inst_count, r.effect), (geom, 0x99, 65, 0, 2, 3, 0x42));
        assert!(r.world.iter().zip(world).all(|(a, b)| (a - b).abs() < 1e-4));
        assert_eq!((r.material.albedo_desc, r.material.alpha_clip), (0x2_4000_0040, true), "material columns");
        assert_eq!(r.state, list.draws[0].state, "the decoded render state replays");
        assert_eq!((back.cameras.len(), back.palettes, back.instances.len()), (1, vec![vec![1.0; 24]], 60));
        assert!(geoms.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }
    /// The timed and the heavy tests run one at a time (timings measured
    /// while another test allocates or writes failed on the remote box).
    fn heavy() -> std::sync::MutexGuard<'static, ()> {
        static HEAVY: Mutex<()> = Mutex::new(());
        HEAVY.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The sphere test: an instance 10 m in front passes, 10 m
    /// behind, beside or with a scale of 0 does not.
    #[test]
    fn instance_frustum_sphere() {
        let view: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let proj: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.1, 0.0];
        let planes = frustum_planes(&xnw_world::Camera { viewport: 0, view, proj }.view_proj_relative());
        let mut t = [0f32; 35];
        t[5] = 1.0;
        let at = |z: f32, s: f32| [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, z, s, 0.0, s, 0.0];
        assert!(in_frustum(&t, &at(-10.0, 1.0), [0.0; 3], &planes), "in front");
        assert!(!in_frustum(&t, &at(10.0, 1.0), [0.0; 3], &planes), "behind");
        assert!(!in_frustum(&t, &at(-10.0, 0.0), [0.0; 3], &planes), "scale 0");
        let mut tr = at(0.0, 1.0);
        tr[3] = 100.0;
        assert!(!in_frustum(&t, &tr, [0.0; 3], &planes), "beside");
    }

    /// camera.txt's matrices read back bit for bit (the replay's frustum is the live one).
    #[test]
    fn camera_text_reads_back_exactly() {
        let m: [f32; 16] = [-0.645_281_3, 0.220_694_8, -0.731_372_1, 0.0, 1e-7, 3.333_333_3, -0.001_471_2, 1.0, 1291.436_2, -380.879_18, 1251.059_6, 1.0, 0.15, -1.0, 2.5e-9, 0.0];
        let back: Vec<f32> = matrix_exact(&m).replace('|', " ").split_whitespace().map(|x| x.parse().unwrap()).collect();
        assert_eq!(back.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), m.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    /// A header without a stream table (word 0 = 0, or a zero
    /// scale) is not ready; Q24 and float3 headers are.
    /// An outcome row is WHY_ROW bytes with its fields where
    /// tools/why_tsv.py reads them.
    #[test]
    fn why_rows_are_fixed_size() {
        let mut b = Vec::new();
        why_row(&mut b, 0x1122_3344_5566_7788, 3, 1, 42, [1.0, 2.0, 3.0], 9.5, 5, 2);
        assert_eq!(b.len(), WHY_ROW);
        assert_eq!(u64::from_le_bytes(b[0..8].try_into().unwrap()), 0x1122_3344_5566_7788);
        assert_eq!((le32(&b, 8), b[12], b[13], b[14], le32(&b, 16)), (3, 1, 5, 2, 42));
        assert_eq!(f32::from_le_bytes(b[28..32].try_into().unwrap()), 3.0);
        assert_eq!(f32::from_le_bytes(b[32..36].try_into().unwrap()), 9.5);
    }

    /// A second submission of the same geometry, placement,
    /// stage, state and sample mask is a repeat; another state or stage is not.
    #[test]
    fn repeats_need_equal_state() {
        let d = Draw { ghash: 7, viewport: 1, world: Some([1.0; 32]), state: [287, 20, 0, 0], ..Default::default() };
        let k = key_of(&d);
        let mut other = d.clone();
        assert_eq!(repeat_key(&d, k), repeat_key(&other, k));
        other.state[3] = 1;
        assert_ne!(repeat_key(&d, k), repeat_key(&other, k));
        let mut moved = d.clone();
        moved.world.as_mut().unwrap()[3] = 2.0;
        assert_ne!(repeat_key(&d, k), repeat_key(&moved, k));
    }

    /// Every viewport with the main camera's matrices counts as the
    /// main view; another camera does not.
    #[test]
    fn main_view_spans_viewports() {
        let mut a = [0f32; 16];
        a[0] = 1.0;
        let mut b = a;
        b[12] = 50.0;
        let cams = [RCamera { viewport: 1, view: a, proj: a }, RCamera { viewport: 2, view: a, proj: a }, RCamera { viewport: 3, view: b, proj: a }];
        assert_eq!(main_viewports(&cams, Some(2)), vec![1, 2]);
        assert!(main_viewports(&cams, None).is_empty());
    }

    /// An instance's screen cover from its sphere.
    #[test]
    fn instance_cover_from_sphere() {
        let mut t = [0f32; 35];
        t[5] = 1.0;
        let mut w = [0f32; 12];
        (w[0], w[4], w[8], w[11]) = (1.0, 1.0, 1.0, 100.0);
        assert!(instance_cover(&w, &t, 1.0) < 0.001, "1 m sphere at 100 m");
        w[11] = 0.5;
        assert_eq!(instance_cover(&w, &t, 1.0), 1.0, "camera inside");
        (w[0], w[11]) = (1e30, 100.0);
        assert_eq!(instance_cover(&w, &t, 1.0), 1.0, "a garbage scale covers the screen");
    }

    /// The packed grass decode (shader 1c621f63: stride 16, u16 x/y/z
    /// over the batch box, scale byte 11) and the records it makes.
    #[test]
    fn grass_instances_decode() {
        let mut b = [0u8; 32];
        b[0..2].copy_from_slice(&0u16.to_le_bytes());
        b[2..4].copy_from_slice(&65535u16.to_le_bytes());
        b[4..6].copy_from_slice(&32768u16.to_le_bytes());
        b[11] = 255;
        b[16..18].copy_from_slice(&65535u16.to_le_bytes());
        let (min, max) = ([100.0, 200.0, 10.0], [140.0, 260.0, 20.0]);
        let (p, f) = grass_instance(&b, min, max);
        assert_eq!((p[0], p[1], f), (100.0, 260.0, 1.0));
        assert!((p[2] - 15.0).abs() < 1e-3);
        let block = grass_block(&b, 2, min, max, [0.5, 1.5]);
        assert_eq!(block.len(), 128);
        let r0 = f32s::<16>(&block);
        assert_eq!((r0[0], r0[3], r0[7], r0[12]), (1.5, 100.0, 260.0, 1.0 / 1.5));
        assert!(plausible_record(&r0));
        let r1 = f32s::<16>(&block[64..]);
        assert_eq!((r1[0], r1[3]), (0.5, 140.0));
        // A bad scale range keeps unit scale.
        assert_eq!(f32s::<16>(&grass_block(&b, 1, min, max, [f32::NAN, 2.0]))[0], 1.0);
    }

    /// Captured grass bytes (a capture's refusals.txt, L 0x12d0bd4c0,
    /// batch 36, its +0x178 array = the buffer view, 16 B each): the decode
    /// lands at street level inside the list's box; normal bytes 0x80 0x80
    /// (up), colour 0x44 0x5b 0x16 (green).
    #[test]
    fn grass_captured_bytes() {
        let b: Vec<u8> = (0..32).map(|i| u8::from_str_radix(&"9f0b3af44f098080445b164646284601062745dbeb098377445b160000324601"[2 * i..2 * i + 2], 16).unwrap()).collect();
        let (lo, hi) = ([1239.1362, -607.29565, 0.718994], [2109.5261, 348.69675, 30.705853]);
        let (p0, f0) = grass_instance(&b, lo, hi);
        let (p1, f1) = grass_instance(&b[16..], lo, hi);
        let near = |a: [f32; 3], e: [f32; 3]| (0..3).all(|k| (a[k] - e[k]).abs() < 0.01);
        assert!(near(p0, [1278.648, 304.745, 1.809]) && near(p1, [1371.816, 211.545, 1.881]), "{p0:?} {p1:?}");
        assert_eq!((f0, f1), (70.0 / 255.0, 0.0));
        assert_eq!((&b[6..8], &b[8..11]), (&[0x80, 0x80][..], &[0x44, 0x5b, 0x16][..]));
        let block = grass_block(&b, 2, lo, hi, [0.5, 1.5]);
        assert!((0..2).all(|i| plausible_record(&f32s::<16>(&block[64 * i..]))));
    }

    /// Switches and the record sanity gate (stage 76 rows of a capture).
    #[test]
    fn switches_and_implausible_records() {
        assert!(!switch_in("screen", "instfallback", false));
        assert!(switch_in("screen,instfallback=on", "instfallback", false));
        assert!(!switch_in("screen,repeatdrop=off", "repeatdrop", true));
        assert!(switch_in("screen", "repeatdrop", true));
        let mut r = [0f32; 16];
        (r[0], r[5], r[10], r[3]) = (1.0, 1.0, 1.0, 50.0);
        assert!(plausible_record(&r));
        let mut bad = [0f32; 16];
        (bad[1], bad[4]) = (3.5e29, 1.5e36);
        assert!(!plausible_record(&bad));
        assert!(!plausible_record(&[0.0; 16]), "a zero rotation is not a transform");
    }

    /// Entity fields from a fake draw-list entry and entity.
    #[test]
    fn entity_from_entry() {
        let mut entry = [0u8; 32];
        entry[..8].copy_from_slice(&0x1234_5678u64.to_le_bytes());
        entry[0x18..0x1C].copy_from_slice(&150.5f32.to_le_bytes());
        entry[0x1E..0x20].copy_from_slice(&0x0004u16.to_le_bytes());
        let mut e = vec![0u8; ENTITY_BYTES];
        (e[0x30], e[0x90], e[0x95], e[0x96], e[0xB4]) = (5, 200, 1, 0x23, 0x49);
        e[0xA4..0xA8].copy_from_slice(&0x0042_0007u32.to_le_bytes());
        let got = entity_of(&entry, &e);
        assert_eq!(got, Entity { at: 0x1234_5678, kind: 5, etype: 8, fade: 200, lod_type: 3, flags: 1, map_ref: 0x0042_0007, entry_f: 150.5, entry_lod: 4 });
        // An unreadable entity keeps the entry's fields.
        let bare = entity_of(&entry, &[]);
        assert_eq!((bare.at, bare.kind, bare.entry_lod), (0x1234_5678, 0, 4));
        // The thread's entry: none until the call-site hook sets one.
        LIST_ENTRY.with(|c| c.set(0));
        assert_eq!(current_entity(), Entity::default());
    }

    /// The prelude is `mov r11,[r15+28h]; add r11,r14` and the
    /// call site's 15 bytes relocate (whole instructions, no relative form).
    #[test]
    fn list_call_site_relocates() {
        assert_eq!(LIST_CALL_PRELUDE, [0x4D, 0x8B, 0x5F, 0x28, 0x4D, 0x01, 0xF3]);
        assert_eq!(crate::watch::relocated_len(&LIST_CALL_BYTES), Some(15));
        // The walk's prologue: eight pushes and `sub rsp, 88h` (19 bytes, exe bytes).
        let prologue = [0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x56, 0x57, 0x55, 0x53, 0x48, 0x81, 0xEC, 0x88, 0x00, 0x00, 0x00];
        assert_eq!(prologue[..14], LIST_WALK_ENTRY);
        assert_eq!(crate::watch::relocated_len(&prologue), Some(19));
    }

    #[test]
    fn vertex_buffer_ready() {
        let mut vb = vec![0u8; 0x80];
        assert!(!vb_ready(&vb), "unfilled");
        vb[0..4].copy_from_slice(&0x4042u32.to_le_bytes());
        assert!(!vb_ready(&vb), "Q24 with a zero scale");
        vb[0x70..0x74].copy_from_slice(&1.0f32.to_le_bytes());
        assert!(vb_ready(&vb));
        vb[0x6C] = 3;
        assert!(!vb_ready(&vb), "axis mode past 2");
        vb[0..4].copy_from_slice(&0x3063u32.to_le_bytes());
        assert!(vb_ready(&vb), "float3");
    }

    /// The batch-first prefilter at the expected scale: 160 static batches of 500
    /// (80,000 instances) on a 4 km x 4 km grid around the camera. "Typical"
    /// from measured lines: about half the tested instances are outside the
    /// frustum and a few percent in another LOD slot. Target: under 3 ms a
    /// frame (bounds cached after the first frame). The per-instance-only
    /// prefilter is timed alongside for the gain.
    #[test]
    fn prefilter_80000_instances_cost() {
        let _heavy = heavy();
        let mut t = lod_record([150.0, 400.0, 1200.0, -1.0], [5.0, 20.0, 20.0], (2000.0, 2050.0), false);
        t[5] = 3.0; // a 3 m bounding sphere per instance
        let batches: Vec<std::sync::Arc<Vec<u8>>> = (0..160u32)
            .map(|b| {
                let (bx, by) = ((b % 16) as f32 * 250.0 - 2000.0, (b / 16) as f32 * 400.0 - 2000.0);
                std::sync::Arc::new((0..500u32).flat_map(|i| instance_at([bx + (i % 25) as f32 * 10.0, by + (i / 25) as f32 * 20.0, 0.0])).flat_map(f32::to_le_bytes).collect())
            })
            .collect();
        // Camera at the centre, 2 m up, looking along +y (row vectors: view = rotation; FOV 60 deg).
        let view: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -2.0, 1.0];
        let f = 1.0 / (30f32.to_radians()).tan();
        let proj: [f32; 16] = [f / (16.0 / 9.0), 0.0, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.15, 0.0];
        let planes = frustum_planes(&xnw_world::Camera { viewport: 0, view, proj }.view_proj_relative());
        let cam = [0.0, 0.0, 2.0];
        let mut out = Vec::with_capacity(12 * 80_000);
        let mut frame = |batch_first: bool| {
            out.clear();
            let mut counts = [0u32; 3];
            let (mut tested, mut drawn) = (0, 0);
            let t0 = Instant::now();
            for (k, b) in batches.iter().enumerate() {
                let lod = Some((&t, k % 3));
                let (n, d, _) = if batch_first {
                    prefilter_batch(b, b.len(), lod, None, cam, &mut out, None, &planes, &mut counts)
                } else {
                    prefilter_why(b, lod, None, cam, &mut out, None, Some(&planes), &mut counts)
                };
                tested += n;
                drawn += d;
            }
            (t0.elapsed().as_secs_f64() * 1e3, tested, drawn, counts)
        };
        let warm = frame(true);
        let (ms, tested, drawn, counts) = frame(true);
        let (ms_old, _, drawn_old, _) = frame(false);
        eprintln!("prefilter, 80,000 instances: batch-first {ms:.2} ms (first frame with bounds {:.2} ms), per-instance only {ms_old:.2} ms; tested {tested}, outside {}, wrong LOD {}, drawn {drawn} (per-instance drew {drawn_old})", warm.0, counts[0], counts[1]);
        assert_eq!(tested, 80_000);
        assert!(drawn > 0 && drawn <= drawn_old, "batch tests only remove what the instance tests would");
        assert!(ms < 3.0, "{ms} ms");
    }

    /// A kind-S batch (desc+0x68 set, desc+0x78 0) draws the
    /// static records at L+0x90 (count u16 L+0x98); the bytes are copied once
    /// and shared by the next frame's capture; a count mismatch is refused.
    #[test]
    fn static_map_batches_use_the_list_records() {
        let records: Vec<u8> = (0..3 * 64).map(|i| i as u8).collect();
        let mut l = vec![0u8; 0xA0];
        l[0x90..0x98].copy_from_slice(&(records.as_ptr() as u64).to_le_bytes());
        l[0x98..0x9A].copy_from_slice(&3u16.to_le_bytes());
        let mut desc = vec![0u8; 0x90];
        desc[0..8].copy_from_slice(&(l.as_ptr() as u64).to_le_bytes());
        desc[8..12].copy_from_slice(&3u32.to_le_bytes());
        desc[0x68..0x70].copy_from_slice(&0x1234u64.to_le_bytes());
        let mut rec = vec![0u8; 0xA4];
        rec[0..8].copy_from_slice(&(desc.as_ptr() as u64).to_le_bytes());
        let mut draw = vec![0u8; 16];
        draw[0..8].copy_from_slice(&(rec.as_ptr() as u64).to_le_bytes());
        let run = |frame: u64| {
            let mut c = Capture::new(frame, usize::MAX);
            c.draws.push(Draw::default());
            capture_instances(&mut c, draw.as_ptr() as u64, 0);
            let b = c.draws[0].inst.map(|i| i.block).unwrap();
            c.inst_list[b as usize].clone()
        };
        let (a, b) = (run(1), run(2));
        assert_eq!(a.as_slice(), records.as_slice());
        assert!(std::sync::Arc::ptr_eq(&a, &b), "copied once, shared by frame 2");
        l[0x98] = 2;
        let mut c = Capture::new(3, usize::MAX);
        c.draws.push(Draw::default());
        capture_instances(&mut c, draw.as_ptr() as u64, 0);
        assert_eq!(c.draws[0].inst.map(|i| i.block), Some(NO_PAL), "count mismatch: refused");
        assert_eq!((c.inst_calls[3], c.inst_calls[5]), (1, 0), "counted as count-differs, no buffer fallback at a bad address");
    }

    fn test_state(dir: &std::path::Path, seq: bool) -> State {
        let (tx, _rx) = mpsc::channel::<Capture>();
        let cfg = Config { dir: dir.to_path_buf(), frames: 3, max: 100_000, at: Vec::new(), raw: 1, render: false, seq, skinlog: 0, census: false, auto: None, auto_veh: 0, cut_present: false, pack: false, scan_exit: false, hookin: false };
        State { cfg, title: 0, tls_index: 0, g: Globals::default(), base: AtomicU64::new(0), wait_world: false, done: AtomicBool::new(false), pending: AtomicU64::new(u64::MAX), slots: Mutex::new(Vec::new()), sent: AtomicU64::new(u64::MAX), late: [AtomicU64::new(0), AtomicU64::new(0)], tx: Mutex::new(tx) }
    }

    /// An instanced call's handler fields as 0x1435A4D50 / 0x144E54970
    /// read them: h = *(rec+8) - 0x2C4, h+0x228, h+0x33C/+0x33D, rec+0xA0.
    #[test]
    fn instanced_call_handler_fields() {
        let mut handler = vec![0u8; 0x400];
        handler[0x228..0x22C].copy_from_slice(&7u32.to_le_bytes());
        (handler[0x33C], handler[0x33D]) = (0x43, 0x02);
        let records = vec![0u8; 64 * 2];
        let mut desc = vec![0u8; 0x90];
        desc[8..12].copy_from_slice(&2u32.to_le_bytes());
        desc[0x78..0x80].copy_from_slice(&(records.as_ptr() as u64).to_le_bytes());
        let mut rec = vec![0u8; 0xA4];
        rec[0..8].copy_from_slice(&(desc.as_ptr() as u64).to_le_bytes());
        rec[8..16].copy_from_slice(&(handler.as_ptr() as u64 + 0x2C4).to_le_bytes());
        rec[0xA0..0xA4].copy_from_slice(&(0b10u32 << 15).to_le_bytes());
        let mut call = vec![0u8; 16];
        call[0..8].copy_from_slice(&(rec.as_ptr() as u64).to_le_bytes());
        call[8..12].copy_from_slice(&1u32.to_le_bytes());
        let mut c = Capture::new(1, usize::MAX);
        c.draws.push(Draw::default());
        capture_instances(&mut c, call.as_ptr() as u64, 0);
        let i = c.draws[0].inst.unwrap();
        assert_eq!((i.handler, i.h228, i.h33c, i.rec_a0, i.lod, i.first_lod, i.count), (handler.as_ptr() as u64, 7, 0x0243, 0b10 << 15, 1, 1, 2));
    }

    /// The scan cursor is a pointer: a drains.tsv printed
    /// 0x7ffdc167fad0 with the title at 0x7ffdb8260000, 2,519 records;
    /// a value outside the array reads as none.
    #[test]
    fn scan_cursor_is_a_record_count() {
        let title = 0x7ffd_b826_0000usize;
        assert_eq!(scan_count(title, 0x7ffd_c167_fad0), 2_519);
        assert_eq!(scan_count(title, real(title, SCAN_ARRAY)), 0);
        assert_eq!(scan_count(title, 0), 0);
        assert_eq!(scan_count(title, real(title, SCAN_ARRAY) + 32 * 47_501), 0);
    }

    /// Coherence: a static draw paired with its frame's camera keeps
    /// its absolute position (residual 0) while the camera moves; a draw gone
    /// for one list between two that have it counts once.
    #[test]
    fn coherence_measures_residual_and_gaps() {
        use xnw_world::{Albedo, Camera as RCamera, DrawRecord, FrameList, GeomKey, Material, NO_PALETTE, NO_REMAP};
        let id: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let proj: [f32; 16] = [1.875, 0.0, 0.0, 0.0, 0.0, 3.33333, 0.0, 0.0, 0.0, 0.0, 0.00002, -1.0, 0.0, 0.0, 0.15, 0.0];
        let list = |frame: u64, cam_x: f32, with_b: bool| {
            // The camera at x = cam_x: a view translating by -cam_x; the static object at x = 100 is drawn at 100 - cam_x.
            let mut view = id;
            view[12] = -cam_x;
            let rec = |hash: u64, x: f32| {
                let mut w = id;
                w[12] = x;
                DrawRecord { geom: GeomKey { hash, mesh_at: 1, ..Default::default() }, world: w, viewport: 1, effect: 0, albedo: Albedo::Flat([1.0; 3]), palette: NO_PALETTE, stage: 67, inst_first: 0, inst_count: 0, bone_first: 0, bone_count: 0, material: Material::default(), state: None, remap: NO_REMAP, ped: false }
            };
            let mut draws = vec![rec(1, 100.0 - cam_x)];
            if with_b {
                draws.push(rec(2, 50.0 - cam_x));
            }
            FrameList { frame, draws, cameras: vec![RCamera { viewport: 1, view, proj }], main: Some(1), ..Default::default() }
        };
        if let Ok(mut g) = COHERENCE.lock() {
            *g = None;
        }
        coherence(&list(10, 0.0, true), HashMap::new());
        // Frame 11: draw 2 was hooked and skipped by us.
        coherence(&list(11, 0.5, false), HashMap::from([((2, 67), SKIP_INST_FRUSTUM)]));
        coherence(&list(13, 1.0, true), HashMap::new());
        {
            let g = COHERENCE.lock().unwrap();
            let k = g.as_ref().unwrap();
            assert!(k.residual.iter().all(|r| *r < 1e-4), "{:?}", k.residual);
            assert_eq!((k.missing_back, k.gaps), (1, 1));
            assert_eq!(k.ours.get(&SKIP_INST_FRUSTUM), Some(&1), "the drop is ours");
            assert_eq!(k.ours_still, 1, "the camera only moved: still");
            assert_eq!(k.title_still, [0, 0], "no title drop yet");
            assert_eq!(ours_line(3, &k.ours, k.ours_still, [2, 1], [5, 2]), "missing-then-back hooked but not listed 1 (inst_frustum 1; camera still 1), not hooked 2 (camera still 2, at stages 65/70/71 1); late draws 5 in 2 captures (recorded after their frame's cut)");
        }
        // The same drop while the camera turns 10 degrees a list is not under a still camera.
        let turned = |frame: u64, deg: f32, with_b: bool| {
            let mut l = list(frame, 0.0, with_b);
            let (s, c) = deg.to_radians().sin_cos();
            let v = &mut l.cameras[0].view;
            (v[0], v[2], v[8], v[10]) = (c, -s, s, c);
            l
        };
        if let Ok(mut g) = COHERENCE.lock() {
            *g = None;
        }
        coherence(&turned(20, 0.0, true), HashMap::new());
        coherence(&turned(21, 10.0, false), HashMap::from([((2, 67), SKIP_INST_FRUSTUM)]));
        coherence(&turned(22, 20.0, true), HashMap::new());
        {
            let g = COHERENCE.lock().unwrap();
            let k = g.as_ref().unwrap();
            assert_eq!((k.ours.get(&SKIP_INST_FRUSTUM), k.ours_still), (Some(&1), 0), "a turning camera's drop");
        }
        // The title's drop (not hooked): counted under a still camera, by stage.
        if let Ok(mut g) = COHERENCE.lock() {
            *g = None;
        }
        coherence(&list(30, 0.0, true), HashMap::new());
        coherence(&list(31, 0.5, false), HashMap::new());
        coherence(&list(32, 1.0, true), HashMap::new());
        let g = COHERENCE.lock().unwrap();
        let k = g.as_ref().unwrap();
        assert_eq!((k.ours.len(), k.title_still), (0, [1, 0]), "a stage-67 title drop under a still camera");
    }

    /// Drain cut: draws of title frame 10 recorded on three threads (one of them
    /// already into frame 11) are handed on as ONE frame when drain 10
    /// returns; frame 11 stays open until its drain returns.
    #[test]
    fn drain_cut_hands_on_whole_frames() {
        let (tx, rx) = mpsc::channel::<Capture>();
        let mut s = test_state(&std::env::temp_dir(), false);
        s.tx = Mutex::new(tx);
        let draw = |s: &State, frame: u64, n: usize| {
            let slot = s.slot();
            let mut g = slot.lock().unwrap();
            if g.as_ref().is_some_and(|c| c.frame != frame) {
                drop(g);
                s.flush_drain(frame - 1);
                g = slot.lock().unwrap();
            }
            let c = g.get_or_insert_with(|| Capture::new(frame, usize::MAX));
            for _ in 0..n {
                c.draws.push(Draw::default());
            }
        };
        std::thread::scope(|sc| {
            sc.spawn(|| draw(&s, 10, 3));
            sc.spawn(|| draw(&s, 10, 4));
        });
        std::thread::scope(|sc| {
            sc.spawn(|| {
                draw(&s, 10, 5);
            });
        });
        assert!(rx.try_recv().is_err(), "nothing before the drain returns");
        s.flush_drain(10);
        let f = rx.try_recv().unwrap();
        assert_eq!((f.frame, f.draws.len()), (10, 12), "one list, all three threads");
        assert!(rx.try_recv().is_err());
        std::thread::scope(|sc| {
            sc.spawn(|| draw(&s, 11, 2));
        });
        s.flush_drain(10);
        assert!(rx.try_recv().is_err(), "frame 11 is not closed by drain 10");
        s.flush_drain(11);
        assert_eq!(rx.try_recv().unwrap().draws.len(), 2);
        // A draw of frame 11 recorded after its cut is dropped and counted.
        assert_eq!(s.late.each_ref().map(|a| a.load(Ordering::Relaxed)), [0, 0]);
        std::thread::scope(|sc| {
            sc.spawn(|| draw(&s, 11, 3));
        });
        s.flush_drain(12);
        assert!(rx.try_recv().is_err(), "a late frame is not sent twice");
        assert_eq!(s.late.each_ref().map(|a| a.load(Ordering::Relaxed)), [3, 1]);
    }

    /// A 40-frame sequence written by the writer: every frame complete
    /// (draws.tsv, palettes, marker), shared geometry blobs written once, no
    /// OBJ, bounded size and time.
    #[test]
    fn seq_writer_completes_frames() {
        let _heavy = heavy();
        let dir = std::env::temp_dir().join("xnw_w8_seq");
        let _ = std::fs::remove_dir_all(&dir);
        let s = test_state(&dir, true);
        let raw: Vec<[u32; 3]> = (0..64).map(|i| [i, i, i]).collect();
        let vb = vertex_buffer(&raw, 2);
        let mesh = meshlet_blob(&[((0..64).collect(), (0..40).map(|i| [i % 60, (i + 1) % 60, (i + 2) % 60]).collect())]);
        let t0 = Instant::now();
        for f in 0..40u64 {
            let mut c = Capture::new(1000 + f, f as usize);
            for g in 0..300u64 {
                let geom = 0x10_0000 + 0x100 * g;
                c.draws.push(Draw { geom, ghash: g, world: Some([1.0; 32]), pal_index: NO_PAL, pal2_index: NO_PAL, ..Default::default() });
                // Geometry bytes only the first time (the hook skips dumped ones).
                let data = if f == 0 { GeomData { mesh: Some(mesh.clone()), vb: Some(vb.clone()), ..Default::default() } } else { GeomData::default() };
                c.geoms.insert(geom, data);
            }
            c.pal_list.push(vec![0u8; 48 * 64]);
            write_frame(&s, c).unwrap();
        }
        let secs = t0.elapsed().as_secs_f64();
        let mut total = 0u64;
        for f in 0..40u64 {
            let fd = dir.join(format!("frame-{}", 1000 + f));
            assert!(fd.join("complete").exists() && fd.join("draws.tsv").exists() && fd.join("blobs").join("pal-0.bin").exists(), "frame {f}");
            assert!(!fd.join("scene.obj").exists());
            total += std::fs::read_dir(&fd).unwrap().flatten().map(|e| e.metadata().unwrap().len()).sum::<u64>();
        }
        let shared = std::fs::read_dir(dir.join("blobs")).unwrap().count();
        total += std::fs::read_dir(dir.join("blobs")).unwrap().flatten().map(|e| e.metadata().unwrap().len()).sum::<u64>();
        eprintln!("40 seq frames of 300 draws: {secs:.2} s, {:.1} MB, {shared} shared blobs", total as f64 / 1e6);
        assert_eq!(shared, 600, "300 geometries x (mesh, vb), written once");
        assert!(total < 1_000_000_000 && secs < 60.0);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The always-on capture on a synthetic stream of 5,000 draws over
    /// 500 geometries (fake TLS block, context, viewport, title globals):
    /// the time per frame against the 5 ms target.
    /// A world entry on 5 threads: 6,000 draws a frame, 2,200
    /// geometries first seen in frame 1 (each with VB and meshlet buffer
    /// objects), a third of the draws instanced (64 CPU records, a LOD
    /// record), each thread its own TLS block, context and slot. Hook time
    /// summed over the threads as the log reports it.
    #[test]
    fn hook_cost_five_threads_world_entry() {
        let _heavy = heavy();
        const THREADS: usize = 5;
        const DRAWS: usize = 1_200;
        let matrices: Vec<f32> = (0..32).map(|i| if i % 5 == 0 { 1.0 } else { 0.5 }).collect();
        let slot_word: u16 = 5 << 2;
        let skin_cb = vec![0u32; 32 * 4096];
        let pal_base: u64 = 0;
        let mut s = test_state(&std::env::temp_dir(), false);
        s.cfg.render = true;
        s.g = Globals { matrix_slot: &slot_word as *const u16 as u64, matrix_cache: matrices.as_ptr() as u64, skin_cb: skin_cb.as_ptr() as u64, pal_base: &pal_base as *const u64 as u64, pal_size: 0, arena: 0 };
        // Buffer objects (grc headers) and geometries.
        let mut grc = vec![0u8; 0x20];
        grc[8..12].copy_from_slice(&64u32.to_le_bytes());
        grc[0xC..0xE].copy_from_slice(&16u16.to_le_bytes());
        let self_at = grc.as_ptr() as u64;
        grc[0x18..0x20].copy_from_slice(&self_at.to_le_bytes());
        let ml: u64 = grc.as_ptr() as u64;
        let geoms: Vec<Vec<u8>> = (0..3_000u64)
            .map(|g| {
                let mut b = vec![0u8; 0x48];
                b[8..16].copy_from_slice(&(grc.as_ptr() as u64).to_le_bytes());
                b[0x30..0x38].copy_from_slice(&(g + 1).to_le_bytes());
                b[0x40..0x48].copy_from_slice(&(&ml as *const u64 as u64).to_le_bytes());
                b
            })
            .collect();
        // Instancing: IcDraw -> IcRec (desc, LOD record) -> desc (64 CPU records).
        let records = vec![0u8; 64 * 64];
        let lodrec = vec![0u8; 140];
        let mut desc = vec![0u8; 0x90];
        desc[8..12].copy_from_slice(&64u32.to_le_bytes());
        desc[0x78..0x80].copy_from_slice(&(records.as_ptr() as u64).to_le_bytes());
        let mut rec = vec![0u8; 0xA4];
        rec[0..8].copy_from_slice(&(desc.as_ptr() as u64).to_le_bytes());
        rec[0x10..0x18].copy_from_slice(&(lodrec.as_ptr() as u64).to_le_bytes());
        rec[0xA0..0xA4].copy_from_slice(&(1u32 << 15).to_le_bytes());
        let mut icdraw = vec![0u8; 16];
        icdraw[0..8].copy_from_slice(&(rec.as_ptr() as u64).to_le_bytes());
        let put = |v: &mut Vec<u8>, at: u64, x: u64| v[at as usize..at as usize + 8].copy_from_slice(&x.to_le_bytes());
        let per_thread: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..THREADS)
            .map(|_| {
                let (mut tls, mut ctx, mut vp) = (vec![0u8; 0x1400], vec![0u8; 0x1_3000], vec![0u8; 0x200]);
                put(&mut ctx, CTX_SLOTS + 8 * 5, matrices.as_ptr() as u64);
                ctx[(CTX_SRV_LAZY + 10) as usize] = 0x80;
                vp[0x40] = 1;
                put(&mut tls, TLS_CTX, ctx.as_ptr() as u64);
                put(&mut tls, TLS_VIEWPORT, vp.as_ptr() as u64);
                put(&mut tls, TLS_DRAWABLE, 0x1234_0000);
                (tls, ctx, vp)
            })
            .collect();
        let frame = |now: u64, s: &State| -> (f64, f64, [u64; PARTS]) {
            let t0 = Instant::now();
            std::thread::scope(|sc| {
                for (k, (tls, _, _)) in per_thread.iter().enumerate() {
                    let (geoms, icdraw) = (&geoms, &icdraw);
                    sc.spawn(move || {
                        for i in 0..DRAWS {
                            // Frame 1: 2,200 distinct geometries over the threads; then repeats.
                            let g = &geoms[(k * 440 + i % 440 + now as usize * 7) % geoms.len()];
                            let inst = (i % 3 == 0).then_some(icdraw.as_ptr() as u64);
                            hook_with(s, now, g.as_ptr() as u64, 0, inst, tls.as_ptr() as u64);
                        }
                    });
                }
            });
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            let (mut hook, mut parts) = (0u64, [0u64; PARTS]);
            for slot in s.slots.lock().unwrap().iter() {
                if let Some(c) = slot.lock().unwrap().take() {
                    hook += c.hook_ns;
                    for (a, b) in parts.iter_mut().zip(c.parts) {
                        *a += b;
                    }
                    assert!(c.draws.len() == DRAWS, "one slot per thread");
                    c.recycle();
                }
            }
            (hook as f64 / 1e6, wall, parts)
        };
        let (cold, cold_wall, cold_parts) = frame(1, &s);
        let (warm, warm_wall, parts) = frame(1_000_000, &s);
        // The best of up to 4 warm frames (the CI box runs builds beside the test).
        let (mut warm, mut warm_wall, mut parts) = (warm, warm_wall, parts);
        for _ in 0..3 {
            if warm <= 5.0 {
                break;
            }
            let (w, ww, p) = frame(1_000_000, &s);
            if w < warm {
                (warm, warm_wall, parts) = (w, ww, p);
            }
        }
        let show = |p: [u64; PARTS]| (0..PARTS).map(|k| format!("{} {:.2}", PART_NAMES[k], p[k] as f64 / 1e6)).collect::<Vec<_>>().join(", ");
        eprintln!("hook, 5 threads x {DRAWS} draws: first-seen frame {cold:.2} ms summed ({cold_wall:.2} ms wall) [{}]; warm frame {warm:.2} ms summed ({warm_wall:.2} ms wall) [{}]", show(cold_parts), show(parts));
        // Wall-time bounds hold only on a quiet host (XNW_TEST_TIMING=1), not under CI load.
        assert!(warm < 5.0 || !std::env::var("XNW_TEST_TIMING").is_ok_and(|v| v == "1"), "warm frame {warm:.2} ms summed over the threads");
    }

    /// A bad pointer in the hook's path (an unmapped geometry, a
    /// geometry whose VB object is unmapped) is a counted skip, not a crash.
    #[test]
    fn hook_reads_survive_bad_pointers() {
        // A reserved, uncommitted page: inside the shape check, faults on read.
        // SAFETY: reserve only; freed below.
        let bad = unsafe { xnw_core::win::VirtualAlloc(std::ptr::null(), 0x1_0000, 0x2000, 0x01) } as u64;
        assert_ne!(bad, 0);
        let before = HOOK_FAULTS.load(Ordering::Relaxed);
        // SAFETY: the point of the test: the copy fixes up the fault.
        assert_eq!(unsafe { dread::<u64>(bad + 8) }, None);
        assert_eq!(dcopy(bad, 256), None);
        assert!(HOOK_FAULTS.load(Ordering::Relaxed) >= before + 2);
        // A fault inside a reader's site is counted for it, with its address; the site ends with its guard.
        let wheels = FAULTS_BY[usize::from(SITE_WHEELS)].load(Ordering::Relaxed);
        {
            let _site = fault_site(SITE_WHEELS);
            assert_eq!(dcopy(bad + 0x40, 16), None);
        }
        assert_eq!(FAULT_SITE.with(std::cell::Cell::get), 0, "the site before comes back");
        // A reader's general read inside a named part keeps the part's name.
        let mask = FAULTS_BY[usize::from(SITE_PRESKIN_MASK)].load(Ordering::Relaxed);
        {
            let _part = fault_site(SITE_PRESKIN_MASK);
            let _read = fault_site_or_kept(SITE_PRESKIN);
            assert_eq!(dcopy(bad + 0x80, 4), None);
        }
        assert!(FAULTS_BY[usize::from(SITE_PRESKIN_MASK)].load(Ordering::Relaxed) > mask);
        {
            let _read = fault_site_or_kept(SITE_PRESKIN);
            assert_eq!(FAULT_SITE.with(std::cell::Cell::get), SITE_PRESKIN, "no part: the reader's site");
        }
        assert!(FAULTS_BY[usize::from(SITE_WHEELS)].load(Ordering::Relaxed) > wheels);
        assert!(faults_text().contains("vehicle wheels") && faults_text().contains("last at 0x"), "{}", faults_text());
        // Through the light capture: geometry unmapped, then a geometry whose VB object is unmapped.
        let tls = vec![0u8; 0x1400];
        let s = test_state(&std::env::temp_dir(), false);
        let mut c = Capture::new(1, usize::MAX);
        capture_light(&s, &mut c, bad, 0, tls.as_ptr() as u64, false);
        assert_eq!(c.draws.len(), 0, "unreadable geometry: skipped");
        let mut g = vec![0u8; 0x48];
        g[8..16].copy_from_slice(&(bad + 0x100).to_le_bytes());
        g[0x30..0x38].copy_from_slice(&77u64.to_le_bytes());
        let mut tls = tls;
        tls[TLS_CTX as usize..TLS_CTX as usize + 8].copy_from_slice(&0u64.to_le_bytes());
        capture_light(&s, &mut c, g.as_ptr() as u64, 0, tls.as_ptr() as u64, false);
        assert_eq!((c.draws.len(), c.draws[0].vb_at), (1, 0), "the VB object read faulted: no VB, no crash");
        assert!(HOOK_FAULTS.load(Ordering::Relaxed) >= before + 4);
        // Cost of the fixed-up copy on a good 8-byte read.
        let x = 5u64;
        let t0 = Instant::now();
        let mut sum = 0u64;
        for _ in 0..1_000_000 {
            // SAFETY: a live local.
            sum = sum.wrapping_add(unsafe { dread::<u64>(std::hint::black_box(&x as *const u64 as u64)) }.unwrap_or(0));
        }
        eprintln!("fault-fixed read: {:.1} ns each ({sum})", t0.elapsed().as_secs_f64() * 1e9 / 1e6);
        // SAFETY: our reservation.
        unsafe { xnw_core::win::VirtualFree(bad as *mut _, 0, 0x8000) };
    }

    #[test]
    fn light_capture_cost_5000_draws() {
        let _heavy = heavy();
        let mut tls = vec![0u8; 0x1400];
        let mut ctx = vec![0u8; 0x1_3000];
        let mut vp = vec![0u8; 0x200];
        let matrices: Vec<f32> = (0..32).map(|i| if i % 5 == 0 { 1.0 } else { 0.5 }).collect();
        let slot_word: u16 = 5 << 2;
        let skin_cb = vec![0u32; 32 * 4096];
        let pal_base: u64 = 0;
        let geoms: Vec<Vec<u8>> = (0..500u64).map(|g| {
            let mut b = vec![0u8; 0x48];
            b[0x30..0x38].copy_from_slice(&(g + 1).to_le_bytes());
            b
        }).collect();
        let put = |v: &mut Vec<u8>, at: u64, x: u64| v[at as usize..at as usize + 8].copy_from_slice(&x.to_le_bytes());
        put(&mut tls, TLS_CTX, ctx.as_ptr() as u64);
        put(&mut tls, TLS_VIEWPORT, vp.as_mut_ptr() as u64);
        put(&mut tls, TLS_DRAWABLE, 0x1234_0000);
        put(&mut ctx, CTX_SLOTS + 8 * 5, matrices.as_ptr() as u64);
        ctx[(CTX_SRV_LAZY + 10) as usize] = 0x80;
        vp[0x40] = 1;
        let mut s = test_state(&std::env::temp_dir(), false);
        s.cfg.render = true;
        s.g = Globals { matrix_slot: &slot_word as *const u16 as u64, matrix_cache: matrices.as_ptr() as u64, skin_cb: skin_cb.as_ptr() as u64, pal_base: &pal_base as *const u64 as u64, pal_size: 0, arena: 0 };
        let mut c = Capture::new(1, usize::MAX);
        // One warm frame fills the geometry cache (guarded reads, as the first frame would).
        for i in 0..500 {
            capture_light(&s, &mut c, geoms[i].as_ptr() as u64, 0, tls.as_ptr() as u64, false);
        }
        // Steady state: frame 2 grows the recycled vector, frame 3 is timed.
        c.recycle();
        let mut c = Capture::new(2, usize::MAX);
        let t0 = Instant::now();
        for i in 0..5000 {
            capture_light(&s, &mut c, geoms[i % 500].as_ptr() as u64, 0, tls.as_ptr() as u64, false);
        }
        let cold = t0.elapsed().as_secs_f64() * 1e3;
        c.recycle();
        let mut c = Capture::new(3, usize::MAX);
        let t0 = Instant::now();
        for i in 0..5000 {
            capture_light(&s, &mut c, geoms[i % 500].as_ptr() as u64, 0, tls.as_ptr() as u64, false);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("light capture: first 5,000-draw frame {cold:.2} ms (vector grows)");
        eprintln!("light capture: 5,000 draws in {ms:.2} ms ({:.2} us/draw); {} recorded, world from {:?}; Draw {} B", ms * 1000.0 / 5000.0, c.draws.len(), c.draws[0].world_src, std::mem::size_of::<Draw>());
        assert_eq!(c.draws.len(), 5000);
        assert_eq!(c.draws[0].world_src, WorldSrc::Bound);
        assert!(ms < 5.0 || !std::env::var("XNW_TEST_TIMING").is_ok_and(|v| v == "1"), "{ms} ms");
    }
    /// An LOD record: distances D, fade widths W, far fade (start, end).
    fn lod_record(d: [f32; 4], w: [f32; 3], far: (f32, f32), hard: bool) -> [f32; 35] {
        let mut t = [0f32; 35];
        t[12..15].copy_from_slice(&w);
        t[17] = f32::from_bits(u32::from(hard));
        t[18..22].copy_from_slice(&d);
        (t[22], t[23]) = (far.0, far.1);
        t
    }

    fn instance_at(p: [f32; 3]) -> [f32; 16] {
        [1.0, 0.0, 0.0, p[0], 0.0, 1.0, 0.0, p[1], 0.0, 0.0, 1.0, p[2], 1.0, 0.0, 1.0, 0.0]
    }

    /// A ring buffer with curIndex near 689,220: every skinned draw
    /// gets its bones (the old copy from the base refused them all).
    #[test]
    fn skin_window_of_a_ring_buffer() {
        let lo = 689_000u32;
        let buf: Vec<u8> = (0..3_000u32).flat_map(|b| (0..12).flat_map(move |k| ((lo + b) as f32 + k as f32 / 100.0).to_le_bytes())).collect();
        let mut read = |first: u32, n: u32| {
            let a = 48 * first.checked_sub(lo)? as usize;
            buf.get(a..a + 48 * n as usize).map(<[u8]>::to_vec)
        };
        let spans = [(689_213, 193), (689_220, 1), (689_225, 4), (689_232, 7), (689_220, 1)];
        let k = skin_copy(&spans, u64::MAX, &mut read);
        assert_eq!((k.window, k.at.len(), k.refused), (Some((689_213, 193)), 4, 0));
        let f = k.at[&(689_225, 4)];
        assert_eq!(k.floats[12 * f as usize], 689_225.0, "rebased to the window");
        // Every skinned draw of the frame is drawn skinned.
        let draws: Vec<Draw> = spans.iter().map(|(i, n)| Draw { skinned: true, skin_cb: [*i, 0, *n, 0], ..Default::default() }).collect();
        let skinned = draws.iter().filter(|d| skin_of(d, &k).0 != xnw_world::NO_PALETTE).count();
        assert_eq!(skinned, draws.len(), "a frame with skinned draws must not end with 0 skinned");
        // Sparse spans: one by one.
        let k = skin_copy(&[(689_001, 4), (691_990, 4)], u64::MAX, &mut read);
        assert_eq!((k.window, k.at.len(), k.bones), (None, 2, 8));
        assert_eq!(k.floats[12 * k.at[&(691_990, 4)] as usize], 691_990.0);
        // Past the buffer's size: refused, and said so.
        let k = skin_copy(&[(689_213, 193)], 689_300, &mut read);
        assert_eq!((k.at.len(), k.refused), (0, 1));
        // Unreadable window: falls back to spans; the unreadable one is refused.
        let k = skin_copy(&[(691_990, 4), (691_998, 8)], u64::MAX, &mut read);
        assert_eq!((k.at.len(), k.refused), (1, 1));
    }

    /// A synthetic ped skin object (u16 base tables per comp x 4 LODs at
    /// *(skin+8) / *(skin+0x18), the remap table object at skin+0x20): the
    /// bind is read as 0x144E351B0 reads it, the table copied from the object
    /// (bounded by its size) and placed per (table, base); a draw whose base
    /// did not come from the bind gets no remap.
    #[test]
    fn ped_skin_remap_from_the_skin_object() {
        let cur: Vec<u16> = (0..16).map(|i| 100 + i).collect(); // comp c, lod l: 100 + 4c + l
        let prev: Vec<u16> = (0..16).map(|i| 200 + i).collect();
        let table: Vec<u32> = (0..512u32).map(|i| 0x7000 | (i % 7)).collect();
        let mut obj = vec![0u8; 0x20];
        obj[8..12].copy_from_slice(&512u32.to_le_bytes());
        obj[0xC..0xE].copy_from_slice(&4u16.to_le_bytes());
        obj[0x18..0x20].copy_from_slice(&(table.as_ptr() as u64).to_le_bytes());
        let skin: [u64; 5] = [0, cur.as_ptr() as u64, 0, prev.as_ptr() as u64, obj.as_ptr() as u64];
        let k = ped_skin_read(skin.as_ptr() as u64, 1, 2);
        assert_eq!((k.1, k.2, k.3, k.4, k.5), (obj.as_ptr() as u64, 1, 2, 106, 206));
        // The frame copy.
        let pal: Vec<f32> = vec![0.0; 12 * 8];
        let base: u64 = pal.as_ptr() as u64;
        let mut s = test_state(&std::env::temp_dir(), false);
        s.g.pal_base = &base as *const u64 as u64;
        let size: u32 = 0;
        s.g.pal_size = &size as *const u32 as u64;
        let ped = Draw { skinned: true, skin_cb: [0, 0, 4, 0], remap: [106, 206], ped_skin: (skin.as_ptr() as u64, obj.as_ptr() as u64, 1, 2), ..Default::default() };
        let vehicle = Draw { skinned: true, skin_cb: [4, 0, 4, 0], remap: [5_000_000, u32::MAX], ..Default::default() };
        let c = skin_copy_live(&s, &[ped.clone(), vehicle.clone()]);
        let at = c.remap_at[&(obj.as_ptr() as u64, 106)] as usize;
        assert_eq!(c.floats[at].to_bits(), table[106]);
        assert_eq!(c.remap_at.len(), 1, "the vehicle's arena base is not a ped remap");
        assert!(c.remap_entries <= 512, "bounded by the table object's size");
    }

    /// Ped remap tables appended after the bones, one window or per base.
    #[test]
    fn remap_window_after_the_bones() {
        let table: Vec<u8> = (0..4096u32).flat_map(|i| (1000 + i).to_le_bytes()).collect();
        let mut read = |first: u32, n: u32| table.get(4 * first as usize..4 * (first + n) as usize).map(<[u8]>::to_vec);
        let mut k = SkinCopy { floats: vec![0.0; 24], ..Default::default() };
        remap_copy(&mut k, 7, &[100, 40, 100], &mut read);
        assert_eq!((k.remap_at.len(), k.remap_entries), (2, 316));
        let at = k.remap_at[&(7, 100)] as usize;
        assert_eq!(k.floats[at].to_bits(), 1100, "entry base+0 of base 100");
        assert_eq!(k.floats[k.remap_at[&(7, 40)] as usize + 3].to_bits(), 1043);
        // Past the table: refused per base, never read past what read() returns.
        let mut k = SkinCopy::default();
        remap_copy(&mut k, 7, &[3900], &mut read);
        assert!(k.remap_at.is_empty());
    }

    #[test]
    fn skin_copy_cost_window_and_spans() {
        let _heavy = heavy();
        // 1,357 skinned draws over 400 bones (a measured shape), against 1,357 spans 1,000 bones apart.
        let buf = vec![0u8; 48 * 1_400_000];
        let mut read = |first: u32, n: u32| buf.get(48 * first as usize..48 * (first + n) as usize).map(<[u8]>::to_vec);
        let dense: Vec<(u32, u32)> = (0..1357u32).map(|i| (1_000 + i % 207, [1, 4, 7, 46, 173, 193][i as usize % 6])).collect();
        let sparse: Vec<(u32, u32)> = (0..1357u32).map(|i| (1_000 * i, [1, 4, 7, 46, 173, 193][i as usize % 6])).collect();
        let a = skin_copy(&dense, u64::MAX, &mut read);
        let b = skin_copy(&sparse, u64::MAX, &mut read);
        eprintln!("skin copy: dense {} spans, window {:?}, {} bones in {:.3} ms; sparse {} spans, {} bones span by span in {:.3} ms", a.at.len(), a.window, a.bones, a.ms, b.at.len(), b.bones, b.ms);
        assert!(a.window.is_some() && b.window.is_none());
        assert!((a.ms < 5.0 && b.ms < 40.0) || !std::env::var("XNW_TEST_TIMING").is_ok_and(|v| v == "1"), "dense {:.2} ms, sparse {:.2} ms", a.ms, b.ms);
    }

    /// The null-render profile's exact strings select the screen seq writer.
    #[test]
    fn null_render_profile_selects_the_light_seq_writer() {
        let c = config(Some(r"out\w,raw=1,seq=3180+40/2,skinlog=12"), Some("screen")).unwrap().unwrap();
        assert_eq!(writer(&c), Writer::ScreenSeq);
        assert_eq!((c.at.len(), c.at[0], c.at[39], c.skinlog, c.raw), (40, 3180, 3258, 12, 1));
        let c = config(Some(r"out\w,seq=3180+2"), Some("screen")).unwrap().unwrap();
        assert_eq!((writer(&c), c.raw), (Writer::ScreenSeq, 1), "seq in screen mode dumps even without raw");
        assert_eq!(writer(&config(Some(r"out\w,seq=3180+2"), None).unwrap().unwrap()), Writer::SeqLight);
        assert_eq!(config(None, None).unwrap(), None);
        // Pure capture: the light recorder and the seq writer, no pass.
        let c = config(Some(r"out\w,raw=1,seq=auto:peds>=8:draws>=2000:frames=6"), Some("capture")).unwrap().unwrap();
        assert_eq!((writer(&c), c.auto), (Writer::ScreenSeq, Some((8, 2000, 6))));
        assert_eq!(xnw_world::parse_screen("capture"), None, "capture never makes the pass");
    }

    /// A screen-mode seq frame ends with draws.tsv, palbuf and complete.
    #[test]
    fn screen_seq_frame_writes_the_light_files() {
        let _heavy = heavy();
        let dir = std::env::temp_dir().join("xnw_w8_screen_seq");
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = test_state(&dir, true);
        s.cfg.render = true;
        let pal: Vec<u8> = vec![0x3F; 48 * 64];
        let base: u64 = pal.as_ptr() as u64 - 48 * 689_200;
        let size: u32 = 0;
        s.g.pal_base = &base as *const u64 as u64;
        s.g.pal_size = &size as *const u32 as u64;
        let mut c = Capture::new(3180, 0);
        for i in 0..10u32 {
            c.draws.push(Draw { geom: 0x10_0000 + u64::from(i), skinned: true, skin_cb: [689_200 + i, 0, 4, 0], world: Some([1.0; 32]), pal_index: NO_PAL, pal2_index: NO_PAL, ..Default::default() });
        }
        let mut cache = xnw_world::GeometryCache::<()>::default();
        render_seq_frame(&s, c, &mut cache, Some(&dir));
        let fd = dir.join("frame-3180");
        for f in ["draws.tsv", "complete", "blobs/palbuf.bin", "blobs/palbuf.tsv"] {
            assert!(fd.join(f).exists(), "{f}");
        }
        // Every list row names its v1 draw.
        let map = std::fs::read_to_string(fd.join("list-draws.tsv")).unwrap();
        let rows = std::fs::read_to_string(fd.join("list.tsv")).unwrap().lines().count() - 1;
        assert_eq!(map.lines().count() - 1, rows, "{map}");
        assert!(map.lines().skip(1).all(|l| l.split('\t').nth(1).and_then(|d| d.parse::<usize>().ok()).is_some_and(|d| d < 10)));
        let tsv = std::fs::read_to_string(fd.join("blobs/palbuf.tsv")).unwrap();
        assert!(tsv.contains("689200\t4\t0") && tsv.contains("689209\t4\t9"), "{tsv}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every skip site in render_frame writes its code to skips.tsv and
    /// the frame line; listed draws are 0.
    #[test]
    fn skip_reasons_reach_the_capture() {
        let _heavy = heavy();
        let dir = std::env::temp_dir().join("xnw_geo3_skips");
        let _ = std::fs::remove_dir_all(&dir);
        let frame = 4242u64;
        let mut cache = xnw_world::GeometryCache::<()>::default();
        let mut c = Capture::new(frame, 0);
        let ident = |t: f32| -> [f32; 32] { std::array::from_fn(|i| if i % 5 == 0 && i < 16 { 1.0 } else if i == 12 { t } else { 0.0 }) };
        // 0: no buffers. 1: vertex buffer with an unknown header (not ready).
        let raw = vec![0u8; 0x100];
        c.draws.push(Draw { ghash: 1, world: Some(ident(0.0)), ..Default::default() });
        c.draws.push(Draw { ghash: 2, vb_at: raw.as_ptr() as u64, vb_len: raw.len() as u64, mesh_at: raw.as_ptr() as u64, mesh_len: 16, world: Some(ident(0.0)), ..Default::default() });
        // Cached geometries (unreadable addresses, so the first loop reads nothing new).
        let cached = |h: u64| Draw { ghash: h, vb_at: 0x10, vb_len: 64, mesh_at: 0x20, mesh_len: 16, world: Some(ident(1.0)), ..Default::default() };
        for h in 10..16 {
            cache.get_or_make(key_of(&cached(h)), frame, || Some(()));
        }
        c.draws.push(cached(10)); // 2: listed
        c.draws.push(cached(10)); // 3: repeat
        c.draws.push(Draw { world: None, ..cached(11) }); // 4: no world
        c.draws.push(Draw { world: Some([1e9; 32]), ..cached(12) }); // 5: implausible
        c.draws.push(Draw { inst: Some(InstInfo { block: NO_PAL, ..Default::default() }), ..cached(13) }); // 6: refused
        c.inst_list.push(std::sync::Arc::new(Vec::new()));
        c.draws.push(Draw { inst: Some(InstInfo { block: 0, ..Default::default() }), ..cached(14) }); // 7: no camera
        c.draws.push(Draw { geom: 9, ..cached(99) }); // 8: never cached, unreadable vertex buffer
        render_frame(c, &mut cache, Some(&dir));
        let tsv = std::fs::read_to_string(dir.join(format!("frame-{frame}")).join("skips.tsv")).unwrap();
        let got: Vec<&str> = tsv.lines().skip(1).map(|l| l.rsplit('\t').next().unwrap()).collect();
        assert_eq!(got, ["no_buffers", "vb_not_ready", "listed", "repeat", "no_world", "implausible_world", "inst_refused", "inst_no_camera", "vb_unreadable"], "{tsv}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn skip_reason_of_an_empty_batch_and_the_line() {
        assert_eq!(inst_zero_reason(0, 0, [0; 3]), SKIP_INST_EMPTY);
        assert_eq!(inst_zero_reason(10, 0, [7, 2, 1]), SKIP_INST_FRUSTUM);
        assert_eq!(inst_zero_reason(10, 0, [2, 7, 1]), SKIP_INST_WRONG_LOD);
        assert_eq!(inst_zero_reason(10, 6, [2, 2, 0]), SKIP_INST_HIDDEN);
        assert_eq!(inst_zero_reason(10, 0, [0, 0, 9]), SKIP_INST_INSANE);
        assert_eq!(inst_zero_reason(10, 0, [1, 1, 0]), SKIP_INST_LOD_NONE);
        assert_eq!(inst_zero_reason(4, 0, [2, 2, 0]), SKIP_INST_FRUSTUM, "a tie goes to the earlier test");
        assert_eq!(skip_line(&[0, 7, 7, 13, 0]), "3 of 5 (repeat 2, inst_frustum 1)");
        assert_eq!(skip_line(&[0]), "0 of 1 (none)");
        assert_eq!(skip_rows(&[0, 17]), "draw\tcode\treason\n0\t0\tlisted\n1\t17\tno_world\n");
    }

    #[test]
    fn sun_from_the_timecycle() {
        let mut tc = vec![0u8; 0x16C0];
        let put = |tc: &mut Vec<u8>, o: usize, v: f32| tc[o..o + 4].copy_from_slice(&v.to_le_bytes());
        for (o, v) in [(0x0, 0.89), (0x4, 0.675), (0x8, 0.392), (0xC, 14.75), (0x1630, 0.0), (0x1634, 0.0), (0x1638, 1.0)] {
            put(&mut tc, o, v);
        }
        let s = sun_of(&tc).unwrap();
        assert_eq!(&s[..3], &[0.0, 0.0, 1.0], "the final light direction");
        assert!((s[4] - 14.75 / 18.75 * 1.25).abs() < 1e-4 && s[6] < s[5] && s[5] < s[4], "hue kept: {s:?}");
        assert!((s[3] - 0.15).abs() < 1e-6);
        for (o, v) in [(0x1610, 0.0), (0x1614, -2.0), (0x1618, 0.0), (0x330, 2.0)] {
            put(&mut tc, o, v);
        }
        let s = sun_of(&tc).unwrap();
        assert_eq!(&s[..3], &[0.0, 0.0, 1.0], "the final light direction wins over +0x1610");
        put(&mut tc, 0x1638, 0.0);
        put(&mut tc, 0x1614, -1.0);
        let s = sun_of(&tc).unwrap();
        assert_eq!(&s[..3], &[0.0, -1.0, 0.0], "no final direction: the sun direction");
        assert!((s[3] - 0.35).abs() < 1e-5);
        assert!(sun_of(&tc[..0x100]).is_none());
    }

    /// A captured night timecycle (f3180 values): sun below, moon disc up,
    /// a dark blue sky, fog over 10 km.
    #[test]
    fn sky_from_the_timecycle() {
        let mut tc = vec![0u8; 0x16C0];
        let put = |tc: &mut Vec<u8>, o: usize, v: f32| tc[o..o + 4].copy_from_slice(&v.to_le_bytes());
        for (o, v) in [(0x0, 1.0), (0x4, 1.0), (0x8, 1.0), (0xC, 1.0), (0x1610, 0.686), (0x1614, -0.0911), (0x1618, -0.7218), (0x1620, -0.5522), (0x1624, 0.53), (0x1628, 0.6436), (0x1630, -0.5522), (0x1634, 0.53), (0x1638, 0.6436), (0x72C, 1.5), (0x14C4, 10_000.0), (0x62C, 0.1601), (0x630, 0.3741), (0x634, 0.9134)] {
            put(&mut tc, o, v);
        }
        let s = sky_of(&tc).unwrap();
        assert_eq!(s.sun_col, [0.0; 4], "sun below the horizon: no disc");
        assert_eq!(s.moon[3], 1.0);
        assert!(s.zenith[2] > s.zenith[0] && s.zenith[2] < 0.2, "dark blue night zenith {:?}", s.zenith);
        assert!((s.fog[3] - 1e-4).abs() < 1e-9);
        // Day: sun up, blue zenith.
        put(&mut tc, 0x1618, 0.7);
        put(&mut tc, 0xC, 14.75);
        let s = sky_of(&tc).unwrap();
        assert!(s.sun_col[3] == 1.0 && s.moon[3] == 0.0 && s.zenith[2] > 0.5 && s.zenith[2] > 2.0 * s.zenith[0], "{s:?}");
    }

    /// A 4x4 map with one dry corner: cells with a dry corner drop out;
    /// positions from the inverse transform; heights 1600 t - 800.
    #[test]
    fn water_grid_from_a_map() {
        let _heavy = heavy();
        let m = WaterMap { map: 1, data: 1, w: 4, h: 4, aabb: [0.0, 0.0, 400.0, 400.0], s: [0.01, 0.01, 0.0, 0.0] };
        let mut tex = vec![32768u16; 16];
        tex[0] = 0;
        let g = water_grid(&m, &tex);
        assert_eq!(g.indices.len() / 6, 8, "9 cells, the one at the dry corner dropped");
        let v = g.vertices[0];
        assert!((v[0] - 150.0).abs() < 1e-3 && (v[1] - 50.0).abs() < 1e-3, "texel (1, 0) centre: {v:?}");
        assert!((v[2] - (1600.0 * 32768.0 / 65535.0 - 800.0)).abs() < 1e-3);
        assert_eq!(water_grid(&m, &[0; 16]).indices.len(), 0);
        // A large map is sampled: at most WATER_GRID cells a side.
        let big = WaterMap { w: 2048, h: 1024, ..m };
        let g = water_grid(&big, &vec![1u16; 2048 * 1024]);
        assert_eq!(g.indices.len() / 6, 511 * 255);
        // The dump round-trips through load_water.
        let dir = std::env::temp_dir().join("xnw_w10_water");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), format!("map 0x1\nwidth 4\nheight 4\naabb {:?}\ntransform {:?}\n", m.aabb, m.s)).unwrap();
        std::fs::write(dir.join("a.bin"), tex.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let (lm, lt) = load_water(&dir).unwrap();
        assert_eq!((lm.w, lm.s, lt), (4, m.s, tex));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn terrain_report_fields() {
        let mut recs = vec![0u8; 3 * TERRAIN_REC];
        for i in 0..3 {
            let r = &mut recs[i * TERRAIN_REC..(i + 1) * TERRAIN_REC];
            r[0x30..0x34].copy_from_slice(&(512.0 * i as f32).to_le_bytes());
            r[0x168..0x16C].copy_from_slice(&16u32.to_le_bytes());
            r[0x16C..0x174].copy_from_slice(&(i as u64).to_le_bytes());
            r[0x144..0x148].copy_from_slice(&(9000 + i as u32).to_le_bytes());
            r[0x8..0xC].copy_from_slice(&1.5f32.to_le_bytes());
        }
        let (tiles, fields) = terrain_report(&recs);
        assert_eq!(tiles.lines().count(), 4);
        assert!(tiles.lines().nth(3).unwrap().starts_with("2	1024	0	0	16	2	"), "{tiles}");
        assert!(fields.lines().any(|l| l.starts_with("+0x144	3	") && l.ends_with("heightmap SRV candidate)")), "{fields}");
        assert!(fields.lines().any(|l| l.starts_with("+0x8	1	") && l.ends_with("float")), "{fields}");
        assert!(fields.lines().any(|l| l.starts_with("+0x0	1	") && l.ends_with("zero")));
    }

    /// Distinct (effect, programs, stage, albedo) rows with
    /// draw counts; names and program hashes read from the objects.
    /// An effect pointer near u64::MAX (or 0) is "-", not a panic;
    /// bytes() refuses a range that would wrap or cross the user space end.
    #[test]
    fn bad_effect_pointers() {
        assert_eq!(effect_name(u64::MAX - 8), "-");
        assert_eq!(effect_name(0), "-");
        assert_eq!(effect_name(0x7FFF_FFFF_FFF0), "-");
        assert!(bytes(0x7FFF_FFFF_FFF8, 16).is_none());
        assert!(bytes(u64::MAX - 4, 8).is_none());
        assert_eq!(program_hashes(u64::MAX - 8), [0; 4]);
    }

    #[test]
    fn program_census_rows() {
        let name = b"trees_cutout\0".to_vec();
        let mut fx = vec![0u8; 0xD10];
        fx[0xD08..0xD10].copy_from_slice(&(name.as_ptr() as u64).to_le_bytes());
        let progs: Vec<Vec<u8>> = (0..4u32).map(|k| {
            let mut p = vec![0u8; 0x20];
            p[0x18..0x1C].copy_from_slice(&(0xA0 + k).to_le_bytes());
            p
        }).collect();
        let set: Vec<u8> = progs.iter().flat_map(|p| (p.as_ptr() as u64).to_le_bytes()).collect();
        let (fxa, seta) = (fx.as_ptr() as u64, set.as_ptr() as u64);
        let frame: HashMap<(u64, u64, u32, u8), u32> = [((fxa, seta, 67, 0), 5), ((fxa, seta, 67, 1), 2), ((fxa, seta, 65, 0), 3)].into_iter().collect();
        let dir = std::env::temp_dir().join("xnw_w11_census");
        let _ = std::fs::remove_dir_all(&dir);
        if let Ok(mut g) = CENSUS.lock() {
            *g = None;
        }
        census_add(&frame, None);
        census_add(&frame, Some(&dir));
        let text = std::fs::read_to_string(dir.join("census-programs.tsv")).unwrap();
        assert!(text.lines().nth(1).unwrap() == "trees_cutout\t000000a0\t000000a1\t000000a2\t000000a3\t67\tfound\t10", "{text}");
        assert!(text.contains("\t67\tno albedo variable\t4"), "{text}");
        assert_eq!(text.lines().count(), 4);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn render_state_records() {
        // The decal DSS {1, 0, 7} (test GE, no write); raster bias -8, slope 1.5; RT0 additive.
        let dss = [1u8, 0, 7, 0, 0xFF, 0xFF, 1, 1, 1, 8, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut ras = [0u8; 16];
        ras[..4].copy_from_slice(&[3, 1, 0, (-8i8) as u8]);
        ras[8..12].copy_from_slice(&1.5f32.to_le_bytes());
        let mut blend = [0u8; 68];
        blend[4..12].copy_from_slice(&[1, 2, 2, 1, 2, 2, 1, 0xF]);
        let s = decode_state(&dss, &ras, &blend);
        assert!(s.depth_enable && !s.depth_write);
        assert_eq!((s.depth_func, s.depth_bias, f32::from_bits(s.slope_bias), s.blend), (7, -8, 1.5, [1, 2, 2, 1, 2, 2, 1, 0xF]));
    }

    #[test]
    fn lod_rule_per_ra16() {
        let t = lod_record([50.0, 150.0, -1.0, -1.0], [5.0, 20.0, 0.0], (400.0, 420.0), false);
        let r = instance_at([0.0; 3]);
        assert_eq!(lod_slot(&t, &r, 10.0), Some(0));
        assert_eq!(lod_slot(&t, &r, 47.0), Some(0), "inside the band, soft: stays");
        assert_eq!(lod_slot(&t, &r, 49.99), Some(1), "band end (x < 1/255): next slot");
        assert_eq!(lod_slot(&t, &r, 100.0), Some(1));
        assert_eq!(lod_slot(&t, &r, 399.0), Some(1), "the last slot has no upper bound");
        assert_eq!(lod_slot(&t, &r, 410.0), Some(1), "fading out, still drawn");
        assert_eq!(lod_slot(&t, &r, 420.0), None, "past the far fade end");
        let hard = lod_record([50.0, 150.0, -1.0, -1.0], [5.0, 20.0, 0.0], (400.0, 420.0), true);
        assert_eq!(lod_slot(&hard, &r, 48.0), Some(1), "hard switch at mid-band");
        assert_eq!(lod_slot(&hard, &r, 46.0), Some(0));
        let mut zero = r;
        zero[14] = 0.0;
        assert_eq!(lod_slot(&t, &zero, 10.0), None, "rec+0x38 <= 0 drops");
        let one = lod_record([-1.0; 4], [0.0; 3], (400.0, 420.0), false);
        assert_eq!(lod_slot(&one, &r, 300.0), Some(0), "single LOD");
        // Prefilter: absolute translations against the absolute camera.
        let cam = [1000.0, 2000.0, 30.0];
        let block: Vec<u8> = [10.0f32, 100.0, 500.0].iter().flat_map(|x| instance_at([cam[0] + x, cam[1], cam[2]])).flat_map(f32::to_le_bytes).collect();
        let mut out = Vec::new();
        assert_eq!(prefilter(&block, Some((&t, 0)), None, cam, &mut out), (3, 1, 0));
        assert_eq!(out.len(), 12);
        assert!((out[9] - 10.0).abs() < 1e-3 && out[10].abs() < 1e-3, "camera-relative translation {:?}", &out[9..12]);
        out.clear();
        assert_eq!(prefilter(&block, Some((&t, 1)), None, cam, &mut out), (3, 1, 0));
        assert_eq!(prefilter(&block, None, None, cam, &mut out), (3, 3, 0));
        // The title's hide bitset: instance 1 (bit 1 of word 0) hidden.
        assert_eq!(prefilter(&block, None, Some(&[0b10]), cam, &mut out), (3, 2, 1));
        assert_eq!(prefilter(&block, None, Some(&[]), cam, &mut out), (3, 3, 0), "a short bitset hides nothing past it");
    }

    #[test]
    fn prefilter_cost_1000_batches() {
        let _heavy = heavy();
        let t = lod_record([60.0, 200.0, 600.0, -1.0], [5.0, 20.0, 20.0], (900.0, 920.0), false);
        let batches: Vec<Vec<u8>> = (0..1000u32)
            .map(|b| (0..64u32).flat_map(|i| instance_at([(b % 40) as f32 * 20.0 + i as f32, (b / 40) as f32 * 20.0, 0.0])).flat_map(f32::to_le_bytes).collect())
            .collect();
        let mut out = Vec::with_capacity(12 * 64_000);
        let t0 = Instant::now();
        let (mut tested, mut drawn) = (0, 0);
        for (b, block) in batches.iter().enumerate() {
            let (n, k, _) = prefilter(block, Some((&t, b % 3)), None, [100.0, 100.0, 2.0], &mut out);
            tested += n;
            drawn += k;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("prefilter: 1,000 batches, {tested} instances tested, {drawn} drawn in {ms:.2} ms");
        assert_eq!(tested, 64_000);
        assert!(drawn > 0 && drawn < tested);
        assert!(ms < 3.0 || !std::env::var("XNW_TEST_TIMING").is_ok_and(|v| v == "1"), "{ms} ms");
    }

    #[test]
    fn switch_grammar() {
        let c = parse(r"out\w1").unwrap();
        assert_eq!((c.frames, c.max, c.raw, c.at.len()), (3, 20_000, 0, 0));
        let c = parse(r"out\w1, frames=5, max=100, at=900+400, raw=1").unwrap();
        assert_eq!((c.frames, c.max, c.raw, c.at.clone()), (5, 100, 1, vec![400, 900]));
        assert_eq!(parse("d,skinlog=8").map(|c| c.skinlog), Ok(8));
        assert_eq!(parse("d,raw=1,seq=auto:peds>=8:draws>=2000:frames=6").map(|c| (c.auto, c.seq)), Ok((Some((8, 2000, 6)), true)));
        assert!(parse("d,seq=auto:bogus>=1").is_err());
        assert_eq!(parse("d,seq=10+120/1,scan=exit").map(|c| (c.scan_exit, c.at.len())), Ok((true, 120)));
        assert!(!parse("d,scan=drain").unwrap().scan_exit && parse("d,scan=x").is_err());
        // The wrapper's first 15 bytes move to the trampoline (its call made absolute).
        assert_eq!(crate::watch::relocated_len(&COMPACT_WRAPPER_BYTES), Some(15));
        assert_eq!(parse("d,seq=auto:peds>=4:draws>=2000:vehicles>=20:frames=6").map(|c| (c.auto, c.auto_veh)), Ok((Some((4, 2000, 6)), 20)));
        assert!(is_vehicle_effect("standard_vehicle_paint") && is_vehicle_effect("vehicle_tyre") && !is_vehicle_effect("standard"));
        assert!(parse("").is_err());
        assert!(parse("d,bogus=1").is_err());
        assert!(parse("d,frames").is_err());
        assert_eq!(offsets(5), vec![3, 30, 120, 240, 360]);
    }

    /// A real blob pair from an earlier probe run (`raw=1`), if any: decodes
    /// without error and to finite positions. Skips without one.
    #[test]
    fn real_blob_if_present() {
        let Ok(frames) = std::fs::read_dir(r"out\w1") else {
            eprintln!("skipped: no out\\w1");
            return;
        };
        for f in frames.flatten() {
            let Ok(blobs) = std::fs::read_dir(f.path().join("blobs")) else { continue };
            for b in blobs.flatten() {
                let p = b.path();
                if p.extension().is_some_and(|e| e == "mesh") {
                    let (Ok(mesh), Ok(vb)) = (std::fs::read(&p), std::fs::read(p.with_extension("vb"))) else { continue };
                    let m = decode_meshlets(&mesh, &vb).unwrap();
                    assert!(m.pos.iter().all(|p| p.iter().all(|v| v.is_finite())));
                    eprintln!("{}: {} meshlets, {} vertices, {} triangles", p.display(), m.meshlets, m.pos.len(), m.tris.len());
                    return;
                }
            }
        }
        eprintln!("skipped: no raw blob under out\\w1");
    }
}
