//! `xnw movie-info <movies.rpf> [--list] [--entry I] [--frame N] [--out file.bmp]
//! [--io callbacks|memory] [--frames-buffers title|bink] [--shaders <shaders.rpf>]`
//!
//! The intro-movie path without the GPU:
//!
//! - lists the Bink 2 movies stored in `movies.rpf` (the archive's table of
//!   contents is encrypted, so entries are found by their `KB2` headers, each
//!   on a 4 KiB boundary; names are not recoverable from the archive);
//! - maps the title's own `Bink2Scarlett.dll` with our loader, binds it
//!   (`xmem.dll` from our HLE, CRT and Windows from the host; audio stays
//!   off: no sound system is set and no track is asked for), attaches it;
//! - opens one movie from the bytes read out of the RPF the way the title
//!   does: through `BinkSetOSFileCallbacks` (open/read/seek/close over our
//!   buffer; `--io memory` uses the from-memory flag instead), with frame
//!   buffers the caller allocates and registers (`BinkGetFrameBuffersInfo`
//!   then `BinkRegisterFrameBuffers`, as `title+0x442f760` does);
//! - decodes frames 1..=N on the CPU (`BinkDoFrame`, `BinkNextFrame`),
//!   prints the plane layout Bink reports and writes frame N as a BMP.
//!
//! `--shaders` runs the pack's Bink shaders through the translator.
//!
//! `--audio out.wav` decodes the movie's first audio track instead of a
//! picture: Bink's track API (`BinkOpenTrack`, then `BinkGetTrackData` per
//! frame with `BinkNextFrame` between, the public Bink SDK calls) hands
//! over each frame's PCM, written as a WAV with its peak and RMS. Video is
//! not decoded meanwhile: with `BinkDoFrame` before each `BinkGetTrackData`
//! the track read past the movie's end and faulted in Bink at frame 2
//! (callbacks) or 29 (memory), measured on movie #1. Through the file
//! callbacks the track alone still faults (`Bink2Scarlett.dll+0x1ee1b`,
//! reading just past a page), so `--audio` opens from memory. The title instead gives Bink a sound system of its own
//! (`BinkSetSoundSystem`); the samples are the
//! same decoder's.
//!
//! The bind resolver repeats the few lines of `run.rs`'s resolver that a
//! title DLL needs (HLE, host override, host); copied, not shared, so that
//! file stays with its owner.

use std::ffi::{CStr, c_char, c_void};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use xnw_hle::{Route, route};
use xnw_loader::{Loaded, Provider};
use xnw_pe::Symbol;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage: xnw movie-info <movies.rpf> [--list] [--entry I] [--frame N] [--out file.bmp] \
                     [--io callbacks|memory] [--frame-buffers title|bink] [--shaders <shaders.rpf>] [--audio out.wav]";
const PAGE: u64 = 0x1000;
/// The pitch we give every plane: D3D12's texture row alignment, so the
/// same bytes can be uploaded as a linear texture without repacking.
const PITCH_ALIGN: u32 = 256;

/// The fixed part of a Bink 2 file header (all little-endian u32 after the
/// magic), as read from the 515 movies of `movies.rpf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinkHeader {
    pub magic: [u8; 4],
    /// Whole file, header included.
    pub size: u64,
    pub frames: u32,
    pub largest_frame: u32,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub flags: u32,
    pub tracks: u32,
}

impl BinkHeader {
    pub fn parse(h: &[u8]) -> Option<Self> {
        if h.len() < 44 || &h[..3] != b"KB2" {
            return None;
        }
        let w = |i: usize| u32::from_le_bytes(h[4 + 4 * i..8 + 4 * i].try_into().expect("4 bytes"));
        Some(BinkHeader {
            magic: h[..4].try_into().expect("4 bytes"),
            size: u64::from(w(0)) + 8,
            frames: w(1),
            largest_frame: w(2),
            width: w(4),
            height: w(5),
            fps_num: w(6),
            fps_den: w(7),
            flags: w(8),
            tracks: w(9),
        })
    }

    pub fn seconds(&self) -> f64 {
        if self.fps_num == 0 {
            return 0.0;
        }
        f64::from(self.frames) * f64::from(self.fps_den) / f64::from(self.fps_num)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Movie {
    pub offset: u64,
    pub header: BinkHeader,
}

/// Every Bink movie in an RPF9 archive, in file order, reading only headers.
pub fn list(file: &mut File) -> Res<(xnw_shader::rpf9::Header, Vec<Movie>)> {
    let len = file.metadata()?.len();
    let mut h = [0u8; 0x40];
    file.read_exact(&mut h)?;
    let header = xnw_shader::rpf9::Header::parse(&h)?;
    let end = len.checked_sub(u64::from(header.trailer_len)).ok_or("trailer past the file")?;
    let mut movies = Vec::new();
    let mut at = first_nonzero(file, u64::from(header.toc_end), end)?;
    let mut buf = [0u8; 64];
    while let Some(off) = at {
        file.seek(SeekFrom::Start(off))?;
        file.read_exact(&mut buf)?;
        let Some(b) = BinkHeader::parse(&buf) else {
            return Err(format!("no Bink header at {off:#x} (entry {})", movies.len()).into());
        };
        movies.push(Movie { offset: off, header: b });
        // The next entry: the next page that does not start with zeros.
        let mut page = (off + b.size).next_multiple_of(PAGE);
        at = None;
        while page + 4 <= end {
            file.seek(SeekFrom::Start(page))?;
            file.read_exact(&mut buf[..4])?;
            if buf[..4] != [0; 4] {
                at = Some(page);
                break;
            }
            page += PAGE;
        }
    }
    Ok((header, movies))
}

fn first_nonzero(file: &mut File, from: u64, end: u64) -> Res<Option<u64>> {
    let mut buf = vec![0u8; 0x10000];
    let mut at = from;
    while at < end {
        file.seek(SeekFrom::Start(at))?;
        let n = file.read(&mut buf[..(end - at).min(0x10000) as usize])?;
        if n == 0 {
            break;
        }
        if let Some(p) = buf[..n].iter().position(|&b| b != 0) {
            return Ok(Some(at + p as u64));
        }
        at += n as u64;
    }
    Ok(None)
}

// ---- Bink2Scarlett.dll, as the title binds it -------------------------------

/// One plane as `BINKFRAMEBUFFERS` holds it (layout from the title's use at
/// `title+0x442f760`: buffer at +0, allocate flag at +8, pitch at +12).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Plane {
    buffer: usize,
    allocate: i32,
    pitch: u32,
}

/// `BINKFRAMEBUFFERS`: 264 bytes (the title keeps it at object+88 and its
/// next field at object+352). Plane order Y, cR, cB, A, H (Bink SDK names).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct FrameBuffers {
    total_frames: i32,
    ya_width: u32,
    ya_height: u32,
    c_width: u32,
    c_height: u32,
    frame_num: u32,
    frames: [[Plane; 5]; 3],
}

const PLANE_NAMES: [&str; 5] = ["Y", "cR", "cB", "A", "H"];

type OpenCb = extern "system" fn(*mut usize, *const c_void) -> u32;
type ReadCb = extern "system" fn(*mut usize, *mut u8, u64) -> u64;
type SeekCb = extern "system" fn(*mut usize, u64);
type CloseCb = extern "system" fn(*mut usize);

struct Bink {
    set_os_file_callbacks: extern "system" fn(OpenCb, ReadCb, SeekCb, CloseCb),
    open_with_options: extern "system" fn(*const c_void, *const u8, u32) -> *mut u32,
    get_frame_buffers_info: extern "system" fn(*mut u32, *mut FrameBuffers),
    register_frame_buffers: extern "system" fn(*mut u32, *const FrameBuffers),
    do_frame: extern "system" fn(*mut u32) -> i32,
    next_frame: extern "system" fn(*mut u32),
    close: extern "system" fn(*mut u32),
    get_error: extern "system" fn() -> *const c_char,
}

/// The movie bytes Bink reads through our callbacks.
struct Source {
    data: *const u8,
    len: u64,
}

struct Cursor {
    src: *const Source,
    pos: AtomicU64,
}

static OPENS: AtomicU64 = AtomicU64::new(0);
static READS: AtomicU64 = AtomicU64::new(0);
static READ_BYTES: AtomicU64 = AtomicU64::new(0);
static READ_MAX: AtomicU64 = AtomicU64::new(0);
static SEEKS: AtomicU64 = AtomicU64::new(0);

extern "system" fn cb_open(user: *mut usize, name: *const c_void) -> u32 {
    OPENS.fetch_add(1, Ordering::Relaxed);
    let c = Box::new(Cursor { src: name.cast(), pos: AtomicU64::new(0) });
    // SAFETY: Bink hands us its slot for the handle.
    unsafe { *user = Box::into_raw(c) as usize };
    1
}

extern "system" fn cb_read(user: *mut usize, dest: *mut u8, bytes: u64) -> u64 {
    // SAFETY: the slot holds the cursor `cb_open` made; `src` outlives the movie.
    let (c, src) = unsafe {
        let c = &*(*user as *const Cursor);
        (c, &*c.src)
    };
    let pos = c.pos.load(Ordering::Relaxed).min(src.len);
    let n = bytes.min(src.len - pos);
    // SAFETY: both ranges are in bounds; Bink's buffer holds `bytes`.
    unsafe { std::ptr::copy_nonoverlapping(src.data.add(pos as usize), dest, n as usize) };
    c.pos.store(pos + n, Ordering::Relaxed);
    READS.fetch_add(1, Ordering::Relaxed);
    READ_BYTES.fetch_add(n, Ordering::Relaxed);
    READ_MAX.fetch_max(n, Ordering::Relaxed);
    n
}

extern "system" fn cb_seek(user: *mut usize, pos: u64) {
    SEEKS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: as in `cb_read`.
    unsafe { (*(*user as *const Cursor)).pos.store(pos, Ordering::Relaxed) };
}

extern "system" fn cb_close(user: *mut usize) {
    // SAFETY: the slot holds the cursor `cb_open` made, closed once.
    unsafe {
        if *user != 0 {
            drop(Box::from_raw(*user as *mut Cursor));
            *user = 0;
        }
    }
}

fn symbol_key(symbol: &Symbol) -> String {
    match symbol {
        Symbol::Name(n) => n.clone(),
        Symbol::Ordinal(o) => format!("#{o}"),
    }
}

/// Maps, binds and attaches `Bink2Scarlett.dll` (resolver lines from `run.rs`).
fn load_bink(dll: &Path, cache: &Path) -> Res<(&'static Loaded, Bink)> {
    let bink: &'static Loaded = Box::leak(Box::new(xnw_loader::load(dll, cache)?));
    xnw_hle::add_image(&bink.name, bink.base(), bink.size);
    let resolve = |dll: &str, symbol: &Symbol| -> Option<(usize, Provider)> {
        let name = dll.to_ascii_lowercase();
        match route(&name) {
            Route::Hle => xnw_hle::lookup(&name, &symbol_key(symbol)).map(|a| (a, Provider::Hle)),
            Route::Unknown | Route::Title => None,
            Route::Host | Route::HostFiltered => match xnw_hle::host_override(&name, &symbol_key(symbol)) {
                Some(a) => Some((a, Provider::Hle)),
                None => xnw_loader::host_export(dll, symbol).map(|a| (a, Provider::Host)),
            },
        }
    };
    let bound = bink.bind(&resolve, &|_, _| false)?;
    println!("bound      {}: {} hle, {} host, {} missing", bink.name, bound.hle, bound.host, bound.missing.len());
    for (d, s) in &bound.missing {
        println!("  missing  {d}!{s}");
    }
    // SAFETY: imports are bound; its only title-side dependency is xmem (HLE).
    unsafe { bink.attach()? };
    let f = |n: &str| bink.export(&Symbol::Name(n.into())).ok_or_else(|| format!("Bink2Scarlett.dll has no {n}"));
    // SAFETY: each export has the signature the title calls it with.
    let api = unsafe {
        Bink {
            set_os_file_callbacks: std::mem::transmute::<usize, _>(f("BinkSetOSFileCallbacks")?),
            open_with_options: std::mem::transmute::<usize, _>(f("BinkOpenWithOptions")?),
            get_frame_buffers_info: std::mem::transmute::<usize, _>(f("BinkGetFrameBuffersInfo")?),
            register_frame_buffers: std::mem::transmute::<usize, _>(f("BinkRegisterFrameBuffers")?),
            do_frame: std::mem::transmute::<usize, _>(f("BinkDoFrame")?),
            next_frame: std::mem::transmute::<usize, _>(f("BinkNextFrame")?),
            close: std::mem::transmute::<usize, _>(f("BinkClose")?),
            get_error: std::mem::transmute::<usize, _>(f("BinkGetError")?),
        }
    };
    Ok((bink, api))
}

fn bink_error(api: &Bink) -> String {
    let p = (api.get_error)();
    if p.is_null() {
        return "(no error text)".into();
    }
    // SAFETY: Bink returns a NUL-terminated string.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Our plane allocations (page-aligned, zeroed, leaked with the movie).
fn alloc_plane(bytes: usize) -> usize {
    let layout = std::alloc::Layout::from_size_align(bytes.max(1), 4096).expect("plane layout");
    // SAFETY: non-zero size.
    let p = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!p.is_null(), "out of memory for a Bink plane");
    p as usize
}

/// Width and height of plane `k` as Bink sizes them.
fn plane_dims(fb: &FrameBuffers, k: usize) -> (u32, u32) {
    if k == 1 || k == 2 { (fb.c_width, fb.c_height) } else { (fb.ya_width, fb.ya_height) }
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

pub fn run(args: &[String]) -> Res<()> {
    let rpf = PathBuf::from(args.first().filter(|a| !a.starts_with("--")).ok_or(USAGE)?);
    if let Some(pack) = arg(args, "--shaders") {
        return shaders(Path::new(pack));
    }
    let mut file = File::open(&rpf)?;
    let (header, movies) = list(&mut file)?;
    println!(
        "archive    {} ({} entries found, header says {}, flags {:#x}, toc end {:#x})",
        rpf.display(),
        movies.len(),
        header.entry_count,
        header.flags,
        header.toc_end
    );
    let total: u64 = movies.iter().map(|m| m.header.size).sum();
    println!("bink       {} movies, {:.2} GiB of Bink data, stored plain (no zlib, no nesting)", movies.len(), total as f64 / f64::from(1u32 << 30));
    if args.iter().any(|a| a == "--list") {
        for (i, m) in movies.iter().enumerate() {
            let b = &m.header;
            println!(
                "  #{i:<3} off {:#011x} size {:>10}  {} {:>5} frames {:>4}x{:<4} {}/{} fps {:>7.1} s  flags {:#08x} tracks {}",
                m.offset,
                b.size,
                String::from_utf8_lossy(&b.magic),
                b.frames,
                b.width,
                b.height,
                b.fps_num,
                b.fps_den,
                b.seconds(),
                b.flags,
                b.tracks
            );
        }
    }
    let Some(entry) = arg(args, "--entry") else {
        return Ok(());
    };
    let index: usize = entry.parse()?;
    let movie = *movies.get(index).ok_or("--entry past the last movie")?;
    let frame: u32 = arg(args, "--frame").map_or(Ok(1), str::parse)?;
    // `--audio` reads from memory: through the callbacks the track decoder
    // faults reading past a buffer of Bink's (see the module comment).
    let memory = arg(args, "--io") == Some("memory") || arg(args, "--audio").is_some();
    let title_buffers = arg(args, "--frame-buffers") != Some("bink");
    let b = movie.header;
    println!(
        "movie      #{index} at {:#x}: {} {}x{} {}/{} fps, {} frames ({:.1} s), flags {:#x}, {} audio track(s), largest frame {} B",
        movie.offset,
        String::from_utf8_lossy(&b.magic),
        b.width,
        b.height,
        b.fps_num,
        b.fps_den,
        b.frames,
        b.seconds(),
        b.flags,
        b.tracks,
        b.largest_frame
    );
    if frame == 0 || frame > b.frames {
        return Err(format!("--frame {frame}: the movie has frames 1..={}", b.frames).into());
    }
    let mut bytes = vec![0u8; b.size as usize];
    file.seek(SeekFrom::Start(movie.offset))?;
    file.read_exact(&mut bytes)?;

    let out = PathBuf::from("out");
    xnw_core::init_log(None);
    xnw_loader::crash::install();
    let dll = rpf.parent().unwrap_or(Path::new(".")).join("Bink2Scarlett.dll");
    let (_bink, api) = load_bink(&dll, &out.join("cache").join("images"))?;

    let source = Source { data: bytes.as_ptr(), len: bytes.len() as u64 };
    // The title's open flags are 0x440 | 0x4000 (title+0x442ed40), plus the
    // from-memory bit 0x04000000 when it hands over a buffer (title+0x442ea70).
    // 0x400: the caller supplies frame buffers; 0x4000: sound tracks come from
    // the options (none here, so no audio). 0x40 is kept as the title has it.
    let mut flags = 0x4000 | 0x40;
    if title_buffers {
        flags |= 0x400;
    }
    let options = [0u8; 48];
    let hbink = if memory {
        (api.open_with_options)(bytes.as_ptr().cast(), options.as_ptr(), flags | 0x0400_0000)
    } else {
        (api.set_os_file_callbacks)(cb_open, cb_read, cb_seek, cb_close);
        (api.open_with_options)((&raw const source).cast(), options.as_ptr(), flags)
    };
    if hbink.is_null() {
        return Err(format!("BinkOpenWithOptions failed: {}", bink_error(&api)).into());
    }
    // SAFETY: the BINK struct starts with these u32 fields.
    let head: Vec<u32> = (0..8).map(|i| unsafe { *hbink.add(i) }).collect();
    println!("opened     io {}, flags {flags:#x}; BINK head u32 {head:?}", if memory { "memory" } else { "callbacks" });

    let mut fb = FrameBuffers::default();
    (api.get_frame_buffers_info)(hbink, &mut fb);
    println!(
        "buffers    {} set(s); Y/A {}x{}, cR/cB {}x{} (as BinkGetFrameBuffersInfo reports)",
        fb.total_frames, fb.ya_width, fb.ya_height, fb.c_width, fb.c_height
    );
    let sets = (fb.total_frames.clamp(0, 3)) as usize;
    if title_buffers {
        for set in 0..sets {
            for k in 0..5 {
                if fb.frames[set][k].allocate != 0 {
                    let (w, h) = plane_dims(&fb, k);
                    let pitch = w.next_multiple_of(PITCH_ALIGN);
                    fb.frames[set][k].pitch = pitch;
                    fb.frames[set][k].buffer = alloc_plane(pitch as usize * h as usize);
                }
            }
        }
        (api.register_frame_buffers)(hbink, &fb);
    }
    for k in 0..5 {
        let p = fb.frames[0][k];
        let (w, h) = plane_dims(&fb, k);
        println!(
            "  plane {:<2} allocate {} {w}x{h}, pitch {}{}",
            PLANE_NAMES[k],
            p.allocate,
            p.pitch,
            if p.allocate != 0 { format!(", {} bytes", p.pitch as usize * h as usize) } else { String::new() }
        );
    }

    if let Some(wav) = arg(args, "--audio") {
        let result = audio(_bink, &api, hbink, b.frames, Path::new(wav));
        (api.close)(hbink);
        return result;
    }
    let started = std::time::Instant::now();
    for i in 1..=frame {
        if i > 1 {
            (api.next_frame)(hbink);
        }
        (api.do_frame)(hbink);
    }
    let spent = started.elapsed();
    let mut now = FrameBuffers::default();
    (api.get_frame_buffers_info)(hbink, &mut now);
    // SAFETY: as above.
    let frame_num = unsafe { *hbink.add(3) };
    println!(
        "decoded    frames 1..={frame} on the CPU in {:.1} ms ({:.2} ms/frame); BINK frame {frame_num}, current set {}",
        spent.as_secs_f64() * 1e3,
        spent.as_secs_f64() * 1e3 / f64::from(frame),
        now.frame_num
    );
    if !memory {
        println!(
            "io         {} open, {} reads ({} bytes, largest {}), {} seeks",
            OPENS.load(Ordering::Relaxed),
            READS.load(Ordering::Relaxed),
            READ_BYTES.load(Ordering::Relaxed),
            READ_MAX.load(Ordering::Relaxed),
            SEEKS.load(Ordering::Relaxed)
        );
    }
    let set = (now.frame_num as usize).min(2);
    let src = if title_buffers { &fb.frames[set] } else { &now.frames[set] };
    let mut planes: Vec<Option<PlaneView<'_>>> = Vec::new();
    for (k, p) in src.iter().enumerate() {
        let (w, h) = plane_dims(&now, k);
        if p.buffer == 0 || p.pitch == 0 {
            planes.push(None);
            continue;
        }
        // SAFETY: Bink decoded into this plane: `pitch * h` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(p.buffer as *const u8, p.pitch as usize * h as usize) };
        let v = PlaneView { bytes, width: w as usize, height: h as usize, pitch: p.pitch as usize };
        let (min, max, mean) = v.stats();
        println!("  set {set} {:<2} at {:#x} pitch {:>5}: min {min:>3} max {max:>3} mean {mean:6.1}", PLANE_NAMES[k], p.buffer, p.pitch);
        planes.push(Some(v));
    }
    let (Some(y), Some(cr), Some(cb)) = (&planes[0], &planes[1], &planes[2]) else {
        return Err("Bink returned no Y/cR/cB planes".into());
    };
    let rgb = to_rgb(y, cr, cb);
    let mean = rgb.iter().map(|&v| f64::from(v)).sum::<f64>() / rgb.len() as f64;
    let path = arg(args, "--out").map_or_else(|| out.join("movie").join(format!("movie{index}_f{frame}.bmp")), PathBuf::from);
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(&path, bmp(&rgb, y.width, y.height))?;
    println!("wrote      {} ({}x{}, mean RGB {mean:.1})", path.display(), y.width, y.height);
    (api.close)(hbink);
    Ok(())
}

/// The head of what `BinkOpenTrack` returns. The Bink 1 SDK's `BINKTRACK`
/// is {frequency, bits, channels, max size}; this Bink 2 build holds
/// {frequency, channels, size} (measured on movie #1: 48000, 2, 154624,
/// next to `BinkGetTrackType` 0x7000bb80 = 48000 Hz in the low half,
/// flags 0x7000 in the high: 16-bit 0x4000, stereo 0x2000, 0x1000; and `BinkGetTrackMaxSize` 146432).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Track {
    frequency: u32,
    channels: u32,
    size: u32,
}

/// Decodes every frame and keeps track 0's PCM (`--audio`).
fn audio(bink: &Loaded, api: &Bink, hbink: *mut u32, frames: u32, wav: &Path) -> Res<()> {
    let f = |n: &str| bink.export(&Symbol::Name(n.into())).ok_or_else(|| format!("Bink2Scarlett.dll has no {n}"));
    // SAFETY: the Bink SDK's signatures for these exports.
    let (open_track, get_track_data, close_track, max_size, track_type, track_id) = unsafe {
        (
            std::mem::transmute::<usize, extern "system" fn(*mut u32, u32) -> *const Track>(f("BinkOpenTrack")?),
            std::mem::transmute::<usize, extern "system" fn(*const Track, *mut u8) -> u32>(f("BinkGetTrackData")?),
            std::mem::transmute::<usize, extern "system" fn(*const Track)>(f("BinkCloseTrack")?),
            std::mem::transmute::<usize, extern "system" fn(*mut u32, u32) -> u32>(f("BinkGetTrackMaxSize")?),
            std::mem::transmute::<usize, extern "system" fn(*mut u32, u32) -> u32>(f("BinkGetTrackType")?),
            std::mem::transmute::<usize, extern "system" fn(*mut u32, u32) -> u32>(f("BinkGetTrackID")?),
        )
    };
    let track = open_track(hbink, 0);
    if track.is_null() {
        return Err(format!("BinkOpenTrack(0) failed: {}", bink_error(api)).into());
    }
    // SAFETY: Bink's track header.
    let t = unsafe { *track };
    println!(
        "track      0: id {}, type {:#x}, max size {} B; track head {} Hz, {} channel(s), {} B",
        track_id(hbink, 0),
        track_type(hbink, 0),
        max_size(hbink, 0),
        t.frequency,
        t.channels,
        t.size
    );
    // Rate in the low 16 bits, flags in the high: 0x4000 16-bit, 0x2000 stereo.
    let (rate, flags) = (track_type(hbink, 0) & 0xffff, track_type(hbink, 0) >> 16);
    if flags & 0x4000 == 0 || t.frequency != rate || t.channels != 1 + (flags >> 13 & 1) {
        close_track(track);
        return Err("the track is not 16-bit PCM, or its head does not match its type".into());
    }
    let mut buffer = vec![0u8; (t.size.max(max_size(hbink, 0)) as usize).max(1 << 16)];
    let mut pcm: Vec<u8> = Vec::new();
    let started = std::time::Instant::now();
    for i in 1..=frames {
        if i > 1 {
            (api.next_frame)(hbink);
        }
        let n = get_track_data(track, buffer.as_mut_ptr()) as usize;
        pcm.extend_from_slice(&buffer[..n.min(buffer.len())]);
    }
    close_track(track);
    let samples = pcm.len() / 2;
    let (peak, rms, _) = xnw_hle::pcm_level(pcm.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
    let seconds = samples as f64 / f64::from(t.channels) / f64::from(t.frequency);
    println!(
        "audio      {frames} frames in {:.1} ms: {} bytes = {seconds:.2} s of {} Hz {}-channel PCM; peak {peak:.4}, RMS {rms:.5} (full scale 1)",
        started.elapsed().as_secs_f64() * 1e3,
        pcm.len(),
        t.frequency,
        t.channels
    );
    let block = 2 * t.channels;
    let mut file = Vec::with_capacity(44 + pcm.len());
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    file.extend_from_slice(b"WAVEfmt ");
    for v in [16u32, 1 | t.channels << 16, t.frequency, t.frequency * block, block | 16 << 16] {
        file.extend_from_slice(&v.to_le_bytes());
    }
    file.extend_from_slice(b"data");
    file.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    file.extend_from_slice(&pcm);
    if let Some(d) = wav.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(wav, file)?;
    println!("wrote      {}", wav.display());
    Ok(())
}

struct PlaneView<'a> {
    bytes: &'a [u8],
    width: usize,
    height: usize,
    pitch: usize,
}

impl PlaneView<'_> {
    fn at(&self, x: usize, y: usize) -> u8 {
        self.bytes[y.min(self.height - 1) * self.pitch + x.min(self.width - 1)]
    }

    fn stats(&self) -> (u8, u8, f64) {
        let (mut min, mut max, mut sum) = (255u8, 0u8, 0u64);
        for y in 0..self.height {
            for &v in &self.bytes[y * self.pitch..y * self.pitch + self.width] {
                min = min.min(v);
                max = max.max(v);
                sum += u64::from(v);
            }
        }
        (min, max, sum as f64 / (self.width * self.height).max(1) as f64)
    }
}

/// BT.601 limited-range YCbCr to RGB, chroma point-sampled at half size.
/// Only for looking at the frame: the title's PS_Bink shader does the real
/// conversion on the GPU.
fn to_rgb(y: &PlaneView<'_>, cr: &PlaneView<'_>, cb: &PlaneView<'_>) -> Vec<u8> {
    let (sx, sy) = (y.width.div_ceil(cr.width.max(1)), y.height.div_ceil(cr.height.max(1)));
    let mut out = Vec::with_capacity(y.width * y.height * 3);
    for row in 0..y.height {
        for col in 0..y.width {
            let l = (f32::from(y.at(col, row)) - 16.0) * 1.164;
            let r_ = f32::from(cr.at(col / sx, row / sy)) - 128.0;
            let b_ = f32::from(cb.at(col / sx, row / sy)) - 128.0;
            let px = [l + 1.596 * r_, l - 0.813 * r_ - 0.392 * b_, l + 2.017 * b_];
            out.extend(px.map(|v| v.clamp(0.0, 255.0) as u8));
        }
    }
    out
}

/// A 24-bit bottom-up BMP of `rgb` (row-major, top row first).
fn bmp(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let row = (width * 3).next_multiple_of(4);
    let size = 54 + row * height;
    let mut f = Vec::with_capacity(size);
    f.extend_from_slice(b"BM");
    f.extend_from_slice(&(size as u32).to_le_bytes());
    f.extend_from_slice(&[0; 4]);
    f.extend_from_slice(&54u32.to_le_bytes());
    f.extend_from_slice(&40u32.to_le_bytes());
    f.extend_from_slice(&(width as i32).to_le_bytes());
    f.extend_from_slice(&(height as i32).to_le_bytes());
    f.extend_from_slice(&1u16.to_le_bytes());
    f.extend_from_slice(&24u16.to_le_bytes());
    f.extend_from_slice(&[0; 24]);
    for y in (0..height).rev() {
        let line = &rgb[y * width * 3..(y + 1) * width * 3];
        f.extend(line.chunks(3).flat_map(|p| [p[2], p[1], p[0]]));
        f.resize(f.len() + row - width * 3, 0);
    }
    f
}

// ---- the pack's Bink shaders through the translator -------------------------

fn shaders(pack: &Path) -> Res<()> {
    use xnw_shader::emit::{self, Fma, Options, validator::Validator};
    use xnw_shader::rpf9::{Archive, EntryKind};
    use xnw_shader::{Container, FourCC, io, ir, native_programs, resource, structure};

    let data = std::fs::read(pack)?;
    let archive = Archive::parse(&data)?;
    let dll = Validator::default_path();
    let mut validator = Validator::load(dll.as_deref()).ok();
    let mut names = Vec::new();
    for (i, e) in archive.entries.iter().enumerate() {
        if e.kind != EntryKind::Sgd5 {
            continue;
        }
        for r in xnw_shader::sgd5::records(archive.read(i).expect("entry in bounds")) {
            let Some(name) = r.name.filter(|n| n.contains("Bink")) else { continue };
            names.push(name.to_owned());
            let Some(bytes) = r.unpack() else {
                println!("{name}: corrupt record");
                continue;
            };
            let Ok(c) = Container::parse(&bytes) else {
                println!("{name}: not a DXBC container");
                continue;
            };
            let Some(dxil) = c.part(FourCC::DXIL).map(|p| p.data) else {
                println!("{name}: no DXIL part");
                continue;
            };
            let programs = match native_programs(dxil) {
                Ok(p) => p,
                Err(e) => {
                    println!("{name}: {e}");
                    continue;
                }
            };
            for p in &programs {
                let verdict = (|| -> Result<(usize, usize), String> {
                    let prog = ir::build(&p.patched_code(), p.wave64, p.vgprs).map_err(|e| format!("ir: {e}"))?;
                    let res = resource::track(&prog, &p.layout, p.stage).map_err(|e| format!("resources: {e}"))?;
                    let raised = io::raise(&prog, p).map_err(|e| format!("I/O: {e}"))?;
                    let st = structure::structure(&prog).map_err(|e| format!("structure: {e}"))?;
                    let mut sh = emit::emit_with(&c, p, &prog, &st, &raised, &res, Options { fma: Fma::Exact, ..Default::default() }).map_err(|e| format!("emit: {e}"))?;
                    if let Some(v) = validator.as_mut() {
                        v.validate(&mut sh.dxil).map_err(|t| format!("validator: {}", t.lines().next().unwrap_or("")))?;
                    }
                    Ok((prog.blocks.iter().map(|b| b.insts.len()).sum(), sh.dxil.len()))
                })();
                match verdict {
                    Ok((insts, len)) => println!(
                        "{name} {:?}: translated{} ({insts} instructions, {len} B DXIL)",
                        p.stage,
                        if validator.is_some() { " and validated" } else { "" }
                    ),
                    Err(e) => println!("{name} {:?}: FAILS {e}", p.stage),
                }
            }
        }
    }
    println!("{} Bink-named records in {}", names.len(), pack.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_bink2_header() {
        let mut h = b"KB2n".to_vec();
        for w in [3_838_204u32, 115, 90_000, 115, 2560, 1440, 30, 1, 0x40013, 1] {
            h.extend_from_slice(&w.to_le_bytes());
        }
        let b = BinkHeader::parse(&h).unwrap();
        assert_eq!((b.size, b.frames, b.width, b.height, b.flags, b.tracks), (3_838_212, 115, 2560, 1440, 0x40013, 1));
        assert!((b.seconds() - 115.0 / 30.0).abs() < 1e-9);
        assert!(BinkHeader::parse(b"BIKi").is_none());
    }

    #[test]
    fn frame_buffers_match_the_titles_layout() {
        // title+0x442f760: object+88 is the struct, plane 0 of set 0 has its
        // allocate flag at +32 and pitch at +36, sets are 80 bytes apart, and
        // the next object field is at +352.
        assert_eq!(std::mem::size_of::<FrameBuffers>(), 352 - 88);
        assert_eq!(std::mem::offset_of!(FrameBuffers, frames), 24);
        assert_eq!(std::mem::size_of::<[Plane; 5]>(), 80);
        assert_eq!(std::mem::offset_of!(Plane, allocate), 8);
    }

    #[test]
    fn grey_converts_to_grey_and_bmp_rows_are_padded() {
        let yb = vec![126u8; 4 * 2];
        let cb = vec![128u8; 2];
        let y = PlaneView { bytes: &yb, width: 3, height: 2, pitch: 4 };
        let c = PlaneView { bytes: &cb, width: 2, height: 1, pitch: 2 };
        let rgb = to_rgb(&y, &c, &c);
        assert!(rgb.iter().all(|&v| v == 128));
        let f = bmp(&rgb, 3, 2);
        assert_eq!(f.len(), 54 + 12 * 2);
        assert_eq!(&f[..2], b"BM");
    }
}
