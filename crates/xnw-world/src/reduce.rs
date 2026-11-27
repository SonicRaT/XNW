//! The reducer's switches (KILL1; `xnw shader-reduce` sets them on a burst
//! it runs as a child): one bridged pixel program drawn alone, one row of
//! it, with one thing changed, so a program the host adapter fails on is
//! cut down to the change that stops the failure.
//!
//! `XNW_REDUCE=<pixel hash, 8 hex digits>[,row=<list row>][,bind=<input>]..
//! [,mask=<target>]..[,depth=off|always][,code=kill|images|image:<hex offset>|exports|export:<n>|rets]..`
//! - `row`: of the program's rows only this list row is drawn (the other
//!   programs' rows stay);
//! - `bind` (the row's inputs, [`frame`]): `tex:<k>` the texture draw word
//!   k names, `slot:<root>.<slot>` a table slot, `root:<i>` a root view,
//!   `samplers` the sampler heap's entries; `textures` every texture,
//!   `buffers` every buffer the pixel program names. A texture becomes the
//!   grey stand-in of its dimension (a handle our table does not know), a
//!   buffer zeros of its length, the samplers the default one;
//! - `mask`: the pipelines write nothing to that target of ours
//!   ([`pipeline`]); `depth=off`: no depth test; `depth=always`: the test
//!   on, every pixel passing;
//! - `mesh=nodraw`: the rows' mesh programs dispatch, their draws are left
//!   out; `mesh=groups:<n>`: the rows' first n groups only (dispatch and
//!   draw); `mesh=scissor`: their draws under a 1x1 scissor (the raster
//!   and vertex work stay, the pixel work goes); `range=zero` (`preskinbind=on`): the bound rows' range
//!   slots read zeros of their length after the binding; `range=nomask`:
//!   our pre-skin pass hides no vertex (every item's, a diagnostic) (SKIN2);
//! - `code`: the program is translated with that feature left out
//!   (`xnw_shader::emit::Stub`); `code=rets`: with its returns as lowered,
//!   not joined in one block (KILL1's WARP fault, to show it again);
//!   `code=ms-rfl` (SKIN2): its mesh program with `v_readfirstlane_b32`
//!   reading the lane's own value (`xnw_shader::emit::first_lane_own`).
//!
//! Off (the variable unset) nothing here runs: the three call sites in
//! `bridged.rs` and the one in the burst test [`get`] first.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use xnw_shader::emit::Stub;

use super::bridged::{Frame, NAMED_CBV, NAMED_IMAGE, ORoot, OSlot, Used};

/// One input replaced by its neutral stand-in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    Tex(u32),
    Slot(u16, u32),
    Root(u16),
    Samplers,
    Textures,
    Buffers,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reduce {
    /// The pixel program: the first 32 bits of its container hash.
    pub hash: u32,
    pub row: Option<u32>,
    pub binds: Vec<Bind>,
    /// Our targets written by nothing, by bit.
    pub masks: u8,
    pub depth_off: bool,
    pub depth_always: bool,
    /// The program's returns as lowered, not joined (`Options::one_return`
    /// off): the form WARP 1.0.22 ends the device on with the depth test on.
    pub returns: bool,
    pub stub: Stub,
    /// `mesh=nodraw`: dispatch only.
    pub nodraw: bool,
    /// `mesh=scissor`: a 1x1 scissor on the draws.
    pub scissor: bool,
    /// `mesh=groups:<n>`: at most n groups.
    pub groups: Option<u32>,
    /// `range=zero`: the bound range slots read zeros.
    pub range_zero: bool,
    /// `code=ms-rfl`: the mesh program's first-lane reads are the lane's own.
    pub ms_first_lane_own: bool,
    /// `range=nomask`: our pre-skin pass hides no vertex (every item's hide mask left out).
    pub range_nomask: bool,
}

/// A handle no texture table knows (its view is the grey stand-in).
const NO_TEXTURE: u64 = 0xFFFF_FF00_0000_0000;

pub fn parse(v: &str) -> Result<Reduce, String> {
    let mut parts = v.split(',').map(str::trim).filter(|p| !p.is_empty());
    let hash = parts.next().and_then(|h| u32::from_str_radix(h.get(..8)?, 16).ok()).ok_or("XNW_REDUCE: no program hash (8 hex digits) first")?;
    let mut r = Reduce { hash, ..Reduce::default() };
    let num = |x: &str| x.parse::<u32>().map_err(|_| format!("XNW_REDUCE: {x}: not a number"));
    let hex = |x: &str| u32::from_str_radix(x.trim_start_matches("0x"), 16).map_err(|_| format!("XNW_REDUCE: {x}: not a hex offset"));
    for p in parts {
        let (k, x) = p.split_once('=').ok_or(format!("XNW_REDUCE: {p}: not key=value"))?;
        match (k, x) {
            ("row", x) => r.row = Some(num(x)?),
            ("mask", x) => r.masks |= 1 << num(x)?.min(7),
            ("depth", "off") => r.depth_off = true,
            ("depth", "always") => r.depth_always = true,
            ("code", "kill") => r.stub.kill = true,
            ("code", "images") => r.stub.images = true,
            ("code", x) if x.starts_with("image:") => r.stub.image = Some(hex(&x[6..])?),
            ("code", "rets") => r.returns = true,
            ("code", "ms-rfl") => r.ms_first_lane_own = true,
            ("code", "exports") => r.stub.exports = 0xFF,
            ("mesh", "nodraw") => r.nodraw = true,
            ("mesh", "scissor") => r.scissor = true,
            ("mesh", x) if x.starts_with("groups:") => r.groups = Some(num(&x[7..])?),
            ("range", "zero") => r.range_zero = true,
            ("range", "nomask") => r.range_nomask = true,
            ("code", x) if x.starts_with("export:") => r.stub.exports |= 1 << num(&x[7..])?.min(7),
            ("bind", "samplers") => r.binds.push(Bind::Samplers),
            ("bind", "textures") => r.binds.push(Bind::Textures),
            ("bind", "buffers") => r.binds.push(Bind::Buffers),
            ("bind", x) if x.starts_with("tex:") => r.binds.push(Bind::Tex(num(&x[4..])?)),
            ("bind", x) if x.starts_with("root:") => r.binds.push(Bind::Root(num(&x[5..])? as u16)),
            ("bind", x) if x.starts_with("slot:") => {
                let (a, b) = x[5..].split_once('.').ok_or(format!("XNW_REDUCE: {x}: slot:<root>.<slot>"))?;
                r.binds.push(Bind::Slot(num(a)? as u16, num(b)?));
            }
            _ => return Err(format!("XNW_REDUCE: {p}: not a switch")),
        }
    }
    Ok(r)
}

/// The process's reduction (`XNW_REDUCE`; None: off). A value that does
/// not parse stops the process: a reducer step must not run as the baseline.
pub fn get() -> Option<&'static Reduce> {
    static R: OnceLock<Option<Reduce>> = OnceLock::new();
    R.get_or_init(|| {
        let v = std::env::var("XNW_REDUCE").ok().filter(|v| !v.trim().is_empty())?;
        match parse(&v) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        }
    })
    .as_ref()
}

fn is(r: &Reduce, hash: u128) -> bool {
    (hash >> 96) as u32 == r.hash
}

/// `mesh=nodraw`: the row of pixel program `ps_hash` is dispatched, not drawn.
pub fn no_draw(ps_hash: u128) -> bool {
    get().is_some_and(|r| r.nodraw && is(r, ps_hash))
}

/// `code=ms-rfl`: the mesh program paired with pixel container `psc` is
/// translated with its first-lane reads the lane's own.
pub fn ms_first_lane_own(psc: &[u8]) -> bool {
    get().is_some_and(|r| r.ms_first_lane_own && is(r, xnw_core::hash128(&[psc])))
}

/// `mesh=scissor`: the row of pixel program `ps_hash` is drawn under a 1x1 scissor.
pub fn scissor(ps_hash: u128) -> bool {
    get().is_some_and(|r| r.scissor && is(r, ps_hash))
}

/// `range=zero`: the bound range slots of pixel program `ps_hash`'s rows read zeros.
pub fn range_zero(ps_hash: u128) -> bool {
    get().is_some_and(|r| r.range_zero && is(r, ps_hash))
}

/// What the pixel program alone names (table slots), and its code's sites.
static NAMED: Mutex<Option<Used>> = Mutex::new(None);

/// The stub for pixel container `psc` (`bridged::translate`): the
/// reduction's for its program, none for another. Keeps what the program
/// names for [`frame`]'s input list.
pub fn stub_for(psc: &[u8]) -> Option<Stub> {
    let r = get().filter(|r| is(r, xnw_core::hash128(&[psc])))?;
    let (mut used, mut wave) = (HashMap::new(), 32);
    if super::bridged::named(psc, &mut used, &mut wave).is_ok() {
        *NAMED.lock().unwrap_or_else(|e| e.into_inner()) = Some(used);
    }
    Some(r.stub)
}

/// Whether pixel container `psc` is translated with its returns joined
/// (`bridged::translate`): every bridged pixel program, but the reduced
/// one under `code=rets`.
pub fn one_return(psc: &[u8]) -> bool {
    !get().is_some_and(|r| r.returns && is(r, xnw_core::hash128(&[psc])))
}

/// The program's sites as the reducer's driver reads them:
/// `reduce: code <hash>: kill image:1e0 .. export:0 ..`.
pub fn sites_line(psc: &[u8], sites: &[(&'static str, u32)]) -> String {
    let s: Vec<String> = sites
        .iter()
        .map(|&(k, v)| match k {
            "kill" => "kill".to_string(),
            "image" => format!("image:{v:x}"),
            _ => format!("{k}:{v}"),
        })
        .collect();
    format!("reduce: code {:08x}: {}", (xnw_core::hash128(&[psc]) >> 96) as u32, s.join(" "))
}

/// The reduction's pipeline state over description `g`
/// (`bridged::graphics`): masked targets, no depth test. Only the reduced
/// program's pipelines (`ps_hash`: their pixel program's): before SHD1 every
/// bridged pipeline took it, so `depth=` and `mask=` changed main's rows too.
pub fn pipeline(g: &mut windows::Win32::Graphics::Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC, ps_hash: u128) {
    let Some(r) = get().filter(|r| is(r, ps_hash)) else { return };
    if r.masks != 0 {
        // Without independent blend target 0's state is every target's.
        if !g.BlendState.IndependentBlendEnable.as_bool() {
            let first = g.BlendState.RenderTarget[0];
            g.BlendState.RenderTarget = [first; 8];
            g.BlendState.IndependentBlendEnable = true.into();
        }
        for k in (0..8).filter(|k| r.masks >> k & 1 != 0) {
            g.BlendState.RenderTarget[k].RenderTargetWriteMask = 0;
        }
    }
    if r.depth_off {
        g.DepthStencilState.DepthEnable = false.into();
    }
    if r.depth_always {
        g.DepthStencilState.DepthFunc = windows::Win32::Graphics::Direct3D12::D3D12_COMPARISON_FUNC_ALWAYS;
    }
}

/// `XNW_REDUCE_DUMP=<dir>`: the reduced program's host shaders as made
/// (`<hash>.ps.dxil`, `<hash>.vs.dxil`, its mesh program's compute form
/// `<hash>.ms.dxil`; `dxc -dumpbin` reads them).
pub fn dump(psc: &[u8], ps: &[u8], vs: &[u8], ms: &[u8]) {
    let Some(r) = get().filter(|r| is(r, xnw_core::hash128(&[psc]))) else { return };
    let Some(dir) = std::env::var_os("XNW_REDUCE_DUMP").map(std::path::PathBuf::from) else { return };
    for (name, bytes) in [("ps", ps), ("vs", vs), ("ms", ms)] {
        let path = dir.join(format!("{:08x}.{name}.dxil", r.hash));
        if let Err(e) = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, bytes)) {
            eprintln!("reduce: {}: {e}", path.display());
        }
    }
}

/// With a reduction on, the process's first faults (access violations,
/// other error exceptions, C++ throws) are named
/// before the adapter turns them into a removed device: `reduce: fault:`
/// with the address, the code (its module, or none: the adapter's
/// generated code), the bytes there, the integer registers and xmm0-7.
/// The burst calls it once.
pub fn watch_faults() {
    use std::sync::atomic::{AtomicU32, Ordering};
    use xnw_core::win;
    static SEEN: AtomicU32 = AtomicU32::new(0);
    unsafe extern "system" fn on_fault(info: *mut win::ExceptionPointers) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let (r, ctx) = unsafe { (&*(*info).record, &*(*info).context) };
        // Warnings and errors (0x8..., 0xC..., and 0xE...: a C++ throw), the first few.
        if r.code < 0x8000_0000 || SEEN.fetch_add(1, Ordering::Relaxed) >= 6 {
            return win::EXCEPTION_CONTINUE_SEARCH;
        }
        let rip = ctx.rip();
        // `module+offset` of a code address (None: in no module).
        let site = |at: usize| -> Option<String> {
            let mut module: win::HMODULE = std::ptr::null_mut();
            let mut name = [0u16; 260];
            // SAFETY: a lookup by address (flags 4 | 2: from an address, no reference taken) into locals.
            let named = unsafe { win::GetModuleHandleExW(6, at as *const u16, &mut module) != 0 && win::GetModuleFileNameW(module, name.as_mut_ptr(), 260) > 0 };
            named.then(|| {
                let n = String::from_utf16_lossy(&name[..name.iter().position(|c| *c == 0).unwrap_or(0)]);
                format!("{}+{:#x}", n.rsplit(['\\', '/']).next().unwrap_or(""), at - module as usize)
            })
        };
        let whose = site(rip).unwrap_or("no module (generated code)".to_string());
        // SAFETY: the faulting instruction's bytes: readable, it was executing.
        let code: Vec<String> = (0..24).map(|k| format!("{:02x}", unsafe { *(rip as *const u8).add(k) })).collect();
        let regs: Vec<String> = win::Context::INT_NAMES.iter().enumerate().map(|(k, n)| format!("{n} {:#x}", ctx.get(win::Context::RAX + 8 * k))).collect();
        let xmm: Vec<String> = (0..8).map(|k| format!("xmm{k} {:016x}{:016x}", ctx.xmm(k, 1), ctx.xmm(k, 0))).collect();
        let what = match r.code {
            win::EXCEPTION_ACCESS_VIOLATION if r.param_count >= 2 => format!("access violation {} {:#x}", if r.params[0] == 1 { "writing" } else { "reading" }, r.params[1]),
            // A C++ throw (MSVC): params are the magic, the object, its ThrowInfo, the image base.
            // The first catchable type's decorated name and the object's first words.
            0xE06D_7363 if r.param_count >= 4 && r.params[1] != 0 && r.params[2] != 0 => {
                let (obj, info, base) = (r.params[1], r.params[2], r.params[3]);
                // SAFETY: the throw's own records, alive while it is dispatched.
                unsafe {
                    let rva = |at: usize| (at as *const i32).read_unaligned() as usize;
                    let types = base + rva(info + 12);
                    let first = base + rva(types + 4);
                    let desc = base + rva(first + 4);
                    let name = std::ffi::CStr::from_ptr((desc + 16) as *const std::ffi::c_char).to_string_lossy().into_owned();
                    let size = rva(first + 20).min(64);
                    let words: Vec<String> = (0..size / 4).map(|k| format!("{:#x}", ((obj + 4 * k) as *const u32).read_unaligned())).collect();
                    format!("a C++ throw of {name} ({size} B: {})", words.join(" "))
                }
            }
            0xE06D_7363 => "a C++ rethrow".to_string(),
            c => format!("exception {c:#x}"),
        };
        // The callers: stack words that are code addresses in a module (the first 12 past the system's own).
        let sp = ctx.get(win::Context::RSP) as usize;
        let mut callers = Vec::new();
        for k in 0..1024 {
            // SAFETY: the faulting thread's own stack above its stack pointer (8 KiB of it).
            let w = unsafe { ((sp + 8 * k) as *const usize).read_unaligned() };
            if w < 0x7ff0_0000_0000 {
                continue;
            }
            if let Some(s) = site(w).filter(|s| !["kernelbase", "ntdll", "vcruntime", "ucrtbase", "msvcrt", "msvcp"].iter().any(|m| s.to_ascii_lowercase().starts_with(m))) {
                callers.push(s);
                if callers.len() == 12 {
                    break;
                }
            }
        }
        eprintln!("reduce: fault: {what} by code at {rip:#x} in {whose}; callers on the stack: {}; code {}; {}; {}", callers.join(" "), code.join(" "), regs.join(", "), xmm.join(", "));
        win::EXCEPTION_CONTINUE_SEARCH
    }
    if get().is_some() {
        // SAFETY: the handler reads the record, the context and the faulting code only.
        unsafe { win::AddVectoredExceptionHandler(1, on_fault) };
    }
}

fn zeros(b: &Arc<Vec<u8>>) -> Arc<Vec<u8>> {
    Arc::new(vec![0; b.len()])
}

/// The reduction over a frame's bridged rows (the burst, before the pass):
/// the program's other rows leave, the kept rows' inputs are replaced.
/// Returns the lines the driver reads: the program's rows, and each kept
/// row's inputs by the names `bind=` takes (`*`: the pixel program names it).
pub fn frame(f: &mut Frame, n: u64) -> Vec<String> {
    let Some(r) = get() else { return Vec::new() };
    let named = NAMED.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default();
    apply(r, &named, f, n)
}

/// [`frame`] for reduction `r` of a program whose pixel side names `named`.
pub fn apply(r: &Reduce, named: &Used, f: &mut Frame, n: u64) -> Vec<String> {
    let mut lines = Vec::new();
    let all: Vec<u32> = f.rows.iter().filter(|x| is(r, x.1.key.1)).map(|x| x.0).collect();
    f.rows.retain(|x| !is(r, x.1.key.1) || r.row.is_none_or(|w| w == x.0));
    lines.push(format!("reduce: frame {n}: rows of {:08x}: {:?}; kept {:?}", r.hash, all, f.rows.iter().filter(|x| is(r, x.1.key.1)).map(|x| x.0).collect::<Vec<_>>()));
    for (row, o) in f.rows.iter_mut().filter(|x| is(r, x.1.key.1)) {
        let mut names = Vec::new();
        let on = |b: Bind| r.binds.contains(&b);
        let mut dims = Vec::new();
        let mut stand = |h: &mut u64, k: u64, all_dims: &[(u64, u8)]| {
            let to = NO_TEXTURE | k;
            if let Some(d) = all_dims.iter().find(|d| d.0 == *h) {
                dims.push((to, d.1));
            }
            *h = to;
        };
        let have = o.dims.clone();
        for (k, h) in &mut o.heap {
            names.push(format!("tex:{k}*"));
            if on(Bind::Tex(*k)) || on(Bind::Textures) {
                stand(h, u64::from(*k), &have);
            }
        }
        for (i, root) in o.roots.iter_mut().enumerate() {
            match root {
                ORoot::Table(v) => {
                    for (slot, s) in v.iter_mut() {
                        let ps = named.get(&(i as u16, *slot)).copied();
                        let star = if ps.is_some() { "*" } else { "" };
                        let this = on(Bind::Slot(i as u16, *slot));
                        match s {
                            OSlot::Buffer(kind, b) => {
                                names.push(format!("slot:{i}.{slot}{star}({kind:?},{}B)", b.len()));
                                if this || on(Bind::Buffers) && ps.is_some() {
                                    *b = zeros(b);
                                }
                            }
                            OSlot::Texture(h) => {
                                names.push(format!("slot:{i}.{slot}{star}(texture{})", if ps == Some(NAMED_IMAGE) || ps == Some(NAMED_CBV) || ps.is_none() { "" } else { " read as a buffer" }));
                                if this || on(Bind::Textures) {
                                    stand(h, 0x1_0000 | (i as u64) << 20 | u64::from(*slot), &have);
                                }
                            }
                        }
                    }
                }
                ORoot::View(b) => {
                    names.push(format!("root:{i}(view,{}B)", b.len()));
                    if on(Bind::Root(i as u16)) {
                        *b = zeros(b);
                    }
                }
                _ => {}
            }
        }
        o.dims.extend(dims);
        if let Some(g) = r.groups {
            o.groups = o.groups.min(g);
        }
        names.push(format!("samplers({})", o.samplers.len()));
        if on(Bind::Samplers) {
            o.samplers = Arc::new(Vec::new());
        }
        lines.push(format!("reduce: frame {n}: row {row} (groups {}, lacks {:#x}) inputs: {}", o.groups, o.lacks, names.join(" ")));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridged::Owned;
    use crate::preskin::Kind;

    #[test]
    fn switches_parse() {
        let r = parse("070aaba8,row=12,bind=tex:3,bind=slot:8.163,bind=root:2,bind=samplers,mask=1,depth=off,code=kill,code=image:41c,code=export:2").unwrap();
        assert_eq!(r.hash, 0x070a_aba8);
        assert_eq!(r.row, Some(12));
        assert_eq!(r.binds, [Bind::Tex(3), Bind::Slot(8, 163), Bind::Root(2), Bind::Samplers]);
        assert_eq!((r.masks, r.depth_off), (2, true));
        assert_eq!(r.stub, Stub { kill: true, image: Some(0x41c), images: false, exports: 4 });
        assert!(parse("row=1").is_err(), "the hash comes first");
        assert!(parse("070aaba8,code=nothing").is_err());
        let m = parse("957fc3f6,mesh=nodraw,mesh=scissor,mesh=groups:12,range=zero").unwrap();
        assert_eq!((m.nodraw, m.scissor, m.groups, m.range_zero), (true, true, Some(12), true));
    }

    /// A stood-in input keeps its shape (a buffer its length, a texture its
    /// dimension); the program's other rows leave, another program's stay.
    #[test]
    fn stand_ins_keep_the_shape() {
        let key = |h: u128| (1u128, h << 96);
        let mut o = Owned { key: key(0x070a_aba8), heap: vec![(0, 0x500), (1, 0x520)], dims: vec![(0x520, 9)], ..Owned::default() };
        o.roots = vec![ORoot::Table(vec![(5, OSlot::Buffer(Kind::Cbv, Arc::new(vec![7; 64]))), (163, OSlot::Texture(0x900))]), ORoot::View(Arc::new(vec![9; 16]))];
        let mut f = Frame::new(vec![(3, o.clone()), (4, o.clone()), (5, Owned { key: key(0x1111_2222), ..o.clone() })]);
        let r = Reduce { hash: 0x070a_aba8, row: Some(4), binds: vec![Bind::Tex(1), Bind::Slot(0, 5), Bind::Slot(0, 163)], ..Reduce::default() };
        let named: Used = [((0u16, 163u32), NAMED_IMAGE)].into_iter().collect();
        let lines = apply(&r, &named, &mut f, 7);
        assert_eq!(f.rows.iter().map(|x| x.0).collect::<Vec<_>>(), [4, 5]);
        assert_eq!(lines[0], "reduce: frame 7: rows of 070aaba8: [3, 4]; kept [4]");
        assert_eq!(lines[1], "reduce: frame 7: row 4 (groups 0, lacks 0x0) inputs: tex:0* tex:1* slot:0.5(Cbv,64B) slot:0.163*(texture) root:1(view,16B) samplers(0)");
        let o = &f.rows[0].1;
        assert_eq!(o.heap, [(0, 0x500), (1, NO_TEXTURE | 1)]);
        assert!(o.dims.contains(&(NO_TEXTURE | 1, 9)), "the stand-in has the texture's dimension");
        let ORoot::Table(v) = &o.roots[0] else { panic!() };
        assert!(matches!(&v[0].1, OSlot::Buffer(Kind::Cbv, b) if b.len() == 64 && b.iter().all(|x| *x == 0)));
        assert!(matches!(v[1].1, OSlot::Texture(h) if h & NO_TEXTURE == NO_TEXTURE));
        // The other program's row is as it was.
        assert_eq!(f.rows[1].1.heap, [(0, 0x500), (1, 0x520)]);
    }
}
