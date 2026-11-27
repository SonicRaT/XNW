//! Where the title's frame goes outside the render stages (ENG4):
//! `XNW_TITLE_TIME=1` times the steps of the title's main loop
//! (`title+0x812620`) and of its render thread at their call sites,
//! and prints one `title time:` line per 60 frames. Off by default.
//!
//! Most of these functions start with a `call` or a rip-relative load, so
//! their entries cannot be moved to a trampoline (`detours.rs`). Each is
//! reached through one `call rel32`: that displacement is pointed at a
//! 14-byte jump of ours within 2 GiB, whose hook calls the title's
//! function at its own address with the four register arguments (none of
//! them takes more, or a float). Per row: the calling thread's CPU time and
//! the wall time, both inclusive (`update` holds its children, `frame end`
//! holds `gpu pacing`). Wall far above CPU is a wait: `kick+wait` is the
//! game thread waiting for the render thread, `gpu pacing` the render
//! thread waiting on its frame token. Job-pool work shows in wall only.
//!
//! `XNW_TITLE_TIME=2` also splits the two update groups (ENG5). `fsm1` and
//! `fsm2` are `title+0x54267a0(skeleton title+0x7127458, group)`: it runs
//! the group's list of update elements `{+0 vtable, +8 name hash, +0x10
//! next, +0x18 function or child list}` (vtable slot 1 runs an element:
//! `title+0x5ea6bf0` jumps to the function, `title+0x5ea6c00` runs a child
//! list; registered by `title+0x589f120` and `title+0x589f200`). With 2 the
//! hook walks the list itself, in the title's order, and times each element:
//! one `title update N:` line per 60 frames, by wall time, each element as
//! `title+function name-hash cpu/wall`.
//!
//! `XNW_TITLE_TIME=3` (ENG7) is 1 plus the calls of stage 36's sky body
//! `title+0xd3ac70` (callback `title+0xd3ac60`: 3.2 of the stage's 3.3 ms in
//! free8): its 16 sub-passes are rows of their own on a `title stage 36
//! sky:` line, named by the function called (`545bfa0` is the sky setup
//! from the time cycle). The once-only exposure seed `title+0x28f9850`
//! takes a float and is not a row.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use xnw_core::win::{FlushInstructionCache, GetCurrentProcess, MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualProtect};

/// The title's executable 1.0.2.1 (PE header timestamp).
const BUILD_1_0_2_1: u32 = 0x6AF7_83D4;

/// A metered call: `title+site` is `call rel32` to `title+target`.
struct Site {
    name: &'static str,
    site: usize,
    target: usize,
}

/// Game thread: the main loop's steps and the game update's children, in
/// call order; then the render thread's. Sites and targets read from the exe.
const SITES: [Site; 31] = [
    Site { name: "timers", site: 0x81267b, target: 0x2246930 },
    Site { name: "update", site: 0x812696, target: 0x21b6ca0 },
    Site { name: "update.fsm1", site: 0x21b6cd0, target: 0x54267a0 },
    Site { name: "update.world", site: 0x21b6cd5, target: 0x28bc4a0 },
    Site { name: "update.fsm2", site: 0x21b6ce9, target: 0x54267a0 },
    Site { name: "update.7b5a80", site: 0x21b6d08, target: 0x7b5a80 },
    Site { name: "update.6c5340", site: 0x21b6d1b, target: 0x6c5340 },
    Site { name: "update.2622120", site: 0x21b6d20, target: 0x2622120 },
    Site { name: "update.b3e560", site: 0x21b6d25, target: 0xb3e560 },
    Site { name: "update.2e7d530", site: 0x21b6d4c, target: 0x2e7d530 },
    Site { name: "pump", site: 0x81269b, target: 0x23cb230 },
    Site { name: "kick+wait", site: 0x8126a2, target: 0x23cf6a0 },
    Site { name: "render.drain", site: 0x23cf4ed, target: 0x43c38a0 },
    // The frame end `title+0xc9e5e0` is reached by a jump each frame (`title+0xc9e33e`; its
    // one call site is the render loop's teardown), so the row is its first call, the drain.
    Site { name: "render.frame end", site: 0xc9e5fe, target: 0x24846d0 },
    Site { name: "render.gpu pacing", site: 0x24848c0, target: 0x2474740 },
    // Stage 36's sky body `title+0xd3ac70`, in call order (rows from `SKY_FIRST` on; `XNW_TITLE_TIME=3` only).
    Site { name: "1025a0", site: 0xd3ada7, target: 0x1025a0 },
    Site { name: "3c9760", site: 0xd3adae, target: 0x3c9760 },
    Site { name: "4e7f410", site: 0xd3adca, target: 0x4e7f410 },
    Site { name: "545bfa0", site: 0xd3adfb, target: 0x545bfa0 },
    Site { name: "337c5d0", site: 0xd3ae00, target: 0x337c5d0 },
    Site { name: "35d6320", site: 0xd3ae14, target: 0x35d6320 },
    Site { name: "1217ca0", site: 0xd3ae21, target: 0x1217ca0 },
    Site { name: "1d26950", site: 0xd3ae26, target: 0x1d26950 },
    Site { name: "25e4f60", site: 0xd3ae2b, target: 0x25e4f60 },
    Site { name: "39ea440", site: 0xd3ae30, target: 0x39ea440 },
    Site { name: "3382da0", site: 0xd3ae35, target: 0x3382da0 },
    Site { name: "cfd9e0", site: 0xd3ae3a, target: 0xcfd9e0 },
    Site { name: "11707d0", site: 0xd3ae3f, target: 0x11707d0 },
    Site { name: "11f840", site: 0xd3ae44, target: 0x11f840 },
    Site { name: "b5c790", site: 0xd3ae49, target: 0xb5c790 },
    Site { name: "32af080", site: 0xd3ae4e, target: 0x32af080 },
];
const N: usize = SITES.len();
/// The first row of stage 36's sky body; the rows before it are the `title time:` line.
const SKY_FIRST: usize = 15;
/// Whether the sky rows are metered.
static SKY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// `jmp qword [rcx+18h]`; `push rsi; push rdi; sub rsp, 28h; mov rsi, [rcx+18h]`.
const ELEMENT_RUN_BYTES: [u8; 4] = [0x48, 0xFF, 0x61, 0x18];
const GROUP_RUN_BYTES: [u8; 10] = [0x56, 0x57, 0x48, 0x83, 0xEC, 0x28, 0x48, 0x8B, 0x71, 0x18];
/// The row called once per frame of the game thread.
const FRAME_ROW: usize = 1;
/// The rows the frame wall line takes (WALL1, `xnw_core::framewall`): row, its slot.
const WALL_ROWS: [(usize, usize); 4] = [(1, xnw_core::framewall::UPDATE), (11, xnw_core::framewall::KICK_WAIT), (12, xnw_core::framewall::DRAIN), (13, xnw_core::framewall::FRAME_END)];
/// Only [`WALL_ROWS`] are metered and no `title time:` line is printed (no `XNW_TITLE_TIME`, a screen pass).
static QUIET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A metered call of row `i` ended: its wall (and the drain's CPU) into the frame wall line.
fn wall_row(i: usize, cycles: u64, wall_ns: u64) {
    let Some(&(_, slot)) = WALL_ROWS.iter().find(|r| r.0 == i) else { return };
    xnw_core::framewall::add(slot, wall_ns);
    if slot == xnw_core::framewall::DRAIN {
        xnw_core::framewall::add(xnw_core::framewall::DRAIN_CPU, (cycles as f64 / crate::stagemeter::cycles_per_second() * 1e9) as u64);
    }
}

/// Whether the frame wall rows are metered without `XNW_TITLE_TIME`: a
/// screen pass is asked for and `XNW_FRAME_WALL` is not `off`.
fn wall_wanted(world_render: &str, frame_wall: &str) -> bool {
    world_render.split(',').any(|p| p.trim().starts_with("screen")) && !matches!(frame_wall.trim(), "off" | "0")
}
const EVERY: u64 = 60;

static TARGETS: [AtomicUsize; N] = [const { AtomicUsize::new(0) }; N];
static CYCLES: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static WALL_NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// The update skeleton's walker and the two element kinds (vtable slot 1).
const WALK_RVA: usize = 0x54267a0;
const ELEMENT_RUN_RVA: usize = 0x5ea6bf0;
const GROUP_RUN_RVA: usize = 0x5ea6c00;
/// The skeleton's list of groups `{+0 id, +8 first element, +0x10 next}`.
const SKELETON_GROUPS: usize = 0x140;
/// Elements timed apart per group; the last slot takes the rest.
const ELEMENTS: usize = 40;
/// `XNW_TITLE_TIME=2`: the title's base, else 0.
static SPLIT_BASE: AtomicUsize = AtomicUsize::new(0);
/// Per group (1, 2) and element: function << 32 | name hash (0 = free), cycles, wall.
static EL_KEY: [[AtomicU64; ELEMENTS]; 2] = [const { [const { AtomicU64::new(0) }; ELEMENTS] }; 2];
static EL_CYCLES: [[AtomicU64; ELEMENTS]; 2] = [const { [const { AtomicU64::new(0) }; ELEMENTS] }; 2];
static EL_WALL_NS: [[AtomicU64; ELEMENTS]; 2] = [const { [const { AtomicU64::new(0) }; ELEMENTS] }; 2];

/// Runs one element as the title does (vtable slot 1) and adds its time to its slot.
///
/// # Safety
/// `element` is a live update element of the title; `base` is the title.
unsafe fn run_element(g: usize, element: usize, base: usize) {
    // SAFETY: the element's vtable, name hash and function, as the title's walker reads them.
    let (run, simple, hash, function) = unsafe {
        let run = ((element as *const *const usize).read()).add(1).read();
        (run, run == base + ELEMENT_RUN_RVA, ((element + 8) as *const u32).read(), ((element + 0x18) as *const usize).read())
    };
    let what = if simple { function } else { run };
    let key = ((what.wrapping_sub(base) as u64) << 32) | u64::from(hash);
    let slot = (0..ELEMENTS - 1)
        .find(|&i| match EL_KEY[g][i].compare_exchange(0, key, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => true,
            Err(k) => k == key,
        })
        .unwrap_or(ELEMENTS - 1);
    // SAFETY: slot 1 of the element's own vtable, called with the element.
    let f: extern "system" fn(usize) -> usize = unsafe { std::mem::transmute(run) };
    let (c0, t0) = (crate::stagemeter::thread_cycles(), Instant::now());
    f(element);
    EL_CYCLES[g][slot].fetch_add(crate::stagemeter::thread_cycles().wrapping_sub(c0), Ordering::Relaxed);
    EL_WALL_NS[g][slot].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// The title's walk of one update group (`title+0x54267a0`), each element
/// timed; a child list's elements are timed one by one.
///
/// # Safety
/// `skeleton` is the title's update skeleton; `base` is the title.
unsafe fn walk(skeleton: usize, group: u32, base: usize) {
    let g = usize::from(group != 1);
    // SAFETY: the lists the title's walker reads, in its order.
    unsafe {
        let mut node = ((skeleton + SKELETON_GROUPS) as *const usize).read();
        while node != 0 && (node as *const u32).read() != group {
            node = ((node + 0x10) as *const usize).read();
        }
        if node == 0 {
            return;
        }
        let mut e = ((node + 8) as *const usize).read();
        while e != 0 {
            if ((e as *const *const usize).read()).add(1).read() == base + GROUP_RUN_RVA {
                let mut child = ((e + 0x18) as *const usize).read();
                while child != 0 {
                    run_element(g, child, base);
                    child = ((child + 0x10) as *const usize).read();
                }
            } else {
                run_element(g, e, base);
            }
            e = ((e + 0x10) as *const usize).read();
        }
    }
}

/// The line of one update group for the last `frames` frames; the sums start again.
fn take_update_line(g: usize, frames: u64) -> String {
    let (hz, f) = (crate::stagemeter::cycles_per_second(), frames.max(1) as f64);
    let mut rows: Vec<(u64, u64, u64)> = (0..ELEMENTS).map(|i| (EL_WALL_NS[g][i].swap(0, Ordering::Relaxed), EL_CYCLES[g][i].swap(0, Ordering::Relaxed), EL_KEY[g][i].load(Ordering::Relaxed))).filter(|r| r.2 != 0 || r.0 != 0).collect();
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.2.cmp(&b.2)));
    let all: u64 = rows.iter().map(|r| r.0).sum();
    let named: Vec<String> = rows.iter().take(12).map(|&(w, c, key)| format!("title+{:#x} {:#010x} {:.2}/{:.2}", key >> 32, key as u32, c as f64 / hz * 1000.0 / f, w as f64 / 1e6 / f)).collect();
    format!("title update {}: {frames} frames, {} elements, ms a frame cpu/wall by element (function, name hash), longest first: {}; all {:.2} wall", g + 1, rows.len(), named.join(", "), all as f64 / 1e6 / f)
}

fn run(i: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    // SAFETY: the title's function this call site named, set at install.
    let f: extern "system" fn(usize, usize, usize, usize) -> usize = unsafe { std::mem::transmute(TARGETS[i].load(Ordering::Relaxed)) };
    let split = SPLIT_BASE.load(Ordering::Relaxed);
    let (c0, t0) = (crate::stagemeter::thread_cycles(), Instant::now());
    let r = if split != 0 && SITES[i].target == WALK_RVA {
        // SAFETY: the walker's own arguments: the skeleton and the group.
        unsafe { walk(a, b as u32, split) };
        0
    } else {
        f(a, b, c, d)
    };
    let (cycles, wall_ns) = (crate::stagemeter::thread_cycles().wrapping_sub(c0), t0.elapsed().as_nanos() as u64);
    wall_row(i, cycles, wall_ns);
    if QUIET.load(Ordering::Relaxed) {
        return r;
    }
    CYCLES[i].fetch_add(cycles, Ordering::Relaxed);
    WALL_NS[i].fetch_add(wall_ns, Ordering::Relaxed);
    CALLS[i].fetch_add(1, Ordering::Relaxed);
    if i == FRAME_ROW && FRAMES.fetch_add(1, Ordering::Relaxed) % EVERY == EVERY - 1 {
        xnw_core::info!("{}", take_line(EVERY));
        if SKY.load(Ordering::Relaxed) {
            xnw_core::info!("{}", take_sky_line(EVERY));
        }
        if split != 0 {
            xnw_core::info!("{}", take_update_line(0, EVERY));
            xnw_core::info!("{}", take_update_line(1, EVERY));
        }
    }
    r
}

/// One hook per row: each knows its row.
macro_rules! hooks {
    ($($i:literal)*) => {
        [$({
            extern "system" fn h(a: usize, b: usize, c: usize, d: usize) -> usize {
                run($i, a, b, c, d)
            }
            h as usize
        }),*]
    };
}

fn hook_table() -> [usize; N] {
    hooks!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
}

/// The line for the last `frames` frames; the sums start again.
fn take_line(frames: u64) -> String {
    let (hz, f) = (crate::stagemeter::cycles_per_second(), frames.max(1) as f64);
    let rows: Vec<String> = (0..SKY_FIRST)
        .map(|i| {
            let (c, w, n) = (CYCLES[i].swap(0, Ordering::Relaxed), WALL_NS[i].swap(0, Ordering::Relaxed), CALLS[i].swap(0, Ordering::Relaxed));
            format!("{} {:.2}/{:.2} ({:.1})", SITES[i].name, c as f64 / hz * 1000.0 / f, w as f64 / 1e6 / f, n as f64 / f)
        })
        .collect();
    format!("title time: {frames} frames, ms a frame cpu/wall (calls a frame): {}", rows.join(", "))
}

/// The sky rows for the last `frames` frames, in call order, and their sum; the sums start again.
fn take_sky_line(frames: u64) -> String {
    let (hz, f) = (crate::stagemeter::cycles_per_second(), frames.max(1) as f64);
    let (mut all, mut rows) = (0.0, Vec::new());
    for i in SKY_FIRST..N {
        let (c, w, n) = (CYCLES[i].swap(0, Ordering::Relaxed), WALL_NS[i].swap(0, Ordering::Relaxed), CALLS[i].swap(0, Ordering::Relaxed));
        all += c as f64 / hz * 1000.0 / f;
        rows.push(format!("title+0x{} {:.2}/{:.2} ({:.1})", SITES[i].name, c as f64 / hz * 1000.0 / f, w as f64 / 1e6 / f, n as f64 / f));
    }
    format!("title stage 36 sky: {frames} frames, ms a frame cpu/wall (calls a frame) by function called, in call order: {}; all {all:.2} cpu", rows.join(", "))
}

/// `jmp qword [rip+0]` followed by the target.
fn jump(to: usize) -> [u8; 14] {
    let mut j = [0xFF, 0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    j[6..].copy_from_slice(&(to as u64).to_le_bytes());
    j
}

/// A fresh executable page within `rel32` reach of every address in `[lo, hi)`.
pub(crate) fn near_page(lo: usize, hi: usize) -> Option<*mut u8> {
    let mut at = (hi + 0xFFFF) & !0xFFFF;
    while at + 0x1000 - lo < 0x7000_0000 {
        // SAFETY: asks for one page at a hinted address; null when taken.
        let p = unsafe { VirtualAlloc(at as *const _, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        if !p.is_null() {
            return Some(p);
        }
        at += 0x1_0000;
    }
    None
}

/// Points the `call rel32` at `site`, which must call `target`, at a jump
/// to `hook` written at `thunk` (14 bytes, within 2 GiB of the site).
///
/// # Safety
/// `site` is 5 bytes of code nothing runs now; `thunk` is 14 writable executable bytes.
pub(crate) unsafe fn redirect(site: *mut u8, target: usize, thunk: *mut u8, hook: usize) -> Result<(), &'static str> {
    // SAFETY: the caller's code bytes.
    let (op, rel) = unsafe { (site.read(), site.add(1).cast::<i32>().read_unaligned()) };
    if op != 0xE8 || (site as usize + 5).wrapping_add(rel as isize as usize) != target {
        return Err("not the expected call");
    }
    let Ok(new) = i32::try_from(thunk as isize - (site as isize + 5)) else { return Err("the jump is out of reach") };
    let mut old = 0;
    // SAFETY: the thunk is ours; the displacement's 4 bytes are made writable and restored.
    unsafe {
        std::ptr::copy_nonoverlapping(jump(hook).as_ptr(), thunk, 14);
        if VirtualProtect(site.add(1).cast(), 4, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return Err("VirtualProtect failed");
        }
        site.add(1).cast::<i32>().write_unaligned(new);
        VirtualProtect(site.add(1).cast(), 4, old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), site.cast(), 5);
    }
    Ok(())
}

/// Installs the meter when `XNW_TITLE_TIME=1`; without it, the four frame
/// wall rows alone when a screen pass is asked for (WALL1, [`wall_wanted`]).
///
/// # Safety
/// `base` is the mapped title and none of its code runs yet.
pub unsafe fn install(base: usize) {
    let (split, sky) = match std::env::var("XNW_TITLE_TIME").unwrap_or_default().trim() {
        "" | "0" => {
            if !wall_wanted(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default(), &std::env::var("XNW_FRAME_WALL").unwrap_or_default()) {
                return;
            }
            QUIET.store(true, Ordering::Relaxed);
            (false, false)
        }
        "1" => (false, false),
        "2" => (true, false),
        "3" => (false, true),
        other => {
            xnw_core::warn!("XNW_TITLE_TIME={other}: not 0, 1, 2 or 3; the title meter stays off");
            return;
        }
    };
    // SAFETY: forwarded; SizeOfImage is at the optional header's +0x38.
    let (stamp, size) = unsafe {
        let nt = (base as *const u8).add(0x3c).cast::<u32>().read_unaligned() as usize;
        (crate::patches::timestamp(base), (base as *const u8).add(nt + 0x50).cast::<u32>().read_unaligned() as usize)
    };
    if stamp != BUILD_1_0_2_1 {
        xnw_core::warn!("title time: title timestamp {stamp:#x} is not build 1.0.2.1; not metered");
        return;
    }
    let Some(page) = near_page(base, base + size) else {
        xnw_core::warn!("title time: no page within 2 GiB of the title; not metered");
        return;
    };
    crate::stagemeter::cycles_per_second();
    let (hooks, mut done) = (hook_table(), 0);
    let quiet = QUIET.load(Ordering::Relaxed);
    let rows = if sky { N } else { SKY_FIRST };
    for (i, s) in SITES.iter().enumerate().take(rows) {
        if quiet && !WALL_ROWS.iter().any(|r| r.0 == i) {
            continue;
        }
        TARGETS[i].store(base + s.target, Ordering::Relaxed);
        // SAFETY: inside the mapped .text of this build, no code running; 16 bytes of our page per row.
        match unsafe { redirect((base + s.site) as *mut u8, base + s.target, page.add(16 * i), hooks[i]) } {
            Ok(()) => done += 1,
            Err(why) => xnw_core::warn!("title time: {} at title+{:#x} skipped: {why}", s.name, s.site),
        }
    }
    // The element kinds are told apart by these two functions: `jmp [rcx+18h]` and the child-list loop.
    // SAFETY: inside the mapped .text of this build.
    let kinds = unsafe { std::slice::from_raw_parts((base + ELEMENT_RUN_RVA) as *const u8, 4) == ELEMENT_RUN_BYTES && std::slice::from_raw_parts((base + GROUP_RUN_RVA) as *const u8, 10) == GROUP_RUN_BYTES };
    if split && kinds {
        SPLIT_BASE.store(base, Ordering::Relaxed);
        xnw_core::info!("title time: the update groups are walked here, element by element (title+{WALK_RVA:#x}); one 'title update N:' line per {EVERY} frames each");
    } else if split {
        xnw_core::warn!("title time: the update elements' code is not the expected one; the groups are not split");
    }
    if sky {
        SKY.store(true, Ordering::Relaxed);
        xnw_core::info!("title time: stage 36's sky body title+0xd3ac70 is split by call; one 'title stage 36 sky:' line per {EVERY} frames");
    }
    if quiet {
        xnw_core::info!("title time: {done} of {} call sites metered for the 'frame wall:' line only (update, kick+wait, render.drain, render.frame end; XNW_FRAME_WALL=off: none)", WALL_ROWS.len());
        return;
    }
    xnw_core::info!("title time: {done} of {rows} call sites metered (main loop title+0x812620, render thread); one line per {EVERY} frames");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every site is a `call rel32` to its target in the executable.
    #[test]
    fn the_sites_call_their_targets_in_the_executable() {
        let Some(exe) = crate::patches::Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): the title meter's sites were not checked against it");
            return;
        };
        for s in &SITES {
            assert_eq!((exe.at(s.site, 1)[0], exe.rel32_target(s.site)), (0xE8, s.target), "{} at title+{:#x}", s.name, s.site);
        }
        // The sky rows are the calls of the stage-36 sky body, whose thunk is the stage's callback; the
        // once-only exposure seed before them gets its float in xmm0 (vmovss xmm0, [rax+rcx+0x228]) and is no row.
        assert_eq!((SITES[SKY_FIRST].name, SITES.len() - SKY_FIRST), ("1025a0", 16));
        assert!(SITES[SKY_FIRST..].iter().all(|s| (0xd3ac70..0xd3ae62).contains(&s.site) && s.name == format!("{:x}", s.target)));
        assert!(SITES[SKY_FIRST..].windows(2).all(|w| w[0].site < w[1].site), "call order");
        assert_eq!((exe.at(0xd3ac60, 1)[0], exe.rel32_target(0xd3ac60)), (0xE9, 0xd3ac70));
        assert_eq!(exe.at(0xd3ad1f, 9), [0xC5, 0xFA, 0x10, 0x84, 0x08, 0x28, 0x02, 0x00, 0x00]);
        assert_eq!((exe.at(0xd3ad28, 1)[0], exe.rel32_target(0xd3ad28)), (0xE8, 0x28f9850));
        assert_eq!(exe.at(ELEMENT_RUN_RVA, 4), ELEMENT_RUN_BYTES);
        assert_eq!(exe.at(GROUP_RUN_RVA, 10), GROUP_RUN_BYTES);
        // The walker reads the skeleton's groups at +0x140: mov rax, [rcx+140h].
        assert_eq!(exe.at(WALK_RVA + 9, 7), [0x48, 0x8B, 0x81, 0x40, 0x01, 0x00, 0x00]);
        // The frame end is entered by a jump each frame, not by the render loop's call.
        assert_eq!((exe.at(0xc9e33e, 1)[0], exe.rel32_target(0xc9e33e)), (0xE9, 0xc9e5e0));
    }

    /// The frame wall rows are the four it names; they are metered alone only for a screen pass.
    #[test]
    fn the_frame_wall_rows_are_the_named_sites() {
        let names: Vec<&str> = WALL_ROWS.iter().map(|r| SITES[r.0].name).collect();
        assert_eq!(names, ["update", "kick+wait", "render.drain", "render.frame end"]);
        assert_eq!(WALL_ROWS[0].0, FRAME_ROW);
        assert!(wall_wanted("screen,drawbones=on", ""));
        assert!(wall_wanted("gbuf=on, screen-hw", " on "));
        assert!(!wall_wanted("screen", "off") && !wall_wanted("capture", "") && !wall_wanted("", ""));
    }

    static RAN: AtomicU64 = AtomicU64::new(0);
    extern "system" fn first() -> usize {
        RAN.fetch_add(1, Ordering::Relaxed);
        0
    }
    extern "system" fn second() -> usize {
        RAN.fetch_add(0x100, Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(3));
        0
    }
    /// The simple element's slot 1 (`jmp [rcx+18h]` in the title).
    extern "system" fn element_run(e: usize) -> usize {
        // SAFETY: a test element: its function at +0x18.
        let f: extern "system" fn() -> usize = unsafe { std::mem::transmute(((e + 0x18) as *const usize).read()) };
        f()
    }
    extern "system" fn group_run(_: usize) -> usize {
        panic!("a child list is walked here, not by its own function");
    }

    /// The sky rows print on their own line, in call order, and stay off the `title time:` line.
    #[test]
    fn the_sky_rows_have_their_own_line() {
        let row = SITES.iter().position(|s| s.name == "545bfa0").unwrap();
        assert!(row >= SKY_FIRST);
        CALLS[row].store(2, Ordering::Relaxed);
        WALL_NS[row].store(3_000_000, Ordering::Relaxed);
        let line = take_sky_line(2);
        assert!(line.starts_with("title stage 36 sky: 2 frames, ms a frame cpu/wall (calls a frame) by function called, in call order: title+0x1025a0 0.00/0.00 (0.0), "), "{line}");
        assert!(line.contains("title+0x545bfa0 0.00/1.50 (1.0)") && line.ends_with("; all 0.00 cpu"), "{line}");
        assert_eq!(CALLS[row].load(Ordering::Relaxed), 0, "the sums start again");
        assert_eq!(hook_table().len(), SITES.len());
    }

    /// A skeleton made in memory as the title's (`title+0x54267a0` reads
    /// it): group 2 holds an element and a child list of two; the walk
    /// runs each once, in order, and the line names them by function and hash.
    #[test]
    fn the_update_walk_runs_every_element_once_and_times_each() {
        // The walk tells a child list by slot 1 == base + GROUP_RUN_RVA; any other slot 1 is called.
        let base = (group_run as usize).wrapping_sub(GROUP_RUN_RVA);
        let simple_table: [usize; 2] = [0, element_run as usize];
        let group_table: [usize; 2] = [0, base + GROUP_RUN_RVA];
        // element: {vtable, hash, next, function or child}
        let c2: [usize; 4] = [simple_table.as_ptr() as usize, 0x2222, 0, second as usize];
        let c1: [usize; 4] = [simple_table.as_ptr() as usize, 0x1111, c2.as_ptr() as usize, first as usize];
        let list: [usize; 4] = [group_table.as_ptr() as usize, 0x9999, 0, c1.as_ptr() as usize];
        let e1: [usize; 4] = [simple_table.as_ptr() as usize, 0x3333, list.as_ptr() as usize, first as usize];
        // groups: {id, first element, next}
        let g2: [usize; 3] = [2, e1.as_ptr() as usize, 0];
        let g1: [usize; 3] = [1, 0, g2.as_ptr() as usize];
        let mut skeleton = [0usize; 0x30];
        skeleton[SKELETON_GROUPS / 8] = g1.as_ptr() as usize;
        // SAFETY: the lists above.
        unsafe {
            walk(skeleton.as_ptr() as usize, 2, base);
            walk(skeleton.as_ptr() as usize, 1, base);
            walk(skeleton.as_ptr() as usize, 7, base);
        }
        assert_eq!(RAN.swap(0, Ordering::Relaxed), 0x102, "first twice (the element and the child), second once");
        let line = take_update_line(1, 1);
        assert!(line.starts_with("title update 2: 1 frames, 3 elements, ms a frame cpu/wall by element (function, name hash), longest first: title+"), "{line}");
        assert!(line.contains(" 0x00002222 ") && !line.contains("0x00009999"), "{line}");
        let slow = line.find("0x00002222").unwrap();
        assert!(line[slow..].contains("0x00003333") || line[slow..].contains("0x00001111"), "the sleeping element is first: {line}");
        assert!(take_update_line(0, 1).starts_with("title update 1: 1 frames, 0 elements"));
    }

    /// A call site made in memory, redirected as at install: the function
    /// still gets its arguments and gives its result, and its row counts it.
    #[test]
    fn a_redirected_call_runs_the_function_and_counts_it() {
        // SAFETY: a fresh executable allocation of ours.
        let code = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.cast::<u8>();
        assert!(!code.is_null());
        // caller at +0: sub rsp, 28h; call +0x100; add rsp, 28h; ret. target at +0x100: lea rax, [rcx+rdx]; ret.
        let caller: [u8; 14] = [0x48, 0x83, 0xEC, 0x28, 0xE8, 0xF7, 0x00, 0x00, 0x00, 0x48, 0x83, 0xC4, 0x28, 0xC3];
        let target: [u8; 5] = [0x48, 0x8D, 0x04, 0x11, 0xC3];
        // SAFETY: inside the allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(caller.as_ptr(), code, caller.len());
            std::ptr::copy_nonoverlapping(target.as_ptr(), code.add(0x100), target.len());
        }
        // SAFETY: the function just written.
        let f: extern "system" fn(usize, usize) -> usize = unsafe { std::mem::transmute(code) };
        assert_eq!(f(2, 3), 5);
        let row = SKY_FIRST - 1;
        TARGETS[row].store(code as usize + 0x100, Ordering::Relaxed);
        let page = near_page(code as usize, code as usize + 4096).expect("a page near our code");
        // SAFETY: our code, nothing runs it now; the page is ours.
        unsafe {
            assert_eq!(redirect(code.add(4), code as usize + 0x200, page, hook_table()[row]), Err("not the expected call"));
            redirect(code.add(4), code as usize + 0x100, page, hook_table()[row]).unwrap();
        }
        assert_eq!(f(2, 3), 5, "through the hook: the same result");
        assert_eq!(CALLS[row].load(Ordering::Relaxed), 1);
        let line = take_line(1);
        assert!(line.starts_with("title time: 1 frames, ms a frame cpu/wall (calls a frame): timers 0.00/0.00 (0.0), update "), "{line}");
        assert!(line.ends_with("(1.0)"), "the last row was called once: {line}");
    }
}
