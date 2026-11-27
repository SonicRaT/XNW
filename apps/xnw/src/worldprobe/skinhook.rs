//! Draw-time bones for the live pass, behind
//! `XNW_WORLD_RENDER=screen,...,drawbones=on` (default off).
//!
//! Today the pass takes every skinned draw's bones from one copy of the
//! title's palette ring made at the END of the frame (`skin_copy_live`). By
//! then the ring slots of many props hold other bones: in one capture
//! (two frames) all 60 skipped non-ped props are identity at the draw while
//! 6 / 19 of them are sheared or displaced in the end-of-frame copy.
//! This copies each skinned draw's bones at the hook instead:
//! the TLS palette pointer (TLS+0x11E8) and count (TLS+0x11E4), as the frame
//! pack's hook palettes do, once per (pointer, count) per thread capture,
//! into one float array the list carries as its own palette marked draw-time
//! (`FrameList::palette_at_draw`). A read that cannot be made is a reason
//! counted, never a fault (the reader is the fault-fixed copy).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use xnw_world::{FrameList, NO_PALETTE, SkinWindow, skin_window};

/// Bones of one draw, at most (the skin constant's curCount bound).
pub const DRAW_BONES_MAX: u32 = 4096;
/// Bones copied per thread capture, at most (6 MiB of floats).
pub const FRAME_BONES_MAX: usize = 1 << 17;

/// Why a draw got no draw-time bones.
pub const WHY_NO_POINTER: usize = 0;
pub const WHY_COUNT: usize = 1;
pub const WHY_UNREADABLE: usize = 2;
pub const WHY_FRAME_CAP: usize = 3;
pub const WHY_NAMES: [&str; 4] = ["no pointer", "count 0 or over 4096", "unreadable", "frame cap"];

/// `drawbones=on` (read once).
pub fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| switch(&v)));
    *ON
}

fn switch(v: &str) -> bool {
    v.split(',').map(str::trim).filter_map(|p| p.strip_prefix("drawbones=")).last() == Some("on")
}

/// One capture's draw-time bones: 12 floats per bone; per (pointer, count)
/// the first bone of its copy. `masks`: the vehicle mask words.
#[derive(Default)]
pub struct HookBones {
    pub floats: Vec<f32>,
    at: HashMap<(u64, u32), u32>,
    /// Draws given bones (a repeat of a span counts too) and refusals by WHY_*.
    pub given: u32,
    pub why: [u32; 4],
    pub masks: VehMasks,
}

/// `skinrules` bit 5 (VEH4): the MSR one-bone parts (doors, bonnet, boot,
/// panels: one bone per vertex or per mesh, no weights) are given index and
/// weight streams at the geometry read (`geomread::read`, the replay's
/// `vehicle_skin` bit 0x20), so they are posed by their bone; without it
/// they draw in the bind pose (a closed door inside the open one).
pub const VEH_ONE_BONE: u32 = 0x20;
/// Bit 5 since the start: one-bone geometries given streams, and ones left
/// in the bind pose because their vertex buffer is drawn in place.
pub static ONE_BONE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// `skinrules` bit 6: `b += rec+0x54` where the vehicle programs do it
/// (under `rec+0x74 != -1`).
pub const VEH_OFFSET: u32 = 0x40;
/// `skinrules` bit 7: the vehicle programs' mask|rank compaction of the bone
/// index through the arena words at `rec+0x74`.
pub const VEH_MASK: u32 = 0x80;

/// The `skinrules=` value of `XNW_WORLD_RENDER` (read once; 0: none), all
/// bits: the pass keeps bits 0-7 (`parse_skin_rules`), the recorder's bit 8
/// lies above them.
pub fn rules() -> u32 {
    static R: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| parse_rules(&std::env::var("XNW_WORLD_RENDER").unwrap_or_default()));
    *R
}

fn parse_rules(v: &str) -> u32 {
    v.split(',').filter_map(|p| p.trim().strip_prefix("skinrules=")).filter_map(|h| u32::from_str_radix(h.trim_start_matches("0x"), 16).ok()).last().unwrap_or(0) & 0x1FF
}

/// Mask words copied per vehicle draw (256 B, as the frame pack's arena
/// record): bones 0..1024; the program reads word `b >> 4 & 0xFF`, words
/// past the copy read as 0 (the hidden entry).
pub const MASK_WORDS: usize = 64;
/// An arena offset at or past this is refused (the arena is a per-frame
/// heap; 256 MiB is far above what it holds).
pub const ARENA_SPAN: u32 = 1 << 28;
/// Why a vehicle draw got no mask words.
pub const MASK_WHY_NAMES: [&str; 3] = ["no arena base", "offset implausible", "unreadable"];

/// One capture's vehicle mask words, `MASK_WORDS` per distinct arena
/// address (a repeat is not read again).
#[derive(Default)]
pub struct VehMasks {
    pub words: Vec<u32>,
    at: HashMap<u64, u32>,
    pub given: u32,
    pub why: [u32; 3],
    /// Time in `copy` on the title's threads (ns), and the reads made.
    pub ns: u64,
    pub reads: u32,
}

impl VehMasks {
    /// The words at `arena + off` (`arena` the frame's arena base, `off` the
    /// draw record's +0x74). `read(at, dst)`: fill `dst`, false when
    /// unreadable. Some(index of the first word in `words`).
    pub fn copy(&mut self, arena: u64, off: u32, read: &mut dyn FnMut(u64, &mut [u32]) -> bool) -> Option<u32> {
        let t0 = std::time::Instant::now();
        let got = self.copy_inner(arena, off, read);
        self.ns += t0.elapsed().as_nanos() as u64;
        got
    }

    fn copy_inner(&mut self, arena: u64, off: u32, read: &mut dyn FnMut(u64, &mut [u32]) -> bool) -> Option<u32> {
        let w = if arena == 0 {
            0
        } else if off >= ARENA_SPAN || off % 4 != 0 {
            1
        } else {
            let at = arena + u64::from(off);
            if let Some(&i) = self.at.get(&at) {
                self.given += 1;
                return Some(i);
            }
            let i = self.words.len();
            self.words.resize(i + MASK_WORDS, 0);
            self.reads += 1;
            if read(at, &mut self.words[i..]) {
                self.at.insert(at, i as u32);
                self.given += 1;
                return Some(i as u32);
            }
            self.words.truncate(i);
            2
        };
        self.why[w] += 1;
        None
    }

    /// Appends another thread's words; returns the word shift for its draws.
    pub fn merge(&mut self, o: VehMasks) -> u32 {
        let shift = self.words.len() as u32;
        self.words.extend(o.words);
        self.given += o.given;
        self.ns += o.ns;
        self.reads += o.reads;
        for (a, b) in self.why.iter_mut().zip(o.why) {
            *a += b;
        }
        shift
    }
}

/// The index table a vehicle draw's bones go through (the pass's remap:
/// `b = table[b] & 0xFFF`, relative to the draw's first bone): `b + off`
/// under `VEH_OFFSET`, then the mask|rank compaction under `VEH_MASK`
/// (`xnw_frame::audit::mask_rank`), else `& 0xFFF`. 4096 entries.
#[cfg(test)]
pub fn vehicle_table(words: &[u32], off: u32, rules: u32) -> Vec<f32> {
    (0..4096u32)
        .map(|b| {
            let b = if rules & VEH_OFFSET != 0 { b.wrapping_add(off) } else { b };
            f32::from_bits(if rules & VEH_MASK != 0 { xnw_frame::audit::mask_rank(b, words) } else { b & 0xFFF })
        })
        .collect()
}

/// The vehicle-rules log line (every 60 frames when bit 6 or 7 is on):
/// `rows` list rows drawn through a vehicle table; the mask copy's time on
/// the title's threads from `m`, the tables' from `t`.
pub fn vehicle_line(frame: u64, rules: u32, rows: u32, m: &VehMasks, t: &VehTables) -> String {
    let why = MASK_WHY_NAMES.iter().zip(m.why).map(|(n, v)| format!("{n} {v}")).collect::<Vec<_>>().join(", ");
    format!(
        "world render: frame {frame}: vehicle rules {rules:#x}: {rows} rows drawn by rule ({} tables); mask words {} for {} draws; refused: {why}; cost: mask copy {:.3} ms on the title's threads ({} reads), tables {:.3} ms in the list build ({} KiB)",
        t.made,
        m.words.len(),
        m.given,
        m.ns as f64 / 1e6,
        m.reads,
        t.ns as f64 / 1e6,
        t.kib
    ) + &if rules & VEH_ONE_BONE != 0 {
        format!("; one-bone parts posed {} (in place, not posed {}) since the start", ONE_BONE[0].load(Ordering::Relaxed), ONE_BONE[1].load(Ordering::Relaxed))
    } else {
        String::new()
    }
}

/// Entries of a vehicle index table (the pass reads `table[b]`, b < 4096).
pub const TABLE_LEN: usize = 4096;

/// The index tables of one list build. A table is `t[b] = f((b + off) &
/// 0xFFF)` (`f` the compaction over the words, or the identity without
/// `VEH_MASK`), so every offset of one word set reads the same array from
/// `off & 0xFFF` on: one array of 2 x 4096 entries (f twice) per (palette,
/// word set) serves all its offsets, the draw's remap pointing `off & 0xFFF`
/// into it. Before, each (word set, offset) had its own 4096 entries made
/// by 4096 compactions.
#[derive(Default)]
pub struct VehTables {
    at: HashMap<u32, HashMap<Vec<u32>, u32>>,
    /// Arrays made, remaps given, ns spent, KiB appended.
    pub made: u32,
    pub given: u32,
    pub ns: u64,
    pub kib: u32,
}

impl VehTables {
    /// The remap (index into `pal`, palette `palette` of the list) for a
    /// draw with `words` and `off` under `rules`; reads from it equal
    /// `vehicle_table(words, off, rules)`.
    pub fn remap(&mut self, palette: u32, pal: &mut Vec<f32>, words: &[u32], off: u32, rules: u32) -> u32 {
        let t0 = std::time::Instant::now();
        let words = if rules & VEH_MASK != 0 { words } else { &[] };
        let start = match self.at.get(&palette).and_then(|m| m.get(words)) {
            Some(&a) => a,
            None => {
                let a = pal.len() as u32;
                let f: Vec<f32> = if rules & VEH_MASK == 0 {
                    (0..TABLE_LEN as u32).map(f32::from_bits).collect()
                } else {
                    // Only a set mask bit has a non-zero entry (the zero entry is 0.0's bits).
                    let mut r = vec![0f32; TABLE_LEN];
                    for (k, w) in words.iter().take(TABLE_LEN / 16).enumerate() {
                        for bit in (0..16).filter(|bit| w >> bit & 1 != 0) {
                            let b = 16 * k as u32 + bit;
                            r[b as usize] = f32::from_bits(xnw_frame::audit::mask_rank(b, words));
                        }
                    }
                    r
                };
                pal.extend_from_slice(&f);
                pal.extend_from_slice(&f);
                self.at.entry(palette).or_default().insert(words.to_vec(), a);
                self.made += 1;
                self.kib += (8 * TABLE_LEN / 1024) as u32;
                a
            }
        };
        self.given += 1;
        self.ns += t0.elapsed().as_nanos() as u64;
        start + if rules & VEH_OFFSET != 0 { off & 0xFFF } else { 0 }
    }
}

impl HookBones {
    /// The draw's bones at the hook: `ptr` the TLS palette pointer, `n` its
    /// count. `read(at, dst)`: fill `dst` (12 floats per bone) from `at`,
    /// false when unreadable. Some((first bone, bones)) in `floats`.
    pub fn copy(&mut self, ptr: u64, n: u32, read: &mut dyn FnMut(u64, &mut [f32]) -> bool) -> Option<(u32, u32)> {
        let refuse = |s: &mut HookBones, w: usize| {
            s.why[w] += 1;
            None
        };
        if ptr == 0 {
            return refuse(self, WHY_NO_POINTER);
        }
        if n == 0 || n > DRAW_BONES_MAX {
            return refuse(self, WHY_COUNT);
        }
        if let Some(&f) = self.at.get(&(ptr, n)) {
            self.given += 1;
            return Some((f, n));
        }
        let first = self.floats.len() / 12;
        if first + n as usize > FRAME_BONES_MAX {
            return refuse(self, WHY_FRAME_CAP);
        }
        self.floats.resize(12 * (first + n as usize), 0.0);
        if !read(ptr, &mut self.floats[12 * first..]) {
            self.floats.truncate(12 * first);
            return refuse(self, WHY_UNREADABLE);
        }
        self.at.insert((ptr, n), first as u32);
        self.given += 1;
        Some((first as u32, n))
    }

    /// Appends another thread's capture; returns the bone shift and the mask
    /// word shift for its draws.
    pub fn merge(&mut self, o: HookBones) -> (u32, u32) {
        let shift = (self.floats.len() / 12) as u32;
        self.floats.extend(o.floats);
        self.given += o.given;
        for (a, b) in self.why.iter_mut().zip(o.why) {
            *a += b;
        }
        (shift, self.masks.merge(o.masks))
    }

    pub fn bones(&self) -> usize {
        self.floats.len() / 12
    }
}

/// The live list's skinned records by bone source: [draw-time, of them
/// identity windows, end-of-frame, of them identity]; `hooked` skinned draws
/// the hook saw minus both sources are the ones left without bones.
pub fn sources(list: &FrameList) -> [u32; 4] {
    let mut n = [0u32; 4];
    for r in list.draws.iter().filter(|r| r.palette != NO_PALETTE) {
        let k = if list.palette_at_draw.get(r.palette as usize).copied().unwrap_or(false) { 0 } else { 2 };
        n[k] += 1;
        n[k + 1] += u32::from(skin_window(list, r) == SkinWindow::Identity);
    }
    n
}

/// The per-frame log line (every 60 frames with `drawbones=on`).
pub fn line(frame: u64, list: &FrameList, hooked: u32, hb: &HookBones) -> String {
    let s = sources(list);
    let none = hooked.saturating_sub(s[0] + s[2]);
    let why = WHY_NAMES.iter().zip(hb.why).map(|(n, v)| format!("{n} {v}")).collect::<Vec<_>>().join(", ");
    format!(
        "world render: frame {frame}: skin sources: draw-time {} ({} identity), end-of-frame {} ({} identity), no bones {none} of {hooked} hooked; draw-time copy {} bones ({} KiB) for {} draws; refused: {why}",
        s[0],
        s[1],
        s[2],
        s[3],
        hb.bones(),
        hb.bones() * 48 / 1024,
        hb.given
    )
}

/// `skinrules` bit 8 in the live pass: an instanced wheel or tyre draw
/// (a program of `xnw_frame::audit::INST_PROGRAMS`) is read at the list
/// sink (the draw's tables are bound only there) and its row drawn once
/// per instance with the bones at its palette base, as the replay's
/// `vehicle_instances` does.
pub const VEH_INST: u32 = 0x100;
pub use xnw_frame::audit::{WHEEL_BONES, WheelInst, wheel_reads};

/// Why an instanced draw was not read (counted, never a fault).
pub const WHEEL_WHY_NAMES: [&str; 6] = ["slot views missing", "no instances", "bases differ", "mask on", "unreadable", "no hook draw"];
pub const WHEEL_NO_HOOK: usize = 5;

/// Counters since the last line: sinks seen, rows given instances, refusals
/// by WHEEL_WHY_NAMES, ns at the sink.
pub static WHEEL_N: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
thread_local! {
    /// Pipeline object -> its instanced program's slots (None: not one).
    static WHEEL_PIPES: std::cell::RefCell<HashMap<usize, Option<xnw_frame::audit::InstSlots>>> = std::cell::RefCell::new(HashMap::new());
}

fn wheel_slots(pipeline: usize) -> Option<xnw_frame::audit::InstSlots> {
    WHEEL_PIPES.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(s) = m.get(&pipeline) {
            return *s;
        }
        if m.len() > 4096 {
            m.clear();
        }
        let s = xnw_gpu::d3d12x::tap::stages(pipeline)
            .iter()
            .find(|s| s.name == "mesh")
            .and_then(|s| xnw_frame::audit::INST_PROGRAMS.iter().find(|t| t.0 == (s.hash >> 96) as u32))
            .map(|t| t.1);
        m.insert(pipeline, s);
        s
    })
}

/// The draw tap's filter: an instanced vehicle program.
pub fn wheel_want(pipeline: usize) -> bool {
    wheel_slots(pipeline).is_some()
}

/// The draw tap (title thread, at the list sink, after the draw's hook):
/// reads the instances and bones and gives them to this thread's last
/// hooked draw.
pub fn on_wheel_sink(d: &xnw_gpu::d3d12x::tap::TapDraw) {
    use xnw_gpu::d3d12x::tap::{self, TapRoot};
    let t0 = std::time::Instant::now();
    let Some(slots) = wheel_slots(d.pipeline) else { return };
    let _site = super::fault_site(super::SITE_WHEELS);
    WHEEL_N[0].fetch_add(1, Ordering::Relaxed);
    let view = |root: u16, slot: u32| match d.roots.get(usize::from(root)) {
        Some(TapRoot::Table { handle, slots, .. }) if slot < *slots => Some(tap::view(handle + tap::SLOT * u64::from(slot))).filter(|v| v.kind != xnw_formats::framepack::view::NONE && v.addr != 0).map(|v| (v.addr, v.bytes)),
        _ => None,
    };
    let mut read = |at: u64, len: usize| -> Option<Vec<u8>> {
        // A read of the title's memory: plausible bounds, then the fault-fixed copy.
        ((0x1_0000..0x8000_0000_0000).contains(&at) && at.checked_add(len as u64).is_some_and(|e| e <= 0x8000_0000_0000)).then(|| super::dcopy(at, len)).flatten()
    };
    let why = match wheel_reads(slots, d.count, &view, &mut read) {
        Ok(w) => super::attach_wheel(w).then_some(1).unwrap_or(2 + WHEEL_NO_HOOK),
        Err(k) => 2 + k,
    };
    WHEEL_N[why].fetch_add(1, Ordering::Relaxed);
    WHEEL_N[8].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// `instrec` (default on, `instrec=off` in `XNW_WORLD_RENDER`): a draw
/// of a static instanced program (`xnw_frame::instrec::STATIC_INST`: the
/// `*_inst` effects) is read at the list sink and its row drawn once per
/// instance record, as the replay's `record_instances` does. The hook's
/// world of such a draw is the identity at the world origin.
pub fn instrec_start() -> bool {
    !std::env::var("XNW_WORLD_RENDER").is_ok_and(|v| v.split(',').any(|p| p.trim() == "instrec=off"))
}

/// Counters since the last line: sinks seen, rows given their worlds,
/// refusals by `xnw_frame::instrec::WHY`, no hook draw.
pub static INSTREC_N: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
thread_local! {
    static INSTREC_PIPES: std::cell::RefCell<HashMap<usize, Option<xnw_frame::audit::InstSlots>>> = std::cell::RefCell::new(HashMap::new());
}

fn instrec_slots(pipeline: usize) -> Option<xnw_frame::audit::InstSlots> {
    INSTREC_PIPES.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(s) = m.get(&pipeline) {
            return *s;
        }
        if m.len() > 4096 {
            m.clear();
        }
        let s = xnw_gpu::d3d12x::tap::stages(pipeline).iter().find(|s| s.name == "mesh").and_then(|s| xnw_frame::instrec::slots_of(s.hash));
        m.insert(pipeline, s);
        s
    })
}

/// The draw tap's filter: a static instanced program.
pub fn instrec_want(pipeline: usize) -> bool {
    instrec_slots(pipeline).is_some()
}

/// The draw tap (title thread, at the list sink, after the draw's hook):
/// the instance worlds for this thread's last hooked draw.
pub fn on_instrec_sink(d: &xnw_gpu::d3d12x::tap::TapDraw) {
    use xnw_gpu::d3d12x::tap::{self, TapRoot};
    let Some(slots) = instrec_slots(d.pipeline) else { return };
    if !xnw_core::live::on(xnw_core::live::INSTREC, || u32::from(instrec_start())) {
        return;
    }
    let _site = super::fault_site(super::SITE_WHEELS);
    INSTREC_N[0].fetch_add(1, Ordering::Relaxed);
    let view = |root: u16, slot: u32| match d.roots.get(usize::from(root)) {
        Some(TapRoot::Table { handle, slots, .. }) if slot < *slots => Some(tap::view(handle + tap::SLOT * u64::from(slot))).filter(|v| v.kind != xnw_formats::framepack::view::NONE && v.addr != 0).map(|v| (v.addr, v.bytes)),
        _ => None,
    };
    let mut read = |at: u64, len: usize| -> Option<Vec<u8>> { ((0x1_0000..0x8000_0000_0000).contains(&at) && at.checked_add(len as u64).is_some_and(|e| e <= 0x8000_0000_0000)).then(|| super::dcopy(at, len)).flatten() };
    // A direct mesh draw gives its groups; an indirect one (its count is the command limit) every record of the view.
    let groups = if d.op == xnw_formats::framepack::op::MESH { d.count } else { 0 };
    let why = match xnw_frame::instrec::static_reads(slots, groups, &view, &mut read) {
        Ok(w) => if super::attach_instrec(w) { 1 } else { 5 },
        Err(k) => 2 + k,
    };
    INSTREC_N[why].fetch_add(1, Ordering::Relaxed);
}

/// The line (every 60 frames): counters since the last line (taken) and this frame's rows placed.
pub fn instrec_line(frame: u64, placed: u32) -> String {
    let n: Vec<u64> = INSTREC_N.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    let why = xnw_frame::instrec::WHY.iter().zip(&n[2..5]).map(|(w, v)| format!("{w} {v}")).collect::<Vec<_>>().join(", ");
    format!("world render: frame {frame}: instance records (60 frames): static instanced sinks {}, given to their draw {}; refused: {why}, no hook draw {}; this frame {placed} rows drawn per record", n[0], n[1], n[5])
}

/// The wheels line (every 60 frames with bit 8): counters since the last
/// line (taken), and `placed` rows of this frame drawn per instance; `left`:
/// this frame's rows with reads, and of them left as they were (a remap
/// already, no palette or instances, palette too far).
pub fn wheel_line(frame: u64, placed: u32, left: [u32; 4]) -> String {
    let n: Vec<u64> = WHEEL_N.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    let why = WHEEL_WHY_NAMES.iter().zip(&n[2..8]).map(|(w, v)| format!("{w} {v}")).collect::<Vec<_>>().join(", ");
    format!("world render: frame {frame}: vehicle wheels (60 frames): instanced sinks {}, given to their draw {}; refused: {why}; {:.3} ms at the sink; this frame {placed} rows drawn per instance (rows with reads {}; left: remap already {}, no palette or instances {}, palette too far {})", n[0], n[1], n[8] as f64 / 1e6, left[0], left[1], left[2], left[3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use xnw_world::{Albedo, DrawRecord, GeomKey, Material, NO_REMAP};

    const ID: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    /// A fake title memory: bones at 0x1000 (bone k's floats all k), unreadable elsewhere.
    fn mem(at: u64, dst: &mut [f32]) -> bool {
        if !(0x1000..0x1000 + 48 * 64).contains(&at) || at + 4 * dst.len() as u64 > 0x1000 + 48 * 64 {
            return false;
        }
        for (i, v) in dst.iter_mut().enumerate() {
            *v = ((at - 0x1000) / 48 + i as u64 / 12) as f32;
        }
        true
    }

    #[test]
    fn copy_and_bounds() {
        let mut hb = HookBones::default();
        let mut reads = 0;
        let mut rd = |at: u64, d: &mut [f32]| {
            reads += 1;
            mem(at, d)
        };
        assert_eq!(hb.copy(0x1000 + 48 * 2, 3, &mut rd), Some((0, 3)));
        assert_eq!(&hb.floats[..], &[[2.0; 12], [3.0; 12], [4.0; 12]].concat()[..]);
        // The same span again: no second read.
        assert_eq!(hb.copy(0x1000 + 48 * 2, 3, &mut rd), Some((0, 3)));
        assert_eq!(hb.copy(0x1000, 1, &mut rd), Some((3, 1)));
        // Reasons, never a fault; nothing kept from a failed read.
        assert_eq!(hb.copy(0, 3, &mut rd), None);
        assert_eq!(hb.copy(0x1000, 0, &mut rd), None);
        assert_eq!(hb.copy(0x1000, DRAW_BONES_MAX + 1, &mut rd), None);
        assert_eq!(hb.copy(0x1000 + 48 * 60, 8, &mut rd), None);
        assert_eq!(hb.copy(0x9000, 1, &mut rd), None);
        assert_eq!(reads, 4);
        assert_eq!(hb.bones(), 4);
        assert_eq!((hb.given, hb.why), (3, [1, 2, 2, 0]));
        // The frame cap.
        let mut big = HookBones { floats: vec![0.0; 12 * (FRAME_BONES_MAX - 2)], ..Default::default() };
        assert_eq!(big.copy(0x1000, 3, &mut |a, d| mem(a, d)), None);
        assert_eq!(big.why[WHY_FRAME_CAP], 1);
        assert_eq!(big.copy(0x1000, 2, &mut |a, d| mem(a, d)), Some(((FRAME_BONES_MAX - 2) as u32, 2)));
        // Merge: the second capture's bones follow the first's.
        let mut a = HookBones::default();
        a.copy(0x1000, 2, &mut |x, d| mem(x, d));
        assert_eq!(a.merge(hb), (2, 0));
        assert_eq!((a.bones(), a.given, a.why), (6, 4, [1, 2, 2, 0]));
    }

    /// Made-up arena: words at 0x8000 + 4k are k, readable 0x8000..0x8400.
    fn arena(at: u64, dst: &mut [u32]) -> bool {
        if at < 0x8000 || at + 4 * dst.len() as u64 > 0x8400 {
            return false;
        }
        for (i, w) in dst.iter_mut().enumerate() {
            *w = ((at - 0x8000) / 4) as u32 + i as u32;
        }
        true
    }

    #[test]
    fn vehicle_masks_and_table() {
        let mut m = VehMasks::default();
        let mut reads = 0;
        let mut rd = |at: u64, d: &mut [u32]| {
            reads += 1;
            arena(at, d)
        };
        assert_eq!(m.copy(0x8000, 0x10, &mut rd), Some(0));
        assert_eq!(&m.words[..3], &[4, 5, 6]);
        assert_eq!(m.copy(0x8000, 0x10, &mut rd), Some(0), "a repeat: no second read");
        assert_eq!(m.copy(0x8000, 0x100, &mut rd), Some(MASK_WORDS as u32));
        // Reasons, never a fault; nothing kept from a failed read.
        assert_eq!(m.copy(0, 0x10, &mut rd), None);
        assert_eq!(m.copy(0x8000, ARENA_SPAN, &mut rd), None);
        assert_eq!(m.copy(0x8000, 0x11, &mut rd), None);
        assert_eq!(m.copy(0x8000, 0x3F0, &mut rd), None, "runs past the readable arena");
        assert_eq!((reads, m.words.len(), m.given, m.why), (3, 2 * MASK_WORDS, 3, [1, 2, 1]));
        let mut o = VehMasks::default();
        o.copy(0x8000, 0, &mut |a, d| arena(a, d));
        assert_eq!(m.merge(o), 2 * MASK_WORDS as u32);
        assert_eq!((m.words.len(), m.given, m.why), (3 * MASK_WORDS, 4, [1, 2, 1]));
        // The table: words 2a17|0, ff90|7 (a car's first words); offset 0: bone 4 is the 4th visible,
        // bone 3 hidden (entry 0), bone 20 the 8th; offset 16 moves bone 4 to word 1.
        let w = [0x2a17u32, 0xff90 | 7 << 16];
        let t = vehicle_table(&w, 0, VEH_OFFSET | VEH_MASK);
        assert_eq!((t[4].to_bits(), t[3].to_bits(), t[20].to_bits(), t.len()), (4, 0, 8, 4096));
        assert_eq!(vehicle_table(&w, 16, VEH_OFFSET | VEH_MASK)[4].to_bits(), 8);
        assert_eq!(vehicle_table(&w, 16, VEH_MASK)[4].to_bits(), 4, "offset off");
        assert_eq!(vehicle_table(&w, 16, VEH_OFFSET)[4095].to_bits(), 15, "offset only: (b + off) & 0xFFF");
        assert_eq!(m.reads, 4, "three reads here, one in the merged capture");
        m.ns = 1_500_000;
        let mut tb = VehTables::default();
        tb.remap(0, &mut Vec::new(), &w, 16, VEH_OFFSET | VEH_MASK);
        tb.ns = 250_000;
        assert_eq!(
            vehicle_line(120, 0xcb, 7, &m, &tb),
            format!("world render: frame 120: vehicle rules 0xcb: 7 rows drawn by rule (1 tables); mask words {} for 4 draws; refused: no arena base 1, offset implausible 2, unreadable 1; cost: mask copy 1.500 ms on the title's threads (4 reads), tables 0.250 ms in the list build (32 KiB)", 3 * MASK_WORDS)
        );
    }

    /// The copy's clock: a read that takes 2 ms is counted as at least 2 ms,
    /// a repeat (no read) adds next to nothing; a merge adds both captures.
    #[test]
    fn mask_copy_is_timed() {
        let mut m = VehMasks::default();
        let mut slow = |at: u64, d: &mut [u32]| {
            std::thread::sleep(std::time::Duration::from_millis(2));
            arena(at, d)
        };
        m.copy(0x8000, 0x10, &mut slow);
        assert!(m.ns >= 2_000_000, "{} ns", m.ns);
        let once = m.ns;
        m.copy(0x8000, 0x10, &mut slow);
        assert!(m.ns - once < 1_000_000, "a repeat reads nothing: {} ns", m.ns - once);
        let mut o = VehMasks::default();
        o.copy(0x8000, 0x20, &mut slow);
        let (mine, other) = (m.ns, o.ns);
        m.merge(o);
        assert_eq!((m.ns, m.reads), (mine + other, 2));
        assert!(m.ns >= 4_000_000);
    }

    /// Reads through a remap equal `vehicle_table` for every rule set and
    /// offsets that wrap; one array per (palette, word set), every offset
    /// of it sharing the array.
    #[test]
    fn remaps_read_as_tables() {
        let w: Vec<u32> = (0..MASK_WORDS as u32).map(|k| (0x2a17 ^ (k * 0x9e37)) & 0xFFFF | (k * 5) << 16).collect();
        for (n, rules) in [VEH_OFFSET | VEH_MASK, VEH_MASK, VEH_OFFSET].into_iter().enumerate() {
            let mut t = VehTables::default();
            let mut pal = vec![7.0f32; 12];
            for off in [0u32, 16, 1000, 4095, 4096, 4097, 0xFFFF_FFF0] {
                let at = t.remap(3, &mut pal, &w, off, rules) as usize;
                let want = vehicle_table(&w, off, rules);
                assert!(pal.len() >= at + TABLE_LEN, "rules {rules:#x} off {off}: the pass reads 4096 entries");
                assert!(pal[at..at + TABLE_LEN].iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "rules {rules:#x} off {off} (case {n})");
            }
            assert_eq!((t.made, t.given, pal.len()), (1, 7, 12 + 2 * TABLE_LEN), "rules {rules:#x}");
            // Another word set, or the same words on another palette: a second array.
            t.remap(3, &mut pal, &w[..2], 3, rules);
            t.remap(4, &mut pal, &w, 3, rules);
            assert_eq!(t.made, if rules & VEH_MASK != 0 { 3 } else { 2 }, "rules {rules:#x}: without the mask the words do not matter");
            assert!(t.ns > 0);
        }
    }

    /// The cost at a test run's frame-840 shape (197 vehicle draws over 17
    /// arena word sets, 51 (word set, offset) pairs): the mask copy as the
    /// hook does it, the tables as before (4096 compactions per pair) and as
    /// `VehTables` makes them. Best of 20 rounds.
    #[test]
    #[ignore]
    fn cost_vehicle_rules_shape() {
        let arena_buf: Vec<u32> = (0..64 * 64).map(|i| (i as u32 * 0x9e37) & 0xFFFF | (i as u32 & 63) << 16).collect();
        let base = arena_buf.as_ptr() as u64;
        let (mut copy_ns, mut old_ns, mut new_ns, mut timer_ns) = (u128::MAX, u128::MAX, u64::MAX, u64::MAX);
        let (mut old_kib, mut new_kib, mut pairs, mut reads) = (0, 0, 0, 0);
        for _ in 0..20 {
            let mut m = VehMasks::default();
            let mut draws = Vec::new();
            let t0 = std::time::Instant::now();
            for i in 0..197u32 {
                let off = 256 * (i % 17);
                // SAFETY: inside `arena_buf`.
                let w = m.copy(base, off, &mut |at, dst| unsafe { super::super::fcopy(dst.as_mut_ptr().cast(), at as *const u8, 4 * dst.len()) });
                draws.push((w.unwrap(), 16 * (i % 48 / 17 + 3 * (i % 17))));
            }
            copy_ns = copy_ns.min(t0.elapsed().as_nanos());
            timer_ns = timer_ns.min(m.ns);
            reads = m.reads;
            let mut keys: Vec<(u32, u32)> = draws.clone();
            keys.sort_unstable();
            keys.dedup();
            pairs = keys.len();
            let mut old = vec![0f32; 12 * 4000];
            let t0 = std::time::Instant::now();
            for &(w, off) in &keys {
                old.extend(vehicle_table(&m.words[w as usize..w as usize + MASK_WORDS], off, VEH_OFFSET | VEH_MASK));
            }
            old_ns = old_ns.min(t0.elapsed().as_nanos());
            old_kib = (old.len() - 12 * 4000) * 4 / 1024;
            let mut t = VehTables::default();
            let mut pal = vec![0f32; 12 * 4000];
            for &(w, off) in &draws {
                let at = t.remap(0, &mut pal, &m.words[w as usize..w as usize + MASK_WORDS], off, VEH_OFFSET | VEH_MASK) as usize;
                assert_eq!(pal[at + 5].to_bits(), vehicle_table(&m.words[w as usize..w as usize + MASK_WORDS], off, VEH_OFFSET | VEH_MASK)[5].to_bits());
            }
            new_ns = new_ns.min(t.ns);
            new_kib = t.kib;
        }
        println!(
            "cost: 197 draws, {reads} reads, mask copy {:.3} ms (its counter {:.3} ms); tables for {pairs} pairs: before {:.3} ms ({old_kib} KiB), now {:.3} ms ({new_kib} KiB)",
            copy_ns as f64 / 1e6,
            timer_ns as f64 / 1e6,
            old_ns as f64 / 1e6,
            new_ns as f64 / 1e6
        );
    }

    /// The cost at a captured frame's shape (1,784 skinned rows over 30 distinct
    /// spans, 17,499 bones read per draw, 1,171 distinct): every draw through
    /// the fault-fixed copy, as the hook does. Prints ns per draw.
    #[test]
    #[ignore]
    fn cost_pk2_shape() {
        let ring: Vec<f32> = (0..12 * 4096).map(|i| i as f32).collect();
        let base = ring.as_ptr() as u64;
        let spans: Vec<(u64, u32)> = (0..30u64).map(|k| (base + 48 * 120 * k, 39)).collect();
        let t0 = std::time::Instant::now();
        let mut hb = HookBones::default();
        for i in 0..1784 {
            let (p, n) = spans[i % 30];
            // SAFETY: inside `ring`.
            hb.copy(p, n, &mut |at, dst| unsafe { super::super::fcopy(dst.as_mut_ptr().cast(), at as *const u8, 4 * dst.len()) });
        }
        let ns = t0.elapsed().as_nanos() as f64 / 1784.0;
        println!("cost: {ns:.0} ns per skinned draw, {} bones ({} KiB) copied for {} draws", hb.bones(), hb.bones() * 48 / 1024, hb.given);
        assert_eq!(hb.given, 1784);
        // Worst case: no repeat, every draw its own 10-bone span (17,840 bones).
        let ring: Vec<f32> = vec![1.0; 12 * 20_000];
        let base = ring.as_ptr() as u64;
        let t0 = std::time::Instant::now();
        let mut hb = HookBones::default();
        for i in 0..1784u64 {
            // SAFETY: inside `ring`.
            hb.copy(base + 480 * i, 10, &mut |at, dst| unsafe { super::super::fcopy(dst.as_mut_ptr().cast(), at as *const u8, 4 * dst.len()) });
        }
        let ns = t0.elapsed().as_nanos() as f64 / 1784.0;
        println!("cost, no repeats: {ns:.0} ns per skinned draw, {} bones ({} KiB)", hb.bones(), hb.bones() * 48 / 1024);
    }

    /// A made-up instanced tyre draw: 44 groups of 11 meshlets (4 instances),
    /// bases 7, -1 (the slot-0 record's +0x30 = 7), 7, 7; bone k's floats all k.
    #[test]
    fn wheel_reads_and_reasons() {
        use std::collections::BTreeMap;
        let slots = xnw_frame::audit::INST_PROGRAMS[1].1;
        let mut mem: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        let le = |w: &[u32]| -> Vec<u8> { w.iter().flat_map(|v| v.to_le_bytes()).collect() };
        mem.insert(0x10000, le(&[11]));
        let mut recs = Vec::new();
        for k in 0..4u32 {
            let d: [f32; 12] = [1.0, 0.0, 0.0, 10.0 + k as f32, 0.0, 1.0, 0.0, 20.0, 0.0, 0.0, 1.0, 30.0];
            recs.extend(d.iter().flat_map(|v| v.to_le_bytes()));
            recs.extend([0u8; 80]);
        }
        mem.insert(0x20000, recs);
        mem.insert(0x30000, le(&[7, u32::MAX, 7, 7]));
        mem.insert(0x40000, vec![0u8; 0xe0]);
        let mut rec0 = vec![0u8; 0x80];
        rec0[0x30..0x34].copy_from_slice(&7u32.to_le_bytes());
        rec0[0x74..0x78].copy_from_slice(&u32::MAX.to_le_bytes());
        mem.insert(0x50000, rec0);
        mem.insert(0x60000, (0..300 * 12).flat_map(|i| ((i / 12) as f32).to_le_bytes()).collect());
        let views: BTreeMap<(u16, u32), (u64, u64)> = [((1, slots.meshlets), (0x10000, 4)), ((1, slots.records), (0x20000, 512)), ((1, slots.bases), (0x30000, 16)), ((1, slots.view), (0x40000, 0xe0)), ((1, 0), (0x50000, 0x80)), ((0, 5), (0x60000, 48 * 300))].into();
        let run = |mem: &BTreeMap<u64, Vec<u8>>, views: &BTreeMap<(u16, u32), (u64, u64)>, groups: u32| {
            let view = |r: u16, s: u32| views.get(&(r, s)).copied();
            let mut read = |at: u64, len: usize| mem.range(..=at).next_back().and_then(|(b, v)| v.get((at - b) as usize..(at - b) as usize + len)).map(<[u8]>::to_vec);
            wheel_reads(slots, groups, &view, &mut read)
        };
        let w = run(&mem, &views, 44).unwrap();
        assert_eq!(w.worlds.len(), 4);
        assert_eq!(w.worlds[3], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 13.0, 20.0, 30.0]);
        assert_eq!((w.bones.len(), w.bones[0], w.bones[12 * 255]), (12 * WHEEL_BONES, 7.0, 262.0));
        // Reasons, never a fault.
        let mut v = views.clone();
        v.remove(&(1, slots.view));
        assert_eq!(run(&mem, &v, 44), Err(0), "slot view missing");
        assert_eq!(run(&mem, &views, 10), Err(1), "fewer groups than one instance");
        let mut m = mem.clone();
        m.insert(0x30000, le(&[7, 8, 7, 7]));
        assert_eq!(run(&m, &views, 44), Err(2), "bases differ");
        let mut m = mem.clone();
        m.get_mut(&0x50000).unwrap()[0x74..0x78].copy_from_slice(&0x40u32.to_le_bytes());
        assert_eq!(run(&m, &views, 44), Err(3), "mask on");
        let mut m = mem.clone();
        m.insert(0x30000, le(&[100; 4]));
        assert_eq!(run(&m, &views, 44), Err(4), "the window runs past the palette view");
        let mut v = views.clone();
        v.insert((1, slots.records), (0x70000, 512));
        assert_eq!(run(&mem, &v, 44), Err(4), "unreadable records");
        // The instance cap and the record view bound n.
        let mut v = views.clone();
        v.insert((1, slots.records), (0x20000, 256));
        assert_eq!(run(&mem, &v, 44).unwrap().worlds.len(), 2);
    }

    #[test]
    fn wheel_line_takes_the_counters() {
        for (k, v) in [3u64, 2, 0, 0, 0, 0, 0, 1, 2_500_000].into_iter().enumerate() {
            WHEEL_N[k].store(v, Ordering::Relaxed);
        }
        assert_eq!(
            wheel_line(60, 2, [3, 1, 0, 0]),
            "world render: frame 60: vehicle wheels (60 frames): instanced sinks 3, given to their draw 2; refused: slot views missing 0, no instances 0, bases differ 0, mask on 0, unreadable 0, no hook draw 1; 2.500 ms at the sink; this frame 2 rows drawn per instance (rows with reads 3; left: remap already 1, no palette or instances 0, palette too far 0)"
        );
        assert!(WHEEL_N.iter().all(|a| a.load(Ordering::Relaxed) == 0));
    }

    #[test]
    fn rules_keep_bit_8() {
        assert_eq!(parse_rules("screen,skinrules=1cb"), 0x1CB);
        assert_eq!(parse_rules("screen,skinrules=cb"), 0xCB);
        assert_eq!(parse_rules("screen"), 0);
        assert_eq!(xnw_world::parse_skin_rules("screen,skinrules=1cb"), 0xCB, "the pass sees bits 0-7 only");
    }

    #[test]
    fn switch_default_off() {
        assert!(!switch("screen"));
        assert!(!switch("screen,drawbones=off"));
        assert!(switch("screen, drawbones=on"));
        assert!(!switch("screen,drawbones=on,drawbones=off"));
    }

    #[test]
    fn sources_line() {
        let rec = |palette: u32, first: u32| DrawRecord { geom: GeomKey::default(), world: [0.0; 16], viewport: 1, effect: 0, albedo: Albedo::Flat([1.0; 3]), palette, stage: 67, inst_first: 0, inst_count: 0, bone_first: first, bone_count: 1, material: Material::default(), state: None, remap: NO_REMAP, ped: false };
        let shear = [1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        // Palette 0: end of frame (identity, shear); palette 1: draw-time (identity, identity).
        let list = FrameList { draws: vec![rec(0, 0), rec(0, 1), rec(1, 0), rec(1, 1), rec(NO_PALETTE, 0)], palettes: vec![[ID, shear].concat(), [ID, ID].concat()], palette_at_draw: vec![false, true], ..Default::default() };
        assert_eq!(sources(&list), [2, 2, 2, 1]);
        let hb = HookBones { floats: vec![0.0; 24], given: 2, why: [0, 1, 3, 0], ..Default::default() };
        assert_eq!(
            line(60, &list, 7, &hb),
            "world render: frame 60: skin sources: draw-time 2 (2 identity), end-of-frame 2 (1 identity), no bones 3 of 7 hooked; draw-time copy 2 bones (0 KiB) for 2 draws; refused: no pointer 0, count 0 or over 4096 1, unreadable 3, frame cap 0"
        );
    }
}
