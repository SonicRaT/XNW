//! Buffer roles: which way each title buffer's data flows, learned per
//! host copy ([`super::shadow::Shadow`]) and kept across runs.
//!
//! A console title's buffers are mostly one-way: the CPU fills them for the
//! GPU (once, or every frame), or the GPU fills them for itself; few are
//! read back. We treated every buffer as two-way: each range shaders may
//! write was snapshotted before and compared after every submission, then
//! copied back (1,069 ms a heavy frame). Here each buffer's flows are measured:
//! - GPU writes: ranges declared written by shaders (`Shadow::writing`);
//! - CPU reads of what the GPU wrote: after a GPU write the written pages
//!   are trapped against access ([`super::watch::trap_reads`], tag
//!   `TAG_BASE + id`); a fault is the title reading (or writing) them;
//! - CPU writes: pages the write watch found written at refreshes.
//!
//! Classes: GPU-only (GPU writes, never read by the CPU), read-back (GPU
//! writes the CPU reads), static (CPU-written once), per-frame (CPU-written
//! again and again). `XNW_ROLES=probe` measures and prints; `XNW_ROLES=on`
//! also acts: a GPU-written range is no longer compared and copied back at
//! every submission; its pages stay trapped and are copied back when the
//! title (or our own read of title memory) touches them, which is exact
//! (a trapped page cannot change under us). Static and per-frame buffers
//! keep today's path (their placement in video memory is not done yet).
//! Classes are kept in `<cache>/roles.txt` by a key stable across runs (see
//! [`key`]); a fault always wins over what the file says.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::shadow::Shadow;

/// Trap tags of the buffer roles: `TAG_BASE + id` (other traps use 0x1000-0x1FFF).
pub(crate) const TAG_BASE: u32 = 0x1_0000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    Off,
    Probe,
    On,
    /// GPU-written ranges go back when the title locks the buffer for the
    /// CPU (its lock function, detoured by the loader): no snapshot,
    /// compare or trap (every CPU read of a GPU-written buffer
    /// goes through that lock).
    Lock,
}

/// `XNW_ROLES`: unset or `off`, `probe`, `on`, `lock`. `on`: GPU-written
/// ranges that are not in place go back when touched (traps), with no
/// snapshot or compare at submission. `lock`: they go back when the title
/// locks them. Off by default until a title run proves one.
pub(crate) fn mode() -> Mode {
    #[cfg(test)]
    if let Some(m) = TEST_MODE.with(std::cell::Cell::get) {
        return m;
    }
    static MODE: std::sync::LazyLock<Mode> = std::sync::LazyLock::new(|| match std::env::var("XNW_ROLES").as_deref() {
        Ok("probe") => Mode::Probe,
        Ok("on") => Mode::On,
        Ok("lock") => Mode::Lock,
        _ => Mode::Off,
    });
    *MODE
}

/// Whether GPU writes to host copies stay there until the CPU needs them
/// (`on` or `lock`): one copy for every view of such pages.
pub(crate) fn one_way() -> bool {
    matches!(mode(), Mode::On | Mode::Lock)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub(crate) enum Class {
    /// Not seen written by anyone yet.
    Unknown,
    Static,
    PerFrame,
    GpuOnly,
    ReadBack,
}

impl Class {
    const NAMES: [&'static str; 5] = ["unknown", "static", "per-frame", "gpu-only", "read-back"];

    fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }

    fn parse(s: &str) -> Option<Class> {
        Self::NAMES.iter().position(|&n| n == s).map(|i| [Class::Unknown, Class::Static, Class::PerFrame, Class::GpuOnly, Class::ReadBack][i])
    }
}

/// What one host copy's buffer was seen doing.
pub(crate) struct Role {
    pub(crate) tag: u32,
    /// The buffer (title address) and its key across runs, made at the
    /// first statistics window, not with the copy: [`key`] asks xmem, whose
    /// lock is held around title memory remaps, which take the host copies'
    /// locks (the copy was made under `shadow::LOOSE`, a remap under
    /// xmem's lock wanted `LOOSE`: a test run stopped at world entry).
    address: u64,
    key: std::sync::OnceLock<String>,
    len: u64,
    /// Ranges declared written by shaders, in all.
    gpu_writes: AtomicU64,
    /// CPU accesses of trapped (GPU-written) pages: reads, writes.
    cpu_faults: [AtomicU64; 2],
    /// Pages the write watch found written at refreshes, in all; and the
    /// frames (plus one) written in, first and last, and how many.
    cpu_pages: AtomicU64,
    cpu_frames: [AtomicU64; 3],
    /// The class the cache file gave (as a `Class` number).
    remembered: std::sync::atomic::AtomicU8,
}

impl Role {
    /// The class as measured so far, the file's where nothing was.
    pub(crate) fn class(&self) -> Class {
        let gpu = self.gpu_writes.load(Ordering::Relaxed) > 0;
        let read = self.cpu_faults[0].load(Ordering::Relaxed) > 0;
        let remembered = [Class::Unknown, Class::Static, Class::PerFrame, Class::GpuOnly, Class::ReadBack][self.remembered.load(Ordering::Relaxed) as usize];
        match (gpu, read) {
            (true, true) => Class::ReadBack,
            // A fault always wins; without one the file's read-back stands.
            (true, false) if remembered == Class::ReadBack => Class::ReadBack,
            (true, false) => Class::GpuOnly,
            _ => match self.cpu_frames[2].load(Ordering::Relaxed) {
                0 => remembered,
                1 => Class::Static,
                _ => Class::PerFrame,
            },
        }
    }

    /// Shaders of a submission may write a range of the buffer.
    pub(crate) fn gpu_wrote(&self) {
        self.gpu_writes.fetch_add(1, Ordering::Relaxed);
    }

    /// A refresh found `pages` pages written by the CPU.
    pub(crate) fn cpu_wrote(&self, pages: u64) {
        if pages == 0 {
            return;
        }
        self.cpu_pages.fetch_add(pages, Ordering::Relaxed);
        let frame = super::queue::presented() + 1;
        if self.cpu_frames[1].swap(frame, Ordering::Relaxed) != frame {
            let _ = self.cpu_frames[0].compare_exchange(0, frame, Ordering::Relaxed, Ordering::Relaxed);
            self.cpu_frames[2].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Whether the buffer's GPU writes go back only on demand (`on`).
    pub(crate) fn deferred(&self) -> bool {
        mode() == Mode::On
    }
}

/// Live roles by tag, for the trap hook and the statistics.
static ROLES: Mutex<BTreeMap<u32, Weak<Shadow>>> = Mutex::new(BTreeMap::new());
static NEXT: AtomicU64 = AtomicU64::new(0);

/// The role of a new host copy of `len` bytes at title `address`; `None`
/// when roles are off.
pub(crate) fn new_role(address: u64, len: u64) -> Option<Role> {
    if mode() == Mode::Off {
        return None;
    }
    let tag = TAG_BASE + NEXT.fetch_add(1, Ordering::Relaxed) as u32;
    Some(Role {
        tag,
        address,
        key: std::sync::OnceLock::new(),
        len,
        gpu_writes: AtomicU64::new(0),
        cpu_faults: [AtomicU64::new(0), AtomicU64::new(0)],
        cpu_pages: AtomicU64::new(0),
        cpu_frames: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        remembered: std::sync::atomic::AtomicU8::new(Class::Unknown as u8),
    })
}

impl Role {
    /// The key across runs, made on first use (no lock of ours held), with
    /// the class the cache file remembers for it.
    fn key(&self) -> &str {
        self.key.get_or_init(|| {
            let k = key(self.address, self.len);
            if let Some(&c) = remembered().get(&k) {
                self.remembered.store(c as u8, Ordering::Relaxed);
            }
            k
        })
    }
}

/// Registers a host copy with a role (after it is made).
pub(crate) fn register(shadow: &Arc<Shadow>) {
    if let Some(role) = shadow.role() {
        ROLES.lock().unwrap_or_else(|e| e.into_inner()).insert(role.tag, Arc::downgrade(shadow));
        // Probe only counts: no hook, so a trap's block never goes
        // read-write in between (a guard's write would be missed).
        if mode() == Mode::On {
            static HOOK: std::sync::Once = std::sync::Once::new();
            HOOK.call_once(|| super::watch::on_trap(TAG_BASE..u32::MAX, on_trap));
        }
    }
}

/// A key for a buffer stable across runs. Addresses in the paired
/// reservation repeat run to run (0x98af2df00+0x17ebf00 and others in
/// two test runs); the title's other regions move
/// (the big heap at 0x15bd0000 then 0x4530000) but keep their offset in a
/// region of the same size (0xb00000 into both): such buffers are keyed by
/// region size and offset.
pub(crate) fn key(address: u64, len: u64) -> String {
    if super::watch::paired(address as usize) {
        format!("p{address:x}+{len:x}")
    } else if let Some((base, size)) = crate::shared::region_of(address as usize) {
        format!("r{size:x}@{:x}+{len:x}", address as usize - base)
    } else {
        format!("a{address:x}+{len:x}")
    }
}

/// The classes the cache file holds, read once.
fn remembered() -> &'static BTreeMap<String, Class> {
    static FILE: std::sync::OnceLock<BTreeMap<String, Class>> = std::sync::OnceLock::new();
    FILE.get_or_init(|| {
        let Some(path) = super::shader_cache::dir().map(|d| d.join("roles.txt")) else {
            return BTreeMap::new();
        };
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let map: BTreeMap<String, Class> = text
            .lines()
            .filter_map(|l| {
                let (k, c) = l.split_once(' ')?;
                Some((k.to_string(), Class::parse(c.trim())?))
            })
            .collect();
        if !map.is_empty() {
            xnw_core::info!("buffer roles: {} classes from {}", map.len(), path.display());
        }
        map
    })
}

/// The trap hook: a GPU-written block of the buffer behind `tag` is about
/// to be touched; with roles on, its GPU results go back first.
fn on_trap(tag: u32, block_start: usize, _len: usize) {
    if mode() != Mode::On {
        return;
    }
    let shadow = ROLES.lock().unwrap_or_else(|e| e.into_inner()).get(&tag).and_then(Weak::upgrade);
    if let Some(shadow) = shadow {
        shadow.copy_back_deferred(block_start);
    }
}

/// The statistics window: takes the trap counts, prints the table by
/// class, and writes the classes to the cache file. Over `n` frames.
pub(crate) fn stats_text(n: u64) -> String {
    if mode() == Mode::Off {
        return String::new();
    }
    let n = n.max(1);
    let caught = super::watch::take_trap_stats(TAG_BASE, u32::MAX);
    let roles: Vec<Arc<Shadow>> = {
        let mut all = ROLES.lock().unwrap_or_else(|e| e.into_inner());
        all.retain(|_, w| w.strong_count() > 0);
        all.values().filter_map(Weak::upgrade).collect()
    };
    let by_tag: BTreeMap<u32, &Arc<Shadow>> = roles.iter().filter_map(|s| Some((s.role()?.tag, s))).collect();
    let mut sites: BTreeMap<usize, u64> = BTreeMap::new();
    let mut window = BTreeMap::<u32, (u64, u64)>::new();
    for (tag, s) in &caught {
        if let Some(r) = by_tag.get(tag).and_then(|s| s.role()) {
            r.cpu_faults[0].fetch_add(s.reads, Ordering::Relaxed);
            r.cpu_faults[1].fetch_add(s.writes, Ordering::Relaxed);
        }
        window.insert(*tag, (s.reads, s.writes));
        for &(site, _) in &s.sites {
            *sites.entry(site).or_default() += s.reads + s.writes;
        }
    }
    // Per class: buffers, bytes, GPU writes, CPU reads, CPU writes (all in
    // the window, per frame); counts since the start for the classes.
    #[derive(Default)]
    struct Row {
        buffers: u64,
        bytes: u64,
        gpu: u64,
        reads: u64,
        writes: u64,
        pages: u64,
    }
    static LAST: Mutex<BTreeMap<u32, [u64; 2]>> = Mutex::new(BTreeMap::new());
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let mut rows: BTreeMap<Class, Row> = BTreeMap::new();
    let mut lines = Vec::new();
    for s in &roles {
        let Some(r) = s.role() else { continue };
        let _ = r.key();
        let class = r.class();
        let now = [r.gpu_writes.load(Ordering::Relaxed), r.cpu_pages.load(Ordering::Relaxed)];
        let before = last.insert(r.tag, now).unwrap_or([0, 0]);
        let (reads, writes) = window.get(&r.tag).copied().unwrap_or((0, 0));
        let row = rows.entry(class).or_default();
        row.buffers += 1;
        row.bytes += r.len;
        row.gpu += now[0] - before[0];
        row.reads += reads;
        row.writes += writes;
        row.pages += now[1] - before[1];
        if class != Class::Unknown {
            lines.push(format!("{} {}", r.key(), class.name()));
        }
    }
    drop(last);
    write_file(&lines);
    let table: Vec<String> = rows
        .iter()
        .map(|(c, r)| {
            format!(
                "{} {} buffers {} MiB, GPU writes {}/frame, CPU reads {}/frame, CPU writes {} faults + {} pages/frame",
                c.name(),
                r.buffers,
                r.bytes >> 20,
                r.gpu / n,
                r.reads / n,
                r.writes / n,
                r.pages / n
            )
        })
        .collect();
    let mut top: Vec<(usize, u64)> = sites.into_iter().collect();
    top.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
    let top: Vec<String> = top.iter().take(8).map(|(a, c)| format!("{a:#x} x{c}")).collect();
    let lifts = super::watch::RUNTIME_LIFTS.swap(0, Ordering::Relaxed) / n;
    let deferred = super::shadow::DEFERRED.swap(0, Ordering::Relaxed) / n;
    let baselines = super::shadow::BASELINES.swap(0, Ordering::Relaxed) / n;
    let one_copy = crate::shared::ONE_COPY.swap(0, Ordering::Relaxed) / n;
    format!(
        "buffer roles ({}): {}; fault sites {}; traps lifted by the runtime {lifts}/frame; put back on demand {deferred} bytes/frame; in-flight pages the CPU touched {baselines}/frame (merged); in-place views kept on a GPU-written copy {one_copy}/frame{}",
        match mode() {
            Mode::On => "on",
            Mode::Lock => "lock",
            _ => "probe",
        },
        if table.is_empty() { "none".into() } else { table.join("; ") },
        if top.is_empty() { "none".into() } else { top.join(", ") },
        if mode() == Mode::Lock { format!("; {}", super::shadow::locks_text(n)) } else { String::new() }
    )
}

/// Writes the classes (with the file's others kept) to `<cache>/roles.txt`.
fn write_file(lines: &[String]) {
    let Some(dir) = super::shader_cache::dir() else { return };
    let mut all: BTreeMap<String, String> = remembered().iter().map(|(k, c)| (k.clone(), c.name().to_string())).collect();
    for l in lines {
        if let Some((k, c)) = l.split_once(' ') {
            all.insert(k.to_string(), c.to_string());
        }
    }
    let text: String = all.iter().map(|(k, c)| format!("{k} {c}\n")).collect();
    let _ = std::fs::create_dir_all(dir);
    let tmp = dir.join("roles.txt.tmp");
    if std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::rename(&tmp, dir.join("roles.txt"));
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: a roles mode for this thread only.
    pub(crate) static TEST_MODE: std::cell::Cell<Option<Mode>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_follow_the_flows_and_faults_win() {
        let r = Role {
            tag: TAG_BASE,
            address: 0,
            key: std::sync::OnceLock::from(String::new()),
            len: 1,
            gpu_writes: AtomicU64::new(0),
            cpu_faults: [AtomicU64::new(0), AtomicU64::new(0)],
            cpu_pages: AtomicU64::new(0),
            cpu_frames: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            remembered: std::sync::atomic::AtomicU8::new(Class::GpuOnly as u8),
        };
        assert_eq!(r.class(), Class::GpuOnly, "the file's class until seen");
        r.cpu_wrote(3);
        assert_eq!(r.class(), Class::Static);
        r.gpu_wrote();
        assert_eq!(r.class(), Class::GpuOnly);
        r.cpu_faults[0].fetch_add(1, Ordering::Relaxed);
        assert_eq!(r.class(), Class::ReadBack, "a read fault wins");
        assert_eq!(Class::parse("read-back"), Some(Class::ReadBack));
    }
}
