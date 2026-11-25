//! The burst's title memory, as the capture run had it (ZC1).
//!
//! - The census: every read a burst serves (`burst::read`: geometry reads
//!   from the geometry table, the sink consumers' reads from the frame's pack
//!   objects) is split into 64 KiB pages and each page classed by the run's
//!   own allocations (its log's `XMemVirtualAlloc` lines), as the earlier 'frame
//!   zero-copy' line classes live copies: physical pages (a 64 KiB-page
//!   reservation, openable through a page window of its section), a region
//!   under 4 GiB (openable whole, by its base), a region of 4 GiB or more,
//!   or none (heap, image, stack). A captured page was mapped when captured,
//!   so a physical page counts as openable [I: contiguity across pages is the
//!   live xmem's, not known from a capture].
//! - `XNW_ZEROCOPY=on`: the run's reservations made again at their addresses
//!   through our own xmem (`replay_reservation`), physical pages mapped where
//!   captured objects lie (section offset = reservation offset: one section
//!   per reservation, so a range is always contiguous: the best case), the
//!   objects written there, and xmem's lookups handed to `xnw_gpu::shared`,
//!   so the world pass and the bridge read them in place through the same
//!   windows and heaps the live runtime opens.

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

const PAGE: u64 = 64 << 10;
const KIND_COMMIT: u32 = 0x1000;
const PAGES_64K: u32 = 0x2040_0000;
const HEAP_LIMIT: u64 = 4 << 30;
const PLACEHOLDER: u32 = 0x4_0000;

/// One reservation of the run: address, bytes, kind, XMem flags, protection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Alloc {
    pub at: u64,
    pub size: u64,
    pub kind: u32,
    pub flags: u64,
    pub protect: u32,
}

impl Alloc {
    /// Physical pages go here: a 64 KiB-page reservation, or one the title
    /// asked for as a host placeholder (kind 0x42000, the 32-bit block).
    fn placeholder(&self) -> bool {
        self.kind & KIND_COMMIT == 0 && (self.kind & PAGES_64K == PAGES_64K || self.kind & PLACEHOLDER != 0)
    }
}

/// The run's `XMemVirtualAlloc(0x0, size, kind K, flags F, protect P) -> at`
/// lines (reservations at an address of xmem's choosing), sorted by address.
pub fn parse(log: &str) -> Vec<Alloc> {
    let hex = |s: &str| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    let mut out: Vec<Alloc> = log
        .lines()
        .filter_map(|l| {
            let rest = l.split("XMemVirtualAlloc(0x0, ").nth(1)?;
            if rest.contains("FAILED") {
                return None;
            }
            let (args, at) = rest.split_once(") -> ")?;
            let mut f = args.split(", ");
            let size = hex(f.next()?)?;
            let kind = hex(f.next()?.strip_prefix("kind ")?)? as u32;
            let flags = hex(f.next()?.strip_prefix("flags ")?)?;
            let protect = hex(f.next()?.strip_prefix("protect ")?)? as u32;
            Some(Alloc { at: hex(at.split_whitespace().next()?)?, size, kind, flags, protect })
        })
        .filter(|a| a.at != 0)
        .collect();
    out.sort_by_key(|a| a.at);
    out
}

/// Page classes.
pub const CLASSES: [&str; 4] = ["physical pages", "region under 4 GiB", "region of 4 GiB or more", "no section or region"];
/// Who read: the list's geometry step, everything else (the sink
/// consumers: bridge, pre-skin, wheels; imposters).
pub const TABLES: [&str; 2] = ["geometry", "sink consumers (bridge, pre-skin, wheels, imposters)"];

static MAP: Mutex<Vec<Alloc>> = Mutex::new(Vec::new());

thread_local! {
    /// Set around the list's geometry step: its reads count as geometry.
    pub static GEOMETRY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
/// Per table: bytes by class; bytes of reads all of whose pages are openable, all bytes; reads.
static BYTES: [[AtomicU64; 4]; 2] = [const { [const { AtomicU64::new(0) }; 4] }; 2];
static WHOLE: [[AtomicU64; 3]; 2] = [const { [const { AtomicU64::new(0) }; 3] }; 2];

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The reservation holding `address`.
fn find(map: &[Alloc], address: u64) -> Option<Alloc> {
    let i = map.partition_point(|a| a.at <= address);
    map.get(i.checked_sub(1)?).copied().filter(|a| address < a.at + a.size)
}

fn class(map: &[Alloc], page: u64) -> usize {
    match find(map, page) {
        Some(a) if a.placeholder() => 0,
        Some(a) if a.size < HEAP_LIMIT => 1,
        Some(_) => 2,
        None => 3,
    }
}

/// Loads the run's reservations from the log beside the capture (`<dir>/../run.log`
/// or `log`); their count, or why not.
pub fn load(dir: &Path, log: Option<&Path>) -> Result<usize, String> {
    let path = log.map(Path::to_path_buf).unwrap_or_else(|| dir.parent().unwrap_or(dir).join("run.log"));
    let text = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let map = parse(&String::from_utf8_lossy(&text));
    let n = map.len();
    *lock(&MAP) = map;
    Ok(n)
}

/// Notes a read of `len` bytes at `addr` (geometry while [`GEOMETRY`] is set).
pub fn note(addr: u64, len: usize) {
    let table = usize::from(!GEOMETRY.with(|g| g.get()));
    let map = lock(&MAP);
    if map.is_empty() || len == 0 {
        return;
    }
    let end = addr + len as u64;
    let mut page = addr & !(PAGE - 1);
    let mut whole = true;
    while page < end {
        let c = class(&map, page);
        let bytes = (page + PAGE).min(end) - page.max(addr);
        BYTES[table][c].fetch_add(bytes, Ordering::Relaxed);
        whole &= c <= 1;
        page += PAGE;
    }
    WHOLE[table][0].fetch_add(if whole { len as u64 } else { 0 }, Ordering::Relaxed);
    WHOLE[table][1].fetch_add(len as u64, Ordering::Relaxed);
    WHOLE[table][2].fetch_add(1, Ordering::Relaxed);
}

/// The census since the last call, zeroing: per table, KiB by class and
/// the share in reads whose every 64 KiB page is openable.
pub fn text() -> String {
    let (mut ok, mut all) = (0, 0);
    let parts: Vec<String> = (0..2)
        .filter_map(|t| {
            let b: Vec<u64> = BYTES[t].iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
            let w: Vec<u64> = WHOLE[t].iter().map(|c| c.swap(0, Ordering::Relaxed)).collect();
            (w[1] > 0).then(|| {
                ok += w[0];
                all += w[1];
                let cls: Vec<String> = CLASSES.iter().zip(&b).filter(|x| *x.1 > 0).map(|(c, b)| format!("{c} {} KiB", b >> 10)).collect();
                format!("{} {} reads {} KiB [{}], {}% in reads wholly openable", TABLES[t], w[2], w[1] >> 10, cls.join(", "), w[0] * 100 / w[1].max(1))
            })
        })
        .collect();
    format!(
        "{} KiB of {} KiB read in reads whose every 64 KiB page is openable ({}%); {}",
        ok >> 10,
        all >> 10,
        ok * 100 / all.max(1),
        if parts.is_empty() { "nothing read".into() } else { parts.join("; ") }
    )
}

// ---- `XNW_ZEROCOPY=on`: the run's memory made again through our xmem ----

struct Mapped {
    a: Alloc,
    /// Page numbers of its section (physical pages: one per 64 KiB, made at first use).
    pages: Vec<usize>,
    /// Pages mapped so far, one bit each.
    mapped: Vec<u64>,
}

struct Mirror {
    allocs: Vec<Mapped>,
    /// GPU view base -> CPU view base and bytes (the paired reservation).
    pairs: Vec<(u64, u64, u64)>,
    /// Bytes written, pages mapped.
    written: u64,
    pages: u64,
}

static MIRROR: Mutex<Option<Mirror>> = Mutex::new(None);

/// Makes the run's reservations again at their addresses and hands xmem's
/// lookups to the shared layer. A summary line, or why not.
pub fn mirror_start() -> Result<String, String> {
    use xnw_hle::title_memory as tm;
    let map = lock(&MAP).clone();
    if map.is_empty() {
        return Err("no reservations (the run's log has none)".into());
    }
    let (mut ok, mut failed, mut bytes) = (Vec::new(), 0usize, 0u64);
    for a in &map {
        if tm::replay_reservation(a.at as usize, a.size as usize, a.kind, a.flags, a.protect) {
            bytes += a.size;
            ok.push(Mapped { a: *a, pages: Vec::new(), mapped: Vec::new() });
        } else {
            failed += 1;
        }
    }
    // The pair as xmem recognises it: a 64 KiB-page reservation the title
    // may read, then one of the same size and kind it may not.
    let by_time: Vec<Alloc> = map.clone();
    let mut pairs = Vec::new();
    for w in by_time.windows(2) {
        let (c, g) = (w[0], w[1]);
        if c.placeholder() && g.placeholder() && c.size == g.size && c.kind == g.kind && c.protect & 0xff != 1 && g.protect & 0xff == 1 {
            pairs.push((g.at, c.at, c.size));
        }
    }
    xnw_gpu::shared::use_memory(tm::backing, tm::physical_window);
    // `put` opens each window as it loads (a runtime's prewarm at map
    // time, done inline here so `--steady` frames find the same heaps open).
    xnw_gpu::shared::WORLD_OPENS_INLINE.store(true, Ordering::Relaxed);
    let line = format!("{} reservations made again ({} GiB of address space), {failed} refused (address taken); pairs {:x?}", ok.len(), bytes >> 30, pairs);
    *lock(&MIRROR) = Some(Mirror { allocs: ok, pairs, written: 0, pages: 0 });
    Ok(line)
}

/// Whether the mirror is up.
pub fn mirrored() -> bool {
    lock(&MIRROR).is_some()
}

/// Writes `bytes` at title address `at` (as the title would have), mapping
/// physical pages there first. False when no reservation holds it.
pub fn put(at: u64, bytes: &[u8]) -> bool {
    use xnw_hle::title_memory as tm;
    let mut g = lock(&MIRROR);
    let Some(m) = g.as_mut() else { return false };
    // An address in the GPU view of the pair is written through the CPU view.
    let at = m.pairs.iter().find(|p| at >= p.0 && at < p.0 + p.2).map_or(at, |p| at - p.0 + p.1);
    let i = m.allocs.partition_point(|x| x.a.at <= at);
    let Some(x) = i.checked_sub(1).and_then(|i| m.allocs.get_mut(i)).filter(|x| at < x.a.at + x.a.size) else { return false };
    let len = (bytes.len() as u64).min(x.a.at + x.a.size - at);
    // A reservation without physical pages or commitment has nothing to write to.
    if len == 0 || !x.a.placeholder() && x.a.kind & KIND_COMMIT == 0 {
        return false;
    }
    if x.a.placeholder() {
        let n = (x.a.size / PAGE) as usize;
        if x.pages.is_empty() {
            let mut count = n;
            let mut pages = vec![0usize; n];
            if tm::XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr()) == 0 || count != n {
                return false;
            }
            x.pages = pages;
            x.mapped = vec![0; n.div_ceil(64)];
        }
        let (p0, p1) = (((at - x.a.at) / PAGE) as usize, ((at - x.a.at + len).div_ceil(PAGE)) as usize);
        let mut p = p0;
        while p < p1 {
            if x.mapped[p / 64] >> (p % 64) & 1 != 0 {
                p += 1;
                continue;
            }
            let mut q = p;
            while q < p1 && x.mapped[q / 64] >> (q % 64) & 1 == 0 {
                q += 1;
            }
            let base = x.a.at + p as u64 * PAGE;
            if tm::XMemMapPhysicalPages(base as *mut std::ffi::c_void, q - p, x.pages[p..].as_ptr()).is_null() {
                return false;
            }
            for k in p..q {
                x.mapped[k / 64] |= 1 << (k % 64);
            }
            m.pages += (q - p) as u64;
            p = q;
        }
    }
    // SAFETY: `[at, at + len)` is committed read-write memory of ours (a
    // mapped section view or a committed region made above).
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, len as usize) };
    m.written += len;
    drop(g);
    // Its heap opened now, while the burst loads (as a runtime that opens a
    // section's heaps when the title allocates it would), not in the frame.
    xnw_gpu::shared::world_open(at, len);
    true
}

/// Bytes written and pages mapped so far.
pub fn mirror_text() -> String {
    lock(&MIRROR).as_ref().map_or("off".into(), |m| format!("{} MiB written, {} MiB of physical pages mapped", m.written >> 20, (m.pages * PAGE) >> 20))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_log_reservations_parse_and_class() {
        let log = "    0.761 info  XMemVirtualAlloc(0x0, 0x400000000, kind 0x20402000, flags 0x21, protect 0x4) -> 0x100000000\n\
                   0.761 info  XMemVirtualAlloc(0x0, 0x400000000, kind 0x20402000, flags 0x3, protect 0x801) -> 0x500000000\n\
                   0.761 info  XMemVirtualAlloc(0x0, 0x48a00000, kind 0x20003000, flags 0x183, protect 0x22004) -> 0x900000000\n\
                   0.8 info  XMemVirtualAlloc(0x0, 0x1000, kind 0x3000, flags 0x0, protect 0x4) -> 0x0  FAILED: x\n";
        let map = parse(log);
        assert_eq!(map.len(), 3);
        assert_eq!(map[2], Alloc { at: 0x9_0000_0000, size: 0x48a0_0000, kind: 0x2000_3000, flags: 0x183, protect: 0x22004 });
        assert!(map[0].placeholder() && map[1].placeholder() && !map[2].placeholder());
        assert_eq!(class(&map, 0x3_3866_0000), 0);
        assert_eq!(class(&map, 0x5_0000_0000), 0);
        assert_eq!(class(&map, 0x9_1000_0000), 1);
        assert_eq!(class(&map, 0x9_8000_0000), 3);
        assert_eq!(class(&map, 0x7FF0_0000_0000), 3);
    }
}
