//! `XNW_LATE_LOG=1`: when the inputs of the title's culling dispatches
//! are written. The title records
//! the culling dispatches first and fills their 576-byte block (dispatch
//! counts, a 456-byte constant buffer at +96) in command-list memory later,
//! up to two flushes after the call; the console's GPU reads it at execute
//! time. For dispatches of pipelines whose name contains "Cull", each bound
//! constant range, root view and indirect argument range is noted when the
//! dispatch is recorded: its first 16 bytes and what backs it (the paired
//! reservation's CPU view = primary, its GPU view = alias; in place in a
//! shared window, a host copy, or neither). The same bytes are logged again
//! when the list is submitted (our GPU runs it then, and the submission
//! waits for it) and once more at the next submission, after it ran: a
//! change only there is a write our GPU came too early for.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Whether `XNW_LATE_LOG=1`.
pub(crate) fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("XNW_LATE_LOG").is_ok_and(|v| v.trim() == "1"));
    *ON
}

/// Lines logged at most (three per range: record, submission, after).
const LINES: u64 = 6000;
static LOGGED: AtomicU64 = AtomicU64::new(0);

fn log(line: std::fmt::Arguments) {
    if LOGGED.fetch_add(1, Ordering::Relaxed) < LINES {
        xnw_core::info!("late: {line}");
    }
}

/// One noted range.
pub(crate) struct Note {
    what: &'static str,
    pipeline: String,
    address: u64,
    len: u64,
    recorded: [u8; 16],
    id: u64,
}

/// The first 16 bytes at `address` (zeros where unreadable: read through
/// the kernel, never faulting).
fn first16(address: u64) -> [u8; 16] {
    let mut out = [0u8; 16];
    super::host::copy_mapped(out.as_mut_ptr(), address as *const u8, 16, false);
    out
}

/// What backs `[address, address + len)`: the paired reservation's view,
/// and in place, host copy or neither.
fn mapping(address: u64, len: u64) -> String {
    let view = if super::watch::gpu_alias(address as usize).is_some() {
        "primary (CPU view)"
    } else if super::watch::paired(address as usize) {
        "alias (GPU view)"
    } else {
        "outside the paired reservation"
    };
    let backing = if let Some(r) = crate::shared::lookup(address, len) {
        format!("in place (window {} + {:#x})", crate::shared::window_id(&r.buffer), r.offset)
    } else if let Some(s) = super::shadow::find(address, len) {
        format!("host copy (buffer at {:#x}, refreshed at submission)", address - s.offset(address))
    } else {
        "no host copy and not in place (read at submission where the path fills from memory)".into()
    };
    format!("{view}, {backing}")
}

/// Whether ranges of the pipeline `name` are noted.
pub(crate) fn wanted(name: &str) -> bool {
    on() && name.contains("Cull")
}

/// Notes `[address, address + len)` read by a culling dispatch being
/// recorded (`what`: root constant buffer, table constants, arguments...).
pub(crate) fn recorded(what: &'static str, pipeline: &str, address: u64, len: u64) -> Note {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let recorded = first16(address);
    log(format_args!(
        "#{id} record frame {}: {pipeline} {what} {address:#x}+{len:#x}: {} [{}]",
        super::queue::presented(),
        hex(&recorded),
        mapping(address, len)
    ));
    Note { what, pipeline: pipeline.to_string(), address, len, recorded, id }
}

/// Ranges submitted, to look at again once their submission has run.
static AFTER: Mutex<Vec<(Note, [u8; 16])>> = Mutex::new(Vec::new());

/// At a submission, before it runs: the list's noted ranges as the GPU
/// will read them now; the ranges of the submission before, after it ran.
pub(crate) fn submitting(notes: Vec<Note>) {
    let before = std::mem::take(&mut *AFTER.lock().unwrap_or_else(|e| e.into_inner()));
    for (n, at_submit) in before {
        let now = first16(n.address);
        log(format_args!(
            "#{} after its submission ran: {} ({})",
            n.id,
            hex(&now),
            if now != at_submit { "CHANGED after our GPU read it: written too late for us" } else { "unchanged" }
        ));
    }
    let mut after = Vec::with_capacity(notes.len());
    for n in notes {
        let now = first16(n.address);
        log(format_args!(
            "#{} submit frame {}: {} {} {:#x}+{:#x}: {} ({})",
            n.id,
            super::queue::presented(),
            n.pipeline,
            n.what,
            n.address,
            n.len,
            hex(&now),
            if now != n.recorded { "changed since recorded" } else { "same as recorded" }
        ));
        after.push((n, now));
    }
    AFTER.lock().unwrap_or_else(|e| e.into_inner()).extend(after);
}

fn hex(b: &[u8; 16]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
