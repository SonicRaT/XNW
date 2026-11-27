//! Live switches inside one running title (SW1): the table is
//! `xnw_core::live`; this starts it for a title run. At start every
//! switch's reader is asked once (the table then holds the run's start
//! values), the key map and the `bridgefam` bit order are logged, and a
//! thread polls the control file once a second.
//!
//! The control file (`XNW_LIVE_FILE`, default `out\live.txt` under the
//! working directory; `off`: no poll): lines or comma lists of
//! `name=value`, `#` comments. What the file holds at start is the
//! baseline, not applied (a stale file never changes a run); from then on
//! an item whose value changed in the file is set, once (a key press is
//! not undone by the next poll). Refused items are logged once per change.

use xnw_core::live;

/// The file's path, or None (`XNW_LIVE_FILE=off`).
fn path() -> Option<std::path::PathBuf> {
    match std::env::var("XNW_LIVE_FILE") {
        Ok(v) if v.trim() == "off" => None,
        Ok(v) if !v.trim().is_empty() => Some(v.trim().into()),
        _ => Some(std::path::Path::new("out").join("live.txt")),
    }
}

/// Sets each item at presented frame `frame` (`who`: the driver, for refusals).
pub(crate) fn apply(items: &[(String, String)], who: &str, frame: u64) {
    for (n, v) in items {
        if let Err(e) = live::set(n, v, frame) {
            xnw_core::warn!("live: {who}: {e}; ignored");
        }
    }
}

/// Called once before the title runs (run.rs).
pub(crate) fn start() {
    let fams = crate::worldprobe::live_prime();
    let _ = xnw_world::lag_on();
    let _ = xnw_world::bridged::hold_mode();
    let _ = xnw_world::bridged::decal_blend([0; 8]);
    let _ = xnw_world::ourlight::live_on();
    let _ = xnw_world::ourlight::live_normals();
    xnw_core::info!("live: keys (title window focused; the title does not see F1-F12): {}", live::key_map());
    if !fams.is_empty() {
        let order: Vec<String> = fams.iter().enumerate().map(|(k, h)| format!("bit {k} {h:08x}")).collect();
        xnw_core::info!("live: bridgefam bits: {}", order.join(", "));
    }
    let Some(file) = path() else {
        xnw_core::info!("live: no control file (XNW_LIVE_FILE=off)");
        return;
    };
    let mut poll = live::Poll::default();
    let base = poll.changes(&std::fs::read_to_string(&file).unwrap_or_default());
    xnw_core::info!("live: control file {} polled once a second ({} items there now: the baseline, not applied); names {}", file.display(), base.len(), live::names());
    let spawned = std::thread::Builder::new().name("xnw-live".into()).spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let text = std::fs::read_to_string(&file).unwrap_or_default();
            let ch = poll.changes(&text);
            if !ch.is_empty() {
                apply(&ch, "file", xnw_gpu::d3d12x::presented());
            }
        }
    });
    if let Err(e) = spawned {
        xnw_core::warn!("live: control file thread: {e}; no poll");
    }
}
