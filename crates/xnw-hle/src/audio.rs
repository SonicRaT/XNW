//! The host end of the title's sound. The title mixes on the CPU and plays
//! through the host's WASAPI, reached through COM like on PC
//! (`api-ms-win-core-com` is bound to the host; VIctory passes `ole32`
//! through the same way). We see one call of it first:
//! `CoCreateInstance(CLSID_MMDeviceEnumerator)`, where the title looks for
//! its render endpoint (title+0x43aea6a, as five-star-recomp's notes on the
//! audio runtime describe it). That is where
//!
//! * `XNW_AUDIO=off` silences every buffer the process hands WASAPI: the
//!   shared `IAudioRenderClient::ReleaseBuffer` slot is patched to add
//!   `AUDCLNT_BUFFERFLAGS_SILENT`, so the title's path stays the same and
//!   nothing reaches the speakers. (Refusing the enumerator instead puts
//!   the title on its fake mixer, which hit an `int3` buffer-overflow
//!   assert at title+0x4dcbc9d after 25-32 s in two runs: not used.) No
//!   volume or mute setting is touched, so nothing persists in Windows'
//!   per-application volume store;
//! * `XNW_AUDIO_CAPTURE=<file.wav>` starts the recorder
//!   ([`crate::audio_capture`]).
//!
//! Every other class goes to the host's `CoCreateInstance` untouched.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::images::caller;
use crate::xgameruntime::HRESULT;

/// A COM interface pointer: a pointer to its method table.
pub(crate) type Com = *mut *const usize;

/// Method `slot` of `this` as type `F`.
///
/// # Safety
/// `this` is a live interface whose method table has `slot` of type `F`.
pub(crate) unsafe fn method<F: Copy>(this: Com, slot: usize) -> F {
    // SAFETY: as the caller promises.
    unsafe { std::mem::transmute_copy(&*(*this).add(slot)) }
}

pub(crate) fn check(what: &str, hr: HRESULT) -> Result<(), String> {
    if hr < 0 { Err(format!("{what} failed: {:#010x}", hr as u32)) } else { Ok(()) }
}

const IID_IMM_DEVICE_ENUMERATOR: [u8; 16] = guid(0xA956_64D2, 0x9614, 0x4F35, [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6]);
const IID_IAUDIO_CLIENT: [u8; 16] = guid(0x1CB9_AD4C, 0xDBFA, 0x4C32, [0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2]);
const IID_IAUDIO_RENDER_CLIENT: [u8; 16] = guid(0xF294_ACFC, 0x3146, 0x4483, [0xA7, 0xBF, 0xAD, 0xDC, 0xA7, 0xC2, 0x60, 0xE2]);
const CLSCTX_ALL: u32 = 0x17;
const AUDCLNT_BUFFERFLAGS_SILENT: u32 = 2;

type ReleaseBufferFn = unsafe extern "system" fn(Com, u32, u32) -> HRESULT;
type GetBufferFn = unsafe extern "system" fn(Com, u32, *mut *mut u8) -> HRESULT;
type InitializeFn = unsafe extern "system" fn(Com, u32, u32, i64, i64, *const c_void, *const [u8; 16]) -> HRESULT;
type PaddingFn = unsafe extern "system" fn(Com, *mut u32) -> HRESULT;
type EventFn = unsafe extern "system" fn(Com, *mut c_void) -> HRESULT;

/// The host's methods before the patches, by patch (see [`PATCHES`]).
static HOST: [std::sync::OnceLock<usize>; 6] = [const { std::sync::OnceLock::new() }; 6];
const P_RELEASE: usize = 0;
const P_GET_BUFFER: usize = 1;
const P_INITIALIZE: usize = 2;
const P_PADDING: usize = 3;
const P_EVENT: usize = 4;
const P_ACTIVATE: usize = 5;
type ActivateFn = unsafe extern "system" fn(Com, *const [u8; 16], u32, *const c_void, *mut Com) -> HRESULT;

extern "system" fn patched_activate(this: Com, iid: *const [u8; 16], context: u32, params: *const c_void, out: *mut Com) -> HRESULT {
    // SAFETY: the caller's arguments.
    let hr = unsafe { host_fn::<ActivateFn>(P_ACTIVATE)(this, iid, context, params, out) };
    // SAFETY: COM callers pass a valid interface id.
    let g = if iid.is_null() { [0; 16] } else { unsafe { *iid } };
    xnw_core::info!(
        "audio: IMMDevice::Activate({:08x}-{:04x}-{:04x}-{:02x}{:02x}-{}) = {:#x} from {}",
        u32::from_le_bytes([g[0], g[1], g[2], g[3]]),
        u16::from_le_bytes([g[4], g[5]]),
        u16::from_le_bytes([g[6], g[7]]),
        g[8],
        g[9],
        g[10..].iter().map(|b| format!("{b:02x}")).collect::<String>(),
        hr as u32,
        caller()
    );
    hr
}

fn host_fn<F: Copy>(patch: usize) -> F {
    let f = *HOST[patch].get().expect("patched after saving");
    // SAFETY: the saved slot has type `F` (the patch's signature).
    unsafe { std::mem::transmute_copy(&f) }
}

static SILENT: AtomicBool = AtomicBool::new(false);

extern "system" fn patched_release_buffer(this: Com, frames: u32, flags: u32) -> HRESULT {
    let flags = if SILENT.load(Ordering::Relaxed) { flags | AUDCLNT_BUFFERFLAGS_SILENT } else { flags };
    // SAFETY: the caller's arguments, maybe with the buffer marked silent.
    let hr = unsafe { host_fn::<ReleaseBufferFn>(P_RELEASE)(this, frames, flags) };
    if crate::acphal::traced() {
        out_stats(|s| {
            let now = std::time::Instant::now();
            if let Some(last) = s.last.replace(now) {
                let ms = (now - last).as_secs_f64() * 1000.0;
                s.gap_max = s.gap_max.max(ms);
                s.gaps_over_10 += u64::from(ms > 10.0);
            }
            s.releases += 1;
            s.frames += u64::from(frames);
            s.silent += u64::from(flags & AUDCLNT_BUFFERFLAGS_SILENT != 0);
        });
    }
    hr
}

extern "system" fn patched_get_buffer(this: Com, frames: u32, data: *mut *mut u8) -> HRESULT {
    // SAFETY: the caller's arguments.
    let hr = unsafe { host_fn::<GetBufferFn>(P_GET_BUFFER)(this, frames, data) };
    out_stats(|s| {
        s.asked_max = s.asked_max.max(frames);
        s.refused += u64::from(hr < 0);
    });
    hr
}

extern "system" fn patched_padding(this: Com, padding: *mut u32) -> HRESULT {
    // SAFETY: the caller's arguments.
    let hr = unsafe { host_fn::<PaddingFn>(P_PADDING)(this, padding) };
    if hr >= 0 && !padding.is_null() {
        // SAFETY: just written by the host.
        let p = unsafe { *padding };
        out_stats(|s| {
            s.pad_min = s.pad_min.min(p);
            s.pad_max = s.pad_max.max(p);
            s.pad_zero += u64::from(p == 0);
            s.pad_calls += 1;
        });
    }
    hr
}

extern "system" fn patched_initialize(this: Com, mode: u32, flags: u32, duration: i64, period: i64, format: *const c_void, session: *const [u8; 16]) -> HRESULT {
    // SAFETY: the caller's arguments.
    let hr = unsafe { host_fn::<InitializeFn>(P_INITIALIZE)(this, mode, flags, duration, period, format, session) };
    let (tag, channels, rate, bits) = if format.is_null() {
        (0, 0, 0, 0)
    } else {
        // SAFETY: a WAVEFORMATEX: tag, channels, rate, bytes/s, align, bits.
        unsafe {
            let f = format.cast::<u8>();
            (f.cast::<u16>().read_unaligned(), f.add(2).cast::<u16>().read_unaligned(), f.add(4).cast::<u32>().read_unaligned(), f.add(14).cast::<u16>().read_unaligned())
        }
    };
    xnw_core::info!(
        "audio: IAudioClient::Initialize(mode {mode}, flags {flags:#x}, buffer {:.1} ms, period {:.1} ms, format tag {tag:#x} {channels} ch {rate} Hz {bits} bit) = {:#x} from {}",
        duration as f64 / 10_000.0,
        period as f64 / 10_000.0,
        hr as u32,
        caller()
    );
    hr
}

extern "system" fn patched_set_event(this: Com, event: *mut c_void) -> HRESULT {
    xnw_core::info!("audio: IAudioClient::SetEventHandle from {} (the title's mixer is event driven)", caller());
    // SAFETY: the caller's arguments.
    unsafe { host_fn::<EventFn>(P_EVENT)(this, event) }
}

/// `XNW_TRACE=audio`: what the title hands WASAPI, logged once a second.
struct OutStats {
    since: std::time::Instant,
    last: Option<std::time::Instant>,
    releases: u64,
    frames: u64,
    silent: u64,
    gap_max: f64,
    gaps_over_10: u64,
    asked_max: u32,
    refused: u64,
    pad_min: u32,
    pad_max: u32,
    pad_zero: u64,
    pad_calls: u64,
}

fn out_stats(f: impl FnOnce(&mut OutStats)) {
    static STATS: std::sync::Mutex<Option<OutStats>> = std::sync::Mutex::new(None);
    if !crate::acphal::traced() {
        return;
    }
    let fresh = || OutStats {
        since: std::time::Instant::now(),
        last: None,
        releases: 0,
        frames: 0,
        silent: 0,
        gap_max: 0.0,
        gaps_over_10: 0,
        asked_max: 0,
        refused: 0,
        pad_min: u32::MAX,
        pad_max: 0,
        pad_zero: 0,
        pad_calls: 0,
    };
    let mut g = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let s = g.get_or_insert_with(fresh);
    f(s);
    let secs = s.since.elapsed().as_secs_f64();
    if secs >= 1.0 {
        xnw_core::trace!(
            "audio out: {secs:.2} s: {} releases, {} frames ({:.0}/s), {} silent; longest gap {:.1} ms, {} gaps over 10 ms; GetBuffer up to {} frames, {} refused; padding {}-{} frames, {} of {} reads empty",
            s.releases,
            s.frames,
            s.frames as f64 / secs,
            s.silent,
            s.gap_max,
            s.gaps_over_10,
            s.asked_max,
            s.refused,
            if s.pad_calls == 0 { 0 } else { s.pad_min },
            s.pad_max,
            s.pad_zero,
            s.pad_calls
        );
        let last = s.last;
        *s = fresh();
        s.last = last;
    }
}

/// Patches the host's shared `IAudioClient` and `IAudioRenderClient`
/// method tables, which every client in the process shares: a render
/// client of our own (initialised, never started) shows where they are.
/// `ReleaseBuffer` marks buffers silent with `XNW_AUDIO=off`; with
/// `XNW_TRACE=audio` the others are traced too (initialisation, event
/// mode, buffer fill and the rate the title delivers frames at).
fn patch_render_clients(trace: bool) -> Result<(), String> {
    type Create = unsafe extern "system" fn(*const [u8; 16], *mut c_void, u32, *const [u8; 16], *mut Com) -> HRESULT;
    type DefaultEndpoint = unsafe extern "system" fn(Com, u32, u32, *mut Com) -> HRESULT;
    type Activate = unsafe extern "system" fn(Com, *const [u8; 16], u32, *const c_void, *mut Com) -> HRESULT;
    type MixFormat = unsafe extern "system" fn(Com, *mut *mut c_void) -> HRESULT;
    type GetService = unsafe extern "system" fn(Com, *const [u8; 16], *mut Com) -> HRESULT;
    type Release = unsafe extern "system" fn(Com) -> u32;
    type Free = unsafe extern "system" fn(*mut c_void);
    let create = host!("combase.dll", "CoCreateInstance": Create);
    let free = host!("combase.dll", "CoTaskMemFree": Free);
    // SAFETY: COM calls on the interfaces they return, on the title's
    // thread, which has COM initialised (it is calling CoCreateInstance);
    // the patched slots keep their signatures.
    unsafe {
        let mut enumerator: Com = std::ptr::null_mut();
        check("MMDeviceEnumerator", create(&CLSID_MM_DEVICE_ENUMERATOR, std::ptr::null_mut(), CLSCTX_ALL, &IID_IMM_DEVICE_ENUMERATOR, &mut enumerator))?;
        let mut device: Com = std::ptr::null_mut();
        check("GetDefaultAudioEndpoint", method::<DefaultEndpoint>(enumerator, 4)(enumerator, 0, 0, &mut device))?;
        let mut client: Com = std::ptr::null_mut();
        check("IMMDevice::Activate", method::<Activate>(device, 3)(device, &IID_IAUDIO_CLIENT, CLSCTX_ALL, std::ptr::null(), &mut client))?;
        let mut format = std::ptr::null_mut();
        check("GetMixFormat", method::<MixFormat>(client, 8)(client, &mut format))?;
        let hr = method::<InitializeFn>(client, 3)(client, 0, 0, 1_000_000, 0, format, std::ptr::null());
        free(format);
        check("IAudioClient::Initialize", hr)?;
        let mut render: Com = std::ptr::null_mut();
        check("GetService(IAudioRenderClient)", method::<GetService>(client, 14)(client, &IID_IAUDIO_RENDER_CLIENT, &mut render))?;
        let mut patches: Vec<(Com, usize, usize, usize)> = vec![(render, 4, patched_release_buffer as usize, P_RELEASE)];
        if trace {
            patches.extend([
                (render, 3, patched_get_buffer as usize, P_GET_BUFFER),
                (client, 3, patched_initialize as usize, P_INITIALIZE),
                (client, 6, patched_padding as usize, P_PADDING),
                (client, 13, patched_set_event as usize, P_EVENT),
                (device, 3, patched_activate as usize, P_ACTIVATE),
            ]);
        }
        for (object, index, to, patch) in patches {
            let slot = (*object).add(index).cast_mut();
            let _ = HOST[patch].set(*slot);
            let mut old = 0;
            if xnw_core::win::VirtualProtect(slot.cast(), 8, xnw_core::win::PAGE_READWRITE, &mut old) == 0 {
                return Err(format!("VirtualProtect: {}", xnw_core::win::last_error()));
            }
            slot.write(to);
            xnw_core::win::VirtualProtect(slot.cast(), 8, old, &mut old);
        }
        // Our objects go; the method tables stay with their DLL.
        for p in [render, client, device, enumerator] {
            method::<Release>(p, 2)(p);
        }
    }
    Ok(())
}

/// `{BCDE0395-E52F-467C-8E3D-C4579291692E}`, as bytes in memory order.
const CLSID_MM_DEVICE_ENUMERATOR: [u8; 16] = guid(0xBCDE_0395, 0xE52F, 0x467C, [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E]);

pub(crate) const fn guid(a: u32, b: u16, c: u16, d: [u8; 8]) -> [u8; 16] {
    let (a, b, c) = (a.to_le_bytes(), b.to_le_bytes(), c.to_le_bytes());
    [a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]
}

/// False with `XNW_AUDIO=off`.
pub fn enabled() -> bool {
    !std::env::var("XNW_AUDIO").is_ok_and(|v| v.eq_ignore_ascii_case("off") || v == "0")
}

/// Gives the process the console's millisecond timers. The title paces
/// itself with short waits (its audio mixer sleeps about 4 ms per 5.3 ms
/// block; Bink's file reads retry after `Sleep(5)`) and cannot tune the
/// timer on the console; a desktop process waits in 15.6 ms ticks unless it
/// asks for finer ones, so the mixer falls behind real time (slow, broken
/// sound) and every short wait becomes a 15.6 ms one. Windows 11 drops the
/// request while the window is hidden unless the process opts out.
/// Process-wide, for the whole run; logs the resolution before and after.
/// `XNW_TIMERS=off` leaves the host's resolution (for comparison runs).
///
/// Ported from VIctory `crates/victory-hle/src/runtime.rs`
/// `use_console_timers` at ffe310e (GPL-3.0-or-later), lines 175-207;
/// declarations by hand in `xnw_core::win` instead of `windows-sys`.
pub fn use_console_timers() {
    use xnw_core::win;
    let resolution = || {
        let (mut coarse, mut fine, mut now) = (0, 0, 0);
        // SAFETY: three out-parameters.
        unsafe { win::NtQueryTimerResolution(&mut coarse, &mut fine, &mut now) };
        f64::from(now) / 10_000.0
    };
    let before = resolution();
    if std::env::var("XNW_TIMERS").is_ok_and(|v| v == "off" || v == "0") {
        xnw_core::info!("timers: XNW_TIMERS=off: resolution left at {before:.3} ms");
        return;
    }
    // Also out of the system's efficiency mode (lower clocks, a hybrid
    // processor's slow cores) while our window is not in front: the console
    // has eight cores alike and no such mode. `XNW_EFFICIENCY_MODE=1` leaves
    // the system its choice. (victory-fork a2537066, in VIctory-7900XTX
    // 51d3f4d, GPL-3.0-or-later.)
    let speed = if std::env::var("XNW_EFFICIENCY_MODE").is_ok_and(|v| v == "1") { 0 } else { win::PROCESS_POWER_THROTTLING_EXECUTION_SPEED };
    let keep = win::ProcessPowerThrottlingState { version: 1, control: win::PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION | speed, state: 0 };
    // SAFETY: plain process settings; `keep` outlives the call.
    let r = unsafe {
        win::SetProcessInformation(win::GetCurrentProcess(), win::PROCESS_POWER_THROTTLING, (&raw const keep).cast(), size_of_val(&keep) as u32);
        win::timeBeginPeriod(1)
    };
    xnw_core::info!("timers: resolution {before:.3} ms -> {:.3} ms (timeBeginPeriod(1) = {r}); efficiency mode {}", resolution(), if speed == 0 { "left to the system (XNW_EFFICIENCY_MODE=1)" } else { "off" });
}

type CoCreateInstanceFn =unsafe extern "system" fn(*const [u8; 16], *mut c_void, u32, *const [u8; 16], *mut *mut c_void) -> HRESULT;

pub extern "system" fn CoCreateInstance(clsid: *const [u8; 16], outer: *mut c_void, context: u32, iid: *const [u8; 16], out: *mut *mut c_void) -> HRESULT {
    // SAFETY: COM callers pass a valid class id.
    if !clsid.is_null() && unsafe { *clsid } == CLSID_MM_DEVICE_ENUMERATOR {
        static SAID: AtomicBool = AtomicBool::new(false);
        let first = !SAID.swap(true, Ordering::Relaxed);
        if first {
            xnw_core::info!("audio: the title opens the host's audio endpoints (MMDeviceEnumerator) from {}", caller());
            let trace = crate::acphal::traced();
            SILENT.store(!enabled(), Ordering::Relaxed);
            if !enabled() || trace {
                match patch_render_clients(trace) {
                    Ok(()) if !enabled() => xnw_core::info!("audio: XNW_AUDIO=off: every buffer the title plays is marked silent"),
                    Ok(()) => {}
                    Err(e) => xnw_core::warn!("audio: could not patch the host's render clients: {e}"),
                }
            }
            crate::audio_capture::start();
        }
    }
    let host = host!("combase.dll", "CoCreateInstance": CoCreateInstanceFn);
    // SAFETY: the caller's arguments, passed on as they came.
    unsafe { host(clsid, outer, context, iid, out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_id_bytes_are_in_memory_order() {
        assert_eq!(&CLSID_MM_DEVICE_ENUMERATOR[..4], &[0x95, 0x03, 0xDE, 0xBC]);
        assert_eq!(&CLSID_MM_DEVICE_ENUMERATOR[4..8], &[0x2F, 0xE5, 0x7C, 0x46]);
    }
}
