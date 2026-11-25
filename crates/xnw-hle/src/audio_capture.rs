//! Debugging aid: `XNW_AUDIO_CAPTURE=<file.wav>` records what the title
//! plays, 48 kHz stereo 16-bit, through process loopback: our own process
//! tree only, never the rest of the host's mix. Stretches where nothing was
//! played are written as silence, so the file keeps real time.
//!
//! The recorder runs in a child copy of this executable: an audio client
//! in the title's process raises that process's timer resolution, which
//! changes how the title's own waits behave.
//!
//! Ported from VIctory, `crates/victory-hle/src/modules/audio/capture.rs`
//! at 5d0516e (xnw/base; GPL-3.0-or-later), lines 1-229, structure and
//! names kept. Changes: the COM interfaces are declared by hand (slots of
//! `IAudioClient`, `IAudioCaptureClient`,
//! `IActivateAudioInterfaceAsyncOperation`, and a static completion
//! handler) instead of the `windows` crate and its `implement` macro;
//! `VICTORY_AUDIO_CAPTURE` is `XNW_AUDIO_CAPTURE`; the recorder starts when
//! the title first opens the host's audio endpoints ([`crate::audio`]),
//! not from the run's diagnostics; the timing probe logs with
//! `XNW_TRACE=audio`; at the end it logs the file's peak and RMS.

use std::ffi::c_void;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use xnw_core::win::{OpenProcess, PROCESS_SYNCHRONIZE, WAIT_OBJECT_0, WaitForSingleObject};

use crate::audio::{Com, check, guid, method};
use crate::xgameruntime::{HRESULT, S_OK};

const RATE: u32 = 48_000;
const FRAME_BYTES: usize = 4;

const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;

const IID_IUNKNOWN: [u8; 16] = guid(0, 0, 0, [0xC0, 0, 0, 0, 0, 0, 0, 0x46]);
const IID_IAGILE_OBJECT: [u8; 16] = guid(0x94EA_2B94, 0xE9CC, 0x49E0, [0xC0, 0xFF, 0xEE, 0x64, 0xCA, 0x8F, 0x5B, 0x90]);
const IID_IACTIVATE_AUDIO_INTERFACE_COMPLETION_HANDLER: [u8; 16] =
    guid(0x41D9_49AB, 0x9862, 0x444A, [0x80, 0xF6, 0xC2, 0x61, 0x33, 0x4D, 0xA5, 0xEB]);
const IID_IAUDIO_CLIENT: [u8; 16] = guid(0x1CB9_AD4C, 0xDBFA, 0x4C32, [0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2]);
const IID_IAUDIO_CAPTURE_CLIENT: [u8; 16] = guid(0xC8AD_BD64, 0xE71E, 0x48A0, [0xA4, 0xDE, 0x18, 0x5C, 0x39, 0x5C, 0xD3, 0x17]);

const AUDCLNT_SHAREMODE_SHARED: u32 = 0;
const AUDCLNT_STREAMFLAGS_LOOPBACK: u32 = 0x0002_0000;
const AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM: u32 = 0x8000_0000;
const AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY: u32 = 1;
const AUDCLNT_BUFFERFLAGS_SILENT: u32 = 2;
const AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK: u32 = 1;
const PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE: u32 = 0;
const WAVE_FORMAT_PCM: u16 = 1;
const VT_BLOB: u16 = 0x41;

#[repr(C, packed)]
struct WaveFormatEx {
    format: u16,
    channels: u16,
    rate: u32,
    bytes_per_second: u32,
    block_align: u16,
    bits: u16,
    extra: u16,
}

#[repr(C)]
struct ActivationParams {
    kind: u32,
    target_process: u32,
    mode: u32,
}

/// Activation completes asynchronously; the recording thread polls for it.
/// One static object: its reference count does not matter.
#[repr(C)]
struct Activated {
    vtable: &'static ActivatedMethods,
}

#[repr(C)]
struct ActivatedMethods {
    query_interface: extern "system" fn(*mut c_void, *const [u8; 16], *mut *mut c_void) -> HRESULT,
    add_ref: extern "system" fn(*mut c_void) -> u32,
    release: extern "system" fn(*mut c_void) -> u32,
    activate_completed: extern "system" fn(*mut c_void, *mut c_void) -> HRESULT,
}

extern "system" fn query_interface(this: *mut c_void, iid: *const [u8; 16], out: *mut *mut c_void) -> HRESULT {
    // SAFETY: COM callers pass an interface id and room for the answer.
    let iid = unsafe { *iid };
    let known = [IID_IUNKNOWN, IID_IAGILE_OBJECT, IID_IACTIVATE_AUDIO_INTERFACE_COMPLETION_HANDLER].contains(&iid);
    // SAFETY: as above.
    unsafe { out.write(if known { this } else { std::ptr::null_mut() }) };
    if known { S_OK } else { E_NOINTERFACE }
}

extern "system" fn add_ref(_: *mut c_void) -> u32 {
    1
}

extern "system" fn activate_completed(_: *mut c_void, _: *mut c_void) -> HRESULT {
    S_OK
}

static ACTIVATED_METHODS: ActivatedMethods = ActivatedMethods { query_interface, add_ref, release: add_ref, activate_completed };

/// Names the process to record, in the recorder's environment.
const TARGET: &str = "XNW_AUDIO_CAPTURE_OF";

/// Starts the recorder if `XNW_AUDIO_CAPTURE` names a file. With
/// `XNW_TRACE=audio`, also logs once a second what a `Sleep(1)` takes in
/// this process (the title's millisecond waits get the same).
pub fn start() {
    if crate::acphal::traced() {
        let _ = std::thread::Builder::new().name("xnw timing probe".into()).spawn(|| {
            loop {
                std::thread::sleep(Duration::from_secs(1));
                // Kernel32's Sleep, as the title calls it: Rust's
                // `thread::sleep` uses a high-resolution timer, which
                // reported 1-2 ms while the title's waits took 15.6 ms.
                let at = Instant::now();
                // SAFETY: a plain wait.
                unsafe { xnw_core::win::Sleep(1) };
                let ms = at.elapsed().as_secs_f32() * 1000.0;
                let at = Instant::now();
                std::thread::sleep(Duration::from_millis(1));
                let high = at.elapsed().as_secs_f32() * 1000.0;
                xnw_core::trace!("Sleep(1) took {ms:.2} ms (high-resolution timer wait {high:.2} ms)");
                thread_census();
            }
        });
    }
    let Some(path) = std::env::var_os("XNW_AUDIO_CAPTURE") else {
        return;
    };
    match std::env::current_exe().and_then(|exe| std::process::Command::new(exe).env(TARGET, std::process::id().to_string()).spawn()) {
        Ok(_) => xnw_core::info!("audio capture: recording this process tree to {}", Path::new(&path).display()),
        Err(e) => xnw_core::warn!("audio capture not started: {e}"),
    }
}

/// `XNW_TRACE=audio`, every 5 s: the process's threads that used
/// more than 5 % of a CPU since the last census, with their priority,
/// affinity mask, start address and name, busiest first (who competes with
/// the title's mixer for a core).
fn thread_census() {
    use std::collections::HashMap;
    use xnw_core::win::*;
    static LAST: std::sync::Mutex<Option<(Instant, HashMap<u32, u64>, u32)>> = std::sync::Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let (since, before, count) = last.get_or_insert_with(|| (Instant::now(), HashMap::new(), 0));
    *count += 1;
    if *count % 5 != 0 {
        return;
    }
    let wall = since.elapsed().as_secs_f64() * 1e7;
    let mut now = HashMap::new();
    let mut busy = Vec::new();
    // SAFETY: no arguments.
    let me = unsafe { GetCurrentProcessId() };
    // SAFETY: a thread snapshot walked with a sized entry; each thread is
    // opened for queries only and closed; out-parameters sized as declared.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return;
        }
        let mut entry = ThreadEntry32 { size: size_of::<ThreadEntry32>() as u32, usage: 0, thread_id: 0, owner_process_id: 0, base_priority: 0, delta_priority: 0, flags: 0 };
        let mut more = Thread32First(snapshot, &mut entry) != 0;
        while more {
            if entry.owner_process_id == me {
                let id = entry.thread_id;
                let thread = OpenThread(THREAD_QUERY_INFORMATION, 0, id);
                if !thread.is_null() {
                    let (mut c, mut e, mut k, mut u) = (0, 0, 0, 0);
                    GetThreadTimes(thread, &mut c, &mut e, &mut k, &mut u);
                    now.insert(id, k + u);
                    let used = (k + u).saturating_sub(before.get(&id).copied().unwrap_or(0)) as f64 / wall * 100.0;
                    if used > 5.0 && !before.is_empty() {
                        // ThreadBasicInformation: exit status, TEB, client id, affinity, priority, base priority.
                        let mut basic = [0usize; 6];
                        NtQueryInformationThread(thread, 0, basic.as_mut_ptr().cast(), size_of_val(&basic) as u32, std::ptr::null_mut());
                        let mut start = 0usize;
                        NtQueryInformationThread(thread, THREAD_QUERY_SET_WIN32_START_ADDRESS, (&raw mut start).cast(), 8, std::ptr::null_mut());
                        let mut name = std::ptr::null_mut();
                        let mut text = String::new();
                        if GetThreadDescription(thread, &mut name) >= 0 && !name.is_null() {
                            let len = (0..256).take_while(|&i| *name.add(i) != 0).count();
                            text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
                            LocalFree(name.cast());
                        }
                        busy.push((used, id, GetThreadPriority(thread), basic[4], crate::images::locate(start), text));
                    }
                    CloseHandle(thread);
                }
            }
            more = Thread32Next(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    busy.sort_by(|a, b| b.0.total_cmp(&a.0));
    let lines: Vec<String> = busy.iter().map(|(used, id, prio, mask, start, name)| format!("{id} {used:.0}% prio {prio} mask {mask:#x} {start} {name:?}")).collect();
    if !lines.is_empty() {
        xnw_core::trace!("threads over 5% of a CPU in {:.1} s: {}", wall / 1e7, lines.join("; "));
    }
    *before = now;
    *since = Instant::now();
}

/// In the recorder (a child started by [`start`]): records until the
/// target exits and returns true; false in any other process.
pub fn child() -> bool {
    let (Some(pid), Some(path)) = (std::env::var(TARGET).ok().and_then(|p| p.parse().ok()), std::env::var_os("XNW_AUDIO_CAPTURE")) else {
        return false;
    };
    if let Err(e) = record(pid, Path::new(&path)) {
        xnw_core::warn!("audio capture stopped: {e}");
    }
    true
}

fn header(frames: u64) -> [u8; 44] {
    let data = u32::try_from(frames * FRAME_BYTES as u64).unwrap_or(u32::MAX - 36);
    let mut h = [0u8; 44];
    let fields: [(&[u8], usize); 11] = [
        (b"RIFF", 0),
        (&(36 + data).to_le_bytes(), 4),
        (b"WAVEfmt ", 8),
        (&16u32.to_le_bytes(), 16),
        (&1u16.to_le_bytes(), 20),
        (&2u16.to_le_bytes(), 22),
        (&RATE.to_le_bytes(), 24),
        (&(RATE * FRAME_BYTES as u32).to_le_bytes(), 28),
        (&(FRAME_BYTES as u16).to_le_bytes(), 32),
        (&16u16.to_le_bytes(), 34),
        (b"data", 36),
    ];
    for (bytes, at) in fields {
        h[at..at + bytes.len()].copy_from_slice(bytes);
    }
    h[40..].copy_from_slice(&data.to_le_bytes());
    h
}

/// Peak and RMS of 16-bit samples, as fractions of full scale.
pub fn level(samples: impl Iterator<Item = i16>) -> (f64, f64, u64) {
    let (mut peak, mut sum, mut n) = (0i32, 0f64, 0u64);
    for s in samples {
        peak = peak.max(i32::from(s).abs());
        sum += f64::from(s) * f64::from(s);
        n += 1;
    }
    (f64::from(peak) / 32768.0, if n == 0 { 0.0 } else { (sum / n as f64).sqrt() / 32768.0 }, n)
}

type ActivateAudioInterfaceAsyncFn = unsafe extern "system" fn(*const u16, *const [u8; 16], *const c_void, *const Activated, *mut Com) -> HRESULT;
type CoInitializeExFn = unsafe extern "system" fn(*const c_void, u32) -> HRESULT;
type QueryInterfaceFn = unsafe extern "system" fn(Com, *const [u8; 16], *mut Com) -> HRESULT;
type GetActivateResultFn = unsafe extern "system" fn(Com, *mut HRESULT, *mut Com) -> HRESULT;
type InitializeFn = unsafe extern "system" fn(Com, u32, u32, i64, i64, *const WaveFormatEx, *const [u8; 16]) -> HRESULT;
type GetServiceFn = unsafe extern "system" fn(Com, *const [u8; 16], *mut Com) -> HRESULT;
type StartFn = unsafe extern "system" fn(Com) -> HRESULT;
type GetBufferFn = unsafe extern "system" fn(Com, *mut *mut u8, *mut u32, *mut u32, *mut u64, *mut u64) -> HRESULT;
type ReleaseBufferFn = unsafe extern "system" fn(Com, u32) -> HRESULT;
type GetNextPacketSizeFn = unsafe extern "system" fn(Com, *mut u32) -> HRESULT;

fn record(pid: u32, path: &Path) -> Result<(), String> {
    const COINIT_MULTITHREADED: u32 = 0;
    static HANDLER: Activated = Activated { vtable: &ACTIVATED_METHODS };
    let co_initialize = host!("combase.dll", "CoInitializeEx": CoInitializeExFn);
    let activate = host!("mmdevapi.dll", "ActivateAudioInterfaceAsync": ActivateAudioInterfaceAsyncFn);
    // SAFETY: COM calls with parameters built here, on interfaces the calls
    // returned; buffers are used only between `GetBuffer` and
    // `ReleaseBuffer`.
    unsafe {
        check("CoInitializeEx", co_initialize(std::ptr::null(), COINIT_MULTITHREADED))?;
        let target = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
        if target.is_null() {
            return Err(format!("OpenProcess({pid}): {}", xnw_core::win::last_error()));
        }
        let params =
            ActivationParams { kind: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK, target_process: pid, mode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE };
        /// A `PROPVARIANT` holding a blob (`VT_BLOB`).
        #[repr(C)]
        struct Blob {
            vt: u16,
            reserved: [u16; 3],
            size: u32,
            data: *const ActivationParams,
        }
        let blob = Blob { vt: VT_BLOB, reserved: [0; 3], size: size_of_val(&params) as u32, data: &params };
        let device: Vec<u16> = "VAD\\Process_Loopback\0".encode_utf16().collect();
        let mut operation: Com = std::ptr::null_mut();
        check("ActivateAudioInterfaceAsync", activate(device.as_ptr(), &IID_IAUDIO_CLIENT, (&raw const blob).cast(), &HANDLER, &mut operation))?;
        let client = loop {
            let (mut hr, mut client): (HRESULT, Com) = (0, std::ptr::null_mut());
            // Fails (`E_ILLEGAL_METHOD_CALL`) until activation completes.
            if method::<GetActivateResultFn>(operation, 3)(operation, &mut hr, &mut client) >= 0 {
                check("process loopback activation", hr)?;
                if client.is_null() {
                    return Err("no audio client".into());
                }
                let mut audio: Com = std::ptr::null_mut();
                check("IAudioClient", method::<QueryInterfaceFn>(client, 0)(client, &IID_IAUDIO_CLIENT, &mut audio))?;
                break audio;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let format = WaveFormatEx {
            format: WAVE_FORMAT_PCM,
            channels: 2,
            rate: RATE,
            bytes_per_second: RATE * FRAME_BYTES as u32,
            block_align: FRAME_BYTES as u16,
            bits: 16,
            extra: 0,
        };
        check(
            "IAudioClient::Initialize",
            method::<InitializeFn>(client, 3)(
                client,
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
                10_000_000,
                0,
                &format,
                std::ptr::null(),
            ),
        )?;
        let mut capture: Com = std::ptr::null_mut();
        check("IAudioClient::GetService", method::<GetServiceFn>(client, 14)(client, &IID_IAUDIO_CAPTURE_CLIENT, &mut capture))?;
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        file.write_all(&header(0)).map_err(|e| e.to_string())?;
        check("IAudioClient::Start", method::<StartFn>(client, 10)(client))?;
        xnw_core::info!("audio capture: recording the title's audio to {}", path.display());

        let (mut written, mut first) = (0u64, None);
        let mut patched = Instant::now();
        // Level so far, logged every 10 s and at the end (the recorder is
        // killed with the run when cargo's job ends).
        let (mut peak, mut sum, mut loud) = (0i32, 0f64, 0u64);
        let mut said = Instant::now();
        let io = |e: std::io::Error| e.to_string();
        loop {
            std::thread::sleep(Duration::from_millis(10));
            let ended = WaitForSingleObject(target, 0) == WAIT_OBJECT_0;
            loop {
                let mut size = 0;
                check("GetNextPacketSize", method::<GetNextPacketSizeFn>(capture, 5)(capture, &mut size))?;
                if size == 0 {
                    break;
                }
                let (mut data, mut frames, mut flags, mut qpc) = (std::ptr::null_mut(), 0, 0, 0);
                check("GetBuffer", method::<GetBufferFn>(capture, 3)(capture, &mut data, &mut frames, &mut flags, std::ptr::null_mut(), &mut qpc))?;
                // Where this packet belongs, from its time stamp (100 ns).
                let at = (qpc - *first.get_or_insert(qpc)) * u64::from(RATE) / 10_000_000;
                if at > written + u64::from(RATE / 500) {
                    let silence = vec![0u8; (at - written) as usize * FRAME_BYTES];
                    file.write_all(&silence).map_err(io)?;
                    written = at;
                }
                let bytes = frames as usize * FRAME_BYTES;
                if flags & AUDCLNT_BUFFERFLAGS_SILENT != 0 {
                    file.write_all(&vec![0u8; bytes]).map_err(io)?;
                } else {
                    let pcm = std::slice::from_raw_parts(data, bytes);
                    for s in pcm.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])) {
                        peak = peak.max(i32::from(s).abs());
                        sum += f64::from(s) * f64::from(s);
                        // Above -54 dBFS: more than rounding of a silent mix.
                        loud += u64::from(s.unsigned_abs() > 64);
                    }
                    file.write_all(pcm).map_err(io)?;
                }
                if flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY != 0 && written > 0 && crate::acphal::traced() {
                    xnw_core::trace!("audio capture: discontinuity");
                }
                written += u64::from(frames);
                check("ReleaseBuffer", method::<ReleaseBufferFn>(capture, 4)(capture, frames))?;
            }
            if ended || patched.elapsed() >= Duration::from_secs(1) {
                // Valid sizes at any time: runs end by being killed.
                file.seek(SeekFrom::Start(0)).map_err(io)?;
                file.write_all(&header(written)).map_err(io)?;
                file.seek(SeekFrom::End(0)).map_err(io)?;
                patched = Instant::now();
            }
            if ended || said.elapsed() >= Duration::from_secs(10) {
                said = Instant::now();
                let samples = written * 2;
                xnw_core::info!(
                    "audio capture: {:.1} s recorded, {} of {} samples above -54 dBFS, peak {:.4}, RMS {:.5} (full scale 1)",
                    written as f64 / f64::from(RATE),
                    loud,
                    samples,
                    f64::from(peak) / 32768.0,
                    if samples == 0 { 0.0 } else { (sum / samples as f64).sqrt() / 32768.0 },
                );
            }
            if ended {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_and_level() {
        let h = header(48_000);
        assert_eq!(&h[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(h[40..44].try_into().unwrap()), 48_000 * 4);
        assert_eq!(u32::from_le_bytes(h[24..28].try_into().unwrap()), RATE);
        let (peak, rms, n) = level([16384i16, -16384, 0, 0].into_iter());
        assert_eq!((peak, n), (0.5, 4));
        assert!((rms - 0.5 / 2f64.sqrt()).abs() < 1e-9);
    }
}
