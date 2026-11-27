//! Console APIs, implemented on Windows.
//!
//! [`route`] says who serves each imported DLL; [`lookup`] hands out our
//! implementations. Functions here are plain `extern "system"` functions
//! with the console's signatures, listed in one table.

#![allow(non_snake_case)]
// Title code calls these through its import table with pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

/// The host's `dll!name` as type `$ty`, found once per call site: what an
/// import of it would have been bound to.
macro_rules! host {
    ($dll:literal, $name:literal: $ty:ty) => {{
        static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let f = *F.get_or_init(|| {
            // SAFETY: a system DLL and NUL-terminated names.
            unsafe {
                xnw_core::win::GetProcAddress(
                    xnw_core::win::LoadLibraryA(concat!($dll, "\0").as_ptr()),
                    concat!($name, "\0").as_ptr(),
                ) as usize
            }
        });
        assert!(f != 0, concat!($dll, "!", $name, " not found"));
        // SAFETY: the export has this signature.
        unsafe { std::mem::transmute::<usize, $ty>(f) }
    }};
}

mod acphal;
mod appnotify;
mod audio;
mod audio_capture;
mod chad;
mod d3d12x;
mod dstorage;
mod files;
mod gameinput;
mod images;
mod libloader;
mod net;
mod registry;
mod texbypass;
pub mod route;
mod xg;
mod xgameruntime;
mod window;
mod xapu;
mod sampler;
mod threads;
mod waits;
mod walks;
mod xmem;

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub use audio::use_console_timers;

/// Reads every switch the thread, wait and sampler shims use, once,
/// before the title runs: a shim must not do an environment lookup (it
/// sets the thread's last error) after its host call.
pub fn read_shim_switches() {
    threads::read_switches();
    waits::read_switches();
    walks::read_switches();
    sampler::ensure();
}
pub use audio_capture::{child as audio_capture_child, level as pcm_level};
pub use files::{set_game_drive, set_overlay};
/// Watched title values for the state-keyed `XNW_KEYS` forms.
pub use gameinput::note_watch;
pub use images::{Diagnostics, add_image, forget_image, set_diagnostics};
pub use libloader::{TitleLoad, set_thread_calls, set_title_loader};
pub use route::{Route, route};
pub use window::read_switches as read_window_switches;
pub use xgameruntime::xgamesave::saves_folder;
pub use xnw_gpu::d3d12x::console::Console;
/// Title memory through our own xmem outside the title (the burst's stand-in, ZC1).
pub mod title_memory {
    pub use crate::xmem::{XMemAllocatePhysicalPages, XMemMapPhysicalPages, backing, physical_window, replay_reservation};
}

/// Chooses the console the title sees (registry, XSystem, DXGI, XMem).
/// Call before any title code runs.
pub fn set_console(console: Console) {
    xnw_gpu::d3d12x::console::set(console);
}

static TITLE_BASE: AtomicUsize = AtomicUsize::new(0);

/// Tells the runtime where the title's executable is mapped. Call before
/// any title code runs.
pub fn set_title(base: usize) {
    TITLE_BASE.store(base, Ordering::Relaxed);
}

pub(crate) fn title_base() -> usize {
    TITLE_BASE.load(Ordering::Relaxed)
}

static TITLE_ID: AtomicU32 = AtomicU32::new(0);

/// Reads what the runtime needs from the title's manifest
/// (`MicrosoftGame.config` in `game`): its title id. A missing manifest
/// leaves id 0, logged.
pub fn set_game_dir(game: &std::path::Path) {
    let text = std::fs::read_to_string(game.join("MicrosoftGame.config")).unwrap_or_default();
    let id = text
        .split_once("<TitleId>")
        .and_then(|(_, rest)| rest.split_once('<'))
        .and_then(|(id, _)| u32::from_str_radix(id.trim(), 16).ok());
    match id {
        Some(id) => xnw_core::info!("title id {id:08X} (MicrosoftGame.config)"),
        None => xnw_core::warn!("no title id: {} has no MicrosoftGame.config with a TitleId", game.display()),
    }
    TITLE_ID.store(id.unwrap_or(0), Ordering::Relaxed);
    // Persistent local storage quota (VIctory-base runtime.rs 99).
    let quota = text
        .split_once("<PersistentLocalStorage>")
        .and_then(|(_, rest)| rest.split_once("<SizeMB>"))
        .and_then(|(_, rest)| rest.split_once('<'))
        .and_then(|(mb, _)| mb.trim().parse().ok());
    xgameruntime::storage::set_quota_mb(quota.unwrap_or(0));
}

pub(crate) fn title_id() -> u32 {
    TITLE_ID.load(Ordering::Relaxed)
}

macro_rules! exports {
    ($($dll:literal => [$($path:path),* $(,)?]),* $(,)?) => {
        /// `(dll, symbol, address)` of everything implemented.
        fn table() -> &'static [(&'static str, &'static str, usize)] {
            static T: std::sync::OnceLock<Vec<(&'static str, &'static str, usize)>> = std::sync::OnceLock::new();
            T.get_or_init(|| vec![$($((
                $dll,
                { let p = stringify!($path); p.rsplit("::").next().unwrap_or(p) },
                $path as usize,
            ),)*)*])
        }
    };
}

exports! {
    "xmem.dll" => [
        xmem::XMemVirtualAlloc,
        xmem::XMemVirtualQuery,
        xmem::XMemAlloc,
        xmem::XMemFree,
        xmem::XMemAllocDefault,
        xmem::XMemFreeDefault,
        xmem::XMemIsToolsMemAvailable,
        xmem::XMemAllocatePhysicalPages,
        xmem::XMemFreePhysicalPages,
        xmem::XMemMapPhysicalPages,
        xmem::XMemMapPhysicalPagesScatter,
        xmem::XMemGetWorkingSetStatistics,
        xmem::XMemTransferMemory,
    ],
    "d3d12_xs.dll" => [
        d3d12x::D3D12ConfigureVirtualMemoryX,
        d3d12x::D3D12XboxCreateDevice,
        d3d12x::D3D12SerializeRootSignature,
        d3d12x::D3D12SerializeVersionedRootSignature,
        d3d12x::D3D12CreateRootSignatureDeserializer,
    ],
    "xg_xs.dll" => [
        xg::XGCreateTextureComputer,
        xg::XGSuggestSwizzleMode,
    ],
    "acphal.dll" => [
        acphal::AcpHalAllocateShapeContexts,
        acphal::AcpHalReleaseShapeContexts,
        acphal::AcpHalCreate,
        acphal::ApuAlloc,
        acphal::ApuCreateHeap,
        acphal::ApuFree,
        acphal::ApuIsVirtualAddressValid,
        acphal::ApuMapApuAddress,
        acphal::ApuMapVirtualAddress,
    ],
    // Host COM: only CoCreateInstance, for the audio endpoint (`audio`).
    "combase.dll" => [
        audio::CoCreateInstance,
    ],
    "xapu.dll" => [
        xapu::XApuConnect,
        xapu::XApuDisconnect,
        xapu::XApuEnqueueCommand,
        xapu::XApuDequeueResult,
        xapu::XApuGetResultCount,
    ],
    "chad.dll" => [
        chad::XDspConnectWithMaximumStreamLimit,
        chad::XDspDisconnect,
        chad::XDspActivate,
        chad::XDspDeactivate,
        chad::XDspSubmitCommand,
    ],
    "xcrdapi.dll" => [
        dstorage::DStorageGetFactory,
    ],
    "gameinput.dll" => [
        gameinput::GameInputCreate,
    ],
    "api-ms-win-core-psm-appnotify-l1-1-0.dll" => [
        appnotify::RegisterAppStateChangeNotification,
        appnotify::UnregisterAppStateChangeNotification,
    ],
    "api-ms-win-core-psm-appnotify-l1-1-1.dll" => [
        appnotify::RegisterAppConstrainedChangeNotification,
        appnotify::UnregisterAppConstrainedChangeNotification,
    ],
    "xgameruntime.dll" => [
        xgameruntime::QueryApiImpl,
        xgameruntime::InitializeApiImpl,
        xgameruntime::InitializeApiImplEx,
        xgameruntime::InitializeApiImplEx2,
        xgameruntime::XErrorReport,
        xgameruntime::UninitializeApiImpl,
    ],
    // Host DLLs whose ways off the machine we vet (see [`host_override`]).
    "ws2_32.dll" => [
        net::connect,
        net::WSAConnect,
        net::WSAConnectByNameA,
        net::WSAConnectByNameW,
        net::WSAConnectByList,
        net::WSAJoinLeaf,
        net::WSAIoctl,
        net::sendto,
        net::WSASendTo,
        net::WSASendMsg,
        net::bind,
        net::getaddrinfo,
        net::GetAddrInfoW,
        net::GetAddrInfoExA,
        net::GetAddrInfoExW,
        net::gethostbyname,
        net::WSAAsyncGetHostByName,
        net::WSALookupServiceBeginA,
        net::WSALookupServiceBeginW,
        net::gethostbyaddr,
        net::WSAAsyncGetHostByAddr,
        net::getnameinfo,
        net::GetNameInfoW,
    ],
    "winhttp.dll" => [
        net::WinHttpConnect,
        net::WinHttpOpen,
        net::WinHttpSetOption,
        net::WinHttpDetectAutoProxyConfigUrl,
        net::WinHttpGetProxyForUrl,
        net::WinHttpGetProxyForUrlEx,
        net::WinHttpGetProxyForUrlEx2,
        net::WinHttpGetProxySettingsEx,
    ],
    // The registry: the console's `ConsoleMode` only (see [`host_override`]).
    "advapi32.dll" => [
        registry::RegOpenKeyW,
        registry::RegOpenKeyExW,
        registry::RegQueryValueExW,
        registry::RegCloseKey,
    ],
    // Not a console DLL: our overrides of host functions, bound in place of
    // the host's for kernel32, kernelbase and the library-loader API sets
    // (see [`host_override`]).
    "kernel32.dll" => [
        libloader::LoadLibraryExA,
        libloader::LoadLibraryExW,
        libloader::LoadLibraryA,
        libloader::LoadLibraryW,
        libloader::GetModuleHandleA,
        libloader::GetModuleHandleW,
        libloader::GetProcAddress,
        libloader::FreeLibrary,
        libloader::DisableThreadLibraryCalls,
        files::CreateFileW,
        files::CreateDirectoryW,
        files::DeleteFileW,
        files::RemoveDirectoryW,
        files::SetFileAttributesW,
        files::MoveFileExW,
        files::FindFirstFileW,
        files::GetFileAttributesW,
        files::GetFileAttributesA,
        files::GetFileAttributesExW,
        files::GetModuleFileNameW,
        files::GetModuleFileNameA,
        files::ReadFile,
        xmem::title::VirtualAlloc,
        xmem::title::VirtualFree,
        // The title's blocking waits (timed with XNW_WAIT_STATS=1).
        waits::WaitForSingleObject,
        waits::WaitForSingleObjectEx,
        waits::WaitForMultipleObjects,
        waits::WaitForMultipleObjectsEx,
        waits::Sleep,
        waits::SleepEx,
        waits::SleepConditionVariableSRW,
        waits::SleepConditionVariableCS,
        waits::WaitOnAddress,
        waits::InitializeCriticalSectionAndSpinCount,
        // Thread names, affinity, the processor count.
        threads::CreateThread,
        threads::SetThreadPriority,
        threads::SetThreadGroupAffinity,
        threads::GetLogicalProcessorInformation,
        // The title's own stack walks and memory queries.
        walks::RtlLookupFunctionEntry,
        walks::RtlVirtualUnwind,
        walks::VirtualQueryEx,
    ],
    // Not a console DLL: the title's window, subclassed (see [`host_override`]).
    "user32.dll" => [
        window::CreateWindowExA,
        window::CreateWindowExW,
        window::ShowWindow,
    ],
    // Not a console DLL either: the CRT's path functions (see [`host_override`]).
    "ucrtbase.dll" => [
        files::_wfopen,
        files::fopen,
        files::_wstat64,
        files::remove,
        files::fread,
    ],
}

/// Console exports titles import by ordinal: `(dll, #ordinal, name)`.
const ORDINALS: &[(&str, &str, &str)] = &[("gameinput.dll", "#1", "GameInputCreate")];

/// Our implementation of `dll!symbol` (a name or `#ordinal`), if there is one.
pub fn lookup(dll: &str, symbol: &str) -> Option<usize> {
    let symbol = ORDINALS.iter().find(|(d, o, _)| d.eq_ignore_ascii_case(dll) && *o == symbol).map_or(symbol, |t| t.2);
    table().iter().find(|(d, s, _)| d.eq_ignore_ascii_case(dll) && *s == symbol).map(|t| t.2)
}

/// Our override of a host function, if we must see its calls first:
/// `dll` is a host DLL (by import or a module's file name), `symbol` a name
/// or `#ordinal`.
pub fn host_override(dll: &str, symbol: &str) -> Option<usize> {
    let d = dll.to_ascii_lowercase();
    if d == "kernel32.dll"
        || d == "kernelbase.dll"
        || d.starts_with("api-ms-win-core-libraryloader-")
        || d.starts_with("api-ms-win-core-file-")
        || d.starts_with("api-ms-win-core-memory-")
        || d.starts_with("api-ms-win-core-synch-")
        || d.starts_with("api-ms-win-core-processthreads-")
        || d.starts_with("api-ms-win-core-processtopology-")
        || d.starts_with("api-ms-win-core-sysinfo-")
        || d.starts_with("api-ms-win-core-rtlsupport-")
    {
        let r = lookup("kernel32.dll", symbol).or_else(|| if d == "kernelbase.dll" { lookup("advapi32.dll", symbol) } else { None });
        if r.is_some() {
            walks::note_bound(&d, symbol);
        }
        r
    } else if d == "advapi32.dll" || d.starts_with("api-ms-win-core-registry-") {
        lookup("advapi32.dll", symbol)
    } else if d == "ucrtbase.dll" || d.starts_with("api-ms-win-crt-stdio-") || d.starts_with("api-ms-win-crt-filesystem-") {
        lookup("ucrtbase.dll", symbol)
    } else if d == "ws2_32.dll" {
        lookup(&d, net::ordinal_name(symbol).unwrap_or(symbol))
    } else if d == "user32.dll" || d.starts_with("ext-ms-win-rtcore-ntuser-window-") {
        lookup("user32.dll", symbol)
    } else if d.starts_with("api-ms-win-core-com-") || d == "combase.dll" || d == "ole32.dll" {
        lookup("combase.dll", symbol)
    } else if d == "winhttp.dll" {
        lookup(&d, symbol)
    } else {
        None
    }
}

/// How many functions are implemented, for reports.
pub fn implemented() -> usize {
    table().len()
}
