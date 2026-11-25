//! Where each imported module is served from. This table is the single
//! place that decides it; the loader will bind from the same table.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Route {
    /// Console-only API: we implement it (xnw-hle / xnw-gpu).
    Hle,
    /// DLL the title ships beside its executable; mapped like the executable.
    Title,
    /// Exists on Windows with the same contract; bound to the host's own.
    Host,
    /// Exists on Windows, but calls must pass through us first (paths,
    /// windows, memory, threads, network policy).
    HostFiltered,
    Unknown,
}

impl Route {
    pub fn label(self) -> &'static str {
        match self {
            Route::Hle => "hle",
            Route::Title => "title-dll",
            Route::Host => "host",
            Route::HostFiltered => "host+filter",
            Route::Unknown => "UNKNOWN",
        }
    }
}

pub(crate) const HLE: &[&str] = &[
    "d3d12_xs.dll",
    "d3d12_x.dll",
    "xg_xs.dll",
    "xmem.dll",
    "xapu.dll",
    "acphal.dll",
    "chad.dll",
    "xcrdapi.dll",
    "gameinput.dll",
    "xgameruntime.dll",
    "kernelx.dll",
    "api-ms-win-core-psm-appnotify-l1-1-0.dll",
    "api-ms-win-core-psm-appnotify-l1-1-1.dll",
];

const TITLE: &[&str] = &[
    "bink2scarlett.dll",
    "xboxbvhbuilder.dll",
    "amd_fidelityfx_loader_xbox.dll",
    "amd_fidelityfx_upscaler_xbox.dll",
    "libhttpclient.gdk.dll",
];

/// Host modules whose calls we must see before Windows does.
const FILTERED: &[&str] = &[
    "api-ms-win-core-file-",
    "api-ms-win-core-memory-",
    "api-ms-win-core-libraryloader-",
    "api-ms-win-core-processthreads-",
    "api-ms-win-core-processtopology-",
    "api-ms-win-core-namedpipe-",
    "api-ms-win-crt-filesystem-",
    "api-ms-win-crt-stdio-",
    "ext-ms-win-rtcore-ntuser-",
    "ws2_32.dll",
    "winhttp.dll",
    "advapi32.dll",
    "mfplat.dll",
    "mfreadwrite.dll",
];

const HOST: &[&str] = &[
    "api-ms-win-",
    "msvcp140",
    "vcruntime140",
    "bcrypt.dll",
    "crypt32.dll",
    "xaudio2_9.dll",
    "kernel32.dll",
    "ntdll.dll",
];

pub fn route(dll: &str) -> Route {
    let d = dll.to_ascii_lowercase();
    if HLE.contains(&d.as_str()) {
        Route::Hle
    } else if TITLE.contains(&d.as_str()) {
        Route::Title
    } else if FILTERED.iter().any(|p| d.starts_with(p)) {
        Route::HostFiltered
    } else if HOST.iter().any(|p| d.starts_with(p)) {
        Route::Host
    } else {
        Route::Unknown
    }
}
