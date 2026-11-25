//! The host D3D12 device. Its own crate below xnw-gpu and xnw-shader's tests (BLD2): a shader test needs a
//! device, not the D3D12.X runtime; `xnw_gpu::host` is this crate.
//!
//! Windows 10's inbox D3D12 stops at shader model 6.5. When a folder holding
//! the Agility SDK's `D3D12Core.dll` is given, the device is created through
//! `ID3D12SDKConfiguration1::CreateDeviceFactory`, which needs neither
//! Developer Mode nor `D3D12SDKVersion` exports in the executable.

use std::ffi::{CString, c_void};
use std::mem::ManuallyDrop;
use std::path::Path;

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_DESC1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIAdapter1, IDXGIFactory1, IDXGIFactory4,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCSTR, Result};

/// What the host device can do, as far as the project cares.
#[derive(Debug, Clone)]
pub struct Caps {
    pub adapter: String,
    pub vram_bytes: u64,
    /// `(major, minor)`.
    pub shader_model: (u32, u32),
    pub wave_lanes: (u32, u32),
    pub mesh_shaders: bool,
    pub ray_tracing: bool,
    pub double_ops: bool,
    /// Agility SDK version in use, or `None` for the inbox runtime.
    pub agility: Option<u32>,
    /// Whether this is the software adapter (WARP), and more of what
    /// the device offers (`gpu-check` prints them).
    pub software: bool,
    pub feature_level: u32,
    pub binding_tier: i32,
    pub tiled_tier: i32,
    pub int64_typed: bool,
    pub int64_heap: bool,
    pub enhanced_barriers: bool,
    /// SM 6.1 barycentrics (OPTIONS3 BarycentricsSupported): `SV_Barycentrics`
    /// and `GetAttributeAtVertex`.
    pub barycentrics: bool,
    /// On the software adapter: the `d3d10warp.dll` the device runs on,
    /// its path and file version.
    pub warp_module: Option<String>,
}

/// `XNW_ADAPTER=warp`: the host device on the Microsoft Basic Render
/// Driver (WARP, software), so hang diagnosis never touches the hardware
/// driver. Anything else: the hardware adapter with the most memory.
pub fn warp() -> bool {
    std::env::var("XNW_ADAPTER").is_ok_and(|v| v.trim().eq_ignore_ascii_case("warp"))
}

/// `XNW_WARP=<folder>`: on the software adapter, that folder's
/// `d3d10warp.dll` (Microsoft's NuGet WARP, newer than the system's) is the
/// one the device runs on. Unset or empty: the system's. Default off.
pub fn warp_dir() -> Option<std::path::PathBuf> {
    parse_warp_dir(std::env::var_os("XNW_WARP").as_deref())
}

fn parse_warp_dir(v: Option<&std::ffi::OsStr>) -> Option<std::path::PathBuf> {
    let s = v?.to_string_lossy();
    let s = s.trim();
    (!s.is_empty()).then(|| std::path::PathBuf::from(s))
}

fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    s.encode_wide().chain(Some(0)).collect()
}

/// Loads `dir\d3d10warp.dll` by full path, once, before any device: the
/// runtime asks for the module by name and the loader hands back the one
/// already loaded. The DLL stays loaded for the process.
fn load_warp(dir: &Path) -> Result<()> {
    static LOADED: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();
    LOADED
        .get_or_init(|| {
            let file = dir.join("d3d10warp.dll");
            if !file.is_file() {
                return Err(format!("XNW_WARP: no {}", file.display()));
            }
            // LOAD_WITH_ALTERED_SEARCH_PATH: its own imports resolve from
            // its folder first (it has none there; system DLLs otherwise).
            // SAFETY: a NUL-terminated path; the module is never freed.
            let m = unsafe { xnw_core::win::LoadLibraryExW(wide(file.as_os_str()).as_ptr(), std::ptr::null_mut(), 0x8) };
            if m.is_null() {
                return Err(format!("XNW_WARP: loading {} failed: {}", file.display(), xnw_core::win::last_error()));
            }
            Ok(())
        })
        .clone()
        .map_err(|e| windows::core::Error::new(windows::Win32::Foundation::E_FAIL, e))
}

/// The `d3d10warp.dll` loaded in this process: path and file version.
fn warp_module() -> Option<String> {
    use xnw_core::win;
    // SAFETY: module handle and resource reads of a loaded module; every
    // read stays inside the resource's reported size.
    unsafe {
        let m = win::GetModuleHandleW(wide(std::ffi::OsStr::new("d3d10warp.dll")).as_ptr());
        if m.is_null() {
            return None;
        }
        let mut name = [0u16; 1024];
        let n = win::GetModuleFileNameW(m, name.as_mut_ptr(), name.len() as u32) as usize;
        let path = String::from_utf16_lossy(&name[..n]);
        let mut version = "version unknown".to_string();
        let r = win::FindResourceW(m, 1 as *const u16, win::RT_VERSION as *const u16);
        if !r.is_null() {
            let size = win::SizeofResource(m, r) as usize;
            let p = win::LockResource(win::LoadResource(m, r)).cast::<u8>();
            if !p.is_null() {
                let b = std::slice::from_raw_parts(p, size);
                let at = (0..b.len().saturating_sub(16)).step_by(4).find(|&i| b[i..i + 4] == 0xFEEF_04BDu32.to_le_bytes());
                if let Some(i) = at {
                    let w = |o: usize| u32::from_le_bytes(b[i + o..i + o + 4].try_into().unwrap());
                    let (ms, ls) = (w(8), w(12));
                    version = format!("{}.{}.{}.{}", ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF);
                }
            }
        }
        Some(format!("{path} {version}"))
    }
}

/// The last device's OPTIONS3 BarycentricsSupported (0: no device yet,
/// 1: no, 2: yes); read by xnw-gpu's translation options.
static HOST_BARYCENTRICS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether the device made last supports SM 6.1 barycentrics
/// (`SV_Barycentrics`, `GetAttributeAtVertex`); `None` before any device.
/// The RX 5700 XT reports no (inbox and Agility 1.610.5); WARP 1.0.22 yes.
pub fn host_barycentrics() -> Option<bool> {
    match HOST_BARYCENTRICS.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        n => Some(n == 2),
    }
}

pub struct Host {
    pub device: ID3D12Device,
    pub caps: Caps,
    pub queue: ID3D12CommandQueue,
}

pub fn feature<T: Default>(device: &ID3D12Device, id: D3D12_FEATURE, init: T) -> Option<T> {
    let mut data = init;
    // SAFETY: `data` is the structure `id` names and its size is passed.
    unsafe { device.CheckFeatureSupport(id, (&raw mut data).cast::<c_void>(), size_of::<T>() as u32) }
        .ok()
        .map(|()| data)
}

/// Logs one validation-layer message (warnings and worse) as it is raised.
unsafe extern "system" fn layer_message(_category: D3D12_MESSAGE_CATEGORY, severity: D3D12_MESSAGE_SEVERITY, id: D3D12_MESSAGE_ID, text: PCSTR, _context: *mut c_void) {
    if severity.0 > D3D12_MESSAGE_SEVERITY_WARNING.0 {
        return;
    }
    // SAFETY: the layer passes a NUL-terminated string that lives for the call.
    let text = unsafe { text.to_string() }.unwrap_or_default();
    xnw_core::warn!("d3d12 layer (severity {}, id {}): {text}", severity.0, id.0);
}

/// The shader patch mode `XNW_GPU_DEBUG`'s `gbv=state|unguarded|guarded`
/// names (None: the layer's default).
fn gbv_mode(v: &str) -> Option<D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE> {
    v.split(',').find_map(|p| p.strip_prefix("gbv=")).and_then(|m| match m {
        "state" => Some(D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_STATE_TRACKING_ONLY),
        "unguarded" => Some(D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_UNGUARDED_VALIDATION),
        "guarded" => Some(D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_GUARDED_VALIDATION),
        _ => None,
    })
}

/// `D3D12SDKVersion` of the Agility SDK releases we know: 1.6xx.y is 6xx.
fn sdk_version_of(dir: &Path) -> u32 {
    dir.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.split('.').nth(1))
        .and_then(|n| n.parse().ok())
        .unwrap_or(610)
}

/// Whether this process may create a D3D12 device. A test, bench or
/// integration-test binary (cargo puts them in `target/<profile>/deps`)
/// may only with `XNW_TEST_GPU=1`: device tests then skip, as they do with
/// no device, and the card stays untouched (the driver timed out while
/// workers' tests used it). The runtime (`target/<profile>/xnw.exe`) and
/// the tools always may.
pub fn device_allowed() -> bool {
    static ALLOWED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALLOWED.get_or_init(|| {
        let test = std::env::current_exe().ok().and_then(|e| e.parent().and_then(|d| d.file_name()).map(|n| n == "deps")).unwrap_or(false);
        let allowed = !test || std::env::var("XNW_TEST_GPU").is_ok_and(|v| v.trim() == "1");
        if test && !allowed {
            // Past the test harness's capture: once per test binary.
            use std::io::Write;
            let _ = std::io::stderr().write_all(b"device tests skipped: no D3D12 device in test binaries unless XNW_TEST_GPU=1\n");
        }
        allowed
    })
}

impl Host {
    /// Creates the device on the hardware adapter with the most memory.
    /// `agility` is a folder named after the SDK release (`...\1.610.5`).
    pub fn new(agility: Option<&Path>) -> Result<Host> {
        if !device_allowed() {
            return Err(windows::core::Error::new(windows::Win32::Foundation::E_ACCESSDENIED, "device tests are off (set XNW_TEST_GPU=1)"));
        }
        // With `XNW_WARP` the inbox runtime stops this WARP at shader model
        // 6.5: a caller that names no Agility folder gets `XNW_AGILITY`'s.
        let agility_env;
        let mut agility = agility;
        if warp()
            && let Some(dir) = warp_dir()
        {
            load_warp(&dir)?;
            if agility.is_none() {
                agility_env = std::env::var_os("XNW_AGILITY").map(std::path::PathBuf::from).filter(|p| !p.as_os_str().is_empty());
                agility = agility_env.as_deref();
            }
        }
        // SAFETY: plain COM calls; every out pointer is a local.
        unsafe {
            let dxgi: IDXGIFactory1 = CreateDXGIFactory1()?;
            let mut best: Option<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> = None;
            if warp() {
                let adapter: IDXGIAdapter1 = dxgi.cast::<IDXGIFactory4>()?.EnumWarpAdapter()?;
                let desc = adapter.GetDesc1()?;
                best = Some((adapter, desc));
            }
            let mut i = 0;
            while best.is_none()
                && let Ok(adapter) = dxgi.EnumAdapters1(i)
            {
                i += 1;
                let desc = adapter.GetDesc1()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    continue;
                }
                if best.as_ref().is_none_or(|(_, b)| desc.DedicatedVideoMemory > b.DedicatedVideoMemory) {
                    best = Some((adapter, desc));
                }
            }
            let (adapter, desc) = best.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x887A0002u32 as i32)))?;

            let mut device: Option<ID3D12Device> = None;
            let mut agility_version = None;
            if let Some(dir) = agility {
                let version = sdk_version_of(dir);
                let mut path = dir.to_string_lossy().into_owned();
                path.push('\\');
                let path = CString::new(path).expect("path without NUL");
                let mut config: Option<ID3D12SDKConfiguration1> = None;
                D3D12GetInterface(&CLSID_D3D12SDKConfiguration, &mut config)?;
                let factory: ID3D12DeviceFactory = config
                    .expect("interface on success")
                    .CreateDeviceFactory(version, PCSTR(path.as_ptr().cast()))?;
                if std::env::var("XNW_GPU_DEBUG").is_ok_and(|v| v.contains("dred")) {
                    // Breadcrumbs and page-fault data for a device removal
                    // (read back in `d3d12x::host::fail`).
                    if let Ok(dred) = factory.GetConfigurationInterface::<ID3D12DeviceRemovedExtendedDataSettings>(&CLSID_D3D12DeviceRemovedExtendedData) {
                        dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                        dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                        xnw_core::info!("host GPU: device-removal breadcrumbs on (XNW_GPU_DEBUG=dred)");
                    }
                }
                if std::env::var("XNW_GPU_DEBUG").is_ok_and(|v| v.contains("layer")) {
                    // The validation layer (d3d12SDKLayers.dll beside the
                    // Agility core): its error messages are read back in
                    // `d3d12x::host::fail`.
                    match factory.GetConfigurationInterface::<ID3D12Debug>(&CLSID_D3D12Debug) {
                        Ok(debug) => {
                            debug.EnableDebugLayer();
                            xnw_core::info!("host GPU: validation layer on (XNW_GPU_DEBUG=layer)");
                            if std::env::var("XNW_GPU_DEBUG").is_ok_and(|v| v.contains("gbv")) {
                                // Shader-side checks of every descriptor and
                                // resource state a draw or dispatch reads.
                                if let Ok(debug1) = debug.cast::<ID3D12Debug1>() {
                                    debug1.SetEnableGPUBasedValidation(true);
                                    xnw_core::info!("host GPU: GPU-based validation on (XNW_GPU_DEBUG=layer,gbv)");
                                }
                            }
                        }
                        Err(e) => xnw_core::warn!("host GPU: no validation layer: {e}"),
                    }
                }
                factory.CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
                agility_version = Some(version);
            } else {
                if std::env::var("XNW_GPU_DEBUG").is_ok_and(|v| v.contains("layer")) {
                    // The system's validation layer (Graphics Tools) for the
                    // inbox runtime: GPU-based validation of our own DXBC
                    // programs on the inbox WARP (WARP 1.0.22 removes the
                    // device under it, HANG1).
                    let mut debug: Option<ID3D12Debug> = None;
                    match D3D12GetDebugInterface(&mut debug).map(|()| debug.expect("interface on success")) {
                        Ok(debug) => {
                            debug.EnableDebugLayer();
                            xnw_core::info!("host GPU: system validation layer on (XNW_GPU_DEBUG=layer)");
                            if std::env::var("XNW_GPU_DEBUG").is_ok_and(|v| v.contains("gbv"))
                                && let Ok(debug1) = debug.cast::<ID3D12Debug1>()
                            {
                                debug1.SetEnableGPUBasedValidation(true);
                                xnw_core::info!("host GPU: GPU-based validation on (XNW_GPU_DEBUG=layer,gbv)");
                            }
                        }
                        Err(e) => xnw_core::warn!("host GPU: no system validation layer: {e}"),
                    }
                }
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            }
            let device = device.expect("device on success");
            if let Some(mode) = std::env::var("XNW_GPU_DEBUG").ok().as_deref().and_then(gbv_mode) {
                // `gbv=state|unguarded|guarded`: how GPU-based validation
                // patches shaders (the layer's default: unguarded).
                let set = D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS { MaxMessagesPerCommandList: 1024, DefaultShaderPatchMode: mode, PipelineStateCreateFlags: D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_NONE };
                match device.cast::<ID3D12DebugDevice1>().and_then(|d| d.SetDebugParameter(D3D12_DEBUG_DEVICE_PARAMETER_GPU_BASED_VALIDATION_SETTINGS, (&raw const set).cast(), size_of_val(&set) as u32)) {
                    Ok(()) => xnw_core::info!("host GPU: GPU-based validation shader patch mode {}", mode.0),
                    Err(e) => xnw_core::warn!("host GPU: GPU-based validation patch mode not set: {e}"),
                }
            }
            if std::env::var("XNW_GPU_DEBUG").is_ok_and(|v| v.contains("layer")) {
                // Every layer message as it is raised, on every path (the
                // stored queue is read only where a failure is reported).
                let mut cookie = 0u32;
                match device.cast::<ID3D12InfoQueue1>().and_then(|q| q.RegisterMessageCallback(Some(layer_message), D3D12_MESSAGE_CALLBACK_FLAG_NONE, std::ptr::null_mut(), &mut cookie)) {
                    Ok(()) => xnw_core::info!("host GPU: layer messages logged as raised"),
                    Err(e) => xnw_core::warn!("host GPU: layer messages only stored: {e}"),
                }
            }

            let mut shader_model = (6, 0);
            for sm in [D3D_SHADER_MODEL_6_8, D3D_SHADER_MODEL_6_7, D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_5] {
                let ask = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: sm };
                if let Some(got) = feature(&device, D3D12_FEATURE_SHADER_MODEL, ask) {
                    let v = got.HighestShaderModel.0 as u32;
                    shader_model = (v >> 4, v & 15);
                    break;
                }
            }
            let o0 = feature(&device, D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_DATA_D3D12_OPTIONS::default());
            let o1 = feature(&device, D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1::default());
            let o3 = feature(&device, D3D12_FEATURE_D3D12_OPTIONS3, D3D12_FEATURE_DATA_D3D12_OPTIONS3::default());
            let o5 = feature(&device, D3D12_FEATURE_D3D12_OPTIONS5, D3D12_FEATURE_DATA_D3D12_OPTIONS5::default());
            let o7 = feature(&device, D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_DATA_D3D12_OPTIONS7::default());
            let o9 = feature(&device, D3D12_FEATURE_D3D12_OPTIONS9, D3D12_FEATURE_DATA_D3D12_OPTIONS9::default());
            let o11 = feature(&device, D3D12_FEATURE_D3D12_OPTIONS11, D3D12_FEATURE_DATA_D3D12_OPTIONS11::default());
            let o12 = feature(&device, D3D12_FEATURE_D3D12_OPTIONS12, D3D12_FEATURE_DATA_D3D12_OPTIONS12::default());
            let levels = [windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_2, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_0];
            let fl = feature(
                &device,
                D3D12_FEATURE_FEATURE_LEVELS,
                D3D12_FEATURE_DATA_FEATURE_LEVELS { NumFeatureLevels: levels.len() as u32, pFeatureLevelsRequested: levels.as_ptr(), ..Default::default() },
            );
            let name_len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
            let caps = Caps {
                adapter: String::from_utf16_lossy(&desc.Description[..name_len]),
                vram_bytes: desc.DedicatedVideoMemory as u64,
                shader_model,
                wave_lanes: o1.map_or((0, 0), |o| (o.WaveLaneCountMin, o.WaveLaneCountMax)),
                mesh_shaders: o7.is_some_and(|o| o.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED),
                ray_tracing: o5.is_some_and(|o| o.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED),
                double_ops: o0.is_some_and(|o| o.DoublePrecisionFloatShaderOps.as_bool()),
                agility: agility_version,
                software: desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
                feature_level: fl.map_or(0, |f| f.MaxSupportedFeatureLevel.0 as u32),
                binding_tier: o0.map_or(0, |o| o.ResourceBindingTier.0),
                tiled_tier: o0.map_or(0, |o| o.TiledResourcesTier.0),
                int64_typed: o9.is_some_and(|o| o.AtomicInt64OnTypedResourceSupported.as_bool()),
                int64_heap: o11.is_some_and(|o| o.AtomicInt64OnDescriptorHeapResourceSupported.as_bool()),
                enhanced_barriers: o12.is_some_and(|o| o.EnhancedBarriersSupported.as_bool()),
                barycentrics: o3.is_some_and(|o| o.BarycentricsSupported.as_bool()),
                warp_module: if warp() { warp_module() } else { None },
            };
            HOST_BARYCENTRICS.store(1 + u8::from(caps.barycentrics), std::sync::atomic::Ordering::Relaxed);
            if let Some(m) = &caps.warp_module {
                xnw_core::info!("host GPU: software adapter on {m}");
            }
            let queue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;
            Ok(Host { device, caps, queue })
        }
    }

    /// The shared-memory path without shaders: a heap opened over a
    /// VirtualAlloc of ours, a cross-adapter (and unordered-access) placed
    /// buffer on it, a copy into it read back from our memory. What
    /// `gpu-check --shared-memory` checks where its SM 6.6 probes cannot run.
    pub fn probe_open_heap(&self) -> std::result::Result<String, String> {
        const BYTES: usize = 64 << 10;
        // SAFETY: plain object creation and copies on resources of ours; the
        // memory is ours and outlives the device's use of it.
        unsafe {
            let memory = xnw_core::win::VirtualAlloc(std::ptr::null(), BYTES, xnw_core::win::MEM_COMMIT | xnw_core::win::MEM_RESERVE, xnw_core::win::PAGE_READWRITE);
            if memory.is_null() {
                return Err("no memory".into());
            }
            let d3: ID3D12Device3 = self.device.cast().map_err(|e| format!("no ID3D12Device3: {e}"))?;
            let heap: ID3D12Heap = d3.OpenExistingHeapFromAddress(memory.cast_const()).map_err(|e| format!("OpenExistingHeapFromAddress refused: {e}"))?;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: BYTES as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut placed: Option<ID3D12Resource> = None;
            self.device.CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut placed).map_err(|e| format!("cross-adapter placed buffer refused: {e}"))?;
            let placed = placed.ok_or("no buffer")?;
            let src = self.buffer(BYTES as u64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ).map_err(|e| e.to_string())?;
            let mut p: *mut c_void = std::ptr::null_mut();
            src.Map(0, None, Some(&mut p)).map_err(|e| e.to_string())?;
            std::ptr::write_bytes(p.cast::<u8>(), 0x5a, BYTES);
            let alloc: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| e.to_string())?;
            list.CopyBufferRegion(&placed, 0, &src, 0, BYTES as u64);
            list.Close().map_err(|e| e.to_string())?;
            self.queue.ExecuteCommandLists(&[Some(list.cast().map_err(|e| e.to_string())?)]);
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|e| e.to_string())?;
            self.queue.Signal(&fence, 1).map_err(|e| e.to_string())?;
            let event: HANDLE = CreateEventW(None, false, false, None).map_err(|e| e.to_string())?;
            fence.SetEventOnCompletion(1, event).map_err(|e| e.to_string())?;
            WaitForSingleObject(event, INFINITE);
            let _ = CloseHandle(event);
            let landed = std::slice::from_raw_parts(memory.cast::<u8>(), BYTES).iter().all(|&b| b == 0x5a);
            Ok(format!("heap over our memory opened, cross-adapter UAV buffer placed, a GPU copy into it {} in our memory", if landed { "landed" } else { "did NOT land" }))
        }
    }

    pub fn buffer(&self, bytes: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> Result<ID3D12Resource> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let props = D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() };
        let mut out: Option<ID3D12Resource> = None;
        // Xbox2Windows: Windows 10 refuses an upload buffer not
        // made in GENERIC_READ and a readback buffer not made in COPY_DEST.
        debug_assert!(heap != D3D12_HEAP_TYPE_UPLOAD || state == D3D12_RESOURCE_STATE_GENERIC_READ, "upload buffers start in GENERIC_READ");
        debug_assert!(heap != D3D12_HEAP_TYPE_READBACK || state == D3D12_RESOURCE_STATE_COPY_DEST, "readback buffers start in COPY_DEST");
        // SAFETY: both descriptions outlive the call.
        if let Err(e) = unsafe { self.device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out) } {
            let kind = if heap == D3D12_HEAP_TYPE_UPLOAD { xnw_core::syscalls::Made::Upload } else { xnw_core::syscalls::Made::Buffer };
            xnw_core::syscalls::creation_failed(kind, &e, &format_args!("buffer of {bytes} bytes, heap {}, state {:#x}", heap.0, state.0));
            return Err(e);
        }
        Ok(out.expect("resource on success"))
    }

    /// `xnw gpu-check --health`: `n` fence-waited submissions of a small
    /// fixed workload needing no shader compiler: a 512x512 render target
    /// clear, an unordered-access clear of a 4 MiB buffer (a driver compute
    /// pass) and a 4 MiB buffer copy. Each submission's wall time, in
    /// milliseconds; an error if the device is removed.
    pub fn health(&self, n: u32) -> Result<Vec<f64>> {
        const BYTES: u64 = 4 << 20;
        // SAFETY: COM calls on live objects; every pointer passed is to a
        // local that outlives the call.
        unsafe {
            let uav = self.buffer(BYTES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
            let dst = self.buffer(BYTES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON)?;
            let rt_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 512,
                Height: 512,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let mut rt: Option<ID3D12Resource> = None;
            self.device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut rt)?;
            let rt = rt.expect("resource on success");
            let heap = |kind, flags| self.device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC { Type: kind, NumDescriptors: 1, Flags: flags, NodeMask: 0 });
            let rtvs = heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE)?;
            let visible = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)?;
            let cpu_only = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE)?;
            self.device.CreateRenderTargetView(&rt, None, rtvs.GetCPUDescriptorHandleForHeapStart());
            let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: (BYTES / 4) as u32, ..Default::default() } },
            };
            self.device.CreateUnorderedAccessView(&uav, None, Some(&uav_desc), visible.GetCPUDescriptorHandleForHeapStart());
            self.device.CreateUnorderedAccessView(&uav, None, Some(&uav_desc), cpu_only.GetCPUDescriptorHandleForHeapStart());
            let alloc: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
            list.Close()?;
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event: HANDLE = CreateEventW(None, false, false, None)?;
            let mut times = Vec::with_capacity(n as usize);
            for i in 1..=u64::from(n) {
                let started = std::time::Instant::now();
                alloc.Reset()?;
                list.Reset(&alloc, None)?;
                list.ClearRenderTargetView(rtvs.GetCPUDescriptorHandleForHeapStart(), &[i as f32 / 256.0, 0.0, 0.0, 1.0], None);
                list.SetDescriptorHeaps(&[Some(visible.clone())]);
                list.ClearUnorderedAccessViewUint(visible.GetGPUDescriptorHandleForHeapStart(), cpu_only.GetCPUDescriptorHandleForHeapStart(), &uav, &[i as u32; 4], &[]);
                list.CopyBufferRegion(&dst, 0, &uav, 0, BYTES);
                list.Close()?;
                self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                self.queue.Signal(&fence, i)?;
                fence.SetEventOnCompletion(i, event)?;
                WaitForSingleObject(event, INFINITE);
                self.device.GetDeviceRemovedReason()?;
                times.push(started.elapsed().as_secs_f64() * 1000.0);
            }
            let _ = CloseHandle(event);
            Ok(times)
        }
    }

    /// Runs a compute shader once (`Dispatch(1,1,1)`) with a zeroed buffer of
    /// `words` u32 as root UAV `u0`, and returns the buffer. For checks and
    /// tests; nothing here is tuned.
    pub fn run_compute(&self, dxil: &[u8], words: usize) -> Result<Vec<u32>> {
        let bytes = (words * 4) as u64;
        // SAFETY: COM calls on live objects; pointers passed are to locals
        // that outlive each call; the mapped range is `bytes` long.
        unsafe {
            let param = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: 1,
                pParameters: &param,
                ..Default::default()
            };
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut blob, None)?;
            let blob = blob.expect("blob on success");
            let rs_bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, rs_bytes)?;

            let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
                ..Default::default()
            };
            let pso: Result<ID3D12PipelineState> = self.device.CreateComputePipelineState(&pso_desc);
            drop(ManuallyDrop::into_inner(pso_desc.pRootSignature));
            let pso = pso?;

            let gpu = self.buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
            let back = self.buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;

            let alloc: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)?;
            list.SetComputeRootSignature(&root);
            list.SetComputeRootUnorderedAccessView(0, gpu.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(gpu.clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            };
            list.ResourceBarrier(std::slice::from_ref(&barrier));
            drop(ManuallyDrop::into_inner(ManuallyDrop::into_inner(barrier.Anonymous.Transition).pResource));
            list.CopyResource(&back, &gpu);
            list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);

            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            self.queue.Signal(&fence, 1)?;
            let event: HANDLE = CreateEventW(None, false, false, None)?;
            fence.SetEventOnCompletion(1, event)?;
            WaitForSingleObject(event, INFINITE);
            let _ = CloseHandle(event);
            self.device.GetDeviceRemovedReason()?;

            let mut ptr: *mut c_void = std::ptr::null_mut();
            back.Map(0, None, Some(&mut ptr))?;
            let out = std::slice::from_raw_parts(ptr.cast::<u32>(), words).to_vec();
            back.Unmap(0, None);
            Ok(out)
        }
    }
}

#[cfg(test)]
mod warp_switch {
    use super::parse_warp_dir;
    use std::ffi::OsStr;
    use std::path::PathBuf;

    #[test]
    fn parse() {
        assert_eq!(parse_warp_dir(None), None);
        assert_eq!(parse_warp_dir(Some(OsStr::new(""))), None);
        assert_eq!(parse_warp_dir(Some(OsStr::new("  "))), None);
        assert_eq!(parse_warp_dir(Some(OsStr::new(r" sdk\warp\1.0.22 "))), Some(PathBuf::from(r"sdk\warp\1.0.22")));
    }
}

/// The experiment that says whether ending the process while the GPU
/// still runs its work causes a driver reset, without risking one: the
/// work is a compute dispatch of about 1.4 s, below Windows' 2 s timeout,
/// so it ends by itself even if nothing cancels it; the process ends at
/// 1.2 s with it in flight (as `XNW_GPU_DEADLINE` does).
///
/// Ignored; run by hand on the hardware adapter, nothing else running:
/// `cargo test -p xnw-gpu --lib deadline_probe -- --ignored --nocapture`
/// (with `XNW_AGILITY`). It calibrates on short dispatches first (each
/// under 0.2 s), prints the time it will end at, and the test process
/// exits mid-dispatch (the harness reports it as a crash: expected).
/// Then look in Event Viewer, Windows Logs > System, for Display event
/// 4101 ("display driver stopped responding and has recovered") at that
/// time. None: ending the process with work in flight does not by itself
/// reset the driver. It cannot tell what happens with truly hung work
/// (that would need a shader that never ends, which risks the reset).
#[cfg(test)]
mod deadline_probe {
    use super::*;
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;

    const SOURCE: &str = "cbuffer c : register(b0) { uint n; } RWStructuredBuffer<uint> o : register(u0); [numthreads(64, 1, 1)] void main(uint id : SV_DispatchThreadID) { uint x = id; for (uint i = 0; i < n; i++) { x = x * 1664525u + 1013904223u; } o[id] = x; }";

    #[test]
    #[ignore = "run by hand: ends the process with GPU work in flight (see the module)"]
    fn deadline_probe() {
        let agility = std::env::var_os("XNW_AGILITY").map(std::path::PathBuf::from);
        let host = Host::new(agility.as_deref()).expect("a host device");
        assert!(!host.caps.software, "the probe is for the hardware adapter");
        // SAFETY: plain object creation, recording and submission on our own
        // device; every pointer passed outlives its call.
        unsafe {
            let mut code: Option<ID3DBlob> = None;
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), PCSTR::null(), None, None, PCSTR(c"main".as_ptr().cast()), PCSTR(c"cs_5_1".as_ptr().cast()), 0, 0, &mut code, None).unwrap();
            let code = code.unwrap();
            let params = [
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 1 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
                D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV, Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } }, ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
            ];
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC { NumParameters: 2, pParameters: params.as_ptr(), ..Default::default() }, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut blob, None).unwrap();
            let blob = blob.unwrap();
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.GetBufferPointer(), BytecodeLength: code.GetBufferSize() },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&desc).unwrap();
            drop(ManuallyDrop::into_inner(desc.pRootSignature));
            const GROUPS: u32 = 4096;
            let out = host.buffer(u64::from(GROUPS) * 64 * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS).unwrap();
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event: HANDLE = CreateEventW(None, false, false, None).unwrap();
            let mut value = 0;
            let mut run = |n: u32, wait: bool| -> f64 {
                let alloc: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).unwrap();
                list.SetComputeRootSignature(&root);
                list.SetComputeRoot32BitConstant(0, n, 0);
                list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
                list.Dispatch(GROUPS, 1, 1);
                list.Close().unwrap();
                let at = std::time::Instant::now();
                host.queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                value += 1;
                host.queue.Signal(&fence, value).unwrap();
                if wait {
                    fence.SetEventOnCompletion(value, event).unwrap();
                    WaitForSingleObject(event, INFINITE);
                }
                std::mem::forget((alloc, list));
                at.elapsed().as_secs_f64()
            };
            // Calibration: double the loop until a dispatch takes 100 ms; the
            // probe aims at 1.4 s, well under the 2 s timeout.
            let mut n = 1024u32;
            let mut t = run(n, true);
            while t < 0.1 && n < u32::MAX / 4 {
                n *= 2;
                t = run(n, true);
                assert!(t < 0.2, "calibration stays short");
            }
            let n15 = (f64::from(n) * 1.4 / t) as u32;
            println!("deadline probe: {n} iterations took {:.1} ms; the next dispatch runs {n15} (~1.4 s); the process ends at 1.2 s", t * 1000.0);
            run(n15, false);
            std::thread::sleep(std::time::Duration::from_millis(1200));
            println!("deadline probe: ending the process now ({:?} since start); check Event Viewer for Display 4101", std::time::SystemTime::now());
            xnw_core::win::TerminateProcess(xnw_core::win::GetCurrentProcess(), 0xDEAD_0002);
        }
    }
}
