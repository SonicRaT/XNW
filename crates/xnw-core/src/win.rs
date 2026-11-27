//! The few Win32 functions we need, declared by hand: no bindings
//! crate, nothing to compile.

#![allow(non_snake_case, clippy::upper_case_acronyms)]

use std::ffi::c_void;

pub type HMODULE = *mut c_void;
pub type HANDLE = *mut c_void;

pub const PAGE_READWRITE: u32 = 0x04;
pub const PAGE_EXECUTE_READ: u32 = 0x20;
pub const PAGE_EXECUTE_READWRITE: u32 = 0x40;
pub const MEM_COMMIT: u32 = 0x1000;
pub const MEM_RESERVE: u32 = 0x2000;

pub const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

#[repr(C)]
pub struct ExceptionRecord {
    pub code: u32,
    pub flags: u32,
    pub record: *mut ExceptionRecord,
    pub address: usize,
    pub param_count: u32,
    pub params: [usize; 15],
}

#[repr(C)]
pub struct ExceptionPointers {
    pub record: *mut ExceptionRecord,
    pub context: *mut Context,
}

/// x64 `CONTEXT`: 1232 bytes, 16-byte aligned. Fields are read by offset.
#[repr(C, align(16))]
pub struct Context(pub [u8; 1232]);

impl Context {
    pub const RAX: usize = 0x78;
    pub const RSP: usize = 0x98;
    pub const RIP: usize = 0xF8;
    /// Names of the 16 integer registers in `CONTEXT` order from `RAX`.
    pub const INT_NAMES: [&'static str; 16] = [
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    ];

    pub fn zeroed() -> Self {
        Context([0; 1232])
    }
    /// The debug registers (DR0-3, DR6, DR7); `FLAGS` is `ContextFlags`.
    pub const DR0: usize = 0x48;
    pub const DR6: usize = 0x68;
    pub const DR7: usize = 0x70;
    pub fn get(&self, offset: usize) -> u64 {
        u64::from_le_bytes(self.0[offset..offset + 8].try_into().unwrap())
    }
    pub fn rip(&self) -> usize {
        self.get(Self::RIP) as usize
    }
    pub fn rsp(&self) -> usize {
        self.get(Self::RSP) as usize
    }
}

pub type VectoredHandler = unsafe extern "system" fn(*mut ExceptionPointers) -> i32;

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn LoadLibraryExW(name: *const u16, file: HANDLE, flags: u32) -> HMODULE;
    pub fn LoadLibraryA(name: *const u8) -> HMODULE;
    pub fn GetProcAddress(module: HMODULE, name: *const u8) -> *const c_void;
    pub fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut HMODULE) -> i32;
    pub fn GetModuleFileNameW(module: HMODULE, name: *mut u16, size: u32) -> u32;
    pub fn VirtualProtect(address: *const c_void, size: usize, protect: u32, old: *mut u32) -> i32;
    pub fn VirtualAlloc(address: *const c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
    pub fn FlushInstructionCache(process: HANDLE, address: *const c_void, size: usize) -> i32;
    pub fn GetCurrentProcess() -> HANDLE;
    pub fn GetCurrentThreadId() -> u32;
    pub fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize);
    pub fn TerminateProcess(process: HANDLE, code: u32) -> i32;
    pub fn AddVectoredExceptionHandler(first: u32, handler: VectoredHandler) -> *mut c_void;
    pub fn RtlCaptureContext(context: *mut Context);
    pub fn RtlLookupFunctionEntry(pc: u64, image_base: *mut u64, history: *mut c_void) -> *const RuntimeFunction;
    pub fn RtlVirtualUnwind(
        handler_type: u32,
        image_base: u64,
        pc: u64,
        function: *const RuntimeFunction,
        context: *mut Context,
        handler_data: *mut *mut c_void,
        establisher: *mut u64,
        pointers: *mut c_void,
    ) -> *mut c_void;
}

#[repr(C)]
pub struct RuntimeFunction {
    pub begin: u32,
    pub end: u32,
    pub unwind: u32,
}

pub fn last_error() -> std::io::Error {
    std::io::Error::last_os_error()
}

// Memory (xnw-hle).
pub const PAGE_NOACCESS: u32 = 0x01;
pub const MEM_RELEASE: u32 = 0x8000;
pub const MEM_DECOMMIT: u32 = 0x4000;
pub const MEM_WRITE_WATCH: u32 = 0x20_0000;
pub const MEM_RESERVE_PLACEHOLDER: u32 = 0x4_0000;
pub const MEM_REPLACE_PLACEHOLDER: u32 = 0x4000;

#[repr(C)]
pub struct MemoryBasicInformation {
    pub base: usize,
    pub allocation_base: usize,
    pub allocation_protect: u32,
    pub partition: u16,
    pub region_size: usize,
    pub state: u32,
    pub protect: u32,
    pub kind: u32,
}

#[repr(C)]
pub struct MemAddressRequirements {
    pub lowest: usize,
    pub highest: usize,
    pub alignment: usize,
}

/// `MEM_EXTENDED_PARAMETER` of type address requirements (type 1).
#[repr(C)]
pub struct MemExtendedParameter {
    pub kind: u64,
    pub pointer: *mut MemAddressRequirements,
}

/// `VirtualAlloc2`, which only `kernelbase` exports (no import library in
/// every SDK), found once at run time.
pub type VirtualAlloc2Fn = unsafe extern "system" fn(HANDLE, *const c_void, usize, u32, u32, *mut MemExtendedParameter, u32) -> *mut c_void;

pub fn virtual_alloc2() -> VirtualAlloc2Fn {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let f = *F.get_or_init(|| {
        // SAFETY: a system DLL that is always present, and a NUL-terminated name.
        unsafe { GetProcAddress(LoadLibraryA(c"kernelbase.dll".as_ptr().cast()), c"VirtualAlloc2".as_ptr().cast()) as usize }
    });
    assert!(f != 0, "kernelbase!VirtualAlloc2 not found");
    // SAFETY: the export has this signature.
    unsafe { std::mem::transmute(f) }
}

// Library loading as the title sees it (xnw-hle `libloader`).
pub const ERROR_MOD_NOT_FOUND: u32 = 126;
pub const ERROR_PROC_NOT_FOUND: u32 = 127;
pub const ERROR_DLL_INIT_FAILED: u32 = 1114;

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn LoadLibraryExA(name: *const u8, file: HANDLE, flags: u32) -> HMODULE;
    pub fn GetModuleHandleA(name: *const u8) -> HMODULE;
    pub fn GetModuleHandleW(name: *const u16) -> HMODULE;
    pub fn FreeLibrary(module: HMODULE) -> i32;
    pub fn SetLastError(code: u32);
    pub fn RtlCaptureStackBackTrace(skip: u32, count: u32, frames: *mut *mut c_void, hash: *mut u32) -> u16;
}

// A loaded module's version resource (xnw-gpu: which WARP the device uses).
pub const RT_VERSION: usize = 16;

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn FindResourceW(module: HMODULE, name: *const u16, kind: *const u16) -> HANDLE;
    pub fn LoadResource(module: HMODULE, resource: HANDLE) -> HANDLE;
    pub fn LockResource(data: HANDLE) -> *const c_void;
    pub fn SizeofResource(module: HMODULE, resource: HANDLE) -> u32;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32;
    pub fn VirtualQueryEx(process: HANDLE, address: *const c_void, info: *mut MemoryBasicInformation, size: usize) -> usize;
}

// The console's millisecond timers for the whole run (xnw-hle `audio::use_console_timers`).
pub const PROCESS_POWER_THROTTLING: u32 = 4;
pub const PROCESS_POWER_THROTTLING_EXECUTION_SPEED: u32 = 1;
pub const PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION: u32 = 4;

#[repr(C)]
pub struct ProcessPowerThrottlingState {
    pub version: u32,
    pub control: u32,
    pub state: u32,
}

// Low priority for offline tools (`xnw texture-pack`).
pub const IDLE_PRIORITY_CLASS: u32 = 0x40;
pub const PROCESS_MODE_BACKGROUND_BEGIN: u32 = 0x0010_0000;

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn SetPriorityClass(process: HANDLE, class: u32) -> i32;
    pub fn SetProcessInformation(process: HANDLE, class: u32, info: *const c_void, size: u32) -> i32;
}

// Thread census (xnw-hle `audio_capture`, `XNW_TRACE=audio`).
pub const THREAD_QUERY_LIMITED_INFORMATION: u32 = 0x0800;

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn GetThreadTimes(thread: HANDLE, created: *mut u64, exited: *mut u64, kernel: *mut u64, user: *mut u64) -> i32;
    pub fn GetThreadPriority(thread: HANDLE) -> i32;
    pub fn GetThreadDescription(thread: HANDLE, description: *mut *mut u16) -> i32;
    /// Title threads named by their entry; affinity and errors.
    pub fn SetThreadDescription(thread: HANDLE, description: *const u16) -> i32;
    pub fn GetProcessAffinityMask(process: HANDLE, process_mask: *mut usize, system_mask: *mut usize) -> i32;
    pub fn GetLastError() -> u32;
    /// A thread's CPU cycles so far (the sampler's running test).
    pub fn QueryThreadCycleTime(thread: HANDLE, cycles: *mut u64) -> i32;
    pub fn LocalFree(memory: *mut c_void) -> *mut c_void;
    /// The classic timer wait the title's own `Sleep` calls get (Rust's
    /// `thread::sleep` uses a high-resolution waitable timer instead).
    pub fn Sleep(ms: u32);
}

#[link(name = "winmm")]
unsafe extern "system" {
    pub fn timeBeginPeriod(ms: u32) -> u32;
}

#[link(name = "ntdll")]
unsafe extern "system" {
    /// Resolutions in 100 ns units: coarsest, finest, current.
    pub fn NtQueryTimerResolution(coarsest: *mut u32, finest: *mut u32, current: *mut u32) -> i32;
}

// Title memory copies that must not fault on unmapped pages (xnw-gpu d3d12x host).
#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn ReadProcessMemory(process: HANDLE, base: *const c_void, buffer: *mut c_void, size: usize, read: *mut usize) -> i32;
    pub fn WriteProcessMemory(process: HANDLE, base: *mut c_void, buffer: *const c_void, size: usize, written: *mut usize) -> i32;
}

// Instruction emulation and thread notices (xnw-loader).
pub const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
pub const DLL_PROCESS_DETACH: u32 = 0;
pub const DLL_PROCESS_ATTACH: u32 = 1;
pub const DLL_THREAD_ATTACH: u32 = 2;
pub const DLL_THREAD_DETACH: u32 = 3;
pub const TH32CS_SNAPTHREAD: u32 = 4;
pub const THREAD_TERMINATE: u32 = 0x0001;
pub const THREAD_QUERY_INFORMATION: u32 = 0x0040;
pub const INVALID_HANDLE_VALUE: HANDLE = -1isize as HANDLE;
/// `THREADINFOCLASS` for a thread's start address.
pub const THREAD_QUERY_SET_WIN32_START_ADDRESS: u32 = 9;

impl Context {
    pub const EFLAGS: usize = 0x44;
    /// `FltSave.XmmRegisters[0]`; each register is 16 bytes, low quadword first.
    pub const XMM0: usize = 0x1A0;

    pub fn set(&mut self, offset: usize, value: u64) {
        self.0[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    /// General register `n` as instructions number them (0 = rax, 15 = r15).
    pub fn gpr(&self, n: usize) -> u64 {
        self.get(Self::RAX + (n & 15) * 8)
    }
    pub fn set_gpr(&mut self, n: usize, value: u64) {
        self.set(Self::RAX + (n & 15) * 8, value);
    }
    /// Low (`half` 0) or high (1) quadword of `xmm[n]`.
    pub fn xmm(&self, n: usize, half: usize) -> u64 {
        self.get(Self::XMM0 + (n & 15) * 16 + half * 8)
    }
    pub fn set_xmm(&mut self, n: usize, half: usize, value: u64) {
        self.set(Self::XMM0 + (n & 15) * 16 + half * 8, value);
    }
    pub fn eflags(&self) -> u32 {
        self.get(Self::EFLAGS) as u32
    }
    pub fn set_eflags(&mut self, value: u32) {
        self.0[Self::EFLAGS..Self::EFLAGS + 4].copy_from_slice(&value.to_le_bytes());
    }
}

#[repr(C)]
pub struct ThreadEntry32 {
    pub size: u32,
    pub usage: u32,
    pub thread_id: u32,
    pub owner_process_id: u32,
    pub base_priority: i32,
    pub delta_priority: i32,
    pub flags: u32,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn CloseHandle(handle: HANDLE) -> i32;
    pub fn GetCurrentProcessId() -> u32;
    pub fn GetCurrentThread() -> HANDLE;
    pub fn SetThreadPriority(thread: HANDLE, priority: i32) -> i32;
    pub fn OpenThread(access: u32, inherit: i32, id: u32) -> HANDLE;
    pub fn TerminateThread(thread: HANDLE, code: u32) -> i32;
    pub fn CreateToolhelp32Snapshot(flags: u32, process: u32) -> HANDLE;
    pub fn Thread32First(snapshot: HANDLE, entry: *mut ThreadEntry32) -> i32;
    pub fn Thread32Next(snapshot: HANDLE, entry: *mut ThreadEntry32) -> i32;
    pub fn SuspendThread(thread: HANDLE) -> u32;
    pub fn ResumeThread(thread: HANDLE) -> u32;
    pub fn GetThreadContext(thread: HANDLE, context: *mut Context) -> i32;
    pub fn SetThreadContext(thread: HANDLE, context: *const Context) -> i32;
}

/// `SYMBOL_INFOW` with room for a 512-character name (dbghelp).
#[repr(C)]
pub struct SymbolInfoW {
    pub size_of_struct: u32,
    pub type_index: u32,
    pub reserved: [u64; 2],
    pub index: u32,
    pub size: u32,
    pub mod_base: u64,
    pub flags: u32,
    pub value: u64,
    pub address: u64,
    pub register: u32,
    pub scope: u32,
    pub tag: u32,
    pub name_len: u32,
    pub max_name_len: u32,
    pub name: [u16; 512],
}

/// `sizeof(SYMBOL_INFOW)` without the name past its first character.
pub const SYMBOL_INFOW_SIZE: u32 = 88;

#[link(name = "dbghelp")]
unsafe extern "system" {
    pub fn SymSetOptions(options: u32) -> u32;
    pub fn SymInitializeW(process: HANDLE, search_path: *const u16, invade: i32) -> i32;
    pub fn SymFromAddrW(process: HANDLE, address: u64, displacement: *mut u64, symbol: *mut SymbolInfoW) -> i32;
}

/// Thread rights for a stack dump (stall dump).
pub const THREAD_SUSPEND_RESUME: u32 = 0x0002;
pub const THREAD_GET_CONTEXT: u32 = 0x0008;
/// `CONTEXT_AMD64 | CONTROL | INTEGER` (rip, rsp and the integer registers).
pub const CONTEXT_CONTROL_INTEGER: u32 = 0x10_0003;

impl Context {
    /// `ContextFlags`.
    pub const FLAGS: usize = 0x30;
}

#[link(name = "ntdll")]
unsafe extern "system" {
    /// `ObjectTypeInformation` (2) names a handle's kind.
    pub fn NtQueryObject(handle: HANDLE, class: u32, info: *mut c_void, len: u32, written: *mut u32) -> i32;
    pub fn NtQueryInformationThread(thread: HANDLE, class: u32, info: *mut c_void, len: u32, written: *mut u32) -> i32;
}

// Host input (xnw-hle `gameinput`).
pub const MAPVK_VK_TO_VSC_EX: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct XINPUT_GAMEPAD {
    pub buttons: u16,
    pub left_trigger: u8,
    pub right_trigger: u8,
    pub left_x: i16,
    pub left_y: i16,
    pub right_x: i16,
    pub right_y: i16,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct XINPUT_STATE {
    pub packet: u32,
    pub gamepad: XINPUT_GAMEPAD,
}

#[repr(C)]
pub struct XINPUT_VIBRATION {
    pub left: u16,
    pub right: u16,
}

#[link(name = "user32")]
unsafe extern "system" {
    pub fn GetAsyncKeyState(key: i32) -> i16;
    pub fn MapVirtualKeyW(code: u32, kind: u32) -> u32;
    pub fn GetForegroundWindow() -> HANDLE;
    pub fn GetWindowThreadProcessId(window: HANDLE, process: *mut u32) -> u32;
    /// `SM_REMOTESESSION` (0x1000) and others.
    pub fn GetSystemMetrics(index: i32) -> i32;
}

#[link(name = "xinput")]
unsafe extern "system" {
    pub fn XInputGetState(user: u32, state: *mut XINPUT_STATE) -> u32;
    pub fn XInputSetState(user: u32, vibration: *const XINPUT_VIBRATION) -> u32;
}

// Physical pages (xnw-hle `xmem`).
pub const MEM_PRESERVE_PLACEHOLDER: u32 = 0x2;
pub const SEC_RESERVE: u32 = 0x400_0000;
pub const SEC_COMMIT: u32 = 0x800_0000;
pub const FILE_MAP_WRITE: u32 = 0x2;

#[repr(C)]
#[derive(Default)]
pub struct ProcessMemoryCounters {
    pub cb: u32,
    pub page_fault_count: u32,
    pub peak_working_set_size: usize,
    pub working_set_size: usize,
    pub quota_peak_paged_pool_usage: usize,
    pub quota_paged_pool_usage: usize,
    pub quota_peak_non_paged_pool_usage: usize,
    pub quota_non_paged_pool_usage: usize,
    pub pagefile_usage: usize,
    pub peak_pagefile_usage: usize,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn CreateFileMappingW(file: HANDLE, security: *const c_void, protect: u32, size_high: u32, size_low: u32, name: *const u16) -> HANDLE;
    pub fn MapViewOfFile(section: HANDLE, access: u32, offset_high: u32, offset_low: u32, size: usize) -> *mut c_void;
    pub fn UnmapViewOfFile(address: *const c_void) -> i32;
    pub fn K32GetProcessMemoryInfo(process: HANDLE, counters: *mut ProcessMemoryCounters, size: u32) -> i32;
}

/// `MapViewOfFile3`, which only `kernelbase` exports, found once at run time.
pub type MapViewOfFile3Fn =
    unsafe extern "system" fn(HANDLE, HANDLE, *const c_void, u64, usize, u32, u32, *mut MemExtendedParameter, u32) -> *mut c_void;
/// `UnmapViewOfFile2`, likewise.
pub type UnmapViewOfFile2Fn = unsafe extern "system" fn(HANDLE, *const c_void, u32) -> i32;

fn kernelbase(name: &std::ffi::CStr) -> usize {
    // SAFETY: a system DLL that is always present, and a NUL-terminated name.
    let f = unsafe { GetProcAddress(LoadLibraryA(c"kernelbase.dll".as_ptr().cast()), name.as_ptr().cast()) as usize };
    assert!(f != 0, "kernelbase!{name:?} not found");
    f
}

pub fn map_view_of_file3() -> MapViewOfFile3Fn {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let f = *F.get_or_init(|| kernelbase(c"MapViewOfFile3"));
    // SAFETY: the export has this signature.
    unsafe { std::mem::transmute(f) }
}

pub fn unmap_view_of_file2() -> UnmapViewOfFile2Fn {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let f = *F.get_or_init(|| kernelbase(c"UnmapViewOfFile2"));
    // SAFETY: the export has this signature.
    unsafe { std::mem::transmute(f) }
}

// DirectStorage error events (xnw-hle `dstorage`).
pub const DUPLICATE_SAME_ACCESS: u32 = 0x2;

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn CreateEventW(security: *const c_void, manual_reset: i32, initial: i32, name: *const u16) -> HANDLE;
    pub fn SetEvent(event: HANDLE) -> i32;
    pub fn ResetEvent(event: HANDLE) -> i32;
    pub fn DuplicateHandle(source_process: HANDLE, source: HANDLE, target_process: HANDLE, target: *mut HANDLE, access: u32, inherit: i32, options: u32) -> i32;
}

// Game saves (xnw-hle `xgameruntime::xgamesave`).
#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn GetDiskFreeSpaceExW(dir: *const u16, free_to_caller: *mut u64, total: *mut u64, total_free: *mut u64) -> i32;
}

// The present window (xnw-gpu `d3d12x::present`).
pub const WS_OVERLAPPEDWINDOW: u32 = 0x00CF_0000;
pub const CW_USEDEFAULT: i32 = 0x8000_0000_u32 as i32;
pub const SW_SHOWNOACTIVATE: i32 = 4;
pub const WM_CLOSE: u32 = 0x0010;
pub const IDC_ARROW: usize = 32512;

#[repr(C)]
pub struct WNDCLASSEXW {
    pub size: u32,
    pub style: u32,
    pub window_proc: Option<unsafe extern "system" fn(HANDLE, u32, usize, isize) -> isize>,
    pub class_extra: i32,
    pub window_extra: i32,
    pub instance: HMODULE,
    pub icon: HANDLE,
    pub cursor: HANDLE,
    pub background: HANDLE,
    pub menu_name: *const u16,
    pub class_name: *const u16,
    pub small_icon: HANDLE,
}

#[repr(C)]
#[derive(Default)]
pub struct RECT {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[repr(C)]
pub struct MSG {
    pub window: HANDLE,
    pub message: u32,
    pub wparam: usize,
    pub lparam: isize,
    pub time: u32,
    pub point: [i32; 2],
    pub private: u32,
}

#[link(name = "user32")]
unsafe extern "system" {
    pub fn RegisterClassExW(class: *const WNDCLASSEXW) -> u16;
    pub fn CreateWindowExW(
        ex_style: u32,
        class: *const u16,
        name: *const u16,
        style: u32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: HANDLE,
        menu: HANDLE,
        instance: HMODULE,
        param: *const c_void,
    ) -> HANDLE;
    pub fn DefWindowProcW(window: HANDLE, message: u32, wparam: usize, lparam: isize) -> isize;
    pub fn ShowWindow(window: HANDLE, command: i32) -> i32;
    pub fn AdjustWindowRect(rect: *mut RECT, style: u32, menu: i32) -> i32;
    pub fn GetMessageW(message: *mut MSG, window: HANDLE, first: u32, last: u32) -> i32;
    pub fn TranslateMessage(message: *const MSG) -> i32;
    pub fn DispatchMessageW(message: *const MSG) -> isize;
    pub fn SetWindowTextW(window: HANDLE, text: *const u16) -> i32;
    pub fn LoadCursorW(instance: HMODULE, name: *const u16) -> HANDLE;
}

pub const SWP_NOMOVE: u32 = 0x0002;
pub const SWP_NOZORDER: u32 = 0x0004;
pub const SWP_NOACTIVATE: u32 = 0x0010;
pub const SWP_ASYNCWINDOWPOS: u32 = 0x4000;
pub const SPI_GETWORKAREA: u32 = 0x0030;
pub const GWL_STYLE: i32 = -16;

pub type EnumWindowsProc = unsafe extern "system" fn(HANDLE, isize) -> i32;

#[link(name = "user32")]
unsafe extern "system" {
    pub fn EnumWindows(callback: EnumWindowsProc, param: isize) -> i32;
    pub fn IsWindowVisible(window: HANDLE) -> i32;
    pub fn InternalGetWindowText(window: HANDLE, text: *mut u16, max: i32) -> i32;
    pub fn GetClassNameW(window: HANDLE, name: *mut u16, max: i32) -> i32;
    pub fn SetWindowPos(window: HANDLE, after: HANDLE, x: i32, y: i32, cx: i32, cy: i32, flags: u32) -> i32;
    pub fn SystemParametersInfoW(action: u32, param: u32, out: *mut c_void, ini: u32) -> i32;
    pub fn GetWindowLongW(window: HANDLE, index: i32) -> i32;
}

// The title's window procedure, subclassed (xnw-hle `window`).
pub type WndProc = unsafe extern "system" fn(HANDLE, u32, usize, isize) -> isize;
pub const GWLP_WNDPROC: i32 = -4;
pub const WM_NCDESTROY: u32 = 0x0082;
pub const WM_SYSCOMMAND: u32 = 0x0112;
pub const SC_KEYMENU: usize = 0xF100;

#[link(name = "user32")]
unsafe extern "system" {
    pub fn CreateWindowExA(
        ex_style: u32,
        class: *const u8,
        name: *const u8,
        style: u32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: HANDLE,
        menu: HANDLE,
        instance: HMODULE,
        param: *const c_void,
    ) -> HANDLE;
    pub fn GetWindowLongPtrW(window: HANDLE, index: i32) -> isize;
    pub fn SetWindowLongPtrW(window: HANDLE, index: i32, value: isize) -> isize;
    pub fn SetPropW(window: HANDLE, name: *const u16, data: HANDLE) -> i32;
    pub fn GetPropW(window: HANDLE, name: *const u16) -> HANDLE;
    pub fn RemovePropW(window: HANDLE, name: *const u16) -> HANDLE;
    pub fn CallWindowProcW(proc: Option<WndProc>, window: HANDLE, message: u32, wparam: usize, lparam: isize) -> isize;
    pub fn SendMessageW(window: HANDLE, message: u32, wparam: usize, lparam: isize) -> isize;
    pub fn DestroyWindow(window: HANDLE) -> i32;
}
pub const MEM_FREE: u32 = 0x1_0000;

// Write watching and guard pages over title memory (xnw-gpu d3d12x watch).
pub const PAGE_READONLY: u32 = 0x02;
pub const MEM_MAPPED: u32 = 0x4_0000;
pub const WRITE_WATCH_FLAG_RESET: u32 = 1;
pub const EXCEPTION_ACCESS_VIOLATION: u32 = 0xC000_0005;

// `XNW_OBJECT_GUARD`: the process's mitigations and the page heap.
#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn GetProcessMitigationPolicy(process: HANDLE, policy: u32, buffer: *mut c_void, length: usize) -> i32;
}
#[link(name = "advapi32")]
unsafe extern "system" {
    pub fn RegGetValueW(key: isize, subkey: *const u16, value: *const u16, flags: u32, kind: *mut u32, data: *mut c_void, size: *mut u32) -> i32;
}
pub const HKEY_LOCAL_MACHINE: isize = 0x8000_0002_u32 as i32 as isize;
pub const RRF_RT_REG_DWORD: u32 = 0x10;

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
}

// The audio recorder (xnw-hle `audio_capture`): waits for the title's process.
pub const PROCESS_SYNCHRONIZE: u32 = 0x0010_0000;
pub const WAIT_OBJECT_0: u32 = 0;

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn OpenProcess(access: u32, inherit: i32, id: u32) -> HANDLE;
    pub fn WaitForSingleObject(handle: HANDLE, ms: u32) -> u32;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn CreateMutexW(attributes: *const c_void, initial_owner: i32, name: *const u16) -> HANDLE;
}

// `xnw smoke`: is a title instance or a lock holder alive.
pub const SYNCHRONIZE: u32 = 0x0010_0000;
#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn OpenMutexW(access: u32, inherit: i32, name: *const u16) -> HANDLE;
    pub fn ReleaseMutex(mutex: HANDLE) -> i32;
}

// The frame lab's window (apps/xnw-lab): a message pump and a GDI blit
// of the read-back picture.
pub const PM_REMOVE: u32 = 1;
pub const DIB_RGB_COLORS: u32 = 0;

#[repr(C)]
pub struct BITMAPINFOHEADER {
    pub size: u32,
    pub width: i32,
    /// Negative: rows top-down.
    pub height: i32,
    pub planes: u16,
    pub bit_count: u16,
    pub compression: u32,
    pub size_image: u32,
    pub x_ppm: i32,
    pub y_ppm: i32,
    pub clr_used: u32,
    pub clr_important: u32,
}

#[link(name = "user32")]
unsafe extern "system" {
    pub fn PeekMessageW(message: *mut MSG, window: HANDLE, first: u32, last: u32, remove: u32) -> i32;
    pub fn PostQuitMessage(code: i32);
    pub fn GetDC(window: HANDLE) -> HANDLE;
    pub fn ReleaseDC(window: HANDLE, dc: HANDLE) -> i32;
}

#[link(name = "gdi32")]
unsafe extern "system" {
    pub fn SetDIBitsToDevice(dc: HANDLE, x: i32, y: i32, w: u32, h: u32, sx: i32, sy: i32, start: u32, lines: u32, bits: *const c_void, info: *const BITMAPINFOHEADER, usage: u32) -> i32;
}

// The loader's executable check: SHA-256 through CNG.
/// The SHA-256 pseudo-handle (Windows 10 and later).
pub const BCRYPT_SHA256_ALG_HANDLE: HANDLE = 0x41 as HANDLE;
#[link(name = "bcrypt")]
unsafe extern "system" {
    pub fn BCryptCreateHash(alg: HANDLE, hash: *mut HANDLE, object: *mut u8, object_len: u32, secret: *const u8, secret_len: u32, flags: u32) -> i32;
    pub fn BCryptHashData(hash: HANDLE, input: *const u8, input_len: u32, flags: u32) -> i32;
    pub fn BCryptFinishHash(hash: HANDLE, output: *mut u8, output_len: u32, flags: u32) -> i32;
    pub fn BCryptDestroyHash(hash: HANDLE) -> i32;
}

/// PERF5: whether our helper threads run below normal priority
/// (`XNW_HELPER_PRIO=normal`: at normal, as before; read once).
pub fn helper_below() -> bool {
    static NORMAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*NORMAL.get_or_init(|| matches!(std::env::var("XNW_HELPER_PRIO").as_deref().map(str::trim), Ok("normal" | "off" | "0")))
}

/// PERF5: the calling thread, one of our helpers (bridged prep and its
/// copy threads, the list recorder, the texture and geometry readers),
/// below the title's workers (`THREAD_PRIORITY_BELOW_NORMAL`) so that
/// it takes no core a title worker could run on; `XNW_HELPER_PRIO=normal`
/// leaves it at normal. The title's threads and our pass thread are not changed.
pub fn helper_thread() {
    if helper_below() {
        // SAFETY: a plain call on the calling thread's pseudo-handle.
        unsafe { SetThreadPriority(GetCurrentThread(), -1) };
    }
}
