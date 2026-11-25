//! Loading a prepared image as a real Windows module and binding it.
//!
//! Adapted from VIctory, `crates/victory-loader/src/windows.rs` at d2a4bae
//! (GPL-3.0-or-later).

use std::ffi::{CString, c_void};
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;

use xnw_pe::{Image, ImportModule, Symbol};

use crate::stubs::{Site, Thunks};
use xnw_core::win::*;

/// Who provides a bound import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// Our implementation of a console API.
    Hle,
    /// Windows' own function.
    Host,
    /// A DLL the title ships, mapped like the executable.
    Title,
}

/// Gives an address for `dll!symbol`, or `None` for a naming stub.
pub type Resolver<'a> = dyn Fn(&str, &Symbol) -> Option<(usize, Provider)> + 'a;

/// A title image mapped into this process.
pub struct Loaded {
    /// `title` for the executable, the file name for a DLL.
    pub name: String,
    module: HMODULE,
    pub size: usize,
    pub is_dll: bool,
    pub entry_rva: u32,
    pub stack_reserve: u64,
    pub tls_callbacks: Vec<u32>,
    pub imports: Vec<ImportModule>,
}

// SAFETY: a module handle is a process-wide base address.
unsafe impl Send for Loaded {}
// SAFETY: as above; nothing in `Loaded` changes after `load`.
unsafe impl Sync for Loaded {}

#[derive(Default)]
pub struct BindReport {
    pub hle: usize,
    pub host: usize,
    pub title: usize,
    pub traced: usize,
    /// Imports nobody provides, as `(dll, symbol)`.
    pub missing: Vec<(String, String)>,
    /// Every thunk made, with its call counter.
    pub sites: Vec<&'static Site>,
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if std::fs::metadata(path).is_ok_and(|m| m.len() == bytes.len() as u64) && std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(());
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Prepares the image at `path`, keeps the copy in `cache` and maps it with
/// the Windows loader. Runs none of its code.
pub fn load(path: &Path, cache: &Path) -> io::Result<Loaded> {
    let original = std::fs::read(path)?;
    // A title DLL that fails to prepare is named in the error (VIctory 5f29f3a).
    let preparing = |e| io::Error::other(format!("preparing {}: {e}", path.display()));
    let img = Image::parse(&original).map_err(preparing)?;
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let copy_path = cache.join(format!("{stem}-{:08x}-{:x}.dll", img.timestamp, original.len()));
    std::fs::create_dir_all(cache)?;
    // The file's length and modification time (ns since 1970): a hash kept
    // for the same pair is not computed again.
    let stamp = std::fs::metadata(path).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| format!("{} {}", original.len(), d.as_nanos()));
    let note = copy_path.with_extension("sha256");
    std::thread::scope(|scope| -> io::Result<()> {
        // The executable's SHA-256 picks the CPU rewrite table: hashed on a
        // second thread while the copy is prepared and written.
        if !img.is_dll() {
            scope.spawn(|| crate::rewrite::identify(&original, stamp.as_deref(), &note));
        }
        let copy = crate::prepare::prepare(&original).map_err(preparing)?;
        write_if_changed(&copy_path, &copy)
    })?;

    let wide: Vec<u16> = copy_path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: a NUL-terminated path; the copy has no entry point and no
    // imports, so loading it runs no title code.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), 0) };
    if module.is_null() {
        return Err(io::Error::other(format!("LoadLibraryExW({}): {}", copy_path.display(), last_error())));
    }
    let tls = img.tls().map_err(io::Error::other)?;
    let has_tls = tls.is_some();
    let loaded = Loaded {
        name: if img.is_dll() { path.file_name().unwrap_or_default().to_string_lossy().into_owned() } else { "title".to_owned() },
        module,
        size: img.size_of_image as usize,
        is_dll: img.is_dll(),
        entry_rva: img.entry_rva,
        stack_reserve: img.stack_reserve,
        tls_callbacks: tls.map(|t| t.callback_rvas).unwrap_or_default(),
        imports: img.imports().map_err(io::Error::other)?,
    };
    crate::crash::register(&loaded.name, loaded.base(), loaded.size);
    crate::threads::mapped(crate::threads::Record {
        name: loaded.name.clone(),
        base: loaded.base(),
        is_dll: loaded.is_dll,
        has_tls,
        entry: if loaded.entry_rva != 0 { loaded.base() + loaded.entry_rva as usize } else { 0 },
        callbacks: loaded.tls_callbacks.iter().map(|&rva| loaded.base() + rva as usize).collect(),
    });
    xnw_core::info!("mapped {} at {:#x} (preferred {:#x}, {:#x} bytes)", loaded.name, loaded.base(), img.image_base, loaded.size);
    // A DLL indexes its TLS properly; only an exe assumes index 0.
    if !loaded.is_dll {
        // SAFETY: just mapped, nothing of it has run.
        unsafe { crate::tls::redirect(loaded.base(), &img)? };
    }
    Ok(loaded)
}

impl Loaded {
    pub fn base(&self) -> usize {
        self.module as usize
    }

    /// Address of one of the image's exports.
    pub fn export(&self, symbol: &Symbol) -> Option<usize> {
        let p = match symbol {
            Symbol::Name(n) => {
                let n = CString::new(n.as_str()).ok()?;
                // SAFETY: valid module and NUL-terminated name.
                unsafe { GetProcAddress(self.module, n.as_ptr().cast()) }
            }
            // SAFETY: an ordinal passed as the low word of the name pointer.
            Symbol::Ordinal(o) => unsafe { GetProcAddress(self.module, *o as usize as *const u8) },
        };
        (!p.is_null()).then_some(p as usize)
    }

    /// Fills every import slot, delay-load ones included (so the title's own
    /// delay-load helper never asks Windows). Imports `resolve` does not
    /// provide get a stub that names them; provided ones for which `trace`
    /// says yes get a logging thunk in front.
    pub fn bind(&self, resolve: &Resolver<'_>, trace: &dyn Fn(&str, &str) -> bool) -> io::Result<BindReport> {
        let mut report = BindReport::default();
        let total: usize = self.imports.iter().map(|m| m.imports.len()).sum();
        let mut thunks = Thunks::new(total)?;
        let mut slots = Vec::with_capacity(total);
        for module in &self.imports {
            for import in &module.imports {
                let symbol = import.symbol.to_string();
                let address = match resolve(&module.dll, &import.symbol) {
                    Some((address, provider)) => {
                        match provider {
                            Provider::Hle => report.hle += 1,
                            Provider::Host => report.host += 1,
                            Provider::Title => report.title += 1,
                        }
                        if trace(&module.dll, &symbol) {
                            report.traced += 1;
                            thunks.traced(&module.dll, &symbol, address)
                        } else if provider != Provider::Title && crate::record::wanted(&symbol) {
                            // XNW_IMPORT_TRACE.
                            thunks.recorded(&module.dll, &symbol, address, provider == Provider::Hle)
                        } else {
                            address
                        }
                    }
                    None => {
                        report.missing.push((module.dll.clone(), symbol.clone()));
                        thunks.unimplemented(&module.dll, &symbol)
                    }
                };
                slots.push((import.slot_rva, address));
            }
        }
        report.sites = thunks.seal()?;
        // Each slot on its own: import tables can sit in sections with
        // different protections.
        for (rva, address) in slots {
            let slot = (self.base() + rva as usize) as *mut u64;
            let mut old = 0;
            // SAFETY: the slot is inside the mapped image; made writable
            // for the write and restored after.
            unsafe {
                if VirtualProtect(slot as *const c_void, 8, PAGE_READWRITE, &mut old) == 0 {
                    return Err(last_error());
                }
                slot.write_volatile(address as u64);
                VirtualProtect(slot as *const c_void, 8, old, &mut old);
            }
        }
        xnw_core::info!(
            "bound {}: {} hle, {} host, {} title-dll, {} missing ({} traced)",
            self.name,
            report.hle,
            report.host,
            report.title,
            report.missing.len(),
            report.traced
        );
        Ok(report)
    }

    /// # Safety
    /// Runs title code; imports must be bound.
    unsafe fn tls_callbacks(&self, reason: u32) {
        for &rva in &self.tls_callbacks {
            type Callback = unsafe extern "system" fn(*mut c_void, u32, *mut c_void);
            // SAFETY: the RVA came from the image's TLS directory.
            let f: Callback = unsafe { std::mem::transmute(self.base() + rva as usize) };
            // SAFETY: the caller's guarantee.
            unsafe { f(self.module, reason, std::ptr::null_mut()) };
        }
    }

    /// Initialises a title DLL: TLS callbacks, then `DllMain`.
    ///
    /// # Safety
    /// Runs title code; imports must be bound, dependencies attached first.
    pub unsafe fn attach(&self) -> io::Result<()> {
        // SAFETY: the caller's guarantee.
        unsafe { self.tls_callbacks(DLL_PROCESS_ATTACH) };
        if self.is_dll && self.entry_rva != 0 {
            type DllMain = unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32;
            // SAFETY: the entry RVA came from the original header.
            let main: DllMain = unsafe { std::mem::transmute(self.base() + self.entry_rva as usize) };
            // SAFETY: the caller's guarantee.
            if unsafe { main(self.module, DLL_PROCESS_ATTACH, std::ptr::null_mut()) } == 0 {
                return Err(io::Error::other(format!("{}: DllMain returned FALSE", self.name)));
            }
        }
        crate::threads::attached(self.base());
        Ok(())
    }

    /// Undoes [`attach`](Self::attach) for a DLL whose own or a later
    /// image's `DllMain` failed while `LoadLibrary` mapped it: its TLS
    /// callbacks and `DllMain` hear `DLL_PROCESS_DETACH`.
    ///
    /// # Safety
    /// Runs title code; the image was attached, or failed to, just now.
    pub unsafe fn detach(&self) {
        // SAFETY: the caller's guarantee.
        unsafe { self.tls_callbacks(DLL_PROCESS_DETACH) };
        if self.is_dll && self.entry_rva != 0 {
            type DllMain = unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32;
            // SAFETY: the entry RVA came from the original header.
            let main: DllMain = unsafe { std::mem::transmute(self.base() + self.entry_rva as usize) };
            // SAFETY: the caller's guarantee.
            unsafe { main(self.module, DLL_PROCESS_DETACH, std::ptr::null_mut()) };
        }
    }

    /// Unmaps the image and forgets it (thread notices, crash names).
    ///
    /// # Safety
    /// Detached, and no code of it runs or is referenced any more.
    pub unsafe fn unload(&self) {
        crate::threads::unmapped(self.base());
        crate::crash::unregister(self.base());
        // SAFETY: the caller's guarantee; drops the reference `load` took.
        unsafe { FreeLibrary(self.module) };
    }

    /// Runs the executable: TLS callbacks, then its entry point, on a new
    /// thread with the stack the image asks for. Returns the entry point's
    /// result if it returns.
    ///
    /// # Safety
    /// Runs the title. Imports must be bound and title DLLs attached.
    pub unsafe fn run(&'static self) -> io::Result<u32> {
        let stack = (self.stack_reserve as usize).max(1 << 20);
        let thread = std::thread::Builder::new().name("title-main".into()).stack_size(stack).spawn(move || {
            xnw_core::info!("entering title at {}+{:#x}", self.name, self.entry_rva);
            // SAFETY: the caller's guarantee.
            unsafe { self.tls_callbacks(DLL_PROCESS_ATTACH) };
            // From here threads started by anyone notify the exe too.
            crate::threads::attached(self.base());
            type Entry = unsafe extern "system" fn() -> u32;
            // SAFETY: the entry RVA came from the original header.
            let entry: Entry = unsafe { std::mem::transmute(self.base() + self.entry_rva as usize) };
            // SAFETY: the caller's guarantee.
            unsafe { entry() }
        })?;
        thread.join().map_err(|_| io::Error::other("title main thread panicked"))
    }
}

/// Address of `symbol` in Windows' own `dll` (loaded on first use; API-set
/// names resolve through the system's schema).
pub fn host_export(dll: &str, symbol: &Symbol) -> Option<usize> {
    let name = CString::new(dll).ok()?;
    // SAFETY: NUL-terminated name of a system DLL; system DLLs are trusted.
    let module = unsafe { LoadLibraryA(name.as_ptr().cast()) };
    if module.is_null() {
        return None;
    }
    let p = match symbol {
        Symbol::Name(n) => {
            let n = CString::new(n.as_str()).ok()?;
            // SAFETY: valid module and NUL-terminated name.
            unsafe { GetProcAddress(module, n.as_ptr().cast()) }
        }
        // SAFETY: an ordinal passed as the low word of the name pointer.
        Symbol::Ordinal(o) => unsafe { GetProcAddress(module, *o as usize as *const u8) },
    };
    (!p.is_null()).then_some(p as usize)
}

#[cfg(test)]
mod tests {
    /// A title DLL that cannot be prepared is named in the error (VIctory 5f29f3a).
    #[test]
    fn a_title_dll_that_does_not_prepare_is_named() {
        let dir = std::env::temp_dir().join(format!("xnw-image-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("broken.dll");
        std::fs::write(&path, b"MZ not a PE").unwrap();
        let error = super::load(&path, &dir.join("cache")).err().expect("not a PE").to_string();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(error.contains("broken.dll"), "{error}");
    }
}
