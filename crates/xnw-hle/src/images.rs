//! What the runtime knows about the process: the title images we mapped
//! (for `GetModuleHandle` and friends, and to name callers in logs) and
//! the loader's diagnostics, handed in by whoever runs the title.

use std::ffi::c_void;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};

use xnw_core::win::RtlCaptureStackBackTrace;

/// One mapped title image: the file name the title asks for it by, its
/// base and size.
struct Image {
    file: String,
    base: usize,
    size: usize,
}

static IMAGES: RwLock<Vec<Image>> = RwLock::new(Vec::new());

/// Records a mapped title image under its file name.
pub fn add_image(file: &str, base: usize, size: usize) {
    let mut images = IMAGES.write().unwrap_or_else(|e| e.into_inner());
    images.retain(|i| i.base != base);
    images.push(Image { file: file.to_owned(), base, size });
}

/// Takes the image based at `base` off the list (a DLL the title loaded
/// whose `DllMain` failed, unmapped again).
pub fn forget_image(base: usize) {
    IMAGES.write().unwrap_or_else(|e| e.into_inner()).retain(|i| i.base != base);
}

/// The base of the title image whose file is `file` (any case).
pub(crate) fn image_named(file: &str) -> Option<usize> {
    let images = IMAGES.read().unwrap_or_else(|e| e.into_inner());
    images.iter().find(|i| i.file.eq_ignore_ascii_case(file)).map(|i| i.base)
}

/// The file name of the title image based at `module`.
pub(crate) fn image_file(module: usize) -> Option<String> {
    let images = IMAGES.read().unwrap_or_else(|e| e.into_inner());
    images.iter().find(|i| i.base == module).map(|i| i.file.clone())
}

/// Whether `module` is the base of a title image.
pub(crate) fn is_image(module: usize) -> bool {
    let images = IMAGES.read().unwrap_or_else(|e| e.into_inner());
    images.iter().any(|i| i.base == module)
}

fn in_image(address: usize) -> bool {
    let images = IMAGES.read().unwrap_or_else(|e| e.into_inner());
    images.iter().any(|i| address >= i.base && address < i.base + i.size)
}

/// Diagnostics the loader provides: naming code addresses and logging a
/// stack. Without them addresses print raw.
#[derive(Clone, Copy)]
pub struct Diagnostics {
    pub locate: fn(usize) -> String,
    pub log_stack: fn(),
}

static DIAGNOSTICS: RwLock<Option<Diagnostics>> = RwLock::new(None);
static TRAP: AtomicBool = AtomicBool::new(true);

/// Installs the loader's diagnostics; `trap` mirrors the unimplemented
/// policy: an unimplemented runtime method ends the run as an unimplemented
/// import does.
pub fn set_diagnostics(d: Diagnostics, trap: bool) {
    *DIAGNOSTICS.write().unwrap_or_else(|e| e.into_inner()) = Some(d);
    TRAP.store(trap, Ordering::Relaxed);
}

pub(crate) fn trap() -> bool {
    TRAP.load(Ordering::Relaxed)
}

pub(crate) fn locate(address: usize) -> String {
    match *DIAGNOSTICS.read().unwrap_or_else(|e| e.into_inner()) {
        Some(d) => (d.locate)(address),
        None => format!("{address:#x}"),
    }
}

pub(crate) fn log_stack() {
    if let Some(d) = *DIAGNOSTICS.read().unwrap_or_else(|e| e.into_inner()) {
        (d.log_stack)()
    }
}

/// The nearest title code on the calling thread's stack, named: who called
/// the runtime function we are in, and who called that (the runtime's
/// calls all pass through one thunk in the title).
pub(crate) fn caller() -> String {
    let mut frames = [std::ptr::null_mut::<c_void>(); 24];
    // SAFETY: fills our own array.
    let n = unsafe { RtlCaptureStackBackTrace(1, frames.len() as u32, frames.as_mut_ptr(), std::ptr::null_mut()) } as usize;
    let title: Vec<String> = frames[..n].iter().map(|&f| f as usize).filter(|&f| in_image(f)).take(3).map(locate).collect();
    if title.is_empty() { "?".to_owned() } else { title.join(" < ") }
}
