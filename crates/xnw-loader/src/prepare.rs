//! Turns a title image into a copy the Windows loader maps but does not run.
//!
//! Adapted from VIctory, `crates/victory-loader/src/prepare.rs` at d2a4bae
//! (GPL-3.0-or-later).
//!
//! The copy is marked as a DLL, has no entry point, an empty import
//! directory and no TLS callbacks. Windows then maps it, relocates it,
//! allocates its static TLS and registers its unwind tables (so debuggers
//! and stack walks see a real module), runs none of its code and resolves
//! none of its imports. We do those ourselves, from the original's tables.

use xnw_pe::{Image, PeError, dir};

const FORCE_INTEGRITY: u16 = 0x0080;
const IMPORT_DESCRIPTOR_SIZE: u32 = 20;
const BOUND_IMPORT: usize = 11;

fn put(bytes: &mut [u8], at: usize, v: &[u8]) {
    bytes[at..at + v.len()].copy_from_slice(v);
}

/// Returns the patched copy of `original`.
pub fn prepare(original: &[u8]) -> Result<Vec<u8>, PeError> {
    let img = Image::parse(original)?;
    let mut bytes = original.to_vec();
    let opt = img.optional_header;

    put(&mut bytes, img.file_header + 18, &(img.characteristics | 0x2000).to_le_bytes());
    put(&mut bytes, opt + 16, &0u32.to_le_bytes()); // entry point
    put(&mut bytes, opt + 64, &0u32.to_le_bytes()); // checksum
    put(&mut bytes, opt + 70, &(img.dll_characteristics & !FORCE_INTEGRITY).to_le_bytes());

    // Point the import directory at its own terminator: an empty table.
    // Windows 10 only assigns static TLS to a DLL that has one.
    let (import_rva, _) = img.dirs[dir::IMPORT];
    if import_rva != 0 {
        let modules = img.imports()?.iter().filter(|m| !m.delay).count() as u32;
        let at = img.dir_offset(dir::IMPORT);
        put(&mut bytes, at, &(import_rva + modules * IMPORT_DESCRIPTOR_SIZE).to_le_bytes());
        put(&mut bytes, at + 4, &IMPORT_DESCRIPTOR_SIZE.to_le_bytes());
    }
    put(&mut bytes, img.dir_offset(BOUND_IMPORT), &[0; 8]);

    if let Some(tls) = img.tls()?
        && tls.callbacks_va != 0
    {
        let field_rva = tls.directory_rva + 24;
        put(&mut bytes, img.offset(field_rva)?, &0u64.to_le_bytes());
        // Its base relocation would turn the 0 into the load delta.
        for (entry, target) in img.dir64_relocations() {
            if target == field_rva {
                put(&mut bytes, entry, &0u16.to_le_bytes());
            }
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_own_test_binary() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let before = Image::parse(&original).unwrap();
        let copy = prepare(&original).unwrap();
        let after = Image::parse(&copy).unwrap();
        assert!(!before.is_dll() && after.is_dll());
        assert_eq!(after.entry_rva, 0);
        assert!(after.imports().unwrap().iter().all(|m| m.delay));
        assert_eq!(copy.len(), original.len());
        if let Some(tls) = after.tls().unwrap() {
            assert_eq!(tls.callbacks_va, 0);
            let field = tls.directory_rva + 24;
            assert!(after.dir64_relocations().iter().all(|&(_, t)| t != field));
        }
    }
}
