//! A generated table: the sites, by relative address, of the Zen 2 instructions
//! the host may lack in the executable whose SHA-256 is `SHA256`; for any other
//! file the loader rewrites nothing. Per site: the kind, the instruction's
//! length, the window (the instruction and any whole instructions moved into its
//! stub) and `targets`, bit n set when a branch or reference lands n bytes in.
//! Regenerated whole by the table generator under `tools/` from a site list and
//! the executable; hand edits are lost.

use super::{Kind, Site};

#[rustfmt::skip]
pub(super) const SHA256: [u8; 32] = [0xe9, 0x50, 0x20, 0xb2, 0xfa, 0x4a, 0x72, 0x1d, 0xb3, 0xdf, 0x59, 0xfd, 0xec, 0xf7, 0x56, 0x21, 0x73, 0x0a, 0xc5, 0x59, 0xd8, 0x83, 0x52, 0x52, 0xf6, 0xc3, 0xb6, 0x58, 0xd5, 0xa0, 0x89, 0x93];

#[rustfmt::skip]
pub(super) const SITES: &[Site] = &[
    Site { rva: 0x27c8f2, kind: Kind::Insertq, len: 7, window: 7, targets: 0x0 },
    Site { rva: 0x27e9f1, kind: Kind::Insertq, len: 7, window: 7, targets: 0x0 },
    Site { rva: 0x2e7ff1, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0x7f1ac9, kind: Kind::Insertq, len: 7, window: 7, targets: 0x0 },
    Site { rva: 0x7f1b29, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0x7f1b38, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0x7f1b44, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0xf75fb3, kind: Kind::Insertq, len: 7, window: 7, targets: 0x0 },
    Site { rva: 0xf762b7, kind: Kind::Insertq, len: 7, window: 7, targets: 0x0 },
    Site { rva: 0xf765b5, kind: Kind::Insertq, len: 7, window: 7, targets: 0x0 },
    Site { rva: 0xf768b0, kind: Kind::Insertq, len: 7, window: 7, targets: 0x0 },
    Site { rva: 0x269dcb2, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0x32bc150, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0x43d440e, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0x43d491b, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0x455bac7, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0x4e4b680, kind: Kind::Clwb, len: 4, window: 4, targets: 0x0 },
    Site { rva: 0x4e8aa70, kind: Kind::Clzero, len: 3, window: 7, targets: 0x0 },
    Site { rva: 0x4e8aa90, kind: Kind::Clzero, len: 3, window: 7, targets: 0x0 },
    Site { rva: 0x4e8aab0, kind: Kind::Clzero, len: 3, window: 7, targets: 0x0 },
    Site { rva: 0x4e8aad0, kind: Kind::Clzero, len: 3, window: 7, targets: 0x0 },
    Site { rva: 0x4e8aaf0, kind: Kind::Clzero, len: 3, window: 7, targets: 0x0 },
    Site { rva: 0x4e8ab2d, kind: Kind::Clzero, len: 3, window: 7, targets: 0x0 },
    Site { rva: 0x4e8ab53, kind: Kind::Clzero, len: 3, window: 6, targets: 0x100 },
    Site { rva: 0x4e8ab72, kind: Kind::Clzero, len: 3, window: 6, targets: 0x100 },
    Site { rva: 0x4e8ab91, kind: Kind::Clzero, len: 3, window: 6, targets: 0x100 },
    Site { rva: 0x4e8abb4, kind: Kind::Clzero, len: 3, window: 6, targets: 0x0 },
    Site { rva: 0x4ecc84a, kind: Kind::Clwb, len: 5, window: 5, targets: 0x0 },
    Site { rva: 0x4fb0bd7, kind: Kind::Monitorx, len: 3, window: 3, targets: 0x0 },
    Site { rva: 0x4fb0c00, kind: Kind::Mwaitx, len: 3, window: 3, targets: 0x0 },
    Site { rva: 0x4fb0d21, kind: Kind::Monitorx, len: 3, window: 3, targets: 0x0 },
    Site { rva: 0x4fb0d52, kind: Kind::Mwaitx, len: 3, window: 3, targets: 0x0 },
    Site { rva: 0x5084782, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0x50f75f5, kind: Kind::Clwb, len: 5, window: 5, targets: 0x0 },
    Site { rva: 0x532f1b0, kind: Kind::Clwb, len: 4, window: 4, targets: 0x0 },
    Site { rva: 0x532f1f0, kind: Kind::Clwb, len: 4, window: 4, targets: 0x0 },
    Site { rva: 0x532f230, kind: Kind::Clwb, len: 4, window: 4, targets: 0x0 },
    Site { rva: 0x532f280, kind: Kind::Clwb, len: 4, window: 4, targets: 0x0 },
    Site { rva: 0x532f2ba, kind: Kind::Clwb, len: 4, window: 4, targets: 0x0 },
    Site { rva: 0x546d4a2, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
    Site { rva: 0x55b1811, kind: Kind::Insertq, len: 6, window: 6, targets: 0x0 },
];
