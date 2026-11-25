//! The console the title sees: an Xbox Series S unless `XNW_CONSOLE` says
//! Series X. Titles pick resolution and memory budgets by it; the host's own
//! figures never reach the title. Every place that tells the title what it
//! runs on (the registry's `ConsoleMode`, XSystem analytics, the DXGI
//! adapter, XMem's budget) reads [`current`], so it cannot see a mix.
//!
//! Ported from VIctory `crates/victory-gpu/src/console.rs` at d2a4bae
//! (GPL-3.0-or-later). Changed: Series S is the default.

use std::sync::atomic::{AtomicBool, Ordering};

const GIB: usize = 1 << 30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Console {
    #[default]
    SeriesS,
    SeriesX,
}

impl Console {
    /// The setting's spelling: `series-s` or `series-x`.
    pub fn parse(text: &str) -> Option<Console> {
        match text.trim().to_ascii_lowercase().as_str() {
            "series-s" | "s" => Some(Console::SeriesS),
            "series-x" | "x" => Some(Console::SeriesX),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Console::SeriesS => "Xbox Series S",
            Console::SeriesX => "Xbox Series X",
        }
    }

    /// Title memory in bytes: (GPU-optimal, standard). Series X in the
    /// advanced configuration; Series S has one range, counted as standard
    /// (7,776 MiB, the published console-mode figure; VIctory notes later
    /// system updates gave titles more, amount unknown).
    pub const fn title_memory(self) -> (usize, usize) {
        match self {
            Console::SeriesS => (0, 7_776 << 20),
            Console::SeriesX => (10 * GIB, 7 * GIB / 2),
        }
    }

    /// `ConsoleMode` under `HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM`
    /// as the title's device-type read (`title+0x55deb88`) maps it: 0x60 ->
    /// type 6 (XboxScarlettLockhart), 0x64 -> type 7 (XboxScarlettAnaconda).
    pub const fn console_mode(self) -> u32 {
        match self {
            Console::SeriesS => 0x60,
            Console::SeriesX => 0x64,
        }
    }

    /// The GDK `XSystemDeviceType` value.
    pub const fn device_type(self) -> u32 {
        match self {
            Console::SeriesS => 6,
            Console::SeriesX => 7,
        }
    }
}

static SERIES_X: AtomicBool = AtomicBool::new(false);

/// Set before the title starts.
pub fn set(console: Console) {
    SERIES_X.store(console == Console::SeriesX, Ordering::Relaxed);
}

pub fn current() -> Console {
    if SERIES_X.load(Ordering::Relaxed) { Console::SeriesX } else { Console::SeriesS }
}

/// Feature level 12_2, what the console hardware supports (both models).
pub const FEATURE_LEVEL: u32 = 0xc200;

/// Feature query 0 as an RDNA 2 GPU reports it. The console's layout has
/// one field more than PC's (index 11).
pub const OPTIONS: [u32; 16] = [
    1,  // double-precision shader operations
    1,  // output-merger logic operations
    2,  // minimum precision: 16-bit
    3,  // tiled resources tier
    3,  // resource binding tier
    1,  // stencil reference from pixel shaders
    1,  // typed UAV loads of more formats
    1,  // rasterizer-ordered views
    3,  // conservative rasterization tier
    40, // GPU address bits per resource
    0,  // 64 KiB standard swizzle
    0,  // console-only field
    0,  // cross-node sharing tier
    0,  // cross-adapter row-major textures
    1,  // viewport and render target index from any stage
    2,  // resource heap tier
];

/// AMD's PCI vendor id, which the adapter reports.
pub const VENDOR: u32 = 0x1002;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_setting_parses_and_maps() {
        assert_eq!(Console::parse("series-s"), Some(Console::SeriesS));
        assert_eq!(Console::parse(" Series-X "), Some(Console::SeriesX));
        assert_eq!(Console::parse("pc"), None);
        assert_eq!(Console::default(), Console::SeriesS);
        assert_eq!((Console::SeriesS.console_mode(), Console::SeriesS.device_type()), (0x60, 6));
        assert_eq!((Console::SeriesX.console_mode(), Console::SeriesX.device_type()), (0x64, 7));
    }
}
