//! The title's paired reservation as the frame side folds it: one memory
//! seen at a CPU view and again at a GPU view (Series S: CPU view
//! `0x1_0000_0000`, GPU view `0x5_0000_0000`, `0x4_0000_0000` bytes). The
//! title writes through the CPU view and places GPU resources in the GPU
//! view; captured views carry either address.
//!
//! xmem publishes the live pair once, when it pairs the two reservations
//! ([`publish`]); a capture records it in its manifest and a reader adopts
//! it from there ([`publish`] again, first one wins). Without either (the
//! captures before ALIAS1, a test) [`Pair::DEFAULT`] holds. GRP1: a test run
//! a test run placed the GPU view at `0x667200000` / `0x8df800000`, and every
//! fold by the fixed `0x4_0000_0000` missed.

use std::sync::OnceLock;

/// CPU view base, GPU view base, length (bytes) of the paired reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    pub cpu: u64,
    pub gpu: u64,
    pub len: u64,
}

impl Pair {
    /// The pair as the title's runs place it when nothing comes between the
    /// two reservations (V: a test run `paired reservation 0x100000000 <-> 0x500000000`).
    pub const DEFAULT: Pair = Pair { cpu: 0x1_0000_0000, gpu: 0x5_0000_0000, len: 0x4_0000_0000 };

    /// How far the GPU view lies above the CPU view.
    pub fn alias(self) -> u64 {
        self.gpu.wrapping_sub(self.cpu)
    }

    /// Whether `a` lies in the GPU view.
    pub fn is_gpu(self, a: u64) -> bool {
        a >= self.gpu && a - self.gpu < self.len
    }

    /// Whether `a` lies in the CPU view.
    pub fn is_cpu(self, a: u64) -> bool {
        a >= self.cpu && a - self.cpu < self.len
    }

    /// `a` with the GPU view folded onto the CPU view; any other address as it is.
    pub fn fold(self, a: u64) -> u64 {
        if self.is_gpu(a) { a - self.alias() } else { a }
    }

    /// The GPU view's address of CPU-view address `a`; any other address as it is.
    pub fn to_gpu(self, a: u64) -> u64 {
        if self.is_cpu(a) { a + self.alias() } else { a }
    }

    /// Whether `a` and `b` name the same bytes (either view).
    pub fn same(self, a: u64, b: u64) -> bool {
        self.fold(a) == self.fold(b)
    }
}

impl Default for Pair {
    fn default() -> Pair {
        Pair::DEFAULT
    }
}

static LIVE: OnceLock<Pair> = OnceLock::new();

/// Publishes the pair (xmem at reservation time, a reader from a capture's
/// manifest). The first one stays; false if `p` differs from it.
pub fn publish(p: Pair) -> bool {
    *LIVE.get_or_init(|| p) == p
}

/// The pair published, if any.
pub fn live() -> Option<Pair> {
    LIVE.get().copied()
}

/// The pair published, else [`Pair::DEFAULT`].
pub fn pair() -> Pair {
    LIVE.get().copied().unwrap_or(Pair::DEFAULT)
}

/// `a` folded by [`pair`] (GPU view onto CPU view).
pub fn fold(a: u64) -> u64 {
    pair().fold(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_either_view_onto_the_cpu_view() {
        let d = Pair::DEFAULT;
        assert_eq!(d.alias(), 0x4_0000_0000);
        assert_eq!(d.fold(0x5_0000_1000), 0x1_0000_1000);
        assert_eq!(d.fold(0x1_0000_1000), 0x1_0000_1000);
        assert_eq!(d.fold(0x9_0000_0000), 0x9_0000_0000);
        assert_eq!(d.to_gpu(0x1_0000_1000), 0x5_0000_1000);
        // a test run's placement (GRP1).
        let p = Pair { cpu: 0x1_0000_0000, gpu: 0x6_6720_0000, len: 0x4_0000_0000 };
        assert_eq!(p.fold(0x6_6720_4000), 0x1_0000_4000);
        assert_eq!(p.fold(0x5_0000_4000), 0x5_0000_4000);
        assert!(p.same(0x1_0000_4000, 0x6_6720_4000) && !p.same(0x1_0000_4000, 0x5_0000_4000));
        assert_eq!(p.to_gpu(0x1_0000_4000), 0x6_6720_4000);
    }
}
