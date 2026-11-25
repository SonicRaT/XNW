//! XNW: switches that keep listed console programs from a host
//! pipeline, to bisect a GPU hang to the programs that cause it. A refused
//! program's pipelines are not made, so its draws and dispatches drop and
//! are counted as failures ("refused by switch", logged once per switch).
//!
//! - `XNW_REFUSE=<hash prefix>,...` or `XNW_REFUSE=@<file>`: programs by
//!   the 128-bit hash the log names them with; a file holds one per line
//!   (the first word, or the text before `|`; `#` starts a comment).
//! - `XNW_REFUSE_NEW=<version>`: programs first translated by translator
//!   version `version` or later ([`super::shader_cache::first_version`]).
//! - `XNW_REFUSE_KIND=atomic64`: programs that use 64-bit image atomics.

use std::sync::OnceLock;

/// XNW: whether `XNW_LOOP_PROBE=<hash prefix>,...|@file` lists the
/// program `hex` (its pixel invocations that run out of loop budget write
/// magenta instead of discarding).
pub(crate) fn probed(hex: &str) -> bool {
    static LIST: OnceLock<Vec<String>> = OnceLock::new();
    LIST.get_or_init(|| {
        let list = parse(std::env::var("XNW_LOOP_PROBE").ok().as_deref(), None, None, |p| std::fs::read_to_string(p).ok()).prefixes;
        if !list.is_empty() {
            xnw_core::warn!("d3d12x: XNW_LOOP_PROBE: {} hash prefixes", list.len());
        }
        list
    })
    .iter()
    .any(|p| hex.starts_with(p.as_str()))
}

/// The switches, read once.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Refuse {
    pub prefixes: Vec<String>,
    pub new_from: Option<u32>,
    pub atomic64: bool,
}

impl Refuse {
    /// Whether any switch is set.
    pub(crate) fn any(&self) -> bool {
        !self.prefixes.is_empty() || self.new_from.is_some() || self.atomic64
    }

    /// Why the program `hex` is refused: the switch that refuses it.
    /// `first` gives the version that first translated it (`None`: never),
    /// `container` its console container; each is asked only if needed.
    pub(crate) fn why(
        &self,
        hex: &str,
        first: impl FnOnce() -> Option<u32>,
        container: impl FnOnce() -> Option<Vec<u8>>,
    ) -> Option<&'static str> {
        if self.prefixes.iter().any(|p| hex.starts_with(p.as_str())) {
            return Some("XNW_REFUSE");
        }
        if let Some(n) = self.new_from
            && first().unwrap_or(xnw_shader::emit::VERSION) >= n
        {
            return Some("XNW_REFUSE_NEW");
        }
        if self.atomic64 && container().is_some_and(|c| uses_atomic64(&c)) {
            return Some("XNW_REFUSE_KIND=atomic64");
        }
        None
    }
}

/// The switches from the environment (logged once when set).
pub(crate) fn switches() -> &'static Refuse {
    static R: OnceLock<Refuse> = OnceLock::new();
    R.get_or_init(|| {
        let var = |k: &str| std::env::var(k).ok();
        let r = parse(var("XNW_REFUSE").as_deref(), var("XNW_REFUSE_NEW").as_deref(), var("XNW_REFUSE_KIND").as_deref(), |p| std::fs::read_to_string(p).ok());
        if r.any() {
            xnw_core::warn!("d3d12x: refusing by switch: {} hash prefixes, new from translator {:?}, atomic64 {}", r.prefixes.len(), r.new_from, r.atomic64);
        }
        r
    })
}

/// The switches from their values; `read` reads an `@file`.
pub(crate) fn parse(list: Option<&str>, new: Option<&str>, kind: Option<&str>, read: impl Fn(&str) -> Option<String>) -> Refuse {
    let mut prefixes = Vec::new();
    let mut add = |s: &str| {
        let s = s.trim().to_ascii_lowercase();
        if !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit()) {
            prefixes.push(s);
        }
    };
    for item in list.unwrap_or("").split(',') {
        match item.trim().strip_prefix('@') {
            Some(file) => {
                for line in read(file).unwrap_or_default().lines() {
                    let line = line.split('#').next().unwrap_or("").trim();
                    add(line.split(['|', ' ', '\t']).next().unwrap_or(""));
                }
            }
            None => add(item),
        }
    }
    let new_from = new.and_then(|v| v.trim().parse().ok());
    let atomic64 = kind.is_some_and(|k| k.split(',').any(|k| k.trim() == "atomic64"));
    Refuse { prefixes, new_from, atomic64 }
}

/// Whether a console container's programs use 64-bit image atomics (two
/// dwords per value: dmask 0x3, cmpswap 0xF), as the translator lowers it.
pub(crate) fn uses_atomic64(container: &[u8]) -> bool {
    use xnw_shader::gfx10::{Fields, decode};
    let programs = xnw_shader::Container::parse(container)
        .ok()
        .and_then(|c| c.part(xnw_shader::FourCC::DXIL).map(|p| p.data))
        .and_then(|d| xnw_shader::native_programs(d).ok())
        .unwrap_or_default();
    programs.iter().any(|p| {
        decode(&p.code).unwrap_or_default().iter().any(|i| {
            let Some(name) = i.name().and_then(|n| n.strip_prefix("image_atomic_")) else {
                return false;
            };
            let wide = if name == "cmpswap" { 0xF } else { 0x3 };
            matches!(i.fields(p.wave64), Fields::Image { dmask, .. } if dmask == wide)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn refuse_switches_parse() {
        let none = |_: &str| None;
        assert!(!parse(None, None, None, none).any());
        let r = parse(Some(" 7E040F4D, d1cb2b87 ,,zz12"), Some("13"), Some("atomic64"), none);
        assert_eq!(r.prefixes, ["7e040f4d", "d1cb2b87"]);
        assert_eq!((r.new_from, r.atomic64), (Some(13), true));
        // A file: first word or text before `|`, `#` comments, plus a listed hash.
        let file = |p: &str| (p == "list.txt").then(|| "# refused\n5c2592b0cbb8772c|pixel|why\n  957fc3f6 old\n\nnot-hex\n".to_owned());
        let r = parse(Some("@list.txt,abc"), Some("x"), Some("other"), file);
        assert_eq!(r.prefixes, ["5c2592b0cbb8772c", "957fc3f6", "abc"]);
        assert_eq!((r.new_from, r.atomic64), (None, false));
        assert!(parse(Some("@missing.txt"), None, None, file).prefixes.is_empty());
    }

    #[test]
    fn refuse_picks_the_switch() {
        let r = parse(Some("7e04"), Some("13"), Some("atomic64"), |_| None);
        assert_eq!(r.why("7e040f4d", || None, || None), Some("XNW_REFUSE"));
        assert_eq!(r.why("11680c07", || Some(11), || None), None);
        assert_eq!(r.why("11680c07", || Some(13), || None), Some("XNW_REFUSE_NEW"));
        // Never translated here: new.
        assert_eq!(r.why("11680c07", || None, || None), Some("XNW_REFUSE_NEW"));
        assert_eq!(r.why("11680c07", || Some(12), || Some(Vec::new())), None);
    }
}
