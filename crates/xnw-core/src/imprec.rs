//! The import recording's format and the table of pure calls, shared
//! by the recorder (`xnw-loader` `record`, `XNW_IMPORT_TRACE`) and the
//! replay test (`xnw-hle/tests/import_replay.rs`).
//!
//! A recording is text, one record per line, tab-separated:
//!
//! ```text
//! # xnw import trace v1
//! R  dll  symbol  provider(H ours|W host)  caller  args  ret  le_before  le_after  bytes
//! C  dll  symbol  calls  distinct
//! ```
//!
//! `args` is a comma list: a scalar in hex, `0` for null, `p` for a
//! pointer-like value (an unknown call's argument at or above 0x10000).
//! For a call in [`PURE`] every argument is as its kind says and `bytes`
//! holds the memory it reads (`i<n>=<hex>`) and writes (`o<n>=<hex>`),
//! `;`-separated. One record per distinct (call, caller, arguments,
//! result, last errors, bytes), at most [`PER_SITE`] per import; `C` lines
//! (written when the recording ends) count every call.

/// What an argument of a pure call is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arg {
    /// A value, recorded exactly.
    S,
    /// A pointer to `n` bytes the call writes.
    Out(usize),
    /// A pointer to a buffer of `args[len] * mult` bytes; the call writes
    /// `ret * mult` of them (string results).
    OutR { len: u8, mult: u8 },
    /// A pointer to a buffer whose size in bytes is the `u32` at
    /// `args[len]` before the call; the call writes the `u32` after.
    OutLenPtr { len: u8 },
    /// A pointer to a `u32` the call reads and writes.
    InOut4,
    /// A byte string read: `args[len]` bytes, or up to its NUL (included)
    /// when `len` is negative or `args[len]` is -1.
    InA(i8),
    /// The same in UTF-16 units.
    InW(i8),
    /// A pointer to `n` bytes the call may write, not compared (a lock,
    /// a clock value).
    Scratch(usize),
}

/// How a pure call's result is compared.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ret {
    Exact,
    /// A `BOOL`: the low 32 bits, 0 or 1.
    Bool,
    /// Changes call to call (clocks, counters, state): not compared.
    Volatile,
    Void,
}

pub struct Pure {
    pub name: &'static str,
    pub args: &'static [Arg],
    pub ret: Ret,
}

use Arg::*;

const fn p(name: &'static str, args: &'static [Arg], ret: Ret) -> Pure {
    Pure { name, args, ret }
}

/// Calls whose result depends only on their arguments and the machine:
/// the replay test issues them again (against our shim when we bind one,
/// and the host's too) and compares result, last error and bytes.
pub const PURE: &[Pure] = &[
    p("GetSystemInfo", &[Out(48)], Ret::Void),
    p("GetNativeSystemInfo", &[Out(48)], Ret::Void),
    p("QueryPerformanceFrequency", &[Out(8)], Ret::Bool),
    p("QueryPerformanceCounter", &[Scratch(8)], Ret::Bool),
    p("GetTickCount", &[], Ret::Volatile),
    p("GetTickCount64", &[], Ret::Volatile),
    p("GetSystemTimeAsFileTime", &[Scratch(8)], Ret::Void),
    p("GetSystemTimePreciseAsFileTime", &[Scratch(8)], Ret::Void),
    p("GetLogicalProcessorInformation", &[OutLenPtr { len: 1 }, InOut4], Ret::Bool),
    p("GetLogicalProcessorInformationEx", &[S, OutLenPtr { len: 2 }, InOut4], Ret::Bool),
    p("IsProcessorFeaturePresent", &[S], Ret::Bool),
    p("GetActiveProcessorCount", &[S], Ret::Exact),
    p("GetMaximumProcessorCount", &[S], Ret::Exact),
    p("GetActiveProcessorGroupCount", &[], Ret::Exact),
    p("GetMaximumProcessorGroupCount", &[], Ret::Exact),
    p("GetCurrentProcess", &[], Ret::Exact),
    p("GetCurrentThread", &[], Ret::Exact),
    p("GetCurrentProcessorNumber", &[], Ret::Volatile),
    p("IsDebuggerPresent", &[], Ret::Bool),
    p("GetErrorMode", &[], Ret::Volatile),
    p("SetErrorMode", &[S], Ret::Volatile),
    p("InitializeCriticalSectionAndSpinCount", &[Scratch(40), S], Ret::Bool),
    p("InitializeCriticalSectionEx", &[Scratch(40), S, S], Ret::Bool),
    p("InitializeSRWLock", &[Scratch(8)], Ret::Void),
    p("InitializeConditionVariable", &[Scratch(8)], Ret::Void),
    p("TlsAlloc", &[], Ret::Volatile),
    p("GetACP", &[], Ret::Exact),
    p("GetOEMCP", &[], Ret::Exact),
    p("GetUserDefaultLCID", &[], Ret::Exact),
    p("GetUserDefaultLangID", &[], Ret::Exact),
    p("GetSystemDefaultLangID", &[], Ret::Exact),
    p("GetUserDefaultUILanguage", &[], Ret::Exact),
    p("GetThreadLocale", &[], Ret::Exact),
    p("GetUserDefaultLocaleName", &[OutR { len: 1, mult: 2 }, S], Ret::Exact),
    p("GetLocaleInfoEx", &[InW(-1), S, OutR { len: 3, mult: 2 }, S], Ret::Exact),
    p("MultiByteToWideChar", &[S, S, InA(3), S, OutR { len: 5, mult: 2 }, S], Ret::Exact),
    p("WideCharToMultiByte", &[S, S, InW(3), S, OutR { len: 5, mult: 1 }, S, InA(-1), Scratch(4)], Ret::Exact),
    p("CompareStringOrdinal", &[InW(1), S, InW(3), S, S], Ret::Exact),
    p("GetLastError", &[], Ret::Volatile),
    p("SetLastError", &[S], Ret::Void),
];

/// Distinct records kept per import.
pub const PER_SITE: u64 = 64;
/// Bytes kept of one buffer.
pub const MAX_BYTES: usize = 4096;

pub fn pure(name: &str) -> Option<(usize, &'static Pure)> {
    PURE.iter().enumerate().find(|(_, p)| p.name == name)
}

/// `BOOL` results we know of besides [`PURE`]'s, for the invariants.
pub const BOOLS: &[&str] = &[
    "VirtualFree", "VirtualProtect", "CloseHandle", "SetEvent", "ResetEvent", "ReleaseMutex", "ReadFile", "WriteFile",
    "SetThreadPriority", "SetFileAttributesW", "DeleteFileW", "CreateDirectoryW", "RemoveDirectoryW", "MoveFileExW",
    "GetFileAttributesExW", "FreeLibrary", "DisableThreadLibraryCalls", "SetThreadGroupAffinity", "FindClose", "FindNextFileW",
    "GetExitCodeThread", "DuplicateHandle", "SetEndOfFile", "FlushFileBuffers", "SetFilePointerEx", "GetFileSizeEx",
    "SetThreadAffinityMask", "GetThreadTimes", "TlsSetValue", "TlsFree", "FlsSetValue", "FlsFree", "HeapFree",
];

/// Calls that set the last error on success too (documented), for the
/// invariants.
pub const SETS_ERROR_ON_SUCCESS: &[&str] = &[
    "CreateFileW", "CreateFileA", "CreateFile2", "CreateEventW", "CreateEventExW", "CreateMutexW", "CreateMutexExW", "CreateSemaphoreW",
    "CreateSemaphoreExW", "CreateFileMappingW", "TlsGetValue", "FlsGetValue", "GetModuleFileNameW", "GetModuleFileNameA",
    "GetEnvironmentVariableW", "GetEnvironmentVariableA", "SetLastError", "GetLastError", "FindFirstFileW", "FindNextFileW",
    "VirtualQuery", "GetFullPathNameW",
];

/// One `R` line.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub dll: String,
    pub symbol: String,
    /// `H` our implementation or override, `W` the host's.
    pub provider: char,
    pub caller: String,
    /// `None` for `p` (a pointer of an unknown call), else the value.
    pub args: Vec<Option<u64>>,
    pub ret: u64,
    pub le_before: u32,
    pub le_after: u32,
    /// `(input?, argument, bytes)`.
    pub bytes: Vec<(bool, usize, Vec<u8>)>,
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    (0..s.len()).step_by(2).map(|i| s.get(i..i + 2).and_then(|h| u8::from_str_radix(h, 16).ok())).collect()
}

fn num(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
}

impl Record {
    pub fn line(&self) -> String {
        let args: Vec<String> = self.args.iter().map(|a| a.map_or("p".into(), |v| format!("{v:#x}"))).collect();
        let bytes: Vec<String> = self.bytes.iter().map(|(i, n, b)| format!("{}{n}={}", if *i { 'i' } else { 'o' }, hex(b))).collect();
        format!(
            "R\t{}\t{}\t{}\t{}\t{}\t{:#x}\t{}\t{}\t{}",
            self.dll,
            self.symbol,
            self.provider,
            self.caller,
            args.join(","),
            self.ret,
            self.le_before,
            self.le_after,
            bytes.join(";")
        )
    }

    pub fn parse(line: &str) -> Option<Record> {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 10 || f[0] != "R" {
            return None;
        }
        let args = if f[5].is_empty() { Vec::new() } else { f[5].split(',').map(|a| if a == "p" { Some(None) } else { num(a).map(Some) }).collect::<Option<Vec<_>>>()? };
        let bytes = if f[9].is_empty() {
            Vec::new()
        } else {
            f[9].split(';')
                .map(|b| {
                    let (k, v) = b.split_once('=')?;
                    let input = k.starts_with('i');
                    Some((input, k[1..].parse().ok()?, unhex(v)?))
                })
                .collect::<Option<Vec<_>>>()?
        };
        Some(Record {
            dll: f[1].into(),
            symbol: f[2].into(),
            provider: f[3].chars().next()?,
            caller: f[4].into(),
            args,
            ret: num(f[6])?,
            le_before: f[7].parse().ok()?,
            le_after: f[8].parse().ok()?,
            bytes,
        })
    }

    pub fn input(&self, arg: usize) -> Option<&[u8]> {
        self.bytes.iter().find(|(i, n, _)| *i && *n == arg).map(|(_, _, b)| b.as_slice())
    }

    pub fn output(&self, arg: usize) -> Option<&[u8]> {
        self.bytes.iter().find(|(i, n, _)| !*i && *n == arg).map(|(_, _, b)| b.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_survives_its_line() {
        let r = Record {
            dll: "kernel32.dll".into(),
            symbol: "GetLogicalProcessorInformation".into(),
            provider: 'H',
            caller: "title+0x1000".into(),
            args: vec![Some(0), None],
            ret: 0,
            le_before: 0,
            le_after: 122,
            bytes: vec![(true, 1, vec![0, 0, 0, 0]), (false, 1, vec![0x60, 1, 0, 0])],
        };
        assert_eq!(Record::parse(&r.line()), Some(r.clone()));
        assert_eq!(r.output(1), Some(&[0x60, 1, 0, 0][..]));
        assert!(pure("GetSystemInfo").is_some() && pure("CreateFileW").is_none());
        for p in PURE {
            for a in p.args {
                if let Arg::OutR { len, .. } | Arg::OutLenPtr { len } = a {
                    assert!((*len as usize) < p.args.len(), "{}", p.name);
                }
            }
        }
    }
}
