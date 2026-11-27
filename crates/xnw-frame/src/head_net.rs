//! HEAD2: the ped head network on the CPU. The title evaluates a ped's face
//! network (`compute_graph_ops:*`, ~185 small compute dispatches a head a
//! frame over one f16 arena) on the GPU; `head_graph` replays those
//! dispatches translated (the oracle). Here the same op list runs on the
//! CPU: 13 op kinds, each written from its program's disassembly
//! and held to the oracle sink by
//! sink (`xnw-lab headnet <frame> --check`).
//!
//! Arithmetic is the oracle's, not the console's: a 16-bit float operation
//! is the f32 operation of the two halves rounded to nearest-even f16 (the
//! translator's lowering, emit/lower/valu16.rs); the console rounds once in
//! f16. The two differ only where the f32 result is itself a rounding (a
//! sum whose f32 rounding lands on an f16 tie), at most one f16 step.
//!
//! The op list of a network instance comes from one recorded frame
//! ([`programs`]): op kind, group count, the constant block's words and the
//! buffers bound. It is the same frame to frame for a rig (pk8a f1027-1032,
//! `--same`), so it is read once; weights and the formula's byte code are
//! read once and kept ([`Net::new`]); a frame gives only the input vector.

use std::collections::HashMap;

use xnw_core::pair::fold;

use crate::pack::Pack;

/// The network's op kinds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Zero,
    VecMatMulAligned,
    SliceCopy,
    PlaceAddAligned,
    PlaceAligned,
    TransposeVecMatMulAligned,
    ConcatAligned,
    Gather,
    VecMatMul,
    Place,
    PlaceAdd,
    Formula,
    Clamp,
}

/// Per kind: the program's name between `CS_` and its hash, the container
/// the kernel was read from, the buffer slots it binds (its constant block
/// is the slot after them; Zero has two root constants instead) and the
/// constant words it reads.
pub const KINDS: [(Kind, &str, &str, usize, usize); 13] = [
    (Kind::Zero, "ZeroOperation", "073fb412", 1, 2),
    (Kind::VecMatMulAligned, "VecMatMulAlignedOperation", "140f77ac", 3, 8),
    (Kind::SliceCopy, "SliceCopyTechnique", "c0013d2a", 2, 8),
    (Kind::PlaceAddAligned, "PlaceAddAlignedTechnique", "60c1dd7e", 2, 3),
    (Kind::PlaceAligned, "PlaceAlignedTechnique", "e423bbae", 2, 3),
    (Kind::TransposeVecMatMulAligned, "TransposeVecMatMulAlignedOperation", "168936d7", 3, 8),
    (Kind::ConcatAligned, "ConcatAlignedOperation", "d3585b1a", 3, 9),
    (Kind::Gather, "GatherOperation", "879d8283", 3, 8),
    (Kind::VecMatMul, "VecMatMulOperation", "e1db80d0", 3, 8),
    (Kind::Place, "PlaceTechnique", "e06062bf", 2, 3),
    (Kind::PlaceAdd, "PlaceAddTechnique", "d4d5e450", 2, 3),
    (Kind::Formula, "FormulaOperation", "307c3ab7", 7, 8),
    (Kind::Clamp, "ClampOperation", "41fce774", 2, 8),
];

fn row(k: Kind) -> &'static (Kind, &'static str, &'static str, usize, usize) {
    KINDS.iter().find(|r| r.0 == k).unwrap()
}

/// The kind of a `compute_graph_ops:CS_<Name>_<hash>_Wrapped` program.
pub fn kind_of(name: &str) -> Option<Kind> {
    let n = name.split(':').nth(1)?.strip_prefix("CS_")?;
    let n = n.strip_suffix("_Wrapped").unwrap_or(n);
    let n = n.rsplit_once('_').map_or(n, |x| x.0);
    KINDS.iter().find(|r| r.1 == n).map(|r| r.0)
}

/// One op: its kind, the dispatch's group count, the constant words it
/// reads (the rest 0) and its buffer per slot (an index into
/// [`Program::bufs`]). `sink` is where the recorded frame had it (not part
/// of the op: [`Program::same`] ignores it).
#[derive(Clone, Debug)]
pub struct Op {
    pub kind: Kind,
    pub sink: u32,
    pub groups: u32,
    pub w: [u32; 9],
    pub buf: [u8; 7],
}

/// A buffer the network binds: title address (modulo the GPU alias) and the view's bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Buf {
    pub addr: u64,
    pub bytes: u64,
}

/// A network instance's op list (one head, one frame's dispatches from its
/// Formula to the op before the next Formula).
#[derive(Clone, Debug, Default)]
pub struct Program {
    pub ops: Vec<Op>,
    pub bufs: Vec<Buf>,
}

impl Program {
    /// Same ops on the same buffers (sinks aside).
    pub fn same(&self, o: &Program) -> bool {
        self.bufs == o.bufs && self.ops.len() == o.ops.len() && self.ops.iter().zip(&o.ops).all(|(a, b)| a.kind == b.kind && a.groups == b.groups && a.w == b.w && a.buf == b.buf)
    }

    /// The ops as text, one a line (`xnw-lab headnet --ops`).
    pub fn lines(&self) -> Vec<String> {
        let mut v: Vec<String> = self.bufs.iter().enumerate().map(|(i, b)| format!("buffer {i}: {:#x} {} B{}", b.addr, b.bytes, if self.written(i) { " (written)" } else { "" })).collect();
        for o in &self.ops {
            let r = row(o.kind);
            let w: Vec<String> = o.w[..r.4].iter().map(|x| format!("{x:x}")).collect();
            v.push(format!("sink {} {:?} g{} bufs {:?} w {}", o.sink, o.kind, o.groups, &o.buf[..r.3], w.join(" ")));
        }
        v
    }

    /// Whether any op writes buffer `i`.
    pub fn written(&self, i: usize) -> bool {
        self.ops.iter().any(|o| writes(o.kind).iter().any(|s| usize::from(o.buf[*s]) == i))
    }

    /// The buffers the formulas take the frame's inputs from: (a sink that binds it, the buffer), one per address.
    pub fn inputs(&self) -> Vec<(u32, Buf)> {
        let mut v: Vec<(u32, Buf)> = Vec::new();
        for o in self.ops.iter().filter(|o| o.kind == Kind::Formula) {
            let b = self.bufs[usize::from(o.buf[4])];
            if !v.iter().any(|x| x.1.addr == b.addr) {
                v.push((o.sink, b));
            }
        }
        v
    }

    /// The network's results: per Gather into another buffer than it reads (an instance's last op), that
    /// buffer's address and the halves it writes there.
    pub fn outputs(&self) -> Vec<(u64, std::ops::Range<usize>)> {
        self.ops.iter().filter(|o| o.kind == Kind::Gather && o.buf[2] != o.buf[0]).map(|o| (self.bufs[usize::from(o.buf[2])].addr, o.w[0] as usize * 8..o.w[0] as usize * 8 + (o.w[5] as usize).min(o.groups as usize * 64))).collect()
    }

    /// How many instances (heads) the list holds: its formulas.
    pub fn heads(&self) -> usize {
        self.ops.iter().filter(|o| o.kind == Kind::Formula).count()
    }
}

/// The slots a kind writes.
fn writes(k: Kind) -> &'static [usize] {
    match k {
        Kind::Zero | Kind::PlaceAligned | Kind::PlaceAddAligned | Kind::Clamp | Kind::VecMatMul => &[0],
        Kind::SliceCopy | Kind::Place | Kind::PlaceAdd | Kind::VecMatMulAligned | Kind::ConcatAligned => &[1],
        Kind::TransposeVecMatMulAligned | Kind::Gather => &[2],
        Kind::Formula => &[0, 3, 5],
    }
}

/// The frame's network as one op list, in the order the dispatches run.
/// The title batches the frame's instances (heads): their formulas first,
/// then every instance's ops interleaved, each instance at its own arena
/// offsets (pk8a f1031: two heads, sinks 107 and 112 the formulas, Clamp
/// 113 to 0x1e3 and 114 to 0xdad). So the list belongs to the frame's set of
/// heads, not to one rig. `Err` names a network program with no kernel here.
pub fn program(p: &Pack, names: &HashMap<u128, String>) -> Result<Program, String> {
    let list = crate::head_graph::order(&crate::head_graph::dispatches(p, names));
    let mut prog = Program::default();
    for (sink, _, groups, _, name) in list.iter().filter(|d| d.4.starts_with("compute_graph_ops:")) {
        let k = kind_of(name).ok_or_else(|| format!("head network: sink {sink}: no CPU kernel for {name}"))?;
        let r = row(k);
        let views = p.tables(*sink).into_iter().find(|t| t.0 == 0).map(|t| t.2).ok_or_else(|| format!("head network: sink {sink}: no table"))?;
        let mut op = Op { kind: k, sink: *sink, groups: *groups, w: [0; 9], buf: [0; 7] };
        for s in 0..r.3 {
            let v = views.iter().find(|v| v.slot == s as u32).ok_or_else(|| format!("head network: sink {sink}: no view at slot {s}"))?;
            let b = Buf { addr: fold(v.addr), bytes: v.bytes };
            let i = match prog.bufs.iter().position(|x| x.addr == b.addr) {
                Some(i) => {
                    prog.bufs[i].bytes = prog.bufs[i].bytes.max(b.bytes);
                    i
                }
                None => {
                    prog.bufs.push(b);
                    prog.bufs.len() - 1
                }
            };
            op.buf[s] = i as u8;
        }
        let words = if k == Kind::Zero {
            p.data.refs_of(*sink).find(|x| x.kind == xnw_formats::framepack::kind::ROOTS && x.slot == 1).and_then(|x| p.bytes_of(x)).ok_or_else(|| format!("head network: sink {sink}: no root constants"))?
        } else {
            let at = xnw_formats::framepack::table_slot(0, r.3 as u32);
            p.data.refs_of(*sink).find(|x| x.kind == xnw_formats::framepack::kind::CBV && x.slot == at).and_then(|x| p.bytes_of(x)).ok_or_else(|| format!("head network: sink {sink}: no constant block at slot {}", r.3))?
        };
        for (i, c) in words.chunks_exact(4).take(r.4).enumerate() {
            op.w[i] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        prog.ops.push(op);
    }
    Ok(prog)
}

/// An f16's value (exact).
pub fn h2f(h: u16) -> f32 {
    let s = (u32::from(h) & 0x8000) << 16;
    let (e, m) = (u32::from(h >> 10) & 0x1f, u32::from(h & 0x3ff));
    let b = match e {
        0 if m == 0 => 0,
        0 => {
            let k = m.leading_zeros() - 21;
            ((113 - k) << 23) | (((m << k) & 0x3ff) << 13)
        }
        31 => 0x7f80_0000 | (m << 13),
        _ => ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(s | b)
}

/// The f16 nearest `x`, ties to even; 65520 and above is infinity.
pub fn f2h(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a >= 0x7f80_0000 {
        return s | if a > 0x7f80_0000 { 0x7e00 } else { 0x7c00 };
    }
    if a >= 0x477f_f000 {
        return s | 0x7c00;
    }
    if a < 0x3300_0000 {
        return s;
    }
    let e = (a >> 23) as i32 - 127;
    let m = (a & 0x7f_ffff) | 0x80_0000;
    let drop = if e >= -14 { 13 } else { (13 + (-14 - e)) as u32 };
    let (half, rem, mut q) = (1u32 << (drop - 1), m & ((1u32 << drop) - 1), m >> drop);
    if rem > half || (rem == half && q & 1 == 1) {
        q += 1;
    }
    if e >= -14 { s | ((((e + 15) as u32) << 10) + (q - 0x400)) as u16 } else { s | q as u16 }
}

/// The f16 at or next below `x` in magnitude (toward zero); above the largest half, the largest.
pub fn f2h_zero(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a >= 0x7f80_0000 {
        return s | if a > 0x7f80_0000 { 0x7e00 } else { 0x7c00 };
    }
    if a >= 0x477f_e000 {
        return s | 0x7bff;
    }
    if a < 0x3380_0000 {
        return s;
    }
    let e = (a >> 23) as i32 - 127;
    let m = (a & 0x7f_ffff) | 0x80_0000;
    if e >= -14 { s | ((((e + 15) as u32) << 10) + ((m >> 13) - 0x400)) as u16 } else { s | (m >> (13 + (-14 - e))) as u16 }
}

/// Whose arithmetic the kernels follow.
/// `Console` (our default): every 16-bit float operation rounds to
/// nearest-even, and a wave's lanes fold pair by pair, each sum rounded, as
/// the console's f16 instructions do.
/// `Oracle`: what the dispatch replay gives on WARP 1.0.22 [V pk8a f1027]:
/// its f32 to f16 conversion truncates (sink 510: 0x124f + 0xa7a2 = 0xa76f
/// there, 0xa770 to nearest), and the translator takes a wave's f16 sum in
/// f32 and converts once (emit/lower/setup.rs `Shape::Half`: 168936d7's lanes).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Round {
    Console,
    Oracle,
}

/// `x` as an f16 under `r`.
fn to_h(x: f32, r: Round) -> u16 {
    match r {
        Round::Console => f2h(x),
        Round::Oracle => f2h_zero(x),
    }
}

/// `x` rounded to an f16 (as an f32).
fn rnd(x: f32, r: Round) -> f32 {
    h2f(to_h(x, r))
}

/// Whether the AVX2 + F16C rows are used (the scalar rows give the same bits; `XNW_HEADNET_SIMD=off`).
fn simd() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::env::var("XNW_HEADNET_SIMD").map_or(true, |v| v != "off") && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

#[cfg(target_arch = "x86_64")]
mod rows {
    use std::arch::x86_64::*;

    /// d[j] = f16(x * w[j] + d[j]) when `acc`, f16(x * w[j]) otherwise.
    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn mul<const R: i32>(d: &mut [f32], x: f32, w: &[f32], acc: bool) -> usize {
        let n = d.len().min(w.len()) & !7;
        let xv = _mm256_set1_ps(x);
        let mut j = 0;
        while j < n {
            // SAFETY: j + 8 <= n <= both lengths.
            unsafe {
                let mut p = _mm256_mul_ps(xv, _mm256_loadu_ps(w.as_ptr().add(j)));
                if acc {
                    p = _mm256_add_ps(p, _mm256_loadu_ps(d.as_ptr().add(j)));
                }
                let h = _mm256_cvtps_ph::<R>(p);
                _mm256_storeu_ps(d.as_mut_ptr().add(j), _mm256_cvtph_ps(h));
            }
            j += 8;
        }
        n
    }

    /// d[j] = f16(d[j] + b[j]).
    #[target_feature(enable = "avx2,f16c")]
    pub unsafe fn add<const R: i32>(d: &mut [f32], b: &[f32]) -> usize {
        let n = d.len().min(b.len()) & !7;
        let mut j = 0;
        while j < n {
            // SAFETY: j + 8 <= n <= both lengths.
            unsafe {
                let p = _mm256_add_ps(_mm256_loadu_ps(d.as_ptr().add(j)), _mm256_loadu_ps(b.as_ptr().add(j)));
                let h = _mm256_cvtps_ph::<R>(p);
                _mm256_storeu_ps(d.as_mut_ptr().add(j), _mm256_cvtph_ps(h));
            }
            j += 8;
        }
        n
    }
}

/// d[j] = f16(x * w[j]) (`acc`: f16(x * w[j] + d[j]); the product of two halves is exact in f32).
fn mul_row(d: &mut [f32], x: f32, w: &[f32], acc: bool, r: Round) {
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the features were detected. 0: to nearest-even; 3: toward zero.
        done = unsafe { if r == Round::Oracle { rows::mul::<3>(d, x, w, acc) } else { rows::mul::<0>(d, x, w, acc) } };
    }
    for (d, w) in d[done..].iter_mut().zip(&w[done..]) {
        *d = rnd(if acc { x * *w + *d } else { x * *w }, r);
    }
}

/// d[j] = f16(d[j] + b[j]).
fn add_row(d: &mut [f32], b: &[f32], r: Round) {
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the features were detected.
        done = unsafe { if r == Round::Oracle { rows::add::<3>(d, b) } else { rows::add::<0>(d, b) } };
    }
    for (d, b) in d[done..].iter_mut().zip(&b[done..]) {
        *d = rnd(*d + *b, r);
    }
}

/// The product of a vector and a prepared weight block, as both matrix
/// kernels make it (140f77ac, e1db80d0: a thread per 4 inputs and 4 outputs,
/// partial sums in group memory folded y with y + 32, 16, .. 1; 168936d7: a
/// lane per 4 inputs, a wave per output, lanes folded in neighbouring pairs).
/// `m` holds `x.len()` rows of `nout` weights (row i = input i); `x.len()`
/// is a multiple of 4. Per 4 inputs: f16(f16(x0 w0 + f16(x1 w1)) + f16(x2 w2
/// + f16(x3 w3))); the partial sums into `s[..nout]`.
fn product(m: &[f32], x: &[f32], nout: usize, far: bool, rd: Round, s: &mut Vec<f32>, t: &mut Vec<f32>) {
    let blocks = x.len() / 4;
    s.clear();
    s.resize(blocks.max(1) * nout, 0.0);
    t.clear();
    t.resize(nout, 0.0);
    for b in 0..blocks {
        let d = &mut s[b * nout..(b + 1) * nout];
        let r = |c: usize| &m[(4 * b + c) * nout..(4 * b + c + 1) * nout];
        mul_row(d, x[4 * b + 1], r(1), false, rd);
        mul_row(d, x[4 * b], r(0), true, rd);
        mul_row(t, x[4 * b + 3], r(3), false, rd);
        mul_row(t, x[4 * b + 2], r(2), true, rd);
        add_row(d, t, rd);
    }
    let mut n = blocks;
    if far {
        let mut h = 32;
        while h >= 1 {
            if n > h {
                let (a, b) = s.split_at_mut(h * nout);
                add_row(&mut a[..(n - h) * nout], &b[..(n - h) * nout], rd);
                n = h;
            }
            h /= 2;
        }
    } else if rd == Round::Oracle {
        // The translated program: the lanes' halves summed in f32, one conversion.
        let (a, b) = s.split_at_mut(nout);
        for l in b.chunks_exact(nout).take(n.saturating_sub(1)) {
            for (a, l) in a.iter_mut().zip(l) {
                *a += *l;
            }
        }
        for a in a.iter_mut() {
            *a = rnd(*a, rd);
        }
    } else {
        while n > 1 {
            for k in 0..n / 2 {
                let (a, b) = s.split_at_mut((2 * k + 1) * nout);
                add_row(&mut a[2 * k * nout..], &b[..nout], rd);
                if k > 0 {
                    s.copy_within(2 * k * nout..(2 * k + 1) * nout, k * nout);
                }
            }
            if n % 2 == 1 {
                s.copy_within((n - 1) * nout..n * nout, n / 2 * nout);
            }
            n = n.div_ceil(2);
        }
    }
}

/// One formula instruction (307c3ab7's byte code): 5 header bytes (op, u16
/// destination, u16 lanes), then per lane its operands as u16.
#[derive(Clone, Debug)]
struct Instr {
    op: u8,
    dst: u32,
    lanes: u32,
    args: Vec<u16>,
}

/// Operands per lane of a formula op (op 2, the output copy, has a
/// (destination, source) pair).
fn arity(op: u8) -> usize {
    match op {
        2..=6 | 9 | 10 => 2,
        7 | 8 | 11..=14 => 1,
        _ => 0,
    }
}

/// The formula's byte code by stage: every program's (one per 32 threads)
/// instructions of that stage, in program order. A barrier separates the
/// stages on the GPU; the programs of a stage run side by side.
fn formula(f: &[u16], o: &Op) -> Result<(u32, u32, Vec<Vec<Instr>>), String> {
    let base = o.w[4] as usize * 8;
    let h = |i: usize| f.get(base + i).copied().unwrap_or(0);
    let byte = |k: usize| u32::from(h(k / 2) >> (8 * (k & 1))) & 0xff;
    let word = |k: usize| byte(k) | (byte(k + 1) << 8);
    let (n_in, n_const, lanes, n_stage) = (u32::from(h(0)), u32::from(h(1)), u32::from(h(2)), usize::from(h(3)));
    if lanes == 0 {
        return Err("head network: formula with 0 lanes".into());
    }
    let mut stages: Vec<Vec<Instr>> = vec![Vec::new(); n_stage];
    for q in 0..(o.groups * 1024 / lanes) as usize {
        let off = usize::from(h(8 + q));
        let own = usize::from(h(off));
        let mut at = off + 1;
        for stage in stages.iter_mut().take(own) {
            let count = h(at);
            at += 1;
            for _ in 0..count {
                at = (at + 7) & !7;
                let (op, dst, n) = (byte(2 * at) as u8, word(2 * at + 1), word(2 * at + 3));
                let a = arity(op);
                if op > 14 || at >= f.len() {
                    return Err(format!("head network: formula op {op} at half {at}"));
                }
                let args = (0..a * n.min(lanes) as usize).map(|k| word(2 * at + 5 + 2 * k) as u16).collect();
                stage.push(Instr { op, dst, lanes: n.min(lanes), args });
                at += 3 + a * n as usize;
            }
        }
    }
    Ok((n_in, n_const, stages))
}

/// A formula op on one lane's operands (`a` the first, `b` the second), as
/// the translated program computes it: in f32, rounded to f16.
fn formula_op(op: u8, a: u16, b: u16, r: Round) -> u16 {
    let (x, y) = (h2f(a), h2f(b));
    // 11-13: the angle as a fraction of a turn, less its whole turns (f16 1/2pi = 0x3118).
    let turn = || {
        let k = h2f(0x3118);
        let n = rnd(rnd(k * x, r).trunc(), r);
        rnd(x * k - n, r) * std::f32::consts::TAU
    };
    match op {
        3 => to_h(y + x, r),
        4 => to_h(x - y, r),
        5 => to_h(y * x, r),
        6 => to_h(x * rnd(1.0 / y, r), r),
        7 => a & 0x7fff,
        8 => a ^ 0x8000,
        9 => min_h(a, b),
        10 => max_h(a, b),
        11 => to_h(turn().sin(), r),
        12 => to_h(turn().cos(), r),
        13 => {
            let t = turn();
            to_h(rnd(t.sin(), r) * rnd(1.0 / rnd(t.cos(), r), r), r)
        }
        14 => to_h(x.sqrt(), r),
        _ => a,
    }
}

/// The larger of two halves; of two zeros +0 unless both are -0 (the oracle: pk8a f1027 sink 89, max(-0, +0),
/// and f1031 sink 107, max(+0, -0), both +0).
fn max_h(a: u16, b: u16) -> u16 {
    let (x, y) = (h2f(a), h2f(b));
    if x > y {
        a
    } else if y > x || x.is_nan() {
        b
    } else if y.is_nan() {
        a
    } else {
        a & b
    }
}

/// The smaller of two halves; of two zeros -0 if either is (by symmetry with [`max_h`], I: no capture has the case).
fn min_h(a: u16, b: u16) -> u16 {
    let (x, y) = (h2f(a), h2f(b));
    if x < y {
        a
    } else if y < x || x.is_nan() {
        b
    } else if y.is_nan() {
        a
    } else {
        a | b
    }
}

/// The matrix kernels' weights ready (f32, a row per input), kept across
/// nets: by weights buffer address, kind and the op's weight words. A new
/// set of heads on screen makes a new op list but finds its matrices here.
#[derive(Default)]
pub struct Cache {
    m: HashMap<(u64, u8, [u32; 5]), std::sync::Arc<Vec<f32>>>,
    /// Matrices made (not found).
    pub made: u32,
}

/// What an op keeps ready: a matrix kernel's weights as f32 rows per input, a formula's instructions.
#[derive(Clone, Debug, Default)]
struct Ready {
    m: std::sync::Arc<Vec<f32>>,
    nin: usize,
    nout: usize,
    stages: Vec<Vec<Instr>>,
    n_in: u32,
    n_const: u32,
}

/// A network instance ready to run: its op list, its buffers as halves,
/// the weights and the formula read once.
pub struct Net {
    /// How results round ([`Round::Console`] unless the oracle is being matched).
    pub round: Round,
    pub prog: Program,
    /// Per buffer of `prog.bufs`: its halves (as far as the ops reach).
    pub mem: Vec<Vec<u16>>,
    ready: Vec<Ready>,
    s: Vec<f32>,
    t: Vec<f32>,
    x: Vec<f32>,
    tmp: Vec<u16>,
}

fn ceil4(x: usize) -> usize {
    (x + 3) & !3
}

/// The halves an op reaches per slot (one past the last), as far as its words say.
fn reach(o: &Op) -> Vec<(usize, usize)> {
    let w = |i: usize| o.w[i] as usize;
    match o.kind {
        Kind::Zero => vec![(0, w(0) * 8 + 4 * w(1))],
        Kind::PlaceAligned | Kind::PlaceAddAligned | Kind::Place | Kind::PlaceAdd => vec![(0, w(0).max(w(1)) + ceil4(w(2)) + 4), (1, w(0).max(w(1)) + ceil4(w(2)) + 4)],
        Kind::SliceCopy => vec![(0, w(1) * 8 + w(6) * w(5) + w(7) + w(3)), (1, w(0) * 8 + w(3))],
        Kind::Clamp => vec![(0, w(0) * 8 + w(1) * 8), (1, w(2) * 8 + w(1) * 8)],
        Kind::VecMatMulAligned => vec![(1, w(0) * 8 + ceil4(w(6))), (2, w(1) * 8 + ceil4(w(5)))],
        Kind::VecMatMul => vec![(0, w(0) * 8 + ceil4(w(6))), (1, w(1) * 8 + ceil4(w(5)))],
        Kind::TransposeVecMatMulAligned => vec![(1, w(1) * 8 + ceil4(w(5))), (2, w(0) * 8 + w(6))],
        Kind::ConcatAligned => {
            let (a, b) = (ceil4(w(4) * w(3).max(1)), ceil4(w(6) * w(5).max(1)));
            vec![(2, w(1) * 8 + a), (0, w(2) * 8 + b), (1, w(0) * 8 + a + b + 4)]
        }
        Kind::Gather => vec![(1, w(2) * 8 + 2 * w(5)), (2, w(0) * 8 + w(5)), (0, w(1) * 8 + 0x10000)],
        Kind::Formula => vec![(0, w(5) * 8 + 0x10100), (3, w(5) * 8 + 0x10100), (5, w(0) * 8 + 0x10000), (4, w(2) * 8 + 0x10000)],
    }
}

impl Net {
    /// `read(i, buffer)`: the bytes of a buffer no op writes (the weights,
    /// the formula's code, the input slots), read once here; the matrices
    /// found in or added to `cache`.
    pub fn new(prog: Program, read: &dyn Fn(usize, &Buf) -> Option<Vec<u8>>, cache: &mut Cache) -> Result<Net, String> {
        let written: Vec<bool> = (0..prog.bufs.len()).map(|i| prog.written(i)).collect();
        let mut mem: Vec<Vec<u16>> = Vec::new();
        for (i, b) in prog.bufs.iter().enumerate() {
            if written[i] {
                mem.push(Vec::new());
            } else {
                let bytes = read(i, b).ok_or_else(|| format!("head network: no bytes for the buffer at {:#x} ({} B)", b.addr, b.bytes))?;
                mem.push(as_halves(&bytes));
            }
        }
        for o in &prog.ops {
            for (slot, end) in reach(o) {
                let i = usize::from(o.buf[slot]);
                let cap = (prog.bufs[i].bytes / 2) as usize;
                if written[i] && mem[i].len() < end.min(cap) {
                    mem[i].resize(end.min(cap), 0);
                }
            }
        }
        let mut ready = Vec::with_capacity(prog.ops.len());
        for o in &prog.ops {
            let w = |i: usize| o.w[i] as usize;
            let mut r = Ready::default();
            match o.kind {
                Kind::VecMatMulAligned | Kind::VecMatMul | Kind::TransposeVecMatMulAligned => {
                    let wi = usize::from(o.buf[if o.kind == Kind::VecMatMul { 2 } else { 0 }]);
                    if written[wi] {
                        return Err(format!("head network: sink {}: weights in a buffer the network writes", o.sink));
                    }
                    let (k, rows, i0) = (w(5), w(6), w(7) * 256);
                    let i1 = k.min(i0 + 256);
                    r.nin = ceil4(i1.saturating_sub(i0));
                    // A group of either program covers 16 outputs.
                    r.nout = if o.kind == Kind::TransposeVecMatMulAligned { rows } else { ceil4(rows) }.min(o.groups as usize * 16);
                    let key = (prog.bufs[wi].addr, KINDS.iter().position(|x| x.0 == o.kind).unwrap() as u8, [o.w[2], o.w[5], o.w[6], o.w[7], r.nout as u32]);
                    r.m = match cache.m.get(&key) {
                        Some(m) => m.clone(),
                        None => {
                            let f = &mem[wi];
                            let at = |i: usize| f.get(i).map_or(0.0, |h| h2f(*h));
                            let mut m = vec![0.0; r.nin * r.nout];
                            for i in i0..i1 {
                                for j in 0..r.nout {
                                    m[(i - i0) * r.nout + j] = if o.kind == Kind::TransposeVecMatMulAligned { at(w(2) * 8 + j * k + i) } else { at(w(2) * 8 + i * rows + j) };
                                }
                            }
                            cache.made += 1;
                            cache.m.entry(key).or_insert(std::sync::Arc::new(m)).clone()
                        }
                    };
                }
                Kind::Formula => {
                    let fi = usize::from(o.buf[6]);
                    if written[fi] || o.buf[1] != o.buf[6] || o.buf[2] != o.buf[6] {
                        return Err(format!("head network: sink {}: the formula's code is not one read-only buffer", o.sink));
                    }
                    (r.n_in, r.n_const, r.stages) = formula(&mem[fi], o)?;
                }
                Kind::Gather => {
                    if w(6) == 0 || w(5) <= 4 {
                        return Err(format!("head network: sink {}: a Gather form with no kernel (words {:x?})", o.sink, o.w));
                    }
                }
                _ => {}
            }
            ready.push(r);
        }
        Ok(Net { round: Round::Console, prog, mem, ready, s: Vec::new(), t: Vec::new(), x: Vec::new(), tmp: Vec::new() })
    }

    /// The frame's inputs: the bytes of the input buffer at `addr`, from its start (each formula reads its own range of it).
    pub fn set_input(&mut self, addr: u64, bytes: &[u8]) {
        if let Some(i) = self.prog.bufs.iter().position(|b| b.addr == fold(addr)).filter(|i| !self.prog.written(*i)) {
            let m = &mut self.mem[i];
            m.clear();
            m.extend(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
        }
    }

    /// The halves of the buffer at title address `addr`.
    pub fn buffer(&self, addr: u64) -> Option<&[u16]> {
        self.prog.bufs.iter().position(|b| b.addr == fold(addr)).map(|i| self.mem[i].as_slice())
    }

    /// The results as whole buffers to hold on the device (title address,
    /// bytes): every instance's output range in its output buffer, the rest
    /// zero as a capture holds it.
    pub fn outputs(&self) -> Vec<(u64, Vec<u8>)> {
        let mut out: Vec<(u64, Vec<u8>)> = Vec::new();
        for (addr, range) in self.prog.outputs() {
            let Some((buf, m)) = self.prog.bufs.iter().zip(&self.mem).find(|(b, _)| b.addr == addr) else { continue };
            let k = match out.iter().position(|x| x.0 == addr) {
                Some(k) => k,
                None => {
                    out.push((addr, vec![0; buf.bytes as usize]));
                    out.len() - 1
                }
            };
            for (h, v) in range.clone().zip(&m[range.start.min(m.len())..range.end.min(m.len())]) {
                if let Some(d) = out[k].1.get_mut(2 * h..2 * h + 2) {
                    d.copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    /// A hash of the results' halves.
    pub fn output_hash(&self) -> u32 {
        let mut b: Vec<u8> = Vec::new();
        for (addr, r) in self.prog.outputs() {
            if let Some(m) = self.buffer(addr) {
                b.extend(m[r.start.min(m.len())..r.end.min(m.len())].iter().flat_map(|h| h.to_le_bytes()));
            }
        }
        xnw_core::hash128(&[&b]) as u32
    }

    /// Every op in order.
    pub fn run(&mut self) {
        for i in 0..self.prog.ops.len() {
            self.step(i);
        }
    }

    /// Op `i` alone.
    pub fn step(&mut self, i: usize) {
        let o = self.prog.ops[i].clone();
        let w = |k: usize| o.w[k] as usize;
        let b = |s: usize| usize::from(o.buf[s]);
        let threads = o.groups as usize * 64;
        let rd = self.round;
        let mut tmp = std::mem::take(&mut self.tmp);
        tmp.clear();
        let get = |m: &Vec<Vec<u16>>, s: usize, i: usize| m[b(s)].get(i).copied().unwrap_or(0);
        let put = |m: &mut Vec<Vec<u16>>, s: usize, i: usize, v: u16| {
            if let Some(x) = m[b(s)].get_mut(i) {
                *x = v;
            }
        };
        match o.kind {
            // 073fb412: thread t below w1 clears the 4 halves at w0 * 8 + 4 t.
            Kind::Zero => {
                let (at, n) = (w(0) * 8, 4 * w(1).min(threads));
                let m = &mut self.mem[b(0)];
                let end = (at + n).min(m.len());
                m[at.min(end)..end].fill(0);
            }
            // e423bbae, 60c1dd7e: thread t with 4 t below w2 moves (adds) the 4 halves of element (w1 + 4 t) / 4 to element (w0 + 4 t) / 4.
            Kind::PlaceAligned | Kind::PlaceAddAligned => {
                let n = w(2).min(4 * threads).div_ceil(4);
                for t in 0..n {
                    for c in 0..4 {
                        tmp.push(get(&self.mem, 1, ((w(1) + 4 * t) & !3) + c));
                    }
                }
                for t in 0..n {
                    for c in 0..4 {
                        let at = ((w(0) + 4 * t) & !3) + c;
                        let v = tmp[4 * t + c];
                        let v = if o.kind == Kind::PlaceAddAligned { to_h(h2f(get(&self.mem, 0, at)) + h2f(v), rd) } else { v };
                        put(&mut self.mem, 0, at, v);
                    }
                }
            }
            // e06062bf, d4d5e450: the same half by half, w2 halves exactly.
            Kind::Place | Kind::PlaceAdd => {
                let n = w(2).min(4 * threads);
                tmp.extend((0..n).map(|k| get(&self.mem, 0, w(1) + k)));
                for (k, v) in tmp.iter().enumerate() {
                    let at = w(0) + k;
                    let v = if o.kind == Kind::PlaceAdd { to_h(h2f(get(&self.mem, 1, at)) + h2f(*v), rd) } else { *v };
                    put(&mut self.mem, 1, at, v);
                }
            }
            // c0013d2a: w3 halves from w1 * 8 + w6 * w5 + w7 (row w6 of rows w5 wide, column w7) to w0 * 8.
            Kind::SliceCopy => {
                let n = w(3).min(4 * threads);
                let from = w(1) * 8 + w(6) * w(5) + w(7);
                tmp.extend((0..n).map(|k| get(&self.mem, 0, from + k)));
                for (k, v) in tmp.iter().enumerate() {
                    put(&mut self.mem, 1, w(0) * 8 + k, *v);
                }
            }
            // 41fce774: 2 w1 elements of 4 halves from w2 * 8 (slot 1) to w0 * 8, each max(x, w6) then min(., w7).
            Kind::Clamp => {
                let n = 4 * (2 * w(1)).min(threads);
                let (lo, hi) = (o.w[6] as u16, o.w[7] as u16);
                tmp.extend((0..n).map(|k| get(&self.mem, 1, w(2) * 8 + k)));
                for (k, v) in tmp.iter().enumerate() {
                    put(&mut self.mem, 0, w(0) * 8 + k, min_h(max_h(*v, lo), hi));
                }
            }
            // d3585b1a: w4 halves from w1 * 8 (slot 2), then w6 halves from w2 * 8 (slot 0), to w0 * 8 (slot 1);
            // the first part has 32 w7 threads of 4 halves, the second the rest of the groups'.
            Kind::ConcatAligned => {
                let (rows_a, rows_b) = if o.w[8] != 0 { (1, 1) } else { (w(3), w(5)) };
                let (a, bn) = (w(4) * rows_a, w(6) * rows_b);
                let ta = (32 * w(7)).min(threads);
                let (na, nb) = (a.div_ceil(4).min(ta), bn.div_ceil(4).min(threads - ta));
                tmp.extend((0..4 * na).map(|k| get(&self.mem, 2, w(1) * 8 + k)));
                tmp.extend((0..4 * nb).map(|k| get(&self.mem, 0, w(2) * 8 + k)));
                for k in 0..4 * na {
                    put(&mut self.mem, 1, ((w(0) * 8 + k) & !3) + (k & 3), tmp[k]);
                }
                for k in 0..4 * nb {
                    put(&mut self.mem, 1, ((w(0) * 8 + a + (k & !3)) & !3) + (k & 3), tmp[4 * na + k]);
                }
            }
            // 879d8283 (the form with w6 != 0 and w5 > 4): thread t below w5 moves the half at w1 * 8 + index t (slot 0)
            // to w0 * 8 + t (slot 2); index t is the u32 at half w2 * 8 + 2 t of slot 1.
            Kind::Gather => {
                let n = w(5).min(threads);
                for t in 0..n {
                    let ix = usize::from(get(&self.mem, 1, w(2) * 8 + 2 * t)) | (usize::from(get(&self.mem, 1, w(2) * 8 + 2 * t + 1)) << 16);
                    tmp.push(get(&self.mem, 0, w(1) * 8 + ix));
                }
                for (t, v) in tmp.iter().enumerate() {
                    put(&mut self.mem, 2, w(0) * 8 + t, *v);
                }
            }
            Kind::VecMatMulAligned | Kind::VecMatMul | Kind::TransposeVecMatMulAligned => {
                let r = &self.ready[i];
                let (xs, os) = match o.kind {
                    Kind::VecMatMulAligned => (2, 1),
                    Kind::VecMatMul => (1, 0),
                    _ => (1, 2),
                };
                let (k, i0) = (w(5), w(7) * 256);
                let mut x = std::mem::take(&mut self.x);
                x.clear();
                // w4 == 0: no input (the product is 0).
                x.extend((0..r.nin).map(|c| if o.w[4] != 0 && i0 + c < k { h2f(get(&self.mem, xs, w(1) * 8 + i0 + c)) } else { 0.0 }));
                product(&r.m, &x, r.nout, o.kind != Kind::TransposeVecMatMulAligned, rd, &mut self.s, &mut self.t);
                // A later chunk of the inputs (w7 != 0) adds to what the earlier ones left.
                for j in 0..r.nout {
                    let at = w(0) * 8 + j;
                    let before = if o.w[7] != 0 { h2f(get(&self.mem, os, at)) } else { 0.0 };
                    let v = to_h(self.s[j] + before, rd);
                    put(&mut self.mem, os, at, v);
                }
                self.x = x;
            }
            // 307c3ab7: the inputs (slot 4 from w2 * 8) and the constants (slot 2 from w3 * 8) to the arena at w5 * 8,
            // 4 halves a thread; then the byte code, stage by stage.
            Kind::Formula => {
                let r = &self.ready[i];
                let (ein, ec) = ((r.n_in as usize + 3) / 4, (r.n_const as usize + 3) / 4);
                for e in 0..(ein + ec).min(o.groups as usize * 1024) {
                    for c in 0..4 {
                        let v = if e < ein { get(&self.mem, 4, w(2) * 8 + 4 * e + c) } else { get(&self.mem, 2, w(3) * 8 + 4 * (e - ein) + c) };
                        put(&mut self.mem, 0, w(5) * 8 + 4 * e + c, v);
                    }
                }
                let base = w(5) * 8;
                for stage in &r.stages {
                    for ins in stage {
                        let a = arity(ins.op);
                        tmp.clear();
                        tmp.extend(ins.args.iter().enumerate().map(|(k, x)| if ins.op == 2 && k % 2 == 0 { *x } else { get(&self.mem, 3, base + usize::from(*x)) }));
                        for l in 0..ins.lanes as usize {
                            match ins.op {
                                2 => put(&mut self.mem, 5, w(0) * 8 + usize::from(tmp[2 * l]), tmp[2 * l + 1]),
                                3..=14 => {
                                    let v = formula_op(ins.op, tmp[a * l], if a > 1 { tmp[a * l + 1] } else { 0 }, rd);
                                    put(&mut self.mem, 3, base + ins.dst as usize + 4 * l, v);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        self.tmp = tmp;
    }

    /// How many formula instructions of each op the instance has (`--ops`).
    pub fn formula_census(&self) -> [u32; 15] {
        let mut n = [0; 15];
        for i in self.ready.iter().flat_map(|r| r.stages.iter().flatten()) {
            n[usize::from(i.op)] += 1;
        }
        n
    }
}

/// The frame's net over the pack's bytes: the op list from its dispatches,
/// each read-only buffer from the first sink that binds it.
pub fn net(p: &Pack, g: Program, cache: &mut Cache) -> Result<Net, String> {
    let ops = g.ops.clone();
    Net::new(
        g,
        &|i, _| {
            let (o, slot) = ops.iter().find_map(|o| (0..row(o.kind).3).find(|s| usize::from(o.buf[*s]) == i).map(|s| (o, s)))?;
            p.slot_bytes(o.sink, 0, slot as u32).ok().map(|x| x.1)
        },
        cache,
    )
}

/// The live path's nets (FACE1 `head=on`): one per op list seen (a set of
/// heads on screen), made at first sight, the matrices shared ([`Cache`]).
pub struct Kept {
    nets: Vec<Net>,
    pub cache: Cache,
    pub round: Round,
    /// Nets made so far.
    pub made: u32,
}

impl Default for Kept {
    fn default() -> Kept {
        Kept { nets: Vec::new(), cache: Cache::default(), round: Round::Console, made: 0 }
    }
}

/// One frame's network on the CPU: heads, ops, ms for the op list, the net (0 when kept), the inputs and the run.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cost {
    pub heads: usize,
    pub ops: usize,
    pub list_ms: f64,
    pub make_ms: f64,
    pub input_ms: f64,
    pub run_ms: f64,
}

impl Kept {
    /// The frame's network results (as [`Net::outputs`]): the op list from
    /// the pack's dispatches, its kept net (made when the list is new), the
    /// inputs from the pack (the buffers the formulas read), the run.
    pub fn outputs(&mut self, p: &Pack, names: &HashMap<u128, String>) -> Result<(Vec<(u64, Vec<u8>)>, Cost), String> {
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let g = program(p, names)?;
        let mut cost = Cost { heads: g.heads(), ops: g.ops.len(), list_ms: ms(t), ..Cost::default() };
        if g.ops.is_empty() {
            return Ok((Vec::new(), cost));
        }
        let t = std::time::Instant::now();
        let k = match self.nets.iter().position(|n| n.prog.same(&g)) {
            Some(k) => k,
            None => {
                if self.nets.len() >= 16 {
                    self.nets.remove(0);
                }
                let mut n = net(p, g.clone(), &mut self.cache)?;
                n.round = self.round;
                self.nets.push(n);
                self.made += 1;
                cost.make_ms = ms(t);
                self.nets.len() - 1
            }
        };
        let t = std::time::Instant::now();
        let n = &mut self.nets[k];
        for (sink, b) in g.inputs() {
            let bytes = p.slot_bytes(sink, 0, 4).map_err(|e| format!("head network: inputs: {e}"))?.1;
            n.set_input(b.addr, &bytes);
        }
        cost.input_ms = ms(t);
        let t = std::time::Instant::now();
        n.run();
        let out = n.outputs();
        cost.run_ms = ms(t);
        Ok((out, cost))
    }
}

/// A buffer's bytes as halves.
fn as_halves(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

/// How far two halves are apart, in f16 steps (the sign-magnitude order).
fn steps(a: u16, b: u16) -> u32 {
    let o = |h: u16| if h & 0x8000 != 0 { -i32::from(h & 0x7fff) } else { i32::from(h & 0x7fff) };
    (o(a) - o(b)).unsigned_abs()
}

/// The software device the oracle runs on (the title's translated programs never run on the card here).
fn oracle_host() -> Result<xnw_gpu::host::Host, String> {
    // SAFETY: set before any thread of ours reads the environment.
    unsafe { std::env::set_var("XNW_ADAPTER", "warp") };
    let host = xnw_gpu::host::Host::new(crate::replay::hw_agility().as_deref()).map_err(|e| format!("head network: device: {e}"))?;
    if !host.caps.software {
        return Err(format!("head network: the oracle runs on the software adapter, got {}", host.caps.adapter));
    }
    Ok(host)
}

/// Per kind: sinks, sinks whose written buffers equal the oracle's, halves that differ, the most f16 steps apart, the largest difference.
type Tally = std::collections::BTreeMap<String, (u32, u32, u64, u32, f32)>;

/// The frame's network on the CPU against the dispatch replay, sink by
/// sink: after each network dispatch the oracle's written buffers against
/// ours. `resync`: ours then take the oracle's (every kernel judged on the
/// oracle's inputs); otherwise ours run on their own results to the end.
/// Lines: one per differing sink (the first 40), the table per kind, the total.
pub fn check(p: &Pack, names: &HashMap<u128, String>, host: &xnw_gpu::host::Host, resync: bool, round: Round) -> Result<Vec<String>, String> {
    let mut net = net(p, program(p, names)?, &mut Cache::default())?;
    net.round = round;
    let at: HashMap<u32, usize> = net.prog.ops.iter().enumerate().map(|(i, o)| (o.sink, i)).collect();
    let mut tally = Tally::new();
    let mut lines: Vec<String> = Vec::new();
    let mut outside = 0u64;
    let mut each = |sink: u32, _wrote: &[u64], runner: &mut xnw_world::compute::Runner| -> Result<(), String> {
        let Some(&i) = at.get(&sink) else { return Ok(()) };
        // `XNW_HEADNET_DEBUG=<sink>,<sink>:<file prefix>`: every buffer before the sink's op
        // (<prefix>.<sink>.before.<buffer>), then what it wrote, ours and the oracle's (.ours.<buffer>, .oracle.<buffer>).
        let debug = std::env::var("XNW_HEADNET_DEBUG").ok().and_then(|v| v.split_once(':').map(|(a, b)| (a.to_string(), b.to_string()))).filter(|(a, _)| a.split(',').any(|x| x.parse() == Ok(sink))).map(|x| x.1);
        let save = |tag: &str, k: usize, m: &[u16]| {
            if let Some(f) = &debug {
                let _ = std::fs::write(format!("{f}.{sink}.{tag}.{k}"), m.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>());
            }
        };
        for (k, m) in net.mem.iter().enumerate() {
            save("before", k, m);
        }
        net.step(i);
        let o = net.prog.ops[i].clone();
        let (mut differ, mut far, mut big) = (0u64, 0u32, 0f32);
        let mut first = None;
        let mut slots: Vec<usize> = writes(o.kind).iter().map(|s| usize::from(o.buf[*s])).collect();
        slots.dedup();
        for k in slots {
            let theirs = as_halves(&runner.read(crate::compute::key(net.prog.bufs[k].addr))?);
            let ours = &mut net.mem[k];
            save("ours", k, ours);
            save("oracle", k, &theirs[..ours.len().min(theirs.len())]);
            for (j, (a, b)) in ours.iter().zip(&theirs).enumerate().filter(|(_, (a, b))| a != b) {
                differ += 1;
                far = far.max(steps(*a, *b));
                big = big.max((h2f(*a) - h2f(*b)).abs());
                first.get_or_insert((j, *a, *b));
            }
            if i + 1 == net.prog.ops.len() {
                outside += theirs.iter().skip(ours.len()).filter(|h| **h != 0).count() as u64;
            }
            if resync {
                let n = ours.len().min(theirs.len());
                ours[..n].copy_from_slice(&theirs[..n]);
            }
        }
        let t = tally.entry(format!("{:?}", o.kind)).or_default();
        t.0 += 1;
        t.1 += u32::from(differ == 0);
        t.2 += differ;
        t.3 = t.3.max(far);
        t.4 = t.4.max(big);
        if let Some((j, a, b)) = first.filter(|_| lines.len() < 40) {
            lines.push(format!("sink {sink} {:?} w {:x?}: {differ} halves differ, at most {far} f16 steps, {big:e}; first at half {j:#x}: ours {a:04x} oracle {b:04x}", o.kind, &o.w[..row(o.kind).4]));
        }
        Ok(())
    };
    let mut runner = xnw_world::compute::Runner::new(host)?;
    let mut ex = crate::head_graph::Extra { held: Vec::new(), each: Some(&mut each) };
    crate::head_graph::run_with(p, &mut runner, names, &mut HashMap::new(), &mut crate::head_graph::Persist::new(), &|_, n| n.starts_with("compute_graph_ops:"), &mut ex)?;
    lines.push(format!("head network, arithmetic {round:?}, {}: op kind | sinks | equal | halves that differ | most f16 steps | largest difference", if resync { "kernel by kernel (ours take the oracle's buffers after each sink)" } else { "chained (ours run on their own results)" }));
    let mut total = (0, 0, 0u64);
    for (k, (sinks, equal, differ, far, big)) in &tally {
        lines.push(format!("  {k} | {sinks} | {equal} | {differ} | {far} | {big:e}"));
        total = (total.0 + sinks, total.1 + equal, total.2 + differ);
    }
    lines.push(format!("  total | {} | {} | {} ; oracle halves non-zero beyond our buffers' reach after the last op: {outside}", total.0, total.1, total.2));
    Ok(lines)
}

/// The head table with the network on the CPU, as the live path makes it
/// (`head=on`): the frame's net from `kept` run here, its outputs held on
/// the device, then only the head programs dispatched
/// (`ped_standard_head_cs:*`, one batch). With the cost of the CPU part.
#[allow(clippy::too_many_arguments)]
pub fn run_on(p: &Pack, runner: &mut xnw_world::compute::Runner, names: &HashMap<u128, String>, programs: &mut crate::head_graph::Programs, persist: &mut crate::head_graph::Persist, kept: &mut Kept) -> Result<(crate::head_graph::Graph, Cost), String> {
    let (held, cost) = kept.outputs(p, names)?;
    let mut ex = crate::head_graph::Extra { held, each: None };
    let g = crate::head_graph::run_with(p, runner, names, programs, persist, &|_, n| n.starts_with("ped_standard_head_cs:"), &mut ex)?;
    Ok((g, cost))
}

/// `xnw-lab headnet <frame dir>... [--ops] [--check] [--chain] [--table] [--time N] [--console]`: per frame its
/// network op list (heads, ops per kind, buffers, whether an earlier frame had the same list); `--ops` every op;
/// `--check` each kernel against the dispatch replay on WARP, `--chain` the whole network against it; `--table`
/// the head table from the CPU network against the replay's; `--time N` ms a run and per kind. The arithmetic is
/// the oracle's unless `--console` (ours live: the console's rounding).
pub fn cmd(args: &[String]) -> Result<(), String> {
    let dirs: Vec<&String> = args.iter().take_while(|a| !a.starts_with("--")).collect();
    if dirs.is_empty() {
        return Err("usage: xnw-lab headnet <frame dir>... [--ops] [--check] [--chain] [--table] [--time N] [--console]".into());
    }
    let has = |k: &str| args.iter().any(|a| a == k);
    // `--console`: our default arithmetic (the console's rounding); otherwise the oracle's (WARP's).
    let round = if has("--console") { Round::Console } else { Round::Oracle };
    let times = args.iter().position(|a| a == "--time").and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<u32>().ok());
    let host = if has("--check") || has("--chain") || has("--table") { Some(oracle_host()?) } else { None };
    let mut seen: Vec<(String, Program)> = Vec::new();
    let mut kept = Kept { round, ..Kept::default() };
    for dir in dirs {
        let p = Pack::open(std::path::Path::new(dir))?.ok_or("no pack")?;
        let names = crate::audit::program_names(&p.dir).ok_or("head network: no shader-names.tsv above the frame")?;
        let g = program(&p, &names)?;
        let mut census: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
        for o in &g.ops {
            *census.entry(format!("{:?}", o.kind)).or_default() += 1;
        }
        let same = seen.iter().find(|x| x.1.same(&g)).map_or("a new op list".to_string(), |x| format!("the op list of {}", x.0));
        seen.push((dir.to_string(), g.clone()));
        let words: Vec<u8> = g.ops.iter().flat_map(|o| o.w.iter().flat_map(|w| w.to_le_bytes()).chain([o.groups as u8])).collect();
        println!("{dir}: {} heads, {} ops (sinks {:?}..{:?}), op list hash {:08x}: {same}; inputs {:x?}; outputs {:x?}; {census:?}", g.heads(), g.ops.len(), g.ops.first().map(|o| o.sink), g.ops.last().map(|o| o.sink), xnw_core::hash128(&[&words]) as u32, g.inputs().iter().map(|x| x.1.addr).collect::<Vec<_>>(), g.outputs());
        if has("--ops") {
            for l in g.lines() {
                println!("    {l}");
            }
        }
        if g.ops.is_empty() {
            continue;
        }
        let (made_before, t0) = (kept.cache.made, std::time::Instant::now());
        let mut n = net(&p, g.clone(), &mut kept.cache)?;
        n.round = round;
        println!("  net made in {:.2} ms ({} matrices made, {} found kept); formula ops {:?}", t0.elapsed().as_secs_f64() * 1e3, kept.cache.made - made_before, g.ops.iter().filter(|o| matches!(o.kind, Kind::VecMatMul | Kind::VecMatMulAligned | Kind::TransposeVecMatMulAligned)).count() as u32 - (kept.cache.made - made_before), n.formula_census());
        if let Some(runs) = times {
            n.run();
            let t = std::time::Instant::now();
            for _ in 0..runs {
                n.run();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(runs.max(1));
            let mut by: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
            for _ in 0..runs {
                for i in 0..n.prog.ops.len() {
                    let t = std::time::Instant::now();
                    n.step(i);
                    *by.entry(format!("{:?}", n.prog.ops[i].kind)).or_default() += t.elapsed().as_secs_f64() * 1e6 / f64::from(runs.max(1));
                }
            }
            let by: Vec<String> = by.iter().map(|(k, v)| format!("{k} {v:.0}")).collect();
            println!("  {runs} runs, arithmetic {round:?}, {}: {ms:.4} ms a run, {:.4} ms a head; output hash {:08x}; us a run by kind (timed op by op): {}", if simd() { "AVX2 + F16C rows" } else { "scalar rows" }, ms / g.heads().max(1) as f64, n.output_hash(), by.join(", "));
        }
        if let Some(host) = &host {
            if has("--check") {
                for l in check(&p, &names, host, true, round)? {
                    println!("  {l}");
                }
            }
            if has("--chain") {
                for l in check(&p, &names, host, false, round)? {
                    println!("  {l}");
                }
            }
            if has("--table") {
                let hash = |g: &crate::head_graph::Graph| (g.network.map(|x| x.3), g.table.as_ref().map(|(_, b)| xnw_core::hash128(&[b]) as u32));
                let replay = crate::head_graph::run(&p, host, &|_, _| true)?;
                let mut runner = xnw_world::compute::Runner::new(host)?;
                let (ours, cost) = run_on(&p, &mut runner, &names, &mut HashMap::new(), &mut crate::head_graph::Persist::new(), &mut kept)?;
                println!("  replay: {}", replay.line());
                println!("  CPU network ({cost:.3?}): {}", ours.line());
                println!("  head table: {} (network output, table): replay {:08x?}, CPU network {:08x?}", if hash(&replay) == hash(&ours) { "EQUAL" } else { "DIFFERENT" }, hash(&replay), hash(&ours));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_trip_and_round_to_even() {
        for h in 0..=0xffffu16 {
            let f = h2f(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f2h(f), h, "{h:#x}");
        }
        assert_eq!(h2f(0x3c00), 1.0);
        assert_eq!(h2f(0x0001), 2f32.powi(-24));
        assert_eq!(h2f(0xc000), -2.0);
        // 1 + 2^-11 is the tie between 0x3c00 and 0x3c01: to even; just above: up.
        assert_eq!(f2h(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f2h(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f2h(f32::from_bits((1.0f32 + 2f32.powi(-11)).to_bits() + 1)), 0x3c01);
        assert_eq!(f2h(65519.99), 0x7bff);
        assert_eq!(f2h(65520.0), 0x7c00);
        assert_eq!(f2h(2f32.powi(-25)), 0);
        assert_eq!(f2h(1.5 * 2f32.powi(-25)), 1);
        assert_eq!(f2h(-0.0), 0x8000);
        // Toward zero: exact halves stay, anything between two halves takes the smaller magnitude.
        for h in (0..0x7c00u16).step_by(7) {
            assert_eq!(f2h_zero(h2f(h)), h);
            let between = f32::from_bits(h2f(h).to_bits() + 1);
            assert_eq!(f2h_zero(between), h, "{h:#x}");
            assert_eq!(f2h_zero(-between), h | 0x8000);
            assert_eq!(f2h_zero(f32::from_bits(h2f(h + 1).to_bits() - 1)), h, "{h:#x}");
        }
        assert_eq!(f2h_zero(1e9), 0x7bff);
        assert_eq!(f2h_zero(2f32.powi(-25)), 0);
    }

    #[test]
    fn names_give_kinds() {
        assert_eq!(kind_of("compute_graph_ops:CS_ZeroOperation_63bfc116_Wrapped"), Some(Kind::Zero));
        assert_eq!(kind_of("compute_graph_ops:CS_VecMatMulAlignedOperation_5ccdaacf_Wrapped"), Some(Kind::VecMatMulAligned));
        assert_eq!(kind_of("compute_graph_ops:CS_VecMatMulOperation_289cde74_Wrapped"), Some(Kind::VecMatMul));
        assert_eq!(kind_of("compute_graph_ops:CS_PlaceAddTechnique_da205a95_Wrapped"), Some(Kind::PlaceAdd));
        // Programs with no kernel here.
        assert_eq!(kind_of("compute_graph_ops:CS_SliceCopyAlignedTechnique_06adfb79_Wrapped"), None);
        assert_eq!(kind_of("compute_graph_ops:CS_TransposeVecMatMulOperation_5b98b712_Wrapped"), None);
        assert_eq!(kind_of("ped_standard_head_cs:CS_ComputeStickyLips_0_Wrapped"), None);
    }

    fn halves(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| f2h(*x).to_le_bytes()).collect()
    }

    /// A net of one op over an arena (buffer 0, 64 halves from `arena`) and
    /// a read-only buffer (1, `consts`).
    fn one(kind: Kind, groups: u32, w: &[u32], buf: [u8; 7], arena: &[f32], consts: Vec<u8>) -> Net {
        let mut ww = [0; 9];
        ww[..w.len()].copy_from_slice(w);
        let prog = Program { ops: vec![Op { kind, sink: 0, groups, w: ww, buf }], bufs: vec![Buf { addr: 0x1000, bytes: 512 }, Buf { addr: 0x2000, bytes: consts.len() as u64 }] };
        let mut n = Net::new(prog, &|_, _| Some(consts.clone()), &mut Cache::default()).unwrap();
        n.mem[0].resize(256, 0);
        for (i, x) in arena.iter().enumerate() {
            n.mem[0][i] = f2h(*x);
        }
        n
    }

    fn vals(n: &Net, at: usize, len: usize) -> Vec<f32> {
        n.mem[0][at..at + len].iter().map(|h| h2f(*h)).collect()
    }

    #[test]
    fn zero_place_and_slice() {
        let a: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let mut n = one(Kind::Zero, 1, &[1, 2], [0; 7], &a, vec![]);
        n.run();
        assert_eq!(vals(&n, 4, 16), [5., 6., 7., 8., 0., 0., 0., 0., 0., 0., 0., 0., 17., 18., 19., 20.]);
        // 6 halves asked: whole elements of 4 move (8 halves).
        let mut n = one(Kind::PlaceAligned, 1, &[16, 4, 6], [0; 7], &a, vec![]);
        n.run();
        assert_eq!(vals(&n, 16, 9), [5., 6., 7., 8., 9., 10., 11., 12., 25.]);
        let mut n = one(Kind::PlaceAddAligned, 1, &[16, 4, 4], [0; 7], &a, vec![]);
        n.run();
        assert_eq!(vals(&n, 16, 5), [22., 24., 26., 28., 21.]);
        let mut n = one(Kind::Place, 1, &[17, 3, 5], [0; 7], &a, vec![]);
        n.run();
        assert_eq!(vals(&n, 16, 7), [17., 4., 5., 6., 7., 8., 23.]);
        let mut n = one(Kind::PlaceAdd, 1, &[17, 3, 2], [0; 7], &a, vec![]);
        n.run();
        assert_eq!(vals(&n, 16, 4), [17., 22., 24., 20.]);
        // Row 1 of rows 5 wide, column 2, 3 halves, from element 1 * 8 to 3 * 8.
        let mut n = one(Kind::SliceCopy, 1, &[3, 1, 1, 3, 1, 5, 1, 2], [0; 7], &a, vec![]);
        n.run();
        assert_eq!(vals(&n, 24, 4), [16., 17., 18., 28.]);
    }

    #[test]
    fn clamp_concat_and_gather() {
        let a = [-1.0, 0.25, 0.5, 2.0, 3.0, -0.5, 0.75, 1.0];
        let mut n = one(Kind::Clamp, 1, &[2, 1, 0, 0, 0, 0, 0, 0x3c00], [0; 7], &a, vec![]);
        n.run();
        assert_eq!(vals(&n, 16, 8), [0.0, 0.25, 0.5, 1.0, 1.0, 0.0, 0.75, 1.0]);
        let a: Vec<f32> = (1..=40).map(|x| x as f32).collect();
        // 4 halves from element 1 * 8, then 8 from 2 * 8, to 4 * 8.
        let mut n = one(Kind::ConcatAligned, 1, &[4, 1, 2, 1, 4, 1, 8, 1, 1], [0; 7], &a, vec![]);
        n.run();
        assert_eq!(vals(&n, 32, 13), [9., 10., 11., 12., 17., 18., 19., 20., 21., 22., 23., 24., 0.]);
        // Indices (u32 as two halves) 5, 0, 9, 9, 2 at half 8 of the read-only buffer.
        let mut c = vec![0u8; 16];
        for i in [5u32, 0, 9, 9, 2] {
            c.extend(i.to_le_bytes());
        }
        let mut n = one(Kind::Gather, 1, &[4, 1, 1, 1, 5, 5, 1, 0], [0, 1, 0, 0, 0, 0, 0], &a, c);
        n.run();
        assert_eq!(vals(&n, 32, 6), [14., 9., 18., 18., 11., 38.]);
    }

    #[test]
    fn matrix_products_both_ways() {
        // 6 inputs at half 8, weights from half 8 of the read-only buffer.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut a = vec![0.0; 8];
        a.extend(x);
        // Vector x matrix: 6 rows (inputs) of 4 outputs.
        let m: Vec<f32> = (0..24).map(|k| (k % 5) as f32 - 1.5).collect();
        let want: Vec<f32> = (0..4).map(|j| (0..6).map(|i| x[i] * m[4 * i + j]).sum()).collect();
        let mut c = vec![0u8; 16];
        c.extend(halves(&m));
        let mut n = one(Kind::VecMatMulAligned, 1, &[4, 1, 1, 0, 1, 6, 4, 0], [1, 0, 0, 0, 0, 0, 0], &a, c.clone());
        n.run();
        assert_eq!(vals(&n, 32, 4), want);
        // The unaligned program: weights in slot 2, 3 outputs asked, 4 written (the 4th from the next row's start).
        let m3: Vec<f32> = (0..24).map(|k| (k % 7) as f32 - 2.0).collect();
        let mut c3 = vec![0u8; 16];
        c3.extend(halves(&m3));
        let mut n = one(Kind::VecMatMul, 1, &[4, 1, 1, 0, 1, 6, 3, 0], [0, 0, 1, 0, 0, 0, 0], &a, c3);
        n.run();
        let want3: Vec<f32> = (0..4).map(|j| (0..6).map(|i| x[i] * m3.get(3 * i + j).copied().unwrap_or(0.0)).sum()).collect();
        assert_eq!(vals(&n, 32, 4), want3);
        // Matrix x vector: 4 rows (outputs) of 6 weights... the aligned program takes inputs in fours: 8, the last two 0.
        let mut a8 = vec![0.0; 8];
        a8.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]);
        let mt: Vec<f32> = (0..32).map(|k| (k % 5) as f32 - 1.5).collect();
        let want_t: Vec<f32> = (0..4).map(|r| (0..6).map(|i| x[i] * mt[8 * r + i]).sum()).collect();
        let mut ct = vec![0u8; 16];
        ct.extend(halves(&mt));
        let mut n = one(Kind::TransposeVecMatMulAligned, 1, &[4, 1, 1, 0, 1, 8, 4, 0], [1, 0, 0, 0, 0, 0, 0], &a8, ct);
        n.run();
        assert_eq!(vals(&n, 32, 5), [want_t[0], want_t[1], want_t[2], want_t[3], 0.0]);
        // No input flag (w4 = 0): zeros.
        let mut n = one(Kind::VecMatMulAligned, 1, &[4, 1, 1, 0, 0, 6, 4, 0], [1, 0, 0, 0, 0, 0, 0], &a, c);
        n.mem[0][32] = 0x3c00;
        n.run();
        assert_eq!(vals(&n, 32, 4), [0.0; 4]);
    }

    #[test]
    fn the_fold_orders_and_the_wide_rows_agree_with_the_scalar_ones() {
        // 20 blocks of 4 inputs, 19 outputs: both fold orders, each against a plain scalar fold.
        let (nin, nout) = (80, 19);
        let m: Vec<f32> = (0..nin * nout).map(|k| h2f(f2h(((k * 7919) % 2003) as f32 / 977.0 - 1.0))).collect();
        let x: Vec<f32> = (0..nin).map(|k| h2f(f2h(((k * 104729) % 1009) as f32 / 500.0 - 1.0))).collect();
        let rd = Round::Console;
        {
        let block = |b: usize, j: usize| {
            let p = |c: usize| x[4 * b + c] * m[(4 * b + c) * nout + j];
            rnd(rnd(p(0) + rnd(p(1), rd), rd) + rnd(p(2) + rnd(p(3), rd), rd), rd)
        };
        for far in [false, true] {
            let (mut s, mut t) = (Vec::new(), Vec::new());
            product(&m, &x, nout, far, rd, &mut s, &mut t);
            for j in 0..nout {
                let mut v: Vec<f32> = (0..64).map(|b| if b < 20 { block(b, j) } else { 0.0 }).collect();
                if far {
                    for h in [32, 16, 8, 4, 2, 1] {
                        for y in 0..h {
                            v[y] = rnd(v[y] + v[y + h], rd);
                        }
                    }
                } else {
                    let mut n = 64;
                    while n > 1 {
                        for k in 0..n / 2 {
                            v[k] = rnd(v[2 * k] + v[2 * k + 1], rd);
                        }
                        n /= 2;
                    }
                }
                assert_eq!(s[j].to_bits(), v[0].to_bits(), "far {far} output {j} {rd:?}");
            }
        }
        }
    }

    #[test]
    fn formula_byte_code() {
        // Buffers: 0 arena, 1 the code, 2 the inputs, 3 the output. Header at half 8 of the code (w4 = 1):
        // 3 inputs, 2 constants, 32 lanes, 2 stages; the 32 programs' offsets at +8 (program 0 at +40, the rest
        // at +120, a program of no stage); the constants 10, 20 at half 0 (w3 = 0).
        let mut f = vec![0u16; 8 + 128];
        f[0] = f2h(10.0);
        f[1] = f2h(20.0);
        f[8..12].copy_from_slice(&[3, 2, 32, 2]);
        f[16] = 40;
        f[17..48].fill(120);
        let bytes = |f: &mut Vec<u16>, half: usize, b: &[u8]| {
            for (k, x) in b.iter().enumerate() {
                let h = &mut f[8 + half + k / 2];
                *h = (*h & !(0xff << (8 * (k & 1)))) | (u16::from(*x) << (8 * (k & 1)));
            }
        };
        // Program 0: 2 stages; stage 0 has 1 instruction, at the next multiple of 8 halves (+48):
        // op 3 (add) to 16, 2 lanes: operands (0, 4) and (1, 5).
        f[8 + 40] = 2;
        f[8 + 41] = 1;
        bytes(&mut f, 48, &[3, 16, 0, 2, 0, 0, 0, 4, 0, 1, 0, 5, 0]);
        // It ends at +48 + 3 + 2 * 2 = +55: stage 1's count there, its instruction at +56:
        // op 2 (output), 2 lanes: output 7 from 16, output 3 from 20 (a lane's result is 4 halves on).
        f[8 + 55] = 1;
        bytes(&mut f, 56, &[2, 0, 0, 2, 0, 7, 0, 16, 0, 3, 0, 20, 0]);
        let mut input = vec![0u16; 8];
        input.extend([1.0, 2.0, 3.0].map(f2h));
        let op = Op { kind: Kind::Formula, sink: 0, groups: 1, w: [0, 0, 1, 0, 1, 0, 0, 0, 0], buf: [0, 1, 1, 0, 2, 3, 1] };
        let b = |addr| Buf { addr, bytes: 1 << 20 };
        let prog = Program { ops: vec![op], bufs: vec![b(0x1000), b(0x2000), b(0x3000), b(0x4000)] };
        let le = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut n = Net::new(prog, &|i, _| Some(if i == 1 { le(&f) } else { le(&input) }), &mut Cache::default()).unwrap();
        assert_eq!(n.formula_census()[2..4], [1, 1]);
        n.run();
        assert_eq!(n.mem[0][..8].iter().map(|h| h2f(*h)).collect::<Vec<_>>(), [1.0, 2.0, 3.0, 0.0, 10.0, 20.0, 0.0, 0.0]);
        assert_eq!((h2f(n.mem[0][16]), h2f(n.mem[0][20])), (11.0, 22.0));
        assert_eq!((h2f(n.mem[3][7]), h2f(n.mem[3][3])), (11.0, 22.0));
        let e = Round::Console;
        assert_eq!(formula_op(4, f2h(5.0), f2h(2.0), e), f2h(3.0));
        assert_eq!(formula_op(6, f2h(5.0), f2h(2.0), e), f2h(2.5));
        assert_eq!(formula_op(8, f2h(5.0), 0, e), f2h(-5.0));
        assert_eq!(formula_op(9, f2h(5.0), f2h(2.0), e), f2h(2.0));
        assert_eq!((formula_op(10, 0x8000, 0, e), formula_op(10, 0, 0x8000, e), formula_op(10, 0x8000, 0x8000, e)), (0, 0, 0x8000));
        assert_eq!((formula_op(9, 0x8000, 0, e), formula_op(10, f2h(-1.0), f2h(2.0), e)), (0x8000, f2h(2.0)));
        // The oracle's conversion truncates: 0x124f + 0xa7a2 (pk8a f1027 sink 510).
        assert_eq!((formula_op(3, 0x124f, 0xa7a2, Round::Oracle), formula_op(3, 0x124f, 0xa7a2, e)), (0xa76f, 0xa770));
        assert_eq!(formula_op(14, f2h(9.0), 0, e), f2h(3.0));
    }
}
