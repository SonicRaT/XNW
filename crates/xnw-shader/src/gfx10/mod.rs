// Ported from VIctory crates/victory-shader/src/gfx10/mod.rs, commit d2a4bae1f49533ef5cb512fd1b1ba1513f2e1989, GPL-3.0-or-later.
// Changed: the opcode name is looked up once ([`Op`]), words are stored inline
// ([`Words`]).
//! GFX10 machine code: instruction boundaries, encodings and opcodes.
//!
//! Instructions are found by following control flow from offset 0, not by a
//! linear sweep: compilers place data (constant tables) in the code region
//! after the end of the program. [`Instruction::fields`] decodes the operand
//! fields and [`Instruction::text`] prints them in AMD's SP3 syntax.
//!
//! Encodings by the top bits of the first dword:
//!
//! ```text
//! 0xxxxxxx  VOP2; 0111111 VOP1; 0111110 VOPC
//! 10        SOP2; 1011 SOPK; 101111101 SOP1; 101111110 SOPC; 101111111 SOPP
//! 110010 VINTRP  110011 VOP3P  110101 VOP3  110110 DS  110111 FLAT/GLOBAL/SCRATCH
//! 111000 MUBUF   111010 MTBUF  111100 MIMG  111101 SMEM  111110 EXP
//! ```

mod fields;
mod opcodes;
mod sp3;

pub use fields::{Cache, Dpp, Fields, NULL, Operand, Sdwa, Valu};

use std::collections::BTreeMap;

use crate::ShaderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    Sop2,
    Sopk,
    Sop1,
    Sopc,
    Sopp,
    Smem,
    Vop2,
    Vop1,
    Vopc,
    Vop3,
    Vop3p,
    Vintrp,
    Ds,
    Mubuf,
    Mtbuf,
    Mimg,
    Exp,
    Flat,
    Global,
    Scratch,
}

/// What the opcode is: a named GFX10 operation, or one our table lacks
/// (the instruction keeps its raw words either way).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    Known(&'static str),
    Unknown,
}

/// The longest instruction: MIMG with three extra NSA address dwords.
pub const MAX_WORDS: usize = 5;

/// An instruction's dwords, inline (no allocation per instruction).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Words {
    len: u8,
    w: [u32; MAX_WORDS],
}

impl std::ops::Deref for Words {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        &self.w[..usize::from(self.len)]
    }
}

impl std::fmt::Debug for Words {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter().map(|w| format!("{w:08X}"))).finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    /// Byte offset in the program.
    pub offset: usize,
    pub encoding: Encoding,
    pub opcode: u16,
    pub op: Op,
    /// Every dword of the instruction, including literals and extra dwords.
    pub words: Words,
}

impl Instruction {
    /// AMD's name for the opcode, if it is in our table.
    pub fn name(&self) -> Option<&'static str> {
        match self.op {
            Op::Known(n) => Some(n),
            Op::Unknown => None,
        }
    }

    /// Byte offset of the next instruction.
    pub fn end(&self) -> usize {
        self.offset + 4 * self.words.len()
    }
}

/// Decodes every instruction reachable from offset 0, sorted by offset.
pub fn decode(code: &[u8]) -> Result<Vec<Instruction>, ShaderError> {
    if !code.len().is_multiple_of(4) {
        return Err(isa(code.len(), "code size is not a multiple of 4"));
    }
    let mut found = BTreeMap::new();
    let mut pending = vec![0usize];
    while let Some(mut offset) = pending.pop() {
        while !found.contains_key(&offset) {
            let inst = instruction(code, offset)?;
            let next = offset + 4 * inst.words.len();
            let flow = control(&inst, code.len(), |at| {
                found.range(..at).next_back().map(|(_, i)| i)
            })?;
            found.insert(offset, inst);
            match flow {
                Flow::Next => offset = next,
                Flow::Stop => break,
                Flow::Jump(target) => offset = target,
                Flow::Branch(target) => {
                    pending.push(target);
                    offset = next;
                }
            }
        }
    }
    let list: Vec<Instruction> = found.into_values().collect();
    if let Some(w) = list
        .windows(2)
        .find(|w| w[0].offset + 4 * w[0].words.len() > w[1].offset)
    {
        return Err(isa(w[1].offset, "a branch lands inside an instruction"));
    }
    Ok(list)
}

/// Where execution goes after an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flow {
    Next,
    /// End of the program, or an indirect jump that is not a far jump.
    Stop,
    Jump(usize),
    /// Conditional: to the target or the next instruction.
    Branch(usize),
}

/// Control flow after `inst`, resolving far jumps. `before(at)` returns the
/// decoded instruction with the highest offset below `at`, if any.
pub(crate) fn control<'a>(
    inst: &Instruction,
    len: usize,
    before: impl Fn(usize) -> Option<&'a Instruction>,
) -> Result<Flow, ShaderError> {
    Ok(match flow(inst, inst.offset + 4 * inst.words.len(), len)? {
        Flow::Stop if inst.name() == Some("s_setpc_b64") => {
            far_jump(before, inst, len)?.map_or(Flow::Stop, Flow::Jump)
        }
        flow => flow,
    })
}

fn flow(inst: &Instruction, next: usize, len: usize) -> Result<Flow, ShaderError> {
    let target = || {
        let delta = 4 * i64::from(inst.words[0] as u16 as i16);
        usize::try_from(next as i64 + delta)
            .ok()
            .filter(|&t| t < len)
            .ok_or(isa(inst.offset, "branch target outside the code"))
    };
    Ok(match (inst.encoding, inst.name().unwrap_or("")) {
        (
            Encoding::Sopp,
            "s_endpgm" | "s_endpgm_saved" | "s_endpgm_ordered_ps_done" | "s_code_end",
        ) => Flow::Stop,
        (Encoding::Sopp, "s_branch") => Flow::Jump(target()?),
        (Encoding::Sopp, name) if name.starts_with("s_cbranch_") => Flow::Branch(target()?),
        (Encoding::Sopk, "s_call_b64") => Flow::Branch(target()?),
        // Indirect jump; `decode` resolves the compiler's far-jump form.
        (Encoding::Sop1, "s_setpc_b64") => Flow::Stop,
        _ => Flow::Next,
    })
}

/// Resolves the far jump compilers emit beyond the reach of a branch offset:
/// `s_getpc_b64 s[n:n+1]`, `s_add_u32|s_sub_u32 sn, sn, literal`, a carry
/// into s(n+1), `s_setpc_b64 s[n:n+1]`, looking back over the straight-line
/// run just decoded. Any other `s_setpc_b64` (e.g. a handover to the next
/// merged stage through registers) gives `None`.
fn far_jump<'a>(
    before: impl Fn(usize) -> Option<&'a Instruction>,
    setpc: &Instruction,
    len: usize,
) -> Result<Option<usize>, ShaderError> {
    let n = setpc.words[0] & 0xFF;
    let mut delta = None;
    let mut at = setpc.offset;
    for _ in 0..4 {
        let Some(prev) = before(at) else {
            break;
        };
        if prev.offset + 4 * prev.words.len() != at {
            break;
        }
        let w = prev.words[0];
        let (sdst, ssrc0, ssrc1) = ((w >> 16) & 0x7F, w & 0xFF, (w >> 8) & 0xFF);
        match (prev.encoding, prev.name()) {
            (Encoding::Sop2, Some(op @ ("s_add_u32" | "s_sub_u32")))
                if sdst == n && ssrc0 == n && ssrc1 == 0xFF && delta.is_none() =>
            {
                // The low dword of a signed 64-bit offset: a backward jump
                // adds a negative one (its high dword is -1).
                let literal = i64::from(prev.words[1] as i32);
                delta = Some(if op == "s_add_u32" { literal } else { -literal });
            }
            (Encoding::Sop1, Some("s_getpc_b64")) if sdst == n => {
                let Some(delta) = delta else { break };
                // s_getpc_b64 yields the address of the instruction after it.
                return usize::try_from(prev.offset as i64 + 4 + delta)
                    .ok()
                    .filter(|&t| t < len)
                    .map(Some)
                    .ok_or(isa(setpc.offset, "far jump target outside the code"));
            }
            _ => {}
        }
        at = prev.offset;
    }
    Ok(None)
}

/// XNW: the one-dword instruction `word`, as if it stood at `offset`
/// (the IR's unspill puts moves in place of lane saves and restores).
pub(crate) fn one_word(word: u32, offset: usize) -> Option<Instruction> {
    let mut i = instruction(&word.to_le_bytes(), 0).ok()?;
    i.offset = offset;
    Some(i)
}

fn instruction(code: &[u8], offset: usize) -> Result<Instruction, ShaderError> {
    let word = |i: usize| {
        code.get(offset + 4 * i..offset + 4 * i + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
            .ok_or(isa(offset, "instruction runs past the end of the code"))
    };
    let w = word(0)?;
    let (encoding, opcode, len) = if w >> 31 == 0 {
        let src0 = w & 0x1FF;
        // Literal constant, SDWA, DPP16, DPP8, DPP8 with fetch-inactive.
        let extra = usize::from(matches!(src0, 0xFF | 0xF9 | 0xFA | 0xE9 | 0xEA));
        match w >> 25 {
            0x3F => (Encoding::Vop1, (w >> 9) & 0xFF, 1 + extra),
            0x3E => (Encoding::Vopc, (w >> 17) & 0xFF, 1 + extra),
            op => {
                let mk_ak = name(Encoding::Vop2, op as u16).is_some_and(|n| {
                    matches!(
                        n,
                        "v_madmk_f32"
                            | "v_madak_f32"
                            | "v_fmamk_f32"
                            | "v_fmaak_f32"
                            | "v_madmk_f16"
                            | "v_madak_f16"
                            | "v_fmamk_f16"
                            | "v_fmaak_f16"
                    )
                });
                (Encoding::Vop2, op, 1 + extra.max(usize::from(mk_ak)))
            }
        }
    } else if w >> 30 == 0b10 {
        let literal = |srcs: &[u32]| 1 + usize::from(srcs.contains(&0xFF));
        match w >> 23 {
            0x17D => (Encoding::Sop1, (w >> 8) & 0xFF, literal(&[w & 0xFF])),
            0x17E => (
                Encoding::Sopc,
                (w >> 16) & 0x7F,
                literal(&[w & 0xFF, (w >> 8) & 0xFF]),
            ),
            0x17F => (Encoding::Sopp, (w >> 16) & 0x7F, 1),
            _ if w >> 28 == 0b1011 => {
                let op = (w >> 23) & 0x1F;
                let imm32 = name(Encoding::Sopk, op as u16) == Some("s_setreg_imm32_b32");
                (Encoding::Sopk, op, 1 + usize::from(imm32))
            }
            _ => (
                Encoding::Sop2,
                (w >> 23) & 0x7F,
                literal(&[w & 0xFF, (w >> 8) & 0xFF]),
            ),
        }
    } else {
        let vop3 =
            |w1: u32| 2 + usize::from([w1, w1 >> 9, w1 >> 18].iter().any(|s| s & 0x1FF == 0xFF));
        match w >> 26 {
            0x32 => (Encoding::Vintrp, (w >> 16) & 0x3, 1),
            0x33 => (Encoding::Vop3p, (w >> 16) & 0x7F, vop3(word(1)?)),
            0x35 => (Encoding::Vop3, (w >> 16) & 0x3FF, vop3(word(1)?)),
            0x36 => (Encoding::Ds, (w >> 18) & 0xFF, 2),
            0x37 => {
                let encoding = match (w >> 14) & 3 {
                    0 => Encoding::Flat,
                    1 => Encoding::Scratch,
                    2 => Encoding::Global,
                    _ => return Err(isa(offset, "reserved FLAT segment")),
                };
                (encoding, (w >> 18) & 0x7F, 2)
            }
            0x38 => (Encoding::Mubuf, (w >> 18) & 0x7F | (w >> 25 & 1) << 7, 2),
            0x3A => (
                Encoding::Mtbuf,
                (w >> 16) & 0x7 | (word(1)? >> 21 & 1) << 3,
                2,
            ),
            // Bits 2:1 count extra dwords of non-sequential address registers.
            0x3C => (
                Encoding::Mimg,
                (w >> 18) & 0x7F | (w & 1) << 7,
                2 + (w as usize >> 1 & 3),
            ),
            0x3D => (Encoding::Smem, (w >> 18) & 0xFF, 2),
            0x3E => (Encoding::Exp, 0, 2),
            _ => return Err(isa(offset, "unknown encoding")),
        }
    };
    let mut words = Words {
        len: len as u8,
        w: [0; MAX_WORDS],
    };
    for i in 0..len {
        words.w[i] = word(i)?;
    }
    let opcode = opcode as u16;
    Ok(Instruction {
        offset,
        encoding,
        opcode,
        op: name(encoding, opcode).map_or(Op::Unknown, Op::Known),
        words,
    })
}

fn name(encoding: Encoding, opcode: u16) -> Option<&'static str> {
    use opcodes as t;
    let (table, op) = match encoding {
        Encoding::Sop2 => (t::SOP2, opcode),
        Encoding::Sopk => (t::SOPK, opcode),
        Encoding::Sop1 => (t::SOP1, opcode),
        Encoding::Sopc => (t::SOPC, opcode),
        Encoding::Sopp => (t::SOPP, opcode),
        Encoding::Smem => (t::SMEM, opcode),
        Encoding::Vop2 => (t::VOP2, opcode),
        Encoding::Vop1 => (t::VOP1, opcode),
        Encoding::Vopc => (t::VOPC, opcode),
        // VOP3 also encodes the VOPC, VOP2 and VOP1 opcodes at fixed bases.
        Encoding::Vop3 => match opcode {
            0x000..0x100 => (t::VOPC, opcode),
            0x100..0x140 => (t::VOP2, opcode - 0x100),
            0x180..0x200 => (t::VOP1, opcode - 0x180),
            _ => (t::VOP3, opcode),
        },
        Encoding::Vop3p => (t::VOP3P, opcode),
        Encoding::Vintrp => (t::VINTRP, opcode),
        Encoding::Ds => (t::DS, opcode),
        Encoding::Mubuf => (t::MUBUF, opcode),
        Encoding::Mtbuf => (t::MTBUF, opcode),
        Encoding::Mimg => (t::MIMG, opcode),
        Encoding::Exp => return Some("exp"),
        Encoding::Flat => (t::FLAT, opcode),
        Encoding::Global => (t::GLOBAL, opcode),
        Encoding::Scratch => (t::SCRATCH, opcode),
    };
    table
        .binary_search_by_key(&op, |&(o, _)| o)
        .ok()
        .map(|i| table[i].1)
}

fn isa(offset: usize, reason: &'static str) -> ShaderError {
    ShaderError::Isa { offset, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;
    /// s_branch / s_cbranch_scc0 with a dword offset relative to the next instruction.
    const fn sopp(op: u32, simm16: i16) -> u32 {
        0xBF80_0000 | op << 16 | simm16 as u16 as u32
    }

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn summary(code: &[u32]) -> Vec<(usize, usize, &'static str)> {
        decode(&bytes(code))
            .unwrap()
            .iter()
            .map(|i| (i.offset, i.words.len(), i.name().unwrap_or("?")))
            .collect()
    }

    #[test]
    fn tables_are_sorted() {
        for t in [
            opcodes::SOP2,
            opcodes::SOPK,
            opcodes::SOP1,
            opcodes::SOPC,
            opcodes::SOPP,
            opcodes::SMEM,
            opcodes::VOP2,
            opcodes::VOP1,
            opcodes::VOPC,
            opcodes::VOP3,
            opcodes::VOP3P,
            opcodes::VINTRP,
            opcodes::DS,
            opcodes::MUBUF,
            opcodes::MTBUF,
            opcodes::MIMG,
            opcodes::FLAT,
            opcodes::GLOBAL,
            opcodes::SCRATCH,
        ] {
            assert!(t.windows(2).all(|w| w[0].0 < w[1].0));
        }
    }

    #[test]
    fn lengths_and_names() {
        let code = [
            0xBE80_03FF,
            0x3F80_0000, // s_mov_b32 s0, literal
            0x0600_0000, // v_add_f32 v0, s0, v0
            0xD54B_0000,
            0x03FE_0000,
            0x4000_0000, // v_fma_f32 with a literal src2
            0xF080_0F0A,
            0,
            0, // image_sample (2D) with one extra address dword
            0xF800_080F,
            0, // exp
            S_ENDPGM,
        ];
        assert_eq!(
            summary(&code),
            [
                (0, 2, "s_mov_b32"),
                (8, 1, "v_add_f32"),
                (12, 3, "v_fma_f32"),
                (24, 3, "image_sample"),
                (36, 2, "exp"),
                (44, 1, "s_endpgm"),
            ]
        );
    }

    #[test]
    fn follows_branches_and_skips_data() {
        let code = [
            sopp(0x04, 1), // s_cbranch_scc0 -> 8
            sopp(0x02, 1), // s_branch -> 12
            S_ENDPGM,
            S_ENDPGM,
            S_ENDPGM, // data follows the program end
            0xFFFF_FFFF,
        ];
        let found: Vec<_> = summary(&code).iter().map(|i| i.0).collect();
        assert_eq!(found, [0, 4, 8, 12]);
    }

    #[test]
    fn follows_far_jumps() {
        let code = [
            0xBE88_1F00, // s_getpc_b64 s[8:9]
            0x8008_FF08,
            20,          // s_add_u32 s8, s8, 20  -> 4 + 20 = 24
            0x8209_8009, // s_addc_u32 s9, s9, 0
            0xBE80_2008, // s_setpc_b64 s[8:9]
            0xFFFF_FFFF, // not code
            S_ENDPGM,
        ];
        let found: Vec<_> = summary(&code).iter().map(|i| (i.0, i.2)).collect();
        assert_eq!(
            found,
            [
                (0, "s_getpc_b64"),
                (4, "s_add_u32"),
                (12, "s_addc_u32"),
                (16, "s_setpc_b64"),
                (24, "s_endpgm"),
            ]
        );
        // Through other registers (a handover to another stage): the path ends.
        assert_eq!(summary(&[0xBE80_2018, S_ENDPGM]).len(), 1);
    }

    #[test]
    fn follows_far_jumps_backward() {
        // The compiler's backward far jump adds a negative offset: the low
        // dword as a 32-bit literal, the high dword as -1 with the carry.
        let code = [
            sopp(0x02, 1), // s_branch -> 8, past the jump target
            S_ENDPGM,      // the far jump's target: only it reaches here
            0xBE88_1F00,   // s_getpc_b64 s[8:9]
            0x8008_FF08,
            0xFFFF_FFF8, // s_add_u32 s8, s8, -8 -> 12 - 8 = 4
            0x8209_C109, // s_addc_u32 s9, s9, -1
            0xBE80_2008, // s_setpc_b64 s[8:9]
        ];
        let found: Vec<_> = summary(&code).iter().map(|i| (i.0, i.2)).collect();
        assert_eq!(
            found,
            [
                (0, "s_branch"),
                (4, "s_endpgm"),
                (8, "s_getpc_b64"),
                (12, "s_add_u32"),
                (20, "s_addc_u32"),
                (24, "s_setpc_b64"),
            ]
        );
    }

    /// One instruction of each encoding (and the extra-dword forms): its
    /// encoding, name, length and SP3 text. Words marked "pack" are taken
    /// from the title's shader pack (`xnw shader-info --disasm`); the rest
    /// are built from the RDNA field layouts.
    #[test]
    fn encodings() {
        let cases: &[(&[u32], Encoding, &str, &str)] = &[
            (&[0xB005_1234], Encoding::Sopk, "s_movk_i32", "s_movk_i32 s5, 0x1234"),
            (&[0xBF06_0201], Encoding::Sopc, "s_cmp_eq_u32", "s_cmp_eq_u32 s1, s2"),
            (&[0xBFA0_0003], Encoding::Sopp, "s_inst_prefetch", "s_inst_prefetch 0x0003"), // pack
            (&[0xBE88_047E], Encoding::Sop1, "s_mov_b64", "s_mov_b64 s[8:9], exec"), // pack
            (&[0x8A88_7E08], Encoding::Sop2, "s_andn2_b64", "s_andn2_b64 s[8:9], s[8:9], exec"), // pack
            (&[0xBE80_03FF, 0x1234_5678], Encoding::Sop1, "s_mov_b32", "s_mov_b32 s0, lit(0x12345678)"),
            (&[0xF40C_0001, 0xFA00_0000], Encoding::Smem, "s_load_dwordx8", "s_load_dwordx8 s[0:7], s[2:3], null"), // pack
            (&[0x7E00_1102], Encoding::Vop1, "v_cvt_i32_f32", "v_cvt_i32_f32 v0, v2"), // pack
            (&[0x3600_0082], Encoding::Vop2, "v_and_b32", "v_and_b32 v0, 2, v0"), // pack
            (&[0x1004_0AFF, 0x3F00_0001], Encoding::Vop2, "v_mul_f32", "v_mul_f32 v2, lit(0x3f000001), v5"),
            (&[0x7C24_0080], Encoding::Vopc, "v_cmpx_eq_f32", "v_cmpx_eq_f32 exec, 0, v0"), // pack
            (&[0xD54B_0002, 0x03CD_E907], Encoding::Vop3, "v_fma_f32", "v_fma_f32 v2, v7, 2.0, -1.0"), // pack
            (&[0xCC20_0C01, 0x2C10_0702], Encoding::Vop3p, "v_fma_mix_f32", "v_fma_mix_f32 v1, -sel_hi(v2), s3, abs(v4)"),
            (&[0xC808_0500], Encoding::Vintrp, "v_interp_p1_f32", "v_interp_p1_f32 v2, v0, attr1.y"),
            (&[0xD8DC_0302, 0x0600_0001], Encoding::Ds, "ds_read2_b32", "ds_read2_b32 v[6:7], v1 offset0:2 offset1:3"),
            (
                &[0xE034_7010, 0x0102_0402],
                Encoding::Mubuf,
                "buffer_load_dwordx2",
                "buffer_load_dwordx2 v[4:5], v[2:3], s[8:11], s1 idxen offen offset:16 glc",
            ),
            (&[0xF000_1108, 0x0000_0000], Encoding::Mimg, "image_load", "image_load v0, v[0:1], s[0:7] dim:SQ_RSRC_IMG_2D unorm"), // pack
            (&[0xF800_08CF, 0x0404_0302], Encoding::Exp, "exp", "exp pos0, v2, v3, v4, v4 done"), // pack
            (&[0x7E06_02E9, 0x0539_7701], Encoding::Vop1, "v_mov_b32", "v_mov_b32 v3, v1 dpp8:[7,6,5,4,3,2,1,0]"),
            (&[0x7E0A_02FA, 0xFF08_B106], Encoding::Vop1, "v_mov_b32", "v_mov_b32 v5, v6 quad_perm:[1,0,3,2] bound_ctrl:1"),
            (&[0x7E02_16F9, 0x0605_0602], Encoding::Vop1, "v_cvt_f32_f16", "v_cvt_f32_f16 v1, v2 src0_sel:WORD_1"),
        ];
        for &(words, encoding, name, text) in cases {
            let i = instruction(&bytes(words), 0).unwrap();
            assert_eq!((i.encoding, i.name(), &i.words[..]), (encoding, Some(name), words), "{words:08X?}");
            assert_eq!(i.text(true), text, "{words:08X?}");
        }
        // Encodings checked by name and length only.
        let shapes: &[(&[u32], Encoding, &str)] = &[
            (&[0xE800_0000, 0x0000_0000], Encoding::Mtbuf, "tbuffer_load_format_x"),
            (&[0xDC30_0000, 0x0200_0000], Encoding::Flat, "flat_load_dword"),
            (&[0xDC30_8000, 0x027D_0000], Encoding::Global, "global_load_dword"),
            (&[0xDC70_4000, 0x007D_0100], Encoding::Scratch, "scratch_store_dword"),
            (&[0xF09C_070A, 0x8024_0801, 0x0000_0005], Encoding::Mimg, "image_sample_lz"),
            (&[0xD54B_8004, 0x2BC0_0501], Encoding::Vop3, "v_fma_f32"),
            (&[0xD401_0106, 0x0001_E501], Encoding::Vop3, "v_cmp_lt_f32"),
            (&[0x3E00_0280], Encoding::Vop2, "v_mac_f32"),
            (&[0x4200_0280, 0x3F80_0000], Encoding::Vop2, "v_madak_f32"),
            (&[0xD541_0000, 0x0402_0280], Encoding::Vop3, "v_mad_f32"),
        ];
        for &(words, encoding, name) in shapes {
            let i = instruction(&bytes(words), 0).unwrap();
            assert_eq!((i.encoding, i.name(), &i.words[..]), (encoding, Some(name), words), "{words:08X?}");
        }
    }

    #[test]
    fn unknown_opcodes_keep_their_words() {
        // SOP2 opcode 0x3f and MIMG opcode 0x7f are not GFX10 operations.
        for words in [&[0x9F80_0000u32][..], &[0xF1FC_0000, 0x0000_0000]] {
            let mut code = bytes(words);
            code.extend(bytes(&[S_ENDPGM]));
            let list = decode(&code).unwrap();
            assert_eq!(list[0].op, Op::Unknown);
            assert_eq!(&list[0].words[..], words);
            assert_eq!(list[1].name(), Some("s_endpgm"));
        }
    }

    /// RDNA2 (GFX10.3) has the fused legacy multiply-adds at these
    /// opcodes: LLVM's gfx1030 assembler tests encode
    /// `v_fma_legacy_f32 v0, v1, v2, v3` and `v_fmac_legacy_f32_e32 v0,
    /// v1, v2` as these words. (VIctory.)
    #[test]
    fn legacy_multiply_adds_are_the_fused_ones() {
        let fma = decode(&bytes(&[0xD540_0000, 0x040E_0501, S_ENDPGM])).unwrap();
        assert_eq!(fma[0].name(), Some("v_fma_legacy_f32"));
        let fmac = decode(&bytes(&[0x0C00_0501, S_ENDPGM])).unwrap();
        assert_eq!(fmac[0].name(), Some("v_fmac_legacy_f32"));
    }

    #[test]
    fn rejects_bad_code() {
        assert!(decode(&[0; 3]).is_err(), "size not a multiple of 4");
        assert!(
            decode(&bytes(&[sopp(0x02, 100)])).is_err(),
            "branch out of range"
        );
        assert!(
            decode(&bytes(&[0xBE80_03FF])).is_err(),
            "literal past the end"
        );
        assert!(
            decode(&bytes(&[0xE400_0000, 0])).is_err(),
            "unknown encoding"
        );
        // s_cbranch_scc0 into the literal of the s_mov_b32 at 4.
        assert!(decode(&bytes(&[sopp(0x04, 1), 0xBE80_03FF, S_ENDPGM, S_ENDPGM])).is_err());
    }
}
