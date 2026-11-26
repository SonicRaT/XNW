// Ported from VIctory crates/victory-shader/src/emit/valu_tests.rs, commit d2a4bae1f49533ef5cb512fd1b1ba1513f2e1989, GPL-3.0-or-later.
// With VIctory (fa5189d, GPL-3.0-or-later): the fused legacy multiply-adds of GFX10.3.
// With VIctory (3ad06fb, da1a1f7): ds_swizzle_b32 rotate and FFT modes.
// With VIctory (499bd98, GPL-3.0-or-later): multi-term, scaled, masked and phi-joined addresses in resource tracking.
// With victory-fork (open pull request, emit/valu_tests.rs, head 5a55384, GPL-3.0-or-later): the three half-view tests.
// With victory-fork (3bff182; VIctory-7900XTX 342245e, b17b69f): the v_cvt_f32_f64 and whole-wave f64 sum tests.
//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, and turn-range reduction for `v_sin_*`/`v_cos_*`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SOPP `op simm16`.
const fn sopp(op: u8, simm16: i16) -> u32 {
    0xBF80_0000 | (op as u32 & 0x7F) << 16 | simm16 as u16 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1` (`vsrc1` a VGPR number). XNW addition.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// VOP3B `op vdst, sdst, src0, src1, src2`.
const fn vop3b(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | (sdst as u32 & 0x7F) << 8 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// GLOBAL `global_load_dword vdst, vaddr, saddr` (`saddr = 125` for null).
const fn global_load_dword(vdst: u8, vaddr: u8, saddr: u8, offset: i16) -> [u32; 2] {
    [
        0x37 << 26 | 0x0C << 18 | 2 << 14 | (offset as u32 & 0xFFF),
        (vdst as u32) << 24 | (saddr as u32 & 0x7F) << 16 | vaddr as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// DS `ds_swizzle_b32 vdst, vsrc offset:offset`.
const fn ds_swizzle_b32(vdst: u8, vsrc: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | 0x35 << 18 | offset as u32,
        (vdst as u32) << 24 | vsrc as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, voff, s[srsrc:srsrc+3], 0 offen`.
const fn buffer_store_dword_offen(vdata: u8, voff: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | voff as u32,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    build_valu_cs_with(body, super::Fma::Exact)
}

/// The same with a chosen multiply-add form (XNW addition).
fn build_valu_cs_with(body: &[u32], fma: super::Fma) -> HostShader {
    build_wide(body, fma, 1, false)
}

/// The same with `lanes` threads per group and the wave size (XNW addition).
fn build_wide(body: &[u32], fma: super::Fma, lanes: u32, wave64: bool) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);
    lower_cs(&words, lanes, fma, wave64)
}

/// Lowers a synthetic RDNA2 compute shader of `threads` threads in one row
/// (thread id in `v0`): 4 dwords of root constants in `s0..s3`, the
/// descriptor table address in `s4`, a raw UAV at heap index 0.
fn build_cs(words: &[u32], threads: u32) -> HostShader {
    lower_cs(words, threads, super::Fma::Exact, false)
}

/// [`build_cs`] with the multiply-add form and wave size (XNW addition).
fn lower_cs(words: &[u32], threads: u32, fma: super::Fma, wave64: bool) -> HostShader {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        fallback: false,
        fma,
        gpu_desc: false,
        probe: false,
        loop_counts: None,
        pagetable: false,
        barycentrics: false,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        half_views: lowered.half_views,
        address_map: lowered.address_map,
        descriptor_words: lowered.descriptor_words,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    // One test at a time: with these tests on the hardware adapter and WARP
    // from several test threads at once, now and then none of their
    // dispatches ever completes and the test binary hangs.
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    // Changed from VIctory: the host device only (no WARP run).
    for label in ["hardware"] {
        let Some(host) = Host::for_tests() else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: Host::roots(shader, ud, 1, 0),
                        // XNW: 16 dwords, room for the memory tests.
                        views: &[View::RawUav(16)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    // Without the Agility SDK (XNW_AGILITY) Windows 10 has no SM 6.6 device:
    // skip, as every GPU test in the workspace does.
    if ran == 0 {
        eprintln!("SKIPPED {}: no SM 6.6 device (set XNW_AGILITY)", std::thread::current().name().unwrap_or("valu test"));
    }
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

#[test]
fn v_fma_legacy_f32_follows_the_d3d9_zero_rule() {
    // v_fma_legacy_f32 v0, v1, v2, v3 (VOP3 opcode 0x140 on RDNA2)
    let shader = build_valu_cs(&vop3(0x140, 0, vgpr(1), vgpr(2), vgpr(3)));
    let f = f32::to_bits;
    let cases: &[([u32; 4], u32)] = &[
        ([f(2.0), f(3.0), f(1.0), 0], f(7.0)),
        // A zero factor gives a zero product, whatever the other one is.
        ([0, f(f32::INFINITY), f(1.5), 0], f(1.5)),
        ([f(f32::NAN), 0, f(-2.0), 0], f(-2.0)),
        // XNW: zero times infinity plus c is c.
        ([f(f32::INFINITY), 0, f(3.0), 0], f(3.0)),
        // XNW: fused, one rounding: (1 + 2^-23)^2 - (1 + 2^-22) = 2^-46
        // (two roundings give 0).
        ([0x3F80_0001, 0x3F80_0001, 0xBF80_0002, 0], 0x2880_0000),
        // XNW (VIctory's claim): a denormal factor counts as zero.
        ([0x0000_0001, f(f32::INFINITY), f(2.5), 0], f(2.5)),
        // XNW (another runtime claim): the zero product is +0, so +0 + -0 = +0.
        ([0, f(5.0), 0x8000_0000, 0], 0),
        ([0x8000_0000, f(5.0), 0x8000_0000, 0], 0),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, _, got| {
        let want = cases[i].1;
        assert_eq!(got, want, "{backend}: case {i}: {got:#x}, want {want:#x}");
    });
}

/// XNW: `v_mul_legacy_f32` (VOP2 0x007): zero or denormal times anything is +0.
#[test]
fn v_mul_legacy_f32_zero_and_denormal_factors() {
    let shader = build_valu_cs(&[vop2(0x007, 0, vgpr(1), 2)]);
    let f = f32::to_bits;
    let cases: &[([u32; 4], u32)] = &[
        ([f(2.0), f(3.0), 0, 0], f(6.0)),
        ([0, f(f32::INFINITY), 0, 0], 0),
        ([f(f32::NAN), 0, 0, 0], 0),
        ([0x0000_0001, f(f32::INFINITY), 0, 0], 0),
        ([f(f32::INFINITY), 0x8040_0000, 0, 0], 0),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, _, got| {
        let want = cases[i].1;
        assert_eq!(got, want, "{backend}: case {i}: {got:#x}, want {want:#x}");
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

// ---- VIctory (3ad06fb, da1a1f7): ds_swizzle_b32 modes ----


/// One wave of 32 lanes holding 1..=32 (the input of the ISA's examples)
/// through `ds_swizzle_b32` with `offset`: what each lane reads.
fn swizzle_32(offset: u16) -> Vec<Vec<u32>> {
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.push(vop2(0x25, 1, 129, 0)); // v_add_nc_u32 v1, 1, v0
    words.extend_from_slice(&ds_swizzle_b32(2, 1, offset));
    words.push(vop2(0x1A, 3, 130, 0)); // v_lshlrev_b32 v3, 2, v0
    words.extend_from_slice(&buffer_store_dword_offen(2, 3, 8));
    words.push(S_ENDPGM);
    let shader = build_cs(&words, 32);

    let mut out = Vec::new();
    // Changed from VIctory: the host device only (no WARP run).
    for (label, warp) in [("hardware", false)] {
        let Some(host) = Host::for_tests() else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // A host without 32-lane waves cannot run it.
            Err(e) if warp => {
                eprintln!("{label}: no pipeline ({e}); skipped");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let uavs = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: Host::roots(&shader, ud, 1, 0),
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push(uavs[0].clone());
    }
    if out.is_empty() {
        eprintln!("no SM 6.6 device with 32-lane waves; skipped");
    }
    out
}

/// `got == want`, except on the software adapter, where a difference is
/// printed and not asserted: WARP 1.0.22 under `WaveSize(32|64)` reads the
/// wrong lane in a `WaveReadLaneAt` whose index differs between lanes
/// (rotations and the FFT orders; not the and/or/xor and quad modes). The
/// case in plain HLSL, no translator: `cargo test -p xnw-gpu --test
/// wave_packing lane_reads -- --nocapture` (rotate: lane 6 reads 16, want 8;
/// bit-reversed: lane 1 reads 1, want 17; native waves right). The same
/// lists as these tests fail with: rotate `[2, 3, 4, 5, 6, 7, 16, 1, ...]`.
fn assert_lanes(got: &[u32], want: &[u32], what: &str) {
    if got != want && xnw_device::warp() {
        eprintln!("KNOWN (WARP, lane_reads): {what}: got {got:?}, want {want:?}; not asserted on the software adapter");
        return;
    }
    assert_eq!(got, want, "{what}");
}

/// The basic modes: a quad permutation and a 32-lane and/or/xor mapping.
#[test]
fn ds_swizzle_basic_modes() {
    // offset[15]: lanes of each quad read lanes 3, 2, 1, 0 of it.
    let reversed: Vec<u32> = (0..32).map(|i| (i ^ 3) + 1).collect();
    // and 0x1f, or 0, xor 0x10: the halves swap.
    let swapped: Vec<u32> = (0..32).map(|i| (i ^ 16) + 1).collect();
    for (offset, want) in [(0x801B, reversed), (0x401F, swapped)] {
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
}

/// Offsets from 0xE000: the FFT orders, as the ISA lists them for inputs
/// 1..=0x20 (RDNA2 ISA, 12.13.1).
#[test]
fn ds_swizzle_fft_mode() {
    let cases: [(u16, [u32; 32]); 3] = [
        (
            0xE000,
            [
                0x01, 0x11, 0x09, 0x19, 0x05, 0x15, 0x0d, 0x1d, 0x03, 0x13, 0x0b, 0x1b, 0x07, 0x17,
                0x0f, 0x1f, 0x02, 0x12, 0x0a, 0x1a, 0x06, 0x16, 0x0e, 0x1e, 0x04, 0x14, 0x0c, 0x1c,
                0x08, 0x18, 0x10, 0x20,
            ],
        ),
        (
            0xE010,
            [
                0x01, 0x09, 0x05, 0x0d, 0x03, 0x0b, 0x07, 0x0f, 0x02, 0x0a, 0x06, 0x0e, 0x04, 0x0c,
                0x08, 0x10, 0x11, 0x19, 0x15, 0x1d, 0x13, 0x1b, 0x17, 0x1f, 0x12, 0x1a, 0x16, 0x1e,
                0x14, 0x1c, 0x18, 0x20,
            ],
        ),
        // 0x1f: no swizzle.
        (0xE01F, std::array::from_fn(|i| i as u32 + 1)),
    ];
    for (offset, want) in cases {
        for got in swizzle_32(offset) {
            assert_lanes(&got, &want, &format!("offset {offset:#06x}"));
        }
    }
    // The other bits of an FFT offset do not matter.
    for got in swizzle_32(0xFFF0) {
        assert_lanes(&got, &cases[1].1, "offset 0xfff0");
    }
}

/// Offsets from 0xC000 to 0xDFFF: the wave's 32 lanes rotate left by
/// offset[9:5], or right with offset[10] (RDNA2 ISA, 12.13.1).
#[test]
fn ds_swizzle_rotate_mode() {
    let rotated = |by: u32| -> Vec<u32> { (0..32).map(|i| (i + by) % 32 + 1).collect() };
    for (offset, by) in [
        (0xC000 | 1 << 5, 1),            // the ISA's example: {2, 3, ... 0x20, 1}
        (0xC000 | 1 << 10 | 1 << 5, 31), // and {0x20, 1, ... 0x1f}
        (0xC000, 0),
        (0xC000 | 5 << 5, 5),
        (0xC000 | 1 << 10 | 5 << 5, 27),
        (0xD800 | 31 << 5, 31), // bits 11 and 12 are not read
    ] {
        for got in swizzle_32(offset) {
            assert_lanes(&got, &rotated(by), &format!("offset {offset:#06x}"));
        }
    }
}

/// Rotations with a mask (offset[4:0]): the mask's bits of the lane index
/// stay, the others rotate, as the ISA manual describes it. Its examples for
/// mask 1 list other lanes; the lanes here are the ones measured on RDNA2
/// hardware (gfx1036, reported in a VIctory fork, which gives the first
/// and last entries of each list and reports the manual's rule matching all 2048
/// offsets 0xC000-0xC7FF, and bits 11 and 12 being ignored).
#[test]
fn ds_swizzle_masked_rotations() {
    // The lane each lane reads.
    let cases: [(u16, [u32; 32]); 4] = [
        // Left by 1, mask 1: 0, 3, 2, 5, 4, 7, 6, 9, 8, ..., 28, 31, 30, 1.
        (
            0xC021,
            [
                0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
                25, 24, 27, 26, 29, 28, 31, 30, 1,
            ],
        ),
        // Right by 1, mask 1: 30, 1, 0, 3, 2, 5, 4, ...
        (
            0xC421,
            [
                30, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20,
                23, 22, 25, 24, 27, 26, 29, 28, 31,
            ],
        ),
        // Left by 1, mask 3: 0, 1, 2, 7, 4, 5, 6, 11, ...
        (
            0xC023,
            [
                0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14, 19, 16, 17, 18, 23, 20, 21, 22,
                27, 24, 25, 26, 31, 28, 29, 30, 3,
            ],
        ),
        // Bits 11 and 12 set: as 0xC021.
        (
            0xD821,
            [
                0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
                25, 24, 27, 26, 29, 28, 31, 30, 1,
            ],
        ),
    ];
    for (offset, lanes) in cases {
        // Lane i holds i + 1.
        let want: Vec<u32> = lanes.iter().map(|j| j + 1).collect();
        for got in swizzle_32(offset) {
            assert_lanes(&got, &want, &format!("offset {offset:#06x}"));
        }
    }
}

// ---- XNW: the 40 most common mnemonics of the title's pack ----
//
// One synthetic shader per mnemonic (inputs in v1..v4 and s0..s3, result in
// v0), run on the host GPU, compared with a software model of the GFX10
// operation. `xnw shader-info --census` gives the list; `s_load_dwordx4` is
// in every shader here (the UAV descriptor), `s_load_dwordx8` (image
// descriptors) is the one not covered.

/// VOPC `op vcc, src0, vsrc1`.
const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// SOPC `op src0, src1`.
const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// SOP1 `op sdst, src0`.
const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

/// SMEM `op s[dst..], s[sbase:sbase+3], byte_off` (soffset null).
const fn smem(op: u8, dst: u8, sbase: u8, byte_off: u32) -> [u32; 2] {
    [0x3D << 26 | (op as u32) << 18 | (dst as u32) << 6 | (sbase as u32 >> 1), 125 << 25 | (byte_off & 0xF_FFFF)]
}

/// MUBUF `buffer_load_dword vdata, off, s[srsrc..], 0 offset:byte_off`.
const fn buffer_load_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [0x38 << 26 | 0x0C << 18 | (byte_off as u32 & 0xFFF), 0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8]
}

const S_WAITCNT_0: u32 = 0xBF8C_0000;
const S_NOP_0: u32 = 0xBF80_0000;
/// Inline constants 0 and 1.
const C0: u16 = 0x80;
const C1: u16 = 0x81;

type Model = fn([u32; 4]) -> u32;

fn fl(x: u32) -> f32 {
    f32::from_bits(x)
}

/// Checks that `body` decodes to `name` at its first non-setup instruction,
/// then runs every input set and compares with `model` (`ulps` of slack for
/// the hardware's approximate rcp/rsq).
fn check_op(name: &str, body: &[u32], first: usize, model: Model, inputs: &[[u32; 4]], ulps: u32) {
    let code: Vec<u8> = body.iter().flat_map(|w| w.to_le_bytes()).chain(S_ENDPGM.to_le_bytes()).collect();
    let decoded = crate::gfx10::decode(&code).expect("decode");
    assert_eq!(decoded[first].name(), Some(name), "encoding of {name}");
    let shader = build_valu_cs(body);
    run_cases(&shader, inputs, |backend, i, inp, got| {
        let want = model(inp);
        let close = got == want || (ulps > 0 && got.abs_diff(want) <= ulps && (got ^ want) >> 31 == 0);
        assert!(close, "{name} {backend} case {i} {inp:08x?}: got {got:#010x}, want {want:#010x}");
    });
}

const F: [[u32; 4]; 6] = [
    [0x3FC0_0000, 0x4020_0000, 0xBF80_0000, 0x0000_0007], // 1.5, 2.5, -1, 7
    [0xC2F6_E979, 0x3DCC_CCCD, 0x4479_C000, 0x1234_5678], // -123.456, 0.1, 999
    [0x3F80_0001, 0x3F80_0001, 0xBF80_0002, 0x0000_00FF],
    [0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFFFF_FFFF], // +0, -0, inf
    [0x4B00_0001, 0xC0A0_0000, 0x3E80_0000, 0x8000_0001],
    [0x7F7F_FFFF, 0x4000_0000, 0xFF7F_FFFF, 0x0000_0100], // max, 2, -max
];
const U: [[u32; 4]; 6] = [
    [7, 3, 100, 0],
    [0xFFFF_FFFF, 1, 2, 31],
    [0x8000_0000, 0x7FFF_FFFF, 0x1234_5678, 4],
    [0, 0, 0, 0],
    [0x00FF_FFFF, 0x00FF_FFFF, 5, 33],
    [0xDEAD_BEEF, 0x0BAD_F00D, 0xCAFE_BABE, 17],
];

fn fb(x: f32) -> u32 {
    x.to_bits()
}

/// D3D9 multiply: a zero or denormal factor gives +0.
fn mul_legacy(a: u32, b: u32) -> u32 {
    if a & 0x7F80_0000 == 0 || b & 0x7F80_0000 == 0 { 0 } else { fb(fl(a) * fl(b)) }
}

#[test]
fn top40_valu_f32() {
    let v = vgpr;
    let cases: &[(&str, Vec<u32>, Model)] = &[
        ("v_mul_legacy_f32", vec![vop2(0x07, 0, v(1), 2)], |i| mul_legacy(i[0], i[1])),
        ("v_mul_f32", vec![vop2(0x08, 0, v(1), 2)], |i| fb(fl(i[0]) * fl(i[1]))),
        ("v_add_f32", vec![vop2(0x03, 0, v(1), 2)], |i| fb(fl(i[0]) + fl(i[1]))),
        ("v_sub_f32", vec![vop2(0x04, 0, v(1), 2)], |i| fb(fl(i[0]) - fl(i[1]))),
        ("v_subrev_f32", vec![vop2(0x05, 0, v(1), 2)], |i| fb(fl(i[1]) - fl(i[0]))),
        ("v_max_f32", vec![vop2(0x10, 0, v(1), 2)], |i| fb(fl(i[0]).max(fl(i[1])))),
        ("v_fma_f32", vop3(0x14B, 0, v(1), v(2), v(3)).to_vec(), |i| fb(fl(i[0]).mul_add(fl(i[1]), fl(i[2])))),
    ];
    for (name, body, model) in cases {
        check_op(name, body, 0, *model, &F, 0);
    }
    // Accumulating forms: v0 = v3 first.
    let acc: &[(&str, u32, Model)] = &[
        ("v_fmac_f32", vop2(0x2B, 0, vgpr(1), 2), |i| fb(fl(i[0]).mul_add(fl(i[1]), fl(i[2])))),
        // Unfused (GFX10.1 opcode kept by the console): product rounded, then the add.
        ("v_mac_f32", vop2(0x1F, 0, vgpr(1), 2), |i| fb(fl(i[0]) * fl(i[1]) + fl(i[2]))),
    ];
    for (name, op, model) in acc {
        check_op(name, &[vop1(0x01, 0, vgpr(3)), *op], 1, *model, &F, 0);
    }
    // v_fma_mix_f32 with every source f32 (op_sel_hi clear): a plain fma.
    let mix = [0xCC20_0000, (vgpr(1) as u32) | (vgpr(2) as u32) << 9 | (vgpr(3) as u32) << 18];
    check_op("v_fma_mix_f32", &mix, 0, |i| fb(fl(i[0]).mul_add(fl(i[1]), fl(i[2]))), &F, 0);
    // Approximate on AMD hardware (1 ULP); inputs kept finite and normal.
    let r: [[u32; 4]; 4] = [[fb(3.0), 0, 0, 0], [fb(-0.1), 0, 0, 0], [fb(1e10), 0, 0, 0], [fb(7.25), 0, 0, 0]];
    check_op("v_rcp_f32", &[vop1(0x2A, 0, vgpr(1))], 0, |i| fb(1.0 / fl(i[0])), &r, 1);
    let r: [[u32; 4]; 4] = [[fb(3.0), 0, 0, 0], [fb(0.1), 0, 0, 0], [fb(1e10), 0, 0, 0], [fb(4.0), 0, 0, 0]];
    check_op("v_rsq_f32", &[vop1(0x2E, 0, vgpr(1))], 0, |i| fb(1.0 / fl(i[0]).sqrt()), &r, 1);
    check_op("v_cvt_f32_u32", &[vop1(0x06, 0, vgpr(1))], 0, |i| fb(i[0] as f32), &U, 0);
    check_op("v_cvt_f32_ubyte0", &[vop1(0x11, 0, vgpr(1))], 0, |i| fb((i[0] & 0xFF) as f32), &U, 0);
}

#[test]
fn top40_valu_int_and_compare() {
    let v = vgpr;
    let cases: &[(&str, Vec<u32>, Model)] = &[
        ("v_mov_b32", vec![vop1(0x01, 0, v(2))], |i| i[1]),
        ("v_and_b32", vec![vop2(0x1B, 0, v(1), 2)], |i| i[0] & i[1]),
        ("v_add_nc_u32", vec![vop2(0x25, 0, v(1), 2)], |i| i[0].wrapping_add(i[1])),
        ("v_lshrrev_b32", vec![vop2(0x16, 0, v(1), 2)], |i| i[1] >> (i[0] & 31)),
        (
            "v_mad_u32_u24",
            vop3(0x143, 0, v(1), v(2), v(3)).to_vec(),
            |i| (i[0] & 0xFF_FFFF).wrapping_mul(i[1] & 0xFF_FFFF).wrapping_add(i[2]),
        ),
        // v_readfirstlane_b32 s20, v2; v_mov_b32 v0, s20
        ("v_readfirstlane_b32", vec![vop1(0x02, 20, v(2)), vop1(0x01, 0, sgpr(20))], |i| i[1]),
    ];
    for (name, body, model) in cases {
        check_op(name, body, 0, *model, &U, 0);
    }
    // Compares into VCC, read back by v_cndmask_b32 v0, 0, v5 (v5 = 1).
    let cmp: &[(&str, u32, Model, &[[u32; 4]])] = &[
        ("v_cmp_eq_u32", vopc(0xC2, vgpr(1), 2), |i| u32::from(i[0] == i[1]), &U),
        ("v_cmp_lt_f32", vopc(0x01, vgpr(1), 2), |i| u32::from(fl(i[0]) < fl(i[1])), &F),
        ("v_cmp_gt_f32", vopc(0x04, vgpr(1), 2), |i| u32::from(fl(i[0]) > fl(i[1])), &F),
    ];
    for (name, op, model, inputs) in cmp {
        let body = [vop1(0x01, 5, C1), *op, vop2(0x01, 0, C0, 5)];
        check_op(name, &body, 1, *model, inputs, 0);
    }
    // v_cndmask_b32 itself, on a compare of s0 with s1.
    let body = [vopc(0xC2, vgpr(1), 2), vop2(0x01, 0, vgpr(3), 4)];
    check_op("v_cndmask_b32", &body, 1, |i| if i[0] == i[1] { i[3] } else { i[2] }, &U, 0);
}

#[test]
fn top40_scalar_and_control() {
    let back = vop1(0x01, 0, sgpr(20)); // v_mov_b32 v0, s20
    let sop: &[(&str, u32, Model)] = &[
        ("s_mov_b32", sop1(0x03, 20, 1), |i| i[1]),
        ("s_and_b32", sop2(0x0E, 20, 0, 1), |i| i[0] & i[1]),
        ("s_or_b32", sop2(0x10, 20, 0, 1), |i| i[0] | i[1]),
        ("s_add_i32", sop2(0x02, 20, 0, 1), |i| i[0].wrapping_add(i[1])),
        ("s_lshl_b32", sop2(0x1E, 20, 0, 3), |i| i[0] << (i[3] & 31)),
        ("s_lshr_b32", sop2(0x20, 20, 0, 3), |i| i[0] >> (i[3] & 31)),
    ];
    for (name, op, model) in sop {
        check_op(name, &[*op, back], 0, *model, &U, 0);
    }
    // s_mov_b64 s[20:21], s[0:1]; v_mov_b32 v0, s21
    check_op("s_mov_b64", &[sop1(0x04, 20, 0), vop1(0x01, 0, sgpr(21))], 0, |i| i[1], &U, 0);
    // s_cmp_eq_u32 s0, s1; s_cselect_b32 s20, s2, s3
    check_op(
        "s_cmp_eq_u32",
        &[sopc(0x06, 0, 1), sop2(0x0A, 20, 2, 3), back],
        0,
        |i| if i[0] == i[1] { i[2] } else { i[3] },
        &U,
        0,
    );
    // No effect on results.
    check_op("s_nop", &[S_NOP_0, vop1(0x01, 0, vgpr(2))], 0, |i| i[1], &U, 0);
    check_op("s_waitcnt", &[S_WAITCNT_0, vop1(0x01, 0, vgpr(2))], 0, |i| i[1], &U, 0);
    // v0 = v1; s_cbranch_execz skips nothing here (EXEC is set): v0 = v2.
    let skip = [vop1(0x01, 0, vgpr(1)), 0xBF88_0001, vop1(0x01, 0, vgpr(2))];
    check_op("s_cbranch_execz", &skip, 1, |i| i[1], &U, 0);
}

#[test]
fn top40_memory() {
    // buffer_store_dword v2 at byte 4, then buffer_load_dword v0 from it.
    let mut body = buffer_store_dword(2, 8, 4).to_vec();
    body.push(S_WAITCNT_0);
    body.extend(buffer_load_dword(0, 8, 4));
    body.push(S_WAITCNT_0);
    check_op("buffer_store_dword", &body, 0, |i| i[1], &U, 0);
    let code: Vec<u8> = body.iter().chain(&[S_ENDPGM]).flat_map(|w| w.to_le_bytes()).collect();
    assert_eq!(crate::gfx10::decode(&code).unwrap()[2].name(), Some("buffer_load_dword"));
    // s_buffer_load_dword / _dwordx4 of the (zeroed) UAV through its
    // descriptor in s[8:11], plus v1: the load's value is 0.
    for (name, op) in [("s_buffer_load_dword", 0x08u8), ("s_buffer_load_dwordx4", 0x0A)] {
        let mut b = smem(op, 12, 8, 16).to_vec();
        b.push(S_WAITCNT_0);
        b.push(vop2(0x25, 0, sgpr(12), 1)); // v_add_nc_u32 v0, s12, v1
        check_op(name, &b, 0, |i| i[0], &U, 0);
    }
}

/// XNW: `--fma exact` keeps the single rounding of `v_fma_f32`; `--fma
/// fast` (`mad`) rounds twice where the host does not fuse `mad` (the RX
/// 5700 XT: 0) and once where it does (WARP 1.0.22: the exact result;
/// D3D leaves `mad` either way). Exact is the same on every host.
#[test]
fn fma_exact_and_fast_differ_by_one_rounding() {
    let input = [[0x3F80_0001, 0x3F80_0001, 0xBF80_0002, 0]];
    for (fma, want) in [(super::Fma::Exact, &[0x2880_0000u32][..]), (super::Fma::Fast, &[0, 0x2880_0000])] {
        let shader = build_valu_cs_with(&vop3(0x14B, 0, vgpr(1), vgpr(2), vgpr(3)), fma);
        run_cases(&shader, &input, |backend, _, _, got| {
            assert!(want.contains(&got), "{fma:?} {backend}: {got:#x}, want one of {want:x?}");
            if fma == super::Fma::Fast {
                eprintln!("{backend}: mad is {}", if got == 0 { "not fused (two roundings)" } else { "fused (one rounding)" });
            }
        });
    }
}

/// XNW: GPU time of `--fma exact` against `--fma fast` on three fma-heavy
/// compute programs in the pack's proportions (`cargo test -p xnw-shader
/// fma_cost -- --ignored --nocapture`). 64-lane groups, wave64, 16,384
/// groups, 20 timed dispatches each.
#[test]
#[ignore = "timing; run by hand"]
fn fma_cost() {
    let acc = [5u8, 6, 7, 8];
    let init: Vec<u32> = (0..4).map(|k| vop1(0x01, acc[k], vgpr(1 + k as u8))).collect();
    let fold = [
        vop2(0x03, 0, vgpr(5), 6),
        vop2(0x03, 0, vgpr(0), 7),
        vop2(0x03, 0, vgpr(0), 8),
    ];
    let n = 512;
    // A: v_fmac_f32 chains only (4 independent accumulators).
    let a: Vec<u32> = (0..n).map(|i| vop2(0x2B, acc[i % 4], vgpr(1 + (i % 3) as u8), 2)).collect();
    // B: the pack's top four float ops, 6 : 4 : 2.4 : 1.3 (mul_legacy,
    // fmac, fma, add), on the accumulators.
    let mut b = Vec::new();
    for i in 0..n {
        let d = acc[i % 4];
        match i % 14 {
            0..=5 => b.push(vop2(0x07, d, vgpr(d), 2)),
            6..=9 => b.push(vop2(0x2B, d, vgpr(1), 3)),
            10 | 11 => b.extend(vop3(0x14B, d, vgpr(d), vgpr(2), vgpr(3))),
            _ => b.push(vop2(0x03, d, vgpr(d), 4)),
        }
    }
    // C: v_fma_mix_f32 (f32 sources) and v_fma_legacy_f32 alternating.
    let mut c = Vec::new();
    for i in 0..n {
        let d = acc[i % 4];
        if i % 2 == 0 {
            c.extend([0xCC20_0000 | u32::from(d), (vgpr(d) as u32) | (vgpr(2) as u32) << 9 | (vgpr(3) as u32) << 18]);
        } else {
            c.extend(vop3(0x140, d, vgpr(d), vgpr(1), vgpr(4)));
        }
    }
    let Some(host) = Host::for_tests() else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    let inputs = [0x3F80_0001u32, 0x3F7F_FFFE, 0x3A83_126F, 0x3727_C5AC];
    for (label, body) in [("A fmac chains", a), ("B pack mix", b), ("C fma_mix + fma_legacy", c)] {
        let mut code = init.clone();
        code.extend(&body);
        code.extend(fold);
        let mut line = format!("{label:<24}");
        let mut times = Vec::new();
        for fma in [super::Fma::Exact, super::Fma::Fast] {
            let shader = build_wide(&code, fma, 64, true);
            let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if u.root == RootType::Constants {
                    ud[u.index as usize] = inputs[u.dword as usize];
                }
            }
            let d = Dispatch {
                roots: Host::roots(&shader, ud, 1, 0),
                views: &[View::RawUav(16)],
                samplers: &[],
                groups: [16384, 1, 1],
            };
            // Warm up, then the best of three timed runs.
            host.dispatch_timed(&root, &pso, &d, 2).expect("dispatch");
            let best = (0..3)
                .map(|_| host.dispatch_timed(&root, &pso, &d, 20).expect("dispatch").1 / 20.0)
                .fold(f64::INFINITY, f64::min);
            times.push(best);
            line += &format!("  {fma:?} {:.3} ms", best * 1e3);
        }
        let ops = 16384.0 * 64.0 * n as f64;
        println!(
            "{line}  exact/fast {:.2}x  ({:.1} / {:.1} G ops/s)",
            times[0] / times[1],
            ops / times[0] / 1e9,
            ops / times[1] / 1e9
        );
    }
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader with root constants
/// in `s0..s3`, a UAV table at `s4`, and a Root SRV descriptor in `s6..s7`.
/// `body` leaves its result in `v0`, which is stored to dword 0 of the UAV.
fn build_root_srv_cs(body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 for VALU address expressions.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1, // 8 user SGPRs: s0..s7
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        fallback: false,
        fma: crate::emit::Fma::Exact,
        gpu_desc: false,
        probe: false,
        loop_counts: None,
        pagetable: false,
        barycentrics: false,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        half_views: lowered.half_views,
        address_map: lowered.address_map,
        descriptor_words: lowered.descriptor_words,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// Runs a Root SRV compute shader on available SM 6.6 devices (hardware GPU and WARP).
fn run_root_srv_cases(
    shader: &HostShader,
    srv_data: &[u32],
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let mut ran = 0;
    // Changed from VIctory: the host device only (Host::for_tests;
    // this machine's WARP has no shader model 6.6).
    for label in ["hardware"] {
        let Some(host) = Host::for_tests() else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let mut roots = Host::roots(shader, ud, 1, 0);
            roots.push((1, Root::View { srv: true, data: srv_data.to_vec() }));
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots,
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    // As the other GPU tests: without XNW_AGILITY there is no SM 6.6 device.
    if ran == 0 {
        eprintln!("SKIPPED {}: no SM 6.6 device (set XNW_AGILITY)", std::thread::current().name().unwrap_or("root srv test"));
    }
}

#[test]
fn root_srv_s_lshl2_add_u32_with_root_constant_dword0() {
    // s_lshl2_add_u32 s12, s0, s6; s_add_u32 s12, s12, s1; s_addc_u32 s13, s7, 0;
    // v_mov_b32 v5, 0; global_load_dword v0, v5, s[12:13]
    let shader = build_root_srv_cs(&[
        sop2(0x2F, 12, 0, 6),
        sop2(0x00, 12, 12, 1),
        sop2(0x04, 13, 7, 0x80),
        vop1(0x001, 5, 0x80),
        global_load_dword(0, 5, 12, 0)[0],
        global_load_dword(0, 5, 12, 0)[1],
    ]);
    let srv_data: Vec<u32> = (0..16).map(|i| 0x1000_0000 + i * 0x0101_0101).collect();
    let cases = [[0, 0, 0, 0], [3, 0, 0, 0], [5, 8, 0, 0], [11, 16, 0, 0]];
    run_root_srv_cases(&shader, &srv_data, &cases, |backend, _, inp, got| {
        let want = srv_data[(inp[0] + inp[1] / 4) as usize];
        assert_eq!(
            got, want,
            "{backend}: s_lshl2_add_u32 index {} + {}B = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn root_srv_two_term_v_add_co_and_v_lshl_add() {
    // v_add_co_u32 v5, vcc, s6, v1
    // v_add_co_ci_u32 v6, vcc, s7, 0, vcc (VOP3B op 0x128)
    // v_lshl_add_u32 v5, v2, 2, v5
    // global_load_dword v0, v[5:6], off
    let add_co = vop3b(0x30F, 5, 106, sgpr(6), vgpr(1), 0);
    let add_ci = vop3b(0x128, 6, 106, sgpr(7), 0x80, sgpr(106));
    let lshl_add = vop3(0x346, 5, vgpr(2), 0x82, vgpr(5));
    let load = global_load_dword(0, 5, 125, 0);
    let shader = build_root_srv_cs(&[
        add_co[0],
        add_co[1],
        add_ci[0],
        add_ci[1],
        lshl_add[0],
        lshl_add[1],
        load[0],
        load[1],
    ]);
    let srv_data: Vec<u32> = (0..24).map(|i| 0xA000_0000 ^ (i * 0x0011_2233)).collect();
    let cases = [[0, 0, 0, 0], [8, 1, 0, 0], [4, 5, 0, 0], [16, 7, 0, 0]];
    run_root_srv_cases(&shader, &srv_data, &cases, |backend, _, inp, got| {
        let idx = (inp[0] / 4 + inp[1]) as usize;
        let want = srv_data[idx];
        assert_eq!(
            got, want,
            "{backend}: two-term v_add_co + v_lshl_add (off={}, idx={}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn root_srv_loop_carried_pointer_induction_phi() {
    // s12, s13 = root_srv + (s0 << 2); s14 = 0; v0 = 0; v5 = 0;
    // loop:
    //   global_load_dword v6, v5, s[12:13]   (2 dwords)
    //   v_add_nc_u32 v0, v0, v6              (1 dword)
    //   s_add_u32 s12, s12, 4                (1 dword)
    //   s_addc_u32 s13, s13, 0               (1 dword)
    //   s_add_u32 s14, s14, 1                (1 dword)
    //   s_cmp_lt_u32 s14, 4                  (1 dword)
    //   s_cbranch_scc1 -8                    (jumps back 8 dwords -> start of loop)
    let load = global_load_dword(6, 5, 12, 0);
    let shader = build_root_srv_cs(&[
        sop2(0x2F, 12, 0, 6),      // s_lshl2_add_u32 s12, s0, s6
        s_mov_b32(13, 7),          // s_mov_b32 s13, s7
        s_mov_b32(14, 0x80),       // s_mov_b32 s14, 0
        vop1(0x001, 0, 0x80),      // v_mov_b32 v0, 0
        vop1(0x001, 5, 0x80),      // v_mov_b32 v5, 0
        load[0],                   // loop start (-7 before branch)
        load[1],                   // (-6)
        vop2(0x25, 0, vgpr(0), 6), // v_add_nc_u32 v0, v0, v6 (-5)
        sop2(0x00, 12, 12, 0x84),  // s_add_u32 s12, s12, 4 (-4)
        sop2(0x04, 13, 13, 0x80),  // s_addc_u32 s13, s13, 0 (-3)
        sop2(0x00, 14, 14, 0x81),  // s_add_u32 s14, s14, 1 (-2)
        sopc(0x0A, 14, 0x84),      // s_cmp_lt_u32 s14, 4 (-1)
        sopp(0x05, -8),            // s_cbranch_scc1 loop
    ]);
    let srv_data: Vec<u32> = (0..16).map(|i| 100 + i * 17).collect();
    let cases = [[0, 0, 0, 0], [2, 0, 0, 0], [5, 0, 0, 0], [11, 0, 0, 0]];
    run_root_srv_cases(&shader, &srv_data, &cases, |backend, _, inp, got| {
        let start = inp[0] as usize;
        let want: u32 = srv_data[start..start + 4]
            .iter()
            .fold(0u32, |a, &x| a.wrapping_add(x));
        assert_eq!(
            got, want,
            "{backend}: loop phi sum starting at {start} = {got}, want {want}",
        );
    });
}

/// `v_pk_fmac_f16` (VOP2): each half of the destination plus the product
/// of the sources' halves. Origin: VIctory valu_tests.rs (0b35b7f); GPL-3.0-or-later.
#[test]
fn packed_fmac_accumulates_each_half() {
    let shader = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)),              // v_mov_b32 v0, v3
        0x3C << 25 | 2 << 9 | vgpr(1) as u32, // v_pk_fmac_f16 v0, v1, v2
    ]);
    // (2.0, 1.0) * (3.0, 4.0) + (1.0, 0.5) = (7.0, 4.5)
    run_cases(
        &shader,
        &[[0x4000_3C00, 0x4200_4400, 0x3C00_3800, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0x4700_4480, "{backend}");
        },
    );
}

// ---- victory-fork (head 5a55384): halves through views of a table. ----
// Their three tests; changed from upstream: our `resource::track` and
// `lower::Parts`, one device (`Host::for_tests`, skipped without
// XNW_TEST_GPU=1), the host-free assertions run everywhere.

/// MUBUF `op vdata, vaddr, s[srsrc:srsrc+3], 0 idxen`.
const fn mubuf_idxen(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | 1 << 13,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// A one-table compute program over `used` (table in root parameter 1,
/// user SGPR 4), lowered with the runtime's defaults.
fn lower_halves(words: &[u32], used: Vec<UsedResource>, threads: u32) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        fallback: false,
        fma: super::Fma::Exact,
        gpu_desc: false,
        probe: false,
        loop_counts: None,
        pagetable: false,
        barycentrics: false,
    })
    .expect("lower");
    HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        half_views: l.half_views,
        address_map: l.address_map,
        descriptor_words: l.descriptor_words,
    }
}

fn table_range(kind: ResourceType, table_offset_dwords: u32, stride: u32) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: table_offset_dwords / 8,
        stride,
        dwords: 4,
        user_data: 4,
        slot: 1,
    }
}

fn validate_all(shaders: &[&HostShader]) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        for s in shaders {
            let mut dxil = s.dxil.clone();
            v.validate(&mut dxil).expect("DXIL validation failed");
        }
    }
}

/// The root arguments: user data, then (XNW) the heap bounds when the
/// program reads them: two view slots.
fn halves_roots(shader: &HostShader, ud: Vec<u32>) -> Vec<(u32, Root)> {
    Host::roots(shader, ud, 2, 0)
}

/// One device test at a time (as `run_cases`).
static HALVES_ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A program that moves a typed buffer's 16-bit channels
/// (`buffer_load_format_d16_xyzw` from the table's first view,
/// `buffer_store_format_d16_xyzw` to its second) names both places for the
/// host (`HostShader::half_views`) and moves every pattern as it is through
/// unsigned views: half floats, the not-a-number ones among them, and
/// plain 16-bit integers (the title keeps both in such buffers; the
/// network that moves its faces reads indices this way). A program that
/// moves no halves names nothing.
#[test]
fn halves_go_through_a_table_as_they_are_and_name_their_views() {
    use super::host::Stride;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_UINT;
    const LOAD_D16_XYZW: u32 = 0x83;
    const STORE_D16_XYZW: u32 = 0x87;
    const THREADS: u32 = 32;
    // The table's second descriptor, 0x20 bytes on.
    let mut second = s_load_dwordx4(12, 4);
    second[1] |= 0x20;
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(&second);
    // v[1:2] = first[thread] (four halves), second[thread] = v[1:2].
    words.extend_from_slice(&mubuf_idxen(LOAD_D16_XYZW, 1, 0, 8));
    words.extend_from_slice(&mubuf_idxen(STORE_D16_XYZW, 1, 0, 12));
    words.push(S_ENDPGM);
    let shader = lower_halves(
        &words,
        vec![table_range(ResourceType::Srv, 0, 4), table_range(ResourceType::Uav, 8, 4)],
        THREADS,
    );
    assert_eq!(shader.half_views, [(1, 0), (1, 8)]);
    validate_all(&[&shader]);
    // A raw store through the same table: no halves, nothing named.
    assert!(build_valu_cs(&[]).half_views.is_empty());

    // Four halves a thread: numbers, both zeros, subnormals, infinities,
    // not-a-numbers of both signs, and integers past the largest number's
    // pattern, as indices are.
    const SPECIAL: [u16; 16] = [
        0x0000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00, 0x7c01, 0x7e00, 0x7fff, 0x8000,
        0x8001, 0xbc00, 0xfc00, 0xfc01, 0xffff,
    ];
    let halves: Vec<u16> = (0..4 * THREADS as usize)
        .map(|i| match i {
            0..16 => SPECIAL[i],
            _ => (i as u16).wrapping_mul(0x0203).wrapping_add(0x7b00),
        })
        .collect();
    let data: Vec<u32> = halves
        .chunks(2)
        .map(|pair| u32::from(pair[0]) | u32::from(pair[1]) << 16)
        .collect();
    let _one = HALVES_ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(host) = Host::for_tests() else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Table, 1) => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: halves_roots(&shader, ud),
                views: &[
                    View::Buffer {
                        data: &data,
                        stride: Stride::Format(DXGI_FORMAT_R16G16B16A16_UINT, 8),
                    },
                    View::HalfUintUav(THREADS),
                ],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch");
    assert_eq!(out[0], data, "the halves stored are the halves loaded");
}

/// The places a program that moves halves names are the descriptors it
/// loads, not the start of the range their used-resource entry describes:
/// with one entry for a range from dword 0, a load at byte 0x20 names
/// dword 8 (a host view there in a 16-bit float format is the one shown
/// as its unsigned twin). A descriptor picked at run time (the load's
/// offset in an SGPR) names nothing and keeps the float path: which slot
/// it reads, and in which format, is known only then.
#[test]
fn half_places_are_the_descriptors_the_program_loads() {
    const LOAD_D16_XYZW: u32 = 0x83;
    const STORE_D16_XYZW: u32 = 0x87;
    // One entry for a range of views from dword 0, one at 16.
    let used = || vec![table_range(ResourceType::Srv, 0, 4), table_range(ResourceType::Uav, 16, 4)];
    // `s_load_dwordx4 s[dst:dst+3], s[4:5], offset` (soffset null), or with
    // the offset in `soffset` instead.
    let at = |dst: u8, offset: u32| {
        let mut w = s_load_dwordx4(dst, 4);
        w[1] |= offset;
        w
    };
    let picked = |dst: u8, soffset: u8| {
        let mut w = s_load_dwordx4(dst, 4);
        w[1] = u32::from(soffset) << 25;
        w
    };
    let mut fixed = vec![s_mov_b32(5, 0x80)];
    fixed.extend_from_slice(&at(8, 0x20));
    fixed.extend_from_slice(&at(12, 0x40));
    fixed.extend_from_slice(&mubuf_idxen(LOAD_D16_XYZW, 1, 0, 8));
    fixed.extend_from_slice(&mubuf_idxen(STORE_D16_XYZW, 1, 0, 12));
    fixed.push(S_ENDPGM);
    assert_eq!(lower_halves(&fixed, used(), 32).half_views, [(1, 8), (1, 16)]);
    // The load's offset in s20, from the first lane's thread index
    // (`v_readfirstlane_b32 s20, v0`): picked at run time. The pick may land
    // on the store's place too, so no place of that table is named: the d16
    // accesses there take the float path, as before half views.
    let mut runtime = vec![s_mov_b32(5, 0x80), vop1(0x02, 20, vgpr(0))];
    runtime.extend_from_slice(&picked(8, 20));
    runtime.extend_from_slice(&at(12, 0x40));
    runtime.extend_from_slice(&mubuf_idxen(LOAD_D16_XYZW, 1, 0, 8));
    runtime.extend_from_slice(&mubuf_idxen(STORE_D16_XYZW, 1, 0, 12));
    runtime.push(S_ENDPGM);
    assert!(lower_halves(&runtime, used(), 32).half_views.is_empty());
}

/// A format load of whole dwords (`buffer_load_format_x`, not d16) whose
/// result the program only reads one half of as a half float
/// (`v_cvt_f32_f16`) is moving 16-bit integers: the head program reads the
/// face network's output so, from an `R16_UINT` view. It names its view for
/// the host and loads the integer: 0x3c00 comes through as 1.0. Used as a
/// whole float instead (`v_add_f32`), the load stays a float load and names
/// nothing; so does a load of halves beside a load of the same view read as
/// a whole float (a view of numbers the program also picks one half float
/// out of, as hundreds of the title's pixel programs do).
#[test]
fn a_format_load_read_as_halves_loads_the_integer() {
    use super::host::Stride;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT;
    const LOAD_FORMAT_X: u32 = 0x00;
    const V_CVT_F32_F16: u16 = 0x0B;
    const V_ADD_F32: u8 = 0x03;
    let lowered = |body: &[u32]| {
        let mut second = s_load_dwordx4(12, 4);
        second[1] |= 0x20;
        let mut words = vec![s_mov_b32(5, 0x80)];
        words.extend_from_slice(&s_load_dwordx4(8, 4));
        words.extend_from_slice(&second);
        // `body` leaves v2, stored raw.
        words.extend_from_slice(body);
        words.extend_from_slice(&buffer_store_dword(2, 12, 0));
        words.push(S_ENDPGM);
        lower_halves(
            &words,
            vec![table_range(ResourceType::Srv, 0, 4), table_range(ResourceType::Uav, 8, 0)],
            1,
        )
    };
    // v1 = first[v0] (one channel).
    let load = mubuf_idxen(LOAD_FORMAT_X, 1, 0, 8);
    let half = vop1(V_CVT_F32_F16, 2, vgpr(1));
    let as_halves = lowered(&[&load[..], &[half]].concat());
    assert_eq!(as_halves.half_views, [(1, 0)]);
    let as_float = lowered(&[&load[..], &[vop2(V_ADD_F32, 2, vgpr(1), 1)]].concat());
    assert!(as_float.half_views.is_empty());
    // v3 = first[v0] too, read as a whole float: v2 = half(v1) + v3.
    let mut beside = load.to_vec();
    beside.extend_from_slice(&mubuf_idxen(LOAD_FORMAT_X, 3, 0, 8));
    beside.push(half);
    beside.push(vop2(V_ADD_F32, 2, vgpr(3), 2));
    assert!(lowered(&beside).half_views.is_empty());
    // Behind a branch, as the head program has it: where the lanes meet
    // again the loaded value goes on through a phi, and a later write of
    // v1 under EXEC reads it only to keep it in the lanes it leaves alone.
    const SAVE_EXEC: u32 = 0xBE90_037E; // s_mov_b32 s16, exec_lo
    const ONLY_LANE_0: u32 = 0x7DA4_0080; // v_cmpx_eq_u32 exec_lo, 0, v0
    const SKIP_3: u32 = 0xBF88_0003; // s_cbranch_execz over the next 3 words
    const RESTORE_EXEC: u32 = 0xBEFE_0310; // s_mov_b32 exec_lo, s16
    let zero = vop1(0x01, 2, 128); // v_mov_b32 v2, 0
    let rewrite = vop1(0x01, 1, 128); // v_mov_b32 v1, 0
    let branched = lowered(
        &[
            &[zero, SAVE_EXEC, ONLY_LANE_0, SKIP_3][..],
            &load,
            &[half, RESTORE_EXEC, SAVE_EXEC, ONLY_LANE_0, rewrite, RESTORE_EXEC],
        ]
        .concat(),
    );
    assert_eq!(branched.half_views, [(1, 0)]);
    // A d16 load names its place where it is the only access there...
    const LOAD_FORMAT_D16_X: u32 = 0x80;
    let d16 = mubuf_idxen(LOAD_FORMAT_D16_X, 1, 0, 8);
    let d16_alone = lowered(&[&d16[..], &[half]].concat());
    assert_eq!(d16_alone.half_views, [(1, 0)]);
    // ...and not beside a whole float read through the same place, which
    // would see the unsigned twin: v2 = half(d16 v1) + v3.
    let mut d16_beside = d16.to_vec();
    d16_beside.extend_from_slice(&mubuf_idxen(LOAD_FORMAT_X, 3, 0, 8));
    d16_beside.push(half);
    d16_beside.push(vop2(V_ADD_F32, 2, vgpr(3), 2));
    assert!(lowered(&d16_beside).half_views.is_empty());
    validate_all(&[&as_halves, &branched, &d16_alone]);
    // Element 0 of an R16_UINT buffer holds the half 1.0 (0x3c00).
    let data = [0x0000_3c00u32, 0];
    let _one = HALVES_ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(host) = Host::for_tests() else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    for (name, shader) in [("straight", &as_halves), ("branched", &branched)] {
        let (root, pso) = host.compute_pipeline(shader).expect("compute_pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Table, 1) => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: halves_roots(shader, ud),
                    views: &[
                        View::Buffer { data: &data, stride: Stride::Format(DXGI_FORMAT_R16_UINT, 2) },
                        View::RawUav(1),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        assert_eq!(out[0][0], 1.0f32.to_bits(), "{name}: the half read as an integer's bits");
    }
}

// ---- victory-fork 3bff182 (VIctory-7900XTX 342245e, b17b69f): 64-bit floats. ----

/// Verbatim from victory-fork.
#[test]
fn v_cvt_f32_f64_matches_ieee754() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x00f -> VOP3 opcode 0x18f).
    let plain = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);
    // Same with abs + neg on src0 (VOP3: abs[0] = bit 8 of w0, neg[0] = bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x18f, 0, vgpr(1), 0, 0);
    w0 |= 1 << 8;
    w1 |= 1 << 29;
    let neg_abs = build_valu_cs(&[w0, w1]);

    let values: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.5,
        -123456.75,
        // Tie-to-even: even mantissa stays, odd mantissa rounds up.
        f64::from_bits(0x3FF0_0000_1000_0000), // 1.0 + 0.5 ulp(f32) -> 1.0
        f64::from_bits(0x3FF0_0000_3000_0000), // (1.0 + 1 ulp) + 0.5 ulp -> 1.0 + 2 ulp
        f64::from_bits(0x3FF0_0000_1000_0001), // > 0.5 ulp -> rounds up
        // Largest finite f32 and overflow to +Inf / -Inf.
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_001,
        -f64::from(f32::MAX) * 1.000_001,
        1e300,
        // Subnormal range (exp64 in [874, 896]), including carry into smallest normal.
        f64::from(f32::MIN_POSITIVE),
        f64::from(f32::MIN_POSITIVE) * 0.5,
        f64::from(f32::MIN_POSITIVE) * 0.75,
        f64::from(f32::from_bits(0x007F_FFFF)),
        f64::from_bits(0x380F_FFFF_F000_0000), // max subnormal + 0.5 ulp -> min normal
        // exp64 == 873: exact 2^-150 ties to 0; strictly greater rounds to min subnormal (1).
        f64::from_bits(0x3690_0000_0000_0000),
        f64::from_bits(0x3690_0000_0000_0001),
        f64::from_bits(0xB690_0000_0000_0001),
        // Deep underflow -> +/-0.0.
        1e-200,
        -1e-200,
        // Infinities and NaNs.
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7FF8_0000_0000_0000), // +qNaN
        f64::from_bits(0x7FF0_0000_0000_0001), // +sNaN -> quieted
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .map(|&x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&plain, &inputs, |backend, i, _, got| {
        let x = values[i];
        let want = (x as f32).to_bits();
        if x.is_nan() {
            assert!(
                f32::from_bits(got).is_nan() && (got & 0x0040_0000) != 0,
                "{backend}: v_cvt_f32_f64({:#018x}) = {got:#010x}, expected qNaN",
                x.to_bits(),
            );
        } else {
            assert_eq!(
                got, want,
                "{backend}: v_cvt_f32_f64({x} / {:#018x}) = {got:#010x}, want {want:#010x}",
                x.to_bits(),
            );
        }
    });

    // -|x| via VOP3 abs + neg modifiers.
    let mod_values: &[f64] = &[2.5, -2.5, 0.0, f64::INFINITY];
    let mod_in: Vec<[u32; 4]> = mod_values
        .iter()
        .map(|&x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&neg_abs, &mod_in, |backend, i, _, got| {
        let x = mod_values[i];
        let want = (-(x.abs()) as f32).to_bits();
        assert_eq!(
            got, want,
            "{backend}: v_cvt_f32_f64(-|{x}|) = {got:#010x}, want {want:#010x}",
        );
    });
}

/// victory-fork's test; changed: our `lower_cs` (its layout and UAV) and
/// `run_cases` (one device; the inputs in the constants), threads 4.
#[test]
fn whole_wave_f64_sum_and_cvt_f32_f64() {
    let s_or_saveexec_b32 =
        |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x3D << 8 | src0 as u32;
    let v_mov_dpp8 = |vdst: u8, vsrc: u8, sel: u32| [vop1(0x001, vdst, 0xE9), vsrc as u32 | (sel << 8)];
    let v_mov_row_mirror = |vdst: u8, vsrc: u8| {
        [
            vop1(0x001, vdst, 0xFA),
            vsrc as u32 | (0x140 << 8) | (1 << 19) | (0xF << 24) | (0xF << 28),
        ]
    };
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // f64 input per lane in v[6:7] = (s0, s1).
    words.push(vop1(0x001, 6, sgpr(0)));
    words.push(vop1(0x001, 7, sgpr(1)));
    // Whole-wave f64 sum over Wave32 with s_mov_b32 copy before EXEC restore.
    words.push(s_or_saveexec_b32(12, 0xC1));
    words.extend_from_slice(&vop3(0x101, 2, 0x80, vgpr(6), sgpr(12)));
    words.extend_from_slice(&vop3(0x101, 3, 0x80, vgpr(7), sgpr(12)));
    for sel in [14570689u32, 9658386, 2340] {
        words.extend_from_slice(&v_mov_dpp8(4, 2, sel));
        words.extend_from_slice(&v_mov_dpp8(5, 3, sel));
        words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0)); // v_add_f64
    }
    words.extend_from_slice(&v_mov_row_mirror(4, 2));
    words.extend_from_slice(&v_mov_row_mirror(5, 3));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x378, 4, vgpr(2), 0x80, 0x80)); // v_permlanex16_b32
    words.extend_from_slice(&vop3(0x378, 5, vgpr(3), 0x80, 0x80));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x360, 14, vgpr(2), 0x80, 0)); // v_readlane_b32 s14, v2, 0
    words.extend_from_slice(&vop3(0x360, 15, vgpr(3), 0x80, 0)); // v_readlane_b32 s15, v3, 0
    words.push(s_mov_b32(16, 14));
    words.push(s_mov_b32(17, 15));
    words.push(s_mov_b32(126, 12)); // restore exec_lo
    words.push(vop1(0x00f, 0, sgpr(16))); // v_cvt_f32_f64 v0, s[16:17]
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);
    let shader = build_cs(&words, 4);
    let cases: &[(f64, f32)] = &[(1.25, 5.0), (-0.75, -3.0)];
    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&shader, &inputs, |backend, i, _, got| {
        let (x, want) = cases[i];
        assert_eq!(got, want.to_bits(), "{backend}: wave f64 sum({x}) = {got:#010x}, want {:#010x}", want.to_bits());
    });
}
