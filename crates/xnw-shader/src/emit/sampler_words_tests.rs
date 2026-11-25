//! XNW: programs that hold a sampler descriptor's dwords as data. The
//! character and skin programs take their samplers from the directly
//! indexed sampler heap (root type `SamplerHeap`, 16 bytes an entry) and
//! then do one of two things with the four dwords:
//!
//! - mesh programs copy them per lane into VGPRs on both paths of a
//!   divergent branch and walk the distinct values with a waterfall loop
//!   before each sample (`v_readfirstlane_b32`, `v_cmp_eq_u64`);
//! - pixel programs reuse the registers for data on one path, so a join
//!   takes a sampler dword on one edge and data on the other, and only
//!   code reached by the data edge reads it.
//!
//! One test per shape (hand-encoded; no title code). Each must lower and
//! pass DXIL validation (when `dxil.dll` loads).

use super::validator::Validator;
use crate::native::{ComputeIo, StageIo};
use crate::{HwStage, NativeProgram, ResourceType, RootParam, RootType, ShaderError, UsedResource};

fn param(root: RootType, sgprs: u32, user_data: u8, slot: u8) -> RootParam {
    RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot }
}

fn used(kind: ResourceType, dwords: u8, user_data: u8, slot: u8) -> UsedResource {
    UsedResource { kind, root: RootType::Table, table_offset_dwords: 0, lower_bound: 0, stride: 0, dwords, user_data, slot }
}

fn lower(
    words: &[u32],
    stage: HwStage,
    io: StageIo,
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<super::lower::Lowered, ShaderError> {
    let native = NativeProgram {
        stage,
        wave64: true,
        vgprs: 32,
        layout,
        used,
        io,
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
    let io = crate::io::raise(&p, &native).unwrap();
    let st = crate::structure::structure(&p).unwrap();
    let l = super::lower::lower(&super::lower::Parts {
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
    })?;
    if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
        v.validate(&mut l.dxil.clone()).unwrap();
    }
    Ok(l)
}

/// The sampler heap in s8 and a texture table in s9 (10 user SGPRs, 512
/// bytes of LDS).
fn heap_and_table() -> (StageIo, Vec<RootParam>, Vec<UsedResource>) {
    (
        StageIo::Cs(ComputeIo { rsrc2: 10 << 1 | 1 << 15, threads: [64, 1, 1] }),
        vec![param(RootType::SamplerHeap, 1, 8, 0), param(RootType::Table, 1, 9, 1)],
        vec![used(ResourceType::Srv, 8, 9, 1)],
    )
}

/// An if/else that copies sampler heap entry 65 or 67 into v[20:23], then
/// a waterfall loop sampling with them. `extra` goes after the first load
/// and its `s_waitcnt` (seven words to the else-part without it).
fn waterfall(extra: &[u32]) -> Vec<u32> {
    let skip = 7 + extra.len() as u32;
    let mut w = vec![
        0xBE82_0309, // s_mov_b32 s2, s9
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE83_0380, // s_mov_b32 s3, 0
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
        0xBF88_0000 | skip, // s_cbranch_execz L1
        0xF408_0304,
        0xFA00_0410, // s_load_dwordx4 s[12:15], s[8:9], 0x410
        0xBF8C_0000, // s_waitcnt 0
    ];
    w.extend(extra);
    w.extend([
        0x7E28_020C, // v_mov_b32 v20, s12
        0x7E2A_020D, // v_mov_b32 v21, s13
        0x7E2C_020E, // v_mov_b32 v22, s14
        0x7E2E_020F, // v_mov_b32 v23, s15
        0x8AFE_7E06, // L1: s_andn2_b64 exec, s[6:7], exec
        0xBF88_0007, // s_cbranch_execz L2
        0xF408_0304,
        0xFA00_0430, // s_load_dwordx4 s[12:15], s[8:9], 0x430
        0xBF8C_0000, // s_waitcnt 0
        0x7E28_020C, // v_mov_b32 v20, s12
        0x7E2A_020D, // v_mov_b32 v21, s13
        0x7E2C_020E, // v_mov_b32 v22, s14
        0x7E2E_020F, // v_mov_b32 v23, s15
        0x88FE_067E, // L2: s_or_b64 exec, exec, s[6:7]
        0xF40C_0401,
        0xFA00_0000, // s_load_dwordx8 s[16:23], s[2:3], null
        0xBF8C_0000, // s_waitcnt 0
        0xBE98_047E, // s_mov_b64 s[24:25], exec
        0x7E38_0514, // L3: v_readfirstlane_b32 s28, v20
        0x7E3A_0515, // v_readfirstlane_b32 s29, v21
        0x7E3C_0516, // v_readfirstlane_b32 s30, v22
        0x7E3E_0517, // v_readfirstlane_b32 s31, v23
        0x7DC4_281C, // v_cmp_eq_u64 vcc, s[28:29], v[20:21]
        0xD4E2_001A,
        0x0002_2C1E, // v_cmp_eq_u64 s[26:27], s[30:31], v[22:23]
        0x879A_1A6A, // s_and_b64 s[26:27], vcc, s[26:27]
        0xBE9A_241A, // s_and_saveexec_b64 s[26:27], s[26:27]
        0xF09C_0F08,
        0x00E4_0004, // image_sample_lz v[0:3], v[4:5], s[16:23], s[28:31]
        0x89FE_1A7E, // s_xor_b64 exec, exec, s[26:27]
        0xBF89_FFF3, // s_cbranch_execnz L3
        0xBEFE_0418, // s_mov_b64 exec, s[24:25]
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn text(words: &[u32]) -> Vec<String> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    crate::gfx10::decode(&code).unwrap().iter().map(|i| i.text(true)).collect()
}

#[test]
fn the_waterfall_decodes_as_written() {
    let t = text(&waterfall(&[]));
    let has = |s: &str| t.iter().any(|l| l.starts_with(s));
    assert!(has("s_load_dwordx4 s[12:15], s[8:9], 0x000430"), "{t:#?}");
    assert!(has("v_cmp_eq_u64 s[26:27], s[30:31], v[22:23]"), "{t:#?}");
    assert!(has("image_sample_lz v[0:3], v[4:5], s[16:23], s[28:31]"), "{t:#?}");
    assert!(has("s_cbranch_execnz label_006C"), "{t:#?}");
}

/// The mesh programs' shape (MSNC_VS_TransformSkinD at 0x202c-0x20a0,
/// 0x2794-0x27c4 and the loop at 0x2a10-0x2a54): the dwords of a sampler
/// heap entry are only copied, read from the first lane and compared for
/// equality, so each stands as its own byte offset in the sampler heap,
/// and every lane samples with the entry its own path loaded.
#[test]
fn sampler_heap_words_walked_by_a_waterfall_loop_lower() {
    let (io, layout, used) = heap_and_table();
    let l = lower(&waterfall(&[]), HwStage::Cs, io, layout, used).unwrap();
    for op in ["dx.op.waveReadLaneFirst.i32", "dx.op.createHandleFromHeap", "dx.op.sampleLevel.f32"] {
        assert!(l.op_names.iter().any(|n| n == op), "{op} in {:?}", l.op_names);
    }
    // The host binds nothing new: no descriptor words, no user data for
    // the heap (its base reads 0).
    assert!(!l.descriptor_words);
    assert!(l.user_data.iter().all(|u| u.root == RootType::Table), "{:?}", l.user_data);
}

/// A sampler dword the program also computes with is no stand-in: the
/// refusal stays (here the sum is stored).
#[test]
fn a_sampler_word_computed_with_stays_refused() {
    let (io, layout, used) = heap_and_table();
    let words = waterfall(&[
        0x8101_810C, // s_add_i32 s1, s12, 1
        0x7E0A_0201, // v_mov_b32 v5, s1
        0xD834_0000,
        0x0000_0504, // ds_write_b32 v4, v5
    ]);
    let t = text(&words);
    assert!(t.iter().any(|l| l.starts_with("s_add_i32 s1, s12, 1")), "{t:#?}");
    assert!(t.iter().any(|l| l.starts_with("ds_write_b32 v4, v5")), "{t:#?}");
    let Err(e) = lower(&words, HwStage::Cs, io, layout, used) else {
        panic!("lowered");
    };
    assert!(e.to_string().contains("a sampler descriptor read as data"), "{e}");
}

/// The pixel programs' shape (PS_DeferredPedTexturedNC: the load at 0x204,
/// the skip at 0x6f4, the data at 0x744, the join at 0x750 and the read at
/// 0xa00, which the skip's path does not reach): a register holds a
/// sampler heap entry's dword on one edge of a join and constant data on
/// the other; only lanes that came by the data edge read it.
#[test]
fn a_sampler_heap_dword_on_a_dead_join_edge_is_dropped() {
    let words = [
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0403,
        0xFA00_0520, // s_load_dwordx4 s[16:19], s[6:7], 0x520
        0xF420_0304,
        0xFA00_0000, // s_buffer_load_dword s12, s[8:11], 0x0
        0xBF06_800C, // s_cmp_eq_u32 s12, 0
        0xBF85_0002, // s_cbranch_scc1 -> the join (s17: sampler dword)
        0xF420_0444,
        0xFA00_0004, // s_buffer_load_dword s17, s[8:11], 0x4
        0xBF06_800C, // s_cmp_eq_u32 s12, 0
        0xBF85_0006, // s_cbranch_scc1 -> s_endpgm
        0xBF0A_FF11,
        0x0100_0000, // s_cmp_lt_u32 s17, 0x01000000
        0xBF85_0003, // s_cbranch_scc1 -> s_endpgm
        0x7E00_0211, // v_mov_b32 v0, s17
        0xF800_180F,
        0x0000_0000, // exp mrt0, v0, v0, v0, v0 done vm
        0xBF81_0000, // s_endpgm
    ];
    let t = text(&words);
    assert!(t.iter().any(|l| l.starts_with("s_load_dwordx4 s[16:19], s[6:7], 0x000520")), "{t:#?}");
    assert!(t.iter().any(|l| l.starts_with("s_cmp_lt_u32 s17, ")), "{t:#?}");
    let io = StageIo::Ps { inputs: vec![], input_addr: 0, db_shader_control: 0, patches: vec![] };
    let layout = vec![
        param(RootType::Constants, 4, 0, 0),
        param(RootType::Table, 1, 4, 1),
        param(RootType::SamplerHeap, 1, 6, 0),
    ];
    let used = vec![used(ResourceType::Cbv, 4, 4, 1)];
    lower(&words, HwStage::Ps, io, layout, used).unwrap();
}
