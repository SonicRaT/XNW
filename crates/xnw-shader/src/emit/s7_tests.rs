//! XNW: one test per class of compute program that failed on address,
//! root-view and LDS classes in world runs, each from the shape of
//! its smallest failing program (hand-encoded; no title code). Each must
//! lower and pass DXIL validation (when `dxil.dll` loads).

use super::validator::Validator;
use crate::native::{ComputeIo, StageIo};
use crate::{HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource};

fn param(root: RootType, sgprs: u32, user_data: u8, slot: u8) -> RootParam {
    RootParam { root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot }
}

fn used(kind: ResourceType, root: RootType, stride: u32, user_data: u8, slot: u8) -> UsedResource {
    UsedResource { kind, root, table_offset_dwords: 0, lower_bound: 0, stride, dwords: 4, user_data, slot }
}

/// Lowers `words` as a wave32 compute program and validates the DXIL.
fn lower(words: &[u32], layout: Vec<RootParam>, used: Vec<UsedResource>, rsrc2: u32, spilled: bool) -> super::lower::Lowered {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo { rsrc2, threads: [32, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build_with(&native.code, native.wave64, native.vgprs, spilled).unwrap();
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
    })
    .unwrap();
    if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
        v.validate(&mut l.dxil.clone()).unwrap();
    }
    l
}

fn names(words: &[u32]) -> Vec<String> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    crate::gfx10::decode(&code).unwrap().iter().map(|i| i.name().unwrap_or("?").to_owned()).collect()
}

/// Class "a root UAV or typed root view" (BVH builder fills, e.g.
/// a6a068d5): the root arguments sit in the spill table; a 16-dword load
/// reuses s8..s23, then the UAV's address (user SGPRs 18, 19) is loaded and
/// a raw descriptor built around it. The copy must read user SGPR 18 at
/// entry (not s18 the first load overwrote), and the store goes to a root UAV.
#[test]
fn root_uav_from_the_spill_table() {
    let words = [
        0x8703_FF01, 0x0000_00FF, // s_and_b32 s3, s1, 0xff
        0xBE82_0300, // s_mov_b32 s2, s0
        0xF410_0201, 0xFA00_0010, // s_load_dwordx16 s[8:23], s[2:3], 0x10
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF404_0001, 0xFA00_0008, // s_load_dwordx2 s[0:1], s[2:3], 0x8
        0x3408_0084, // v_lshlrev_b32 v4, 4, v0
        0x7E00_020A, // v_mov_b32 v0, s10
        0xBE82_03C1, // s_mov_b32 s2, -1
        0xBE83_03FF, 0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
        0xBF8C_C07F,
        0xBE81_1D90, // s_bitset1_b32 s1, 16
        0xE070_1000, 0x8000_0004, // buffer_store_dword v0, v4, s[0:3], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    assert!(names(&words).contains(&"buffer_store_dword".to_owned()));
    let layout = vec![
        param(RootType::Descriptor, 2, 16, 0),
        param(RootType::Descriptor, 2, 18, 1),
        param(RootType::Constants, 16, 20, 2),
    ];
    let used = vec![used(ResourceType::Uav, RootType::Descriptor, 1, 18, 1)];
    let l = lower(&words, layout, used, 2 << 1, true);
    assert_eq!(l.root_uavs, vec![1]);
    // v_mov_b32 v0, s10: constant dword 2 (user data 22).
    assert!(l.user_data.iter().any(|u| u.index == 22 && u.slot == 2 && u.dword == 2), "{:?}", l.user_data);
}

/// Class "an address from a root parameter that is no address" (BVH
/// builder, e.g. 165f1029): a pointer in root constants, loaded through
/// and stored to: title memory through the address map.
#[test]
fn pointer_in_root_constants_reads_through_the_address_map() {
    let words = [
        0xF400_0100, 0xFA00_0000, // s_load_dword s4, s[0:1], 0x0
        0x7E04_0200, // v_mov_b32 v2, s0
        0x7E06_0201, // v_mov_b32 v3, s1
        0xDC30_8000, 0x017D_0002, // global_load_dword v1, v[2:3], off
        0xBF8C_C07F,
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x4A02_0204, // v_add_nc_u32 v1, s4, v1
        0xDC70_8000, 0x007D_0102, // global_store_dword v[2:3], v1, off
        0xBF81_0000,
    ];
    let n = names(&words);
    assert!(n.contains(&"global_load_dword".to_owned()) && n.contains(&"global_store_dword".to_owned()), "{n:?}");
    let layout = vec![param(RootType::Constants, 2, 0, 0)];
    let used = vec![used(ResourceType::Cbv, RootType::Constants, 0, 0, 0)];
    let l = lower(&words, layout, used, 2 << 1, false);
    assert!(l.address_map);
}

/// Class "an address from a phi" (scattering_transmittance_system
/// ReconstructRaymarchClassified): one of two code tables picked by a
/// branch, then a scalar load: `s_getpc_b64` values with the code tag read
/// the program's own code.
#[test]
fn phi_of_code_tables_reads_the_code() {
    let words = [
        0xBE82_1F00, // s_getpc_b64 s[2:3]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 +1
        0x8002_9002, // s_add_u32 s2, s2, 16
        0xF400_0101, 0xFA00_0000, // s_load_dword s4, s[2:3], 0x0
        0x7E04_0200, // v_mov_b32 v2, s0
        0x7E06_0201, // v_mov_b32 v3, s1
        0xBF8C_C07F,
        0x7E02_0204, // v_mov_b32 v1, s4
        0xDC70_8000, 0x007D_0102, // global_store_dword v[2:3], v1, off
        0xBF81_0000,
        0x1234_5678, 0x9ABC_DEF0, 0x0BAD_F00D, 0x0000_0001, 0x0000_0002, // the table
    ];
    let layout = vec![param(RootType::Constants, 2, 0, 0)];
    let used = vec![used(ResourceType::Cbv, RootType::Constants, 0, 0, 0)];
    let l = lower(&words[..14], layout, used, 2 << 1, false);
    assert!(l.address_map);
}

/// Classes "an LDS write of bytes or halves" (TAA2 PrepareFrameStats,
/// shadowmask DenoiseShadowMaskLDSPacked) and "an LDS operation"
/// (`ds_bpermute_b32`, shadowmask GenerateShadowMasks).
#[test]
fn lds_bytes_halves_and_backward_permute() {
    let words = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xD878_0000, 0x0000_0001, // ds_write_b8 v1, v0
        0xD87C_0002, 0x0000_0001, // ds_write_b16 v1, v0 offset:2
        0xD8E8_0000, 0x0200_0001, // ds_read_u8 v2, v1
        0xD8EC_0002, 0x0300_0001, // ds_read_i16 v3, v1 offset:2
        0xDACC_0000, 0x0400_0201, // ds_bpermute_b32 v4, v1, v2
        0xBF81_0000,
    ];
    let n = names(&words);
    for want in ["ds_write_b8", "ds_write_b16", "ds_read_u8", "ds_read_i16", "ds_bpermute_b32"] {
        assert!(n.iter().any(|x| x == want), "{want} in {n:?}");
    }
    // 512 bytes of LDS.
    lower(&words, Vec::new(), Vec::new(), 1 << 15, false);
}
