//! XNW: the non-uniform heap index fault, end to end on the software adapter. A
//! synthetic GFX10 compute program in CS_Prop_BatchCull's shape (0451d340
//! 0x1070-0x114c): lanes other than lane 0 read a heap slot number with a
//! scalar buffer load, load that slot's descriptor from the view heap and
//! take the texture's width. Lowered with the masked rule
//! ([`super::Options::masked_lanes`], the software adapter's form) the
//! heap index is declared non-uniform and every running lane reads the
//! slot it named; lowered without it (the hardware adapter's form, and the
//! translator before) WARP 1.0.22 takes the index from lane 0, which
//! does not run. `XNW_TEST_GPU=1 XNW_ADAPTER=warp XNW_WARP=<dir>
//! XNW_AGILITY=<dir> cargo test -p xnw-shader heap_uniform -- --nocapture`.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
use crate::{io, ir, resource, structure};

/// Lanes 1..63 write the width of the texture at the slot the slot list
/// names (dword 4 * lane of the output); lane 0 writes nothing.
const WORDS: [u32; 22] = [
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0: the slot list (raw SRV)
    0xF408_0302,
    0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 32: the output (raw UAV)
    0x7DAA_0080, // v_cmpx_ne_u32 exec, 0, v0: every lane but lane 0
    0xBF88_000E, // s_cbranch_execz end
    0xF420_0404,
    0xFA00_0000, // s_buffer_load_dword s16, s[8:11], 0: the slot
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x8F11_8510, // s_lshl_b32 s17, s16, 5
    0xF40C_0503,
    0x2200_0000, // s_load_dwordx8 s[20:27], s[6:7], s17: the view heap at the slot
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_0280, // v_mov_b32 v1, 0
    0xF038_1308,
    0x0005_0201, // image_get_resinfo v[2:3], v1, s[20:27] dmask:0x3 dim:2D unorm
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x3408_0082, // v_lshlrev_b32 v4, 2, v0
    0xE070_1000,
    0x8003_0204, // buffer_store_dword v2, v4, s[12:15], 0 offen
];
const S_ENDPGM: u32 = 0xBF81_0000;

/// The program lowered as with [`super::Options::masked_lanes`] `masked`.
fn build(masked: bool) -> HostShader {
    let mut words = WORDS.to_vec();
    words.push(S_ENDPGM);
    let param = |root, user_data, slot| RootParam { root, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data, slot };
    let used = |kind, table_offset_dwords| UsedResource { kind, root: RootType::Table, table_offset_dwords, lower_bound: 0, stride: 0, dwords: 4, user_data: 4, slot: 1 };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 8,
        layout: vec![param(RootType::Table, 4, 1), param(RootType::ResourceHeap, 6, 0)],
        used: vec![used(ResourceType::Srv, 0), used(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo { rsrc2: 7 << 1, threads: [64, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    super::HEAP_UNIFORMITY.with(|f| f.set((false, masked)));
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
        fma: super::Fma::Exact,
        fallback: false,
        gpu_desc: false,
        probe: false,
        loop_counts: None,
        pagetable: false,
        barycentrics: false,
    });
    super::HEAP_UNIFORMITY.with(|f| f.set((false, false)));
    let lowered = lowered.expect("lower");
    let (marked, varying, handles) = super::last_nonuniform();
    println!("masked rule {masked}: {marked} of {handles} heap handles non-uniform ({varying} varying)");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        address_map: lowered.address_map,
        descriptor_words: lowered.descriptor_words,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    shader
}

/// The widths the lanes wrote; `None` without a device.
fn run(shader: &HostShader) -> Option<Vec<u32>> {
    let host = Host::for_tests()?;
    let (root, pso) = host.compute_pipeline(shader).expect("compute_pipeline");
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            // The table starts at heap slot 1 (32 bytes a view).
            RootType::Table => 32,
            RootType::ResourceHeap => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    let blank = [[0.0f32; 4]; 16 * 8];
    // Slot 0: a 4x4 texture (what lane 0's unread index names); 1: the slot
    // list (slot 3); 2: the output; 3: the 16x8 texture the lanes name.
    let views = [View::Texture(4, 4, &blank[..16]), View::Buffer { data: &[3], stride: Stride::Raw }, View::RawUav(64), View::Texture(16, 8, &blank)];
    // Heap bounds (`Shader::bounded`): 4 view slots, every null at slot 0.
    let bounds = vec![4, 0, 0, 0, 0, 0, 1, 0];
    let out = host
        .dispatch(&root, &pso, &Dispatch { roots: vec![(0, Root::Constants(ud)), (1, Root::Constants(bounds))], views: &views, samplers: &[], groups: [1, 1, 1] })
        .expect("dispatch");
    Some(out[0].clone())
}

#[test]
fn heap_uniform_a_slot_read_by_some_lanes_reaches_each_of_them() {
    // The software adapter's rule: never run on the card.
    if !xnw_device::warp() {
        eprintln!("XNW_ADAPTER is not warp; skipped");
        return;
    }
    let fixed = build(true);
    let Some(widths) = run(&fixed) else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    println!("masked rule on: lanes 0..4 {:?}, distinct {:?}", &widths[..4], distinct(&widths[1..]));
    assert_eq!(widths[0], 0, "lane 0 does not run");
    assert!(widths[1..].iter().all(|&w| w == 16), "every running lane reads slot 3's width: {widths:?}");
    // Without the rule, on the software adapter: the index of lane 0.
    let before = run(&build(false)).unwrap_or_default();
    println!("masked rule off: lanes 0..4 {:?}, distinct {:?}", &before[..4.min(before.len())], distinct(before.get(1..).unwrap_or(&[])));
}

fn distinct(v: &[u32]) -> Vec<u32> {
    let mut d = v.to_vec();
    d.sort_unstable();
    d.dedup();
    d
}
