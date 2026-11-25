// Ported from victory-fork crates/victory-shader/src/emit/mod.rs (`mod barycentric_tests`), commit
// 578d459c344aa44753e9a61ea6ba10a8c68f6ea6, GPL-3.0-or-later: the computed-barycentric tests,
// translated with `Options::barycentrics` on. Not ported: upstream's tests of barycentrics read as values
// without a computed interpolation (ours read 0 there: the RX 5700 XT has no SV_Barycentrics). Added
// (XNW): the same programs on a host without barycentrics.
use super::container::SigElement;
use super::*;
use crate::native::{PsInput, StageIo, Usage};

/// `SPI_PS_INPUT_ADDR` bit PERSP_CENTER.
const PERSP_CENTER: u32 = 1 << 1;
const S_ENDPGM: u32 = 0xBF81_0000;

/// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
fn exp_mrt0(r: u32) -> [u32; 2] {
    [0xF800_180F, r * 0x0101_0101]
}

/// VOP2 `v_add_f32 v<dst>, v<a>, v<b>`.
fn v_add_f32(dst: u32, a: u32, b: u32) -> u32 {
    3 << 25 | dst << 17 | b << 9 | (256 + a)
}

fn element(name: &str, kind: u8, row: u32) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    }
}

/// Translates a pixel program with the input VGPRs `input_addr` enables
/// and one float4 TEXCOORD0 attribute (attr 0, row 0); `barycentrics`:
/// whether the host has them.
fn pixel_on(words: &[u32], input_addr: u32, barycentrics: bool) -> Result<HostShader, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 0b1111,
                attr: 0,
                flat: false,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            input_addr,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let mut target = element("SV_Target", 16, 0);
    target.usage = 15;
    let isg1 = container::signature(&[element("TEXCOORD", 0, 0)], true);
    let osg1 = container::signature(&[target], false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let res = resource::track(&p, &native.layout, native.stage)?;
    let io = io::raise(&p, &native)?;
    let st = structure::structure(&p)?;
    let opts = Options { barycentrics, ..Default::default() };
    emit_with(&c, &native, &p, &st, &io, &res, opts)
}

fn pixel(words: &[u32], input_addr: u32) -> Result<HostShader, ShaderError> {
    pixel_on(words, input_addr, true)
}

/// The translation's `SFI0` features and `ISG1` rows.
fn parts(shader: &HostShader) -> (u64, Vec<SigElement>) {
    let c = Container::parse(&shader.dxil).unwrap();
    let sfi0 = c.part(FourCC::SFI0).unwrap().data.try_into().unwrap();
    (u64::from_le_bytes(sfi0), signature(c.part(FourCC::ISG1).unwrap().data).unwrap())
}

fn validate(shader: &HostShader) {
    #[cfg(windows)]
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).unwrap();
    }
    let _ = shader;
}

/// The `PSV0` interpolation mode of TEXCOORD0 (row 0, x-w, float).
fn texcoord_mode(shader: &HostShader) -> u8 {
    let c = Container::parse(&shader.dxil).unwrap();
    let psv = c.part(FourCC::PSV0).unwrap().data;
    let row = [1, 0, 0x44, 0, 3];
    let at = psv.windows(5).position(|w| w == row).unwrap();
    psv[at + 5]
}

/// I and J only fed to `v_interp_*` stay host interpolation: no
/// barycentrics are declared.
#[test]
fn interpolation_alone_declares_no_barycentrics() {
    let [e0, e1] = exp_mrt0(2);
    let words = [
        0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
        0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
        e0,
        e1,
        S_ENDPGM,
    ];
    let shader = pixel(&words, PERSP_CENTER).unwrap();
    let (features, inputs) = parts(&shader);
    assert_eq!(features & 0x2_0000, 0);
    assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
    validate(&shader);
}

/// An interpolation whose i is computed (`v_mul_f32 v3, 0.5, v0`) reads
/// the attribute's vertices: TEXCOORD0 becomes constant, read with
/// `attributeAtVertex`; i and j are `SV_Barycentrics` values.
#[test]
fn computed_barycentrics_read_the_vertices() {
    let [e0, e1] = exp_mrt0(2);
    let words = [
        0x1006_00F0, // v_mul_f32 v3, 0.5, v0
        0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
        0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
        e0,
        e1,
        S_ENDPGM,
    ];
    let shader = pixel(&words, PERSP_CENTER).unwrap();
    let (features, inputs) = parts(&shader);
    assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
    assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
    assert_eq!(texcoord_mode(&shader), 1);
    validate(&shader);
}

/// Weights from constants alone: the vertices, no `SV_Barycentrics`.
#[test]
fn constant_weights_need_no_barycentrics() {
    let [e0, e1] = exp_mrt0(2);
    let words = [
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
        0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
        e0,
        e1,
        S_ENDPGM,
    ];
    let shader = pixel(&words, PERSP_CENTER).unwrap();
    let (features, inputs) = parts(&shader);
    assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
    assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
    assert_eq!(texcoord_mode(&shader), 1);
    validate(&shader);
}

/// The same attribute interpolated with the loaded i and j and with a
/// computed j: both from its vertices, the first weighed with
/// `SV_Barycentrics`.
#[test]
fn computed_and_loaded_barycentrics_on_one_attribute() {
    let [e0, e1] = exp_mrt0(2);
    let words = [
        0xC810_0100, // v_interp_p1_f32 v4, v0, attr0.y
        0xC811_0101, // v_interp_p2_f32 v4, v1, attr0.y
        0x1006_02F0, // v_mul_f32 v3, 0.5, v1
        0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
        0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
        v_add_f32(2, 2, 4),
        e0,
        e1,
        S_ENDPGM,
    ];
    let shader = pixel(&words, PERSP_CENTER).unwrap();
    let (_, inputs) = parts(&shader);
    assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
    assert_eq!(texcoord_mode(&shader), 1);
    validate(&shader);
}

/// P10 of an attribute the program interpolates is - P0.
#[test]
fn p10_of_an_interpolated_attribute() {
    let [e0, e1] = exp_mrt0(2);
    let words = [
        0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
        0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
        0xC812_0000, // v_interp_mov_f32 v4, p10, attr0.x
        v_add_f32(2, 2, 4),
        e0,
        e1,
        S_ENDPGM,
    ];
    let shader = pixel(&words, PERSP_CENTER).unwrap();
    assert_eq!(texcoord_mode(&shader), 1);
    validate(&shader);
    // Alone, what the hardware holds is not known.
    let [e0, e1] = exp_mrt0(4);
    let words = [0xC812_0000, e0, e1, S_ENDPGM];
    assert!(pixel(&words, PERSP_CENTER).is_err());
}

/// XNW: the form of the title's TrailmapDeformData (ecd15b4d at
/// 0x29c-0x2d4): i and j zeroed on a path a lane test picks, then
/// interpolated: computed barycentrics through a join of the loaded i and
/// j with 0. On a host without barycentrics (the card) the program keeps
/// the refusal it had; on one with them it reads SV_Barycentrics and the
/// attribute's vertices, and validates.
#[test]
fn barycentrics_joined_with_zero_need_the_host_barycentrics() {
    let [e0, e1] = exp_mrt0(2);
    let words = [
        0x7C08_0280, // v_cmp_gt_f32 vcc, 0, v1 (a lane test)
        0xBF87_0002, // s_cbranch_vccnz -> the first half
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
        0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
        e0,
        e1,
        S_ENDPGM,
    ];
    let shader = pixel(&words, PERSP_CENTER).unwrap();
    let (features, inputs) = parts(&shader);
    assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
    assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1, "{inputs:?}");
    assert_eq!(texcoord_mode(&shader), 1);
    validate(&shader);
    let refused = pixel_on(&words, PERSP_CENTER, false).map(|_| ()).unwrap_err();
    let refused = format!("{refused:?}");
    assert!(refused.contains("an interpolation with computed barycentrics"), "{refused}");
    // Without a computed interpolation nothing changes with the option.
    let plain = [0xC808_0000, 0xC809_0001, e0, e1, S_ENDPGM];
    let on = pixel(&plain, PERSP_CENTER).unwrap();
    let off = pixel_on(&plain, PERSP_CENTER, false).unwrap();
    assert_eq!(on.dxil, off.dxil);
}

/// XNW: a vertex shader that copies its inputs to its outputs: the pixel
/// program's own input elements (vertex data in their order), then
/// `SV_Position` (the last four floats of a vertex).
#[cfg(windows)]
fn copy_vs(ps: &HostShader) -> HostShader {
    use super::bitcode::{Attr, Term};
    let c = Container::parse(&ps.dxil).unwrap();
    let mut outputs: Vec<SigElement> = signature(c.part(FourCC::ISG1).unwrap().data).unwrap().into_iter().filter(|e| e.kind == 0).collect();
    let mut pos = element("SV_Position", 3, outputs.iter().filter_map(|e| e.row).max().map_or(0, |r| r + 1));
    pos.interp = 4;
    outputs.push(pos);
    let mut s = shader::Shader::new(false);
    let b = s.entry;
    s.inputs = outputs.iter().map(|e| SigElement { name: if e.kind == 3 { "POSITION".into() } else { e.name.clone() }, kind: 0, interp: 0, usage: 15, ..e.clone() }).collect();
    for (id, e) in outputs.iter_mut().enumerate() {
        for col in 0..e.cols {
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
            let x = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), x];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        e.usage = 15;
    }
    s.outputs = outputs;
    s.m.terminate(b, Term::Ret);
    HostShader {
        stage: HwStage::Vs,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        address_map: false,
        descriptor_words: false,
    }
}

/// XNW: the construct on a device with barycentrics (WARP 1.0.22:
/// `XNW_TEST_GPU=1 XNW_ADAPTER=warp XNW_WARP=<warp folder>
/// XNW_AGILITY=<agility folder>`; skipped without one). TEXCOORD0.x
/// is 0.25, 1.25, 0.25 at the three vertices: interpolated as loaded it is
/// 0.25 + i; with i halved before the first half (computed barycentrics) it
/// is 0.25 + i / 2 in every pixel the triangle covers.
#[test]
#[cfg(windows)]
fn computed_barycentrics_draw_on_a_device_that_has_them() {
    let Some(host) = host::Host::for_tests() else {
        eprintln!("no device; skipped");
        return;
    };
    let caps = xnw_device::host_barycentrics();
    if caps != Some(true) {
        eprintln!("device without barycentrics ({caps:?}); skipped");
        return;
    }
    let [e0, e1] = exp_mrt0(2);
    let plain = [0xC808_0000, 0xC809_0001, e0, e1, S_ENDPGM];
    let halved = [0x1006_00F0, 0xC808_0003, 0xC809_0001, e0, e1, S_ENDPGM];
    // TEXCOORD0 (x, 0, 0, 1), then the position; counter-clockwise from
    // the provoking vertex.
    let vertices: [f32; 24] = [
        0.25, 0.0, 0.0, 1.0, -0.9, -0.9, 0.0, 1.0, //
        1.25, 0.0, 0.0, 1.0, 0.9, -0.9, 0.0, 1.0, //
        0.25, 0.0, 0.0, 1.0, 0.0, 0.9, 0.0, 1.0,
    ];
    let mut out = Vec::new();
    for words in [&plain[..], &halved[..]] {
        let ps = pixel(words, PERSP_CENTER).unwrap();
        let vs = copy_vs(&ps);
        validate(&vs);
        let (root, pso) = host.pipeline(&vs, &ps).unwrap();
        let d = host::Draw { roots: vec![], textures: &[], samplers: &[], vertices: &vertices, stride: 32, size: (32, 32), mesh_groups: 0 };
        out.push(host.draw(&root, &pso, &d).unwrap());
    }
    let (mut n, mut worst) = (0, 0f32);
    for (a, b) in out[0].iter().zip(&out[1]) {
        if a[0] == 0.0 {
            continue;
        }
        n += 1;
        worst = worst.max((b[0] - (0.25 + (a[0] - 0.25) / 2.0)).abs());
    }
    eprintln!("computed barycentrics: {n} pixels of 32x32 drawn, worst |halved - (0.25 + i/2)| {worst:e}");
    assert!(n > 200, "{n} pixels drawn");
    assert!(worst < 1e-5, "{worst}");
}

/// `v_pk_fmac_f16_dpp` (VOP2 + DPP16): `src0` comes from the DPP-selected
/// lane in both packed f16 halves, not the executing lane's own VGPR.
/// Origin: VIctory valu_tests.rs `packed_fmac_dpp_reads_the_swizzled_lane`
/// (upstream, 754e60e4), GPL-3.0-or-later; run like the barycentric draw above
/// (`XNW_TEST_GPU=1 XNW_ADAPTER=warp ...`; skipped without a device).
/// Pixel shader on a 2x2 quad: v0 = SV_Position.x (0.5 / 1.5), v1 = y; v2 =
/// pack_f16(x + 2y, x + 4y); quad_perm:[1,0,3,2] swaps horizontal neighbours
/// before multiplying by (lo 4.0, hi 2.0) and adding (lo 0.5, hi 1.0).
#[test]
#[cfg(windows)]
fn packed_fmac_dpp_reads_the_swizzled_lane() {
    let vop1 = |op: u32, dst: u32, src: u32| 0x7E00_0000 | dst << 17 | op << 9 | src;
    let words: [u32; 16] = [
        0x08 << 25 | 2 << 17 | 1 << 9 | 244,  // v_mul_f32 v2, 2.0, v1
        0x03 << 25 | 2 << 17 | 2 << 9 | 256,  // v_add_f32 v2, v0, v2
        0x08 << 25 | 3 << 17 | 1 << 9 | 246,  // v_mul_f32 v3, 4.0, v1
        0x03 << 25 | 3 << 17 | 3 << 9 | 256,  // v_add_f32 v3, v0, v3
        0x2F << 25 | 2 << 17 | 3 << 9 | 258,  // v_cvt_pkrtz_f16_f32 v2, v2, v3
        0x7E06_02FF,                          // v_mov_b32 v3, 0x4000_4400
        0x4000_4400,
        0x7E08_02FF,                          // v_mov_b32 v4, 0x3C00_3800
        0x3C00_3800,
        0x3C << 25 | 4 << 17 | 3 << 9 | 0xFA, // v_pk_fmac_f16_dpp v4, v2, v3
        0xFF00_B102,                          //   quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
        vop1(0x00B, 5, 256 + 4),              // v_cvt_f32_f16 v5, v4 (low half)
        0x16 << 25 | 6 << 17 | 4 << 9 | 144,  // v_lshrrev_b32 v6, 16, v4
        vop1(0x00B, 6, 256 + 6),              // v_cvt_f32_f16 v6, v6 (high half)
        0xF800_180F,                          // exp mrt0 v5, v6, v5, v6 done vm
        0x0605_0605,
    ];
    let mut code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    code.extend_from_slice(&S_ENDPGM.to_le_bytes());
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps { inputs: vec![], input_addr: (1 << 8) | (1 << 9), db_shader_control: 0, patches: vec![] },
        code,
    };
    let mut pos = element("SV_Position", 3, 0);
    pos.interp = 4;
    pos.usage = 0xF;
    let mut target = element("SV_Target", 16, 0);
    target.usage = 15;
    let isg1 = container::signature(&[pos], false);
    let osg1 = container::signature(&[target], false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let res = resource::track(&p, &native.layout, native.stage).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let st = structure::structure(&p).unwrap();
    let ps = emit_with(&c, &native, &p, &st, &io, &res, Options::default()).expect("emit");
    let vs = copy_vs(&ps);
    validate(&ps);
    validate(&vs);
    let Some(host) = host::Host::for_tests() else {
        eprintln!("no device; skipped");
        return;
    };
    let vertices: [f32; 12] = [-1.0, -1.0, 0.5, 1.0, -1.0, 3.0, 0.5, 1.0, 3.0, -1.0, 0.5, 1.0];
    let want = [
        [10.5f32, 8.0, 10.5, 8.0], // (0,0) reads (1,0): (2.5*4+0.5, 3.5*2+1)
        [6.5, 6.0, 6.5, 6.0],      // (1,0) reads (0,0): (1.5*4+0.5, 2.5*2+1)
        [18.5, 16.0, 18.5, 16.0],  // (0,1) reads (1,1): (4.5*4+0.5, 7.5*2+1)
        [14.5, 14.0, 14.5, 14.0],  // (1,1) reads (0,1): (3.5*4+0.5, 6.5*2+1)
    ];
    let (root, pso) = host.pipeline(&vs, &ps).unwrap();
    let d = host::Draw { roots: vec![], textures: &[], samplers: &[], vertices: &vertices, stride: 16, size: (2, 2), mesh_groups: 0 };
    let texels = host.draw(&root, &pso, &d).unwrap();
    assert_eq!(texels, want, "v_pk_fmac_f16_dpp quad_perm:[1,0,3,2]");
}
