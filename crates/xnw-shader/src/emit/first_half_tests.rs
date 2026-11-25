// XNW: tests of the first-half port from victory-fork a26859f (GPL-3.0-or-later; upstream has no
// test of it). The form is the one the 28 refused pixel programs use (PS_VehicleLivery*Deferred,
// PS_DeferredTextured, PS_MirrorReflection*): an f16 pair written in place, `v_interp_p1ll_f16 vN` then
// `v_interp_p2_f16 vN, ..., vN`, whose second half keeps the high half of vN (the first half's).
use super::container::SigElement;
use super::*;
use crate::native::{PsInput, StageIo, Usage};

/// `SPI_PS_INPUT_ADDR` bit PERSP_CENTER.
const PERSP_CENTER: u32 = 1 << 1;
const S_ENDPGM: u32 = 0xBF81_0000;

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

/// Translates (no retry) a pixel program with one TEXCOORD0 attribute
/// (attr 0, row 0), 16-bit when `fp16`, exporting v3 to mrt0.
fn pixel(words: &[u32], fp16: bool) -> Result<HostShader, ShaderError> {
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
                fp16,
                pack2: false,
                row2: 0,
            }],
            input_addr: PERSP_CENTER,
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
    emit_with(&c, &native, &p, &st, &io, &res, Options::default())
}

fn validate(shader: &HostShader) {
    #[cfg(windows)]
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).unwrap();
    }
    let _ = shader;
}

/// `exp mrt0, v3, v3, v3, v3 done vm`, `s_endpgm`.
const EXPORT_V3: [u32; 3] = [0xF800_180F, 0x0303_0303, S_ENDPGM];

/// The vehicle/decal form: the first half's register is the second half's
/// dst, so it is read as data (the half `low_half` keeps). On the retry the
/// first half is evaluated as its pair; the result passes the validator.
#[test]
fn an_f16_pair_written_in_place_translates_on_the_retry() {
    let mut words = vec![
        0xD742_0002, 0x0002_0000, // v_interp_p1ll_f16 v2, v0, attr0.x
        0xD75A_0002, 0x040A_0200, // v_interp_p2_f16 v2, v1, attr0.x, v2
        0x7E06_1702, // v_cvt_f32_f16 v3, v2
    ];
    words.extend(EXPORT_V3);
    let pairs = lower::with_first_half_pairs(|| pixel(&words, true)).unwrap();
    validate(&pairs);
}

/// Where the approximation shows: a first half read in full (its partial
/// sum P0 + i P10 exported). Refused on the first translation; the retry
/// exports the whole interpolation (P0 + i P10 + j P20) instead: the
/// console's value needs the attribute's vertices (barycentrics).
#[test]
fn an_f32_first_half_read_in_full_is_the_pair_on_the_retry() {
    let mut words = vec![
        0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
        0x7E06_0302, // v_mov_b32 v3, v2
        0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
    ];
    words.extend(EXPORT_V3);
    match pixel(&words, false) {
        Err(ShaderError::Unsupported { what, .. }) => assert_eq!(what, lower::FIRST_HALF),
        other => panic!("{other:?}"),
    }
    let shader = lower::with_first_half_pairs(|| pixel(&words, false)).unwrap();
    validate(&shader);
}
