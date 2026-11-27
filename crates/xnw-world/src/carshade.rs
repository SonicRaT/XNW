//! CAR1 (`XNW_WORLD_RENDER=...,carshade=on`, default off; needs
//! `ourlight=metal`): vehicle paint in our lighting pass, a clearcoat over
//! the lit base.
//!
//! The title: its vehicle body programs (PS_VehicleLiveryOffOnlyDeferred
//! a8000c5c NoDeform / 7095f810 deform) write the six-target G-buffer of
//! `PS_DeferredTextured`: mrt1 R = 1 - roughness (the quad-variance
//! widening 0x6b20..0x6ed0), mrt2 RGB = sqrt(1 - sqrt(1 - F0)) (0x8318..
//! 0x83d0), mrt3 B A = two material-class bytes of a draw constant (s82,
//! 0x8330, 0x8364) [V disasm]; the tiled lighting picks a program per class,
//! and a CLEARCOAT family exists (TL_PS_PBR_MATERIAL_CLEARCOAT_*, shader
//! index) [V names; that the paint takes it: I]. The paint's own numbers
//! (roughness, clearcoat) come from the material slot's 100-byte T8+0xAA0
//! row that the function packs from the material definition [V pseudo;
//! which field is which: U]. Its reflections are the stochastic
//! screen-space / traced pass and the sky paraboloid (LIT1): ours have neither.
//!
//! Ours, two parts:
//! - the raw rows (`#define CARSHADE`): a row of the vehicle paint class
//!   (`standard_vehicle*`, worldtex `CLASS_PAINT`: bodies and interiors, one
//!   program family; not the wheel and tyre effects), matFlags bit [`COAT`],
//!   writes 1 into the albedo target's alpha (the title's RT0 .a; 0 on every
//!   other opaque row: the frame clears it, decal rows do not write alpha).
//!   A first cut keyed on the vehmat palette entries glossed tyres and trim
//!   (the palette is a material ramp, PAL1), so the class decides;
//! - the lighting pass reads that alpha through a table of its own (root
//!   parameter [`root_param`], register [`REGISTER`]) and, at our rows'
//!   pixels (not the bridged rows'), puts [`HLSL`]'s clearcoat over the lit
//!   base: Schlick Fresnel (F0 0.04), the environment along the reflected
//!   ray (ground below the horizon, the fog colour at it, the sky above),
//!   a sharp sun lobe (GGX, roughness 1 - gloss within [`COAT_ROUGH`]) and a
//!   broad one for the base tinted by the paint (flake: no grey veil).
//!
//! CAR1B: only exterior paint takes it. The row is of the paint class AND
//! its vertex's material slot is a paint slot (`vehmat::COAT`: element 0's
//! +32 colour is set; interiors, trims, pedals, seats are not), and the
//! gloss the row writes is 1 - the slot's roughness byte (element 0 +24
//! byte 3, high nibble: the paint's 0x1e gives 0.94). Rows without a vehmat
//! table take no coat. Elsewhere the pass's colour is unchanged (a select, no branch).

use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Graphics::Direct3D12::*;

/// Process-wide: set by `Pass::new` before it compiles the raw shader.
static ON: AtomicBool = AtomicBool::new(false);

/// `carshade=on` (with `ourlight=metal`).
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

pub(crate) fn set(v: bool) {
    ON.store(v, Ordering::Relaxed);
}

/// `carshade=on|off` in `XNW_WORLD_RENDER` (the last one wins; default off).
pub fn parse_switch(v: &str) -> bool {
    v.split(',').filter_map(|p| p.trim().strip_prefix("carshade=")).next_back() == Some("on")
}

/// The raw shader's matFlags bit: a vehicle paint-class row ([`crate::Material::coat`]).
pub const COAT: u32 = 32;

/// The pass's register for the albedo target's view.
pub const REGISTER: &str = "t16";

/// CAR1B: the clearcoat's roughness range (perceptual); within it 1 - the pixel's gloss.
pub const COAT_ROUGH: [f32; 2] = [0.04, 0.3];

/// The pass's paint term; after the pass's declarations (Light, `clipToWorld`,
/// `sunSpecular`, `skyRadiance`), before its entries.
pub const HLSL: &str = r#"
// CAR1 (`carshade=on`, carshade.rs): the albedo target, .a = an exterior paint pixel's gloss, 1 - roughness (0: none).
Texture2D<float4> carIn : register(CARSHADE);
// CAR1B: the coat's roughness from the gloss, clamped (a lacquer); the flake sheen under it, tinted by the paint.
#define CAR_COAT_MIN 0.04
#define CAR_COAT_MAX 0.3
#define CAR_BASE_ROUGH 0.3
// The environment along r: the ground below the horizon, the horizon (the fog colour as radiance) up to the sky,
// the horizon's edge widened by the roughness.
float3 carEnv(float3 r, float rough) {
    float s = 0.01 + 0.5 * rough;
    float3 sky = lerp(skyRadiance(lPad.yzw), lSky.rgb, saturate(r.z * 2));
    return lerp(lGround.rgb, sky, smoothstep(-s, s, r.z));
}
// hdr: the pixel lit as a row of ours; n its normal, base its linear albedo, sh its sun fraction.
float3 carShade(float3 hdr, int3 p, uint w, uint h, float3 n, float3 base, float sh) {
    float k = saturate(carIn.Load(p).a);
    float rc = clamp(1 - k, CAR_COAT_MIN, CAR_COAT_MAX);
    float2 ndc = (float2(p.xy) + 0.5) / float2(max(w, 1), max(h, 1)) * float2(2, -2) + float2(-1, 1);
    float4 q = mul(float4(ndc, 0.5, 1), clipToWorld);
    float3 v = -normalize(q.xyz / (abs(q.w) > 1e-12 ? q.w : 1e-12) + 1e-9);
    float3 nf = dot(n, v) < 0 ? -n : n;
    float3 ld = normalize(lDir.xyz + 1e-9);
    float3 hv = normalize(ld + v);
    float nl = saturate(dot(nf, ld)), nv = saturate(dot(nf, v)), nh = saturate(dot(nf, hv)), vh = saturate(dot(v, hv));
    float fc = 0.04 + 0.96 * pow(1 - nv, 5);
    float3 sun = lRad.rgb * sh;
    float3 coat = fc * carEnv(reflect(-v, nf), rc) + sun * sunSpecular(0.04, nl, nv, nh, vh, rc);
    float3 sheen = sun * sunSpecular(0.3 * saturate(base), nl, nv, nh, vh, CAR_BASE_ROUGH);
    float3 o = (hdr + sheen) * (1 - fc) + coat;
    return k > 0.5 / 255 ? o : hdr;
}
"#;

/// The lighting pass's source with the paint term (`metal`: the variant
/// with the camera rays; without it the source is unchanged): the define,
/// the view's table appended to every root, [`HLSL`] before the vertex
/// entry. Applied after `ourlight::source_post` (its root edits look for
/// the roots' old ends).
pub fn pass_source(src: String, metal: bool) -> String {
    if !metal {
        return src;
    }
    let table = format!(", DescriptorTable(SRV({REGISTER}), visibility=SHADER_VISIBILITY_PIXEL)\"");
    let mut out = format!("#define CARSHADE {REGISTER}\n");
    for line in src.lines() {
        match line.strip_suffix('"').filter(|_| line.starts_with("#define RS \"")) {
            Some(head) => out.push_str(&format!("{head}{table}")),
            None => {
                if line.starts_with("struct VsOut ") {
                    out.push_str(HLSL);
                }
                out.push_str(line);
            }
        }
        out.push('\n');
    }
    out
}

/// The table's root parameter: after b0, the pass's table, the camera
/// constants (metal) and, with `lights=on`, the lights' CBV.
pub fn root_param(lamps: bool) -> u32 {
    3 + u32::from(lamps)
}

/// The raw shader with `#define CARSHADE 1` first when the switch is on
/// (its paint rows write their gloss).
pub fn raw_source(src: String) -> String {
    if on() && !src.contains("#define CARSHADE") { format!("#define CARSHADE 1\n{src}") } else { src }
}

/// The pass's binding: the albedo target, its view's heap slot, the root
/// parameter, the heap step.
pub(crate) struct View {
    pub albedo: ID3D12Resource,
    pub slot: u32,
    pub param: u32,
    pub step: u32,
}

impl View {
    /// The view of `albedo` (the G-buffer's target 1) at `cpu(slot)`.
    pub(crate) fn new(device: &ID3D12Device, albedo: &ID3D12Resource, slot: u32, param: u32, cpu: &dyn Fn(u32) -> D3D12_CPU_DESCRIPTOR_HANDLE) -> View {
        let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: crate::gbuf::ALBEDO,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } },
        };
        // SAFETY: a live device, resource and heap slot.
        let step = unsafe {
            device.CreateShaderResourceView(albedo, Some(&view), cpu(slot));
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        View { albedo: albedo.clone(), slot, param, step }
    }

    /// Binds the table on `l` (the pass's root set): `table` is the heap's
    /// handle at the pass's first slot `first`.
    pub(crate) fn bind(&self, l: &ID3D12GraphicsCommandList, table: D3D12_GPU_DESCRIPTOR_HANDLE, first: u32) {
        let at = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: table.ptr + u64::from(self.slot.saturating_sub(first)) * u64::from(self.step) };
        // SAFETY: an open list with the pass's root signature set; the slot holds our view.
        unsafe { l.SetGraphicsRootDescriptorTable(self.param, at) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_last_wins() {
        assert!(!parse_switch("ourlight=metal"));
        assert!(parse_switch("ourlight=metal,carshade=on"));
        assert!(!parse_switch("carshade=on, carshade=off"));
    }

    /// Main's variant (metal, title normals, sun shadow, lights, post) and the
    /// plain metal one compile with the term; the table is the root's last.
    #[test]
    fn pass_compiles_with_the_term() {
        for (t, s, la, post) in [(true, true, true, true), (true, true, true, false), (false, false, false, false), (true, false, true, false)] {
            let src = pass_source(crate::ourlight::source_post(true, t, s, la, post), true);
            assert!(src.contains("SRV(t16), visibility=SHADER_VISIBILITY_PIXEL)\"") && src.contains("float3 carShade("), "{t} {s} {la} {post}");
            if la {
                assert!(src.contains("CBV(b2, visibility=SHADER_VISIBILITY_PIXEL), DescriptorTable(SRV(t16)"), "lights' CBV stays parameter 3");
            }
            for entry in [(c"vs", c"vs_5_1"), (c"ps", c"ps_5_1")] {
                crate::compile_source(&src, entry.0, entry.1).unwrap();
            }
        }
        // LGT1's `emissive=on` lengthens the pass's table only: the paint table stays last.
        let src = pass_source(crate::ourlight::source_full(true, true, true, true, Some(1.0), true), true);
        assert!(src.contains("#define EMISSIVE t6") && src.contains("CBV(b2, visibility=SHADER_VISIBILITY_PIXEL), DescriptorTable(SRV(t16)"));
        crate::compile_source(&src, c"ps", c"ps_5_1").unwrap();
        assert_eq!(pass_source("x".into(), false), "x");
        assert_eq!((root_param(false), root_param(true)), (3, 4));
    }

    /// The raw shader with the paint gloss still writes three targets.
    #[test]
    fn raw_shader_compiles_with_the_gloss() {
        let src = format!("#define CARSHADE 1\n#define OURLIGHT 1\n#define GBUF 1\n{}", crate::RAW_SHADER);
        let ps = crate::compile_source(&src, c"ps", c"ps_5_1").unwrap();
        assert_eq!(crate::gbuf::targets(crate::blob_bytes(&ps)), 7);
    }
}
