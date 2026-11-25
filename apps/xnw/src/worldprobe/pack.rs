//! The frame pack v2 writer, behind
//! `XNW_WORLD_PROBE=...,pack=2` (default off).
//!
//! Two sampling points. The geometry-draw hook numbers each draw it
//! records (`hook_entry`, a run-wide sequence kept per thread). The list
//! sink (`xnw_gpu::d3d12x::tap`, every graphics draw the title records,
//! null mode included) copies the binding: root constants, each table's
//! resolved descriptor slots, the bytes behind its constant buffers, the
//! pipeline's program identity, and each texture slot's identity (pack
//! origin, else resident texels and a fill fingerprint). Everything is
//! content-addressed (`xnw_core::hash128`); new bytes go to the capture's
//! `blobs\r-<hash>` when the seq writer writes the frame. A failed read is
//! a record with a reason, never a panic.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use xnw_formats::framepack::{self as fp, Fill, FramePack, Ref, SlotView, kind, reason, view};
use xnw_gpu::d3d12x::tap::{self, TapDraw, TapRoot, TapView};

/// Bytes behind one constant buffer, at most.
const CBV_CAP: u64 = 64 << 10;
/// Texel bytes of one texture, at most (larger ones: fill and origin only).
const TEX_CAP: u64 = 4 << 20;
/// Textures up to this size whose sampled windows read zero are read whole
/// before they are called zero-filled.
const ZERO_CHECK: u64 = 256 << 10;
/// New texel bytes per frame, at most.
const FRAME_TEX_CAP: u64 = 256 << 20;
/// Heap slots one bindless index buffer resolves, at most.
const BINDLESS_MAX: usize = 16_384;
/// The bindless index CB's slot in the T8 table (T8+0xAC0, s4 [V]).
const BINDLESS_SLOT: u32 = 0xAC0 / 32;
/// Bytes of one root view or buffer view (mesh streams), at most.
const BUF_CAP: u64 = 16 << 20;
/// New buffer bytes per frame, at most (ranges over 64 KB only: small
/// views such as the ped remap range are never cut by it).
const FRAME_BUF_CAP: u64 = 256 << 20;
/// A head graph dispatch's plain unordered buffer views are read (head1
/// #926: the CPU writes the rig's weights and controls there; one is
/// 25,165,824 B, so its own cap).
const HEAD_UAV_CAP: u64 = 32 << 20;
/// How a sink's tables are read: a draw's, a dispatch's (no unordered
/// bytes, no bindless or record), a head graph dispatch's (as a dispatch,
/// with its plain unordered buffers read).
const TB_DRAW: u8 = 0;
const TB_DISPATCH: u8 = 1;
const TB_HEAD: u8 = 2;
/// Frames being recorded (not yet sent to the writer), at most; past it
/// the oldest is skipped and counted. A sent frame's data moves into its
/// capture (`take`), so a slow writer never costs a frame: an earlier fault was
/// a cap on frames *waiting for the writer* (64; the title ran 69 ahead
/// while frames 1002/1003 were written).
const OPEN_MAX: usize = 16;
/// Slots a compute table root is resolved to (the pre-skin programs
/// reach ud4+0x320 = slot 25 and ud5+0x120 = slot 9, used list [V]);
/// the slots after a table belong to other tables.
const DISPATCH_SLOTS: u32 = 32;
/// Container hashes (= shader index hashes) of effect
/// `ped_skinning_cs`: CS_PedStandardSkinning_0..12, CS_PosOnlyLodSkinning,
/// CS_PosOnlyLodSkinningSecmo, CS_GenWorldAabbs [V]. Only their direct
/// dispatches are sinks: a capture has 1,628-3,090 direct dispatches a frame
/// (run.log "frame calls to 2388..2391"), about 60 of them these.
pub(super) const PED_SKINNING: [u128; 16] = [
    0xa45da7131ba3596484a96440c5f20afb,
    0xa553f1ac88292cda70b4c1b175cf7fc8,
    0x448e13a4b81dfc4ddbbe1ddf2fbc4d40,
    0xa1bddae1ec76f785a971032791928970,
    0x54afc93ed3785edad182ced2576df307,
    0x0d96d9b4548718db39324555d165775d,
    0x9ebed5a0a2375ba0c622c484be9e18b3,
    0xd2c703567ec00bc4a8361074350067a5,
    0xdcfcc6a803ab0b263a10b1c561d7ba77,
    0x658aa4879482263feb31ad13c3fec09f,
    0x14222a26b440981ce48b516fc2aa3af0,
    0x659b134db69fc77748f972df1499ef37,
    0x0832d53d82233216f374a6b59f7ed1db,
    0xef8255f869e206be1c80b2cf91611f4a,
    0xe712fb959de87915115b7210fb92cc7c,
    0xb8f9acafc6771f508efe8d2e7a17590b,
];
/// Container hashes (= shader index hashes) of the strand-hair programs
/// whose inputs a capture keeps: `smh_pose_*`
/// CS_PreSkinSimGuides, CS_PreSkinSimNodes, CS_UpdatePosePosition (9) and
/// `smh_gen_*` CS_InterpolateStrands (80) [V shader_index.py --sql over
/// names 'smh_pose%', 'smh_gen%'].
const HAIR_INPUTS: [u128; 89] = [
    0x3a10724ee52ff399775b1193043bfc71, 0x708b16253dd9beb090e39f6ff9077629, 0x0f5d6f22cdbede03f6af03a5b25282d0, 0x39fdc1bbf1fd8c3748872783764f0f13,
    0xa1bd4326c52b253feee511ef4cfc9992, 0x10dfef00b5890f3ef7273e993264d077, 0x46b73725a1d33ea68c4b25435fe55d54, 0xa49e33ea638e8a90c4224adfab34d17a,
    0x2e1116060ad5a15b4cdaff3dbf4786e1, 0x4292afa57b7b258b7bec144ae29a0892, 0x98e40e37b12a6312e7377edcbbb20948, 0xe216650a351755bc3b1c5bc96f3f1639,
    0x763a6b6293523e174148a9adeb340fdf, 0x2cb5983112f81af1dac7bc58ee9fce44, 0x1b6756f805c347617a3aef0b3b4b1766, 0x21554820c7715412a66314f01c82997a,
    0x9109c5ef8d84a8fd45ba290422781b9b, 0xcb2a35e99bea70d6ea59a8e746c369fb, 0x70079b50f7e32bed114eec001d6f2306, 0x48e658a0db8468b59f6fc84a651fa171,
    0x83894af9a5679926e7ea595cda57ee15, 0xf989b3dc2167e65aeee7adb6e002748a, 0xde50b904a0e64cb2aa91d8c2bb427da6, 0xb68b88fc5de29f4e7f8fab3469a2c26c,
    0x8c5bbb6c1298058fa2aeded0ad37e011, 0x0653a80299e9ff621355d0d3e1cb7280, 0xb57e3b3dd72f1ab1fc2daee7c8b464c5, 0xd6e32c6245def7c1424124c18ade4a1e,
    0xed0786dfc6c19bd8ed63ef894e141381, 0x6dfde0ffda7f29b4e471599942e1d1db, 0xac55f6fd9a34742ccdec671ffed26c26, 0xb137d5fc8899486be1eca77f09d3c9d2,
    0x4212d9cf250aab7621608a5cbf74089b, 0x7ae45a1eb3621a9fa5db5b16b847084c, 0x8e212861bd6831603882d83bd274033c, 0xc69afa18b41322924dcb75042f5a576f,
    0xb7407409bf1260c4a8cb05ed3a753fb2, 0xd6b86aca42f3cf5b730f40ea73235129, 0xdb5a6dd95d1d181ff5f34390b38d768f, 0x1541688532405f3e3ef38813c25cc693,
    0xa239c953917f782422df6920207617f7, 0x61659908836966b214ed96be914c33a6, 0xd26858433ee3f9e7fac524688c127bba, 0x277e0f1c2bfed4d398d53102d897fdbe,
    0x0b9bd12457e050e7edc246df56956ec5, 0xe8fab7607848e27f7941b996df81d712, 0x10b78fcbad8a138b2113e3ac30509907, 0x8ccd7c48027fa667372e72e8b15ce9e1,
    0x15637155c27a5784d3430782640fbce2, 0xc1eda29fee2c736ae103cba23783d3ad, 0x052ddb1b2d4776f982e377ac162bf768, 0xf8c68c75943ca626aba319736cb14f5f,
    0xbe6489840d8e358fd22bfffeaa732e57, 0x9c5904029d3738e81f43901eb6529c53, 0x8495420024ef008681161689ebdc00a8, 0x3d3754b9d6991bfc0ab7d5b79e9df724,
    0xf65b3bda3b2cdc7af431d076a475c3fa, 0x3b03d83ef1ce096b8379b6e827e1d73d, 0x355ad8017ccd37ffbaf6104d532f9d53, 0xa68fef426b22cecc5fda220e1c4fb206,
    0x39091272b1f0287232df838fb795cc2e, 0xd198ef132a860c5d0673fad3a13bdb2f, 0xaa0703c9814ce4e1e487f02628343521, 0x1a43a27af8a16fd5ff584b4b6046f492,
    0xa49a49dc33e1d59e3bc03448423a61d3, 0xfcfcd6128eb0d578e248ce44f3d3b4e3, 0x462d5a9d5cf3e085605892563d005378, 0xfb3a193d68619fd4e93bfcadf90cd29b,
    0xf3956a96107569a3ef3fdfa132d3cb9f, 0x4d289f80beae7daf56e9f418cc7572a3, 0xfcc1a56368bf6ee9c8d052894ad0e9fa, 0xb02176a951dfa31ce155bc9787c4464b,
    0xb9abdff902b7cf24239d9536b4285464, 0xf0edfdf94ca1dae6c2fe39339b0a4586, 0x1a02cdef90afb4f77123141d69420ac8, 0xe604904e9de8dd2c94e6bae465c77a36,
    0x0988e72660716da8e5afc7cce7416dd3, 0x7bfd6fd91b858a2c573e598d06bdf4b3, 0x1265eb54f04f94ff222fe2ca2f843871, 0x8afbf8939524f90fe0209e43fd833883,
    0x0479e16ed4c491a11eafe3d625048498, 0xc0f42c93228498443f2b6b3a5ca2266b, 0xed83ecd172eedc19b330895bb89e88da, 0xed403b6d88dab0c1689e31f52fa043cd,
    0x70a718878bf903a26d2254b40f2c1e59, 0xbc6245e7b8c8f1ccef4638bbd2773c71, 0x243b071d6ae57113c5644a5a5c80c351, 0x165335c899403b82cc192eaeb577ea56,
    0x364355050c195f360c4910d9769b95ce,
];
/// `smh_sim_*` CS_InitSimBuffer, CS_Solve*, CS_LodBlend: sinks for their
/// order only (roots, no tables) [V shader_index.py].
const HAIR_ORDER: [u128; 17] = [
    0x1a130b269cd47ef4812dd15752152b23, 0xbd4d55900c5e927c7ca235af566c08a2, 0x5adcf5cccaed9253d6719718e7193013, 0x102ab06d66c15a80f376c0879fe18ffa,
    0x8ba39ba05b199d96d56762ed7d3c9b48, 0x25b63d3d14678cbcc520266ff95684f0, 0x5ff30f5ba243ec03a8f14f42c416b792, 0x9aafa53ac20e837b14cd31dda3ca3157,
    0xd3f920b5ae197062f576114eaf3b8b31, 0xc40def105e2826d0c14cfbad46d6110a, 0xd8f39b4be685c330a42ce9ff24e9aae6, 0xa7b60867d8bb1cf70efde8850f729cbf,
    0x4b347011e0f5a254be472bafd0db04e8, 0xdcfc8670a5f01ddf6276cb11caad1922, 0x1886a183ed726df1bb42ab1799d778bb, 0x829cb2574c8b01861abecf0ad27e5a7b,
    0x0c41534fa266b658e793684fd19ee9a0,
];
/// Container hashes of the lighting compute programs whose direct
/// dispatches are sinks (tiled_lighting:CS_*, deferred_tiled:
/// CS_DepthProcessingHBO, shadowmask:CS_GenerateShadowMasks*,
/// deferred_lighting:LightMix*, cascadeshadows_classify:CS_*; 37) [V
/// shader_index.py --sql over those names].
const LIGHT_PROGRAMS: [u128; 37] = [
    0xc00fa2557d58746b534656c88d7d9785, 0x4e8ed645147aae8799a8ec167d494032, 0x9842030d45bdb4bf8e319cead44feb35, 0x5ecc2838d44e490cbff64d18e5f92270,
    0xc84e6b41a9c8a05eee7a02ce04e947bc, 0x68f897e057538c772cfd753355475d5d, 0xdcc4ead61fe7a2c53545c1a9abba4cf6, 0x3a237fdcfbc729580a4b05d125682cd1,
    0x57d4d97691c8cf3bffaa3e28dc55522e, 0xc366d334fa2053647518e4cda50433ec, 0xc4b7599307477e90a7642eb3f3b6f125, 0x44bbf9cf6a140a48223e950cc66dbbc6,
    0xb81a6353023964a9200aec831a1ad1d4, 0xaccf00849ee4e7fdf2cb1d41f6cf3beb, 0x4ba45354ba99fe030e3d099c665fe562, 0x470a7f7e7d42318daeea2a617194cfb8,
    0xb8d6d00fcabe063bbd0463690c30b05e, 0x6b7ed85e6c8498f93591c172950234e2, 0xb8441dc954c40af02b931382802be151, 0x435922e10a429005de408add8a09d91f,
    0x47dd89d3bda95101c7ab5c358dc2d2ad, 0x69d79920225215375ce740a10bd0dbb4, 0xe7c7dce7e36c72e9705677bb45f10f5e, 0x02a8bcb8465f679b034f42fe3af13b28,
    0x0574e3dbddca772962c10ca83440041c, 0x59fb3a2411fcbf3f4bd9f812d10ee07e, 0x934a052e29f9e6a2ceb3fa7042654471, 0xbb5a60f92686f4586ab377a7706a4f92,
    0x9d1a95d8fa8a56cc38afe012bbe4c1d9, 0xc0492072c633349e6b6c3a32138604e1, 0xc9274a319595fa91a8fbf29d82837c83, 0x0b7b2bb725d0d3d01065bb5c85c01b96,
    0xca502cf7d0fa3b6a79e2ea138f1a02dc, 0x66db7fad04bdbc6a0703d44606f72b11, 0x77f0da87dcd6c9748b8af8f8e5a974fe, 0xabd4e9cef24dc129c7dad88d91e9b257,
    0x54f0245c4de701ba5560e7e7bfed0e11,
];
/// Container hashes of the sky and ambient producers (lit4 #914), sinks
/// as LIGHT_PROGRAMS: sky_scattering CS_Transmittance, CS_MultipleScattering,
/// CS_SkyView, CS_AerialPerspectiveVolumes; sky_system
/// CS_DirectionalLightTransmittanceLUT, CS_AmbientCubeIBR_Paraboloid,
/// _CubeMap, CS_AmbientCubeIBRLightning; sky_dome CS_SkyMain; ambient_map
/// CS_AmbientMapClassifyClear, CS_AmbientMapClassify16_32, CS_AmbientMap,
/// CS_LightMap, CS_LightMap_NoAmbient, CS_LightMapPropagate;
/// scattering_transmittance_system CS_SkyAtmosphereLUT (16) [V
/// shader_index.py --sql over the hash prefixes].
const SKY_PROGRAMS: [u128; 16] = [
    0xe6d7475ab43f1fb9f8582daad4f5ec6b, 0x8c5f5cfdaeebe66ebbad6fb4a3d511d8, 0x767996c1ce51eace99da7699f9e81c52, 0x956244a7319fa673f9ce39465cf97b68,
    0x834488728ebd10f67b1c05206400c6fe, 0x9ae7b3a323b1ab2aeb8bba7ee722c413, 0x4a5362fec5c3ac24f992e9da651e0e4d, 0x40e25e2154e4a3db040ba207f8a6c8bb,
    0x560ac0aae322671d71d20f0b5cdf45de, 0x2649828536e40445ba307e6b0ac488b4, 0xd470f982b9cca53a406198907d5cf904, 0x0e3d7b2c8405da358abc92fe2a616aaf,
    0xe1034c21b824e1dcf3f27414123918ef, 0x5ce5d87ea2576dc3a9cf4de7bc1e8a4b, 0xd0c9606baabbbd6528c4f0f2faace6ac, 0xfe2a19b40be2b479b2c2306be64d1a98,
];
/// Container hashes of the head compute graph (`ped_standard_head_cs:*`,
/// `compute_graph_ops:*`; 73) [V shader_index.py --sql]: it fills the
/// head position table (frame-table slot 7) and the face table on the GPU;
/// their direct dispatches are sinks with roots and tables (as the hair inputs).
const HEAD_PROGRAMS: [u128; 73] = [
    0x8a152901b7b1726392a584e82f136fd0, 0x13d513909a2f1a72186e75c6c7bf3f0a, 0x12252386bfe8f96e5e18167baea05dda, 0x51fbf6c90507ff9fdc840c85ac4049e5,
    0x4cf89afaf51ad48071155dec3637414a, 0x41fce774344b3cb36865d1832e3748ec, 0xd3585b1a359d6798a11e2a38595f4246, 0xa122cdedf4db8a1e038878ab49142304,
    0xa459fee2b760bf7979af7bf4beaea159, 0x916ca9c2f439ebbe97e113129e8f3e06, 0xee19277b8ec6ec33cbff6e04173f25f3, 0x92acb70da48b7ee80c73dbf5b804f48c,
    0xb086e6844125481b73669012189819a7, 0x3f6b978c193c3a36ac2e6b2e1632e329, 0xc31c5129ea3cc14c38a3ec8e684cba42, 0xfa91d49abe491a30d44e9d45a94aef75,
    0xf38a016a1d4963531db4c828b5aade01, 0x307c3ab71cd9c3933f891fb9cdb6443a, 0x30f8542f55d5b72d8e94611bd0fbabd2, 0x879d8283f03652451ca3b41549149fb0,
    0xfc3a97fd4cd872c094789f8ebbeb8711, 0xd6158724033dc2291145c256655be76a, 0xbbb34d1b79288e28dc4a8441bb3fb0a2, 0x871fc8cfdd6ae317f5d1d0e1a06858e8,
    0x21b86188ae6c19fd00f950ac1b376d28, 0x0c4914bf1bca8ad8dc9a15b41e1aab2e, 0xb9dbdca5b78dfd246929e453ff1d659f, 0x9e37c27e7f6e368e4e9cee402798d5aa,
    0x60c1dd7eff24fc8f87fc2f0211375c65, 0xd4d5e450bdabc175d37c2c34389d4a69, 0xe423bbae8c60cb6d6aafbcf0c773433f, 0xe06062bf687a2e8ac4e83ad238c97361,
    0x80cb6b3ba3ce13b23e119f2fbe7d8413, 0xe223e9166fed62bbd66872a624231037, 0x1a230ea6b4b18a893f06e6e57bc85dbd, 0x66e3445e2760c97260d1b48e26b7f6ac,
    0xa9e9903aa7fd91eeee4966d143eaf410, 0xbd8c2726d131199595bc3f01dacdb85f, 0x868a86a1956b50ae0d94a43a09459398, 0x0d7e975c62732b9af9068dac95a8df58,
    0x83bd0fcc205995e840685806122de4d3, 0xb04ed27cbe6728dc67deb2896230eef4, 0xd38a37b01030e442baee3b3507170668, 0x52e7c4fcedd8efc6e2d06ed9cc1fa6b0,
    0x4c8614d257f7a013e3684ce127eed791, 0x5e96b26dc803a3c270c44ee9cc5aee8c, 0x5a23f6063b93cc448b7474c8cb1d09ec, 0x70267aac6d3d57ea75d5106194b76d02,
    0xe941d98bb231e10f467b692a9e82d190, 0xd4cca9c3b0f252d64c1cd9224d4b51d2, 0xc0013d2a2d27ec5ff2c46a1bea35ceaf, 0x89532824626840dbb9f55b4e71554140,
    0x51c6ce4a28883d4442c01a21c07d67db, 0xf3d10c4c70ee18fcf424f36210c6a924, 0x96ad89b92b918914ff3fc93f6ceeb63c, 0x4ec54b2292375af29ff0d076d4376662,
    0xffa5558edc63b5881a70819aeff3c8ee, 0xe768619bedc51843048b506cf8a1dda4, 0xbc3b634edf40718240dcd6fba9131af9, 0x815c8827de54ea96eb481ebe52dace4e,
    0x168936d7fea333aa8d041c7e2f231315, 0x1edc98abdc3f0db0ea1eb18835d2ffc4, 0x140f77ac62b70f8608bb8f40eef84203, 0xe1db80d026380023495c7ab6e583e73e,
    0x073fb412a4fd84e3bbdaec4429c79b1c, 0x59e8bcb7549175ae42b143d4598446d3, 0xca779b55e3bfe2058d50bee5583141f7, 0xfadf8deffbb8e7a87213f6db99ee7efb,
    0x220c0303b3f1ccdf930f33abbb4eaa3d, 0x818485c2df101acc425f1b1a53f6fe17, 0x0a66bfdbce258f90e78db4915848f171, 0x7f8fa0391982e6c7e7eb36d4843faed3,
    0x0b542c4f4837d965b09fada63177f82c,
];
/// A direct dispatch's pipeline: not a sink, a pre-skin program, a
/// strand-hair input program, a strand-hair program kept for order only.
const PIPE_NONE: u8 = 0;
const PIPE_PED: u8 = 1;
const PIPE_HAIR: u8 = 2;
const PIPE_HAIR_ORDER: u8 = 3;
const PIPE_LIGHT: u8 = 4;
const PIPE_HEAD: u8 = 5;
/// A sky or ambient producer (SKY_PROGRAMS): recorded as PIPE_LIGHT.
const PIPE_SKY: u8 = 6;
/// Slots a hair input program's table root is resolved to (root 0 slot
/// 36 = the strand pool).
const HAIR_SLOTS: u32 = 40;
/// Strand group records looked at per record buffer.
const STRAND_MAX: usize = 512;

/// Which of the dispatch kinds a compute program (container hash) is.
fn pipe_class(hash: u128) -> u8 {
    if PED_SKINNING.contains(&hash) {
        PIPE_PED
    } else if HAIR_INPUTS.contains(&hash) {
        PIPE_HAIR
    } else if HAIR_ORDER.contains(&hash) {
        PIPE_HAIR_ORDER
    } else if LIGHT_PROGRAMS.contains(&hash) {
        PIPE_LIGHT
    } else if HEAD_PROGRAMS.contains(&hash) {
        PIPE_HEAD
    } else if SKY_PROGRAMS.contains(&hash) {
        PIPE_SKY
    } else {
        PIPE_NONE
    }
}
/// The pre-skin item lists, qword_1478A9A50[13 * list + bucket],
/// lists 0-2 ([V 0x144E3CDD0]); each head a chain of 144-B items
/// linked by their first qword.
const ITEM_HEADS: u64 = 39;
/// Items one walk reads at most (a chain that loops ends here).
const ITEMS_MAX: usize = 4096;
/// Walks of the item lists per frame at most (one per dispatch whose item
/// is not among those read so far: lists 1/2 are built after list 0).
const WALKS_MAX: u32 = 64;

static ON: AtomicBool = AtomicBool::new(false);
static SEQ: AtomicU32 = AtomicU32::new(0);
static THREADS: AtomicU32 = AtomicU32::new(0);
/// One past the last frame sent to the writer (0: none); sinks of a sent
/// or older frame are late and not recorded.
static SENT: AtomicU64 = AtomicU64::new(0);
/// Whether a frame's data will be written (the probe's rule; unset: all).
static WANT: AtomicUsize = AtomicUsize::new(0);
/// Counters logged at the next `take`: open frames skipped over OPEN_MAX,
/// late sink/hook records, wanted frames dropped unsent.
static SKIPPED: AtomicU64 = AtomicU64::new(0);
static LATE: AtomicU64 = AtomicU64::new(0);
/// The pre-skin item list heads (real address; 0: unknown).
static ITEM_LISTS: AtomicU64 = AtomicU64::new(0);
/// Pipeline object -> its compute program is one of PED_SKINNING.
static DISPATCH_PIPES: Mutex<Option<HashMap<usize, u8>>> = Mutex::new(None);

thread_local! {
    /// This thread's last hook draw: (frame, sequence number; 0 none recorded).
    static HOOK: std::cell::Cell<(u64, u32)> = const { std::cell::Cell::new((u64::MAX, 0)) };
    static THREAD: u32 = THREADS.fetch_add(1, Ordering::Relaxed);
}

/// Registers the sink tap (once, when `pack=2`).
pub(super) fn enable() {
    if !ON.swap(true, Ordering::Relaxed) {
        tap::set_tap(Some(on_sink));
        tap::set_dispatch_want(Some(want_dispatch));
        xnw_core::info!("world probe: frame pack v2 on (list sink tap)");
        xnw_core::info!("world probe: pack: ped pre-skin dispatch sink on ({} programs of ped_skinning_cs; item lists at {:#x})", PED_SKINNING.len(), ITEM_LISTS.load(Ordering::Relaxed));
    }
}

/// Where the title keeps the pre-skin item list heads (0x1478A9A50, real address).
pub(super) fn set_preskin_lists(heads: u64) {
    ITEM_LISTS.store(heads, Ordering::Relaxed);
}

/// tap.rs `DispatchWant`: a direct dispatch on a title thread:
/// counted for its frame; a sink when the frame is recorded and the
/// pipeline's compute program is one of PED_SKINNING.
fn want_dispatch(pipeline: usize) -> bool {
    want_dispatch_at(super::frame_key(), pipeline, |p| tap::stages(p).iter().filter(|s| s.name == "compute").map(|s| pipe_class(s.hash)).max().unwrap_or(PIPE_NONE))
}

fn want_dispatch_at(frame: u64, pipeline: usize, class: impl Fn(usize) -> u8) -> bool {
    {
        let mut accs = lock(&ACCS);
        let Some(acc) = acc_of(&mut accs, frame) else { return false };
        acc.dispatches_seen += 1;
    }
    *lock(&DISPATCH_PIPES).get_or_insert_with(HashMap::new).entry(pipeline).or_insert_with(|| class(pipeline)) != PIPE_NONE
}

/// A dispatch sink's class (`want_dispatch`); a pipeline never asked about
/// is a pre-skin one (the tests' sinks).
fn class_of(pipeline: usize) -> u8 {
    lock(&DISPATCH_PIPES).as_ref().and_then(|m| m.get(&pipeline).copied()).unwrap_or(PIPE_PED)
}

/// Sets the rule naming the frames whose data the writer will write (others
/// are not recorded: their sinks cost nothing).
pub(super) fn set_want(f: fn(u64) -> bool) {
    WANT.store(f as usize, Ordering::Relaxed);
}

fn wanted(frame: u64) -> bool {
    match WANT.load(Ordering::Relaxed) {
        0 => true,
        f => {
            // SAFETY: only `set_want` stores, a `fn(u64) -> bool`.
            let f = unsafe { std::mem::transmute::<usize, fn(u64) -> bool>(f) };
            f(frame)
        }
    }
}

pub(super) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// At the hook: a new draw on this thread in `frame`; its sequence number
/// (what the recorded draw keeps as `pack_seq`).
pub(super) fn hook_entry(frame: u64) -> u32 {
    if !on() {
        return 0;
    }
    let seq = SEQ.fetch_add(1, Ordering::Relaxed).wrapping_add(1).max(1);
    HOOK.with(|h| h.set((frame, seq)));
    seq
}

/// Skinned draws (palette count > 0) at the hook: the palette spans the
/// draw reads (TLS+0x11E8 current, +0x11F0 previous; 48 B per bone, at
/// most 512) and the model head (0x40 B, +0x33 the rigid bone byte),
/// as hook-side records of draw `seq` (owner HOOK | v1 draw); v2.1:
/// the draw handler's head, h = entity+0x28 [V], 0x400 B
/// (h+0x37C bit 0x40: palette path; +0x190/+0x1B0 palettes, +0x220 mask).
pub(super) fn hook_objects(frame: u64, seq: u32, palette: (u64, u32), previous: u64, model: u64, entity: u64) {
    if !on() || seq == 0 || palette.1 == 0 {
        return;
    }
    let bones = u64::from(palette.1.min(512));
    let mut accs = lock(&ACCS);
    let Some(acc) = acc_of(&mut accs, frame) else { return };
    let mut run_g = lock(&RUN);
    let run = run_g.get_or_insert_with(Run::default);
    let owner = fp::HOOK | seq;
    for (slot, at) in [(0u16, palette.0), (1, previous)] {
        let got = read(at, 48 * bones).map(|b| keep(acc, run, kind::PALETTE, b));
        acc.refs.push(reference(owner, kind::PALETTE, slot, at, 48 * bones, got));
    }
    let got = read(model, 0x40).map(|b| keep(acc, run, kind::MODEL, b));
    acc.refs.push(reference(owner, kind::MODEL, 0, model, 0x40, got));
    let handler = read(entity.wrapping_add(0x28), 8).map(|b| u64::from_le_bytes(b[..8].try_into().unwrap_or([0; 8])));
    let h = *handler.as_ref().unwrap_or(&0);
    let got = handler.and_then(|h| read(h, HANDLER_BYTES)).map(|b| keep(acc, run, kind::HANDLER, b));
    acc.refs.push(reference(owner, kind::HANDLER, 0, h, HANDLER_BYTES, got));
}

/// v2.1: one call of the imposter submit 0x1451D0AC0, as a
/// frame-scope IMPOSTER record (slot = call in frame, addr = IcRec). Blob:
/// 12 u64 (type ecx, stage edx, r8, r9 = h+0x308, a5 = h+0x228 count,
/// a7 strip, IcRec, desc = *IcRec, n = u32 desc+8, records = *(desc+0x78),
/// T = arena + u32 *(*(IcRec+0x60)+8), flags: bit0 IcRec, 1 desc, 2 T,
/// 3 records read), then IcRec 0x100 B, desc 0x100 B, T 0x90 B, records
/// 64 B x min(n, 4096) (a section that failed reads as zeros). The type
/// record R = 0x145248630(type) and its atlas ids need a call into title
/// code: not taken.
pub(super) fn imposter(frame: u64, args: [u64; 6], icrec: u64) {
    if !on() {
        return;
    }
    let mut accs = lock(&ACCS);
    let Some(acc) = acc_of(&mut accs, frame) else { return };
    let calls = acc.imposters;
    acc.imposters += 1;
    if calls >= u32::from(u16::MAX) {
        return;
    }
    let mut run_g = lock(&RUN);
    let run = run_g.get_or_insert_with(Run::default);
    let base = *acc.arena.get_or_insert_with(arena_base);
    let (blob, inst) = imposter_read(args, icrec, base, true);
    let len = blob.len() as u64;
    let h = keep(acc, run, kind::IMPOSTER, blob);
    acc.refs.push(reference(fp::FRAME_SCOPE, kind::IMPOSTER, calls as u16, icrec, len, Ok(h)));
    acc.imposter_parts[0] += u32::from(inst.reasons[fp::IMP_RECORDS] == reason::OK && !inst.records.is_empty());
    acc.imposter_parts[1] += u32::from(inst.reasons[fp::IMP_TYPE] == reason::OK);
    let list = inst.list;
    let bytes = inst.encode();
    let len = bytes.len() as u64;
    let h = keep(acc, run, kind::IMP_INST, bytes);
    acc.refs.push(reference(fp::FRAME_SCOPE, kind::IMP_INST, calls as u16, list, len, Ok(h)));
}

/// The title's per-frame arena base (0: unknown or unreadable).
pub(super) fn arena_base() -> u64 {
    read(ARENA_AT.load(Ordering::Relaxed), 8).map_or(0, |b| u64::from_le_bytes(b[..8].try_into().unwrap_or([0; 8])))
}

/// One imposter submit call read from title memory: its kind-42 blob
/// (layout at `imposter`) and its kind-47 instances. `arena` is the frame's
/// arena base (T lives there). The frame pack writer and the live pass
/// (`worldprobe::imposter`) both read through here. `full`: also the
/// parts the card pass does not use (the visible list, the indirect
/// arguments, the handler's 0x400 B at r9 - 0x308); the live pass passes
/// false: a capture's 793 imposter faults sat 0x88-0x308 below a 64 KiB
/// boundary (0x2eb8efcf8 = 0x2eb8f0000 - 0x308), as `r9 - 0x308` does when r9
/// is the first bytes of a fresh allocation (not h + 0x308 at every caller) [I].
pub(super) fn imposter_read(args: [u64; 6], icrec: u64, arena: u64, full: bool) -> (Vec<u8>, fp::ImpInstances) {
    let u64_at = |at: u64| read(at, 8).ok().map(|b| u64::from_le_bytes(b[..8].try_into().unwrap_or([0; 8])));
    let ic = read(icrec, 0x100).ok();
    let desc = u64_at(icrec).unwrap_or(0);
    let dh = read(desc, 0x100).ok();
    let n = dh.as_ref().map_or(0, |d| u32::from_le_bytes([d[8], d[9], d[10], d[11]]));
    let recs_at = dh.as_ref().map_or(0, |d| u64::from_le_bytes(d[0x78..0x80].try_into().unwrap_or([0; 8])));
    let t_at = ic.as_ref().map(|c| u64::from_le_bytes(c[0x60..0x68].try_into().unwrap_or([0; 8]))).and_then(|p| read(p + 8, 4).ok()).map_or(0, |b| arena + u64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])));
    let t = if arena != 0 && t_at != 0 { read(t_at, 0x90).ok() } else { None };
    let recs = read(recs_at, 64 * u64::from(n.min(4096))).ok();
    let flags = u64::from(ic.is_some()) | u64::from(dh.is_some()) << 1 | u64::from(t.is_some()) << 2 | u64::from(recs.is_some()) << 3;
    let mut blob = Vec::with_capacity(96 + 0x290 + recs.as_ref().map_or(0, Vec::len));
    for w in [args[0], args[1], args[2], args[3], args[4], args[5], icrec, desc, u64::from(n), recs_at, t_at, flags] {
        blob.extend_from_slice(&w.to_le_bytes());
    }
    for (part, len) in [(&ic, 0x100usize), (&dh, 0x100), (&t, 0x90)] {
        match part {
            Some(b) => blob.extend_from_slice(b),
            None => blob.resize(blob.len() + len, 0),
        }
    }
    if let Some(r) = &recs {
        blob.extend_from_slice(r);
    }
    let inst = imposter_instances(ic.as_deref(), dh.as_deref(), n, args[3].wrapping_sub(0x308), args[0] as u32, full);
    (blob, inst)
}

/// v2.3: an imposter call's instances: L = *(desc+0) and its 64-B records
/// (L+0x90, u16 count L+0x98; desc+0x78 is 0 for these GPU-buffer
/// batches), the visible words at IcRec+0x48 (u32 x IcRec+0x50), the
/// indirect arguments behind IcRec+0x58 and the batch handler's head (`h`
/// = r9 - 0x308) and the type record of `ty` (`imposter_type`). Every
/// part reads to bytes or a reason.
fn imposter_instances(ic: Option<&[u8]>, desc: Option<&[u8]>, n: u32, h: u64, ty: u32, full: bool) -> fp::ImpInstances {
    let q = |b: Option<&[u8]>, at: usize| b.and_then(|b| b.get(at..at + 8)).map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap_or([0; 8])));
    let reason_of = |r: &Result<Vec<u8>, u32>| r.as_ref().err().copied().unwrap_or(reason::OK);
    let list = q(desc, 0);
    let head = read(list, fp::IMP_HEAD_BYTES as u64);
    let lh = head.as_ref().ok().map(Vec::as_slice);
    let records_at = q(lh, 0x90);
    let count = lh.and_then(|h| h.get(0x98..0x9A)).map_or(0, |c| u32::from(u16::from_le_bytes([c[0], c[1]])));
    let records = match &head {
        Err(r) => Err(*r),
        Ok(_) if count == 0 => Ok(Vec::new()),
        Ok(_) => read(records_at, (fp::IMP_RECORD_BYTES * (count as usize).min(fp::IMP_MAX)) as u64),
    };
    let visible_at = q(ic, 0x48);
    let visible_n = ic.and_then(|c| c.get(0x50..0x54)).map_or(0, |c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    let vis = (visible_n as usize).min(fp::IMP_MAX);
    let unread = Err(reason::NOT_CAPTURED);
    let visible = if !full { unread.clone() } else if vis == 0 && visible_at != 0 { Ok(Vec::new()) } else { read(visible_at, 4 * vis as u64) };
    let args_at = q(ic, 0x58);
    let args = if full { read(args_at, fp::IMP_ARGS_BYTES as u64) } else { unread.clone() };
    let handler = if full { read(h, fp::IMP_HANDLER_BYTES as u64) } else { unread };
    let (type_at, type_record) = imposter_type(ty);
    fp::ImpInstances {
        list,
        records_at,
        visible_at,
        args_at,
        handler_at: h,
        type_at,
        count,
        n,
        visible_n,
        reasons: [reason_of(&head), reason_of(&args), reason_of(&visible), reason_of(&records), reason_of(&handler), reason_of(&type_record)],
        head: head.unwrap_or_default(),
        args: args.unwrap_or_default(),
        handler: handler.unwrap_or_default(),
        type_record: type_record.unwrap_or_default(),
        visible: visible.unwrap_or_default().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        records: records.unwrap_or_default(),
    }
}

/// The imposter type table's globals (real address of 0x1476765A8: the
/// records' base, then u32 group count at +8, the groups at +0x10; 0: off).
static IMP_TYPES: AtomicU64 = AtomicU64::new(0);

pub(super) fn set_imposter_types(at: u64) {
    IMP_TYPES.store(at, Ordering::Relaxed);
}

/// Groups probed at most per lookup.
const TYPE_PROBES: u32 = 256;

/// The type record R of imposter type `ty` as the title's lookup
/// 0x145248630 finds it, in plain memory: key = ty & 0x3FFFFF; 144-B groups
/// of 16 control bytes and 16 slots {u32 key, u32 index}; group (key >> 7)
/// & (count - 1) on; a slot matches when its control byte is key & 0x7F and
/// its key is key; a group with an empty control byte (sign bit) ends the
/// probe. R = base + 224 * index. (address, 224 B or a reason; NULL: no
/// entry, as the title's 0.)
fn imposter_type(ty: u32) -> (u64, Result<Vec<u8>, u32>) {
    let g = match read(IMP_TYPES.load(Ordering::Relaxed), 0x18) {
        Ok(g) => g,
        Err(why) => return (0, Err(why)),
    };
    let q = |at: usize| u64::from_le_bytes(g[at..at + 8].try_into().unwrap_or([0; 8]));
    let (base, count, groups) = (q(0), u32::from_le_bytes([g[8], g[9], g[10], g[11]]), q(0x10));
    if count == 0 || !count.is_power_of_two() {
        return (0, Err(if count == 0 { reason::NULL } else { reason::IMPLAUSIBLE }));
    }
    let key = match ty & 0x3F_FFFF {
        0x3F_FFFF => u32::MAX,
        k => k,
    };
    let ctrl = (key & 0x7F) as u8;
    let mut at = (key >> 7) & (count - 1);
    for _ in 0..count.min(TYPE_PROBES) {
        let grp = match read(groups.wrapping_add(144 * u64::from(at)), 144) {
            Ok(b) => b,
            Err(why) => return (0, Err(why)),
        };
        for i in 0..16 {
            let w = |o: usize| u32::from_le_bytes([grp[o], grp[o + 1], grp[o + 2], grp[o + 3]]);
            if grp[i] == ctrl && key != u32::MAX && w(16 + 8 * i) & 0x3F_FFFF == key {
                let r = base.wrapping_add(224 * u64::from(w(20 + 8 * i)));
                return (r, read(r, fp::IMP_TYPE_BYTES as u64));
            }
        }
        if grp[..16].iter().any(|c| c & 0x80 != 0) {
            return (0, Err(reason::NULL));
        }
        at = (at + 1) & (count - 1);
    }
    (0, Err(reason::BUDGET))
}

/// Bytes of a draw handler's head kept per skinned draw.
const HANDLER_BYTES: u64 = 0x400;

/// At the hook: the descriptor handles a draw names (albedo view+0x18,
/// layer diffuse slots), each resolved once per frame as a frame-scope
/// HANDLE_VIEW record (addr = handle), its texture noted. In one capture the albedo
/// handles were view-heap slots no table or bindless record named, so the
/// replay found 0 of 705. Also the draw's vertex and meshlet streams
/// (address, bytes), for the root views over them (`root_view_at`).
pub(super) fn hook_views(frame: u64, handles: &[u64], streams: &[(u64, u64)]) {
    if !on() || (handles.iter().all(|h| *h == 0) && streams.iter().all(|s| s.0 == 0 || s.1 == 0)) {
        return;
    }
    let mut accs = lock(&ACCS);
    let Some(acc) = acc_of(&mut accs, frame) else { return };
    for &(at, len) in streams {
        if at != 0 && len != 0 {
            let e = acc.streams.entry(at).or_insert(0);
            *e = (*e).max(len);
        }
    }
    let mut run_g = lock(&RUN);
    let run = run_g.get_or_insert_with(Run::default);
    for &h in handles {
        if h == 0 || !acc.handles.insert(h) {
            continue;
        }
        // The slot is read in place by the view lookup: readable first.
        if let Err(why) = read(h, tap::SLOT) {
            acc.refs.push(reference(fp::FRAME_SCOPE, kind::HANDLE_VIEW, 0, h, 0, Err(why)));
            continue;
        }
        let v = slot_view(0, &tap::view(h));
        if v.kind == view::NONE {
            acc.refs.push(reference(fp::FRAME_SCOPE, kind::HANDLE_VIEW, 0, h, 0, Err(reason::NOT_RESIDENT)));
            continue;
        }
        if v.kind == view::TEXTURE || v.kind == view::PLACED {
            texture(acc, run, &v);
        }
        let got = keep(acc, run, kind::HANDLE_VIEW, fp::encode_slots(&[v]));
        acc.refs.push(reference(fp::FRAME_SCOPE, kind::HANDLE_VIEW, 0, h, fp::SLOT_BYTES as u64, Ok(got)));
    }
}

/// Frame `frame`'s data while it is recorded; None when the frame is not
/// wanted or was already sent (late: counted).
fn acc_of(accs: &mut Vec<Acc>, frame: u64) -> Option<&mut Acc> {
    if frame < SENT.load(Ordering::Relaxed) {
        LATE.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let at = match accs.iter().position(|a| a.frame == frame) {
        Some(i) => i,
        None => {
            if accs.len() >= OPEN_MAX {
                let old = accs.remove(0);
                SKIPPED.fetch_add(u64::from(old.wanted), Ordering::Relaxed);
            }
            accs.push(Acc { frame, wanted: wanted(frame), ..Acc::default() });
            accs.len() - 1
        }
    };
    let a = &mut accs[at];
    a.wanted.then_some(a)
}

/// One sink draw before the writer maps its hook draw.
struct SinkRaw {
    thread: u32,
    hook: u32,
    op: u8,
    count: u32,
    pipeline: u64,
}

/// One frame's pack as the sinks fill it.
#[derive(Default)]
struct Acc {
    frame: u64,
    /// Recorded (the writer will want it); else a marker that records nothing.
    wanted: bool,
    sinks: Vec<SinkRaw>,
    refs: Vec<Ref>,
    /// Bytes first met in this frame (written at the frame's end).
    blobs: HashMap<u128, (u16, Vec<u8>)>,
    pipelines: HashSet<u64>,
    programs: Vec<fp::Program>,
    /// By (texel address, bytes).
    textures: HashMap<(u64, u64), fp::Texture>,
    texel_bytes: u64,
    buffer_bytes: u64,
    /// Buffer ranges met this frame -> their ref outcome.
    buffers: HashMap<(u64, u64), Result<u128, u32>>,
    /// Table content (raw records) seen this frame -> its refs, reused.
    tables: HashMap<u128, Vec<Ref>>,
    bindless: HashSet<u128>,
    /// Descriptor handles the hook draws named (albedo, layers), resolved once.
    handles: HashSet<u64>,
    /// The frame's arena base (read at the first record that needs it).
    arena: Option<u64>,
    arena_noted: bool,
    /// Imposter submit calls this frame.
    imposters: u32,
    /// Imposter calls with instance records, with their type record.
    imposter_parts: [u32; 2],
    /// Pipelines of the instanced wheel programs -> (base slot, meshlet slot).
    wheels: HashMap<u64, (u32, u32)>,
    /// Wheel palette windows with bytes, without.
    wheel_windows: [u32; 2],
    /// Direct dispatches the title made this frame (any program).
    dispatches_seen: u32,
    /// The pre-skin items read so far (address, 144 B), and the walks made.
    items: Vec<(u64, Vec<u8>)>,
    item_walks: u32,
    /// Time spent recording dispatch sinks.
    dispatch_ns: u64,
    /// Draw words: sinks with the 64 bytes, without, views resolved.
    draw_words: [u32; 3],
    /// Sampler heap records by heap object (one read per frame per heap).
    sampler_heaps: HashMap<usize, Ref>,
    /// Sampler records: entries kept (per heap read), sinks without a heap.
    sampler_counts: [u32; 2],
    /// Root views read through the CPU alias, without bytes, over a hook
    /// draw's stream, over a window ([`ROOT_WINDOW`]).
    alias: [u32; 4],
    /// The hook draws' vertex and meshlet streams this frame: address ->
    /// bytes. The title keeps them in its own pages, in no buffer object.
    streams: BTreeMap<u64, u64>,
    /// Pipelines of the displacing pre-skin programs.
    displaced: HashSet<u64>,
    /// Draw and DrawIndexed sinks with their DRAW_ARGS.
    draw_args: u32,
    /// Hair: input dispatch sinks, order-only sinks, strand group records,
    /// their heap views with bytes, without.
    hair: [u32; 5],
    /// Strand record buffers met this frame (address and content).
    strand_seen: HashSet<u128>,
    /// Lighting dispatch sinks; the scene lights copied (their count; MAX unread).
    light_sinks: u32,
    /// Head compute graph dispatch sinks.
    head_sinks: u32,
    /// Of the lighting dispatch sinks, sky and ambient producers; indirect
    /// dispatch sinks (with the compute roots).
    sky_sinks: u32,
    indirect_dispatches: u32,
    /// Head graph plain unordered buffer views read, not read (per sink view).
    head_uavs: [u32; 2],
    /// Strand palette windows with bytes, without; collision spans with
    /// bytes, without (count 0 included).
    strand_extra: [u32; 4],
    lights_done: bool,
    lights_copied: u32,
    /// HOOKIN records this frame (seq, geometry, blob: refs made by the
    /// writer); HOOK_GLOBALS written.
    hookin: u32,
    hookin_raw: Vec<(u32, u64, Vec<u8>)>,
    hook_globals_done: bool,
    /// Displacement map records: with a view, without.
    disp_maps: [u32; 2],
}

/// One sent frame's pack data, moved into its capture (`take`).
pub(super) struct Taken(Acc);

/// Run-wide: blobs of frames sent to be written (committed at `take`; a
/// frame that is dropped commits nothing: one capture had 3,712 refs of frame 1002
/// whose blobs an earlier, unwritten frame had claimed), and resolved
/// tables and texels by content (their blobs re-queued when not committed).
#[derive(Default)]
struct Run {
    stored: HashSet<u128>,
    resolved: HashMap<u128, (u128, Vec<SlotView>)>,
    texels: HashMap<(u64, u64, u64), u128>,
}

static ACCS: Mutex<Vec<Acc>> = Mutex::new(Vec::new());
static RUN: Mutex<Option<Run>> = Mutex::new(None);
/// Blobs the writer has put on disk this run.
static WRITTEN: Mutex<Option<HashSet<u128>>> = Mutex::new(None);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Copies `len` bytes of title memory (fault-safe); a reason when not.
fn read(addr: u64, len: u64) -> Result<Vec<u8>, u32> {
    if addr == 0 {
        return Err(reason::NULL);
    }
    if len == 0 || !(0x1_0000..0x8000_0000_0000).contains(&addr) || addr.checked_add(len).is_none_or(|e| e > 0x8000_0000_0000) {
        return Err(reason::IMPLAUSIBLE);
    }
    super::dcopy(addr, len as usize).ok_or(reason::UNREADABLE)
}

/// Queues `bytes` (kind `k`) unless the run has them; their hash.
fn keep(acc: &mut Acc, run: &mut Run, k: u16, bytes: Vec<u8>) -> u128 {
    let h = xnw_core::hash128(&[&bytes]);
    if !run.stored.contains(&h) {
        acc.blobs.entry(h).or_insert((k, bytes));
    }
    h
}

/// Whether blob `h` is committed or queued in this frame.
fn held(acc: &Acc, run: &Run, h: u128) -> bool {
    run.stored.contains(&h) || acc.blobs.contains_key(&h)
}

fn reference(owner: u32, k: u16, slot: u16, addr: u64, len: u64, got: Result<u128, u32>) -> Ref {
    match got {
        Ok(hash) => Ref { owner, kind: k, slot, addr, len: len as u32, reason: reason::OK, hash },
        Err(r) => Ref { owner, kind: k, slot, addr, len: len as u32, reason: r, hash: 0 },
    }
}

/// As [`reference`]; a view longer than the cap keeps its first `len`
/// bytes with reason CUT (one capture: T8 slot 84, a 16 MB view, cut silently).
fn reference_cut(owner: u32, k: u16, slot: u16, addr: u64, len: u64, got: Result<u128, u32>, cut: bool) -> Ref {
    let mut r = reference(owner, k, slot, addr, len, got);
    if cut && r.reason == reason::OK {
        r.reason = reason::CUT;
    }
    r
}

fn slot_view(slot: u32, v: &TapView) -> SlotView {
    // A buffer's unordered access view says so in `layout` (unused by buffers).
    let layout = if v.unordered && v.kind == view::BUFFER { view::UNORDERED } else { v.layout };
    SlotView { slot, kind: v.kind, addr: v.addr, bytes: v.bytes, resource: v.resource, width: v.width, height: v.height, depth: v.depth, mips: v.mips, format: v.format, layout, words: v.words }
}

/// The list sink (title thread).
fn on_sink(d: &TapDraw) {
    sink_at(super::frame_key(), d);
}

fn sink_at(frame: u64, d: &TapDraw) {
    // An indirect dispatch carries the compute roots (draw.rs `tap`); it
    // is recorded as a lighting dispatch is (frame table to 192 slots).
    let indirect = d.op == fp::op::INDIRECT_DISPATCH;
    let compute = d.op == fp::op::DISPATCH || indirect;
    let class = if d.op == fp::op::DISPATCH { class_of(d.pipeline) } else { PIPE_NONE };
    let lit = matches!(class, PIPE_LIGHT | PIPE_SKY) || indirect;
    let t0 = compute.then(std::time::Instant::now);
    let (hf, hs) = HOOK.with(|h| h.get());
    // A dispatch is not a draw: never paired with this thread's hook draw.
    let hook = if hf == frame && !compute { hs } else { 0 };
    let thread = THREAD.with(|t| *t);
    let mut accs = lock(&ACCS);
    let Some(acc) = acc_of(&mut accs, frame) else { return };
    let mut run_g = lock(&RUN);
    let run = run_g.get_or_insert_with(Run::default);
    let owner = acc.sinks.len() as u32;
    acc.sinks.push(SinkRaw { thread, hook, op: d.op, count: d.count, pipeline: d.pipeline as u64 });
    if acc.pipelines.insert(d.pipeline as u64) {
        programs(acc, run, d.pipeline);
    }
    for (i, r) in d.roots.iter().enumerate() {
        let slot = i as u16;
        match *r {
            TapRoot::None => {}
            TapRoot::Constants(c) => {
                let bytes: Vec<u8> = c.iter().flat_map(|w| w.to_le_bytes()).collect();
                let len = bytes.len() as u64;
                let h = keep(acc, run, kind::ROOTS, bytes);
                acc.refs.push(reference(owner, kind::ROOTS, slot, 0, len, Ok(h)));
            }
            TapRoot::View(a) => {
                let (at, full, alias, how) = root_view_at(a, tap::buffer_left, |x| stream_left(&acc.streams, x));
                let len = full.min(BUF_CAP);
                let got = buffer(acc, run, at, len);
                acc.alias[0] += u32::from(alias);
                acc.alias[1] += u32::from(got.is_err());
                acc.alias[2] += u32::from(how == ROOT_STREAM);
                acc.alias[3] += u32::from(how == ROOT_WINDOWED);
                acc.refs.push(reference_cut(owner, kind::ROOT_VIEW, slot, at, len, got, full > len || how == ROOT_WINDOWED));
            }
            // A hair program kept for its order: its roots, not its tables.
            TapRoot::Table { .. } if class == PIPE_HAIR_ORDER => {}
            TapRoot::Table { handle, slots, .. } => {
                // The frame table (root 0) to tap::TABLE_SLOTS for draws and
                // lighting dispatches (the lighting reads its slots up to 181);
                // other tables to one row.
                let cap = match (compute && !lit, class, i) {
                    (false, _, 0) => tap::TABLE_SLOTS,
                    (false, _, _) => fp::TABLE_ROW,
                    (true, PIPE_HAIR | PIPE_HEAD, _) => HAIR_SLOTS,
                    _ => DISPATCH_SLOTS,
                };
                let mode = match (compute, class) {
                    (false, _) => TB_DRAW,
                    (true, PIPE_HEAD) => TB_HEAD,
                    _ => TB_DISPATCH,
                };
                table(acc, run, owner, slot, handle, slots.min(cap), d.view_heap, mode)
            }
        }
    }
    if !compute {
        let words = match d.roots.first() {
            Some(TapRoot::Table { handle, slots, .. }) if WORDS_SLOT < *slots => Some(slot_view(WORDS_SLOT, &tap::view(handle + tap::SLOT * u64::from(WORDS_SLOT)))),
            _ => None,
        };
        let off = match d.roots.get(WORDS_ROOT) {
            Some(TapRoot::Constants(c)) => c.get(WORDS_DWORD).copied(),
            _ => None,
        };
        let heap = d.view_heap;
        draw_words(acc, run, owner, words_at(words, off), &|w| tap::heap_slot(heap, w).map(|(start, h)| (start, slot_view(w, &tap::view(h)))));
    }
    samplers(acc, run, owner, d.sampler_heap, tap::sampler_heap(d.sampler_heap), &tap::sampler);
    if let Some(&(base_slot, meshlet_slot)) = acc.wheels.get(&(d.pipeline as u64)) {
        wheel_windows(acc, run, owner, d.count, wheel_views(d, base_slot, meshlet_slot));
    }
    if compute && acc.displaced.contains(&(d.pipeline as u64)) {
        let elems = match d.roots.first() {
            Some(TapRoot::Table { handle, slots, .. }) if MAP_SLOT < *slots => Some(slot_view(MAP_SLOT, &tap::view(handle + tap::SLOT * u64::from(MAP_SLOT)))),
            _ => None,
        };
        let heap = d.view_heap;
        displacement_maps(acc, run, owner, elems, &|i| tap::heap_slot(heap, i).and_then(|(_, h)| read(h, tap::SLOT).ok().map(|_| (h, slot_view(i, &tap::view(h))))));
    }
    if matches!(d.op, fp::op::DRAW | fp::op::INDEXED) {
        let bytes: Vec<u8> = d.args.iter().flat_map(|w| w.to_le_bytes()).collect();
        let h = keep(acc, run, kind::DRAW_ARGS, bytes);
        acc.refs.push(reference(owner, kind::DRAW_ARGS, 0, 0, fp::DRAW_ARGS_BYTES as u64, Ok(h)));
        acc.draw_args += 1;
    }
    acc.head_sinks += u32::from(class == PIPE_HEAD);
    acc.sky_sinks += u32::from(class == PIPE_SKY);
    acc.indirect_dispatches += u32::from(indirect);
    if d.op == fp::op::DISPATCH {
        let bytes: Vec<u8> = [d.count, d.yz[0], d.yz[1]].iter().flat_map(|w| w.to_le_bytes()).collect();
        let h = keep(acc, run, kind::GROUPS, bytes);
        acc.refs.push(reference(owner, kind::GROUPS, 0, 0, 12, Ok(h)));
    }
    if matches!(class, PIPE_LIGHT | PIPE_SKY) {
        acc.light_sinks += 1;
        if !acc.lights_done {
            acc.lights_done = true;
            scene_lights(acc, run);
        }
    }
    if class == PIPE_HAIR || class == PIPE_HAIR_ORDER {
        acc.hair[usize::from(class == PIPE_HAIR_ORDER)] += 1;
    }
    if class == PIPE_HAIR {
        let at = |slot: u32| match d.roots.first() {
            Some(TapRoot::Table { handle, slots, .. }) if slot < *slots => Some(slot_view(slot, &tap::view(handle + tap::SLOT * u64::from(slot)))),
            _ => None,
        };
        let heap = d.view_heap;
        let inputs = StrandInputs { records: at(STRAND_SLOT), palette: at(PALETTE_SLOT), shapes: at(SHAPES_SLOT) };
        strand_groups(acc, run, owner, inputs, &|i| tap::heap_slot(heap, i).and_then(|(_, h)| read(h, tap::SLOT).ok().map(|_| (h, slot_view(i, &tap::view(h))))));
    }
    if let Some(t0) = t0 {
        if class == PIPE_PED {
            dispatch_record(acc, run, owner, [d.count, d.yz[0], d.yz[1]], d.view_heap);
        }
        acc.dispatch_ns += t0.elapsed().as_nanos() as u64;
    }
}

/// The scene-light array (real addresses: array base, buffer index
/// dword, count dword; 0x1487BAF40, dword_148FDC4D0, dword_1488CD81C [V ]).
static LIGHTS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

pub(super) fn set_lights(array: u64, index: u64, count: u64) {
    for (a, v) in LIGHTS.iter().zip([array, index, count]) {
        a.store(v, Ordering::Relaxed);
    }
}

/// The frame's LIGHTS records: the current buffer of the CPU scene-light
/// array (before the GPU's copy reads it) and its index and count.
fn scene_lights(acc: &mut Acc, run: &mut Run) {
    let [array, index, count] = LIGHTS.each_ref().map(|a| a.load(Ordering::Relaxed));
    let word = |at: u64| read(at, 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let (i, n) = (word(index), word(count));
    let head = match (i, n) {
        (Ok(i), Ok(n)) => Ok(keep(acc, run, kind::LIGHTS, [i.to_le_bytes(), n.to_le_bytes()].concat())),
        (Err(e), _) | (_, Err(e)) => Err(e),
    };
    acc.refs.push(reference(fp::FRAME_SCOPE, kind::LIGHTS, 1, count, 8, head));
    let len = fp::LIGHT_ARRAY_BYTES as u64;
    let at = match i {
        Ok(i) if i < 4 && array != 0 => Ok(array + len * u64::from(i)),
        Ok(_) => Err(reason::IMPLAUSIBLE),
        Err(e) => Err(e),
    };
    let got = at.and_then(|a| read(a, len)).map(|b| keep(acc, run, kind::LIGHTS, b));
    acc.refs.push(reference(fp::FRAME_SCOPE, kind::LIGHTS, 0, *at.as_ref().unwrap_or(&0), len, got));
    acc.lights_copied = n.unwrap_or(u32::MAX);
}

/// `hookin=1`: the hook's inputs per hook draw (HOOKIN), and the globals
/// they need (real addresses: matrix slot word, matrix cache, palette base).
static HOOKIN: AtomicBool = AtomicBool::new(false);
static HOOK_GLOBALS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
/// Hook-in records made and the nanoseconds they took on the title's threads.
static HOOKIN_COST: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

pub(super) fn set_hookin(on: bool, matrix_slot: u64, matrix_cache: u64, pal_base: u64) {
    for (a, v) in HOOK_GLOBALS.iter().zip([matrix_slot, matrix_cache, pal_base]) {
        a.store(v, Ordering::Relaxed);
    }
    HOOKIN.store(on, Ordering::Relaxed);
}

/// The HOOKIN blob of one hook draw (layout: `fp::HOOKIN_PARTS`), read in
/// place with the fault-fixed copy into one buffer.
fn hookin_blob(tls: u64, geom: u64, globals: [u64; 3]) -> Vec<u8> {
    // 4 + 16 * 19 header bytes + 1,025 part bytes.
    let mut b = vec![0u8; 4 + 16 * fp::HOOKIN_PARTS + 1025];
    b[..4].copy_from_slice(&(fp::HOOKIN_PARTS as u32).to_le_bytes());
    let mut at = 4 + 16 * fp::HOOKIN_PARTS;
    let mut k = 0;
    let mut part = |b: &mut Vec<u8>, addr: u64, len: usize| -> Option<Vec<u8>> {
        if b.len() < at + len {
            b.resize(at + len, 0);
        }
        // SAFETY: into our buffer; the source read is fault-fixed.
        let ok = addr >= 0x1_0000 && unsafe { super::fcopy(b.as_mut_ptr().add(at), addr as *const u8, len) };
        let h = 4 + 16 * k;
        b[h..h + 8].copy_from_slice(&addr.to_le_bytes());
        b[h + 8..h + 12].copy_from_slice(&(len as u32).to_le_bytes());
        b[h + 12..h + 16].copy_from_slice(&u32::from(ok).to_le_bytes());
        let got = ok.then(|| b[at..at + len].to_vec());
        at += len;
        k += 1;
        got
    };
    let q = |v: &Option<Vec<u8>>, o: usize| v.as_ref().and_then(|v| v.get(o..o + 8)).map_or(0, |w| u64::from_le_bytes(w.try_into().unwrap_or([0; 8])));
    for (o, len) in [(0x950u64, 4usize), (0x1110, 4), (0x1294, 1)] {
        part(&mut b, tls + o, len);
    }
    let t = part(&mut b, tls + 0x11D8, 0xF0);
    let tq = |o: u64| q(&t, (o - 0x11D8) as usize);
    let g = part(&mut b, geom, 0x48);
    part(&mut b, q(&g, 8), 0x20);
    part(&mut b, q(&g, 0x10), 0x20);
    let m = q(&g, 0x40);
    let mut mp = [0u8; 8];
    // SAFETY: into our 8 bytes, fault-fixed.
    let m = if m >= 0x1_0000 && unsafe { super::fcopy(mp.as_mut_ptr(), m as *const u8, 8) } { u64::from_le_bytes(mp) } else { 0 };
    part(&mut b, m, 0x20);
    let ctx = tq(0x12A8);
    let mut sw = [0u8; 2];
    // SAFETY: as above.
    let slot = if globals[0] != 0 && unsafe { super::fcopy(sw.as_mut_ptr(), globals[0] as *const u8, 2) } { u64::from(u16::from_le_bytes(sw) >> 2) } else { 0 };
    let bound = part(&mut b, ctx + 0x40 + 8 * slot, 8);
    part(&mut b, ctx + 0x6200 + 2 * slot, 2);
    let index = part(&mut b, ctx + 0x1_2190, 4).map_or(0, |v| i32::from_le_bytes([v[0], v[1], v[2], v[3]]));
    part(&mut b, ctx + 0x1_2328, 8);
    part(&mut b, ctx + 0x1_2360, 0x1A);
    part(&mut b, q(&bound, 0), 0x80);
    part(&mut b, globals[1].wrapping_add((i64::from(index) * 128) as u64), 0x80);
    let vp = tq(0x12B8);
    part(&mut b, vp, 0xC0);
    part(&mut b, vp + 0x708, 8);
    let model = part(&mut b, tq(0x1208), 0x38);
    let mh = model.as_ref().map_or(0, |v| u64::from(v[0x33]));
    part(&mut b, tq(0x11E8).wrapping_add(48 * mh), 48);
    b.truncate(at);
    b
}

/// At the hook (`hookin=1`): hook draw `seq`'s HOOKIN record, and once per
/// frame the HOOK_GLOBALS record. Its cost on the title's thread is counted.
pub(super) fn hook_inputs(frame: u64, seq: u32, tls: u64, geom: u64) {
    if !on() || seq == 0 || tls == 0 || !HOOKIN.load(Ordering::Relaxed) {
        return;
    }
    let t0 = std::time::Instant::now();
    // Only frames the writer will write (the reads cost the title's thread).
    if !wanted(frame) {
        return;
    }
    let globals = HOOK_GLOBALS.each_ref().map(|a| a.load(Ordering::Relaxed));
    let blob = hookin_blob(tls, geom, globals);
    let mut accs = lock(&ACCS);
    if let Some(acc) = acc_of(&mut accs, frame) {
        let mut run_g = lock(&RUN);
        let run = run_g.get_or_insert_with(Run::default);
        if !acc.hook_globals_done {
            acc.hook_globals_done = true;
            let mut g = vec![0u8; 10];
            // SAFETY: into our buffer, fault-fixed.
            let ok = globals[0] != 0 && globals[2] != 0 && unsafe { super::fcopy(g.as_mut_ptr(), globals[0] as *const u8, 2) && super::fcopy(g.as_mut_ptr().add(2), globals[2] as *const u8, 8) };
            let got = if ok { Ok(keep(acc, run, kind::HOOK_GLOBALS, g)) } else { Err(reason::UNREADABLE) };
            acc.refs.push(reference(fp::FRAME_SCOPE, kind::HOOK_GLOBALS, 0, globals[0], 10, got));
        }
        // Hashed and kept by the writer, off the title's thread.
        acc.hookin_raw.push((seq, geom, blob));
        acc.hookin += 1;
    }
    drop(accs);
    HOOKIN_COST[0].fetch_add(1, Ordering::Relaxed);
    HOOKIN_COST[1].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// The strand group records of the hair programs: root 0 slot 21; the
/// bone palette slot 1; the collision shapes slot 22 (bc6245e7: T+0x20,
/// T+0x2a0, T+0x2c0) [V read].
const STRAND_SLOT: u32 = 21;
const PALETTE_SLOT: u32 = 1;
const SHAPES_SLOT: u32 = 22;
/// Group record words: the remap offset (elements), the palette bases
/// (current, previous), the collision span (count in the top byte of
/// +0xa8, byte offset +0xf8; bc6245e7 0x8b0-0x9a0) [V read].
const REC_REMAP_AT: usize = 0xe4;
const REC_BASES: [usize; 2] = [0xe8, 0xec];
const REC_SHAPE_COUNT: usize = 0xa8;
const REC_SHAPE_AT: usize = 0xf8;
/// Remap entries a record can reach (the bone bytes index it).
const REMAP_REACH: u64 = 256;

/// A hair input sink's root 0 views the strand records use.
struct StrandInputs {
    records: Option<SlotView>,
    palette: Option<SlotView>,
    shapes: Option<SlotView>,
}

fn le32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
}

/// `len` bytes at byte `off` of view `v` (cut at its end): (address,
/// length, cut) or a reason (NULL: no buffer view; IMPLAUSIBLE: past it).
fn view_range(v: Option<&SlotView>, off: u64, len: u64) -> Result<(u64, u64, bool), u32> {
    let v = v.filter(|v| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER) && v.addr != 0).ok_or(reason::NULL)?;
    if off >= v.bytes {
        return Err(reason::IMPLAUSIBLE);
    }
    let n = len.min(v.bytes - off);
    Ok((v.addr + off, n, n < len))
}

/// Bones a record reaches past its palette base: the largest of the 256
/// remap entries from rec+0xe4 (16-bit, & 0xfff) + 1; 256 when the remap
/// is not readable.
fn bones_reached(remap: Option<&SlotView>, at: u32) -> u64 {
    let got = view_range(remap, 2 * u64::from(at), 2 * REMAP_REACH).and_then(|(a, n, _)| read(a, n));
    match got {
        Ok(b) => b.chunks_exact(2).map(|w| u64::from(u16::from_le_bytes([w[0], w[1]]) & 0xfff)).max().map_or(REMAP_REACH, |m| m + 1),
        Err(_) => REMAP_REACH,
    }
}

/// A hair input dispatch's STRAND_VIEW / STRAND_BYTES records: for each
/// strand group record of `records` (`fp::STRAND_RECORD` bytes each, at
/// most `STRAND_MAX`), the views at the heap indices in its
/// `fp::STRAND_WORDS` (`resolve`: index -> handle and view), with their
/// bytes; once per frame per record buffer content (later sinks naming
/// the same records add nothing). Counts: records, views with bytes,
/// without.
fn strand_groups(acc: &mut Acc, run: &mut Run, owner: u32, inputs: StrandInputs, resolve: &dyn Fn(u32) -> Option<(u64, SlotView)>) {
    let Some(v) = inputs.records.filter(|v| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER)) else {
        acc.refs.push(reference(owner, kind::STRAND_VIEW, 0, 0, 0, Err(reason::NULL)));
        return;
    };
    let n = (v.bytes as usize / fp::STRAND_RECORD).min(STRAND_MAX);
    let bytes = if n == 0 { Err(reason::IMPLAUSIBLE) } else { read(v.addr, (n * fp::STRAND_RECORD) as u64) };
    let bytes = match bytes {
        Ok(b) => b,
        Err(why) => {
            acc.refs.push(reference(owner, kind::STRAND_VIEW, 0, v.addr, 0, Err(why)));
            return;
        }
    };
    if !acc.strand_seen.insert(xnw_core::hash128(&[&v.addr.to_le_bytes(), &bytes])) {
        return;
    }
    for (r, rec) in bytes.chunks_exact(fp::STRAND_RECORD).enumerate() {
        acc.hair[2] += 1;
        for (k, &o) in fp::STRAND_WORDS.iter().enumerate() {
            let index = u32::from_le_bytes([rec[o], rec[o + 1], rec[o + 2], rec[o + 3]]);
            let slot = (r * 8 + k) as u16;
            let before = acc.refs.len();
            heap_view_as(acc, run, owner, [kind::STRAND_VIEW, kind::STRAND_BYTES], slot, index, resolve(index));
            let ok = acc.refs[before..].iter().all(|r| reason::has_bytes(r.reason));
            acc.hair[if ok { 3 } else { 4 }] += 1;
        }
        strand_extras(acc, run, owner, r as u16, rec, &inputs, resolve);
    }
}

/// One group record's STRAND_PALETTE windows (the bones at its two
/// palette bases) and its STRAND_SPAN (the collision shapes it names).
fn strand_extras(acc: &mut Acc, run: &mut Run, owner: u32, r: u16, rec: &[u8], inputs: &StrandInputs, resolve: &dyn Fn(u32) -> Option<(u64, SlotView)>) {
    let remap = resolve(le32(rec, fp::STRAND_WORDS[0])).map(|(_, v)| v);
    let bones = bones_reached(remap.as_ref(), le32(rec, REC_REMAP_AT));
    let bone = fp::BONE_BYTES as u64;
    for (j, &o) in REC_BASES.iter().enumerate() {
        let base = u64::from(le32(rec, o));
        let slot = 2 * r + j as u16;
        let r = match view_range(inputs.palette.as_ref(), base * bone, bones * bone) {
            Ok((at, n, cut)) => reference_cut(owner, kind::STRAND_PALETTE, slot, at, n, read(at, n).map(|b| keep(acc, run, kind::STRAND_PALETTE, b)), cut),
            Err(why) => reference(owner, kind::STRAND_PALETTE, slot, base, bones * bone, Err(why)),
        };
        acc.strand_extra[if reason::has_bytes(r.reason) { 0 } else { 1 }] += 1;
        acc.refs.push(r);
    }
    let count = u64::from(le32(rec, REC_SHAPE_COUNT) >> 24);
    let off = u64::from(le32(rec, REC_SHAPE_AT));
    let len = count * fp::STRAND_SHAPE as u64;
    let span = if count == 0 { Err(reason::NULL) } else { view_range(inputs.shapes.as_ref(), off, len) };
    let r = match span {
        Ok((at, n, cut)) => reference_cut(owner, kind::STRAND_SPAN, r, at, n, read(at, n).map(|b| keep(acc, run, kind::STRAND_SPAN, b)), cut),
        Err(why) => reference(owner, kind::STRAND_SPAN, r, off, len, Err(why)),
    };
    acc.strand_extra[if reason::has_bytes(r.reason) { 2 } else { 3 }] += 1;
    acc.refs.push(r);
}

/// The offset between a buffer's GPU address and its CPU address.
const GPU_ALIAS: u64 = 0x4_0000_0000;

/// A root view with no buffer object or stream behind it: this much of
/// the title memory there (the device binds such a view as the memory
/// around it, draw.rs `root_view`: 64 KiB for a constant buffer, 1 MiB
/// for the rest; the probe takes the smaller, marked cut).
const ROOT_WINDOW: u64 = 64 << 10;

/// How a root view was read: in a title buffer object, over a hook draw's
/// stream, over a window of [`ROOT_WINDOW`] bytes, not at all.
const ROOT_BUFFER: u8 = 0;
const ROOT_STREAM: u8 = 1;
const ROOT_WINDOWED: u8 = 2;
pub(super) const ROOT_NONE: u8 = 3;

/// Bytes from `address` to the end of the hook stream holding it.
fn stream_left(streams: &BTreeMap<u64, u64>, address: u64) -> Option<u64> {
    streams.range(..=address).next_back().map(|(&at, &len)| at + len).filter(|&end| end > address).map(|end| end - address)
}

/// Where a root view's bytes are read: (address, bytes to read, through
/// the CPU alias, how). A root view names a GPU address: the title's
/// pages are mapped twice (paired reservation, CPU view + `GPU_ALIAS` =
/// GPU view [V a run's log: 'paired reservation 0x100000000 <->
/// 0x500000000']); its vertex streams lie in its own pages, in no buffer
/// object (one captured frame: 2,449 of 2,449 root views at 0x5..0x9_xxxx_xxxx,
/// 2,389 of them at their hook draw's vertex stream + `GPU_ALIAS`).
/// In order: a buffer object (`left`) at the address, then at the
/// address - `GPU_ALIAS`; a hook stream (`stream`) at either; else a
/// window at the address (the memory is mapped at both).
pub(super) fn root_view_at(a: u64, left: impl Fn(u64) -> Option<u64>, stream: impl Fn(u64) -> Option<u64>) -> (u64, u64, bool, u8) {
    let cpu = a.checked_sub(GPU_ALIAS).filter(|&c| c >= 0x1_0000);
    if let Some(n) = left(a) {
        return (a, n, false, ROOT_BUFFER);
    }
    if let Some((c, n)) = cpu.and_then(|c| left(c).map(|n| (c, n))) {
        return (c, n, true, ROOT_BUFFER);
    }
    if let Some(n) = stream(a) {
        return (a, n, false, ROOT_STREAM);
    }
    if let Some((c, n)) = cpu.and_then(|c| stream(c).map(|n| (c, n))) {
        return (c, n, true, ROOT_STREAM);
    }
    if a == 0 {
        return (a, 0, false, ROOT_NONE);
    }
    (a, ROOT_WINDOW, false, ROOT_WINDOWED)
}

/// The draw words: the frame table's (root 0) slot 86 buffer, at the byte
/// offset in root 4's constant dword 1 (the pixel program's bindless
/// words for this draw).
const WORDS_SLOT: u32 = 86;
const WORDS_ROOT: usize = 4;
const WORDS_DWORD: usize = 1;

/// The draw words' address: slot 86's view address + the offset; NULL
/// without a buffer view or the constant, IMPLAUSIBLE past the view.
fn words_at(slot86: Option<SlotView>, off: Option<u32>) -> Result<u64, u32> {
    let v = slot86.filter(|v| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER | view::CONSTANT) && v.addr != 0).ok_or(reason::NULL)?;
    let off = u64::from(off.ok_or(reason::NULL)?);
    if off + fp::DRAW_WORDS_BYTES as u64 > v.bytes {
        return Err(reason::IMPLAUSIBLE);
    }
    Ok(v.addr + off)
}

/// A draw sink's DRAW_WORDS records: slot 0 the 64 bytes at `at`; slot 1
/// the views at the heap indices they hold (`resolve`: index -> heap
/// start and view, None when not in the bound heap), distinct, in word
/// order; their textures noted. NULL when no word lies in the heap,
/// NOT_RESIDENT when none names a view.
fn draw_words(acc: &mut Acc, run: &mut Run, owner: u32, at: Result<u64, u32>, resolve: &dyn Fn(u32) -> Option<(u64, SlotView)>) {
    let len = fp::DRAW_WORDS_BYTES as u64;
    let bytes = match at.and_then(|a| read(a, len)) {
        Ok(b) => b,
        Err(why) => {
            acc.draw_words[1] += 1;
            acc.refs.push(reference(owner, kind::DRAW_WORDS, 0, *at.as_ref().unwrap_or(&0), len, Err(why)));
            return;
        }
    };
    acc.draw_words[0] += 1;
    let h = keep(acc, run, kind::DRAW_WORDS, bytes.clone());
    acc.refs.push(reference(owner, kind::DRAW_WORDS, 0, *at.as_ref().unwrap_or(&0), len, Ok(h)));
    let mut seen = HashSet::new();
    let mut views = Vec::new();
    let mut heap_start = 0;
    for w in bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])) {
        if !seen.insert(w) {
            continue;
        }
        let Some((start, v)) = resolve(w) else { continue };
        heap_start = start;
        if v.kind == view::TEXTURE || v.kind == view::PLACED {
            texture(acc, run, &v);
        }
        if v.kind != view::NONE {
            views.push(SlotView { slot: w, ..v });
        }
    }
    acc.draw_words[2] += views.len() as u32;
    let got = if views.is_empty() {
        Err(if heap_start == 0 { reason::NULL } else { reason::NOT_RESIDENT })
    } else {
        Ok(keep(acc, run, kind::DRAW_WORDS, fp::encode_slots(&views)))
    };
    acc.refs.push(reference(owner, kind::DRAW_WORDS, 1, heap_start, (views.len() * fp::SLOT_BYTES) as u64, got));
}

/// A sink's SAMPLERS record: the bound sampler heap (`range`: first slot,
/// slot count) read once per frame per heap object, every slot with
/// non-zero words among the first `SAMPLER_MAX` as a `SamplerEntry`
/// (`resolve`: raw words -> form and description, None when they are not a
/// record of ours). NULL without a heap, ZERO_FILLED when every slot is
/// zero, CUT past `SAMPLER_MAX` slots.
fn samplers(acc: &mut Acc, run: &mut Run, owner: u32, heap: usize, range: Option<(u64, u32)>, resolve: &dyn Fn([u32; 4]) -> Option<(u32, [u32; 27])>) {
    if let Some(r) = acc.sampler_heaps.get(&heap) {
        let r = Ref { owner, ..*r };
        acc.sampler_counts[1] += u32::from(r.reason == reason::NULL);
        acc.refs.push(r);
        return;
    }
    let r = match range {
        None => reference(owner, kind::SAMPLERS, 0, 0, 0, Err(reason::NULL)),
        Some((start, count)) => {
            let n = count.min(fp::SAMPLER_MAX);
            match read(start, tap::SAMPLER_SLOT * u64::from(n)) {
                Err(why) => reference(owner, kind::SAMPLERS, 0, start, 0, Err(why)),
                Ok(b) => {
                    let entries: Vec<fp::SamplerEntry> = b
                        .chunks_exact(tap::SAMPLER_SLOT as usize)
                        .enumerate()
                        .filter_map(|(i, c)| {
                            let raw: [u32; 4] = std::array::from_fn(|k| u32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]]));
                            (raw != [0; 4]).then(|| {
                                let (form, desc) = resolve(raw).unwrap_or((fp::SAMPLER_RAW, [0; 27]));
                                fp::SamplerEntry { index: i as u32, form, raw, desc }
                            })
                        })
                        .collect();
                    acc.sampler_counts[0] += entries.len() as u32;
                    let len = (entries.len() * fp::SAMPLER_BYTES) as u64;
                    let got = if entries.is_empty() { Err(reason::ZERO_FILLED) } else { Ok(keep(acc, run, kind::SAMPLERS, fp::encode_samplers(&entries))) };
                    reference_cut(owner, kind::SAMPLERS, 0, start, len, got, count > n)
                }
            }
        }
    };
    acc.sampler_counts[1] += u32::from(r.reason == reason::NULL);
    acc.sampler_heaps.insert(heap, r);
    acc.refs.push(r);
}

/// The pre-skin programs that sample a displacement map (variants 8 and 9,
/// container hash): the map's heap index is +0x8C of a 144-B element of the
/// compute table's (root 0) slot 8 buffer, the element picked in the
/// program (0x45a8, 0x52c8, 0xebd8 of dcfcc6a8: `s_load_dwordx8` from the
/// view heap at that word << 5).
const DISPLACED: [u128; 2] = [0xdcfc_c6a8_03ab_0b26_3a10_b1c5_61d7_ba77, 0x658a_a487_9482_263f_eb31_ad13_c3fe_c09f];
const MAP_SLOT: u32 = 8;
const MAP_STRIDE: u64 = 144;
const MAP_WORD: usize = 0x8C;
/// Elements looked at per dispatch.
const MAP_MAX: u64 = 64;

/// A displacing dispatch's DISP_MAP records: for each element of `elems`
/// (root 0 slot 8), the view at its heap index (`resolve`: index ->
/// handle and view, None when not in the bound heap), its texture noted.
/// NULL without the buffer, IMPLAUSIBLE when it holds no whole element or an
/// index is outside the heap, NOT_RESIDENT when the slot holds no view.
fn displacement_maps(acc: &mut Acc, run: &mut Run, owner: u32, elems: Option<SlotView>, resolve: &dyn Fn(u32) -> Option<(u64, SlotView)>) {
    let Some(v) = elems.filter(|v| matches!(v.kind, view::BUFFER | view::PLACED_BUFFER)) else {
        acc.disp_maps[1] += 1;
        acc.refs.push(reference(owner, kind::DISP_MAP, 0, 0, 0, Err(reason::NULL)));
        return;
    };
    let n = (v.bytes / MAP_STRIDE).min(MAP_MAX);
    let bytes = if n == 0 { Err(reason::IMPLAUSIBLE) } else { read(v.addr, n * MAP_STRIDE) };
    let bytes = match bytes {
        Ok(b) => b,
        Err(why) => {
            acc.disp_maps[1] += 1;
            acc.refs.push(reference(owner, kind::DISP_MAP, 0, v.addr, 0, Err(why)));
            return;
        }
    };
    for (e, el) in bytes.chunks_exact(MAP_STRIDE as usize).enumerate() {
        let slot = e as u16;
        let index = u32::from_le_bytes([el[MAP_WORD], el[MAP_WORD + 1], el[MAP_WORD + 2], el[MAP_WORD + 3]]);
        let r = match resolve(index) {
            None => reference(owner, kind::DISP_MAP, slot, u64::from(index), 0, Err(reason::IMPLAUSIBLE)),
            Some((h, sv)) if sv.kind == view::NONE => reference(owner, kind::DISP_MAP, slot, h, 0, Err(reason::NOT_RESIDENT)),
            Some((h, sv)) => {
                let sv = SlotView { slot: index, ..sv };
                if sv.kind == view::TEXTURE || sv.kind == view::PLACED {
                    texture(acc, run, &sv);
                }
                let got = keep(acc, run, kind::DISP_MAP, fp::encode_slots(&[sv]));
                reference(owner, kind::DISP_MAP, slot, h, fp::SLOT_BYTES as u64, Ok(got))
            }
        };
        acc.disp_maps[usize::from(r.reason != reason::OK)] += 1;
        acc.refs.push(r);
    }
}

/// The pre-skin items of the title's lists now: each head's chain of
/// 144-B items (next = the first qword), every read fault-safe and bounded.
fn walk_items() -> Vec<(u64, Vec<u8>)> {
    let mut out = Vec::new();
    let Ok(heads) = read(ITEM_LISTS.load(Ordering::Relaxed), 8 * ITEM_HEADS) else { return out };
    for head in heads.chunks_exact(8) {
        let mut at = u64::from_le_bytes(head.try_into().unwrap_or([0; 8]));
        while at != 0 && out.len() < ITEMS_MAX {
            let Ok(item) = read(at, fp::ITEM_BYTES as u64) else { break };
            let next = u64::from_le_bytes(item[..8].try_into().unwrap_or([0; 8]));
            out.push((at, item));
            at = next;
        }
    }
    out
}

/// The DISPATCH record of dispatch sink `owner`, and the views behind
/// the heap indices its constant buffer names. b7 is the 96-B constant
/// buffer among the sink's bound ones (table slots and root views) that
/// equals an item's +32..+127 (the title copies it from there, 0x144E3CDD0
/// [V]); without an item, the one at table slot 8 (ud5+0x100).
fn dispatch_record(acc: &mut Acc, run: &mut Run, owner: u32, groups: [u32; 3], view_heap: usize) {
    let cands: Vec<(u32, Vec<u8>)> = acc
        .refs
        .iter()
        .filter(|r| r.owner == owner && ((r.kind == kind::CBV && (96..=256).contains(&r.len)) || r.kind == kind::ROOT_VIEW))
        .map(|r| (if r.kind == kind::CBV { u32::from(r.slot) } else { u32::MAX }, r.addr))
        .collect::<Vec<_>>()
        .into_iter()
        .filter_map(|(slot, addr)| Some((slot, read(addr, fp::B7_BYTES as u64).ok()?)))
        .collect();
    let find = |items: &[(u64, Vec<u8>)]| cands.iter().find_map(|(slot, b)| items.iter().find(|(_, item)| item[32..32 + fp::B7_BYTES] == b[..]).map(|(at, item)| (*slot, b.clone(), *at, item.clone())));
    let mut found = find(&acc.items);
    if found.is_none() && !cands.is_empty() && acc.item_walks < WALKS_MAX {
        acc.item_walks += 1;
        acc.items = walk_items();
        found = find(&acc.items);
    }
    let mut d = fp::Dispatch { groups, ..fp::Dispatch::default() };
    let mut item_at = 0;
    if let Some((slot, b7, at, item)) = found {
        (d.flags, d.b7_slot, d.b7, d.item, item_at) = (fp::B7_FOUND | fp::ITEM_FOUND, slot, b7, item, at);
    } else if let Some((slot, b7)) = cands.iter().find(|(slot, _)| *slot != u32::MAX && fp::table_slot_of(*slot as u16).1 == 8) {
        (d.flags, d.b7_slot, d.b7) = (fp::B7_FOUND, *slot, b7.clone());
    }
    let word = |b: &[u8], o: usize| b.get(o..o + 4).map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
    (d.word, d.vertices) = (word(&d.b7, 0x34), word(&d.b7, 0x38));
    let heap_indices = [0x40, 0x44, 0x48].map(|o| word(&d.b7, o));
    let b7_found = d.flags & fp::B7_FOUND != 0;
    let h = keep(acc, run, kind::DISPATCH, d.encode());
    acc.refs.push(reference(owner, kind::DISPATCH, 0, item_at, fp::DISPATCH_BYTES as u64, Ok(h)));
    if !b7_found {
        return;
    }
    // The heap indices at b7+0x40/+0x44/+0x48 (vertex buffer, palette,
    // extra stream: writer 0x1427C15A0 [V]; which is which: I).
    for (k, index) in heap_indices.into_iter().enumerate() {
        let resolved = tap::heap_slot(view_heap, index).and_then(|(_, handle)| read(handle, tap::SLOT).ok().map(|_| (handle, slot_view(index, &tap::view(handle)))));
        heap_view(acc, run, owner, k as u16, index, resolved);
    }
}

/// Heap index `index` (the k-th of b7) of dispatch sink `owner`: its
/// view (HEAP_VIEW; a reason when the index is not in the bound heap or the
/// slot holds no view) and the bytes of its range (HEAP_BYTES).
fn heap_view(acc: &mut Acc, run: &mut Run, owner: u32, k: u16, index: u32, resolved: Option<(u64, SlotView)>) {
    heap_view_as(acc, run, owner, [kind::HEAP_VIEW, kind::HEAP_BYTES], k, index, resolved);
}

/// `heap_view` with the record kinds `[view, bytes]`.
fn heap_view_as(acc: &mut Acc, run: &mut Run, owner: u32, kinds: [u16; 2], k: u16, index: u32, resolved: Option<(u64, SlotView)>) {
    let [kv, kb] = kinds;
    let Some((handle, v)) = resolved else {
        acc.refs.push(reference(owner, kv, k, u64::from(index), 0, Err(reason::IMPLAUSIBLE)));
        return;
    };
    if v.kind == view::NONE {
        acc.refs.push(reference(owner, kv, k, handle, 0, Err(reason::NOT_RESIDENT)));
        return;
    }
    let got = keep(acc, run, kv, fp::encode_slots(&[v]));
    acc.refs.push(reference(owner, kv, k, handle, fp::SLOT_BYTES as u64, Ok(got)));
    match v.kind {
        view::CONSTANT => {
            let len = v.bytes.min(CBV_CAP);
            let got = read(v.addr, len).map(|b| keep(acc, run, kb, b));
            acc.refs.push(reference_cut(owner, kb, k, v.addr, len, got, v.bytes > len));
        }
        view::BUFFER | view::PLACED_BUFFER if v.kind == view::BUFFER && v.layout == view::UNORDERED => {
            acc.refs.push(reference(owner, kb, k, v.addr, v.bytes.min(u64::from(u32::MAX)), Err(reason::GPU_OUTPUT)));
        }
        view::BUFFER | view::PLACED_BUFFER => {
            let len = v.bytes.min(BUF_CAP);
            let got = buffer(acc, run, v.addr, len);
            acc.refs.push(reference_cut(owner, kb, k, v.addr, len, got, v.bytes > len));
        }
        view::TEXTURE | view::PLACED => texture(acc, run, &v),
        _ => {}
    }
}

/// A pipeline's stages: identity rows, container bytes once per run.
fn programs(acc: &mut Acc, run: &mut Run, pipeline: usize) {
    for (i, s) in tap::stages(pipeline).into_iter().enumerate() {
        let mut row = fp::Program { pipeline: pipeline as u64, stage: s.name.to_string(), form: if s.container { 1 } else { 2 }, container: s.hash, size: s.size as u32, prefix: s.prefix, reason: reason::OK };
        if s.size == 0 {
            row.reason = reason::NULL;
        } else if !held(acc, run, s.hash)
            && let Some(b) = tap::stage_bytes(pipeline, i)
        {
            acc.blobs.insert(s.hash, (0, b));
        }
        if DISPLACED.contains(&s.hash) {
            acc.displaced.insert(pipeline as u64);
        }
        if let Some(w) = WHEELS.iter().find(|w| w.0 == s.hash) {
            acc.wheels.insert(pipeline as u64, (w.1, w.2));
        }
        acc.programs.push(row);
    }
    if let Some((targets, depth)) = tap::targets(pipeline) {
        let mut words = vec![targets.len() as u32];
        words.extend((0..8).map(|i| targets.get(i).copied().unwrap_or(0)));
        words.push(depth);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let h = keep(acc, run, kind::TARGETS, bytes);
        acc.refs.push(reference(fp::FRAME_SCOPE, kind::TARGETS, 0, pipeline as u64, fp::TARGETS_BYTES as u64, Ok(h)));
    }
}

/// Instanced vehicle programs (wheels, tyres; mesh container hash): the
/// per-draw table (root 1) slot of the per-instance palette base (u32) and
/// the slot of the meshlet info (dword 0 & 0x1FFFF = meshlets per instance).
const WHEELS: [(u128, u32, u32); 2] = [(0xeddd_a273_13a0_da86_fada_f4af_39c2_f68a, 8, 4), (0xf9dd_5454_30b0_753b_962c_87bc_9cb5_3c76, 9, 6)];
/// The palette they index: the frame table's (root 0) slot 5, 48 B a bone.
const WHEEL_PALETTE: u32 = 5;
const WHEEL_BONES: u64 = 256;
/// Instances read per draw.
const WHEEL_MAX: u64 = 64;

/// The wheel draw's slot views the windows need (root 0 palette, root 1
/// bases and meshlet info), resolved from its table roots.
fn wheel_views(d: &TapDraw, base_slot: u32, meshlet_slot: u32) -> [Option<SlotView>; 3] {
    let at = |root: usize, slot: u32| match d.roots.get(root) {
        Some(TapRoot::Table { handle, slots, .. }) if slot < *slots => Some(slot_view(slot, &tap::view(handle + tap::SLOT * u64::from(slot)))).filter(|v| v.kind != view::NONE),
        _ => None,
    };
    [at(0, WHEEL_PALETTE), at(1, base_slot), at(1, meshlet_slot)]
}

/// The palette bones an instanced wheel draw reads past the buffer
/// cap: for each distinct per-instance base, 256 x 48 B of the palette
/// view at 48 * base (kind PAL_WINDOW, slot k = k-th distinct base, addr =
/// the window; cut at the view's end). A missing view or count is one
/// record with a reason.
fn wheel_windows(acc: &mut Acc, run: &mut Run, owner: u32, groups: u32, views: [Option<SlotView>; 3]) {
    let [Some(pal), Some(bases), Some(meshlets)] = views else {
        acc.wheel_windows[1] += 1;
        acc.refs.push(reference(owner, kind::PAL_WINDOW, 0, 0, 0, Err(reason::NULL)));
        return;
    };
    let per = read(meshlets.addr, 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x1_FFFF);
    let n = if per == 0 { 0 } else { u64::from(groups / per) }.min(WHEEL_MAX).min(bases.bytes / 4);
    let words = if n == 0 { Err(reason::IMPLAUSIBLE) } else { read(bases.addr, 4 * n) };
    let words = match words {
        Ok(w) => w,
        Err(why) => {
            acc.wheel_windows[1] += 1;
            acc.refs.push(reference(owner, kind::PAL_WINDOW, 0, bases.addr, 4 * n, Err(why)));
            return;
        }
    };
    let mut seen: Vec<u32> = Vec::new();
    for c in words.chunks_exact(4) {
        let b = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        if !seen.contains(&b) {
            seen.push(b);
        }
    }
    for (k, b) in seen.into_iter().enumerate() {
        let off = 48 * u64::from(b);
        let at = pal.addr.wrapping_add(off);
        let want = 48 * WHEEL_BONES;
        let len = want.min(pal.bytes.saturating_sub(off));
        let got = if len == 0 { Err(reason::IMPLAUSIBLE) } else { read(at, len).map(|w| keep(acc, run, kind::PAL_WINDOW, w)) };
        acc.wheel_windows[usize::from(got.is_err())] += 1;
        acc.refs.push(reference_cut(owner, kind::PAL_WINDOW, k as u16, at, len, got, len < want));
    }
}

/// A table root: its resolved slots (by raw content, run-wide), the bytes
/// behind its constant buffers and its textures (once per content per frame).
fn table(acc: &mut Acc, run: &mut Run, owner: u32, root: u16, handle: u64, slots: u32, view_heap: usize, mode: u8) {
    let raw_len = u64::from(slots) * tap::SLOT;
    let raw = match read(handle, raw_len) {
        Ok(r) => r,
        Err(why) => {
            acc.refs.push(reference(owner, kind::TABLE, root, handle, raw_len, Err(why)));
            return;
        }
    };
    let raw_hash = xnw_core::hash128(&[&raw]);
    // A dispatch's table keeps other bytes than a draw's (`view_bytes`).
    let seen = raw_hash ^ u128::from(mode);
    if let Some(refs) = acc.tables.get(&seen) {
        let slot = |r: &Ref| if r.kind == kind::TABLE { root } else { fp::table_slot(u32::from(root), fp::table_slot_of(r.slot).1) };
        // Frame-scope records met inside the table (bindless, arena base) stay once.
        let again: Vec<Ref> = refs.iter().filter(|r| r.owner != fp::FRAME_SCOPE).map(|r| Ref { owner, slot: slot(r), ..*r }).collect();
        acc.refs.extend(again);
        return;
    }
    let (hash, views) = match run.resolved.get(&raw_hash) {
        Some((h, views)) => {
            let (h, views) = (*h, views.clone());
            if !held(acc, run, h) {
                acc.blobs.insert(h, (kind::TABLE, fp::encode_slots(&views)));
            }
            (h, views)
        }
        None => {
            let views: Vec<SlotView> = (0..slots).map(|k| slot_view(k, &tap::view(handle + tap::SLOT * u64::from(k)))).filter(|v| v.kind != view::NONE).collect();
            let blob = fp::encode_slots(&views);
            let h = keep(acc, run, kind::TABLE, blob);
            run.resolved.insert(raw_hash, (h, views.clone()));
            (h, views)
        }
    };
    let first = acc.refs.len();
    acc.refs.push(reference(owner, kind::TABLE, root, handle, (views.len() * fp::SLOT_BYTES) as u64, Ok(hash)));
    for v in &views {
        view_bytes(acc, run, owner, root, v, view_heap, mode);
    }
    let refs = acc.refs[first..].to_vec();
    acc.tables.insert(seen, refs);
}

/// The bytes behind one resolved table slot. A dispatch sink (`compute`)
/// reads no unordered access view (the GPU writes those; under the
/// null GPU nothing does) and no view over the buffer cap, and has neither
/// bindless table nor per-draw record.
fn view_bytes(acc: &mut Acc, run: &mut Run, owner: u32, root: u16, v: &SlotView, view_heap: usize, mode: u8) {
    let compute = mode != TB_DRAW;
    let slot = fp::table_slot(u32::from(root), v.slot);
    let plain_uav = v.kind == view::BUFFER && v.layout == view::UNORDERED;
    match v.kind {
        view::CONSTANT => {
            let len = v.bytes.min(CBV_CAP);
            let bytes = read(v.addr, len);
            if v.slot == BINDLESS_SLOT
                && !compute
                && let Ok(b) = &bytes
            {
                bindless(acc, run, b, view_heap);
            }
            let got = bytes.map(|b| keep(acc, run, kind::CBV, b));
            acc.refs.push(reference_cut(owner, kind::CBV, slot, v.addr, len, got, v.bytes > len));
        }
        view::TEXTURE | view::PLACED => texture(acc, run, v),
        view::BUFFER if mode == TB_HEAD && plain_uav => {
            let len = v.bytes.min(HEAD_UAV_CAP);
            let got = buffer(acc, run, v.addr, len);
            acc.head_uavs[usize::from(got.is_err())] += 1;
            acc.refs.push(reference_cut(owner, kind::BUFFER, slot, v.addr, len, got, v.bytes > len));
        }
        view::BUFFER | view::PLACED_BUFFER if compute && (v.bytes > BUF_CAP || plain_uav) => {
            let why = if plain_uav { reason::GPU_OUTPUT } else { reason::BUDGET };
            acc.refs.push(reference(owner, kind::BUFFER, slot, v.addr, v.bytes.min(u64::from(u32::MAX)), Err(why)));
        }
        view::BUFFER | view::PLACED_BUFFER => {
            let len = v.bytes.min(BUF_CAP);
            let got = buffer(acc, run, v.addr, len);
            acc.refs.push(reference_cut(owner, kind::BUFFER, slot, v.addr, len, got, v.bytes > len));
            if v.slot == 0 && v.bytes == REC_BYTES && !compute {
                arena_words(acc, run, owner, slot, v.addr);
            }
        }
        _ => {}
    }
}

/// The per-draw record a T9 table names at slot 0 (128 B: +0x30 palette
/// index, +0x54 bone offset, +0x74 arena byte offset or -1).
const REC_BYTES: u64 = 128;
/// Arena bytes kept at the record's +0x74 (vehicle mask|rank words, 4 B per
/// 16 bones; ped remap entries start there too).
const ARENA_WORDS: u64 = 256;
/// The title's per-frame arena base global (qword 0x1475DF140, real address; 0: unknown).
static ARENA_AT: AtomicU64 = AtomicU64::new(0);

pub(super) fn set_arena(global: u64) {
    ARENA_AT.store(global, Ordering::Relaxed);
}

/// v2.1: a 128-B slot-0 record whose +0x74 is not -1: the
/// words at arena base + rec+0x74 (kind ARENA, slot as the record's; the
/// frame's arena base is a frame-scope ARENA record, addr = the base).
fn arena_words(acc: &mut Acc, run: &mut Run, owner: u32, slot: u16, rec: u64) {
    let Ok(r) = read(rec + 0x74, 4) else { return };
    let off = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
    if off == u32::MAX {
        return;
    }
    let base = *acc.arena.get_or_insert_with(|| {
        let at = ARENA_AT.load(Ordering::Relaxed);
        read(at, 8).map_or(0, |b| u64::from_le_bytes(b[..8].try_into().unwrap_or([0; 8])))
    });
    if base == 0 {
        acc.refs.push(reference(owner, kind::ARENA, slot, 0, ARENA_WORDS, Err(reason::NULL)));
        return;
    }
    if !acc.arena_noted {
        acc.arena_noted = true;
        acc.refs.push(Ref { owner: fp::FRAME_SCOPE, kind: kind::ARENA, slot: 0, addr: base, len: 0, reason: reason::OK, hash: 0 });
    }
    let at = base + u64::from(off);
    let got = read(at, ARENA_WORDS).map(|b| keep(acc, run, kind::ARENA, b));
    acc.refs.push(reference(owner, kind::ARENA, slot, at, ARENA_WORDS, got));
}

/// A buffer range (root view or buffer view: vertex streams, blend indices
/// and weights, delta streams): bytes once per frame per range, stored
/// once per run per content; over the caps a reason.
fn buffer(acc: &mut Acc, run: &mut Run, addr: u64, len: u64) -> Result<u128, u32> {
    if let Some(got) = acc.buffers.get(&(addr, len)) {
        return *got;
    }
    let got = if len == 0 {
        Err(if addr == 0 { reason::NULL } else { reason::IMPLAUSIBLE })
    } else if len > CBV_CAP && acc.buffer_bytes + len > FRAME_BUF_CAP {
        Err(reason::BUDGET)
    } else {
        read(addr, len).map(|b| {
            acc.buffer_bytes += len;
            keep(acc, run, kind::BUFFER, b)
        })
    };
    acc.buffers.insert((addr, len), got);
    got
}

/// The bindless index CB: each dword a view heap index (bl[k], s0 [V]);
/// its slots resolved once per frame as a frame-scope record, their textures noted.
fn bindless(acc: &mut Acc, run: &mut Run, bytes: &[u8], view_heap: usize) {
    if !acc.bindless.insert(xnw_core::hash128(&[bytes])) {
        return;
    }
    let mut seen = HashSet::new();
    let mut views = Vec::new();
    // The view heap's first slot (record addr): handle = start + 32 * slot.
    let mut heap_start = 0;
    for w in bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])) {
        if seen.len() >= BINDLESS_MAX || !seen.insert(w) {
            continue;
        }
        let Some((start, handle)) = tap::heap_slot(view_heap, w) else { continue };
        heap_start = start;
        let v = slot_view(w, &tap::view(handle));
        if v.kind == view::TEXTURE || v.kind == view::PLACED {
            texture(acc, run, &v);
        }
        if v.kind != view::NONE {
            views.push(v);
        }
    }
    let len = (views.len() * fp::SLOT_BYTES) as u64;
    let h = keep(acc, run, kind::HEAP_SLOT, fp::encode_slots(&views));
    acc.refs.push(reference(fp::FRAME_SCOPE, kind::HEAP_SLOT, (acc.bindless.len() - 1) as u16, heap_start, len, Ok(h)));
}

/// A texture slot: origin, else texels (capped), and a fill fingerprint.
fn texture(acc: &mut Acc, run: &mut Run, v: &SlotView) {
    let key = (v.addr, v.bytes);
    if acc.textures.contains_key(&key) {
        return;
    }
    let desc = keep(acc, run, kind::HEAP_SLOT, fp::encode_slots(&[SlotView { slot: 0, ..*v }]));
    let mut t = fp::Texture { desc, at: v.addr, len: v.bytes, ..fp::Texture::default() };
    let sampled: Result<Vec<u8>, u32> = Fill::windows(v.bytes as usize).iter().map(|&(o, n)| read(v.addr + o as u64, n as u64)).collect::<Result<Vec<_>, _>>().map(|p| p.concat());
    match sampled {
        Err(why) => t.reason = why,
        Ok(s) => {
            t.fill = Fill::of(&s);
            t.origin = xnw_formats::origin::origin_of(v.addr, 0x1000).map(|o| {
                let name = xnw_formats::origin::file(o.file).and_then(|f| f.path.file_name().map(|n| n.to_string_lossy().into_owned()));
                format!("{}#{}", name.unwrap_or_else(|| format!("file{}", o.file)), o.texture)
            });
            // Zero in the sampled windows: a small texture is read whole
            // before it is called zero-filled (a 64x320 hair width profile).
            let whole = if t.fill.empty() && v.bytes <= ZERO_CHECK { read(v.addr, v.bytes).ok().filter(|b| b.iter().any(|x| *x != 0)) } else { None };
            if t.fill.empty() && whole.is_none() {
                t.reason = reason::ZERO_FILLED;
            } else if let Some(b) = whole.filter(|_| t.origin.is_none() && acc.texel_bytes + v.bytes <= FRAME_TEX_CAP) {
                acc.texel_bytes += v.bytes;
                (t.texels, t.texel_bytes) = (keep(acc, run, u16::MAX, b), v.bytes);
            } else if t.origin.is_none() {
                let k = (v.addr, v.bytes, t.fill.hash);
                if let Some(h) = run.texels.get(&k).copied().filter(|h| held(acc, run, *h)) {
                    (t.texels, t.texel_bytes) = (h, v.bytes);
                } else if v.bytes > TEX_CAP || acc.texel_bytes + v.bytes > FRAME_TEX_CAP {
                    t.reason = reason::BUDGET;
                } else {
                    match read(v.addr, v.bytes) {
                        Ok(b) => {
                            acc.texel_bytes += v.bytes;
                            let h = keep(acc, run, u16::MAX, b);
                            run.texels.insert(k, h);
                            (t.texels, t.texel_bytes) = (h, v.bytes);
                        }
                        Err(why) => t.reason = why,
                    }
                }
            }
        }
    }
    acc.textures.insert(key, t);
}

/// Frame `frame` is sent to the writer: its data moves out (into the
/// capture, when `keep`: the writer will write it; its blobs count as
/// stored from now), older open frames (never sent) are dropped, and the
/// counters of frames lost since the last send are logged.
pub(super) fn take(frame: u64, keep: bool) -> Option<Taken> {
    if !on() {
        return None;
    }
    let mut accs = lock(&ACCS);
    SENT.fetch_max(frame.saturating_add(1), Ordering::Relaxed);
    let mut taken = None;
    let mut unsent = 0u32;
    accs.retain_mut(|a| {
        if a.frame == frame {
            taken = Some(std::mem::take(a));
            false
        } else if a.frame < frame {
            unsent += u32::from(a.wanted);
            false
        } else {
            true
        }
    });
    drop(accs);
    let (skipped, late) = (SKIPPED.swap(0, Ordering::Relaxed), LATE.swap(0, Ordering::Relaxed));
    // Outside the windows `seq=auto` records every frame and the probe
    // sends few (992, 993, 1141, ...): the frames between are dropped here
    // or over the open cap, 518 + 129 + 64 of them before the ped window,
    // all 12 dumped frames whole. Not a loss, so not a warning: a dumped
    // frame without data is the `pack frame N: not written` warning.
    if skipped + late + u64::from(unsent) > 0 {
        xnw_core::info!("world probe: pack: at frame {frame}: recorded frames the probe did not send: {skipped} dropped over the open cap ({OPEN_MAX}), {unsent} older dropped now; {late} late records");
    }
    let acc = taken.filter(|a| a.wanted && keep)?;
    lock(&RUN).get_or_insert_with(Run::default).stored.extend(acc.blobs.keys().copied());
    Some(Taken(acc))
}

/// Writes frame `frame`'s pack into `dir` (the seq writer's frame
/// directory); `seqs[i]` is v1 draw i's hook sequence number.
pub(super) fn write(dir: &Path, frame: u64, seqs: &[u32], taken: Option<Taken>) -> std::io::Result<()> {
    let t0 = std::time::Instant::now();
    let Some(Taken(mut acc)) = taken else {
        xnw_core::warn!("world probe: pack frame {frame}: not written: no pack data reached the writer (frame not recorded: see the 'world probe: pack:' warnings)");
        return Ok(());
    };
    // The hook-in records the hook left raw (hashing there cost its thread).
    for (seq, geom, blob) in std::mem::take(&mut acc.hookin_raw) {
        let (h, len) = (xnw_core::hash128(&[&blob]), blob.len() as u64);
        acc.blobs.entry(h).or_insert((kind::HOOKIN, blob));
        acc.refs.push(reference(fp::HOOK | seq, kind::HOOKIN, 0, geom, len, Ok(h)));
    }
    // v2.1: the frame's new blobs in one bundle file (one file per blob
    // cost ~1.3 ms each, 23,252 files = 29 s for frame 1002).
    let mut by_kind: HashMap<u16, (u64, u64)> = HashMap::new();
    let mut new: Vec<(u128, &[u8])> = Vec::with_capacity(acc.blobs.len());
    {
        let mut g = lock(&WRITTEN);
        let written = g.get_or_insert_with(HashSet::new);
        for (h, (k, b)) in &acc.blobs {
            if written.insert(*h) {
                let e = by_kind.entry(*k).or_default();
                e.0 += 1;
                e.1 += b.len() as u64;
                new.push((*h, b));
            }
        }
    }
    let new_count = new.len();
    fp::write_bundle(&fp::blob_dir(dir), &frame.to_string(), &new)?;
    let blob_ms = t0.elapsed().as_millis();
    let index: HashMap<u32, u32> = seqs.iter().enumerate().filter(|(_, s)| **s != 0).map(|(i, s)| (*s, i as u32)).collect();
    let mut per_draw = vec![0u32; seqs.len()];
    let mut unpaired = 0;
    let sinks: Vec<fp::Sink> = acc
        .sinks
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let hook_draw = index.get(&s.hook).copied();
            match hook_draw {
                Some(d) => per_draw[d as usize] += 1,
                None if fp::op::is_dispatch(s.op) => {}
                None => unpaired += 1,
            }
            fp::Sink { index: i as u32, thread: s.thread, hook_draw, op: s.op, count: s.count, pipeline: s.pipeline }
        })
        .collect();
    let pairs = [per_draw.iter().filter(|n| **n == 0).count() as u32, per_draw.iter().filter(|n| **n == 1).count() as u32, per_draw.iter().filter(|n| **n > 1).count() as u32];
    // Dispatch sinks, those with their item and all inputs' bytes.
    let dispatches = sinks.iter().filter(|s| s.op == fp::op::DISPATCH).count();
    let (sky_sinks, indirect_dispatches, strand_extra, head_uavs) = (acc.sky_sinks, acc.indirect_dispatches, acc.strand_extra, acc.head_uavs);
    let with_item = acc.refs.iter().filter(|r| r.kind == kind::DISPATCH && r.addr != 0).count();
    let dispatch_ms = acc.dispatch_ns as f64 / 1e6;
    let seen = acc.dispatches_seen;
    let (imposters, imp_parts, wheel_windows) = (acc.imposters, acc.imposter_parts, acc.wheel_windows);
    let (words, sampler_counts, sampler_heaps, alias, disp_maps, draw_args, hair) = (acc.draw_words, acc.sampler_counts, acc.sampler_heaps.len(), acc.alias, acc.disp_maps, acc.draw_args, acc.hair);
    let hookin = acc.hookin;
    let (n, ns) = (HOOKIN_COST[0].swap(0, Ordering::Relaxed), HOOKIN_COST[1].swap(0, Ordering::Relaxed));
    let hookin_line = if HOOKIN.load(Ordering::Relaxed) { format!("; hook inputs {hookin} (cost {:.2} us a hook draw over {n})", ns as f64 / 1e3 / n.max(1) as f64) } else { String::new() };
    let head_sinks = acc.head_sinks;
    let (light_sinks, lights) = (acc.light_sinks, if acc.lights_done { acc.lights_copied.to_string() } else { "not read".into() });
    let mut reasons: HashMap<(u16, u32), u64> = HashMap::new();
    for r in acc.refs.iter().filter(|r| r.reason != reason::OK) {
        *reasons.entry((r.kind, r.reason)).or_default() += 1;
    }
    let mut textures: Vec<fp::Texture> = acc.textures.into_values().collect();
    textures.sort_by_key(|t| t.at);
    for t in textures.iter().filter(|t| t.reason != reason::OK) {
        *reasons.entry((u16::MAX, t.reason)).or_default() += 1;
    }
    let mut bytes: Vec<(u16, u64, u64)> = by_kind.into_iter().map(|(k, (n, b))| (k, n, b)).collect();
    bytes.sort_unstable();
    let mut reasons: Vec<(u16, u32, u64)> = reasons.into_iter().map(|((k, r), n)| (k, r, n)).collect();
    reasons.sort_unstable();
    let manifest = fp::Manifest {
        version: fp::VERSION,
        frame,
        commit: option_env!("XNW_COMMIT").unwrap_or("-").to_string(),
        hook_draws: seqs.len() as u32,
        sink_draws: (sinks.len() - dispatches - indirect_dispatches as usize) as u32,
        pairs,
        unpaired_sinks: unpaired,
        dispatches: [dispatches as u32, acc.dispatches_seen],
        bytes,
        reasons,
    };
    // Hook-side records: sequence number -> v1 draw (dropped when not captured).
    let refs: Vec<Ref> = acc
        .refs
        .into_iter()
        .filter_map(|r| {
            if r.owner == fp::FRAME_SCOPE || r.owner & fp::HOOK == 0 {
                return Some(r);
            }
            index.get(&(r.owner & !fp::HOOK)).map(|d| Ref { owner: fp::HOOK | d, ..r })
        })
        .collect();
    let mut pack = FramePack { manifest, refs, sinks, programs: acc.programs, textures };
    pack.write(dir)?;
    xnw_core::info!("world probe: pack frame {frame}: {dispatches} pre-skin dispatches recorded ({with_item} with their item) of {seen} direct dispatches, recorded in {dispatch_ms:.1} ms");
    xnw_core::info!(
        "world probe: pack frame {frame}: bridge: draw words {} of {} draw sinks ({} views resolved); sampler entries {} in {sampler_heaps} heap reads ({} sinks without a sampler heap); root views through the CPU alias {} ({} without bytes; {} over a hook draw's stream, {} over a 64 KiB window); pre-skin displacement maps {} ({} without a view)",
        words[0],
        pack.manifest.sink_draws,
        words[2],
        sampler_counts[0],
        sampler_counts[1],
        alias[0],
        alias[1],
        alias[2],
        alias[3],
        disp_maps[0],
        disp_maps[1]
    );
    xnw_core::info!(
        "world probe: pack frame {frame}: draw args on {draw_args} Draw/DrawIndexed sinks; hair: {} input dispatch sinks, {} order-only, strand group records {} (heap views {} with bytes, {} without; palette windows {} with bytes, {} without; collision spans {} with bytes, {} without); lighting: {light_sinks} dispatch sinks ({sky_sinks} sky/ambient), {indirect_dispatches} indirect dispatch sinks, scene lights {lights}; head graph: {head_sinks} dispatch sinks (unordered buffer views {} read, {} not){hookin_line}",
        hair[0],
        hair[1],
        hair[2],
        hair[3],
        hair[4],
        strand_extra[0],
        strand_extra[1],
        strand_extra[2],
        strand_extra[3],
        head_uavs[0],
        head_uavs[1]
    );
    xnw_core::info!("world probe: pack frame {frame}: {imposters} imposter calls ({} with instance records, {} with their type record); wheel palette windows {} ({} without bytes)", imp_parts[0], imp_parts[1], wheel_windows[0], wheel_windows[1]);
    xnw_core::info!(
        "world probe: pack frame {frame}: {} sink draws ({} paired 1:1 with {} hook draws, {unpaired} unpaired), {} programs, {} textures, {:.1} MB new blobs ({} blobs, one bundle file), written in {} ms (blobs {blob_ms} ms)",
        pack.manifest.sink_draws,
        pack.manifest.pairs[1],
        seqs.len(),
        pack.programs.len(),
        pack.textures.len(),
        pack.manifest.bytes.iter().map(|b| b.2).sum::<u64>() as f64 / 1e6,
        new_count,
        t0.elapsed().as_millis()
    );
    Ok(())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// The writer's state is global: its tests run one at a time, each from
    /// a clean state.
    static SERIAL: Mutex<()> = Mutex::new(());

    pub(in crate::worldprobe) fn fresh() -> std::sync::MutexGuard<'static, ()> {
        let g = lock(&SERIAL);
        ON.store(true, Ordering::Relaxed);
        SENT.store(0, Ordering::Relaxed);
        WANT.store(0, Ordering::Relaxed);
        lock(&ACCS).clear();
        *lock(&RUN) = None;
        LATE.store(0, Ordering::Relaxed);
        SKIPPED.store(0, Ordering::Relaxed);
        *lock(&WRITTEN) = None;
        *lock(&DISPATCH_PIPES) = None;
        ITEM_LISTS.store(0, Ordering::Relaxed);
        g
    }

    /// The writer end to end on synthetic memory: a paired and an unpaired
    /// sink draw (root constants, a table of empty slots, a root view), the
    /// texture paths (texels, zero-filled, null), the hook records (palette,
    /// model, handler head, a handle that names no view), written and read back.
    #[test]
    fn sinks_and_textures_round_trip_through_the_reader() {
        let _one = fresh();
        let frame = super::super::frame_key();
        let dir = std::env::temp_dir().join(format!("xnw-pk1-{}", std::process::id())).join(format!("frame-{frame}"));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        let table = vec![0u8; 4 * 32];
        let consts = [1u32, 2, 3];
        let roots = [TapRoot::Constants(&consts), TapRoot::Table { handle: table.as_ptr() as u64, heap_start: table.as_ptr() as u64, slots: 4 }, TapRoot::View(0x1234_0000)];
        let seq = hook_entry(frame);
        on_sink(&TapDraw { op: fp::op::MESH, count: 12, pipeline: 0, yz: [0; 2], args: [0; 4], view_heap: 0, sampler_heap: 0, roots: &roots });
        HOOK.with(|h| h.set((u64::MAX, 0)));
        on_sink(&TapDraw { op: fp::op::INDEXED, count: 36, pipeline: 0, yz: [0; 2], args: [0; 4], view_heap: 0, sampler_heap: 0, roots: &roots[..1] });
        let pal: Vec<u8> = (0..96u32).map(|i| i as u8).collect();
        let model = vec![7u8; 0x40];
        let handler: Vec<u8> = (0..0x400u32).map(|i| (i >> 2) as u8).collect();
        let mut entity = vec![0u8; 0x40];
        entity[0x28..0x30].copy_from_slice(&(handler.as_ptr() as u64).to_le_bytes());
        hook_objects(frame, seq, (pal.as_ptr() as u64, 2), 0, model.as_ptr() as u64, entity.as_ptr() as u64);
        let slot = vec![0u8; 32];
        hook_views(frame, &[slot.as_ptr() as u64, 0, slot.as_ptr() as u64], &[]);
        let texels: Vec<u8> = (0..8192u32).map(|i| (i % 251) as u8 + 1).collect();
        let zeros = vec![0u8; 4096];
        {
            let mut accs = lock(&ACCS);
            let acc = accs.iter_mut().find(|a| a.frame == frame).unwrap();
            let mut run = lock(&RUN);
            let run = run.get_or_insert_with(Run::default);
            for (addr, bytes) in [(texels.as_ptr() as u64, 8192), (zeros.as_ptr() as u64, 4096), (0, 64)] {
                texture(acc, run, &SlotView { kind: view::PLACED, addr, bytes, ..SlotView::default() });
            }
        }
        write(&dir, frame, &[0, seq], take(frame, true)).unwrap();
        let p = FramePack::open(&dir).unwrap().unwrap();
        assert_eq!((p.manifest.hook_draws, p.manifest.sink_draws, p.manifest.pairs, p.manifest.unpaired_sinks), (2, 2, [1, 1, 0], 1));
        assert_eq!(p.sinks[0].hook_draw, Some(1));
        assert_eq!(p.sinks[1].hook_draw, None);
        let first: Vec<(u16, u32)> = p.refs_of(0).map(|r| (r.kind, r.reason)).collect();
        assert_eq!(first, [(kind::ROOTS, 0), (kind::TABLE, 0), (kind::ROOT_VIEW, reason::UNREADABLE), (kind::SAMPLERS, reason::NULL), (kind::DRAW_WORDS, reason::NULL)]);
        let roots = p.refs_of(0).next().unwrap();
        assert_eq!(FramePack::blob(&dir, roots.hash).unwrap(), [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
        let t = |at: u64| p.textures.iter().find(|t| t.at == at).unwrap().clone();
        let full = t(texels.as_ptr() as u64);
        assert_eq!((full.reason, full.texel_bytes), (0, 8192));
        assert_eq!(FramePack::blob(&dir, full.texels).unwrap(), texels);
        assert_eq!(t(zeros.as_ptr() as u64).reason, reason::ZERO_FILLED);
        assert_eq!(t(0).reason, reason::NULL);
        let hook: Vec<(u16, u16, u32)> = p.refs_of(fp::HOOK | 1).map(|r| (r.kind, r.slot, r.reason)).collect();
        assert_eq!(hook, [(kind::PALETTE, 0, 0), (kind::PALETTE, 1, reason::NULL), (kind::MODEL, 0, 0), (kind::HANDLER, 0, 0)]);
        assert_eq!(FramePack::blob(&dir, p.refs_of(fp::HOOK | 1).next().unwrap().hash).unwrap(), pal);
        let h = p.refs_of(fp::HOOK | 1).find(|r| r.kind == kind::HANDLER).unwrap();
        assert_eq!((h.addr, FramePack::blob(&dir, h.hash).unwrap()), (handler.as_ptr() as u64, handler.clone()));
        // The handle is resolved once per frame; a slot that holds no view says so.
        let views: Vec<(u64, u32)> = p.refs_of(fp::FRAME_SCOPE).filter(|r| r.kind == kind::HANDLE_VIEW).map(|r| (r.addr, r.reason)).collect();
        assert_eq!(views, [(slot.as_ptr() as u64, reason::NOT_RESIDENT)]);
        let c = fp::Coverage::of(&p, &|_| true);
        assert_eq!((c.sinks, c.paired, c.textures, c.textures_held), (2, 1, 3, 1));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// A ped pre-skin dispatch end to end on synthetic memory: only
    /// the effect's pipelines are sinks (every direct dispatch is counted);
    /// the sink keeps the compute roots, b7 and the 144-B item found in the
    /// title's lists by b7's bytes, the bytes behind its input views, no
    /// bytes of a view the GPU writes or of one over the cap, and the heap
    /// views b7 names; it never pairs with the thread's hook draw. Written
    /// and read back.
    #[test]
    fn preskin_dispatch_keeps_its_item_inputs_and_heap_views() {
        let _one = fresh();
        let frame = 9_400_000u64;
        let dir = std::env::temp_dir().join(format!("xnw-ped1-{}", std::process::id())).join(format!("frame-{frame}"));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        // Two items chained under head 13 (list 1, bucket 0); the second is ours.
        let word = 0x6100_0040u32;
        let mut item = vec![0u8; 144];
        for (o, w) in [(32 + 0x34, word), (32 + 0x38, 1000), (32 + 0x40, 70_001), (32 + 0x44, 70_002), (32 + 0x48, 70_003), (132, 1000), (136, word)] {
            item[o..o + 4].copy_from_slice(&w.to_le_bytes());
        }
        let mut other = vec![1u8; 144];
        other[..8].copy_from_slice(&(item.as_ptr() as u64).to_le_bytes());
        let mut heads = vec![0u8; 8 * ITEM_HEADS as usize];
        heads[13 * 8..14 * 8].copy_from_slice(&(other.as_ptr() as u64).to_le_bytes());
        set_preskin_lists(heads.as_ptr() as u64);
        // The constant buffer the title bound (its copy of item+32..+127), a palette, a stream.
        let b7 = item[32..128].to_vec();
        let palette: Vec<u8> = (0..48u8).collect();
        let stream: Vec<u8> = (0..200u8).collect();
        // Every direct dispatch is counted; only the effect's pipeline is a sink.
        let seq = hook_entry(frame);
        assert!(!want_dispatch_at(frame, 0x10, |_| PIPE_NONE));
        assert!(want_dispatch_at(frame, 0x20, |_| PIPE_PED));
        assert!(want_dispatch_at(frame, 0x20, |_| unreachable!("asked once per pipeline")));
        let consts = [9u32, 8];
        sink_at(frame, &TapDraw { op: fp::op::DISPATCH, count: 32, yz: [1, 1], args: [0; 4], pipeline: 0, view_heap: 0, sampler_heap: 0, roots: &[TapRoot::Constants(&consts)] });
        {
            let mut accs = lock(&ACCS);
            let acc = accs.iter_mut().find(|a| a.frame == frame).unwrap();
            // No constant buffer bound: a record without b7, no heap views, no walk.
            assert_eq!((acc.sinks[0].hook, acc.item_walks), (0, 0));
            let r = *acc.refs.iter().find(|r| r.kind == kind::DISPATCH).unwrap();
            assert_eq!((r.owner, r.addr, fp::Dispatch::decode(&acc.blobs[&r.hash].1).unwrap().flags), (0, 0, 0));
            // The second dispatch, with its views as the table resolved them (ud5 = root 5, ud4 = root 4).
            let mut run = lock(&RUN);
            let run = run.get_or_insert_with(Run::default);
            acc.sinks.push(SinkRaw { thread: 0, hook: 0, op: fp::op::DISPATCH, count: 32, pipeline: 0 });
            let view = |slot: u32, kind: u32, b: &[u8]| SlotView { slot, kind, addr: b.as_ptr() as u64, bytes: b.len() as u64, ..SlotView::default() };
            view_bytes(acc, run, 1, 5, &view(8, view::CONSTANT, &b7), 0, TB_DISPATCH);
            view_bytes(acc, run, 1, 4, &view(1, view::BUFFER, &palette), 0, TB_DISPATCH);
            view_bytes(acc, run, 1, 4, &SlotView { layout: view::UNORDERED, bytes: 26_214_400, ..view(24, view::BUFFER, &stream) }, 0, TB_DISPATCH);
            view_bytes(acc, run, 1, 4, &SlotView { bytes: BUF_CAP + 1, ..view(2, view::BUFFER, &stream) }, 0, TB_DISPATCH);
            dispatch_record(acc, run, 1, [32, 1, 1], 0);
            assert_eq!(acc.item_walks, 1);
            // Heap index 2 as the heap resolved it (no heap object in a test: 0 and 1 stay unresolved).
            acc.refs.retain(|r| !(r.owner == 1 && r.kind == kind::HEAP_VIEW && r.slot == 2));
            heap_view(acc, run, 1, 2, 70_003, Some((0x5000, view(70_003, view::BUFFER, &stream))));
        }
        write(&dir, frame, &[seq], take(frame, true)).unwrap();
        let p = FramePack::open(&dir).unwrap().unwrap();
        assert_eq!((p.manifest.sink_draws, p.manifest.dispatches, p.manifest.pairs, p.manifest.unpaired_sinks), (0, [2, 3], [1, 0, 0], 0));
        assert_eq!(p.sinks.iter().map(|s| (s.op, s.count, s.hook_draw)).collect::<Vec<_>>(), [(fp::op::DISPATCH, 32, None); 2]);
        let got: Vec<(u16, u16, u32)> = p.refs_of(1).map(|r| (r.kind, r.slot, r.reason)).collect();
        assert_eq!(
            got,
            [
                (kind::CBV, 5 * 96 + 8, 0),
                (kind::BUFFER, 4 * 96 + 1, 0),
                (kind::BUFFER, 4 * 96 + 2, reason::BUDGET),
                (kind::BUFFER, 4 * 96 + 24, reason::GPU_OUTPUT),
                (kind::DISPATCH, 0, 0),
                (kind::HEAP_VIEW, 0, reason::IMPLAUSIBLE),
                (kind::HEAP_VIEW, 1, reason::IMPLAUSIBLE),
                (kind::HEAP_VIEW, 2, 0),
                (kind::HEAP_BYTES, 2, 0),
            ]
        );
        let blob = |k: u16, slot: u16| FramePack::blob(&dir, p.refs_of(1).find(|r| r.kind == k && r.slot == slot).unwrap().hash).unwrap();
        let r = p.refs_of(1).find(|r| r.kind == kind::DISPATCH).unwrap();
        let d = fp::Dispatch::decode(&blob(kind::DISPATCH, 0)).unwrap();
        assert_eq!((r.addr, d.groups, d.flags, d.b7_slot, d.word, d.vertices), (item.as_ptr() as u64, [32, 1, 1], fp::B7_FOUND | fp::ITEM_FOUND, 5 * 96 + 8, word, 1000));
        assert_eq!((&d.b7, &d.item), (&b7, &item));
        assert_eq!(blob(kind::BUFFER, 4 * 96 + 1), palette);
        assert_eq!(fp::decode_slots(&blob(kind::HEAP_VIEW, 2))[0].slot, 70_003);
        assert_eq!(blob(kind::HEAP_BYTES, 2), stream);
        assert_eq!(p.refs_of(1).find(|r| r.kind == kind::HEAP_VIEW && r.slot == 0).unwrap().addr, 70_001);
        // A chain that loops ends at the cap; an unreadable head list reads nothing.
        let mut ring = vec![0u8; 144];
        let at = ring.as_ptr() as u64;
        ring[..8].copy_from_slice(&at.to_le_bytes());
        heads[13 * 8..14 * 8].copy_from_slice(&at.to_le_bytes());
        assert_eq!(walk_items().len(), ITEMS_MAX);
        set_preskin_lists(0x10);
        assert!(walk_items().is_empty());
        set_preskin_lists(0);
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// An earlier fault: 200 frames recorded and sent while the writer is still
    /// on the first (the title ran 69 frames ahead of it in one capture; the old
    /// 64-frame cap evicted frames 1004-1007). Every dumped frame keeps its
    /// sinks; frames not dumped hold nothing; at most one frame stays open.
    #[test]
    fn frames_sent_faster_than_written_keep_their_pack() {
        let _one = fresh();
        let base = 9_000_000u64;
        let root = std::env::temp_dir().join(format!("xnw-pk2-seq-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let consts = [5u32];
        let roots = [TapRoot::Constants(&consts)];
        let dumped = |f: u64| f < 6 || f % 50 == 49;
        let mut queue = Vec::new();
        for f in 0..200u64 {
            let seq = hook_entry(base + f);
            sink_at(base + f, &TapDraw { op: fp::op::INDIRECT_MESH, count: 1, pipeline: 0, yz: [0; 2], args: [0; 4], view_heap: 0, sampler_heap: 0, roots: &roots });
            let taken = take(base + f, dumped(f));
            assert_eq!(taken.is_some(), dumped(f), "frame {f}");
            assert!(lock(&ACCS).is_empty());
            queue.push((f, seq, taken));
        }
        // A sink of a frame already sent is late: not recorded.
        sink_at(base + 3, &TapDraw { op: fp::op::MESH, count: 1, pipeline: 0, yz: [0; 2], args: [0; 4], view_heap: 0, sampler_heap: 0, roots: &roots });
        assert!(lock(&ACCS).is_empty());
        for (f, seq, taken) in queue.into_iter().filter(|q| q.2.is_some()) {
            let dir = root.join(format!("frame-{}", base + f));
            write(&dir, base + f, &[seq], taken).unwrap();
            let p = FramePack::open(&dir).unwrap().unwrap();
            assert_eq!((p.manifest.sink_draws, p.manifest.pairs), (1, [0, 1, 0]), "frame {f}");
            assert_eq!(p.sinks[0].op, fp::op::INDIRECT_MESH);
            // The root constants' blob is reachable from every frame.
            assert_eq!(FramePack::blob(&dir, p.refs_of(0).next().unwrap().hash).unwrap(), 5u32.to_le_bytes());
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// Root views over the title's own pages: a GPU address whose CPU
    /// alias lies in the hook draw's vertex stream, in no buffer object,
    /// is read there to the stream's end; an address in no buffer and no
    /// stream gets a window, marked cut; the log counts say which.
    #[test]
    fn root_views_over_hook_streams() {
        let _one = fresh();
        let frame = 9_700_000u64;
        let dir = std::env::temp_dir().join(format!("xnw-pk7-{}", std::process::id())).join(format!("frame-{frame}"));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        let vb: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
        let cpu = vb.as_ptr() as u64;
        let loose = vec![0x77u8; ROOT_WINDOW as usize];
        let seq = hook_entry(frame);
        hook_views(frame, &[], &[(cpu, vb.len() as u64), (0, 64)]);
        let roots = [TapRoot::None, TapRoot::None, TapRoot::View(cpu + 16 + GPU_ALIAS), TapRoot::View(loose.as_ptr() as u64)];
        sink_at(frame, &TapDraw { op: fp::op::INDIRECT_MESH, count: 1, pipeline: 0, yz: [0; 2], args: [0; 4], view_heap: 0, sampler_heap: 0, roots: &roots });
        {
            let mut accs = lock(&ACCS);
            let acc = acc_of(&mut accs, frame).unwrap();
            assert_eq!((acc.alias, acc.streams.len()), ([1, 0, 1, 1], 1));
        }
        write(&dir, frame, &[seq], take(frame, true)).unwrap();
        let p = FramePack::open(&dir).unwrap().unwrap();
        let rv = |slot: u16| *p.refs_of(0).find(|r| r.kind == kind::ROOT_VIEW && r.slot == slot).unwrap();
        assert_eq!((rv(2).addr, rv(2).len, rv(2).reason), (cpu + 16, 240, reason::OK));
        assert_eq!(FramePack::blob(&dir, rv(2).hash).unwrap(), vb[16..].to_vec());
        assert_eq!((rv(3).addr, rv(3).reason), (loose.as_ptr() as u64, reason::CUT));
        assert_eq!(FramePack::blob(&dir, rv(3).hash).unwrap(), loose);
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// A slot-0 record with +0x74 != -1 keeps the arena words
    /// at base + rec+0x74; -1 keeps nothing; the base is noted once.
    #[test]
    fn arena_words_follow_the_record() {
        let _one = fresh();
        let frame = 9_200_000u64;
        let arena: Vec<u8> = (0..1024u32).map(|i| i as u8).collect();
        let global = (arena.as_ptr() as u64).to_le_bytes();
        set_arena(global.as_ptr() as u64);
        let mut rec = vec![0u8; 128];
        rec[0x74..0x78].copy_from_slice(&64u32.to_le_bytes());
        let mut none = vec![0u8; 128];
        none[0x74..0x78].copy_from_slice(&u32::MAX.to_le_bytes());
        {
            let mut accs = lock(&ACCS);
            let acc = acc_of(&mut accs, frame).unwrap();
            let mut run = lock(&RUN);
            let run = run.get_or_insert_with(Run::default);
            arena_words(acc, run, 0, 96, rec.as_ptr() as u64);
            arena_words(acc, run, 1, 96, none.as_ptr() as u64);
            arena_words(acc, run, 2, 192, rec.as_ptr() as u64);
            let got: Vec<(u32, u16, u64)> = acc.refs.iter().map(|r| (r.owner, r.slot, r.addr)).collect();
            let at = arena.as_ptr() as u64;
            assert_eq!(got, [(fp::FRAME_SCOPE, 0, at), (0, 96, at + 64), (2, 192, at + 64)]);
            assert_eq!(acc.blobs.get(&acc.refs[1].hash).map(|b| b.1.clone()), Some(arena[64..64 + ARENA_WORDS as usize].to_vec()));
        }
        set_arena(0);
    }

    /// An imposter call keeps its arguments, IcRec, desc, T and
    /// n records; an unreadable IcRec still leaves a record (flags 0).
    #[test]
    fn imposter_calls_keep_their_inputs() {
        let _one = fresh();
        let frame = 9_300_000u64;
        let arena = vec![0x5Au8; 0x200];
        let global = (arena.as_ptr() as u64).to_le_bytes();
        set_arena(global.as_ptr() as u64);
        let recs: Vec<u8> = (0..128u32).map(|i| i as u8).collect();
        let inst: Vec<u8> = (0..128u32).map(|i| 200 - i as u8).collect();
        let mut l = vec![0u8; 0xA0];
        l[0x90..0x98].copy_from_slice(&(inst.as_ptr() as u64).to_le_bytes());
        l[0x98..0x9A].copy_from_slice(&2u16.to_le_bytes());
        let visible = [1u32, 0];
        let args = [6u32, 2, 0, 0, 0];
        let mut desc = vec![0u8; 0x100];
        desc[..8].copy_from_slice(&(l.as_ptr() as u64).to_le_bytes());
        desc[8..12].copy_from_slice(&2u32.to_le_bytes());
        desc[0x78..0x80].copy_from_slice(&(recs.as_ptr() as u64).to_le_bytes());
        let t_off = [0x10u8, 0, 0, 0, 0, 0, 0, 0, 0x40, 0, 0, 0];
        let mut ic = vec![0u8; 0x100];
        ic[..8].copy_from_slice(&(desc.as_ptr() as u64).to_le_bytes());
        ic[0x48..0x50].copy_from_slice(&(visible.as_ptr() as u64).to_le_bytes());
        ic[0x50..0x54].copy_from_slice(&2u32.to_le_bytes());
        ic[0x58..0x60].copy_from_slice(&(args.as_ptr() as u64).to_le_bytes());
        ic[0x60..0x68].copy_from_slice(&(t_off.as_ptr() as u64).to_le_bytes());
        let handler: Vec<u8> = (0..0x400u32).map(|i| (i % 251) as u8).collect();
        // Type 0x181: home group 1 (full, no match) then group 0, slot 3 -> index 2.
        let types: Vec<u8> = (0..3 * 224u32).map(|i| (i / 224) as u8 + 1).collect();
        let mut groups = vec![0x10u8; 2 * 144];
        groups[..16].fill(0x80);
        groups[3] = 0x01;
        groups[16 + 24..16 + 32].copy_from_slice(&[0x81, 0x01, 0, 0, 2, 0, 0, 0]);
        let mut globals = (types.as_ptr() as u64).to_le_bytes().to_vec();
        globals.extend(2u64.to_le_bytes());
        globals.extend((groups.as_ptr() as u64).to_le_bytes());
        set_imposter_types(globals.as_ptr() as u64);
        assert_eq!(imposter_type(0x182), (0, Err(reason::NULL)));
        imposter(frame, [0x181, 66, 3, handler.as_ptr() as u64 + 0x308, 2, 1], ic.as_ptr() as u64);
        imposter(frame, [7, 66, 3, 0, 0, 0], 0);
        let mut accs = lock(&ACCS);
        let acc = acc_of(&mut accs, frame).unwrap();
        let r = acc.refs[0];
        assert_eq!((r.owner, r.kind, r.slot, r.addr), (fp::FRAME_SCOPE, kind::IMPOSTER, 0, ic.as_ptr() as u64));
        let b = &acc.blobs[&r.hash].1;
        let w = |i: usize| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
        assert_eq!((w(0), w(1), w(2), w(6), w(8), w(10), w(11)), (0x181, 66, 3, ic.as_ptr() as u64, 2, arena.as_ptr() as u64 + 0x40, 0xF));
        assert_eq!(b.len(), 96 + 0x290 + 128);
        assert_eq!(&b[96 + 0x200..96 + 0x200 + 4], &[0x5A; 4]);
        assert_eq!(&b[96 + 0x290..], &recs[..]);
        // Its instances: L's head, both records, the visible words, the arguments.
        let r = acc.refs[1];
        assert_eq!((r.owner, r.kind, r.slot, r.addr, r.reason), (fp::FRAME_SCOPE, kind::IMP_INST, 0, l.as_ptr() as u64, 0));
        let i = fp::ImpInstances::decode(&acc.blobs[&r.hash].1).unwrap();
        assert_eq!((i.records_at, i.count, i.n, i.visible_n, i.handler_at, i.type_at, i.reasons), (inst.as_ptr() as u64, 2, 2, 2, handler.as_ptr() as u64, types.as_ptr() as u64 + 448, [0; 6]));
        assert_eq!(i.type_record, types[448..].to_vec());
        assert_eq!(acc.imposter_parts, [1, 1]);
        assert_eq!((&i.head[..], &i.records[..], &i.visible[..], &i.handler[..]), (&l[..], &inst[..], &visible[..], &handler[..]));
        assert_eq!(i.args, args.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>());
        // The unreadable call: its records and instances with reasons, same slot.
        assert_eq!((acc.refs[2].slot, acc.blobs[&acc.refs[2].hash].1[88]), (1, 0));
        let i = fp::ImpInstances::decode(&acc.blobs[&acc.refs[3].hash].1).unwrap();
        assert_eq!((acc.refs[3].kind, acc.refs[3].slot, i.reasons), (kind::IMP_INST, 1, [reason::NULL, reason::NULL, reason::NULL, reason::NULL, reason::IMPLAUSIBLE, reason::NULL]));
        drop(accs);
        set_arena(0);
        set_imposter_types(0);
    }

    /// An instanced wheel draw keeps 256 bones of its palette at each
    /// distinct instance base: whole, cut at the view's end, past the end a
    /// reason; a draw without its views one record with a reason.
    #[test]
    fn wheel_draws_keep_their_palette_windows() {
        let _one = fresh();
        let frame = 9_500_000u64;
        let palette: Vec<u8> = (0..48 * 400u32).map(|i| (i / 48) as u8).collect();
        let bases: Vec<u8> = [10u32, 10, 100, 290, 1000].iter().flat_map(|b| b.to_le_bytes()).collect();
        let meshlets = 2u32.to_le_bytes();
        let view = |slot: u32, b: &[u8]| Some(SlotView { slot, kind: view::BUFFER, addr: b.as_ptr() as u64, bytes: b.len() as u64, ..SlotView::default() });
        let _ = hook_entry(frame);
        let mut accs = lock(&ACCS);
        let acc = acc_of(&mut accs, frame).unwrap();
        let mut run = lock(&RUN);
        let run = run.get_or_insert_with(Run::default);
        wheel_windows(acc, run, 3, 10, [view(5, &palette), view(8, &bases), view(4, &meshlets)]);
        wheel_windows(acc, run, 4, 10, [None, view(8, &bases), view(4, &meshlets)]);
        let at = palette.as_ptr() as u64;
        let got: Vec<(u32, u16, u64, u32, u32)> = acc.refs.iter().map(|r| (r.owner, r.slot, r.addr, r.len, r.reason)).collect();
        assert_eq!(
            got,
            [
                (3, 0, at + 480, 12_288, 0),
                (3, 1, at + 4800, 12_288, 0),
                (3, 2, at + 48 * 290, 48 * 110, reason::CUT),
                (3, 3, at + 48_000, 0, reason::IMPLAUSIBLE),
                (4, 0, 0, 0, reason::NULL),
            ]
        );
        assert!(acc.refs.iter().all(|r| r.kind == kind::PAL_WINDOW));
        assert_eq!(acc.wheel_windows, [3, 2]);
        assert_eq!(acc.blobs[&acc.refs[1].hash].1, palette[4800..4800 + 12_288].to_vec());
        assert_eq!(acc.blobs[&acc.refs[2].hash].1.len(), 48 * 110);
        // Instances past the meshlet count are not read: 3 groups of 2 = one instance.
        acc.refs.clear();
        wheel_windows(acc, run, 5, 3, [view(5, &palette), view(8, &bases), view(4, &meshlets)]);
        assert_eq!(acc.refs.iter().map(|r| r.addr).collect::<Vec<_>>(), [at + 480]);
    }

    /// The bridge inputs on synthetic memory: the draw words at slot 86 +
    /// the offset (past the view a reason), their views by heap index
    /// (texture noted, empty and outside-the-heap words left out), the
    /// sampler heap read once per heap (zero slots left out, a record of
    /// ours with its description, raw words kept as they are), and a root
    /// view read through the CPU alias. Written and read back.
    #[test]
    fn bridge_inputs_keep_draw_words_samplers_and_alias() {
        let _one = fresh();
        let frame = 9_600_000u64;
        let dir = std::env::temp_dir().join(format!("xnw-pk6-{}", std::process::id())).join(format!("frame-{frame}"));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        // Slot 86's buffer: the words at +0x100 (heap indices 9, 9, 12, 7, 70000, then zeros).
        let mut buf = vec![0u8; 0x200];
        for (k, w) in [9u32, 9, 12, 7, 70_000].iter().enumerate() {
            buf[0x100 + 4 * k..0x104 + 4 * k].copy_from_slice(&w.to_le_bytes());
        }
        let slot86 = SlotView { slot: 86, kind: view::PLACED_BUFFER, addr: buf.as_ptr() as u64, bytes: buf.len() as u64, ..SlotView::default() };
        assert_eq!(words_at(Some(slot86), Some(0x100)), Ok(buf.as_ptr() as u64 + 0x100));
        assert_eq!(words_at(Some(slot86), Some(0x1C1)), Err(reason::IMPLAUSIBLE));
        assert_eq!(words_at(Some(slot86), None), Err(reason::NULL));
        assert_eq!(words_at(Some(SlotView { kind: view::TEXTURE, ..slot86 }), Some(0)), Err(reason::NULL));
        assert_eq!(words_at(None, Some(0)), Err(reason::NULL));
        let texels: Vec<u8> = (0..4096u32).map(|i| (i % 249) as u8 + 1).collect();
        let heap = 0x7_0000u64;
        let resolve = |w: u32| match w {
            9 => Some((heap, SlotView { slot: 0, kind: view::PLACED, addr: texels.as_ptr() as u64, bytes: 4096, width: 32, height: 32, ..SlotView::default() })),
            12 => Some((heap, SlotView { slot: 0, kind: view::CONSTANT, addr: 0x9000, bytes: 256, ..SlotView::default() })),
            0 | 7 => Some((heap, SlotView::default())),
            _ => None,
        };
        // Sampler heap: slot 0 zero, slot 1 a record of ours, slot 2 raw words.
        let mut sheap = vec![0u8; 3 * 16];
        for (k, w) in [5u32, 2, 1, 0x4549_5676, 0xA, 0xB, 0xC, 0xD].iter().enumerate() {
            sheap[16 + 4 * k..20 + 4 * k].copy_from_slice(&w.to_le_bytes());
        }
        let console: [u32; 27] = std::array::from_fn(|i| i as u32 + 1);
        let sresolve = |raw: [u32; 4]| (raw[3] == 0x4549_5676).then_some((fp::SAMPLER_CONSOLE, console));
        let range = Some((sheap.as_ptr() as u64, 3));
        // A root view the GPU address of a 64-B buffer names.
        let vb = vec![0x33u8; 64];
        let cpu = vb.as_ptr() as u64;
        let left = |a: u64| (a == cpu).then_some(64);
        let none = |_: u64| None;
        assert_eq!(root_view_at(cpu + GPU_ALIAS, left, none), (cpu, 64, true, ROOT_BUFFER));
        assert_eq!(root_view_at(cpu, left, none), (cpu, 64, false, ROOT_BUFFER));
        assert_eq!(root_view_at(0x1234, left, none), (0x1234, ROOT_WINDOW, false, ROOT_WINDOWED));
        assert_eq!(root_view_at(0, left, none), (0, 0, false, ROOT_NONE));
        // The title's vertex streams are in no buffer object; the
        // hook draw's stream is read at the GPU address - GPU_ALIAS, from
        // inside it to its end; a buffer object comes first.
        let mut streams = BTreeMap::new();
        streams.insert(cpu, 64u64);
        let stream = |x: u64| stream_left(&streams, x);
        let no_buffer = |_: u64| None;
        assert_eq!(root_view_at(cpu + GPU_ALIAS, no_buffer, stream), (cpu, 64, true, ROOT_STREAM));
        assert_eq!(root_view_at(cpu + GPU_ALIAS + 16, no_buffer, stream), (cpu + 16, 48, true, ROOT_STREAM));
        assert_eq!(root_view_at(cpu + 8, no_buffer, stream), (cpu + 8, 56, false, ROOT_STREAM));
        assert_eq!(root_view_at(cpu + GPU_ALIAS + 64, no_buffer, stream).3, ROOT_WINDOWED);
        assert_eq!(root_view_at(cpu + GPU_ALIAS, |x: u64| (x == cpu + GPU_ALIAS).then_some(32), stream), (cpu + GPU_ALIAS, 32, false, ROOT_BUFFER));
        let seq = hook_entry(frame);
        {
            let mut accs = lock(&ACCS);
            let acc = acc_of(&mut accs, frame).unwrap();
            let mut run = lock(&RUN);
            let run = run.get_or_insert_with(Run::default);
            for owner in 0..3u32 {
                acc.sinks.push(SinkRaw { thread: 0, hook: seq, op: fp::op::INDIRECT_MESH, count: 1, pipeline: 0 });
                let at = if owner == 2 { words_at(Some(slot86), Some(0x1C1)) } else { words_at(Some(slot86), Some(0x100)) };
                draw_words(acc, run, owner, at, &resolve);
                samplers(acc, run, owner, if owner == 2 { 0 } else { 0x55 }, if owner == 2 { None } else { range }, &sresolve);
            }
            let (at, full, alias, _) = root_view_at(cpu + GPU_ALIAS, left, none);
            let got = buffer(acc, run, at, full);
            acc.refs.push(reference(0, kind::ROOT_VIEW, 3, at, full, got));
            assert_eq!((acc.draw_words, acc.sampler_counts, acc.sampler_heaps.len(), alias), ([2, 1, 4], [2, 1], 2, true));
        }
        write(&dir, frame, &[seq], take(frame, true)).unwrap();
        let p = FramePack::open(&dir).unwrap().unwrap();
        let got: Vec<(u16, u16, u32)> = p.refs_of(0).map(|r| (r.kind, r.slot, r.reason)).collect();
        assert_eq!(got, [(kind::ROOT_VIEW, 3, 0), (kind::SAMPLERS, 0, 0), (kind::DRAW_WORDS, 0, 0), (kind::DRAW_WORDS, 1, 0)]);
        let r = |owner: u32, k: u16, slot: u16| *p.refs_of(owner).find(|r| r.kind == k && r.slot == slot).unwrap();
        assert_eq!(FramePack::blob(&dir, r(0, kind::DRAW_WORDS, 0).hash).unwrap(), buf[0x100..0x140].to_vec());
        assert_eq!(r(0, kind::DRAW_WORDS, 0).addr, buf.as_ptr() as u64 + 0x100);
        let views = fp::decode_slots(&FramePack::blob(&dir, r(0, kind::DRAW_WORDS, 1).hash).unwrap());
        assert_eq!((r(0, kind::DRAW_WORDS, 1).addr, views.iter().map(|v| (v.slot, v.kind)).collect::<Vec<_>>()), (heap, vec![(9, view::PLACED), (12, view::CONSTANT)]));
        assert_eq!(p.textures.iter().map(|t| (t.at, t.reason)).collect::<Vec<_>>(), [(texels.as_ptr() as u64, 0)]);
        // The second sink shares the blobs; the third has no words and no heap.
        assert_eq!(r(1, kind::DRAW_WORDS, 1).hash, r(0, kind::DRAW_WORDS, 1).hash);
        assert_eq!(p.refs_of(2).map(|r| (r.kind, r.slot, r.reason)).collect::<Vec<_>>(), [(kind::SAMPLERS, 0, reason::NULL), (kind::DRAW_WORDS, 0, reason::IMPLAUSIBLE)]);
        let s = r(0, kind::SAMPLERS, 0);
        assert_eq!((s.addr, s.hash), (sheap.as_ptr() as u64, r(1, kind::SAMPLERS, 0).hash));
        let entries = fp::decode_samplers(&FramePack::blob(&dir, s.hash).unwrap()).unwrap();
        assert_eq!(entries, [fp::SamplerEntry { index: 1, form: fp::SAMPLER_CONSOLE, raw: [5, 2, 1, 0x4549_5676], desc: console }, fp::SamplerEntry { index: 2, form: fp::SAMPLER_RAW, raw: [0xA, 0xB, 0xC, 0xD], desc: [0; 27] }]);
        let rv = r(0, kind::ROOT_VIEW, 3);
        assert_eq!((rv.addr, FramePack::blob(&dir, rv.hash).unwrap()), (cpu, vb.clone()));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// A displacing pre-skin dispatch keeps the view at each element's
    /// heap index (+0x8C of 144-B elements of root 0 slot 8), its texture
    /// noted; an index outside the heap, an empty slot and a missing buffer
    /// each give a reason.
    #[test]
    fn displacing_dispatch_keeps_its_maps() {
        let _one = fresh();
        let frame = 9_700_000u64;
        let mut elems = vec![0u8; 3 * 144 + 10];
        for (e, i) in [9812u32, 9816, 5].iter().enumerate() {
            elems[e * 144 + 0x8C..e * 144 + 0x90].copy_from_slice(&i.to_le_bytes());
        }
        let texels: Vec<u8> = (0..4096u32).map(|i| (i % 241) as u8 + 1).collect();
        let resolve = |i: u32| match i {
            9812 => Some((0x5_0000 + 32 * 9812, SlotView { kind: view::TEXTURE, addr: texels.as_ptr() as u64, bytes: 4096, width: 64, height: 16, ..SlotView::default() })),
            9816 => Some((0x5_0000 + 32 * 9816, SlotView::default())),
            _ => None,
        };
        let buf = SlotView { slot: 8, kind: view::BUFFER, addr: elems.as_ptr() as u64, bytes: elems.len() as u64, ..SlotView::default() };
        let _ = hook_entry(frame);
        let mut accs = lock(&ACCS);
        let acc = acc_of(&mut accs, frame).unwrap();
        let mut run = lock(&RUN);
        let run = run.get_or_insert_with(Run::default);
        displacement_maps(acc, run, 1, Some(buf), &resolve);
        displacement_maps(acc, run, 2, None, &resolve);
        displacement_maps(acc, run, 3, Some(SlotView { bytes: 100, ..buf }), &resolve);
        let got: Vec<(u32, u16, u64, u32)> = acc.refs.iter().map(|r| (r.owner, r.slot, r.addr, r.reason)).collect();
        assert_eq!(
            got,
            [
                (1, 0, 0x5_0000 + 32 * 9812, 0),
                (1, 1, 0x5_0000 + 32 * 9816, reason::NOT_RESIDENT),
                (1, 2, 5, reason::IMPLAUSIBLE),
                (2, 0, 0, reason::NULL),
                (3, 0, elems.as_ptr() as u64, reason::IMPLAUSIBLE),
            ]
        );
        assert!(acc.refs.iter().all(|r| r.kind == kind::DISP_MAP));
        let v = fp::decode_slots(&acc.blobs[&acc.refs[0].hash].1);
        assert_eq!(v.iter().map(|v| (v.slot, v.addr)).collect::<Vec<_>>(), [(9812, texels.as_ptr() as u64)]);
        assert_eq!((acc.disp_maps, acc.textures.len()), ([1, 4], 1));
        assert!(DISPLACED.contains(&0xdcfc_c6a8_03ab_0b26_3a10_b1c5_61d7_ba77) && PED_SKINNING.iter().filter(|h| DISPLACED.contains(h)).count() == 2);
    }

    /// Strand hair: an input program's dispatch sink keeps
    /// the views at the heap indices of each strand group record's words
    /// with their bytes, once per record buffer; an order-only program keeps
    /// its roots and no table; a Draw keeps its instance count and firsts.
    #[test]
    fn hair_programs_keep_strand_groups_and_draw_args() {
        let _one = fresh();
        let frame = 9_800_000u64;
        let mut recs = vec![0xFFu8; 2 * fp::STRAND_RECORD];
        recs[0xe0..0xe4].copy_from_slice(&5u32.to_le_bytes());
        recs[0x120..0x124].copy_from_slice(&6u32.to_le_bytes());
        recs[fp::STRAND_RECORD + 0x120..fp::STRAND_RECORD + 0x124].copy_from_slice(&6u32.to_le_bytes());
        let asset: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        let cb = vec![7u8; 256];
        let resolve = |i: u32| match i {
            5 => Some((0x9_0000 + 32 * 5, SlotView { kind: view::BUFFER, addr: asset.as_ptr() as u64, bytes: asset.len() as u64, ..SlotView::default() })),
            6 => Some((0x9_0000 + 32 * 6, SlotView { kind: view::CONSTANT, addr: cb.as_ptr() as u64, bytes: 256, ..SlotView::default() })),
            _ => None,
        };
        let buf = SlotView { slot: STRAND_SLOT, kind: view::BUFFER, addr: recs.as_ptr() as u64, bytes: recs.len() as u64, ..SlotView::default() };
        let seq = hook_entry(frame);
        assert!(want_dispatch_at(frame, 0x30, |_| PIPE_HAIR));
        // Pipeline 0 (no stages to read): the order-only program here.
        assert!(want_dispatch_at(frame, 0, |_| PIPE_HAIR_ORDER));
        assert_eq!((class_of(0x30), class_of(0), pipe_class(HAIR_INPUTS[0]), pipe_class(HAIR_ORDER[0]), pipe_class(PED_SKINNING[0])), (PIPE_HAIR, PIPE_HAIR_ORDER, PIPE_HAIR, PIPE_HAIR_ORDER, PIPE_PED));
        let consts = [3u32, 0, 64, 0];
        let table = vec![0u8; 4 * 32];
        let roots = [TapRoot::Table { handle: table.as_ptr() as u64, heap_start: table.as_ptr() as u64, slots: 4 }, TapRoot::Constants(&consts)];
        sink_at(frame, &TapDraw { op: fp::op::DISPATCH, count: 2, yz: [1, 1], args: [0; 4], pipeline: 0, view_heap: 0, sampler_heap: 0, roots: &roots });
        sink_at(frame, &TapDraw { op: fp::op::DRAW, count: 6, yz: [0; 2], args: [9724, 0, 0, 12], pipeline: 0, view_heap: 0, sampler_heap: 0, roots: &[] });
        {
            let mut accs = lock(&ACCS);
            let acc = acc_of(&mut accs, frame).unwrap();
            let mut run = lock(&RUN);
            let run = run.get_or_insert_with(Run::default);
            for owner in [2u32, 3] {
                acc.sinks.push(SinkRaw { thread: 0, hook: 0, op: fp::op::DISPATCH, count: 1, pipeline: 0x30 });
                strand_groups(acc, run, owner, StrandInputs { records: Some(buf), palette: None, shapes: None }, &resolve);
            }
            strand_groups(acc, run, 3, StrandInputs { records: None, palette: None, shapes: None }, &resolve);
            // 2 records x 7 words: record 0 words 0 (buffer) and 1 (constants) resolve, record 1 word 1.
            assert_eq!((acc.draw_args, acc.hair), (1, [0, 1, 2, 3, 11]));
            assert!(!acc.refs.iter().any(|r| r.kind == kind::DISPATCH), "a hair sink is not a pre-skin record");
        }
        let dir = std::env::temp_dir().join(format!("xnw-pk7h-{}", std::process::id())).join(format!("frame-{frame}"));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        write(&dir, frame, &[seq], take(frame, true)).unwrap();
        let p = FramePack::open(&dir).unwrap().unwrap();
        // The order-only sink: its constants, no table.
        assert_eq!(p.refs_of(0).map(|r| (r.kind, r.slot)).collect::<Vec<_>>(), [(kind::ROOTS, 1), (kind::SAMPLERS, 0), (kind::GROUPS, 0)]);
        let a = p.refs_of(1).find(|r| r.kind == kind::DRAW_ARGS).unwrap();
        let words: Vec<u32> = FramePack::blob(&dir, a.hash).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(words, [9724, 0, 0, 12]);
        let sv: Vec<(u16, u16, u32)> = p.refs_of(2).filter(|r| r.reason == 0).map(|r| (r.kind, r.slot, r.len)).collect();
        assert_eq!(sv, [(kind::STRAND_VIEW, 0, fp::SLOT_BYTES as u32), (kind::STRAND_VIEW, 1, fp::SLOT_BYTES as u32), (kind::STRAND_VIEW, 9, fp::SLOT_BYTES as u32), (kind::STRAND_BYTES, 0, 1000), (kind::STRAND_BYTES, 1, 256), (kind::STRAND_BYTES, 9, 256)]);
        let b = p.refs_of(2).find(|r| r.kind == kind::STRAND_BYTES && r.slot == 0).unwrap();
        assert_eq!(FramePack::blob(&dir, b.hash).unwrap(), asset);
        // The second sink naming the same records adds none; no records buffer: one NULL record.
        assert_eq!(p.refs_of(3).map(|r| (r.kind, r.reason)).collect::<Vec<_>>(), [(kind::STRAND_VIEW, reason::NULL)]);
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// Strand hair (hair3 #892 c, e): per group record the palette bones
    /// at both bases (as many as the remap reaches) and the collision
    /// shapes it names, cut at the views' ends; count 0 names none.
    #[test]
    fn strand_records_keep_palette_bones_and_collision_spans() {
        let _one = fresh();
        let frame = 9_850_000u64;
        let mut recs = vec![0u8; 2 * fp::STRAND_RECORD];
        let put = |r: &mut Vec<u8>, at: usize, v: u32| r[at..at + 4].copy_from_slice(&v.to_le_bytes());
        // Record 0: remap at heap 5 from element 2, bases 1 and 3, 2 shapes at byte 0x50.
        put(&mut recs, 0xe0, 5);
        put(&mut recs, REC_REMAP_AT, 2);
        put(&mut recs, 0xe8, 1);
        put(&mut recs, 0xec, 3);
        put(&mut recs, REC_SHAPE_COUNT, 2 << 24 | 0x1234);
        put(&mut recs, REC_SHAPE_AT, 0x50);
        // Record 1: no remap (256 bones from base 0: cut), 9 shapes from 0xa0 (cut), base 99 past the palette.
        let r1 = fp::STRAND_RECORD;
        put(&mut recs, r1 + 0xe0, 77);
        put(&mut recs, r1 + 0xec, 99);
        put(&mut recs, r1 + REC_SHAPE_COUNT, 9 << 24);
        put(&mut recs, r1 + REC_SHAPE_AT, 0xa0);
        let remap: Vec<u8> = [9u16, 9, 0, 1, 0xf002, 0].iter().flat_map(|w| w.to_le_bytes()).collect();
        let palette: Vec<u8> = (0..8 * fp::BONE_BYTES).map(|i| (i / fp::BONE_BYTES) as u8).collect();
        let shapes: Vec<u8> = (0..4 * fp::STRAND_SHAPE).map(|i| (i / fp::STRAND_SHAPE) as u8 + 1).collect();
        let resolve = |i: u32| (i == 5).then(|| (0x9_0000, SlotView { kind: view::PLACED_BUFFER, addr: remap.as_ptr() as u64, bytes: remap.len() as u64, ..SlotView::default() }));
        let view_of = |b: &Vec<u8>| Some(SlotView { kind: view::PLACED_BUFFER, addr: b.as_ptr() as u64, bytes: b.len() as u64, ..SlotView::default() });
        let mut accs = lock(&ACCS);
        let acc = acc_of(&mut accs, frame).unwrap();
        let mut run = lock(&RUN);
        let run = run.get_or_insert_with(Run::default);
        let inputs = StrandInputs { records: view_of(&recs), palette: view_of(&palette), shapes: view_of(&shapes) };
        strand_groups(acc, run, 4, inputs, &resolve);
        let got: Vec<(u16, u16, u32, u32)> = acc.refs.iter().filter(|r| matches!(r.kind, kind::STRAND_PALETTE | kind::STRAND_SPAN)).map(|r| (r.kind, r.slot, r.len, r.reason)).collect();
        let b = fp::BONE_BYTES as u32;
        // Remap entries 2.. read 0, 1, 0xf002 & 0xfff = 2, 0: 3 bones.
        assert_eq!(
            got,
            [
                (kind::STRAND_PALETTE, 0, 3 * b, reason::OK),
                (kind::STRAND_PALETTE, 1, 3 * b, reason::OK),
                (kind::STRAND_SPAN, 0, 2 * fp::STRAND_SHAPE as u32, reason::OK),
                (kind::STRAND_PALETTE, 2, 8 * b, reason::CUT),
                (kind::STRAND_PALETTE, 3, 256 * b, reason::IMPLAUSIBLE),
                (kind::STRAND_SPAN, 1, 2 * fp::STRAND_SHAPE as u32, reason::CUT),
            ]
        );
        let bytes = |slot: u16, k: u16| acc.blobs[&acc.refs.iter().find(|r| r.kind == k && r.slot == slot).unwrap().hash].1.clone();
        assert_eq!(bytes(1, kind::STRAND_PALETTE), palette[3 * fp::BONE_BYTES..6 * fp::BONE_BYTES]);
        assert_eq!(bytes(0, kind::STRAND_SPAN), shapes[fp::STRAND_SHAPE..3 * fp::STRAND_SHAPE]);
        assert_eq!(acc.strand_extra, [3, 1, 2, 0]);
    }

    /// Head graph (head1 #926): a head dispatch reads its plain unordered
    /// buffer views (the CPU's rig weights and controls), once per address
    /// in a frame, cut at HEAD_UAV_CAP; another dispatch does not.
    #[test]
    fn head_graph_reads_its_unordered_buffers() {
        let _one = fresh();
        let frame = 9_880_000u64;
        let weights: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
        let mut accs = lock(&ACCS);
        let acc = acc_of(&mut accs, frame).unwrap();
        let mut run = lock(&RUN);
        let run = run.get_or_insert_with(Run::default);
        let v = SlotView { slot: 3, kind: view::BUFFER, layout: view::UNORDERED, addr: weights.as_ptr() as u64, bytes: weights.len() as u64, ..SlotView::default() };
        view_bytes(acc, run, 1, 0, &v, 0, TB_HEAD);
        view_bytes(acc, run, 2, 0, &v, 0, TB_HEAD);
        view_bytes(acc, run, 3, 0, &v, 0, TB_DISPATCH);
        let got: Vec<(u32, u32)> = acc.refs.iter().map(|r| (r.owner, r.reason)).collect();
        assert_eq!(got, [(1, reason::OK), (2, reason::OK), (3, reason::GPU_OUTPUT)]);
        assert_eq!(acc.blobs[&acc.refs[0].hash].1, weights);
        assert_eq!((acc.head_uavs, acc.buffer_bytes), ([2, 0], 300));
    }

    /// An indirect dispatch (lit4 #914, ambient sinks 54/58) is no draw:
    /// unpaired, its tables resolved as a lighting dispatch's (frame table
    /// past one row, other tables one row, no unordered bytes); a direct
    /// dispatch keeps its group counts; sky programs are lighting sinks.
    #[test]
    fn indirect_dispatch_keeps_its_tables_and_direct_its_groups() {
        let _one = fresh();
        let frame = 9_870_000u64;
        let seq = hook_entry(frame);
        let table = vec![0u8; 4 * 32];
        let roots = [TapRoot::Table { handle: table.as_ptr() as u64, heap_start: table.as_ptr() as u64, slots: 4 }, TapRoot::Table { handle: table.as_ptr() as u64, heap_start: table.as_ptr() as u64, slots: 4 }];
        // Pipeline 0 (no stages to read) for both.
        sink_at(frame, &TapDraw { op: fp::op::INDIRECT_DISPATCH, count: 1, yz: [0; 2], args: [0; 4], pipeline: 0, view_heap: 0, sampler_heap: 0, roots: &roots });
        assert!(want_dispatch_at(frame, 0, |_| PIPE_SKY));
        sink_at(frame, &TapDraw { op: fp::op::DISPATCH, count: 8, yz: [4, 2], args: [0; 4], pipeline: 0, view_heap: 0, sampler_heap: 0, roots: &[] });
        let dir = std::env::temp_dir().join(format!("xnw-pk8i-{}", std::process::id())).join(format!("frame-{frame}"));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        write(&dir, frame, &[seq], take(frame, true)).unwrap();
        let p = FramePack::open(&dir).unwrap().unwrap();
        assert_eq!(p.sinks.iter().map(|s| (s.op, s.hook_draw)).collect::<Vec<_>>(), [(fp::op::INDIRECT_DISPATCH, None), (fp::op::DISPATCH, None)]);
        assert_eq!((p.manifest.sink_draws, p.manifest.unpaired_sinks), (0, 0));
        assert_eq!(p.refs_of(0).filter(|r| r.kind == kind::TABLE).map(|r| r.slot).collect::<Vec<_>>(), [0, 1]);
        let g = p.refs_of(1).find(|r| r.kind == kind::GROUPS).unwrap();
        assert_eq!(FramePack::blob(&dir, g.hash).unwrap(), [8u32, 4, 2].iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>());
        assert!(p.refs_of(0).all(|r| r.kind != kind::GROUPS));
        assert_eq!((pipe_class(SKY_PROGRAMS[0]), pipe_class(SKY_PROGRAMS[15]), pipe_class(LIGHT_PROGRAMS[0])), (PIPE_SKY, PIPE_SKY, PIPE_LIGHT));
        assert!(SKY_PROGRAMS.iter().all(|h| !LIGHT_PROGRAMS.contains(h) && !HEAD_PROGRAMS.contains(h)));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// Lighting (lit1): a lighting dispatch sink copies the CPU scene-light
    /// array's current buffer once per frame, with its index and count, and
    /// is no pre-skin record; a pipeline's targets are one frame record.
    #[test]
    fn lighting_dispatch_copies_the_scene_lights() {
        let _one = fresh();
        let frame = 9_900_000u64;
        let lights: Vec<u8> = (0..2 * fp::LIGHT_ARRAY_BYTES).map(|i| (i / fp::LIGHT_ARRAY_BYTES) as u8 + 1).collect();
        let (index, count) = (1u32.to_le_bytes(), 44u32.to_le_bytes());
        set_lights(lights.as_ptr() as u64, index.as_ptr() as u64, count.as_ptr() as u64);
        let seq = hook_entry(frame);
        assert!(want_dispatch_at(frame, 0, |_| PIPE_LIGHT));
        assert_eq!((pipe_class(LIGHT_PROGRAMS[0]), pipe_class(HEAD_PROGRAMS[0])), (PIPE_LIGHT, PIPE_HEAD));
        let consts = [1u32];
        for _ in 0..2 {
            sink_at(frame, &TapDraw { op: fp::op::DISPATCH, count: 8, yz: [1, 1], args: [0; 4], pipeline: 0, view_heap: 0, sampler_heap: 0, roots: &[TapRoot::Constants(&consts)] });
        }
        {
            let mut accs = lock(&ACCS);
            let acc = acc_of(&mut accs, frame).unwrap();
            assert_eq!((acc.light_sinks, acc.lights_copied), (2, 44));
            assert!(!acc.refs.iter().any(|r| r.kind == kind::DISPATCH));
        }
        let dir = std::env::temp_dir().join(format!("xnw-pk7l-{}", std::process::id())).join(format!("frame-{frame}"));
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        write(&dir, frame, &[seq], take(frame, true)).unwrap();
        let p = FramePack::open(&dir).unwrap().unwrap();
        let l: Vec<&Ref> = p.refs.iter().filter(|r| r.kind == kind::LIGHTS).collect();
        assert_eq!(l.iter().map(|r| (r.owner, r.slot, r.reason)).collect::<Vec<_>>(), [(fp::FRAME_SCOPE, 0, 0), (fp::FRAME_SCOPE, 1, 0)]);
        assert_eq!(l[0].addr, lights.as_ptr() as u64 + fp::LIGHT_ARRAY_BYTES as u64);
        assert_eq!(FramePack::blob(&dir, l[0].hash).unwrap(), lights[fp::LIGHT_ARRAY_BYTES..].to_vec());
        assert_eq!(FramePack::blob(&dir, l[1].hash).unwrap(), [1, 0, 0, 0, 44, 0, 0, 0]);
        set_lights(0, 0, 0);
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    /// `hookin=1` on made-up memory: the TLS words, geometry, context,
    /// world, viewport, model and rigid-bone parts land where the format
    /// says; unreadable pointers give zeroed parts; the cost per hook draw is
    /// printed (`XNW_TEST_TIMING=1` makes it a limit of 0.5 us).
    #[test]
    fn hook_inputs_follow_the_hook() {
        let _one = fresh();
        let frame = 9_950_000u64;
        let mut tls = vec![0u8; 0x1400];
        let mut ctx = vec![0u8; 0x1_2400];
        let world: Vec<u8> = (0..0x80u32).map(|i| i as u8).collect();
        let cache: Vec<u8> = (0..256u32).map(|i| (i ^ 0x5A) as u8).collect();
        let vp = vec![0x11u8; 0x710];
        let mut model = vec![0u8; 0x38];
        model[0x33] = 2;
        let pal: Vec<u8> = (0..48 * 4u32).map(|i| (i / 48) as u8 + 0x40).collect();
        let vb = vec![0x22u8; 0x20];
        let mut geom = vec![0u8; 0x48];
        geom[8..16].copy_from_slice(&(vb.as_ptr() as u64).to_le_bytes());
        let slot_word = 3u16 << 2;
        let put = |b: &mut Vec<u8>, o: usize, v: u64| b[o..o + 8].copy_from_slice(&v.to_le_bytes());
        tls[0x950..0x954].copy_from_slice(&69u32.to_le_bytes());
        put(&mut tls, 0x12A8, ctx.as_ptr() as u64);
        put(&mut tls, 0x12B8, vp.as_ptr() as u64);
        put(&mut tls, 0x1208, model.as_ptr() as u64);
        put(&mut tls, 0x11E8, pal.as_ptr() as u64);
        put(&mut ctx, 0x40 + 8 * 3, world.as_ptr() as u64);
        ctx[0x1_2190..0x1_2194].copy_from_slice(&1i32.to_le_bytes());
        let pal_base = 0x1234_5678_u64;
        set_hookin(true, &slot_word as *const u16 as u64, cache.as_ptr() as u64, &pal_base as *const u64 as u64);
        let seq = hook_entry(frame);
        hook_inputs(frame, seq, tls.as_ptr() as u64, geom.as_ptr() as u64);
        let b = hookin_blob(tls.as_ptr() as u64, geom.as_ptr() as u64, HOOK_GLOBALS.each_ref().map(|a| a.load(Ordering::Relaxed)));
        let parts = fp::decode_hookin(&b).unwrap();
        assert_eq!(parts.len(), fp::HOOKIN_PARTS);
        let p = |k: usize| parts[k].clone();
        assert_eq!((p(0).3, p(0).2), (69u32.to_le_bytes().to_vec(), true));
        assert_eq!((p(5).0, p(5).3), (vb.as_ptr() as u64, vb.clone()));
        assert!(!p(6).2 && p(6).3 == vec![0; 0x20], "a null ib pointer reads nothing");
        assert_eq!((p(8).0, p(13).3), (ctx.as_ptr() as u64 + 0x40 + 24, world.clone()));
        assert_eq!(p(14).3, cache[128..].to_vec());
        assert_eq!((p(15).3.len(), p(16).3), (0xC0, vec![0x11; 8]));
        assert_eq!(p(18).3, vec![0x42; 48]);
        {
            let mut accs = lock(&ACCS);
            let acc = acc_of(&mut accs, frame).unwrap();
            assert_eq!(acc.hookin, 1);
            let g = acc.refs.iter().find(|r| r.kind == kind::HOOK_GLOBALS).unwrap();
            let gb = acc.blobs[&g.hash].1.clone();
            assert_eq!((u16::from_le_bytes([gb[0], gb[1]]), u64::from_le_bytes(gb[2..10].try_into().unwrap())), (slot_word, pal_base));
        }
        // The best of 20 batches of 1,000 (a loaded machine stretches some).
        let best = |f: &dyn Fn()| {
            (0..20)
                .map(|_| {
                    let t0 = std::time::Instant::now();
                    for _ in 0..1000 {
                        f();
                    }
                    t0.elapsed().as_secs_f64() * 1e3
                })
                .fold(f64::MAX, f64::min)
        };
        let g = HOOK_GLOBALS.each_ref().map(|a| a.load(Ordering::Relaxed));
        let blob_us = best(&|| drop(std::hint::black_box(hookin_blob(tls.as_ptr() as u64, geom.as_ptr() as u64, g))));
        let us = best(&|| hook_inputs(frame, seq, tls.as_ptr() as u64, geom.as_ptr() as u64));
        println!("hook inputs: {us:.3} us a hook draw, of which the reads {blob_us:.3} us ({} B)", b.len());
        assert!(us < 0.5 || !std::env::var("XNW_TEST_TIMING").is_ok_and(|v| v == "1"), "{us} us");
        set_hookin(false, 0, 0, 0);
    }

    /// A blob first met in a frame that is never written is written by
    /// the next frame that is (f1002 had 3,712 refs whose blob was missing).
    #[test]
    fn blobs_of_a_dropped_frame_are_written_by_the_next_kept_one() {
        let _one = fresh();
        let base = 9_100_000u64;
        let root = std::env::temp_dir().join(format!("xnw-pk2-drop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let consts = [0xC0FFEEu32, 7];
        let roots = [TapRoot::Constants(&consts)];
        let seq0 = hook_entry(base);
        sink_at(base, &TapDraw { op: fp::op::MESH, count: 1, pipeline: 0, yz: [0; 2], args: [0; 4], view_heap: 0, sampler_heap: 0, roots: &roots });
        assert!(take(base, false).is_none());
        let seq1 = hook_entry(base + 1);
        sink_at(base + 1, &TapDraw { op: fp::op::MESH, count: 1, pipeline: 0, yz: [0; 2], args: [0; 4], view_heap: 0, sampler_heap: 0, roots: &roots });
        let dir = root.join(format!("frame-{}", base + 1));
        write(&dir, base + 1, &[seq1], take(base + 1, true)).unwrap();
        let p = FramePack::open(&dir).unwrap().unwrap();
        let r = p.refs_of(0).next().unwrap();
        assert_eq!(FramePack::blob(&dir, r.hash).unwrap().len(), 8);
        assert_ne!(seq0, seq1);
        // A frame the probe does not want records nothing.
        set_want(|f| f != 9_100_002);
        sink_at(base + 2, &TapDraw { op: fp::op::MESH, count: 1, pipeline: 0, yz: [0; 2], args: [0; 4], view_heap: 0, sampler_heap: 0, roots: &roots });
        assert!(take(base + 2, true).is_none());
        let _ = std::fs::remove_dir_all(root);
    }
}
