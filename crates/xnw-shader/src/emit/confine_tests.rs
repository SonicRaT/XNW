//! XNW (BND1): a confined program reads its own views or a null, end to
//! end on the software adapter. The program of `heap_uniform_tests` (a
//! slot number from a buffer, that slot's descriptor from the view heap,
//! the texture's width through a 2D instruction) lowered with
//! [`super::Options::confine`], in a heap laid out as the bridge's: another
//! draw's texture below the draw's own block, a 2D view and a cube in the
//! block, another draw's texture past it. A card run's fault is the
//! unconfined half: an index that is the title's own heap number reads
//! whatever lies there. `XNW_TEST_GPU=1 XNW_ADAPTER=warp XNW_WARP=<dir>
//! XNW_AGILITY=<dir> cargo test -p xnw-shader confine -- --nocapture`.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;

use super::HostShader;
use super::heap_uniform_tests::build_with;
use super::host::{Dispatch, Host, Root, Stride, View};
use crate::native::RootType;

/// The width lanes 1.. read at heap slot `slot`; `None` without a device.
fn width_at(shader: &HostShader, slot: u32) -> Option<u32> {
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
    let blank = [[0.0f32; 4]; 32 * 4];
    let null = |desc: D3D12_SHADER_RESOURCE_VIEW_DESC| {
        move |dev: &ID3D12Device, at: D3D12_CPU_DESCRIPTOR_HANDLE| {
            // SAFETY: a null view of a description that outlives the call.
            unsafe { dev.CreateShaderResourceView(None, Some(&desc), at) };
        }
    };
    let desc = |dim, a| D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, ViewDimension: dim, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: a };
    let null_2d = null(desc(D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() } }));
    let null_cube = null(desc(D3D12_SRV_DIMENSION_TEXTURECUBE, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { TextureCube: D3D12_TEXCUBE_SRV { MipLevels: 1, ..Default::default() } }));
    let list = [slot];
    // 0: the null 2D image. 1, 2: the program's table (the slot list, the
    // output). 3: another draw's 16x8 texture. 4, 5: the draw's own block,
    // an 8x8 texture and a cube. 6: another draw's 32x4 texture. 7: the
    // null images that are not 2D start here (the cube's stands for all).
    let views = [
        View::Slot(&null_2d),
        View::Buffer { data: &list, stride: Stride::Raw },
        View::RawUav(64),
        View::Texture(16, 8, &blank[..128]),
        View::Texture(8, 8, &blank[..64]),
        View::Cube(4, &blank[..96]),
        View::Texture(32, 4, &blank),
        View::Slot(&null_cube),
    ];
    // The heap bounds: 8 view slots, the null 2D image at 0 (the buffer
    // nulls are not read here). The confinement: the block starts at slot
    // 4; its slot 0 holds a 2D view, its slot 1 a cube; null images from 7.
    let mut roots = vec![(0, Root::Constants(ud)), (1, Root::Constants(vec![8, 0, 0, 0, 0, 0, 1, 0]))];
    if shader.reads_confine() {
        roots.push((2, Root::Constants(vec![4, 0b01, 0, 0b10, 0, 0, 0, 7])));
    }
    let out = host.dispatch(&root, &pso, &Dispatch { roots, views: &views, samplers: &[], groups: [1, 1, 1] }).expect("dispatch");
    assert_eq!(out[0][0], 0, "lane 0 does not run");
    assert!(out[0][2..].iter().all(|w| *w == out[0][1]), "every lane reads one slot: {:?}", out[0]);
    Some(out[0][1])
}

#[test]
fn confine_a_program_reads_its_own_views_or_the_null() {
    // Pure: the option adds the confinement words (`b1` beside the bounds), and only it does.
    let (confined, open) = (build_with(true, true), build_with(true, false));
    assert!(confined.reads_confine() && confined.reads_heap_bounds());
    assert!(!open.reads_confine() && open.reads_heap_bounds());
    // The software adapter's rule: never run on the card.
    if !xnw_device::warp() {
        eprintln!("XNW_ADAPTER is not warp; skipped");
        return;
    }
    let Some(own) = width_at(&confined, 4) else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    assert_eq!(own, 8, "its own 2D view, by the index that names it");
    // Below the block, past it (inside the heap, another draw's texture
    // each), far past the heap (the title's own numbers of a test run), the
    // guard itself: the null 2D image (width 0).
    for index in [3, 6, 7, 0, 36, 9812, 11683, 0x000F_FFFF, u32::MAX] {
        assert_eq!(width_at(&confined, index), Some(0), "index {index} is not the draw's own");
    }
    // In the block, but a cube where the instruction reads 2D: the null 2D image.
    assert_eq!(width_at(&confined, 5), Some(0), "a slot of another dimension");
    // Without the option the same index reads the other draw's texture (bounded above only).
    let before = (width_at(&open, 3), width_at(&open, 6));
    println!("unconfined: index 3 reads width {:?}, index 6 {:?} (another draw's textures)", before.0, before.1);
    assert_eq!(before, (Some(16), Some(32)));
}
