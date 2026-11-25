# XNW

XNW is a research project that runs a console game's own x64 code natively on Windows.

It supplies the parts of the console platform the game expects (a loader, system services and the graphics API)
on top of Direct3D 12, and it draws the game's world with a renderer of its own.

You bring the game. This repository contains no game files, keys or console software, and it never will.

## Status

Everything here was measured on the test machine listed below, with one retail game as the test case.

**Where it stands: the game gets through its intro cinematics and hands you control of the player. You can walk
around and interact with the world. The picture is still incomplete, so it is not yet playable in any real sense.**

### How far the game gets

| Stage | State | Notes |
|---|---|---|
| Loads and starts | Working | The unmodified x64 code runs inside our process. |
| First screens | Working | The account prompt was the first frame we ever drew. |
| Front-end menus | Partly | The game gets through them on key presses. Our renderer does not draw the user interface yet. |
| Story load | Working | About 20-35 seconds after launch. |
| Intro cinematics | Working | They play through to the end, about six minutes. |
| Player control | Working | After the cinematics you are given control and can walk around and interact with the world. |
| Playable | Not yet | Characters and vehicles are drawn; hair, the interface, lighting and the game's own materials are still missing from the picture. |
| Audio | Working | The game's audio plays. |
| Asset streaming | Working | The game's streaming requests are served as it moves through the world. |
| Online | Off | Network access is disabled by design. |

### Milestones

| When | What |
|---|---|
| 2026-11-23 | The game's unmodified x64 code loads, binds its imports and runs inside our process. It creates its device and presents continuously in its own window. |
| 2026-11-24 | First drawn frame. Boot, menus and the story load work with the console's GPU work switched off. |
| 2026-11-24 | CPU instructions that the test CPU lacks are rewritten when the game loads. This removed a slow stretch of 115-150 ms per frame in the world; the same stretch now runs at 20-43 ms. |
| 2026-11-24 | World geometry read straight from the game's draw lists. |
| 2026-11-25 | Our own renderer draws the static world (streets, buildings, vegetation, props) on a real GPU. |
| 2026-11-25 | The game's own textures on that world: roughly nine out of ten texture slots filled. |
| 2026-11-25 | A frame lab: a captured frame replays outside the game, so rendering work no longer needs the game running. |
| 2026-11-25 | Skinned props drawn on real hardware from the bones the game used at draw time. |
| 2026-11-25 | First whole character in the frame lab, from our own version of the game's GPU skinning step (the face is still missing). |
| 2026-11-25 | Characters in the running game on a real GPU: posed, clothed, with their faces, from our own GPU skinning step. Hair is still missing. |
| 2026-11-25 | Vehicles in the running game: bodies placed by the game's own rule, wheels drawn per instance. Distant trees drawn as flat cards. |
| 2026-11-25 | Camera cuts: 22 of 23 cuts finish within two frames (the worst used to hold the picture for 392 ms). |
| 2026-11-25 | The game's own translated pixel shaders reproduce its facades, signs and storefronts on a captured frame, inside our renderer. |
| 2026-11-25 | Faster frames: the stalls on entering the world are gone, and slow frames fell from 94 to 22 per 150 s run (3,739 frames). |
| 2026-11-25 | Reading the game's geometry can run on a worker thread, off the frame (a switch, off by default). An optimised build profile (`cargo build --profile fast`) is what timed runs use. |
| 2026-11-25 | Best run yet on a real GPU: 3,990 frames in 150 s, 8 slow frames (94 at the start of the day), no device errors. The hook that reads the game's draw lists costs 3.2-3.4 us a draw. |
| 2026-11-25 | Instance culling on the GPU (a switch, `gpucull=on`): our renderer drops hidden instances in a compute pass (0.6 ms of GPU time) before drawing. |
| 2026-11-25 | The shader translator reaches 25,530 of 25,889 programs, all valid; it follows the rule for non-uniform descriptor-heap indices. |
| 2026-11-25 | `cargo x burst` replays a burst of the game's draw-list steps outside the game, for timing and checks. |
| 2026-11-25 | Zero-copy geometry: buffers are used where they sit instead of being copied (a switch, off by default; run on the software adapter only). |

### Performance so far

| What | Measured |
|---|---|
| World on screen, our renderer, real GPU | about **26 frames per second** with characters and vehicles (3,990 frames in a 150 s run, 8 of them slow, no device errors); the game paces itself at 30 |
| Renderer cost inside each frame | 6-15 ms for our pass, plus 9-17 ms to read the game's draw lists; the draw-list hook costs 3.2-3.4 us a draw; GPU instance culling (a switch, `gpucull=on`, run on the card) 0.6 ms |
| Texture slots filled with the game's textures | 23,562 of 25,653 at the end of a run (92 %) |
| Captured frame replayed in the frame lab | **4-7 ms per frame**; a shader edit shows on screen in about half a second |
| Console GPU programs translated to DXIL | **25,530 of 25,889** (98.6 %) in the test game, all of them valid DXIL; 786 of the 788 that a typical frame uses. Programs that read barycentrics need hardware support for it and are run on the software adapter |
| Translating the whole program set | about 2.5 minutes on a 6-core machine with 6 threads, validation included; translated programs are cached per translator generation, and a change is checked against the previous output in seconds |
| The game's own GPU work on real hardware | 1.3-4 s per frame, and not stable yet |

### In progress

- Drawing with the game's own translated shaders, family by family, inside our renderer: its pixel shaders for
  materials, and its mesh shaders through a compute-plus-indexed-draw form (the test GPU has no mesh shaders).
- Hair: the game builds strand hair on the GPU with a chain of compute shaders; running that chain is next.
- Occasional one-frame flicker of some objects.

### Pending

- Lighting, sky, water, decals and the game's user interface. The picture is still dark and unlit.
- Running the game's own GPU work on real hardware at a usable speed.
- The frame lab (a window that replays captured frames with a free camera) is not in this tree yet.

## Requirements

|  | Test machine | Needed |
|---|---|---|
| OS | Windows 10/11 x64 | Windows 10/11 x64 |
| CPU | Intel Core i5-10600K (6 cores, 12 threads) | x86-64 |
| Memory | 64 GB | 64 GB tested; less is untested |
| GPU | AMD Radeon RX 5700 XT, 8 GB | Direct3D 12, feature level 12_0, shader model 6.6; mesh shaders are not required |
| Runtime | D3D12 Agility SDK 1.610.5 | D3D12 Agility SDK (Windows 10's built-in D3D12 stops at shader model 6.5) |

### What you need

- **The D3D12 Agility SDK.** Take `D3D12Core.dll` from Microsoft's `Microsoft.Direct3D.D3D12` package on NuGet
  (`build\native\bin\x64`) and put it in a folder named after the release, for example `deps\agility\1.610.5`
  (the version is read from the folder's name). Point `XNW_AGILITY` at that folder.
- **Your own copy of the game.** The player takes the path of the game's executable as its argument and reads the
  rest of the package from that executable's folder, then from the folders listed in `XNW_GAME_DRIVE`
  (`;`-separated). The loader rewrites the few instructions some CPUs lack only in the executable it was built
  for (it checks the file's SHA-256); any other file runs unchanged.
- **The texture format tables.** They are not part of this repository. Get `bcpack-tables.bin` from VIctory
  ([crates/victory-formats/src/bcpack-tables.bin](https://github.com/Marvisak/VIctory/blob/main/crates/victory-formats/src/bcpack-tables.bin))
  and put it in `deps\bcpack\` under the working folder or the source tree, or point `XNW_BCPACK_TABLES` at the
  file.

## Building

The project is plain Rust, and the toolchain is pinned in `rust-toolchain.toml`.

```
cargo build --locked
cargo test --locked
```

Tests that need the game's files, the format tables or a GPU are marked `#[ignore]` or skip themselves.

## Running

The front end is `xnw`; `cargo x <command>` is short for `cargo run --quiet -p xnw -- <command>`. Name the two
folders first:

```
set XNW_AGILITY=C:\path\to\agility\1.610.5
set XNW_GAME_DRIVE=C:\path\to\the\game\package
```

| Command | What it does |
|---|---|
| `cargo x gpu-check` | The host device: adapter, shader model, wave size. |
| `cargo x inspect <exe> --imports` | The executable's headers, sections and where each import goes. |
| `cargo x smoke --exe <exe>` | The check before any run: the game for 20 s with its GPU work switched off, on the software adapter, without a window; prints PASS or FAIL. Its expectations are in `tools/runs/`. |
| `cargo x run <exe> --execute` | The game in a window. |
| `cargo x world-render <frame folder> --adapter warp --out frame.png` | One captured frame drawn by our renderer, outside the game. |

A run writes its log and caches under `out\` in the working folder. `XNW_WORLD_PROBE=<folder>,raw=1,at=<frame>`
captures frames during a run (one `frame-NNNN` folder each) for `world-render`.

The game stays offline: its connections to anything but this machine fail.

## Conventions

Code style, naming, error handling, logging, `unsafe` rules and commit messages are described in
[CONVENTIONS.md](CONVENTIONS.md).

## Legal

XNW is free software under the GNU General Public License, version 3 or, at your option, any later version.
See [LICENSE](LICENSE).

This repository provides no game files, no game code and no copyrighted game or console material. You bring your
own legally obtained copy of the game; please keep contributions free of such material too.

All product names and trademarks belong to their respective owners. This project is not affiliated with, or
endorsed by, any console maker or game publisher.

XNW builds on related open-source projects: VIctory (GPL-3.0-or-later), from which parts of the code are ported,
and XCL, along with Xbox2Windows, five-star-recomp and AMD's address library as shipped in Mesa. Ported files name
their origin in their headers; the notices their licences ask for are in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
