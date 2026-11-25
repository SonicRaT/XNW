//! Named title patches: a table of `{name, rva, expected
//! bytes, new bytes}` for one build of the title. An entry applies only
//! when the image's PE timestamp is that build's, the expected bytes are
//! there, and `XNW_PATCH` (a comma list of names) asks for it; each one
//! applied is logged. Applied after mapping, before any title code runs.
//!
//! `watchdog` is after VIctory (d2c93fa, GPL-3.0-or-later,
//! `crates/victory-loader/src/hooks.rs`): the title's watchdog thread (fn
//! `title+0x2215a60`) ends the process on purpose when the main thread has
//! not advanced for about 30 one-second ticks: in the handler it calls
//! (`mov dword [0], imm32` at `title+0x23cf683`) and, after the call, a
//! fallback (`mov dword [0], 0; int3` at `title+0x2215c35`). The jump goes
//! from the handler load at `title+0x2215c27` to the loop's continuation
//! at `title+0x2215c41`; bytes checked against our exe.

use xnw_core::win::{FlushInstructionCache, GetCurrentProcess, PAGE_EXECUTE_READWRITE, VirtualProtect};

/// The title's executable 1.0.2.1 (PE header timestamp).
const BUILD_1_0_2_1: u32 = 0x6AF7_83D4;

struct Patch {
    name: &'static str,
    what: &'static str,
    rva: usize,
    expect: &'static [u8],
    new: &'static [u8],
}

const PATCHES: &[Patch] = &[
    Patch {
        name: "watchdog",
        what: "skip the handler call and the fallback fault (jmp title+0x2215c41)",
        rva: 0x2215c27,
        // mov rax, [rip + 0x4fa071a] (the handler pointer)
        expect: &[0x48, 0x8B, 0x05, 0x1A, 0x07, 0xFA, 0x04],
        // jmp +0x18, then NOPs
        new: &[0xEB, 0x18, 0x90, 0x90, 0x90, 0x90, 0x90],
    },
    Patch {
        name: "watchdog",
        what: "the handler's deliberate write to 0",
        rva: 0x23cf683,
        // mov dword ptr [0], 0x4d8bd421
        expect: &[0xC7, 0x04, 0x25, 0, 0, 0, 0, 0x21, 0xD4, 0x8B, 0x4D],
        // an eleven-byte NOP
        new: &[0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0, 0, 0, 0, 0],
    },
    // The upscaler selector (fn `title+0x7e93f0`) returns kind 7, the
    // console's FSR 4 DLL, when the console type is Series S and bit 0x40
    // of a setting is set, else kind 6, the title's own upscaler
    // (after five-star-recomp, GPL-3.0; bytes read from
    // our exe). The FSR 4 DLL is console GPU code behind
    // `DeserializeComputePipelineStateX` and cannot run on the host (pipeline
    // slot 11 and list slot 53 not implemented in our runs), so its output
    // target stays flat. `mov ecx, 7` becomes `mov ecx, 6`: always kind 6.
    Patch {
        name: "upscaler",
        what: "the upscaler selector answers kind 6 (the title's own), never the console FSR 4 DLL",
        rva: 0x7e9417,
        expect: &[0xB9, 0x07, 0x00, 0x00, 0x00, 0x83, 0xD9, 0x00],
        new: &[0xB9, 0x06, 0x00, 0x00, 0x00, 0x83, 0xD9, 0x00],
    },
    // Constant tables at 256-byte alignment:
    // the title packs per-draw constants with one bump allocator (fn
    // `title+0x4d9d9a0`) that takes the alignment in r8d; the six
    // constant-buffer callers pass 16 (`mov r8d, 10h`), which the console
    // allows and the host does not (256), so the host copied about 3,700
    // tables per frame. `mov r8d, 100h` aligns them at the source; the
    // allocator aligns relative to its 64 KiB-aligned pool base.
    Patch {
        name: "cbalign",
        what: "graphics set-constants allocates its constants 256-byte aligned",
        rva: 0x4d9f530,
        expect: &[0x41, 0xB8, 0x10, 0x00, 0x00, 0x00],
        new: &[0x41, 0xB8, 0x00, 0x01, 0x00, 0x00],
    },
    Patch {
        name: "cbalign",
        what: "graphics constant-buffer setter caller allocates its constants 256-byte aligned",
        rva: 0x378dd,
        expect: &[0x41, 0xB8, 0x10, 0x00, 0x00, 0x00],
        new: &[0x41, 0xB8, 0x00, 0x01, 0x00, 0x00],
    },
    Patch {
        name: "cbalign",
        what: "graphics constant-buffer setter caller allocates its constants 256-byte aligned",
        rva: 0x248d6ed,
        expect: &[0x41, 0xB8, 0x10, 0x00, 0x00, 0x00],
        new: &[0x41, 0xB8, 0x00, 0x01, 0x00, 0x00],
    },
    Patch {
        name: "cbalign",
        what: "second context set-constants allocates its constants 256-byte aligned",
        rva: 0x4eeccd0,
        expect: &[0x41, 0xB8, 0x10, 0x00, 0x00, 0x00],
        new: &[0x41, 0xB8, 0x00, 0x01, 0x00, 0x00],
    },
    Patch {
        name: "cbalign",
        what: "second context constant-buffer setter caller allocates its constants 256-byte aligned",
        rva: 0x4c11d,
        expect: &[0x41, 0xB8, 0x10, 0x00, 0x00, 0x00],
        new: &[0x41, 0xB8, 0x00, 0x01, 0x00, 0x00],
    },
    Patch {
        name: "cbalign",
        what: "second context constant-buffer setter caller allocates its constants 256-byte aligned",
        rva: 0x248d79d,
        expect: &[0x41, 0xB8, 0x10, 0x00, 0x00, 0x00],
        new: &[0x41, 0xB8, 0x00, 0x01, 0x00, 0x00],
    },
    // Render size: the title's own upscaler
    // sets the world's render size to 2/3 of the 1920x1080 output (1280x720)
    // from a constant stored at fn `title+0x28f1e78` (`mov dword [rcx+90h],
    // 3F2AAAABh`); the code clamps the scale to at least 0.2. `renderscale`
    // stores 1/3 (640x360), `renderscale540` 1/2 (960x540). Output size is
    // unchanged; the title's upscaler brings it back up.
    Patch {
        name: "renderscale",
        what: "the world renders at 1/3 of the output size (640x360)",
        rva: 0x28f1e78,
        expect: &[0xC7, 0x81, 0x90, 0x00, 0x00, 0x00, 0xAB, 0xAA, 0x2A, 0x3F],
        new: &[0xC7, 0x81, 0x90, 0x00, 0x00, 0x00, 0xAB, 0xAA, 0xAA, 0x3E],
    },
    // `renderscale216`: 0.2 (`CD CC 4C 3E`, bytes of the float read from our
    // exe; the same value sits at 0x146159F8C, the clamp floor of slot +80
    // `title+0x7e9440`, so the clamp passes it unchanged): 384x216. Only this
    // kind-6 site matters: the FSR slot-10 scale load in `title+0x28ef680`
    // belongs to the kind-7 FSR 4 path that `upscaler` never selects, so
    // the two sites need not agree.
    Patch {
        name: "renderscale216",
        what: "the world renders at 1/5 of the output size (384x216, the title's floor)",
        rva: 0x28f1e78,
        expect: &[0xC7, 0x81, 0x90, 0x00, 0x00, 0x00, 0xAB, 0xAA, 0x2A, 0x3F],
        new: &[0xC7, 0x81, 0x90, 0x00, 0x00, 0x00, 0xCD, 0xCC, 0x4C, 0x3E],
    },
    Patch {
        name: "renderscale540",
        what: "the world renders at 1/2 of the output size (960x540)",
        rva: 0x28f1e78,
        expect: &[0xC7, 0x81, 0x90, 0x00, 0x00, 0x00, 0xAB, 0xAA, 0x2A, 0x3F],
        new: &[0xC7, 0x81, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3F],
    },
    // World meshlets drawn directly:
    // the chooser fn `title+0x325b4f0` sends a meshlet batch to GPU culling
    // (three `meshlet_culling` dispatches, then DispatchMesh through
    // ExecuteIndirect) when TLS+0x8A0 bits 0 and 1 are set, else to a direct
    // DispatchMesh(meshlets x instances, 1, 1). `directmesh` makes the test
    // at `title+0x325b58a` an unconditional jump to the direct block [V in
    // the machine code; untested in a run]. The direct count is not split by the title
    // and may pass 65535.
    Patch {
        name: "directmesh",
        what: "world meshlet batches take the direct mesh draw, not GPU culling",
        rva: 0x325b58a,
        expect: &[0x75, 0x76],
        new: &[0xEB, 0x76],
    },
    // Game time follows the clock (read in the machine code [V]): the
    // timer update `title+0x22264d0` caps a frame's step at 1/15 s, loaded at
    // `title+0x2226833` (`vmovss xmm0, [title+0x6159ec0]`, a constant 64 sites
    // share). At a one-second host frame the game then runs at 1/15 speed and
    // a cinematic needs 15 times the frames. `realtime` loads the 1.0 at
    // `title+0x6159bc4` instead, so the cap is one second. [I] that this timer
    // drives the cinematic; untested in a run.
    // CORRECTION: this site is the inline
    // clamp that only the paused path takes; while the game runs the clamp is
    // in the helper `title+0x2226040` (load at `title+0x2226045`), so
    // `realtime` had no effect on a running game. Use `timestep1` (both sites).
    Patch {
        name: "realtime",
        what: "a frame's time step is capped at 1 s instead of 1/15 s",
        rva: 0x2226833,
        expect: &[0xC5, 0xFA, 0x10, 0x05, 0x85, 0x36, 0xF3, 0x03],
        new: &[0xC5, 0xFA, 0x10, 0x05, 0x89, 0x33, 0xF3, 0x03],
    },
    // Fixed exposure: the title computes
    // exposure on the GPU from the previous frame's value; one zero or NaN
    // sticks and the scene stays black (seen: a picture for a few frames,
    // then black). Its own fixed-exposure path (fn `title+0x28f93c0`, pass
    // 12) writes the timecycle's starting exposure every frame: the branch at
    // `title+0x119ab14` always takes it, and the value loaded at
    // `title+0x119ab3a` is the frame parameter +0x228 instead of +0x214.
    Patch {
        name: "exposure",
        what: "every frame takes the title's fixed-exposure path",
        rva: 0x119ab14,
        expect: &[0x77, 0x24],
        new: &[0xEB, 0x24],
    },
    Patch {
        name: "exposure",
        what: "the fixed exposure is the title's starting value",
        rva: 0x119ab3a,
        expect: &[0xC5, 0xFA, 0x10, 0x81, 0x14, 0x02, 0x00, 0x00],
        new: &[0xC5, 0xFA, 0x10, 0x81, 0x28, 0x02, 0x00, 0x00],
    },
    // Pass switches (bytes checked in
    // our exe, all UNTESTED in a run). Each gate-writer is `call fn; mov
    // [rip+gate], al`; `xor eax, eax` + NOPs makes the gate 0 so every reader
    // skips its pass. All off by default; ask by name in XNW_PATCH.
    // `noreflect`: stochastic reflections and denoise, gate byte_14876e1c0
    // (writer fn `title+0x22aee0`).
    Patch {
        name: "noreflect",
        what: "reflections and their denoise off (gate byte_14876e1c0 stays 0)",
        rva: 0x22aee6,
        expect: &[0xE8, 0x75, 0xEC, 0xFF, 0xFF],
        new: &[0x31, 0xC0, 0x90, 0x90, 0x90],
    },
    // `noirradiance`: irradiance cache, gate byte_14876d283 (fn `title+0x22cba0`).
    Patch {
        name: "noirradiance",
        what: "irradiance cache off (gate byte_14876d283 stays 0)",
        rva: 0x22cbe8,
        expect: &[0xE8, 0x73, 0xCF, 0xFF, 0xFF],
        new: &[0x31, 0xC0, 0x90, 0x90, 0x90],
    },
    // `noindirect`: indirect diffuse, gate byte_14876a180 (fn `title+0x22a3b0`).
    Patch {
        name: "noindirect",
        what: "indirect diffuse off (gate byte_14876a180 stays 0)",
        rva: 0x22a3b5,
        expect: &[0xE8, 0xA6, 0xF7, 0xFF, 0xFF],
        new: &[0x31, 0xC0, 0x90, 0x90, 0x90],
    },
    // `nort`: gates the BVH build only: per-frame ray-tracing BVH update
    // and skinning, gate byte_149dd8f30 (TLAS update `title+0x3375fd0`; call of
    // `title+0x522dda0` at `title+0x5779e44`). The per-material ray-tracing
    // resolve dispatches sit under the irradiance, reflection and
    // indirect-diffuse gates (`noirradiance`, `noreflect`, `noindirect`).
    Patch {
        name: "nort",
        what: "per-frame ray tracing BVH update and skinning off (gate byte_149dd8f30 stays 0)",
        rva: 0x5779e44,
        expect: &[0xE8, 0x57, 0xCF, 0xF3, 0xFA],
        new: &[0x31, 0xC0, 0x90, 0x90, 0x90],
    },
    // `noshadows`: the branch after `test byte [byte_1484dda00], 1` at
    // `title+0x200dde7` always skips the cascade shadow stages. Weaker: fn
    // `title+0x97db10` still tests bit 0 and consumers may sample undrawn maps.
    Patch {
        name: "noshadows",
        what: "cascade shadows off (only stage 28 is queued; consumers may sample undrawn maps)",
        rva: 0x200ddf1,
        expect: &[0x74, 0x75],
        new: &[0xEB, 0x75],
    },
    // Volumetrics, bloom, lens flares
    // (bytes read from our exe, all UNTESTED in a run).
    // "Returns at entry": the first byte becomes `ret` (0xC3); the callers
    // ignore eax after each call (0x1415b5956..0x1415b5a95, 0x1417bf1fa,
    // 0x1417bf400, 0x1417bf4b9 [V]), so no `xor eax, eax` is
    // needed. Open risk [U]: consumers sampling unwritten targets (froxel
    // volumes qword_14849BA08/BA30, the bloom target in the tonemap
    // composite). `volskip` goes with `volhist`.
    Patch {
        name: "volskip",
        what: "volumetrics pass 0x143279490 returns at entry",
        rva: 0x3279490,
        expect: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xEC],
        new: &[0xC3, 0x57, 0x53, 0x48, 0x83, 0xEC],
    },
    Patch {
        name: "volskip",
        what: "volumetrics pass 0x143272d10 returns at entry",
        rva: 0x3272d10,
        expect: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xEC],
        new: &[0xC3, 0x57, 0x53, 0x48, 0x83, 0xEC],
    },
    Patch {
        name: "volskip",
        what: "volumetrics pass 0x1432730a0 returns at entry",
        rva: 0x32730a0,
        expect: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xEC],
        new: &[0xC3, 0x57, 0x53, 0x48, 0x83, 0xEC],
    },
    Patch {
        name: "volskip",
        what: "volumetrics pass 0x143279860 returns at entry",
        rva: 0x3279860,
        expect: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xEC],
        new: &[0xC3, 0x57, 0x53, 0x48, 0x83, 0xEC],
    },
    Patch {
        name: "volskip",
        what: "volumetrics apply 0x143279d40 returns at entry",
        rva: 0x3279d40,
        expect: &[0x56, 0x57, 0x53, 0x48, 0x83, 0xEC],
        new: &[0xC3, 0x57, 0x53, 0x48, 0x83, 0xEC],
    },
    // `volhist`: the view block's volumetric-history bytes (writer
    // `title+0xf742be..0xf74375`): +1 picks the history volume or the fallback
    // qword_148FCFC38, +3 does the same for the volume textures.
    Patch {
        name: "volhist",
        what: "view block +1 = 0: no volumetric history, fallback volume bound",
        rva: 0xf742ca,
        expect: &[0x75, 0x0D],
        new: &[0xEB, 0x0D],
    },
    Patch {
        name: "volhist",
        what: "view block +3 = 0: 0x14200c780/7d0 return the fallback texture",
        rva: 0xf7434a,
        expect: &[0x34, 0x01],
        new: &[0x30, 0xC0],
    },
    Patch {
        name: "nobloom",
        what: "bloom 0x1428e27c0 (both calls) returns at entry",
        rva: 0x28e27c0,
        expect: &[0x41, 0x57, 0x41, 0x56],
        new: &[0xC3, 0x57, 0x41, 0x56],
    },
    Patch {
        name: "noflare",
        what: "lens flares 0x1428f6960 returns at entry",
        rva: 0x28f6960,
        expect: &[0x41, 0x57, 0x41, 0x55],
        new: &[0xC3, 0x57, 0x41, 0x55],
    },
    // Streamed-resource placement
    // (bytes read from our exe, all UNTESTED in a run). The main thread places
    // streamed resources in `title+0x9d94a0` until a QPC deadline of 250-1250 us
    // per FRAME (budget obj+0x166C8 of qword_14764B240, clamped by fn
    // `title+0x2294750`, deadline fn `title+0x2576d40`). At our 1-2 frames per
    // second that is 0.25-2.5 ms of placement per second of wall time, so
    // streamed models and textures arrive 15-30 times slower than on the
    // console [I that this is the main cause of missing world geometry and slow
    // loads]. `place_unsliced`: the `jz` at `title+0x2576d7e` (budget 0 = no
    // limit) becomes an unconditional jump to the same target, so the deadline
    // is always unlimited. `place_budget`: the clamp's 1250 us ceiling becomes
    // 30000 us (the conservative one). `tex_upgrades`: NOPs both `jge`
    // branches of maxTexUpgradesPerFrame (dword_146849848 = 5).
    Patch {
        name: "place_unsliced",
        what: "the placement deadline is always unlimited (no per-frame time slice)",
        rva: 0x2576d77,
        expect: &[0x83, 0xBE, 0xC8, 0x66, 0x01, 0x00, 0x00, 0x0F, 0x84, 0x2B, 0x01, 0x00, 0x00],
        new: &[0xE9, 0x33, 0x01, 0x00, 0x00, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90],
    },
    Patch {
        name: "place_budget",
        what: "slice clamp max 1250 -> 30000 us (mov ecx)",
        rva: 0x2294814,
        expect: &[0xB9, 0xE2, 0x04, 0x00, 0x00],
        new: &[0xB9, 0x30, 0x75, 0x00, 0x00],
    },
    Patch {
        name: "place_budget",
        what: "slice clamp max 1250 -> 30000 us (cmp eax)",
        rva: 0x2294819,
        expect: &[0x3D, 0xE2, 0x04, 0x00, 0x00],
        new: &[0x3D, 0x30, 0x75, 0x00, 0x00],
    },
    Patch {
        name: "tex_upgrades",
        what: "ignore maxTexUpgradesPerFrame (jge -> nop) 1/2",
        rva: 0x2203930,
        expect: &[0x0F, 0x8D, 0x7A, 0xFF, 0xFF, 0xFF],
        new: &[0x90, 0x90, 0x90, 0x90, 0x90, 0x90],
    },
    Patch {
        name: "tex_upgrades",
        what: "ignore maxTexUpgradesPerFrame (jge -> nop) 2/2",
        rva: 0x220398c,
        expect: &[0x0F, 0x8D, 0x1E, 0xFF, 0xFF, 0xFF],
        new: &[0x90, 0x90, 0x90, 0x90, 0x90, 0x90],
    },
    // Game-time step cap (expect bytes
    // re-read from our exe, UNTESTED in a run). The title advances every game
    // clock by min(frame time, 1/15 s) per frame (helper `title+0x2226040`,
    // bound loaded at `title+0x2226045` from `title+0x6159ec0`); at our 1-2 s
    // frames the game runs at 1/15 to 1/30 speed. These raise the cap by
    // pointing the `vmovss` at another float constant (rip-relative disp
    // changed): `timestep4` 0.25 s (`title+0x6159ca4`), `timestep2` 0.5 s
    // (`title+0x6159b10`), `timestep1` 1.0 s (`title+0x6159bc4`). Two sites
    // each: the helper (the running game) and the inline clamp at
    // `title+0x2226833` (pause path). Derived values follow by themselves
    // (ms step, 1/step, accumulated ms). The three names share the same sites
    // and expect bytes: apply order is table order, so with two named the
    // first in the table (4, then 2, then 1) applies and the later one logs
    // "skipped: the bytes are not the expected ones" at each site; no harm.
    Patch {
        name: "timestep4",
        what: "a frame's game-time step is capped at 0.25 s instead of 1/15 s (helper)",
        rva: 0x2226045,
        expect: &[0xC5, 0xFA, 0x10, 0x1D, 0x73, 0x3E, 0xF3, 0x03],
        new: &[0xC5, 0xFA, 0x10, 0x1D, 0x57, 0x3C, 0xF3, 0x03],
    },
    Patch {
        name: "timestep4",
        what: "a frame's game-time step is capped at 0.25 s instead of 1/15 s (inline)",
        rva: 0x2226833,
        expect: &[0xC5, 0xFA, 0x10, 0x05, 0x85, 0x36, 0xF3, 0x03],
        new: &[0xC5, 0xFA, 0x10, 0x05, 0x69, 0x34, 0xF3, 0x03],
    },
    Patch {
        name: "timestep2",
        what: "a frame's game-time step is capped at 0.5 s instead of 1/15 s (helper)",
        rva: 0x2226045,
        expect: &[0xC5, 0xFA, 0x10, 0x1D, 0x73, 0x3E, 0xF3, 0x03],
        new: &[0xC5, 0xFA, 0x10, 0x1D, 0xC3, 0x3A, 0xF3, 0x03],
    },
    Patch {
        name: "timestep2",
        what: "a frame's game-time step is capped at 0.5 s instead of 1/15 s (inline)",
        rva: 0x2226833,
        expect: &[0xC5, 0xFA, 0x10, 0x05, 0x85, 0x36, 0xF3, 0x03],
        new: &[0xC5, 0xFA, 0x10, 0x05, 0xD5, 0x32, 0xF3, 0x03],
    },
    Patch {
        name: "timestep1",
        what: "a frame's game-time step is capped at 1 s instead of 1/15 s (helper)",
        rva: 0x2226045,
        expect: &[0xC5, 0xFA, 0x10, 0x1D, 0x73, 0x3E, 0xF3, 0x03],
        new: &[0xC5, 0xFA, 0x10, 0x1D, 0x77, 0x3B, 0xF3, 0x03],
    },
    Patch {
        name: "timestep1",
        what: "a frame's game-time step is capped at 1 s instead of 1/15 s (inline)",
        rva: 0x2226833,
        expect: &[0xC5, 0xFA, 0x10, 0x05, 0x85, 0x36, 0xF3, 0x03],
        new: &[0xC5, 0xFA, 0x10, 0x05, 0x89, 0x33, 0xF3, 0x03],
    },
];

/// The PE header timestamp of the image mapped at `base`.
///
/// # Safety
/// `base` is a mapped PE image with its headers.
pub(crate) unsafe fn timestamp(base: usize) -> u32 {
    // SAFETY: the DOS header's `e_lfanew`, then the file header's timestamp.
    unsafe {
        let nt = (base as *const u8).add(0x3c).cast::<u32>().read_unaligned() as usize;
        (base as *const u8).add(nt + 8).cast::<u32>().read_unaligned()
    }
}

/// Applies the patches `XNW_PATCH` names to the title at `base`.
///
/// # Safety
/// `base` is the mapped title and none of its code runs yet.
pub unsafe fn apply(base: usize) {
    let wanted: Vec<String> = std::env::var("XNW_PATCH").unwrap_or_default().split(',').map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()).collect();
    if wanted.is_empty() {
        return;
    }
    for name in &wanted {
        if !PATCHES.iter().any(|p| p.name == name) {
            xnw_core::warn!("XNW_PATCH: no patch named {name}");
        }
    }
    // SAFETY: forwarded.
    let stamp = unsafe { timestamp(base) };
    if stamp != BUILD_1_0_2_1 {
        xnw_core::warn!("XNW_PATCH: title timestamp {stamp:#x} is not build 1.0.2.1 ({BUILD_1_0_2_1:#x}); nothing patched");
        return;
    }
    for p in PATCHES.iter().filter(|p| wanted.iter().any(|w| w == p.name)) {
        let at = (base + p.rva) as *mut u8;
        // SAFETY: inside the mapped .text of this build.
        let now = unsafe { std::slice::from_raw_parts(at, p.expect.len()) };
        if now != p.expect || p.new.len() != p.expect.len() {
            xnw_core::warn!("patch {} at title+{:#x} skipped: the bytes are not the expected ones", p.name, p.rva);
            continue;
        }
        let mut old = 0;
        // SAFETY: those bytes only, protection restored below.
        if unsafe { VirtualProtect(at.cast(), p.new.len(), PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
            xnw_core::warn!("patch {} at title+{:#x}: VirtualProtect failed", p.name, p.rva);
            continue;
        }
        // SAFETY: writable now; the bytes were checked above.
        unsafe {
            std::ptr::copy_nonoverlapping(p.new.as_ptr(), at, p.new.len());
            VirtualProtect(at.cast(), p.new.len(), old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), at.cast(), p.new.len());
        }
        xnw_core::info!("patch {} applied at title+{:#x}: {}", p.name, p.rva, p.what);
    }
}
