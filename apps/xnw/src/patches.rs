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
    // Engine levers (ENG1; bytes read from our exe, all UNTESTED in a run).
    // Under the null GPU our pass leaves out the stages below, yet the title
    // still scans, lists and records them. Each is off unless named.
    // `nolocalshadows`: the local-light shadow manager `title+0x1518430`
    // returns at once while the game is paused (its first call, the pause
    // test `title+0x764870`, then `test al, al; jne` to the epilogue).
    // `mov al, 1` in place of that call takes the same exit every frame: no
    // slot's update bit (record +576 bit 0 of `title+0x84be240`) is set, so
    // the 16 shadow phases (vt+0x58 `title+0x51d12f0`, vt+0x60
    // `title+0x55bcad0`) stay out of the scan, the list build and stage 21
    // (246-2,056 hooked draws a frame in pk7f/pk8a/pk7h).
    Patch {
        name: "nolocalshadows",
        what: "the local-light shadow manager takes its paused exit: no shadow phase is scanned, listed or recorded (stage 21)",
        rva: 0x1518488,
        expect: &[0xE8, 0xE3, 0xC3, 0x24, 0xFF],
        new: &[0xB0, 0x01, 0x90, 0x90, 0x90],
    },
    // `nopuddles` (ENG5): the puddle phase (stage 42: a top-down view
    // of the ground around the player, 63-66 hooked draws a frame in
    // pk7h/pk8a, read back by `title+0x328ef50` inside the same callback) is
    // active while byte_1484DB920 is 1: its vt+0x58 `title+0x655590` copies
    // the byte to phase+0x752 (the scan's gate) and its vt+0x60
    // `title+0x328ef40` returns it (the registrar's gate, `title+0x804a30`).
    // The one writer, `title+0x6558c0`, stores 1 when the player object
    // qword_14AB56E38 exists, else 0. `mov al, 1` becomes `mov al, 0`: the
    // title's own no-player state on every frame. The position it stores
    // beside the byte (xmmword_1484DB910, what its second caller
    // `title+0xd6e110` uses; that one ignores al) is still computed. The CPU
    // readers of the byte then answer "no puddle": `title+0x52db830` (puddle
    // depth at a point, 9 callers) returns 0.0, `title+0x328f550` returns 0.
    // Under the null GPU the map is never drawn, so they have nothing to
    // read today either.
    Patch {
        name: "nopuddles",
        what: "the puddle phase is never active (its gate byte_1484DB920 stays 0): not scanned, listed or recorded (stage 42)",
        rva: 0x65591e,
        expect: &[0xB0, 0x01, 0xC5, 0xF8, 0x28, 0x06],
        new: &[0xB0, 0x00, 0xC5, 0xF8, 0x28, 0x06],
    },
    // `novfx` (ENG6): stage 13 of the main view (alpha/VFX: the particle
    // list, bucket 30 of the phase's list slot, through `title+0x25cedb0`;
    // far-light sprites `title+0x10755e0`; the lightning strike
    // `title+0xcd94a0`; the OIT resolve dispatch `title+0x25d1c60`) is
    // queued by the second-view registrar `title+0x200f580` between stages
    // 11 and 14, each by its own call of `drawlist_queue_stage`
    // `title+0x4ff02e0`. The middle call becomes a 5-byte nop: the stage is
    // never queued (rsi/rdi, which the third call uses, are untouched; the
    // return value is unused). Our pass drops every stage-13 draw
    // (`OTHER_VIEW_STAGES`). Escapes: the body's direct-call closure (399
    // functions, depth 8) writes 50 globals; the ones read outside it are
    // render-context, job and ring state, and the VFX random state
    // qword_149AAC800 (advanced by the lightning draw `title+0x336b000`).
    // Stage 14 sets its own state (`title+0x205d30`, `title+0xa689a0`,
    // `title+0x1f8dcc0` at its head), so nothing carries over from 13. The
    // draw handlers the list walk calls virtually were not followed.
    Patch {
        name: "novfx",
        what: "stage 13 (alpha/VFX of the main view: particles, far-light sprites, lightning, OIT resolve) is never queued",
        rva: 0x200f5cb,
        // call title+0x4ff02e0 (drawlist_queue_stage(phase, 13, title+0x11544b0, 0, 0))
        expect: &[0xE8, 0x10, 0x0D, 0xFE, 0x02],
        new: &[0x0F, 0x1F, 0x44, 0x00, 0x00],
    },
    // `nolighting` (ENG8): stage 1, the lighting compute chain (callback
    // `title+0x15b4aa0` -> body `title+0x15b4ab0`, 9,171 bytes: 190 callees,
    // the indirect diffuse, reflection, irradiance cache and deferred light
    // dispatches, both ray-tracing builds and one `drawlist_exec`; 4.4-5.0 ms
    // of CPU a frame in a test run, and no hooked draw: ENG1). Its registrar
    // `title+0x200e020` is one call, `drawlist_queue_stage(phase, 1,
    // title+0x15b4aa0, 0, 0)`, which becomes a 5-byte nop: the stage is never
    // queued, as `novfx` does for stage 13. Read outside its closure (ENG6):
    // the light level at a point dword_14763B90C (a readback, empty under
    // null), a history flag byte_14876F940, context and job state. Our
    // screen pass lights from the timecycle (`ourlight`) and reads none of
    // it; a capture for the bridged lighting programs (LIT) needs the
    // stage and must not name this patch. Untested in a run: the run with
    // `stskip` (`XNW_STAGE_SKIP=1`, the same thing through the stage meter)
    // is the functional test.
    Patch {
        name: "nolighting",
        what: "stage 1 (the lighting compute chain) is never queued",
        rva: 0x200e038,
        // call title+0x4ff02e0 (drawlist_queue_stage(phase, 1, title+0x15b4aa0, 0, 0))
        expect: &[0xE8, 0xA3, 0x22, 0xFE, 0x02],
        new: &[0x0F, 0x1F, 0x44, 0x00, 0x00],
    },
    // `noprobecap` (ENG6): the probe cube capture, the second callback of
    // stage 36 (`title+0x4413b0` -> `title+0x15298e0`, 10,495 bytes; the
    // other one, the sky LUT callback `title+0xd3ac60`, is untouched). The
    // callback walks the five lists of 272-byte capture records of the
    // owner qword_148F05D50 (+0x30..+0x70, step 16). A record that is not
    // done (+0xD8 == 0) is tested three ways before anything is drawn: no
    // target (+0xE0 == 0, `je title+0x1529c32`), an element that is not
    // ready (`title+0x5047c40` == 0), a face size above the owner's limit
    // (+0xC0 > 1 << owner+0x688, `ja title+0x152bd86`). Each of the three
    // exits stores done (+0xD8 = 1) and the frame number (+0xD0 =
    // dword_14A9437AC) and goes to the next record; only a record that
    // passes all three gets the six-face loop (5 targets + depth,
    // geometry_draw and DrawPrimitive, 24-54 hooked draws a frame in
    // pk7h/pk8a, all dropped by our pass). The first `je` becomes `jmp`:
    // every pending record takes the title's own no-target exit, so the
    // records' done marks and frame stamps are written as before and no
    // face is drawn. The owner's copy loop after the lists (+0xB4/+0xB8)
    // still runs. Escapes of the callback's direct-call closure (271
    // functions): render-context, job and ring state and the material
    // table cache qword_14929CF00, nothing of the capture's own.
    Patch {
        name: "noprobecap",
        what: "the probe cube capture (stage 36's second callback) marks every pending record done and draws no face; the sky LUT callback is untouched",
        rva: 0x1529ada,
        // je title+0x1529c32 (after cmp qword [r13+0xe0], 0)
        expect: &[0x0F, 0x84, 0x52, 0x01, 0x00, 0x00],
        // jmp title+0x1529c32; nop
        new: &[0xE9, 0x53, 0x01, 0x00, 0x00, 0x90],
    },
    // `noshadowscan`: the cascade phase's activity test (vt+0x58
    // `title+0x97db10`: `test byte_1484dda00, 1; jz`) always skips, so the
    // phase is not scanned or listed. Goes with `noshadows` (the registrar).
    Patch {
        name: "noshadowscan",
        what: "the cascade shadow phase is never active for the scan (with noshadows)",
        rva: 0x97db1c,
        expect: &[0x74, 0x34],
        new: &[0xEB, 0x34],
    },
    // `noprepass`: the depth prepass stage 65 (thunk `title+0x1353d40` ->
    // body `title+0x1353d50`) keeps its head and skips its draws. The head
    // is not prepass work: `title+0xfc12b0` (a compute dispatch),
    // `title+0xa28030`, `title+0x27c2190` (the ped pre-skin batches of the
    // queues qword_1478A9A28 and the next, owner `ped_skinning_cs` `title+0x27c0400`)
    // and `title+0x8cf5d0` run before anything is drawn. Then the body tests
    // the phase's list flags (+0x73A0 bits 0-2, or the view-block case in
    // bl) and, with none set, jumps to its pass end `title+0x13548ac`
    // (`call title+0x5003da0`): the title's own nothing-to-draw path. That
    // `je` becomes `jmp`, so the depth-only setup, the three list walks (642
    // hooked draws a frame in pk7f), the GPU culling and the HiZ chain (read
    // by GPU culling only, FLK1) are left out and the pre-skin still runs.
    // Until ENG5 this entry returned at the thunk (`e9 0b 00 00 00` -> `c3
    // ...`): the pre-skin batches went with it, the PreSkinD draws of the
    // G-buffer stages named outputs nobody had computed, and our join drew 6
    // bodies of 184 (world-run-free2.log against 105 of 123 unpatched).
    Patch {
        name: "noprepass",
        what: "the depth prepass stage 65 takes its nothing-to-draw branch: its head (ped pre-skin batches) runs, its depth draws, GPU culling and HiZ do not",
        rva: 0x1353e8e,
        // test al, 1; je title+0x13548ac
        expect: &[0xA8, 0x01, 0x0F, 0x84, 0x16, 0x0A, 0x00, 0x00],
        // test al, 1; jmp title+0x13548ac; nop
        new: &[0xA8, 0x01, 0xE9, 0x17, 0x0A, 0x00, 0x00, 0x90],
    },
    // `features_nort`: the render feature mask (`mov eax, 0xDE4F` in
    // `title+0x1ca31f0`, stored to dword_1476B8FE0/FE4 and tested through
    // `title+0x522dda0`) without mask bits 0x4 and 0x8: the near and far
    // ray-tracing acceleration structures (getters `title+0x6b6da0`,
    // `title+0x15de0e0`; gates two bytes at byte_149DD8F30). Which programs the
    // lighting then runs (the pack names `_Voxels` and `_Fallback` variants
    // beside `_BVH`) is unproven.
    // `nojitter`: the camera jitter generator (TAA object vtable slot 8,
    // `title+0x6e30d0`; mode dword obj+0x1B0) takes the zero branch for mode
    // 3, the sequence `frac(n * (0.754877627, 0.569840252) + 0.5) - 0.5`
    // `jne` after `cmp r8d, 3` becomes `jmp`, so mode 3 falls through
    // the mode 1 and 2 tests to `vxorps xmm0, xmm0, xmm0`. Why: 7 * 0.5698 =
    // 3.9889, so the y jitter repeats every 7 frames within 0.011 px, and a
    // geometry at the edge of the title's draw-time view test then drops on
    // the same 1 or 3 frames of every 7 under a still camera (FLK4) [I].
    Patch {
        name: "nojitter",
        what: "the camera jitter is (0, 0) in the no-jitter mode (the title's own no-jitter branch)",
        rva: 0x6e30ed,
        expect: &[0x75, 0x77],
        new: &[0xEB, 0x77],
    },
    Patch {
        name: "features_nort",
        what: "render feature mask 0xDE4F -> 0xDE43: no near or far ray-tracing acceleration structure",
        rva: 0x1ca31f0,
        expect: &[0xB8, 0x4F, 0xDE, 0x00, 0x00],
        new: &[0xB8, 0x43, 0xDE, 0x00, 0x00],
    },
    // `notrail` (ENG5/ENG7): stage 44, the trail map (deep surface and grass
    // flattening; phase vtable 0x1463DEA90). The phase has no state of its
    // own to turn off: its "active" slot vt+0x60 is the shared `mov al, 1`
    // stub and its view setup vt+0x58 (`title+0xcefc50`) always stores
    // phase+0x752 = 1, the flag the scan setup asks before it builds the
    // phase's descriptor. Two changes give it the state the puddle
    // phase has without a player: the store becomes 0 (no scan, no lists),
    // and the call that queues the stage in its registrar vt+0x68
    // (`title+0x200f6c0`: `drawlist_queue_stage(list, 44, title+0x280000)`)
    // becomes a 5-byte nop. The callback's closure (306 functions) hands the
    // CPU nothing: context, job and ring state, and a name map made on first
    // use (xmmword_149588C20, freed null-safe at shutdown). The update-side
    // trail job (`title+0x35bb450`) is not touched.
    Patch {
        name: "notrail",
        what: "the trail-map phase stays out of the scan (phase+0x752 = 0)",
        rva: 0xcefde0,
        expect: &[0xC6, 0x86, 0x52, 0x07, 0x00, 0x00, 0x01],
        new: &[0xC6, 0x86, 0x52, 0x07, 0x00, 0x00, 0x00],
    },
    Patch {
        name: "notrail",
        what: "stage 44 (trail map) is not queued",
        rva: 0x200f6d8,
        expect: &[0xE8, 0x03, 0x0C, 0xFE, 0x02],
        new: &[0x0F, 0x1F, 0x44, 0x00, 0x00],
    },
    // `rtfallback` (ENG7): the lighting picks its programs by one test, "is
    // acceleration-structure set i of this manager there" (`title+0x4fd8570`:
    // `cmp qword [sets + i*8], 0; jne title+0x4fd8590`, else `xor eax, eax;
    // ret`), asked through `title+0x4fd8550` for the lighting manager
    // qword_148527DC8 (40 callers) and `title+0x49be10` for the shadow one
    // qword_1484CEA58. A yes picks the `_BVH` programs (indirect diffuse
    // `title+0x356b6b0`: passes 37-40 `CS_ScreenProbes_TraceRays_RayBinning_
    // BVH_*`), a no the title's own non-ray-traced ones (pass 42 `..._TraceRays_
    // Fallback`, `CS_PPR_RaytraceFallback`, `CS_LaserLightShadowRaytracing
    // Fallback`) and skips the irradiance-cache and reflection trace
    // dispatches. No device caps answer, settings field or platform query
    // feeds it. The `jne` becomes a 6-byte nop: every set reads as absent.
    // Goes with `features_nort` (no structure builds).
    Patch {
        name: "rtfallback",
        what: "the acceleration-structure set test answers no: the lighting takes its non-ray-traced programs (_Fallback)",
        rva: 0x4fd857b,
        expect: &[0x0F, 0x85, 0x0F, 0x00, 0x00, 0x00],
        new: &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
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
        // push r15; push r14 (ENG5: the entry said 41 55, push r13, which the exe does not hold: it never applied)
        expect: &[0x41, 0x57, 0x41, 0x56],
        new: &[0xC3, 0x57, 0x41, 0x56],
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
        // `noencode` and `noencodebatch` are made of hooks, not bytes (`encode.rs`, installed after the detours).
        if !PATCHES.iter().any(|p| p.name == name) && !crate::encode::NAMES.contains(&name.as_str()) {
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

/// The patches `XNW_AB_PATCH` may write while the title runs (`ab.rs`).
/// Each entry's changed bytes lie in one aligned 8-byte word (unit test),
/// written with one locked compare-exchange, so a thread fetching there
/// sees the old instructions or the new, never half; the new bytes are
/// whole instructions over whole instructions (a call or jcc made a
/// mov/jmp, an immediate or displacement changed, a `ret` at an entry), so
/// a return address after a replaced call stays an instruction start; and
/// each is a switch the title takes every frame, not a size made once.
/// Left out: crossing a word (`noreflect`, `noindirect`, `nort`,
/// `realtime`, `timestep*`, `tex_upgrades`, `place_unsliced`, `watchdog`),
/// or read once at start (`upscaler`, `cbalign`, `renderscale*`). Code-safe
/// is not state-safe: what a pass's consumers make of it switching off
/// mid-run is what the B half of the run shows.
pub(crate) const LIVE: &[&str] = &[
    "nolocalshadows", "noprepass", "noshadows", "noshadowscan", "nojitter", "features_nort", "noirradiance", "nopuddles", "volskip",
    "volhist", "nobloom", "noflare", "directmesh", "exposure", "place_budget", "nolighting",
];

/// The aligned word at `rva & !7` holding every byte `p` changes, if one does.
fn live_word(p: &Patch) -> Option<usize> {
    let first = p.rva + (0..p.expect.len()).find(|&i| p.expect[i] != p.new[i])?;
    let last = p.rva + (0..p.expect.len()).rfind(|&i| p.expect[i] != p.new[i])?;
    (first & !7 == last & !7).then_some(first & !7)
}

/// Applies the `LIVE` patches named in `names` while the title runs: per
/// entry, the bytes are checked, then the aligned word holding the change
/// is swapped in one locked compare-exchange. Returns the names applied.
///
/// # Safety
/// `base` is the mapped title of build 1.0.2.1.
pub(crate) unsafe fn apply_live(base: usize, names: &[String]) -> Vec<String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut done: Vec<String> = Vec::new();
    for p in PATCHES.iter().filter(|p| LIVE.contains(&p.name) && names.iter().any(|n| n == p.name)) {
        let Some(word) = live_word(p) else { continue };
        let at = base + word;
        let mut old = 0;
        // SAFETY: one word of the mapped .text; protection restored below.
        if unsafe { VirtualProtect(at as *mut _, 8, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
            xnw_core::warn!("ab: patch {} at title+{:#x}: VirtualProtect failed", p.name, p.rva);
            continue;
        }
        // SAFETY: an aligned word of the image, writable now.
        let cell = unsafe { &*(at as *const AtomicU64) };
        let now = cell.load(Ordering::Acquire).to_le_bytes();
        let mut want = now;
        let mut ok = true;
        for i in 0..p.expect.len() {
            let a = base + p.rva + i;
            if a & !7 == at {
                ok &= now[a - at] == p.expect[i];
                want[a - at] = p.new[i];
            } else {
                // SAFETY: a byte of the entry outside the word: unchanged by the patch, checked only.
                ok &= unsafe { (a as *const u8).read() } == p.expect[i];
            }
        }
        let swapped = ok && cell.compare_exchange(u64::from_le_bytes(now), u64::from_le_bytes(want), Ordering::AcqRel, Ordering::Acquire).is_ok();
        // SAFETY: as above.
        unsafe {
            VirtualProtect(at as *mut _, 8, old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), at as *const _, 8);
        }
        if swapped {
            xnw_core::info!("ab: patch {} applied live at title+{:#x}: {}", p.name, p.rva, p.what);
            if !done.iter().any(|d| d == p.name) {
                done.push(p.name.to_owned());
            }
        } else {
            xnw_core::warn!("ab: patch {} at title+{:#x} not applied: the bytes are not the expected ones", p.name, p.rva);
        }
    }
    done
}

/// The title's executable on disk, for tests that check our byte tables
/// against it: `XNW_TITLE_EXE`, else `game\\the title's executable.
/// None when the file is not there (the test then checks nothing).
#[cfg(test)]
pub(crate) struct Exe {
    bytes: Vec<u8>,
    /// (rva, raw size, file offset) per section.
    sections: Vec<(usize, usize, usize)>,
}

#[cfg(test)]
impl Exe {
    pub(crate) fn open() -> Option<Exe> {
        let path = std::env::var("XNW_TITLE_EXE").unwrap_or_else(|_| r"title.exe".to_owned());
        let bytes = std::fs::read(&path).ok()?;
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
        let u16_at = |o: usize| u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap()) as usize;
        let nt = u32_at(0x3c);
        assert_eq!(u32_at(nt + 8) as u32, BUILD_1_0_2_1, "{path} is not build 1.0.2.1");
        let first = nt + 24 + u16_at(nt + 20);
        let sections = (0..u16_at(nt + 6)).map(|i| first + 40 * i).map(|o| (u32_at(o + 12), u32_at(o + 16), u32_at(o + 20))).collect();
        Some(Exe { bytes, sections })
    }

    /// The file's bytes at `title+rva`.
    pub(crate) fn at(&self, rva: usize, len: usize) -> &[u8] {
        let &(start, _, offset) = self.sections.iter().find(|&&(start, size, _)| rva >= start && rva + len <= start + size).unwrap_or_else(|| panic!("title+{rva:#x} is in no section's file bytes"));
        &self.bytes[offset + rva - start..][..len]
    }

    /// Where the `call`/`jmp rel32` at `title+rva` goes.
    pub(crate) fn rel32_target(&self, rva: usize) -> usize {
        let b = self.at(rva, 5);
        (rva + 5).wrapping_add(i32::from_le_bytes(b[1..5].try_into().unwrap()) as isize as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_patch_replaces_its_bytes_with_as_many_others() {
        for p in PATCHES {
            assert_eq!(p.expect.len(), p.new.len(), "{} at title+{:#x}", p.name, p.rva);
            assert_ne!(p.expect, p.new, "{} at title+{:#x}", p.name, p.rva);
        }
    }

    /// Every live patch exists and changes, in each entry, bytes of one
    /// aligned word only; applied live on a copy it gives the same bytes
    /// as the boot-time write.
    #[test]
    fn every_live_patch_changes_one_aligned_word() {
        for name in LIVE {
            let entries: Vec<&Patch> = PATCHES.iter().filter(|p| p.name == *name).collect();
            assert!(!entries.is_empty(), "{name}: no such patch");
            for p in entries {
                assert!(live_word(p).is_some(), "{name} at title+{:#x}: its change crosses an 8-byte word", p.rva);
            }
        }
        for left_out in ["noreflect", "nort", "timestep1", "watchdog", "cbalign", "renderscale"] {
            assert!(!LIVE.contains(&left_out), "{left_out}");
        }
        // apply_live on a fake image: 64 KiB-aligned like the title's base, the
        // entry's expected bytes at its rva; only the patch's bytes change.
        let p = PATCHES.iter().find(|p| p.name == "noprepass").unwrap();
        let mut img = vec![0xCCu8; (p.rva & 0xFFFF) + 0x1_0000 + 32];
        let base = img.as_ptr() as usize;
        let fake = (base + 0xFFFF) & !0xFFFF;
        let off = fake - base + (p.rva & 0xFFFF);
        img[off..off + p.expect.len()].copy_from_slice(p.expect);
        let fake_base = fake + (p.rva & 0xFFFF) - p.rva;
        // SAFETY: the entry's word and bytes lie inside `img`.
        let done = unsafe { apply_live(fake_base, &["noprepass".to_owned(), "watchdog".to_owned()]) };
        assert_eq!(done, ["noprepass"]);
        assert_eq!(&img[off..off + p.new.len()], p.new);
        assert!(img[off - 8..off].iter().chain(&img[off + p.new.len()..off + p.new.len() + 8]).all(|&b| b == 0xCC));
        // SAFETY: as above; a second time the bytes are no longer the expected ones.
        assert!(unsafe { apply_live(fake_base, &["noprepass".to_owned()]) }.is_empty());
    }

    /// Every entry's expected bytes are the executable's at its address.
    #[test]
    fn the_expected_bytes_are_the_executables() {
        let Some(exe) = Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): the patch table was not checked against it");
            return;
        };
        let wrong: Vec<String> = PATCHES.iter().filter(|p| exe.at(p.rva, p.expect.len()) != p.expect).map(|p| format!("{} at title+{:#x}", p.name, p.rva)).collect();
        assert!(wrong.is_empty(), "expected bytes that are not the executable's: {wrong:?}");
    }

    /// `novfx` removes the one call that queues stage 13 and nothing else:
    /// the registrar's three calls go to `drawlist_queue_stage` with stages
    /// 11, 13 and 14, the middle one's callback is the stage-13 thunk, and
    /// the bytes between the calls only reload the arguments.
    #[test]
    fn novfx_removes_the_queueing_of_stage_13_only() {
        let p = PATCHES.iter().find(|p| p.name == "novfx").unwrap();
        assert_eq!(p.new, [0x0F, 0x1F, 0x44, 0x00, 0x00], "one 5-byte nop");
        let Some(exe) = Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): novfx was not checked against it");
            return;
        };
        let lea_r8 = |at: usize| {
            let b = exe.at(at, 7);
            assert_eq!(b[..3], [0x4C, 0x8D, 0x05], "lea r8, [rip+..] at title+{at:#x}");
            (at + 7).wrapping_add(i32::from_le_bytes(b[3..7].try_into().unwrap()) as isize as usize)
        };
        // (call site, `mov edx, stage` site, `lea r8, callback` site, stage, callback thunk)
        for (call, mov, lea, stage, cb) in [(0x200f5af, 0x200f59f, 0x200f598, 11u8, 0x1258ad0), (0x200f5cb, 0x200f5be, 0x200f5b4, 13, 0x11544b0), (0x200f5e7, 0x200f5df, 0x200f5d5, 14, 0x1304dd0)] {
            assert_eq!((exe.at(call, 1)[0], exe.rel32_target(call)), (0xE8, 0x4ff02e0), "title+{call:#x}");
            assert_eq!(exe.at(mov, 5), [0xBA, stage, 0, 0, 0], "title+{mov:#x}");
            assert_eq!(lea_r8(lea), cb, "title+{lea:#x}");
        }
        assert_eq!(p.rva, 0x200f5cb);
        // The stage-13 thunk enters the body ENG6 analysed.
        assert_eq!((exe.at(0x11544b0, 1)[0], exe.rel32_target(0x11544b0)), (0xE9, 0x11544c0));
        // After the removed call the third call's arguments are set from rsi/rdi and constants only.
        assert_eq!(exe.at(0x200f5d0, 5), [0x40, 0x88, 0x7C, 0x24, 0x20], "mov [rsp+20h], dil");
        assert_eq!(exe.at(0x200f5dc, 3), [0x48, 0x89, 0xF1], "mov rcx, rsi");
    }

    /// `nolighting` removes the one call of the stage-1 registrar: the whole
    /// function is `sub rsp, 28h; lea r8, callback; mov edx, 1; xor r9d, r9d;
    /// mov byte [rsp+20h], 0; call drawlist_queue_stage; nop; add rsp, 28h;
    /// ret`, and the callback is the thunk of the body ENG6 analysed.
    #[test]
    fn nolighting_removes_the_queueing_of_stage_1_only() {
        let p = PATCHES.iter().find(|p| p.name == "nolighting").unwrap();
        assert_eq!((p.rva, p.new), (0x200e038, &[0x0F, 0x1F, 0x44, 0x00, 0x00][..]), "one 5-byte nop");
        assert_eq!(PATCHES.iter().filter(|p| p.name == "nolighting").count(), 1);
        let Some(exe) = Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): nolighting was not checked against it");
            return;
        };
        assert_eq!(exe.at(0x200e020, 4), [0x48, 0x83, 0xEC, 0x28]);
        let lea = exe.at(0x200e024, 7);
        assert_eq!(lea[..3], [0x4C, 0x8D, 0x05]);
        assert_eq!(0x200e02busize.wrapping_add(i32::from_le_bytes(lea[3..7].try_into().unwrap()) as isize as usize), 0x15b4aa0);
        assert_eq!(exe.at(0x200e02b, 13), [0xBA, 1, 0, 0, 0, 0x45, 0x31, 0xC9, 0xC6, 0x44, 0x24, 0x20, 0x00], "mov edx, 1; xor r9d, r9d; mov byte [rsp+20h], 0");
        assert_eq!((exe.at(0x200e038, 1)[0], exe.rel32_target(0x200e038)), (0xE8, 0x4ff02e0));
        assert_eq!(exe.at(0x200e03d, 6), [0x90, 0x48, 0x83, 0xC4, 0x28, 0xC3], "nop; add rsp, 28h; ret");
        assert_eq!((exe.at(0x15b4aa0, 1)[0], exe.rel32_target(0x15b4aa0)), (0xE9, 0x15b4ab0));
    }

    /// `notrail`: the phase vtable's slots are the ones read (view setup,
    /// the shared always-active stub, the registrar), the store patched is
    /// the view setup's last one before its epilogue, and the call removed is
    /// the registrar's only call, queueing stage 44 with the trail callback.
    #[test]
    fn notrail_keeps_the_phase_out_of_the_scan_and_stage_44_out_of_the_queue() {
        let ps: Vec<&Patch> = PATCHES.iter().filter(|p| p.name == "notrail").collect();
        assert_eq!(ps.iter().map(|p| p.rva).collect::<Vec<_>>(), [0xcefde0, 0x200f6d8]);
        assert_eq!(ps[1].new, [0x0F, 0x1F, 0x44, 0x00, 0x00], "one 5-byte nop");
        let Some(exe) = Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): notrail was not checked against it");
            return;
        };
        // Vtable 0x1463DEA90: +0x58 the view setup, +0x60 `mov al, 1; ret`, +0x68 the registrar (file values, base 0x140000000).
        let slot = |at: usize| u64::from_le_bytes(exe.at(0x63dea90 + at, 8).try_into().unwrap()) as usize - 0x1_4000_0000;
        assert_eq!((slot(0x58), slot(0x60), slot(0x68)), (0xcefc50, 0x4870, 0x200f6c0));
        assert_eq!(exe.at(0x4870, 3), [0xB0, 0x01, 0xC3]);
        // The view setup ends: mov byte [rsi+0x750], 0xFF; the patched store; add rsp, 0xD8; pops; ret.
        assert_eq!(exe.at(0xcefdd9, 7), [0xC6, 0x86, 0x50, 0x07, 0x00, 0x00, 0xFF]);
        assert_eq!(exe.at(0xcefde7, 7), [0x48, 0x81, 0xC4, 0xD8, 0x00, 0x00, 0x00]);
        assert_eq!(exe.at(0xcefdf3, 1), [0xC3]);
        // The registrar: sub rsp, 0x28; lea r8, callback; mov edx, 44; xor r9d, r9d; mov byte [rsp+0x20], 0; the call; nop; add rsp, 0x28; ret.
        assert_eq!(exe.at(0x200f6c0, 7), [0x48, 0x83, 0xEC, 0x28, 0x4C, 0x8D, 0x05]);
        assert_eq!((0x200f6c4 + 7usize).wrapping_add(i32::from_le_bytes(exe.at(0x200f6c7, 4).try_into().unwrap()) as isize as usize), 0x280000, "the stage-44 callback thunk");
        assert_eq!(exe.at(0x200f6cb, 5), [0xBA, 44, 0, 0, 0]);
        assert_eq!((exe.at(0x200f6d8, 1)[0], exe.rel32_target(0x200f6d8)), (0xE8, 0x4ff02e0), "drawlist_queue_stage");
        assert_eq!(exe.at(0x200f6dd, 6), [0x90, 0x48, 0x83, 0xC4, 0x28, 0xC3]);
        // The callback thunk chain ends in the body whose closure was read.
        assert_eq!(exe.rel32_target(0x280000), 0x280010);
        assert_eq!(exe.rel32_target(0x280010), 0x32bab40);
    }

    /// `rtfallback` removes the one branch to the structure state test, so
    /// the set test falls into its own "absent" answer; both wrappers and the
    /// indirect-diffuse trace dispatch (Fallback = pass 42, BVH = 37 + ..)
    /// are the code ENG7 read.
    #[test]
    fn rtfallback_makes_the_set_test_answer_absent() {
        let p = PATCHES.iter().find(|p| p.name == "rtfallback").unwrap();
        assert_eq!((p.rva, p.new), (0x4fd857b, &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00][..]), "one 6-byte nop");
        assert_eq!(p.rva + 6 + u32::from_le_bytes(p.expect[2..6].try_into().unwrap()) as usize, 0x4fd8590, "the removed jne went to the state test");
        let Some(exe) = Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): rtfallback was not checked against it");
            return;
        };
        let rip = |at: usize, len: usize| (at + len).wrapping_add(i32::from_le_bytes(exe.at(at + len - 4, 4).try_into().unwrap()) as isize as usize);
        // The set test: mov rax, [rcx+8]; mov edx, edx; cmp qword [rax+rdx*8], 0; the patched jne; xor eax, eax; ret.
        assert_eq!(exe.at(0x4fd8570, 11), [0x48, 0x8B, 0x41, 0x08, 0x89, 0xD2, 0x48, 0x83, 0x3C, 0xD0, 0x00]);
        assert_eq!(exe.at(0x4fd8581, 3), [0x31, 0xC0, 0xC3], "what the nop falls into");
        // The state test it no longer reaches reads the manager's two counts and its object's count.
        assert_eq!(exe.at(0x4fd8590, 7), [0x83, 0xB9, 0xAC, 0x00, 0x00, 0x00, 0x00]);
        assert_eq!(exe.at(0x4fd859a, 7), [0x83, 0xB9, 0x9C, 0x00, 0x00, 0x00, 0x00]);
        // The lighting wrapper: mov edx, ecx; mov rcx, [qword_148527DC8]; test rcx, rcx; jne the set test; xor eax, eax; ret.
        assert_eq!(exe.at(0x4fd8550, 5), [0x89, 0xCA, 0x48, 0x8B, 0x0D]);
        assert_eq!(rip(0x4fd8552, 7), 0x8527dc8);
        assert_eq!((exe.at(0x4fd855c, 2), rip(0x4fd855c, 6)), (&[0x0F, 0x85][..], 0x4fd8570));
        // The shadow wrapper loads qword_1484CEA58 and calls the same set test.
        assert_eq!(rip(0x49be14, 7), 0x84cea58);
        assert_eq!((exe.at(0x49be22, 1)[0], exe.rel32_target(0x49be22)), (0xE8, 0x4fd8570));
        // The trace dispatch asks the lighting wrapper, then: mov r15d, 42 (Fallback); test bpl, 1; jne past the
        // BVH choice `lea r12d, [rax+rax+37]`.
        assert_eq!((exe.at(0x356b770, 1)[0], exe.rel32_target(0x356b770)), (0xE8, 0x4fd8550));
        assert_eq!(exe.at(0x356ba53, 12), [0x41, 0xBF, 0x2A, 0x00, 0x00, 0x00, 0x40, 0xF6, 0xC5, 0x01, 0x75, 0x30]);
        assert_eq!(exe.at(0x356ba73, 5), [0x44, 0x8D, 0x64, 0x00, 0x25]);
    }

    /// `noprobecap` sends every pending capture record to the title's own
    /// no-target exit: the new jump lands where the `je` did, that exit
    /// stores the done mark and the frame number in the record (r13) and
    /// continues with the next record, and the patched test comes straight
    /// after the record's own done test.
    #[test]
    fn noprobecap_marks_each_record_done_and_draws_no_face() {
        let p = PATCHES.iter().find(|p| p.name == "noprobecap").unwrap();
        let to = |bytes: &[u8], len: usize| (p.rva + len).wrapping_add(i32::from_le_bytes(bytes[len - 4..len].try_into().unwrap()) as isize as usize);
        assert_eq!((to(p.expect, 6), to(&p.new[..5], 5), p.new[5]), (0x1529c32, 0x1529c32, 0x90), "the jump goes where the je went");
        let Some(exe) = Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): noprobecap was not checked against it");
            return;
        };
        // The stage's capture callback reaches this body: thunk -> owner test -> body.
        assert_eq!((exe.at(0x4413b0, 1)[0], exe.rel32_target(0x4413b0)), (0xE9, 0x4413c0));
        // The record loop's head: cmp byte [r13+0xd8], 0; jne next; cmp qword [r13+0xe0], 0; the patched je.
        assert_eq!(exe.at(0x1529ac4, 8), [0x41, 0x80, 0xBD, 0xD8, 0x00, 0x00, 0x00, 0x00]);
        assert_eq!(exe.at(0x1529acc, 2), [0x0F, 0x85], "done records are skipped first");
        assert_eq!(exe.at(0x1529ad2, 8), [0x49, 0x83, 0xBD, 0xE0, 0x00, 0x00, 0x00, 0x00]);
        // The exit: mov byte [r13+0xd8], 1; mov eax, [rip+..] (dword_14A9437AC); mov [r13+0xd0], eax; jmp next record.
        assert_eq!(exe.at(0x1529c32, 8), [0x41, 0xC6, 0x85, 0xD8, 0x00, 0x00, 0x00, 0x01]);
        let frame = exe.at(0x1529c3a, 6);
        assert_eq!(frame[..2], [0x8B, 0x05]);
        assert_eq!((0x1529c3a + 6usize).wrapping_add(i32::from_le_bytes(frame[2..6].try_into().unwrap()) as isize as usize), 0xa9437ac, "the frame number");
        assert_eq!(exe.at(0x1529c40, 7), [0x41, 0x89, 0x85, 0xD0, 0x00, 0x00, 0x00]);
        assert_eq!((exe.at(0x1529c47, 1)[0], exe.rel32_target(0x1529c47)), (0xE9, 0x152bdab));
        // Next record: add r13, 0x110; cmp r13, r8; jne loop head.
        assert_eq!(exe.at(0x152bdab, 7), [0x49, 0x81, 0xC5, 0x10, 0x01, 0x00, 0x00]);
        assert_eq!(exe.at(0x152bdb2, 3), [0x4D, 0x39, 0xC5]);
        // The size gate that also leads to a done mark without faces (the title's third exit) is still there.
        assert_eq!(exe.at(0x1529b9c, 2), [0x0F, 0x87]);
    }

    /// `noprepass` keeps the stage's head and takes the title's own branch
    /// to its pass end: the four head calls come before the patched test,
    /// the new jump lands where the `je` did, and that is the pass end.
    #[test]
    fn noprepass_skips_the_draws_and_keeps_the_ped_pre_skin() {
        let p = PATCHES.iter().find(|p| p.name == "noprepass").unwrap();
        let target = |bytes: &[u8], at: usize, len: usize| (p.rva + at + len).wrapping_add(i32::from_le_bytes(bytes[at + len - 4..at + len].try_into().unwrap()) as isize as usize);
        assert_eq!((p.expect[2], p.expect[3], p.new[2], p.new[7]), (0x0F, 0x84, 0xE9, 0x90));
        assert_eq!(target(p.expect, 2, 6), 0x13548ac);
        assert_eq!(target(p.new, 2, 5), 0x13548ac, "the jump goes where the je went");
        let Some(exe) = Exe::open() else {
            eprintln!("no title executable here (XNW_TITLE_EXE): noprepass was not checked against it");
            return;
        };
        // The thunk of the stage's callback still enters the body.
        assert_eq!((exe.at(0x1353d40, 1)[0], exe.rel32_target(0x1353d40)), (0xE9, 0x1353d50));
        // The head: four calls in a row, the third the ped pre-skin batches, all before the patched test.
        for (site, to) in [(0x1353da0, 0xfc12b0), (0x1353da5, 0xa28030), (0x1353daa, 0x27c2190), (0x1353daf, 0x8cf5d0)] {
            assert!(site < p.rva);
            assert_eq!((exe.at(site, 1)[0], exe.rel32_target(site)), (0xE8, to), "title+{site:#x}");
        }
        // The pre-skin runner loads the queue that ped_skinning_cs owns (mov rcx, [rip+..] -> 0x78a9a28 somewhere in it).
        let runner = exe.at(0x27c2190, 219);
        assert!((0..runner.len() - 7).any(|i| runner[i + 1] == 0x8B && (0x27c2190 + i + 7).wrapping_add(i32::from_le_bytes(runner[i + 3..i + 7].try_into().unwrap()) as isize as usize) == 0x78a9a28), "no load of qword_1478A9A28");
        // The branch target is the pass end, then the epilogue.
        assert_eq!((exe.at(0x13548ac, 1)[0], exe.rel32_target(0x13548ac)), (0xE8, 0x5003da0));
    }
}
