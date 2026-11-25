# Import replay

`import_replay.rs` checks our import shims against a recording of the
title's own calls. The recording is game-derived: it lives in `out/`
(ignored) and is never committed; without it the test passes with a
"skipped" line.

Refresh it (one smoke run, about 20 s, null GPU on WARP):

    cargo build
    target\debug\xnw.exe smoke --record            # writes out\imports.tsv (until 18 s)
    cargo test -p xnw-hle --test import_replay -- --nocapture

`--record FILE,until=S` picks another file or end; `XNW_IMPORT_TRACE=FILE[,until=S]`
records in any run. The test looks for `out\imports.tsv` in this checkout,
then in the main checkout's `out\` (from a worktree), or takes
`XNW_IMPORT_RECORDING=<file>`. Re-record after adding a shim, so its calls
are in the recording; a recording made with a buggy shim still shows the
bug through the invariants and the shim-versus-host replay, not through
the plain replay.

Format and the pure-call table: `crates/xnw-core/src/imprec.rs`.
Recorder: `crates/xnw-loader/src/record.rs`.
