# Conventions

Simple, efficient, readable code in modern Rust. The rules follow a modern C++ (C++20 and later) house style and
state the Rust form of each rule; where a C++ habit and Rust idiom disagree, Rust idiom wins and the rule says so.
They describe what the code already does.

## Language and dependencies
- Rust 2024 edition, toolchain pinned in `rust-toolchain.toml`. Standard library first.
- Plain Rust: no `async`, no procedural macros of our own, no trait hierarchies. A table beats an abstraction.
- A new external crate needs a written reason in its commit. Generics are for small helpers, as templates are in C++.

## Layout
- A Cargo workspace: libraries in `crates/<name>/`, programs in `apps/<name>/`, names prefixed `xnw-`.
- One module per concern; the file name is the module name (`snake_case.rs`). C++ puts one type per PascalCase
  file; Rust idiom wins: a module may hold the few types that belong together.
- Every file opens with a `//!` header: what the module does, which switch turns it on, and for ported code the
  origin (project, file, commit) and licence.
- `use` order: `std`, then external crates, then workspace crates, then `crate::`/`super::`; groups separated by a
  blank line.

## Formatting
- 4 spaces, no tabs, LF line endings, UTF-8. Aim for 140 columns; tables and long format strings may run past.
- Braces follow Rust style (opening brace on the same line). The C++ style uses Allman braces; Rust idiom wins.
- Short guards on one line: `let Ok(x) = ... else { return out };`, `if n == 0 { continue; }`.
- Struct literals and arrays one entry per line once they no longer fit on one.

## Naming
- Types, traits and enum variants `UpperCamelCase` (C++ PascalCase). Functions, variables, fields and modules
  `snake_case`; C++ uses camelCase and `m_`/`g_` prefixes, Rust idiom wins: no prefixes, no Hungarian notation.
- Constants and statics `SCREAMING_SNAKE_CASE`.
- Foreign functions and structures keep their system names exactly (`GetThreadTimes`, `CONTEXT`), with
  `#[allow(non_snake_case)]` where needed, as C++ keeps a foreign ABI's PascalCase.
- Environment switches are `XNW_<NAME>`; list values are comma-separated (`XNW_TRACE=perf,file`).
- Test names are sentences: `a_record_survives_its_line`, `counters_add_and_peak`.

## Idioms
- Borrow for read-only input: `&[u8]`, `&str`, `&[T]` (C++ `std::span`, `std::string_view`).
- `Option` and `Result` instead of sentinel values and out-parameters (C++ `std::optional`, `std::expected`).
- Fixed tables are `const`/`static` arrays of small records consumed by iteration; an enum used as an index is
  `#[derive(Clone, Copy)]` and indexes with `as usize`, with a test that the table and the enum agree.
- Lazy globals with `OnceLock` (the C++ function-local static); counters are atomics with `Relaxed` ordering.
- Guard clauses and early returns over nesting; `let ... else` for "nothing to do".
- Ownership instead of manual release: a handle that must be closed is owned by a value whose `Drop` closes it.

## Errors
- A library returns `Result` with its own error enum implementing `Display` and `std::error::Error` (`PeError`).
  Programs report at the top: `Result<_, String>` or a boxed error, printed once.
- No panics on input the title or a file controls; check bounds with `get` and return an error or a default.
  `unwrap` only where the invariant is local and visible (a slice whose length was just checked).
- Nothing unwinds across a foreign boundary: an `extern "system"` function handles its own errors.
- Check the result of every system call; log a failure once with the call, the value and what it was for.

## Logging
- `xnw_core::info!`, `warn!` and `trace!` take `format!` arguments. A line is the time since start, a tag and the
  text, to stderr and the run's log file.
- A message starts with its subsystem (`perf: ...`, `host buffer creation failed ...`) and carries the numbers
  someone needs to act on it.
- Logging never stalls a hot path: per-call tracing is off unless `XNW_TRACE` names its area; repeated warnings are
  logged for the first few occurrences and counted after that.

## `unsafe` and foreign functions
- `unsafe_op_in_unsafe_fn` is denied workspace-wide. Every `unsafe` block carries a `// SAFETY:` comment that
  says why it holds.
- Win32 is declared by hand in one module (`xnw_core::win`): only what is used, `#[repr(C)]` structures with their
  size noted. Exports missing from some import libraries are looked up once at run time.
- Wide (`...W`) APIs for paths; narrow ones only for environment variables and logging.

## Comments
- Say why, and what the code relies on: units, sizes, ranges, the condition a branch handles. `///` on public
  items, `//!` for a module; no parameter-list boilerplate.
- A comment may cite public specifications and API documentation, an upstream project's file and commit where code
  was ported (keep every notice its licence requires), and measurements with what was measured.
- Addresses, structure layouts and file formats of the game are interoperability facts: a comment may cite them
  (`title+0x1000`, an offset, a field size) where the code depends on them.
- A comment must not quote bytes or code taken from a game, decompiler or disassembler output, console SDK text,
  private notes or tickets, personal paths, names or machine names.

## Tests
- Unit tests live next to the code in `#[cfg(test)] mod tests`. They run without game data, a GPU or a network.

## Commits
- Subject `<Area>: <what it does>`, for example `Pe: read delay-load imports`; body in plain sentences says why and
  what was measured. No trailers.
