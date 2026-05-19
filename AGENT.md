# Rust Rewrite Notes

This directory contains the staged Rust rewrite of `ds4`.

The current implementation is a **preview skeleton**, not a feature-parity port.
Keep the project honest about that status: preserve clean boundaries first, then
replace stubs with real model loading, tokenization, inference, KV, and server
logic in small correctness-oriented steps.

## Goals

- Mirror the high-level C architecture with Rust-native modules:
  - `Engine`
  - `Session`
  - KV/checkpoint layer
  - protocol/request layer
  - continuation state
  - CLI binary
  - server binary
- Keep the public API narrow so CLI and server code do not depend on tensor or
  backend internals.
- Make continuation, checkpoint, and long-session reuse first-class concepts
  from the start instead of retrofitting them later.
- Prefer a staged rewrite that becomes progressively real over a large placeholder
  code dump with unclear ownership.

## Current Layout

- `src/lib.rs`: crate surface and public re-exports.
- `src/model.rs`: shared DS4 model-shape constants used by the reference path so
  kernel-style modules do not hard-code architecture values independently.
- `src/weights.rs`: GGUF tensor binding views (`BoundTensor`,
  `BoundWeights`) plus lightweight checksum helpers used by the engine to
  decide whether the reference path is trustworthy. The binding map now scans
  all `blk.{i}` tensors into ordered `blocks[]`, with per-block attention
  tensors bound eagerly and FFN/MoE tensor names reserved in optional block
  bindings so the pure-Rust CPU reference path can grow layer-by-layer instead
  of staying hard-coded to `blk.0`.
- `src/engine.rs`: engine options, backend enum, summary helpers, and minimal
  GGUF-aware open path plus tokenizer wiring. The current real inference slice
  binds `token_embd.weight`, `output_norm.weight`, and `output.weight`, then
  runs a minimal CPU reference logits path for simple `f32` / `f16` / `bf16`
  rows plus `Q8_0` rows before falling back to deterministic preview logits.
  Large output heads now default back to preview logits unless `quality=true`,
  which keeps giant GGUF models responsive until a real backend exists. The
  engine also exposes whether model-backed generation is trustworthy so the
  server can choose stable text fallbacks instead of streaming gibberish.
- `src/gguf.rs`: minimal GGUF v3 header/metadata parser used by the engine.
  Model bytes are now mmap-backed instead of eagerly copied into a single
  `Vec<u8>`, bringing startup behavior closer to the C implementation.
- `src/tokenizer.rs`: JoyAI-style pre-tokenizer plus GPT-2 byte-level BPE merge
  path, with special-token aware rendered chat tokenization.
- `src/kernels/`: focused reference-kernel modules. `kernels/attention.rs`
  now owns the multi-block CPU reference attention forward used by
  `Engine::infer_logits()`: it iterates all bound blocks, keeps a per-layer KV
  cache structure, and can now switch between two paths. The plain fallback
  path keeps the older residual-only flow, while the trustworthy DS4 path
  wires the minimal Hyper-connection loop:
  `plain embedding -> HC state -> attn HC pre/post -> ffn HC pre/post ->
  output HC head -> output_norm/output`. The HC math itself lives in the new
  `kernels/hc.rs` helper module so Sinkhorn split, HC post mixing, and final
  HC collapse stay out of `engine.rs`.
  The attention output projection should also stay structurally close to the C
  path: when `attn_output_a` is `Q8_0`, prefer the grouped-rows helper instead
  of looping over output groups and issuing one matvec per group.
  For single-token decode, other `Q8_0` attention projections such as
  `attn_q_b` and `attn_output_b` should likewise prefer a decode-scratch style
  flow: quantize the activation once, then reuse the prequantized path instead
  of calling the generic dense matvec helper and redoing equivalent hot work.
  The reusable decode intermediates now live in `kernels/decode_scratch.rs`.
  Keep `Session` and the reference decode path on a single scratch instance per
  request so hot loops can reuse buffers like shared FFN gate/up/mid/output and
  grouped attention low-rank activations instead of allocating new `Vec`s per
  layer and token.
  Continue extending that scratch-first rule to attention intermediates such as
  `attn_norm`, `qr`, `qr_norm`, and `kv_raw`; match the C decode flow where
  these buffers are long-lived decode temporaries rather than fresh allocations.
  For HC control flow, prefer fixed-size arrays for the 4-way post/comb/split
  vectors instead of heap-backed `Vec`s, mirroring the C path's stack-shaped
  `mix[24]` / `split[24]` control math.
  In `kernels/matmul.rs`, `*_into` helpers on the hot decode path should write
  directly into caller-provided output slices where possible. Avoid per-thread
  temporary chunk `Vec`s in scoped worker loops when a disjoint output chunk can
  be borrowed and filled in place instead.
  Single-token routed MoE should follow the C `layer_routed_moe_one_prealloc`
  shape: reuse decode scratch for `routed_xq`, selected-expert `mid_all`,
  quantized `midq`, and per-expert down outputs, rather than allocating one
  fresh `Vec` per expert for mid/out on every token.
  `kernels/matmul.rs` holds tensor row decode / dot-product helpers and
  `kernels/norm.rs` owns RMSNorm reference math.
- `src/kernels/ffn.rs`: routed expert work is still the dominant CPU hotspot.
  Keep the reference path allocation-light: reuse quantized tensor accessors,
  derive expert row bases once per expert, and write SwigLU output directly
  into the intermediate `mid` buffer instead of materializing separate
  `gate`/`up` vectors first.
- `src/weights.rs`: besides `blocks[]`, binding now also records optional
  HC tensors (`blk.{i}.hc_attn_*`, `blk.{i}.hc_ffn_*`, `output_hc_*`) and
  caches several frequently reused 1D tensors such as attention/FFN norms and
  sinks. The reference path should prefer these cached decoded vectors over
  re-decoding the same 1D tensors on every token.
- `src/session.rs`: mutable session timeline, prefix reuse, explicit
  `prefill`/`decode_next` skeleton methods, minimal DSV4-like snapshot payload,
  and prefill-boundary
  hook. `sync`/`eval` remain compatibility wrappers for CLI and server code.
  The session now tries to keep a live `TransformerKvCache` on the trustworthy
  reference path so append-only prefill and decode can advance token-by-token
  instead of recomputing the full prefix every step; rewind/invalidate/snapshot
  restore still clear that forward-state cache and fall back to rebuild.
- `src/kv.rs`: in-memory checkpoint store plus a separate
  `TransformerKvCache`/`TransformerKvLayer` structure for pure-Rust reference
  decode work. Keep these two responsibilities separate: continuation cache is
  request/session state, transformer KV is model forward state. The preview
  KV layer now also has a KVC-like outer disk header around persisted
  `KvEntry` snapshots behind optional `--kv-disk-dir` wiring. The embedded
  payload is a minimal DSV4-like Rust snapshot (tokens + logits) and remains a
  restart-friendly skeleton, not full C-compatible graph-state parity. KVC
  files may also carry a minimal KTM-like section for exact Rust-side tool
  replay blocks keyed by `tool_call_id`. On disk, file identity now follows the
  C model and uses `sha1(rendered_text).kv`; keep that rendered-text boundary
  stable when evolving the cache format.
- `src/protocol.rs`: tiny request/response protocol model for chat/responses
  style APIs. The current chat path already accepts a minimal OpenAI-like
  `messages` array, parses assistant `tool_calls` plus tool-result
  `tool_call_id`. The responses path also accepts minimal `input` item arrays
  (`message`, `function_call`, `function_call_output`) so tool-result
  continuations can stay on the real model path. Both APIs can emit either
  standard assistant messages or minimal tool-call responses, including SSE.
  The preview protocol layer also now covers legacy `/v1/completions` prompt
  requests plus Anthropic-style `/v1/messages` payloads with minimal
  `tool_use` / `tool_result` parsing and rendering.
- `src/continuation.rs`: minimal continuation store keyed by response id and
  latest conversation alias. It also keeps a lightweight `tool_call_id` alias
  so bare tool-result tails can still hit the latest matching continuation.
- `src/server.rs`: small HTTP preview that wires protocol parsing, session sync,
  KV store, and continuation remember/restore. Request reads now honor
  `Content-Length` instead of assuming one short socket read. When chat tools
  are present and no tool result has arrived yet, the preview server now always
  short-circuits to a minimal assistant tool call until model-emitted
  structured tool calls exist; this keeps OpenAI-compatible agent clients usable
  even when GGUF-backed generation is still partial. `/v1/models` and chat SSE
  now use the stable model id `deepseek-v4-flash` so local client discovery
  matches request/response payloads. The same preview server now also exposes
  `/v1/completions` and `/v1/messages`, reusing the existing continuation,
  session sync, and preview/model reply pipeline across OpenAI legacy prompt
  clients and Anthropic-style message clients. Tool-result continuations still generate
  through the model when trustworthy logits are available. The server now
  fails closed for non-tool assistant replies: if no GGUF model is loaded or
  the current inference path is not trustworthy for that model, it returns a
  structured `503` JSON error instead of streaming preview/garbled text. The
  server now also keeps serving after per-request failures and supports
  `DS4_TRACE=1` / `DS4_TRACE_HTTP=1` request-stage logging for local debugging.
- `src/bin/ds4-server.rs` and `src/bin/ds4.rs`: both CLIs now expose
  `--quality`. Large output heads no longer require that flag just to unlock
  reference logits; the remaining trust gate is mainly about missing bound
  blocks / FFN coverage rather than output-head size alone.
- `tests/test-vectors/`: Rust now carries the same official DeepSeek V4 Flash
  prompt/logprob fixture set as the C tree. Treat these files as parity and
  regression assets: integration tests should at least validate manifest wiring,
  prompt/request consistency, and step-token reconstruction even before Rust
  reaches true logprob parity.
- `src/engine.rs`: default model discovery now prefers a colocated
  `ds4flash.gguf` next to the working directory or binary before falling back to
  the compile-time crate path, which avoids stale preview-only startup after the
  tree is moved.
- `src/bin/ds4.rs`: preview CLI.
- `src/bin/ds4-server.rs`: preview server entry.

## Rewrite Order

- Stabilize public data models before implementing real inference internals.
- Keep tokenizer work on a real path: load special tokens from GGUF metadata,
  preserve JoyAI split shape, and verify merge behavior with focused unit tests.
- Prefer "real load path with preview fallback" over blocking the whole rewrite
  on having a local GGUF file present during every iteration.
- When extending inference, keep the fallback boundary explicit: if a tensor
  type or layout is not supported yet, return to the preview logits path rather
  than pretending parity.
- Add real request parsing and response projection before adding streaming.
- Implement snapshot/payload compatibility before promising continuation parity.
- The temporary Rust disk-KV format is allowed to stay simple while the C
  payload/header compatibility work is still pending, but new code should keep
  the serialization boundary explicit so the format can later be swapped for a
  true DS4-compatible payload.
- Rust-side KTM entries should store exact replay blocks in the current prompt
  canonicalization format, so future continuation work can reuse them before
  full DSML parity lands.
- The current Rust-side "exact replay" format should prefer canonical DSML
  fragments (`<｜DSML｜tool_calls>...`) over placeholder
  `AssistantToolCall[...]` lines, because restart hydration should already look
  like the model-facing protocol even before live sampled-token parity exists.
- Tool-result requests should hydrate KTM replay blocks before render/sync when
  a matching `tool_call_id` exists, so restarted flows preserve the sampled
  tool-call line instead of trusting canonical JSON projection alone.
- Prompt projection for tool outputs should prefer `<tool_result>...</tool_result>`
  tails over `Tool[...]` placeholder lines, and replay hydration should accept
  both so older cached prompts keep working during the transition.
- Anthropic SSE tool-use paths should stream `input_json_delta` in chunks,
  mirroring the live incremental shape more closely than a single full-JSON
  delta block.
- Add one backend path end to end before branching into Metal/CUDA/CPU variants.

## Rules

- Do not claim parity in docs or code comments unless a path is actually wired.
- Prefer explicit structs and enums over loose stringly typed plumbing.
- Keep continuation and KV state serializable and inspectable.
- Avoid introducing async/runtime complexity before the protocol and session
  model are stable.
- Keep the first real inference path small and inspectable: embedding lookup,
  optional norm, vocab projection, then sampling.
- Prefer moving reference math into `src/kernels/` and architecture constants
  into `src/model.rs` before growing `engine.rs` further.
- Keep GGUF binding metadata in `src/weights.rs`; avoid re-mixing tensor
  binding, tensor decoding, and higher-level math inside one file again.
- Keep comments compact and explain why a boundary exists, not the obvious.
- Split and encapsulate files before they grow beyond 400 lines. Prefer moving
  cohesive helpers, structs, tests, or protocol branches into focused modules
  instead of letting one file accumulate multiple responsibilities.
- When a file is nearing 400 lines, treat modularization as the default path
  for the next meaningful change unless there is a strong reason to keep the
  code together temporarily.
- For core modules like protocol, server, session, and engine, default to a
  directory module layout (`mod.rs` or a thin root plus submodules) once the
  file starts mixing data models, parsing, rendering, helpers, and tests.

## Validation

- Run `cargo check` and then `cargo build` once a Rust toolchain is available.
- Prefer small focused tests around session sync, continuation restore, and
  protocol parsing before backend-heavy work.
- When touching official-vector fixtures, prefer targeted integration tests in
  `tests/` so they can run independently even if some unit-test-only preview
  modules are temporarily broken elsewhere in the crate.
- When adding real inference code, validate correctness before speed claims.
- Keep focused unit tests around tensor binding, row decoding, and logits shape
  before moving on to larger decode or streaming features.
