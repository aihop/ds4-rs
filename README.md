# ds4-rust Preview

This directory contains a **minimal Rust rewrite skeleton** for `ds4`.

It is **not** a full engine port yet. The goal of this first cut is to mirror
the current C project boundaries with a small, compilable Rust layout so the
next iterations can replace stubs with real inference, KV, checkpoint, and HTTP
logic without redesigning the crate from scratch.

## Current scope

- `src/lib.rs`: public crate surface mirroring the C-style engine/session/server split.
- `src/engine.rs`: minimal engine options, backend selection, summary, and
  GGUF-aware tokenizer wiring.
- `src/gguf.rs`: minimal GGUF v3 header/metadata loader used by the Rust engine.
- `src/tokenizer.rs`: GGUF-backed JoyAI/GPT-2 style byte-level BPE path with
  special-token aware rendered-chat tokenization.
- `src/session.rs`: mutable session timeline, prefix reuse, and a minimal
  DSV4-like snapshot payload carrying checkpoint tokens plus next-token logits.
- `src/kv.rs`: minimal KV/checkpoint store abstraction.
- `src/kv.rs`: in-memory KV/checkpoint store plus a minimal optional disk-backed
  cache layer for persisted continuation snapshots. The current disk file now
  uses a KVC-like outer header plus rendered-text section wrapped around the
  Rust snapshot payload, and can optionally carry a KTM-like tool replay
  section keyed by `tool_call_id`. Disk filenames now follow the C-side rule:
  `sha1(rendered_text).kv`.
- `src/protocol.rs`: tiny request/response protocol model for chat, responses, completions, and Anthropic-style messages APIs.
- `src/continuation.rs`: minimal continuation index for `previous_response_id` and conversation alias restore.
- `src/server.rs`: tiny HTTP preview for `/health`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, and `/v1/messages`.
- `src/bin/ds4.rs`: preview CLI with one-shot prompt mode and a small interactive loop.
- `src/bin/ds4-server.rs`: preview server entry point.
- `tests/test-vectors/`: official DeepSeek V4 Flash prompt/logprob fixtures copied
  from the C tree, plus Rust-side integration coverage for manifest/request sanity.

## What this proves

- A Rust port can preserve the same top-level concepts:
  - `Engine`
  - `Session`
  - KV/cache layer
  - CLI entry
  - HTTP server entry
  - continuation-oriented session sync
- The rewrite can start from the public boundary first instead of translating
  tensor internals immediately.
- The staged path is now beyond a pure shell:
  - GGUF metadata and tensor directory loading are real
  - tokenizer special tokens and BPE merges are real
  - `token_embd.weight` / `output_norm.weight` / `output.weight` can now drive a
    minimal CPU reference logits path when tensor layouts are simple enough
  - `/v1/chat/completions` now accepts a minimal `messages` array shape and can
    answer with standard `choices[].message` or SSE `delta` chunks
  - `/v1/completions` now accepts legacy prompt-style requests and returns
    OpenAI-style text completion JSON or SSE chunks
  - `/v1/messages` now accepts a minimal Anthropic-style `messages` payload,
    including `tool_use` / `tool_result` blocks, and can answer with Anthropic
    message JSON or SSE lifecycle events
  - the official `tests/test-vectors` fixture set now lives in the Rust tree too,
    and an integration test verifies manifest/file wiring plus request/message
    consistency against the recorded official JSON
  - the preview server can now optionally persist KV entries to disk with
    `--kv-disk-dir`, so visible-prefix snapshots survive process restarts even
    though the Rust payload format is still a small Rust-native skeleton
  - snapshot bytes now carry a minimal DSV4-like payload header and logits, and
    disk KV files now use a KVC-like fixed header with rendered text before the
    payload so the outer/inner serialization boundary is closer to the C design
  - KVC files can now also embed a minimal KTM-like tool replay section, and
    `KvCache` can look up a stored exact tool block by `tool_call_id` after a
    restart
  - disk identity now uses the SHA1 of the rendered prefix text instead of the
    earlier Rust-only FNV key, so on-disk naming is closer to the C cache model
  - tool-result follow-up requests now hydrate exact tool replay blocks from
    KTM before render/sync, so restarted continuation paths no longer depend
    only on canonical JSON-to-prompt projection
  - KTM replay blocks are now stored as canonical DSML tool-call fragments
    (`<｜DSML｜tool_calls> ... </｜DSML｜tool_calls>`) instead of the earlier
    Rust-only `AssistantToolCall[...]` placeholder line
  - request prompt projection now renders tool outputs as
    `<tool_result>...</tool_result>` tails instead of `Tool[...]` placeholders,
    with minimal closing-tag escaping to stay closer to the C-side live suffix
  - Anthropic `tool_use` streaming now emits chunked `input_json_delta`
    `content_block_delta` events rather than sending the whole input JSON as a
    single block, so live tool-call SSE is closer to the C-side behavior
  - the preview server now exposes the stable model id `deepseek-v4-flash`
    across `/v1/models` and chat-compatible responses
  - implicit continuation can now recover not only from explicit response ids,
    but also from session-style headers and repeated `tool_call_id` tails
  - when chat tools are provided, the preview server can now emit a minimal
    assistant `tool_calls` response before the matching tool result arrives,
    which keeps OpenAI-compatible agent clients usable while structured tool
    decoding is still incomplete
  - CLI and preview server both return generated tokens through the same session
    sync plus argmax loop instead of a fixed placeholder string

## What is still missing

- Full DS4 weight binding and parity inference
- Metal/CUDA/CPU execution backends
- Disk KV payload format compatibility
- Full KTM/tool replay parity with the C server
- Continuation fidelity matching the C server
- Streaming, tool calls, and protocol-complete compatibility

## Current load path

- If the configured model file exists, `Engine::open()` now tries to parse the
  GGUF v3 header and metadata, and load tokenizer token/merge tables plus the
  core special token ids from the model.
- The GGUF loader now also parses the tensor directory, computes aligned
  `tensor_data_pos`, records tensor byte sizes and absolute offsets, and exposes
  tensor lookup by name through the engine/model view.
- The Rust tokenizer now includes a minimal real JoyAI pre-tokenizer and
  GPT-2 byte-level BPE merge path, with focused unit tests covering merge
  behavior and rendered-chat special token handling.
- The engine now binds a small output head view and first tries a tensor-backed
  CPU reference path:
  - embed the last token from `token_embd.weight`
  - optionally apply `output_norm.weight`
  - project with `output.weight`
- The current tensor-backed path only covers straightforward `f32` / `f16` /
  `bf16` rows. Unsupported layouts or missing model files still fall back to the
  deterministic preview logits path so the project remains runnable while the
  real backend is incomplete.
- The preview HTTP path now reads full request bodies using `Content-Length`
  instead of assuming a single small socket read, which makes larger chat-style
  payloads much less fragile during local integration.
- If no GGUF file is present locally, the binaries still fall back to preview
  mode so the Rust project remains runnable during staged development.

## Expected usage

Once a Rust toolchain is installed:

```bash
cargo run --manifest-path rust/Cargo.toml --bin ds4 -- -p "hello"
cargo run --manifest-path rust/Cargo.toml --bin ds4-server -- --port 8080
```

This preview is intentionally small: it is the starting point for a staged
rewrite, not a claim of feature parity.
