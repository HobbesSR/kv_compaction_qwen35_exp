# Roo Proxy Experiment

## Goal

Run Roo Code against a local OpenAI-compatible endpoint that:

- serves `Qwen3.5-9B`
- keeps Roo's own auto-condense path disabled
- owns compaction policy in the proxy/model process
- preserves logical context length separately from physical compacted KV size

This experiment is scoped to the attention-bearing surface already validated in
the Qwen3.5 adaptation work:

- compact only the 8 full-attention layers
- leave SWA and recurrent state untouched
- treat segment compaction as the target service architecture

## Current Code Surface

Relevant modules:

- segment bundle storage and lineage:
  - `src/kv_compaction_qwen35_clean/segment_compaction_cache.py`
- OpenAI-style message canonicalization:
  - `src/kv_compaction_qwen35_clean/openai_chat_canonicalization.py`
- minimal OpenAI-compatible local proxy:
  - `src/kv_compaction_qwen35_clean/qwen35_openai_proxy.py`
  - `scripts/run_qwen35_openai_proxy.py`
- minimal local tool harness:
  - `src/kv_compaction_qwen35_clean/roo_lite_agent.py`
  - `scripts/run_roo_lite_agent.py`

The new canonicalization path exists to bridge:

- OpenAI chat message JSON
- exact model token IDs
- turn spans usable by segment caching

## Required Service Ownership

The proxy must own the model process or at least the tokenizer/chat template.

Why:

- cache identity is token-ID based
- token boundaries depend on the model chat template
- hidden scaffolding (role tags, generation prompt, tool formatting, thinking
  flags) is part of the true prompt

This design is not a good fit for fronting an opaque external server that does
its own prompt templating internally.

## Roo Configuration

For the local experiment:

- disable Roo's automatic intelligent context condensing
- point Roo at the local OpenAI-compatible Base URL
- keep the proxy ahead of true context-limit errors

The proxy should not rely on Roo to fail cleanly at the hard limit because Roo
has its own documented recovery behavior around context errors.

Because the local proxy owns a single live model instance, request handling
should be serialized unless and until there is explicit multi-request model
state isolation. The current baseline proxy uses a request lock around the
completion path.

## Request Flow

For each `/v1/chat/completions` request:

1. Receive OpenAI-format messages.
2. Canonicalize them with the exact model tokenizer/template.
3. Compute:
   - `message_token_ids`
   - `generation_token_ids`
   - `turn_spans`
4. Look up the longest cached segment prefix.
5. Reuse loaded segment bundles for the cached prefix.
6. Run live observation from the first uncached turn onward.
7. Compact newly completed segments at turn boundaries.
8. Return the assistant response in OpenAI format.

The current code supports steps 2-5:

- `canonicalize_openai_chat_messages(...)`
- `find_cached_prefix_for_openai_messages(...)`
- `resolve_uncached_segment_range(...)`

## Cache Identity

Cache keys are based on:

- `parent_hash`
- exact segment token IDs
- config fingerprint
- boundary metadata

This is intentionally conservative:

- same visible message text under different hidden scaffolding hashes
  differently
- same segment content in a different parent lineage hashes differently

## Segment Policy

Target architecture:

- compact segments between natural turn boundaries
- use a `min_segment_tokens` floor to avoid trivial bundles
- special-case the root/shared scaffold segment once reuse patterns are clear

This is preferred over replay-window narrowing because replay-window narrowing
changed the evidence policy and immediately degraded quality on the 4-prompt
surface.

## Counters The Proxy Must Track

The proxy should maintain and surface at least:

- logical token count
- physical retained KV slot count
- physical retained unique token count
- last compacted boundary turn index
- deepest cached segment hash
- active uncached segment token range for the next observer pass

Roo's own token display will reflect logical conversation size, not compacted
KV size. That is acceptable for the first experiment as long as the proxy keeps
its own authoritative counters.

## First Experiment

The first honest local experiment is:

1. Dense `Qwen3.5-9B`
2. OpenAI-compatible local proxy
3. Live observer only on the 8 full-attention layers
4. Segment compaction at turn boundaries
5. Roo pointed at the proxy with auto-condense disabled

Success criteria:

- no secondary teacher-forced evidence pass
- cached segment reuse across repeated or branched chats
- stable responses on the same smoke prompts already validated by the batch
  collector path
- honest reporting of logical vs physical context

## Live Endpoint Result

The first non-streaming local endpoint is now working on this box with dense
`Qwen3.5-9B`.

Run command:

```bash
cd qwen35_clean
env PYTHONPATH=src \
  python scripts/run_qwen35_openai_proxy.py \
  --config configs/qwen35_smoke/qwen3_5_9b.yaml \
  --cache-root artifacts/qwen35_proxy_cache \
  --host 127.0.0.1 \
  --port 8010
```

Validated endpoints:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/chat/completions` with `stream=true` via a buffered SSE shim

Observed live requests:

1. system + user
   - system: `Answer briefly.`
   - user: `Respond with exactly: OK`
   - response: `OK`
2. user only
   - user: `Respond with exactly: OK`
   - response: `OK`

Observed response telemetry:

- OpenAI-style `usage`
- proxy headers:
  - `X-KV-Logical-Tokens`
  - `X-KV-Cached-Segments`
  - `X-KV-First-Uncached-Turn`
  - `X-KV-Cached-KV-Slots`
  - `X-KV-Cached-Unique-Tokens`
- response payload field:
  - `kv_compaction.mode = "observe_only_baseline"`
  - `kv_compaction.uncached_segment_*`

Current operational meaning:

- Roo can point at the local proxy now
- the endpoint already tokenizes requests exactly the way the model sees them
- cached-prefix lookup is active
- simple recency-selected bundle writing is now active
- the endpoint can emulate OpenAI streaming by returning the completed response
  as a single SSE content chunk followed by the normal stop chunk and `[DONE]`
- live compaction is not yet active
- this is the correct staging point before adding the live observer

## Reuse Probe

The proxy has now been exercised on a repeated two-request conversation against
a fresh cache root.

Conversation:

1. request:
   - system: `Answer briefly.`
   - user: `Respond with exactly: OK`
   - response: `OK`
2. request:
   - same system + prior user + prior assistant `OK`
   - new user: `Repeat the same one-word answer.`
   - response: `OK`

Observed telemetry:

1. first request
   - `cached_segments = 0`
   - `bundles_written = 2`
   - `bundle_write_status = "bundles_written_with_trailer"`
2. second request
   - `cached_segments = 1`
   - `first_uncached_turn_index = 1`
   - `cached_kv_slots = 192`
   - `cached_unique_tokens = 8`
   - `bundles_written = 3`
   - `bundle_write_status = "bundles_written_with_trailer"`

Interpretation:

- the endpoint is now writing real segment bundles from live traffic
- the next request is reusing at least one cached prefix segment
- this is still not query-observed compaction, but the end-to-end cache/write
  path is now real

## Known Gaps

Not implemented yet:

- live observer wired into the OpenAI-style request loop
- segment bundle reuse inside a full generation service
- compaction trigger policy at `32k+`
- RoPE/YaRN-style position reclamation
- quantized model path with a verified AWQ/GPTQ backend

Those are next-step engineering tasks, not hidden assumptions in the current
design.

## Confirmed Qwen3.5 Cache Layout

The first live observer graft depends on the real cache layout, which has now
been inspected directly on dense `Qwen3.5-9B`.

Observed after a normal prefill:

- `past_key_values` type:
  - `Qwen3_5DynamicCache`
- full-attention layers:
  - `key_cache[layer]` and `value_cache[layer]` are tensor-backed
  - example shape at short prompt length:
    - `(1, 4, seq_len, 256)`
- linear/recurrent layers:
  - `key_cache[layer]` and `value_cache[layer]` are `None`
  - state instead lives in:
    - `conv_states[layer]`
    - `recurrent_states[layer]`

Operational consequence:

- the first observer milestone can slice K/V directly from
  `past_key_values.key_cache/value_cache` for the 8 full-attention layers
- it does not need to touch recurrent-layer state at all
