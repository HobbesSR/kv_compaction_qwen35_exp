# Qwen3.5 Clean Repo

This is the parallel clean-artifact lane for the Qwen3.5 program. It lives
inside the research workspace for convenience, but it is structured to split
into its own repository cleanly.

## What This Repo Is For

This repo exists to answer two questions cleanly:

1. Can the compaction protocol validated on Qwen2.5 be reproduced on a native
   Qwen3.5 evaluation surface?
2. Can the same protocol support a qualitatively service-shaped local Qwen3.5
   pipeline for long-running agentic/orchestrator use?

This is not a copy of the Qwen2.5 clean repo and not a dump of the research
branch. It is a parallel story with shared protocol and separate calibration.

## Intended Deliverables

### 1. `qwen35_smoke_v1`

- native Qwen3.5 prompt surface
- same compaction protocol:
  - boundary bundle
  - prototype bank
  - query coreset
  - key selection
  - runtime fit
- reference vs sketch vs explicit control

### 2. `qwen35_service_demo`

- long context ingestion with visible progress
- boundary-triggered compaction
- continued answering over compacted state
- geared toward practical local orchestrator sessions

## Repo Principles

- no runtime dependency on the parent research package
- no prompt inheritance from Qwen2.5 unless explicitly used as an overlap check
- Qwen3.5-native calibration first, parity claims second
- one clean story first, variants later

## Current State

This repo has the first real Qwen3.5 substrate in place:

- config and context loading
- runtime planning
- teacher-forced boundary collection
- prototype bank state extraction
- query coreset extraction
- key selection
- beta fitting
- native Qwen3.5 smoke-eval runner

Model-backed Qwen3.5 commands still require the dedicated Qwen3.5 environment.
In the default workspace env, those commands fail fast with a clear dependency
message instead of a raw Transformers stack trace.

Operational note:

- on this machine, `Qwen3.5-9B` currently runs on the torch fallback path
  because the fast-path extensions are not installed
- the native smoke runner now exposes `--prompt-limit` and
  `--max-new-tokens` for bring-up and smoke iteration under that slower path

The current validated artifact still lives in `clean_repo/` for Qwen2.5. The
job here is to build the analogous Qwen3.5 lane without coupling the two
implementation surfaces.

## Current Demonstrated Result

On the current validated local `Qwen3.5-9B` run:

- reference: `4/4` central details, `0` hallucination runs
- sketch: `3/4` central details, `0` hallucination runs
- control: `3/4` central details, `0` hallucination runs

That means the native Qwen3.5 smoke surface is now calibrated above floor and
the compaction protocol works end to end on the model family. The remaining
gap is that sketch does not yet beat the explicit control path on this lane.

## Installed Commands

After install:

```bash
cd qwen35_clean
python3 -m pip install . --user
```

the repo exposes:

- `kv-qwen35-smoke`
- `kv-qwen35-demo`
- `kv-qwen35-export-examples`

See:

- [docs/plan.md](./docs/plan.md)
- [docs/repo_contract.md](./docs/repo_contract.md)
- [docs/architecture.md](./docs/architecture.md)
- [docs/results.md](./docs/results.md)
- [docs/reproduction.md](./docs/reproduction.md)
