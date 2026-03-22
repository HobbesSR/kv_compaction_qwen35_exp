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

The current validated artifact still lives in `clean_repo/` for Qwen2.5. The
job here is to build the analogous Qwen3.5 lane without coupling the two
implementation surfaces.

See [docs/plan.md](./docs/plan.md) and [docs/repo_contract.md](./docs/repo_contract.md).
