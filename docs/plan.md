# Qwen3.5 Clean Plan

## Goal

Build a standalone Qwen3.5 clean artifact that mirrors the validated protocol
from the Qwen2.5 clean repo while keeping Qwen3.5 evaluation native to the
model family.

## First Public Story

`qwen35_smoke_v1`:

- one native Qwen3.5 task surface
- one local model rung that clears the quality floor
- full-cache reference vs compacted sketch vs explicit control
- rerunnable from config and scripts

## Second Story

`qwen35_service_demo`:

- visible long-context ingestion
- boundary compaction
- continued answering on compacted state
- oriented toward local orchestrator-style use

## Migration Order

1. scaffold and contract
2. runtime/model bring-up
3. native prompt surface and baseline floor
4. boundary bundle
5. prototype bank / coreset
6. key selection and runtime fit
7. smoke evaluation
8. service demo shell

## Explicit Non-Goals

- no forced reuse of the Qwen2.5 prompt ecology
- no coupling to `clean_repo/` at runtime
- no attempt to solve every research branch before first end-to-end Qwen3.5 path
