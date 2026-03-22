# Repo Contract

## What Belongs Here

- stable Qwen3.5 implementation needed to rerun the clean smoke result
- native Qwen3.5 prompt surfaces
- a small interactive Qwen3.5 demo path
- concise docs that state exactly what the repo claims

## What Does Not Belong Here

- hidden imports from the parent research repo
- Qwen2.5 prompt calibration carried over as default judgment
- research-only branches that do not affect the clean story
- large exploratory result ladders

## Evaluation Contract

Every Qwen3.5 result in this repo should state:

- the exact model
- whether thinking/reasoning mode is enabled
- the prompt surface
- the controls
- the claimed result

## Relationship To Qwen2.5

This repo may share protocol with the Qwen2.5 clean repo, but it should not
pretend that prompt alignment or baseline quality transfers automatically.

The clean comparison principle is:

- shared compaction protocol
- separate model-family calibration
