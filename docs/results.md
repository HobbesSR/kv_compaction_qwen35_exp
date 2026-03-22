# Results

## Supported Claim

This repo currently supports one clean Qwen3.5 claim:

- on the local `qwen35_smoke_v1` surface with `Qwen3.5-9B`, the calibrated
  full-cache reference reaches `4/4` central details with `0` hallucination
  runs
- on that same run, both the compacted sketch and the explicit control path
  preserve `3/4` central details with `0` hallucination runs

That is enough to show that:

- the Qwen3.5-native prompt surface is calibrated above floor
- the compaction protocol runs end to end on Qwen3.5
- the current bottleneck is not prompt calibration anymore

It is not yet enough to claim a Qwen3.5 win for sketch over explicit control.

## Current Demonstrated Output

Smoke summary:

- prompt set: `qwen35_calibration_v2`
- keys per head: `6`
- key selection: `highest_attention`
- reference:
  - runtime: `51.145611s`
  - central details preserved: `4/4`
  - hallucination runs: `0`
- sketch:
  - runtime: `45.145773s`
  - central details preserved: `3/4`
  - hallucination runs: `0`
  - effective compact tokens: `96`
- control:
  - runtime: `44.972305s`
  - central details preserved: `3/4`
  - hallucination runs: `0`
  - effective compact tokens: `96`

Service demo summary:

- prefix tokens: `7168`
- preserved tail tokens: `1024`
- capture token count: `28`
- monitored observations: `672`
- monitored query samples: `672`
- compacted heads: `16`
- effective compact tokens: `96`

The one remaining miss on both sketch and control is:

- `qwen35_same_task_handoff_rollback`
  - missing reference-hit fact: `handoff_checklist`

## Where These Numbers Live

- checked-in summaries:
  - `examples/qwen35_smoke/behavioral_eval_summary.json`
  - `examples/qwen35_smoke/service_demo_summary.json`
- regenerable local artifacts:
  - `artifacts/qwen35_smoke/behavioral_eval_qwen35_calibration_v2_k6_t40.json`
  - `artifacts/qwen35_smoke/service_demo_summary.json`
  - `artifacts/qwen35_smoke/reference_calibration_qwen35_calibration_v2_t40.json`

## Non-Claims

This repo does not currently claim:

- that Qwen3.5 sketch already beats the explicit control path
- that the current path is speed-optimized
- that the prompt surface is final
- that the current result supersedes the validated Qwen2.5 clean lane

The current Qwen3.5 lane is a real, native, end-to-end validation surface with
an honest `4/4 -> 3/4` result, not a polished win claim.
