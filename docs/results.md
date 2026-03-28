# Results

## Supported Claim

This repo currently supports one clean Qwen3.5 claim:

- on the local `qwen35_smoke_v1` surface with `Qwen3.5-9B`, the calibrated
  full-cache reference reaches `4/4` central details with `0` hallucination
  runs
- on the current validated smoke run, the compacted sketch also preserves
  `4/4` central details with `0` hallucination runs at `128` compact tokens
  over a `7168`-token prefix
- on that same run, the explicit control path preserves `3/4` central details
  with `0` hallucination runs

That is enough to show that:

- the Qwen3.5-native prompt surface is calibrated above floor
- the compaction protocol runs end to end on Qwen3.5
- the sketch prototype bank, after turn-, layer-, and within-layer head fixes,
  can now match reference on the native smoke surface
- the remaining bottlenecks are no longer broad structural failures of the
  sketch path

This is enough to claim a local Qwen3.5 sketch win on the current smoke
surface, but not yet enough to claim a speed-optimized implementation.

## Current Demonstrated Output

Smoke summary (checked-in summary:
`examples/qwen35_smoke/behavioral_eval_summary.json`):

- prompt set: `qwen35_calibration_v3`
- keys per head: `8`
- key selection: `highest_attention`
- reference:
  - runtime: `41.118896s`
  - central details preserved: `4/4`
  - hallucination runs: `0`
- sketch:
  - runtime: `10.926923s`
  - central details preserved: `4/4`
  - hallucination runs: `0`
  - effective compact tokens: `128`
  - compacted heads: `16`
- control:
  - runtime: `10.864395s`
  - central details preserved: `3/4`
  - hallucination runs: `0`
  - effective compact tokens: `128`
  - compacted heads: `16`

Service demo summary:

- prefix tokens: `7168`
- preserved tail tokens: `1024`
- capture token count: `28`
- monitored observations: `672`
- monitored query samples: `672`
- compacted heads: `16`
- effective compact tokens: `96`

The current smoke artifact has no sketch misses. The control miss is:

- `qwen35_branch_switch_appendix_details`
  - missing reference-hit facts: `supplier_phones`, `cage_inventory`,
    `shift_lead_names`
  - the sketch closes this by retaining `(layer, 0)` and `(layer, 7)` across
    all full-attention layers, which recovers the late appendix enumeration
    under the current budget

## Bank Policy History

The sketch result improved over four successive bank fixes:

| Fix | Sketch | Note |
|-----|--------|------|
| baseline | `2/4` | bank collapsed to `turn_7` only |
| turn-diversity replacement policy | `2/4` (null) | equalization was at coreset level, wrong stage |
| turn + layer floor in bank | `3/4` | early-turn misses resolved |
| H0 protection within-layer at `k=6` | `3/4` | appendix miss improved from `0/3` to `2/3` recalled facts |
| H0 protection within-layer at `k=8` | `4/4` | sketch closes the final appendix miss |

The per-turn equalization in `coreset.py` was confirmed null and removed.
The `source_turn_id` field in `PrototypeEntry` was retained for diagnostics.

## Where These Numbers Live

- regenerable local artifacts:
  - `artifacts/qwen35_smoke/behavioral_eval_qwen35_calibration_v3_k8_t40.json`
  - `artifacts/qwen35_smoke/service_demo_summary.json`
- checked-in summary outputs:
  - `examples/qwen35_smoke/behavioral_eval_summary.json`
  - `examples/qwen35_smoke/service_demo_summary.json`

## Non-Claims

This repo does not currently claim:

- that the current path is speed-optimized (eager boundary capture,
  missing Qwen3.5 native extensions, and eager compacted continuation
  are all known; runtimes are honest but not representative of an
  optimized deployment)
- that the prompt surface is final
- that the current result supersedes the validated Qwen2.5 clean lane

The current Qwen3.5 lane is a real, native, end-to-end validation surface with
an honest `4/4 -> 4/4` sketch result at `k=8`. The implementation is still
slow and research-shaped, but the main remaining limitations are now
performance and broader-surface validation rather than a smoke-level quality
gap.
