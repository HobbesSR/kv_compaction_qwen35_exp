# Reproduction

## Environment

`qwen35_clean` assumes:

- Python `3.9+`
- `torch`
- `transformers` with `Qwen3.5` model support
- `PyYAML`

Model-backed commands require the dedicated Qwen3.5 environment used in the
research workspace.

Recommended install inside that environment:

```bash
cd qwen35_clean
python3 -m pip install . --user
```

This installs the console entry points:

- `kv-qwen35-smoke`
- `kv-qwen35-demo`
- `kv-qwen35-export-examples`

## Smoke Result

Run:

```bash
cd qwen35_clean
kv-qwen35-smoke
```

Output artifact:

- `artifacts/qwen35_smoke/behavioral_eval_qwen35_calibration_v3_k8_t40.json`

Current observed local summary on the default `Qwen/Qwen3.5-9B` config:

- reference:
  - runtime: `53.494985s`
  - central details preserved: `4/4`
  - hallucination runs: `0`
- sketch:
  - runtime: `61.673165s`
  - central details preserved: `4/4`
  - hallucination runs: `0`
  - effective compact tokens: `128`
- control:
  - runtime: `50.569344s`
  - central details preserved: `3/4`
  - hallucination runs: `0`
  - effective compact tokens: `128`

These are the current local demonstration numbers, not paper claims.

The smoke command uses `k=8` because that is the current validated best
artifact. The interactive service demo remains on a cheaper `k=6` setup.

## Service Demo

Run:

```bash
cd qwen35_clean
kv-qwen35-demo
```

The demo prints ingest progress during boundary collection, writes:

- `artifacts/qwen35_smoke/service_demo_summary.json`

and then accepts:

- `/compact <prompt>`
- `/full <prompt>`
- `/status`
- `/quit`

Current observed local summary:

- prefix tokens: `7168`
- preserved tail tokens: `1024`
- capture token count: `28`
- monitored observations: `672`
- monitored query samples: `672`
- compacted heads: `16`
- effective compact tokens: `96`

## Artifact Policy

`artifacts/` is intentionally ignored in git.

The repo policy is:

- check in code, configs, datasets, and docs
- regenerate artifacts locally from documented commands
- keep a tiny checked-in summary set under `examples/qwen35_smoke/`
- document the current observed outputs in this file and the README

To refresh the checked-in summaries after a validated local run:

```bash
cd qwen35_clean
kv-qwen35-export-examples
```
