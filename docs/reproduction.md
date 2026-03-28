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
- `kv-qwen35-proxy`
- `kv-qwen35-roo-lite`

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
  - runtime: `41.118896s`
  - central details preserved: `4/4`
  - hallucination runs: `0`
- sketch:
  - runtime: `10.926923s`
  - central details preserved: `4/4`
  - hallucination runs: `0`
  - effective compact tokens: `128`
- control:
  - runtime: `10.864395s`
  - central details preserved: `3/4`
  - hallucination runs: `0`
  - effective compact tokens: `128`

These are the current local demonstration numbers, not paper claims.

The smoke command uses `k=8` because that is the current validated best
artifact. The interactive service demo remains on a cheaper `k=6` setup.

## Proxy And Roo-Lite

These commands are intentionally separate from the clean smoke claim. They use
the same Qwen3.5 runtime surface but exercise a service-shaped endpoint and a
minimal local tool agent.

Run the OpenAI-compatible proxy:

```bash
cd qwen35_clean
kv-qwen35-proxy --config configs/qwen35_smoke/qwen3_5_9b.yaml --port 8010
```

Run the minimal tool agent against a running proxy:

```bash
cd qwen35_clean
kv-qwen35-roo-lite \
  --base-url http://127.0.0.1:8010/v1 \
  --model Qwen/Qwen3.5-9B
```

The service/agent track is documented in:

- `docs/roo_proxy_experiment.md`

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

The curated checked-in summaries live under:

- `examples/qwen35_smoke/behavioral_eval_summary.json`
- `examples/qwen35_smoke/service_demo_summary.json`
