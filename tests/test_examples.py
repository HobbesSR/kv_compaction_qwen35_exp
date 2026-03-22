from __future__ import annotations

import json
from pathlib import Path


def test_behavioral_eval_summary_example_shape() -> None:
    path = Path("examples/qwen35_smoke/behavioral_eval_summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["prompt_set"] == "qwen35_calibration_v3"
    assert payload["reference"]["preserved_central_detail_count"] >= 4
    assert payload["control"]["preserved_central_detail_count"] >= 4
    assert payload["sketch"]["preserved_central_detail_count"] >= 2


def test_service_demo_summary_example_shape() -> None:
    path = Path("examples/qwen35_smoke/service_demo_summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sample_id"]
    assert payload["effective_compact_tokens"] > 0
