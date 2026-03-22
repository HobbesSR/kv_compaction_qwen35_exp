from __future__ import annotations

import json
from pathlib import Path

from kv_compaction_qwen35_clean.boundary_collection import load_boundary_collection
from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.prototype_bank import build_state_from_observations, write_state


def _fixture_boundary_collection() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "boundary_collection.json"


def test_build_state_from_boundary_features_has_probe_coverage() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    bundle = load_boundary_collection(_fixture_boundary_collection())

    state = build_state_from_observations(config, bundle.harvest.observations)

    assert len(state.entries) <= config.sketch.max_prototypes
    assert len({entry.prototype_id for entry in state.entries}) == len(state.entries)
    assert any(entry.layer == 4 and entry.head == 0 for entry in state.entries)


def test_write_state_serializes_json(tmp_path: Path) -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    bundle = load_boundary_collection(_fixture_boundary_collection())
    state = build_state_from_observations(config, bundle.harvest.observations)
    output_path = tmp_path / "prototype_bank.json"

    write_state(state, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sketch_kind"] == "prototype_bank"
    assert payload["tap_point"] == "post_query_pre_head_merge"
    assert payload["entries"]
