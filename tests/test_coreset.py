from __future__ import annotations

import json
from pathlib import Path

from kv_compaction_qwen35_clean.boundary_collection import load_boundary_collection
from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.coreset import extract_query_coreset, write_query_coreset
from kv_compaction_qwen35_clean.prototype_bank import build_state_from_observations


def _fixture_boundary_collection() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "boundary_collection.json"


def test_extract_query_coreset_selects_ranked_entries() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    bundle = load_boundary_collection(_fixture_boundary_collection())
    state = build_state_from_observations(config, bundle.harvest.observations)

    coreset = extract_query_coreset(bundle.harvest.sample_id, bundle.harvest.boundary_id, state, config)

    assert coreset.source == "prototype_bank"
    assert coreset.max_entries <= config.sketch.max_prototypes
    assert len(coreset.selected_entries) == coreset.max_entries
    assert coreset.total_weight > 0.0


def test_write_query_coreset_serializes_json(tmp_path: Path) -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    bundle = load_boundary_collection(_fixture_boundary_collection())
    state = build_state_from_observations(config, bundle.harvest.observations)
    coreset = extract_query_coreset(bundle.harvest.sample_id, bundle.harvest.boundary_id, state, config)
    output_path = tmp_path / "query_coreset.json"

    write_query_coreset(coreset, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source"] == "prototype_bank"
    assert payload["selected_entries"]
