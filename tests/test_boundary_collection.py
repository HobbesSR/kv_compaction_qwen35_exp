from __future__ import annotations

import json
from pathlib import Path

from kv_compaction_qwen35_clean.boundary_collection import (
    load_boundary_collection,
    select_long_context_capture_indices,
    write_boundary_collection,
)
from kv_compaction_qwen35_clean.data_types import (
    BoundaryCollection,
    FeatureHarvest,
    FeatureObservation,
    QuerySample,
    QuerySampleBank,
)


def test_select_long_context_capture_indices_includes_last_prefix_token() -> None:
    assert select_long_context_capture_indices(0) == []
    assert select_long_context_capture_indices(1) == []
    assert select_long_context_capture_indices(512) == [255, 511]
    assert select_long_context_capture_indices(7168)[-1] == 7167


def test_write_and_load_boundary_collection_roundtrip(tmp_path: Path) -> None:
    bundle = BoundaryCollection(
        harvest=FeatureHarvest(
            sample_id="sample",
            boundary_id="boundary",
            logical_context_tokens=1024,
            physical_context_tokens=1024,
            feature_granularity="per_head",
            tap_point="post_query_pre_head_merge",
            query_projection_dim=4,
            output_projection_dim=4,
            observed_layers=[4],
            observed_heads=[0],
            observation_count=1,
            observations=[
                FeatureObservation(
                    token_index=255,
                    layer=4,
                    head=0,
                    tap_point="post_query_pre_head_merge",
                    query_projection=[0.1, 0.2, 0.3, 0.4],
                    prefix_mass_share=0.25,
                    raw_prefix_mass=63.75,
                    output_projection=[0.5, 0.6, 0.7, 0.8],
                    source_turn_id="turn_1",
                    source_speaker="tool",
                )
            ],
        ),
        query_bank=QuerySampleBank(
            sample_id="sample",
            boundary_id="boundary",
            query_dim=8,
            sample_count=1,
            samples=[
                QuerySample(
                    query_id="sample:4:0:255",
                    layer=4,
                    head=0,
                    token_index=255,
                    prefix_mass_share=0.25,
                    raw_prefix_mass=63.75,
                    query_projection=[0.1, 0.2, 0.3, 0.4],
                    raw_query_vector=[0.1] * 8,
                    source_turn_id="turn_1",
                    source_speaker="tool",
                )
            ],
        ),
        boundary_keys={(4, 0): [[0.1, 0.2], [0.3, 0.4]]},
        boundary_values={(4, 0): [[0.5, 0.6], [0.7, 0.8]]},
        boundary_projected_values={(4, 0): [[0.9, 1.0], [1.1, 1.2]]},
        output_targets={(4, 0, 255): [0.01, 0.02]},
        runtime_cache=None,
        capture_token_indices=[255, 511],
        monitored_observation_count=1,
        monitored_query_sample_count=1,
    )
    output_path = tmp_path / "boundary_collection.json"

    write_boundary_collection(bundle, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    loaded = load_boundary_collection(output_path)

    assert payload["query_bank"]["sample_count"] == 1
    assert payload["capture_token_indices"] == [255, 511]
    assert loaded.harvest.observation_count == 1
    assert loaded.query_bank.sample_count == 1
    assert loaded.boundary_keys[(4, 0)][0] == [0.1, 0.2]
    assert loaded.output_targets[(4, 0, 255)] == [0.01, 0.02]


def test_write_boundary_collection_rejects_runtime_cache(tmp_path: Path) -> None:
    bundle = BoundaryCollection(
        harvest=FeatureHarvest(
            sample_id="sample",
            boundary_id="boundary",
            logical_context_tokens=1,
            physical_context_tokens=1,
            feature_granularity="per_head",
            tap_point="tap",
            query_projection_dim=1,
            output_projection_dim=1,
            observed_layers=[],
            observed_heads=[],
            observation_count=0,
            observations=[],
        ),
        query_bank=QuerySampleBank(
            sample_id="sample",
            boundary_id="boundary",
            query_dim=1,
            sample_count=0,
            samples=[],
        ),
        boundary_keys={},
        boundary_values={},
        boundary_projected_values={},
        output_targets={},
        runtime_cache=object(),
    )

    try:
        write_boundary_collection(bundle, tmp_path / "boundary_collection.json")
    except ValueError as exc:
        assert "runtime_cache" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected runtime_cache serialization to fail.")
