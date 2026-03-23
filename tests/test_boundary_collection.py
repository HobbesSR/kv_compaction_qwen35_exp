from __future__ import annotations

import json
from pathlib import Path

import torch

from kv_compaction_qwen35_clean.boundary_collection import (
    AttentionTraceChunkBuffer,
    _build_capture_rows_from_trace_payload,
    load_boundary_collection,
    resolve_replay_checkpoint_start,
    select_boundary_biased_capture_indices,
    select_long_context_capture_indices,
    write_boundary_collection,
)
from kv_compaction_qwen35_clean.config import load_config
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


def test_select_boundary_biased_capture_indices_prefers_last_turns() -> None:
    turn_spans = [
        (0, 512, "turn_0", "system"),
        (512, 1536, "turn_1", "tool"),
        (1536, 2560, "turn_2", "tool"),
        (2560, 3328, "turn_3", "assistant"),
        (3328, 4608, "turn_4", "tool"),
        (4608, 5504, "turn_5", "user"),
        (5504, 6912, "turn_6", "assistant"),
        (6912, 7168, "turn_7", "tool"),
    ]

    indices = select_boundary_biased_capture_indices(7168, turn_spans, lookback_turns=3)

    assert indices[0] >= 4608
    assert 5503 in indices
    assert 6911 in indices
    assert 7167 in indices


def test_select_boundary_biased_capture_indices_falls_back_without_turns() -> None:
    indices = select_boundary_biased_capture_indices(1024, [], lookback_turns=2, stride=256)

    assert indices == [255, 511, 767, 1023]


def test_resolve_replay_checkpoint_start_uses_turn_boundary_before_first_capture() -> None:
    turn_spans = [
        (0, 512, "turn_0", "system"),
        (512, 1536, "turn_1", "tool"),
        (1536, 2560, "turn_2", "tool"),
        (2560, 3328, "turn_3", "assistant"),
        (3328, 4608, "turn_4", "tool"),
        (4608, 5504, "turn_5", "user"),
        (5504, 6912, "turn_6", "assistant"),
        (6912, 7168, "turn_7", "tool"),
    ]

    assert resolve_replay_checkpoint_start([4863, 5119, 7167], turn_spans) == 4608
    assert resolve_replay_checkpoint_start([255, 511], turn_spans) == 0


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


def test_attention_trace_chunk_buffer_roundtrip() -> None:
    buffer = AttentionTraceChunkBuffer(capacity=4, query_length=2)
    buffer.add_query_position(
        query_position=0,
        layer_indices=torch.tensor([3, 7], dtype=torch.long),
        head_indices=torch.tensor([0, 7], dtype=torch.long),
        prefix_mass_shares=torch.tensor([0.25, 0.75], dtype=torch.float32),
        raw_query_vectors=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        raw_outputs=torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
    )

    payload = buffer.snapshot_for_query_position(0)

    assert payload is not None
    assert payload["layer_indices"].tolist() == [3, 7]
    assert payload["head_indices"].tolist() == [0, 7]
    assert payload["raw_outputs"].tolist() == [[5.0, 6.0], [7.0, 8.0]]


def test_attention_trace_chunk_buffer_filters_absolute_positions() -> None:
    buffer = AttentionTraceChunkBuffer(capacity=4, query_length=4, tracked_absolute_positions={11, 13})
    buffer.add_query_position(
        query_position=0,
        absolute_query_position=10,
        layer_indices=torch.tensor([3], dtype=torch.long),
        head_indices=torch.tensor([0], dtype=torch.long),
        prefix_mass_shares=torch.tensor([0.25], dtype=torch.float32),
        raw_query_vectors=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        raw_outputs=torch.tensor([[5.0, 6.0]], dtype=torch.float32),
    )
    buffer.add_query_position(
        query_position=1,
        absolute_query_position=11,
        layer_indices=torch.tensor([7], dtype=torch.long),
        head_indices=torch.tensor([7], dtype=torch.long),
        prefix_mass_shares=torch.tensor([0.75], dtype=torch.float32),
        raw_query_vectors=torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        raw_outputs=torch.tensor([[7.0, 8.0]], dtype=torch.float32),
    )

    assert buffer.snapshot_for_query_position(10) is None
    payload = buffer.snapshot_for_query_position(11)
    assert payload is not None
    assert payload["layer_indices"].tolist() == [7]
    assert payload["head_indices"].tolist() == [7]


def test_build_capture_rows_from_trace_payload() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    rows = _build_capture_rows_from_trace_payload(
        trace_payload={
            "layer_indices": torch.tensor([3], dtype=torch.long),
            "head_indices": torch.tensor([7], dtype=torch.long),
            "prefix_mass_shares": torch.tensor([0.5], dtype=torch.float32),
            "raw_query_vectors": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            "raw_outputs": torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32),
        },
        token_index=255,
        config=config,
    )

    assert len(rows) == 1
    assert rows[0]["layer"] == 3
    assert rows[0]["head"] == 7
    assert rows[0]["raw_prefix_mass"] == 127.5
    assert len(rows[0]["query_projection"]) == config.feature_schema.query_projection_dim
    assert len(rows[0]["output_projection"]) == config.feature_schema.output_projection_dim
