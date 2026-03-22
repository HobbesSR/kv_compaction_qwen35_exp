from __future__ import annotations

import json
from pathlib import Path

from kv_compaction_qwen35_clean.boundary_collection import load_boundary_collection
from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.data_types import FeatureObservation
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


def test_build_state_recycles_represented_turn_before_evicting_singletons() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    config.sketch.max_prototypes = 3
    observations = [
        FeatureObservation(
            token_index=0,
            layer=4,
            head=0,
            tap_point="post_query_pre_head_merge",
            query_projection=[1.0, 0.0],
            prefix_mass_share=0.4,
            raw_prefix_mass=0.4,
            output_projection=[1.0, 0.0],
            source_turn_id="turn_1",
        ),
        FeatureObservation(
            token_index=1,
            layer=12,
            head=0,
            tap_point="post_query_pre_head_merge",
            query_projection=[0.0, 1.0],
            prefix_mass_share=0.4,
            raw_prefix_mass=0.4,
            output_projection=[0.0, 1.0],
            source_turn_id="turn_2",
        ),
        FeatureObservation(
            token_index=2,
            layer=20,
            head=0,
            tap_point="post_query_pre_head_merge",
            query_projection=[1.0, 1.0],
            prefix_mass_share=0.4,
            raw_prefix_mass=0.4,
            output_projection=[1.0, 1.0],
            source_turn_id="turn_3",
        ),
        FeatureObservation(
            token_index=3,
            layer=28,
            head=0,
            tap_point="post_query_pre_head_merge",
            query_projection=[1.0, -1.0],
            prefix_mass_share=0.4,
            raw_prefix_mass=0.4,
            output_projection=[1.0, -1.0],
            source_turn_id="turn_3",
        ),
        FeatureObservation(
            token_index=4,
            layer=28,
            head=3,
            tap_point="post_query_pre_head_merge",
            query_projection=[1.0, -0.5],
            prefix_mass_share=0.4,
            raw_prefix_mass=0.4,
            output_projection=[1.0, -0.5],
            source_turn_id="turn_3",
        ),
    ]

    state = build_state_from_observations(config, observations)

    surviving_turns = sorted(entry.source_turn_id for entry in state.entries)
    assert surviving_turns.count("turn_1") == 1
    assert surviving_turns.count("turn_2") == 1
    assert surviving_turns.count("turn_3") == 1


def test_build_state_preserves_layer_floor_when_capacity_allows() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    config.sketch.max_prototypes = 4
    observations = [
        FeatureObservation(0, 3, 0, "post_query_pre_head_merge", [1.0, 0.0], 0.4, 0.4, [1.0, 0.0], "turn_0"),
        FeatureObservation(1, 7, 0, "post_query_pre_head_merge", [0.0, 1.0], 0.4, 0.4, [0.0, 1.0], "turn_1"),
        FeatureObservation(2, 11, 0, "post_query_pre_head_merge", [1.0, 1.0], 0.4, 0.4, [1.0, 1.0], "turn_2"),
        FeatureObservation(3, 15, 0, "post_query_pre_head_merge", [1.0, -1.0], 0.4, 0.4, [1.0, -1.0], "turn_3"),
        FeatureObservation(4, 15, 3, "post_query_pre_head_merge", [0.8, -0.8], 0.4, 0.4, [0.8, -0.8], "turn_4"),
        FeatureObservation(5, 15, 7, "post_query_pre_head_merge", [0.6, -0.6], 0.4, 0.4, [0.6, -0.6], "turn_5"),
    ]

    state = build_state_from_observations(config, observations)

    assert {entry.layer for entry in state.entries} == {3, 7, 11, 15}


def test_build_state_promotes_underrepresented_heads_within_layer_quota() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    config.sketch.max_prototypes = 4
    observations = [
        FeatureObservation(0, 3, 3, "post_query_pre_head_merge", [1.0, 0.0], 0.4, 0.4, [1.0, 0.0], "turn_0"),
        FeatureObservation(1, 3, 7, "post_query_pre_head_merge", [0.0, 1.0], 0.4, 0.4, [0.0, 1.0], "turn_0"),
        FeatureObservation(2, 7, 3, "post_query_pre_head_merge", [1.0, 1.0], 0.4, 0.4, [1.0, 1.0], "turn_1"),
        FeatureObservation(3, 7, 7, "post_query_pre_head_merge", [1.0, -1.0], 0.4, 0.4, [1.0, -1.0], "turn_1"),
        FeatureObservation(4, 3, 0, "post_query_pre_head_merge", [0.5, 0.5], 0.4, 0.4, [0.5, 0.5], "turn_2"),
    ]

    state = build_state_from_observations(config, observations)

    layer_3_heads = {entry.head for entry in state.entries if entry.layer == 3}
    assert 0 in layer_3_heads
