from __future__ import annotations

import torch

from kv_compaction_qwen35_clean.data_types import (
    BoundaryCollection,
    FeatureHarvest,
    QueryCoreset,
    QueryCoresetEntry,
    QuerySample,
    QuerySampleBank,
)
from kv_compaction_qwen35_clean.runtime_compaction import build_path_runtime


def test_build_path_runtime_returns_selected_head_runtime() -> None:
    bundle = BoundaryCollection(
        harvest=FeatureHarvest(
            sample_id="sample",
            boundary_id="boundary",
            logical_context_tokens=16,
            physical_context_tokens=16,
            feature_granularity="per_head",
            tap_point="post_query_pre_head_merge",
            query_projection_dim=2,
            output_projection_dim=2,
            observed_layers=[4],
            observed_heads=[0],
            observation_count=2,
            observations=[],
        ),
        query_bank=QuerySampleBank(
            sample_id="sample",
            boundary_id="boundary",
            query_dim=2,
            sample_count=4,
            samples=[
                QuerySample(
                    query_id=f"q{index}",
                    layer=4,
                    head=0,
                    token_index=index + 1,
                    prefix_mass_share=0.6,
                    raw_prefix_mass=1.0 + index,
                    query_projection=[1.0, 0.0],
                    raw_query_vector=[1.0, 0.0],
                    source_turn_id="turn",
                    source_speaker="tool",
                )
                for index in range(4)
            ],
        ),
        boundary_keys={(4, 0): [[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [-1.0, 0.0]]},
        boundary_values={(4, 0): [[1.0, 0.0], [0.5, 0.1], [0.0, 1.0], [-1.0, 0.0]]},
        boundary_projected_values={(4, 0): [[1.0, 0.0], [0.5, 0.1], [0.0, 1.0], [-1.0, 0.0]]},
        output_targets={(4, 0, index + 1): [1.0, 0.0] for index in range(4)},
    )
    query_source = QueryCoreset(
        sample_id="sample",
        boundary_id="boundary",
        source="prototype_bank",
        max_entries=2,
        total_weight=1.2,
        selected_entries=[
            QueryCoresetEntry(
                coreset_id="c0",
                prototype_id="p0",
                layer=4,
                head=0,
                weight=0.6,
                avg_prefix_mass_share=0.6,
                avg_raw_prefix_mass=2.0,
                query_projection=[1.0, 0.0],
                output_projection_hint=[],
                last_token_index=1,
            ),
            QueryCoresetEntry(
                coreset_id="c1",
                prototype_id="p1",
                layer=4,
                head=0,
                weight=0.6,
                avg_prefix_mass_share=0.6,
                avg_raw_prefix_mass=3.0,
                query_projection=[1.0, 0.0],
                output_projection_hint=[],
                last_token_index=2,
            ),
        ],
    )

    selection, runtimes = build_path_runtime(
        "sample",
        "boundary",
        "prototype_bank",
        2,
        bundle,
        query_source,
        target_layers=(4,),
        target_heads=(0,),
        target_layer_heads=((4, 0),),
        compute_device="cpu",
    )

    assert selection.groups[0].selected_indices[0] == 0
    runtime = runtimes[4][0]
    assert runtime.layer == 4
    assert runtime.head == 0
    assert runtime.compact_keys.shape == torch.Size([2, 2])
    assert runtime.compact_values.shape == torch.Size([2, 2])
    assert runtime.beta.shape == torch.Size([2])
