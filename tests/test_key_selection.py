from __future__ import annotations

import torch

from kv_compaction_qwen35_clean.data_types import QueryCoreset, QueryCoresetEntry, QuerySample
from kv_compaction_qwen35_clean.head_budget import resolve_head_budgets
from kv_compaction_qwen35_clean.key_selection import (
    compare_key_selection_results,
    match_coreset_to_query_samples,
    select_keys,
    select_keys_with_highest_attention,
)


def test_match_coreset_to_query_samples_prefers_same_group() -> None:
    coreset = QueryCoreset(
        sample_id="sample",
        boundary_id="boundary",
        source="prototype_bank",
        max_entries=1,
        total_weight=1.0,
        selected_entries=[
            QueryCoresetEntry(
                coreset_id="q0",
                prototype_id="p0",
                layer=4,
                head=3,
                weight=1.0,
                avg_prefix_mass_share=0.5,
                avg_raw_prefix_mass=10.0,
                query_projection=[1.0, 0.0],
                output_projection_hint=[],
                last_token_index=15,
            )
        ],
    )
    samples = [
        QuerySample(
            query_id="a",
            layer=4,
            head=3,
            token_index=15,
            prefix_mass_share=0.4,
            raw_prefix_mass=6.0,
            query_projection=[0.9, 0.0],
            raw_query_vector=[1.0, 0.0],
            source_turn_id="t0",
            source_speaker="user",
        ),
        QuerySample(
            query_id="b",
            layer=4,
            head=0,
            token_index=31,
            prefix_mass_share=0.9,
            raw_prefix_mass=20.0,
            query_projection=[1.0, 0.0],
            raw_query_vector=[1.0, 0.0],
            source_turn_id="t1",
            source_speaker="assistant",
        ),
    ]

    matches = match_coreset_to_query_samples(coreset, samples)

    assert len(matches) == 1
    assert matches[0][1].query_id == "a"


def test_select_keys_and_compare_results() -> None:
    q_entry = QueryCoresetEntry(
        coreset_id="q0",
        prototype_id="p0",
        layer=4,
        head=0,
        weight=1.0,
        avg_prefix_mass_share=0.8,
        avg_raw_prefix_mass=8.0,
        query_projection=[1.0, 0.0],
        output_projection_hint=[],
        last_token_index=7,
    )
    sample = QuerySample(
        query_id="qa",
        layer=4,
        head=0,
        token_index=7,
        prefix_mass_share=0.8,
        raw_prefix_mass=8.0,
        query_projection=[1.0, 0.0],
        raw_query_vector=[1.0, 0.0],
        source_turn_id="t0",
        source_speaker="user",
    )
    keys = {(4, 0): [[1.0, 0.0], [0.0, 1.0], [0.9, 0.0]]}

    sketch = select_keys_with_highest_attention(
        "sample",
        "boundary",
        "prototype_bank",
        [(q_entry, sample)],
        keys,
        keys_per_head=2,
    )
    control = select_keys_with_highest_attention(
        "sample",
        "boundary",
        "teacher_forced_subsample",
        [(q_entry, sample)],
        keys,
        keys_per_head=2,
    )
    comparison = compare_key_selection_results(sketch, control)

    assert sketch.groups[0].selected_indices[0] == 0
    assert comparison.overlap_by_group[0]["jaccard"] == 1.0


def test_select_keys_accepts_tensor_boundary_keys() -> None:
    q_entry = QueryCoresetEntry(
        coreset_id="q0",
        prototype_id="p0",
        layer=4,
        head=0,
        weight=1.0,
        avg_prefix_mass_share=0.8,
        avg_raw_prefix_mass=8.0,
        query_projection=[1.0, 0.0],
        output_projection_hint=[],
        last_token_index=7,
    )
    sample = QuerySample(
        query_id="qa",
        layer=4,
        head=0,
        token_index=7,
        prefix_mass_share=0.8,
        raw_prefix_mass=8.0,
        query_projection=[1.0, 0.0],
        raw_query_vector=[1.0, 0.0],
        source_turn_id="t0",
        source_speaker="user",
    )
    keys = {(4, 0): torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.0]], dtype=torch.float32)}

    result = select_keys_with_highest_attention(
        "sample",
        "boundary",
        "prototype_bank",
        [(q_entry, sample)],
        keys,
        keys_per_head=2,
    )

    assert result.groups[0].selected_indices == [0, 2]


def test_select_keys_supports_omp_selection() -> None:
    q_entry = QueryCoresetEntry(
        coreset_id="q0",
        prototype_id="p0",
        layer=4,
        head=0,
        weight=1.0,
        avg_prefix_mass_share=0.8,
        avg_raw_prefix_mass=8.0,
        query_projection=[1.0, 0.0],
        output_projection_hint=[],
        last_token_index=7,
    )
    sample = QuerySample(
        query_id="qa",
        layer=4,
        head=0,
        token_index=7,
        prefix_mass_share=0.8,
        raw_prefix_mass=8.0,
        query_projection=[1.0, 0.0],
        raw_query_vector=[1.0, 0.0],
        source_turn_id="t0",
        source_speaker="user",
    )
    keys = {(4, 0): torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.0]], dtype=torch.float32)}

    result = select_keys(
        "sample",
        "boundary",
        "prototype_bank",
        [(q_entry, sample)],
        keys,
        keys_per_head=2,
        selection_method="omp",
    )

    assert result.groups[0].selected_indices[0] == 0
    assert len(result.groups[0].selected_indices) == 2
    assert len(result.groups[0].selected_scores) == 2


def test_resolve_head_budgets_preserves_total_budget() -> None:
    budgets = resolve_head_budgets(
        group_keys=[(4, 0), (4, 1), (12, 0)],
        keys_per_head=2,
        head_budget_proportions={
            (4, 0): 0.6,
            (4, 1): 0.3,
            (12, 0): 0.1,
        },
        min_keys_per_head=1,
    )

    assert sum(budgets.values()) == 6
    assert budgets[(4, 0)] >= budgets[(4, 1)] >= budgets[(12, 0)]


def test_resolve_head_budgets_rejects_disjoint_schedule() -> None:
    try:
        resolve_head_budgets(
            group_keys=[(4, 0), (4, 1)],
            keys_per_head=2,
            head_budget_proportions={(20, 0): 1.0},
            min_keys_per_head=1,
        )
    except ValueError as exc:
        assert "does not cover the active layer/head groups" in str(exc)
    else:
        raise AssertionError("Expected disjoint head-budget schedules to be rejected.")
