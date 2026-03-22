from __future__ import annotations

from kv_compaction_qwen35_clean.beta_fit import (
    compare_beta_fit_results,
    fit_beta_for_selected_keys,
    split_query_bank_train_eval,
)
from kv_compaction_qwen35_clean.data_types import (
    BetaFitGroupResult,
    BetaFitResult,
    KeySelectionResult,
    QuerySample,
    SelectedKeyGroup,
)


def test_fit_beta_for_selected_keys_improves_relative_error() -> None:
    query_bank = [
        QuerySample(
            query_id=f"q{i}",
            layer=4,
            head=0,
            token_index=i,
            prefix_mass_share=0.5,
            raw_prefix_mass=float(i + 1),
            query_projection=[1.0, 0.0],
            raw_query_vector=[1.0, 0.0],
            source_turn_id="t0",
            source_speaker="user",
        )
        for i in range(6)
    ]
    selection = KeySelectionResult(
        sample_id="sample",
        boundary_id="boundary",
        source="prototype_bank",
        keys_per_head=2,
        groups=[
            SelectedKeyGroup(
                layer=4,
                head=0,
                selected_indices=[0, 2],
                selected_scores=[1.0, 0.8],
                query_count=2,
                total_query_weight=1.0,
            )
        ],
    )
    boundary_keys = {(4, 0): [[1.0, 0.0], [0.2, 0.0], [0.5, 0.0], [-0.1, 0.0]]}

    result = fit_beta_for_selected_keys(
        "sample",
        "boundary",
        selection,
        query_bank,
        boundary_keys,
    )

    assert result.group_count == 1
    assert result.aggregate_post_eval_mean_abs_rel_error <= result.aggregate_pre_eval_mean_abs_rel_error
    assert result.aggregate_post_over_pre_eval_rel_error_ratio <= 1.0
    assert result.regularization_strength == 0.0
    assert result.train_fraction == 0.5
    assert result.improved_eval_rel_group_count == 1
    assert result.underdetermined_group_count == 0


def test_compare_beta_fit_results_picks_lower_error() -> None:
    sketch = BetaFitResult(
        sample_id="sample",
        boundary_id="boundary",
        source="prototype_bank",
        solver="clamped_lstsq",
        regularization_strength=0.0,
        train_fraction=0.5,
        runtime_seconds=0.1,
        group_count=1,
        aggregate_pre_eval_mean_abs_rel_error=0.5,
        aggregate_post_eval_mean_abs_rel_error=0.1,
        aggregate_post_over_pre_eval_rel_error_ratio=0.2,
        aggregate_pre_eval_mean_abs_log_error=0.2,
        aggregate_post_eval_mean_abs_log_error=0.05,
        aggregate_post_over_pre_eval_log_error_ratio=0.25,
        improved_eval_rel_group_count=1,
        improved_eval_log_group_count=1,
        underdetermined_group_count=0,
        rank_deficient_group_count=0,
        median_condition_number=1.0,
        max_condition_number=1.0,
        groups=[
            BetaFitGroupResult(
                layer=4,
                head=0,
                selected_keys_fingerprint="abc",
                selected_key_count=2,
                train_query_count=2,
                eval_query_count=2,
                design_rank=2,
                condition_number=1.0,
                underdetermined=False,
                pre_train_mean_abs_rel_error=0.4,
                post_train_mean_abs_rel_error=0.1,
                pre_eval_mean_abs_rel_error=0.5,
                post_eval_mean_abs_rel_error=0.1,
                pre_eval_mean_abs_log_error=0.2,
                post_eval_mean_abs_log_error=0.05,
                beta_min=0.0,
                beta_max=0.0,
                beta_mean=0.0,
                runtime_seconds=0.01,
                degeneracy_flags=[],
            )
        ],
    )
    control = BetaFitResult(
        sample_id="sample",
        boundary_id="boundary",
        source="teacher_forced_subsample",
        solver="clamped_lstsq",
        regularization_strength=0.0,
        train_fraction=0.5,
        runtime_seconds=0.1,
        group_count=1,
        aggregate_pre_eval_mean_abs_rel_error=0.5,
        aggregate_post_eval_mean_abs_rel_error=0.2,
        aggregate_post_over_pre_eval_rel_error_ratio=0.4,
        aggregate_pre_eval_mean_abs_log_error=0.2,
        aggregate_post_eval_mean_abs_log_error=0.08,
        aggregate_post_over_pre_eval_log_error_ratio=0.4,
        improved_eval_rel_group_count=0,
        improved_eval_log_group_count=0,
        underdetermined_group_count=0,
        rank_deficient_group_count=0,
        median_condition_number=1.0,
        max_condition_number=1.0,
        groups=sketch.groups,
    )

    comparison = compare_beta_fit_results(sketch, control)

    assert comparison.relative_error_winner == "prototype_bank"
    assert comparison.log_error_winner == "prototype_bank"


def test_regularized_beta_fit_records_solver_metadata() -> None:
    query_bank = [
        QuerySample(
            query_id=f"q{i}",
            layer=4,
            head=0,
            token_index=i,
            prefix_mass_share=0.5,
            raw_prefix_mass=float(i + 1),
            query_projection=[1.0, 0.0],
            raw_query_vector=[1.0, 0.0],
            source_turn_id="t0",
            source_speaker="user",
        )
        for i in range(8)
    ]
    selection = KeySelectionResult(
        sample_id="sample",
        boundary_id="boundary",
        source="prototype_bank",
        keys_per_head=2,
        groups=[
            SelectedKeyGroup(
                layer=4,
                head=0,
                selected_indices=[0, 1],
                selected_scores=[1.0, 0.5],
                query_count=4,
                total_query_weight=1.0,
            )
        ],
    )
    boundary_keys = {(4, 0): [[1.0, 0.0], [0.7, 0.0], [0.2, 0.0], [-0.1, 0.0]]}

    result = fit_beta_for_selected_keys(
        "sample",
        "boundary",
        selection,
        query_bank,
        boundary_keys,
        solver="clamped_ridge",
        regularization_strength=0.01,
    )

    assert result.solver == "clamped_ridge"
    assert result.regularization_strength == 0.01
    assert result.train_fraction == 0.5
    assert result.aggregate_post_eval_mean_abs_rel_error >= 0.0
    assert result.max_condition_number is not None


def test_split_query_bank_train_eval_supports_train_fraction() -> None:
    query_bank = [
        QuerySample(
            query_id=f"q{i}",
            layer=4,
            head=0,
            token_index=i,
            prefix_mass_share=0.5,
            raw_prefix_mass=float(i + 1),
            query_projection=[1.0, 0.0],
            raw_query_vector=[1.0, 0.0],
            source_turn_id="t0",
            source_speaker="user",
        )
        for i in range(8)
    ]

    train, eval_ = split_query_bank_train_eval(query_bank, train_fraction=0.75)

    assert len(train[(4, 0)]) == 6
    assert len(eval_[(4, 0)]) == 2
    assert [sample.token_index for sample in train[(4, 0)]] == [1, 2, 3, 5, 6, 7]
    assert [sample.token_index for sample in eval_[(4, 0)]] == [0, 4]
