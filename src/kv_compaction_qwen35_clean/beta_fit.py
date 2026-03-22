from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from kv_compaction_qwen35_clean.data_types import (
    BetaFitComparison,
    BetaFitGroupResult,
    BetaFitResult,
    KeySelectionResult,
    QuerySample,
)


def split_query_bank_train_eval(
    query_bank: list[QuerySample],
    train_fraction: float = 0.5,
) -> tuple[dict[tuple[int, int], list[QuerySample]], dict[tuple[int, int], list[QuerySample]]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction!r}.")

    grouped: dict[tuple[int, int], list[QuerySample]] = defaultdict(list)
    for sample in query_bank:
        grouped[(sample.layer, sample.head)].append(sample)

    train: dict[tuple[int, int], list[QuerySample]] = {}
    eval_: dict[tuple[int, int], list[QuerySample]] = {}
    for group_key, samples in grouped.items():
        ordered = sorted(samples, key=lambda sample: sample.token_index)
        train_samples: list[QuerySample] = []
        eval_samples: list[QuerySample] = []
        accumulator = 0.0
        for sample in ordered:
            accumulator += train_fraction
            if accumulator >= 1.0:
                train_samples.append(sample)
                accumulator -= 1.0
            else:
                eval_samples.append(sample)

        if not train_samples and eval_samples:
            train_samples.append(eval_samples.pop(0))
        if not eval_samples and train_samples:
            eval_samples.append(train_samples.pop(-1))

        train[group_key] = train_samples
        eval_[group_key] = eval_samples
    return train, eval_


def _selected_keys_fingerprint(indices: list[int]) -> str:
    payload = ",".join(str(index) for index in indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _build_query_tensor(samples: list[QuerySample]):
    import torch

    return torch.tensor([sample.raw_query_vector for sample in samples], dtype=torch.float32)


def _fit_nonnegative_scale_matrix(design, target):
    import torch

    solution = torch.linalg.lstsq(design, target.unsqueeze(1), driver="gels").solution.squeeze(1)
    return solution.clamp_min(1e-12)


def _fit_nonnegative_ridge_scale_matrix(design, target, regularization_strength: float):
    import torch

    gram = design.T @ design
    rhs = design.T @ target
    diag_mean = torch.clamp_min(torch.diagonal(gram).mean(), 1e-12)
    ridge = torch.eye(gram.shape[0], dtype=design.dtype, device=design.device)
    stabilized_gram = gram + ridge * (diag_mean * regularization_strength)
    solution = torch.linalg.solve(stabilized_gram, rhs)
    return solution.clamp_min(1e-12)


def _mean_abs_rel_error(target, pred) -> float:
    import torch

    rel = torch.abs(pred - target) / torch.clamp_min(target.abs(), 1e-12)
    return float(rel.mean().item())


def _mean_abs_log_error(target_log, pred_log) -> float:
    import torch

    return float(torch.abs(pred_log - target_log).mean().item())


def fit_beta_for_selected_keys(
    sample_id: str,
    boundary_id: str,
    selection_result: KeySelectionResult,
    query_bank: list[QuerySample],
    boundary_keys: dict[tuple[int, int], list[list[float]]],
    solver: str = "clamped_lstsq",
    regularization_strength: float = 0.0,
    train_fraction: float = 0.5,
) -> BetaFitResult:
    import torch

    train_groups, eval_groups = split_query_bank_train_eval(query_bank, train_fraction=train_fraction)
    result_groups: list[BetaFitGroupResult] = []
    overall_start = time.perf_counter()

    for group in selection_result.groups:
        group_key = (group.layer, group.head)
        train_samples = train_groups.get(group_key, [])
        eval_samples = eval_groups.get(group_key, [])
        keys = boundary_keys.get(group_key, [])
        if not train_samples or not eval_samples or not keys:
            continue

        group_start = time.perf_counter()
        selected_indices = group.selected_indices
        full_key_tensor = torch.tensor(keys, dtype=torch.float32)
        selected_key_tensor = full_key_tensor[selected_indices]
        train_query_tensor = _build_query_tensor(train_samples)
        eval_query_tensor = _build_query_tensor(eval_samples)

        train_full_logits = (train_query_tensor @ full_key_tensor.T) / math.sqrt(train_query_tensor.shape[1])
        train_selected_logits = (train_query_tensor @ selected_key_tensor.T) / math.sqrt(train_query_tensor.shape[1])
        train_reference_max = train_full_logits.max(dim=1, keepdim=True).values
        train_target_mass = torch.exp(train_full_logits - train_reference_max).sum(dim=1)
        train_design = torch.exp(train_selected_logits - train_reference_max)
        train_pre_mass = train_design.sum(dim=1)
        train_target_log = torch.log(torch.clamp_min(train_target_mass, 1e-12))

        eval_full_logits = (eval_query_tensor @ full_key_tensor.T) / math.sqrt(eval_query_tensor.shape[1])
        eval_selected_logits = (eval_query_tensor @ selected_key_tensor.T) / math.sqrt(eval_query_tensor.shape[1])
        eval_reference_max = eval_full_logits.max(dim=1, keepdim=True).values
        eval_target_mass = torch.exp(eval_full_logits - eval_reference_max).sum(dim=1)
        eval_design = torch.exp(eval_selected_logits - eval_reference_max)
        eval_pre_mass = eval_design.sum(dim=1)
        eval_target_log = torch.log(torch.clamp_min(eval_target_mass, 1e-12))

        if solver == "clamped_lstsq":
            scale = _fit_nonnegative_scale_matrix(train_design, train_target_mass)
        elif solver == "clamped_ridge":
            if regularization_strength <= 0.0:
                raise ValueError("Regularized beta fitting requires a positive regularization strength.")
            scale = _fit_nonnegative_ridge_scale_matrix(
                train_design,
                train_target_mass,
                regularization_strength=regularization_strength,
            )
        else:
            raise ValueError(f"Unsupported beta solver {solver!r}.")
        beta = torch.log(scale)

        train_post_mass = train_design @ scale
        train_post_log = torch.log(torch.clamp_min(train_post_mass, 1e-12))
        eval_post_mass = eval_design @ scale
        eval_post_log = torch.log(torch.clamp_min(eval_post_mass, 1e-12))
        eval_pre_log = torch.log(torch.clamp_min(eval_pre_mass, 1e-12))

        design_rank = int(torch.linalg.matrix_rank(train_design).item())
        try:
            singular_values = torch.linalg.svdvals(train_design)
            condition_number = float((singular_values.max() / torch.clamp_min(singular_values.min(), 1e-12)).item())
        except Exception:
            condition_number = None

        flags = []
        if train_design.shape[0] < train_design.shape[1]:
            flags.append("underdetermined")
        if design_rank < min(train_design.shape):
            flags.append("rank_deficient")
        if torch.isnan(beta).any() or torch.isinf(beta).any():
            flags.append("beta_non_finite")

        result_groups.append(
            BetaFitGroupResult(
                layer=group.layer,
                head=group.head,
                selected_keys_fingerprint=_selected_keys_fingerprint(selected_indices),
                selected_key_count=len(selected_indices),
                train_query_count=len(train_samples),
                eval_query_count=len(eval_samples),
                design_rank=design_rank,
                condition_number=condition_number,
                underdetermined=train_design.shape[0] < train_design.shape[1],
                pre_train_mean_abs_rel_error=round(_mean_abs_rel_error(train_target_mass, train_pre_mass), 6),
                post_train_mean_abs_rel_error=round(_mean_abs_rel_error(train_target_mass, train_post_mass), 6),
                pre_eval_mean_abs_rel_error=round(_mean_abs_rel_error(eval_target_mass, eval_pre_mass), 6),
                post_eval_mean_abs_rel_error=round(_mean_abs_rel_error(eval_target_mass, eval_post_mass), 6),
                pre_eval_mean_abs_log_error=round(_mean_abs_log_error(eval_target_log, eval_pre_log), 6),
                post_eval_mean_abs_log_error=round(_mean_abs_log_error(eval_target_log, eval_post_log), 6),
                beta_min=round(float(beta.min().item()), 6),
                beta_max=round(float(beta.max().item()), 6),
                beta_mean=round(float(beta.mean().item()), 6),
                runtime_seconds=round(time.perf_counter() - group_start, 6),
                degeneracy_flags=flags,
            )
        )

    runtime_seconds = round(time.perf_counter() - overall_start, 6)
    group_count = len(result_groups)
    if group_count == 0:
        raise ValueError("No beta-fit groups were produced.")
    aggregate_pre_rel = round(sum(group.pre_eval_mean_abs_rel_error for group in result_groups) / group_count, 6)
    aggregate_post_rel = round(sum(group.post_eval_mean_abs_rel_error for group in result_groups) / group_count, 6)
    aggregate_pre_log = round(sum(group.pre_eval_mean_abs_log_error for group in result_groups) / group_count, 6)
    aggregate_post_log = round(sum(group.post_eval_mean_abs_log_error for group in result_groups) / group_count, 6)
    aggregate_post_over_pre_rel = round(aggregate_post_rel / max(aggregate_pre_rel, 1e-12), 6)
    aggregate_post_over_pre_log = round(aggregate_post_log / max(aggregate_pre_log, 1e-12), 6)
    improved_eval_rel_group_count = sum(
        group.post_eval_mean_abs_rel_error <= group.pre_eval_mean_abs_rel_error for group in result_groups
    )
    improved_eval_log_group_count = sum(
        group.post_eval_mean_abs_log_error <= group.pre_eval_mean_abs_log_error for group in result_groups
    )
    underdetermined_group_count = sum(group.underdetermined for group in result_groups)
    rank_deficient_group_count = sum("rank_deficient" in group.degeneracy_flags for group in result_groups)
    condition_numbers = [group.condition_number for group in result_groups if group.condition_number is not None]
    median_condition_number = round(statistics.median(condition_numbers), 6) if condition_numbers else None
    max_condition_number = round(max(condition_numbers), 6) if condition_numbers else None
    return BetaFitResult(
        sample_id=sample_id,
        boundary_id=boundary_id,
        source=selection_result.source,
        solver=solver,
        regularization_strength=round(regularization_strength, 9),
        train_fraction=round(train_fraction, 6),
        runtime_seconds=runtime_seconds,
        group_count=group_count,
        aggregate_pre_eval_mean_abs_rel_error=aggregate_pre_rel,
        aggregate_post_eval_mean_abs_rel_error=aggregate_post_rel,
        aggregate_post_over_pre_eval_rel_error_ratio=aggregate_post_over_pre_rel,
        aggregate_pre_eval_mean_abs_log_error=aggregate_pre_log,
        aggregate_post_eval_mean_abs_log_error=aggregate_post_log,
        aggregate_post_over_pre_eval_log_error_ratio=aggregate_post_over_pre_log,
        improved_eval_rel_group_count=improved_eval_rel_group_count,
        improved_eval_log_group_count=improved_eval_log_group_count,
        underdetermined_group_count=underdetermined_group_count,
        rank_deficient_group_count=rank_deficient_group_count,
        median_condition_number=median_condition_number,
        max_condition_number=max_condition_number,
        groups=result_groups,
    )


def compare_beta_fit_results(
    sketch_result: BetaFitResult,
    control_result: BetaFitResult,
) -> BetaFitComparison:
    control_groups = {(group.layer, group.head): group for group in control_result.groups}
    deltas = []
    for group in sketch_result.groups:
        control_group = control_groups.get((group.layer, group.head))
        if control_group is None:
            continue
        deltas.append(
            {
                "layer": group.layer,
                "head": group.head,
                "sketch_post_eval_mean_abs_rel_error": group.post_eval_mean_abs_rel_error,
                "control_post_eval_mean_abs_rel_error": control_group.post_eval_mean_abs_rel_error,
                "sketch_minus_control_rel_error": round(
                    group.post_eval_mean_abs_rel_error - control_group.post_eval_mean_abs_rel_error,
                    6,
                ),
                "sketch_post_eval_mean_abs_log_error": group.post_eval_mean_abs_log_error,
                "control_post_eval_mean_abs_log_error": control_group.post_eval_mean_abs_log_error,
                "sketch_minus_control_log_error": round(
                    group.post_eval_mean_abs_log_error - control_group.post_eval_mean_abs_log_error,
                    6,
                ),
            }
        )

    def _winner(left: float, right: float, left_name: str, right_name: str) -> str:
        if abs(left - right) < 1e-9:
            return "tie"
        return left_name if left < right else right_name

    return BetaFitComparison(
        sample_id=sketch_result.sample_id,
        boundary_id=sketch_result.boundary_id,
        sketch_source=sketch_result.source,
        control_source=control_result.source,
        sketch_post_eval_mean_abs_rel_error=sketch_result.aggregate_post_eval_mean_abs_rel_error,
        control_post_eval_mean_abs_rel_error=control_result.aggregate_post_eval_mean_abs_rel_error,
        sketch_post_eval_mean_abs_log_error=sketch_result.aggregate_post_eval_mean_abs_log_error,
        control_post_eval_mean_abs_log_error=control_result.aggregate_post_eval_mean_abs_log_error,
        relative_error_winner=_winner(
            sketch_result.aggregate_post_eval_mean_abs_rel_error,
            control_result.aggregate_post_eval_mean_abs_rel_error,
            sketch_result.source,
            control_result.source,
        ),
        log_error_winner=_winner(
            sketch_result.aggregate_post_eval_mean_abs_log_error,
            control_result.aggregate_post_eval_mean_abs_log_error,
            sketch_result.source,
            control_result.source,
        ),
        per_group_deltas=deltas,
    )


def write_beta_fit_result(result: BetaFitResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_serializable(), indent=2) + "\n", encoding="utf-8")
    return output_path
