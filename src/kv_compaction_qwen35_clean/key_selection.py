from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

from kv_compaction_qwen35_clean.data_types import (
    KeySelectionComparison,
    KeySelectionResult,
    QueryCoreset,
    QuerySample,
    SelectedKeyGroup,
)
from kv_compaction_qwen35_clean.head_budget import resolve_head_budgets


KEY_SELECTION_METHODS = (
    "highest_attention",
    "omp",
)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def match_coreset_to_query_samples(
    query_source: QueryCoreset,
    query_bank: list[QuerySample],
) -> list[tuple[object, QuerySample]]:
    samples_by_group: dict[tuple[int, int], list[QuerySample]] = defaultdict(list)
    entries_by_group: dict[tuple[int, int], list[object]] = defaultdict(list)
    for sample in query_bank:
        samples_by_group[(sample.layer, sample.head)].append(sample)
    for entry in query_source.selected_entries:
        entries_by_group[(entry.layer, entry.head)].append(entry)

    matches = []
    for group_key, entries in entries_by_group.items():
        candidates = samples_by_group.get(group_key, [])
        if not candidates:
            continue
        for entry in entries:
            best_sample = max(
                candidates,
                key=lambda sample: (
                    _cosine_similarity(entry.query_projection, sample.query_projection),
                    sample.prefix_mass_share,
                    sample.token_index,
                ),
            )
            matches.append((entry, best_sample))
    return matches


def select_keys_with_highest_attention(
    sample_id: str,
    boundary_id: str,
    source: str,
    matched_queries: list[tuple[object, QuerySample]],
    boundary_keys: dict[tuple[int, int], object],
    keys_per_head: int,
) -> KeySelectionResult:
    return select_keys(
        sample_id,
        boundary_id,
        source,
        matched_queries,
        boundary_keys,
        keys_per_head,
        selection_method="highest_attention",
    )


def select_keys(
    sample_id: str,
    boundary_id: str,
    source: str,
    matched_queries: list[tuple[object, QuerySample]],
    boundary_keys: dict[tuple[int, int], object],
    keys_per_head: int,
    selection_method: str = "highest_attention",
    head_budget_proportions: dict[tuple[int, int], float] | None = None,
    min_keys_per_head: int = 1,
) -> KeySelectionResult:
    import torch

    if selection_method not in KEY_SELECTION_METHODS:
        raise ValueError(
            f"Unsupported key selection method {selection_method!r}. "
            f"Expected one of {KEY_SELECTION_METHODS}."
        )

    grouped_queries: dict[tuple[int, int], list[tuple[object, QuerySample]]] = defaultdict(list)
    for entry, sample in matched_queries:
        grouped_queries[(sample.layer, sample.head)].append((entry, sample))

    active_groups = [
        group_key
        for group_key in sorted(grouped_queries)
        if group_key in boundary_keys and boundary_keys[group_key] is not None
    ]
    budgets_by_group = resolve_head_budgets(
        group_keys=active_groups,
        keys_per_head=keys_per_head,
        head_budget_proportions=head_budget_proportions,
        min_keys_per_head=min_keys_per_head,
    )

    groups: list[SelectedKeyGroup] = []
    for group_key in sorted(grouped_queries):
        layer, head = group_key
        keys = boundary_keys.get(group_key)
        if keys is None or (not isinstance(keys, torch.Tensor) and not keys):
            continue

        if isinstance(keys, torch.Tensor):
            key_tensor = keys.to(dtype=torch.float32)
        else:
            key_tensor = torch.tensor(keys, dtype=torch.float32)

        queries = grouped_queries[group_key]
        budget = max(0, int(budgets_by_group.get(group_key, keys_per_head)))
        if budget <= 0:
            continue

        query_tensor = torch.tensor(
            [sample.raw_query_vector for _, sample in queries],
            dtype=torch.float32,
            device=key_tensor.device,
        )
        logits = (query_tensor @ key_tensor.T) / math.sqrt(query_tensor.shape[1])
        weights = torch.softmax(logits, dim=1)
        entry_weights = torch.tensor(
            [float(entry.weight) for entry, _ in queries],
            dtype=torch.float32,
            device=key_tensor.device,
        ).unsqueeze(1)
        scores = (weights * entry_weights).sum(dim=0)
        total_weight = float(entry_weights.sum().item())

        if selection_method == "highest_attention":
            top_k = min(budget, scores.numel())
            top_scores, top_indices = torch.topk(scores, k=top_k, largest=True)
            selected_indices = top_indices.tolist()
            selected_scores = [round(float(score), 6) for score in top_scores.tolist()]
        else:
            selected_indices, selected_scores = _select_keys_with_omp(
                key_tensor=key_tensor,
                query_tensor=query_tensor,
                entry_weights=entry_weights.squeeze(1),
                selection_budget=budget,
            )

        groups.append(
            SelectedKeyGroup(
                layer=layer,
                head=head,
                selected_indices=selected_indices,
                selected_scores=selected_scores,
                query_count=len(queries),
                total_query_weight=round(total_weight, 6),
            )
        )

    return KeySelectionResult(
        sample_id=sample_id,
        boundary_id=boundary_id,
        source=source,
        keys_per_head=keys_per_head,
        groups=groups,
    )


def _select_keys_with_omp(
    *,
    key_tensor,
    query_tensor,
    entry_weights,
    selection_budget: int,
) -> tuple[list[int], list[float]]:
    import torch

    if selection_budget <= 0 or key_tensor.numel() == 0 or query_tensor.numel() == 0:
        return [], []

    inv_sqrt_d = 1.0 / math.sqrt(max(int(query_tensor.shape[1]), 1))
    logits = (query_tensor @ key_tensor.T) * inv_sqrt_d
    reference_max = logits.max(dim=1, keepdim=True).values
    exp_scores = torch.exp(logits - reference_max)
    target = exp_scores.sum(dim=1)
    row_weights = torch.sqrt(torch.clamp_min(entry_weights.to(dtype=torch.float32), 0.0))
    weighted_design = exp_scores * row_weights.unsqueeze(1)
    weighted_target = target * row_weights

    selected_indices: list[int] = []
    selected_scores: list[float] = []
    mask = torch.zeros(weighted_design.shape[1], dtype=torch.bool, device=weighted_design.device)
    current = torch.zeros_like(weighted_target)

    for _ in range(min(int(selection_budget), int(weighted_design.shape[1]))):
        residual = weighted_target - current
        corr = torch.matmul(weighted_design.T, residual)
        corr[mask] = -float("inf")
        index = int(torch.argmax(corr).item())
        if not math.isfinite(float(corr[index].item())):
            break
        selected_indices.append(index)
        selected_scores.append(round(float(corr[index].item()), 6))
        mask[index] = True
        selected_design = weighted_design[:, selected_indices]
        scale = torch.linalg.lstsq(selected_design, weighted_target.unsqueeze(1), driver="gels").solution.squeeze(1)
        scale = scale.clamp_min(1e-12)
        current = selected_design @ scale

    return selected_indices, selected_scores


def compare_key_selection_results(
    sketch: KeySelectionResult,
    control: KeySelectionResult,
) -> KeySelectionComparison:
    control_groups = {(group.layer, group.head): group for group in control.groups}
    overlap_by_group = []

    for group in sketch.groups:
        control_group = control_groups.get((group.layer, group.head))
        if control_group is None:
            continue
        left = set(group.selected_indices)
        right = set(control_group.selected_indices)
        intersection = left & right
        union = left | right
        jaccard = (len(intersection) / len(union)) if union else 1.0
        overlap_by_group.append(
            {
                "layer": group.layer,
                "head": group.head,
                "sketch_indices": sorted(left),
                "control_indices": sorted(right),
                "intersection": sorted(intersection),
                "jaccard": round(jaccard, 6),
            }
        )

    return KeySelectionComparison(
        sample_id=sketch.sample_id,
        boundary_id=sketch.boundary_id,
        sketch_source=sketch.source,
        control_source=control.source,
        overlap_by_group=overlap_by_group,
    )


def write_key_selection_result(result: KeySelectionResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_serializable(), indent=2) + "\n", encoding="utf-8")
    return output_path
