from __future__ import annotations

import math


def resolve_head_budgets(
    *,
    group_keys: list[tuple[int, int]],
    keys_per_head: int,
    head_budget_proportions: dict[tuple[int, int], float] | None = None,
    min_keys_per_head: int = 1,
) -> dict[tuple[int, int], int]:
    ordered_groups = sorted({(int(layer), int(head)) for layer, head in group_keys})
    if not ordered_groups:
        return {}

    base_budget = max(0, int(keys_per_head))
    if head_budget_proportions is None:
        return {group_key: base_budget for group_key in ordered_groups}

    raw_weights = [max(0.0, float(head_budget_proportions.get(group_key, 0.0))) for group_key in ordered_groups]
    total_weight = sum(raw_weights)
    if total_weight <= 0.0:
        raise ValueError(
            "Head-budget schedule does not cover the active layer/head groups. "
            "Refusing to silently fall back to uniform allocation."
        )

    normalized = [weight / total_weight for weight in raw_weights]
    total_budget = base_budget * len(ordered_groups)
    min_budget = max(0, int(min_keys_per_head))

    if total_budget <= 0:
        return {group_key: 0 for group_key in ordered_groups}

    allocations = [0] * len(ordered_groups)
    if min_budget > 0 and total_budget >= len(ordered_groups) * min_budget:
        allocations = [min_budget] * len(ordered_groups)
        remaining = total_budget - sum(allocations)
    else:
        remaining = total_budget

    if remaining > 0:
        desired = [weight * remaining for weight in normalized]
        extra = [int(math.floor(value)) for value in desired]
        allocations = [current + add for current, add in zip(allocations, extra)]
        leftover = remaining - sum(extra)
        if leftover > 0:
            ranked = sorted(
                range(len(ordered_groups)),
                key=lambda index: (desired[index] - extra[index], normalized[index], -index),
                reverse=True,
            )
            for index in ranked[:leftover]:
                allocations[index] += 1

    if sum(allocations) != total_budget:
        raise AssertionError(
            f"Head-budget allocation drifted: expected {total_budget}, got {sum(allocations)}."
        )
    return {group_key: allocation for group_key, allocation in zip(ordered_groups, allocations)}
