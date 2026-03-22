from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from kv_compaction_qwen35_clean.data_types import FeatureObservation, SmokeTestConfig


@dataclass
class PrototypeEntry:
    prototype_id: str
    layer: int
    head: int
    center_query_projection: list[float]
    center_output_projection: list[float]
    avg_prefix_mass_share: float
    avg_raw_prefix_mass: float
    weight: float
    update_count: int
    last_token_index: int
    center_query_norm: float = field(default=0.0, repr=False)
    center_output_norm: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.center_query_norm = round(_norm(self.center_query_projection), 6)
        self.center_output_norm = round(_norm(self.center_output_projection), 6)


@dataclass
class PrototypeBankState:
    sketch_kind: str
    update_rule: str
    similarity_metric: str
    merge_threshold: float
    forgetting_factor: float
    min_prefix_mass: float
    feature_granularity: str
    projection_source: str
    query_projection_dim: int
    output_projection_dim: int
    mass_measure: str
    auxiliary_mass_metric: str
    tap_point: str
    next_prototype_index: int
    entries: list[PrototypeEntry]
    _pair_to_indices: dict[tuple[int, int], list[int]] = field(default_factory=dict, repr=False)
    _entry_decay_steps: list[int] = field(default_factory=list, repr=False)
    _decay_step: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_internal_state()

    def _rebuild_internal_state(self) -> None:
        self._pair_to_indices = {}
        for index, entry in enumerate(self.entries):
            self._pair_to_indices.setdefault((entry.layer, entry.head), []).append(index)
        if len(self._entry_decay_steps) != len(self.entries):
            self._entry_decay_steps = [self._decay_step for _ in self.entries]

    def to_serializable(self) -> dict[str, object]:
        return {
            "sketch_kind": self.sketch_kind,
            "update_rule": self.update_rule,
            "similarity_metric": self.similarity_metric,
            "merge_threshold": self.merge_threshold,
            "forgetting_factor": self.forgetting_factor,
            "min_prefix_mass": self.min_prefix_mass,
            "feature_granularity": self.feature_granularity,
            "projection_source": self.projection_source,
            "query_projection_dim": self.query_projection_dim,
            "output_projection_dim": self.output_projection_dim,
            "mass_measure": self.mass_measure,
            "auxiliary_mass_metric": self.auxiliary_mass_metric,
            "tap_point": self.tap_point,
            "next_prototype_index": self.next_prototype_index,
            "entries": [
                {
                    "prototype_id": entry.prototype_id,
                    "layer": entry.layer,
                    "head": entry.head,
                    "center_query_projection": list(entry.center_query_projection),
                    "center_output_projection": list(entry.center_output_projection),
                    "avg_prefix_mass_share": entry.avg_prefix_mass_share,
                    "avg_raw_prefix_mass": entry.avg_raw_prefix_mass,
                    "weight": entry.weight,
                    "update_count": entry.update_count,
                    "last_token_index": entry.last_token_index,
                }
                for entry in self.entries
            ],
        }


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: list[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _cosine_similarity_with_norms(
    left: list[float],
    right: list[float],
    *,
    left_norm: float,
    right_norm: float,
) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return _dot(left, right) / (left_norm * right_norm)


def _blended_similarity_from_values(
    query_projection: list[float],
    prefix_mass_share: float,
    output_projection: list[float],
    query_norm: float,
    output_norm: float,
    entry: PrototypeEntry,
) -> float:
    query_similarity = _cosine_similarity_with_norms(
        query_projection,
        entry.center_query_projection,
        left_norm=query_norm,
        right_norm=entry.center_query_norm,
    )
    output_similarity = _cosine_similarity_with_norms(
        output_projection,
        entry.center_output_projection,
        left_norm=output_norm,
        right_norm=entry.center_output_norm,
    )
    mass_similarity = 1.0 - abs(prefix_mass_share - entry.avg_prefix_mass_share)
    return (query_similarity + output_similarity + mass_similarity) / 3.0


def _mass_gate(prefix_mass_share: float, min_prefix_mass: float) -> float:
    if prefix_mass_share < min_prefix_mass:
        return 0.0
    return min(1.0, prefix_mass_share)


def _residual_score_from_values(
    output_projection: list[float],
    output_norm: float,
    entry: PrototypeEntry,
) -> float:
    output_similarity = _cosine_similarity_with_norms(
        output_projection,
        entry.center_output_projection,
        left_norm=output_norm,
        right_norm=entry.center_output_norm,
    )
    return max(0.0, 1.0 - output_similarity)


def _weighted_average(current: list[float], update: list[float], strength: float) -> list[float]:
    base = max(0.0, 1.0 - strength)
    return [round((base * a) + (strength * b), 6) for a, b in zip(current, update)]


def _pair_indices(state: PrototypeBankState, layer: int, head: int) -> list[int]:
    return state._pair_to_indices.get((layer, head), [])


def _entry_decay_factor(state: PrototypeBankState, index: int, config: SmokeTestConfig) -> float:
    delta = state._decay_step - state._entry_decay_steps[index]
    if delta <= 0:
        return 1.0
    return config.sketch.forgetting_factor ** delta


def _materialize_entry_decay(
    state: PrototypeBankState,
    index: int,
    config: SmokeTestConfig,
) -> PrototypeEntry:
    delta = state._decay_step - state._entry_decay_steps[index]
    entry = state.entries[index]
    if delta <= 0:
        return entry
    factor = config.sketch.forgetting_factor ** delta
    entry.weight = round(entry.weight * factor, 6)
    entry.avg_prefix_mass_share = round(entry.avg_prefix_mass_share * factor, 6)
    entry.avg_raw_prefix_mass = round(entry.avg_raw_prefix_mass * factor, 6)
    state._entry_decay_steps[index] = state._decay_step
    return entry


def _append_entry(state: PrototypeBankState, entry: PrototypeEntry) -> None:
    state.entries.append(entry)
    state._entry_decay_steps.append(state._decay_step)
    state._pair_to_indices.setdefault((entry.layer, entry.head), []).append(len(state.entries) - 1)


def _replace_entry(state: PrototypeBankState, index: int, entry: PrototypeEntry) -> None:
    previous = state.entries[index]
    previous_pair = (previous.layer, previous.head)
    current_pair = (entry.layer, entry.head)
    if previous_pair != current_pair:
        state._pair_to_indices[previous_pair] = [
            candidate for candidate in state._pair_to_indices.get(previous_pair, []) if candidate != index
        ]
        if not state._pair_to_indices[previous_pair]:
            del state._pair_to_indices[previous_pair]
        state._pair_to_indices.setdefault(current_pair, []).append(index)
    state.entries[index] = entry
    state._entry_decay_steps[index] = state._decay_step


def _refresh_entry_norms(entry: PrototypeEntry) -> None:
    entry.center_query_norm = round(_norm(entry.center_query_projection), 6)
    entry.center_output_norm = round(_norm(entry.center_output_projection), 6)


def _replacement_index(state: PrototypeBankState, config: SmokeTestConfig) -> int:
    candidate_indices = list(range(len(state.entries)))
    return min(
        candidate_indices,
        key=lambda index: _materialize_entry_decay(state, index, config).weight,
    )


def initialize_state(config: SmokeTestConfig) -> PrototypeBankState:
    return PrototypeBankState(
        sketch_kind=config.sketch.kind,
        update_rule=config.sketch.update_rule,
        similarity_metric=config.sketch.similarity_metric,
        merge_threshold=config.sketch.merge_threshold,
        forgetting_factor=config.sketch.forgetting_factor,
        min_prefix_mass=config.sketch.min_prefix_mass,
        feature_granularity=config.feature_schema.granularity,
        projection_source=config.feature_schema.projection_source,
        query_projection_dim=config.feature_schema.query_projection_dim,
        output_projection_dim=config.feature_schema.output_projection_dim,
        mass_measure=config.feature_schema.mass_measure,
        auxiliary_mass_metric=config.feature_schema.auxiliary_mass_metric,
        tap_point=config.feature_schema.tap_point,
        next_prototype_index=0,
        entries=[],
    )


def materialize_state(state: PrototypeBankState, config: SmokeTestConfig) -> PrototypeBankState:
    for index in range(len(state.entries)):
        _materialize_entry_decay(state, index, config)
    return state


def apply_observation(
    state: PrototypeBankState,
    observation: FeatureObservation,
    config: SmokeTestConfig,
) -> None:
    state._decay_step += 1

    gate = _mass_gate(observation.prefix_mass_share, config.sketch.min_prefix_mass)
    if gate == 0.0:
        return

    query_norm = round(_norm(observation.query_projection), 6)
    output_norm = round(_norm(observation.output_projection), 6)

    if not state.entries:
        _append_entry(
            state,
            PrototypeEntry(
                prototype_id=f"p{state.next_prototype_index}",
                layer=observation.layer,
                head=observation.head,
                center_query_projection=[round(value, 6) for value in observation.query_projection],
                center_output_projection=[round(value, 6) for value in observation.output_projection],
                avg_prefix_mass_share=round(observation.prefix_mass_share, 6),
                avg_raw_prefix_mass=round(observation.raw_prefix_mass, 6),
                weight=round(gate, 6),
                update_count=1,
                last_token_index=observation.token_index,
            ),
        )
        state.next_prototype_index += 1
        return

    matching_indices = _pair_indices(state, observation.layer, observation.head)
    best_index = None
    best_similarity = -1.0
    for index in matching_indices:
        entry = _materialize_entry_decay(state, index, config)
        similarity = _blended_similarity_from_values(
            observation.query_projection,
            observation.prefix_mass_share,
            observation.output_projection,
            query_norm,
            output_norm,
            entry,
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = index
    best_entry = state.entries[best_index] if best_index is not None else None

    if best_entry is None or best_similarity < config.sketch.merge_threshold:
        candidate = PrototypeEntry(
            prototype_id=f"p{state.next_prototype_index}",
            layer=observation.layer,
            head=observation.head,
            center_query_projection=[round(value, 6) for value in observation.query_projection],
            center_output_projection=[round(value, 6) for value in observation.output_projection],
            avg_prefix_mass_share=round(observation.prefix_mass_share, 6),
            avg_raw_prefix_mass=round(observation.raw_prefix_mass, 6),
            weight=round(gate, 6),
            update_count=1,
            last_token_index=observation.token_index,
        )
        state.next_prototype_index += 1

        if len(state.entries) < config.sketch.max_prototypes:
            _append_entry(state, candidate)
        else:
            replace_index = _replacement_index(state, config)
            _replace_entry(state, replace_index, candidate)
        return

    novelty = max(0.0, 1.0 - best_similarity)
    residual = _residual_score_from_values(observation.output_projection, output_norm, best_entry)
    strength = gate * (
        1.0
        + (config.sketch.novelty_weight * novelty)
        + (config.sketch.residual_weight * residual)
    )
    strength = min(1.0, strength)

    best_entry.center_query_projection = _weighted_average(
        best_entry.center_query_projection,
        observation.query_projection,
        strength,
    )
    best_entry.center_output_projection = _weighted_average(
        best_entry.center_output_projection,
        observation.output_projection,
        strength,
    )
    _refresh_entry_norms(best_entry)
    best_entry.avg_prefix_mass_share = round(
        ((1.0 - strength) * best_entry.avg_prefix_mass_share) + (strength * observation.prefix_mass_share),
        6,
    )
    best_entry.avg_raw_prefix_mass = round(
        ((1.0 - strength) * best_entry.avg_raw_prefix_mass) + (strength * observation.raw_prefix_mass),
        6,
    )
    best_entry.weight = round(best_entry.weight + gate, 6)
    best_entry.update_count += 1
    best_entry.last_token_index = observation.token_index


def build_state_from_observations(
    config: SmokeTestConfig,
    observations: list[FeatureObservation],
) -> PrototypeBankState:
    state = initialize_state(config)
    for observation in observations:
        apply_observation(state, observation, config)
    return materialize_state(state, config)


def write_state(state: PrototypeBankState, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(state.to_serializable(), indent=2) + "\n", encoding="utf-8")
    return output_path
