from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from kv_compaction_qwen35_clean.behavioral_eval import _build_base_cache, _continue_with_prompt
from kv_compaction_qwen35_clean.boundary_collection import collect_teacher_forced_boundary_collection
from kv_compaction_qwen35_clean.coreset import extract_query_coreset
from kv_compaction_qwen35_clean.model_runtime import (
    default_probe_heads_for_model,
    default_probe_layers_for_model,
    load_qwen35_bundle,
    materialize_long_context_ids,
    unload_qwen35_bundle,
)
from kv_compaction_qwen35_clean.prototype_bank import build_state_from_observations
from kv_compaction_qwen35_clean.runtime_compaction import build_path_runtime


@dataclass
class ServiceDemoSummary:
    sample_id: str
    boundary_id: str
    keys_per_head: int
    compacted_head_count: int
    effective_compact_tokens: int
    prefix_token_count: int
    preserved_tail_tokens: int
    capture_token_count: int
    monitored_observation_count: int
    monitored_query_sample_count: int

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


def write_service_demo_summary(summary: ServiceDemoSummary, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.to_serializable(), indent=2) + "\n", encoding="utf-8")
    return output_path


def format_progress_event(event: dict[str, object]) -> str:
    stage = str(event.get("stage", "progress"))
    processed = int(event.get("processed_token_count", 0))
    prefix = max(1, int(event.get("prefix_token_count", 1)))
    pct = round((processed / prefix) * 100.0, 1)
    if stage == "capture":
        return (
            f"[capture] {processed}/{prefix} tokens ({pct}%) "
            f"obs={int(event.get('monitored_observation_count', 0))} "
            f"queries={int(event.get('monitored_query_sample_count', 0))}"
        )
    return f"[prefill] {processed}/{prefix} tokens ({pct}%)"


class ServiceDemoSession:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        device: str,
        prefix_token_count: int,
        prefix_token_ids: list[int],
        tail_token_ids: list[int],
        compacted_layers,
        enable_thinking: bool | None,
        full_base_cache,
        full_base_position: int,
        compact_base_cache,
        compact_base_position: int,
        summary: ServiceDemoSummary,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prefix_token_count = prefix_token_count
        self.prefix_token_ids = prefix_token_ids
        self.tail_token_ids = tail_token_ids
        self.compacted_layers = compacted_layers
        self.enable_thinking = enable_thinking
        self.full_base_cache = full_base_cache
        self.full_base_position = full_base_position
        self.compact_base_cache = compact_base_cache
        self.compact_base_position = compact_base_position
        self.summary = summary

    def answer(self, prompt_text: str, *, compacted: bool = True, max_new_tokens: int = 40) -> tuple[str, float]:
        return _continue_with_prompt(
            model=self.model,
            tokenizer=self.tokenizer,
            prefix_token_ids=self.prefix_token_ids,
            tail_token_ids=self.tail_token_ids,
            prompt=type("Prompt", (), {"prompt_text": prompt_text})(),
            device=self.device,
            compacted_layers=self.compacted_layers if compacted else None,
            prefix_token_count=self.prefix_token_count,
            enable_thinking=self.enable_thinking,
            max_new_tokens=max_new_tokens,
            base_cache=self.compact_base_cache if compacted else self.full_base_cache,
            base_position=self.compact_base_position if compacted else self.full_base_position,
        )

    def close(self) -> None:
        unload_qwen35_bundle(self.model)


def build_service_demo_session(
    sample,
    config,
    *,
    keys_per_head: int = 6,
    key_selection_method: str = "highest_attention",
    progress_callback=None,
) -> ServiceDemoSession:
    eager_config = replace(config, model=replace(config.model, attn_implementation="eager"))
    model, tokenizer, model_type = load_qwen35_bundle(eager_config)
    try:
        probe_layers = default_probe_layers_for_model(model, model_type)
        probe_heads = default_probe_heads_for_model(model)
        target_layer_heads = tuple((int(layer), int(head)) for layer in probe_layers for head in probe_heads)
        bundle = collect_teacher_forced_boundary_collection(
            sample,
            eager_config,
            model=model,
            tokenizer=tokenizer,
            probe_layers=probe_layers,
            probe_heads=probe_heads,
            retain_runtime_cache=True,
            progress_callback=progress_callback,
        )
        state = build_state_from_observations(eager_config, bundle.harvest.observations)
        sketch_source = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, eager_config)
        _, compacted_layers = build_path_runtime(
            sample.sample_id,
            sample.boundary.boundary_id,
            sketch_source.source,
            keys_per_head,
            bundle,
            sketch_source,
            target_layers=probe_layers,
            target_heads=probe_heads,
            target_layer_heads=target_layer_heads,
            compute_device=eager_config.model.device,
            key_selection_method=key_selection_method,
        )
        token_ids, _ = materialize_long_context_ids(sample, tokenizer)
        prefix_token_ids = token_ids[: sample.boundary.prefix_token_count]
        tail_token_ids = token_ids[sample.boundary.prefix_token_count :]
        full_base_cache, full_base_position = _build_base_cache(
            model=model,
            device=eager_config.model.device,
            prefix_cache=bundle.runtime_cache,
            tail_token_ids=tail_token_ids,
            prefix_token_count=sample.boundary.prefix_token_count,
            compacted_layers=None,
        )
        compact_base_cache, compact_base_position = _build_base_cache(
            model=model,
            device=eager_config.model.device,
            prefix_cache=bundle.runtime_cache,
            tail_token_ids=tail_token_ids,
            prefix_token_count=sample.boundary.prefix_token_count,
            compacted_layers=compacted_layers,
        )
        compacted_head_count = sum(len(layer_rows) for layer_rows in compacted_layers.values())
        effective_compact_tokens = sum(
            len(runtime.selected_indices)
            for layer_rows in compacted_layers.values()
            for runtime in layer_rows.values()
        )
        summary = ServiceDemoSummary(
            sample_id=sample.sample_id,
            boundary_id=sample.boundary.boundary_id,
            keys_per_head=keys_per_head,
            compacted_head_count=compacted_head_count,
            effective_compact_tokens=effective_compact_tokens,
            prefix_token_count=sample.boundary.prefix_token_count,
            preserved_tail_tokens=sample.boundary.preserved_tail_tokens,
            capture_token_count=len(bundle.capture_token_indices or []),
            monitored_observation_count=int(bundle.monitored_observation_count or bundle.harvest.observation_count),
            monitored_query_sample_count=int(bundle.monitored_query_sample_count or bundle.query_bank.sample_count),
        )
        return ServiceDemoSession(
            model=model,
            tokenizer=tokenizer,
            device=eager_config.model.device,
            prefix_token_count=sample.boundary.prefix_token_count,
            prefix_token_ids=prefix_token_ids,
            tail_token_ids=tail_token_ids,
            compacted_layers=compacted_layers,
            enable_thinking=eager_config.model.enable_thinking,
            full_base_cache=full_base_cache,
            full_base_position=full_base_position,
            compact_base_cache=compact_base_cache,
            compact_base_position=compact_base_position,
            summary=summary,
        )
    except Exception:
        unload_qwen35_bundle(model)
        raise
