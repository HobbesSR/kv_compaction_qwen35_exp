from __future__ import annotations

import json
import time
from dataclasses import replace

from kv_compaction_qwen35_clean.behavioral_eval import (
    _build_base_cache,
    _clear_cuda_memory,
    _feed_tokens_with_cache,
    _run_prompt_path,
    select_prompt_subset,
)
from kv_compaction_qwen35_clean.boundary_collection import (
    collect_teacher_forced_boundary_collection,
    resolve_replay_checkpoint_start,
    select_boundary_biased_capture_indices,
)
from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.context_loader import load_context_sample
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


def _serializable_runs(runs):
    from dataclasses import asdict

    return [asdict(run) for run in runs]


def _checkpoint_for_capture_indices(
    capture_indices: list[int],
    turn_spans: list[tuple[int, int, str, str]],
) -> dict[str, object] | None:
    if not capture_indices:
        return None
    checkpoint_start = resolve_replay_checkpoint_start(capture_indices, turn_spans)
    first_capture_index = min(int(index) for index in capture_indices)
    for start, end, turn_id, speaker in turn_spans:
        if int(start) == checkpoint_start:
            return {
                "turn_id": str(turn_id),
                "speaker": str(speaker),
                "start": int(start),
                "end": int(end),
                "first_capture_index": int(first_capture_index),
            }
    return None


def _run_sketch_gate(
    *,
    model,
    tokenizer,
    collection_config,
    sample,
    bundle,
    token_ids: list[int],
    probe_layers: tuple[int, ...],
    probe_heads: tuple[int, ...],
    target_layer_heads: tuple[tuple[int, int], ...],
    prompts,
):
    state = build_state_from_observations(collection_config, bundle.harvest.observations)
    sketch_source = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, collection_config)
    _, sketch_layers = build_path_runtime(
        sample.sample_id,
        sample.boundary.boundary_id,
        sketch_source.source,
        8,
        bundle,
        sketch_source,
        target_layers=probe_layers,
        target_heads=probe_heads,
        target_layer_heads=target_layer_heads,
        compute_device=collection_config.model.device,
        key_selection_method=collection_config.compaction.key_selection,
    )
    prefix_token_ids = token_ids[: sample.boundary.prefix_token_count]
    tail_token_ids = token_ids[sample.boundary.prefix_token_count :]
    sketch_base_cache, sketch_base_position = _build_base_cache(
        model=model,
        device=collection_config.model.device,
        prefix_cache=bundle.runtime_cache,
        tail_token_ids=tail_token_ids,
        prefix_token_count=sample.boundary.prefix_token_count,
        compacted_layers=sketch_layers,
    )
    runs, runtime_seconds = _run_prompt_path(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        prefix_token_ids=prefix_token_ids,
        tail_token_ids=tail_token_ids,
        device=collection_config.model.device,
        compacted_layers=sketch_layers,
        prefix_token_count=sample.boundary.prefix_token_count,
        enable_thinking=collection_config.model.enable_thinking,
        max_new_tokens=40,
        reference_runs=None,
        base_cache=sketch_base_cache,
        base_position=sketch_base_position,
    )
    bundle.runtime_cache = None
    bundle.boundary_keys = {}
    bundle.boundary_values = {}
    bundle.boundary_projected_values = {}
    bundle.output_targets = {}
    return runs, runtime_seconds


def main() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)
    prompts = select_prompt_subset(
        "qwen35_calibration_v3",
        sample.prompt_family,
        prompt_labels=["qwen35_branch_switch_harness_note"],
    )
    collection_config = replace(config, model=replace(config.model, attn_implementation="eager"))
    model, tokenizer, model_type = load_qwen35_bundle(collection_config)
    try:
        probe_layers = default_probe_layers_for_model(model, model_type)
        probe_heads = default_probe_heads_for_model(model)
        target_layer_heads = tuple((int(layer), int(head)) for layer in probe_layers for head in probe_heads)
        token_ids, turn_spans = materialize_long_context_ids(sample, tokenizer)

        full_start = time.perf_counter()
        full_bundle = collect_teacher_forced_boundary_collection(
            sample,
            collection_config,
            model=model,
            tokenizer=tokenizer,
            probe_layers=probe_layers,
            probe_heads=probe_heads,
            retain_runtime_cache=True,
        )
        full_collection_seconds = time.perf_counter() - full_start
        full_runs, full_runtime_seconds = _run_sketch_gate(
            model=model,
            tokenizer=tokenizer,
            collection_config=collection_config,
            sample=sample,
            bundle=full_bundle,
            token_ids=token_ids,
            probe_layers=probe_layers,
            probe_heads=probe_heads,
            target_layer_heads=target_layer_heads,
            prompts=prompts,
        )
        _clear_cuda_memory()

        replay_capture_indices = select_boundary_biased_capture_indices(
            sample.boundary.prefix_token_count,
            turn_spans,
            lookback_turns=3,
        )
        checkpoint = _checkpoint_for_capture_indices(replay_capture_indices, turn_spans)
        if checkpoint is None:
            raise ValueError("Unable to resolve replay checkpoint from boundary-biased capture indices.")

        checkpoint_cache = None
        for start, end, _turn_id, _speaker in turn_spans:
            start = int(start)
            end = min(int(end), int(checkpoint["start"]))
            if end <= start:
                continue
            checkpoint_cache, _ = _feed_tokens_with_cache(
                model,
                token_ids[start:end],
                device=collection_config.model.device,
                past_key_values=checkpoint_cache,
                start_position=start,
            )

        replay_start = time.perf_counter()
        replay_bundle = collect_teacher_forced_boundary_collection(
            sample,
            collection_config,
            model=model,
            tokenizer=tokenizer,
            probe_layers=probe_layers,
            probe_heads=probe_heads,
            capture_indices=replay_capture_indices,
            retain_runtime_cache=True,
            initial_past_key_values=checkpoint_cache,
            replay_start_position=int(checkpoint["start"]),
        )
        replay_collection_seconds = time.perf_counter() - replay_start
        replay_runs, replay_runtime_seconds = _run_sketch_gate(
            model=model,
            tokenizer=tokenizer,
            collection_config=collection_config,
            sample=sample,
            bundle=replay_bundle,
            token_ids=token_ids,
            probe_layers=probe_layers,
            probe_heads=probe_heads,
            target_layer_heads=target_layer_heads,
            prompts=prompts,
        )

        print(
            json.dumps(
                {
                    "prompt_label": prompts[0].label,
                    "full_collection": {
                        "collection_seconds": round(full_collection_seconds, 6),
                        "sketch_runtime_seconds": round(full_runtime_seconds, 6),
                        "runs": _serializable_runs(full_runs),
                    },
                    "replay_collection": {
                        "capture_indices": replay_capture_indices,
                        "checkpoint": checkpoint,
                        "replay_window_tokens": sample.boundary.prefix_token_count - int(checkpoint["start"]),
                        "collection_seconds": round(replay_collection_seconds, 6),
                        "sketch_runtime_seconds": round(replay_runtime_seconds, 6),
                        "runs": _serializable_runs(replay_runs),
                    },
                },
                indent=2,
            )
        )
    finally:
        unload_qwen35_bundle(model)


if __name__ == "__main__":
    main()
