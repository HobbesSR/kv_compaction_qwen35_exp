from __future__ import annotations

import json
from dataclasses import replace

from kv_compaction_qwen35_clean.behavioral_eval import (
    _build_base_cache,
    _clear_cuda_memory,
    _feed_tokens_with_cache,
    _run_prompt_path,
    select_prompt_subset,
)
from kv_compaction_qwen35_clean.boundary_collection import collect_teacher_forced_boundary_collection
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

        bundle = collect_teacher_forced_boundary_collection(
            sample,
            collection_config,
            model=model,
            tokenizer=tokenizer,
            probe_layers=probe_layers,
            probe_heads=probe_heads,
            retain_runtime_cache=True,
        )
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
            key_selection_method=config.compaction.key_selection,
        )

        token_ids, turn_spans = materialize_long_context_ids(sample, tokenizer)
        prefix_token_ids = token_ids[: sample.boundary.prefix_token_count]
        tail_token_ids = token_ids[sample.boundary.prefix_token_count :]

        uniform_base_cache, uniform_base_position = _build_base_cache(
            model=model,
            device=collection_config.model.device,
            prefix_cache=bundle.runtime_cache,
            tail_token_ids=tail_token_ids,
            prefix_token_count=sample.boundary.prefix_token_count,
            compacted_layers=sketch_layers,
        )
        uniform_runs, uniform_runtime = _run_prompt_path(
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
            base_cache=uniform_base_cache,
            base_position=uniform_base_position,
        )

        bundle.runtime_cache = None
        uniform_base_cache = None
        _clear_cuda_memory()

        turnwise_prefix_cache = None
        turnwise_turns = []
        for start, end, turn_id, speaker in turn_spans:
            start = int(start)
            end = int(end)
            if start >= sample.boundary.prefix_token_count:
                break
            end = min(end, sample.boundary.prefix_token_count)
            if end <= start:
                continue
            turnwise_prefix_cache, _ = _feed_tokens_with_cache(
                model,
                token_ids[start:end],
                device=collection_config.model.device,
                past_key_values=turnwise_prefix_cache,
                start_position=start,
            )
            turnwise_turns.append(
                {
                    "turn_id": str(turn_id),
                    "speaker": str(speaker),
                    "start": start,
                    "end": end,
                }
            )

        turnwise_base_cache, turnwise_base_position = _build_base_cache(
            model=model,
            device=collection_config.model.device,
            prefix_cache=turnwise_prefix_cache,
            tail_token_ids=tail_token_ids,
            prefix_token_count=sample.boundary.prefix_token_count,
            compacted_layers=sketch_layers,
        )
        turnwise_runs, turnwise_runtime = _run_prompt_path(
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
            base_cache=turnwise_base_cache,
            base_position=turnwise_base_position,
        )

        capture_start_turn = None
        capture_indices = list(bundle.capture_token_indices or [])
        if capture_indices:
            first_capture_index = min(capture_indices)
            for start, end, turn_id, speaker in turn_spans:
                if int(start) <= first_capture_index < int(end):
                    capture_start_turn = {
                        "turn_id": str(turn_id),
                        "speaker": str(speaker),
                        "start": int(start),
                        "end": int(end),
                    }
                    break

        print(
            json.dumps(
                {
                    "prompt_label": prompts[0].label,
                    "uniform_chunk_sketch": {
                        "runtime_seconds": uniform_runtime,
                        "runs": _serializable_runs(uniform_runs),
                    },
                    "turnwise_sketch": {
                        "runtime_seconds": turnwise_runtime,
                        "runs": _serializable_runs(turnwise_runs),
                    },
                    "turns_processed": turnwise_turns,
                    "capture_start_turn": capture_start_turn,
                },
                indent=2,
            )
        )
    finally:
        unload_qwen35_bundle(model)


if __name__ == "__main__":
    main()
