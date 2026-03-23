from __future__ import annotations

import argparse
import json
from dataclasses import replace
from dataclasses import asdict

import torch

from kv_compaction_qwen35_clean.behavioral_eval import (
    _build_base_cache,
    _clear_cuda_memory,
    _run_prompt_path,
    select_prompt_subset,
)
from kv_compaction_qwen35_clean.boundary_collection import (
    CAPTURE_ATTENTION_CHUNK_SIZE,
    _capture_chunks,
    collect_teacher_forced_boundary_collection,
    select_long_context_capture_indices,
)
from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.context_loader import load_context_sample
from kv_compaction_qwen35_clean.coreset import extract_query_coreset
from kv_compaction_qwen35_clean.model_runtime import (
    all_probe_heads_for_model,
    default_probe_heads_for_model,
    default_probe_layers_for_model,
    load_qwen35_bundle,
    materialize_long_context_ids,
    unload_qwen35_bundle,
)
from kv_compaction_qwen35_clean.prototype_bank import build_state_from_observations
from kv_compaction_qwen35_clean.query_controls import extract_teacher_forced_subsample_control
from kv_compaction_qwen35_clean.runtime_compaction import build_path_runtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys-per-head", type=int, default=8)
    parser.add_argument("--prompt-set", default="qwen35_calibration_v3")
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--prompt-label", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--probe-coverage", choices=("narrow", "all_heads"), default="narrow")
    parser.add_argument(
        "--reference-mode",
        choices=("cached", "uncached", "both", "compare_sources"),
        default="cached",
        help="Only affects the reference path in this probe runner.",
    )
    parser.add_argument(
        "--path",
        action="append",
        choices=("reference", "sketch", "control"),
        default=None,
        help="Repeat to run multiple paths. Defaults to all three.",
    )
    return parser.parse_args()


def _serializable_runs(runs):
    return [asdict(run) for run in runs]


def _build_chunked_prefix_cache(*, model, device: str, prefix_token_ids: list[int], chunk_size: int):
    past_key_values = None
    processed = 0
    while processed < len(prefix_token_ids):
        chunk_end = min(len(prefix_token_ids), processed + int(chunk_size))
        input_tensor = torch.tensor([prefix_token_ids[processed:chunk_end]], device=device, dtype=torch.long)
        cache_position = torch.arange(processed, chunk_end, device=device, dtype=torch.long)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_tensor,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                cache_position=cache_position,
            )
        past_key_values = outputs.past_key_values
        processed = chunk_end
    return past_key_values


def _build_capture_schedule_prefix_cache(
    *,
    model,
    device: str,
    prefix_token_ids: list[int],
    prefill_chunk_size: int,
):
    capture_indices = select_long_context_capture_indices(len(prefix_token_ids))
    past_key_values = None
    processed = 0
    for capture_start, capture_end in _capture_chunks(
        capture_indices,
        max_chunk_size=min(int(prefill_chunk_size), CAPTURE_ATTENTION_CHUNK_SIZE),
    ):
        while processed < capture_start:
            chunk_end = min(capture_start, processed + int(prefill_chunk_size))
            input_tensor = torch.tensor([prefix_token_ids[processed:chunk_end]], device=device, dtype=torch.long)
            cache_position = torch.arange(processed, chunk_end, device=device, dtype=torch.long)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    cache_position=cache_position,
                )
            past_key_values = outputs.past_key_values
            processed = chunk_end

        input_tensor = torch.tensor([prefix_token_ids[capture_start:capture_end]], device=device, dtype=torch.long)
        cache_position = torch.arange(capture_start, capture_end, device=device, dtype=torch.long)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_tensor,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                cache_position=cache_position,
            )
        past_key_values = outputs.past_key_values
        processed = capture_end

    while processed < len(prefix_token_ids):
        chunk_end = min(len(prefix_token_ids), processed + int(prefill_chunk_size))
        input_tensor = torch.tensor([prefix_token_ids[processed:chunk_end]], device=device, dtype=torch.long)
        cache_position = torch.arange(processed, chunk_end, device=device, dtype=torch.long)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_tensor,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                cache_position=cache_position,
            )
        past_key_values = outputs.past_key_values
        processed = chunk_end
    return past_key_values


def main() -> None:
    args = _parse_args()
    selected_paths = args.path or ["reference", "sketch", "control"]

    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)
    prompts = select_prompt_subset(
        args.prompt_set,
        sample.prompt_family,
        prompt_limit=args.prompt_limit,
        prompt_labels=args.prompt_label,
    )

    collection_config = replace(config, model=replace(config.model, attn_implementation="eager"))
    collection_model, collection_tokenizer, model_type = load_qwen35_bundle(collection_config)
    try:
        probe_layers = default_probe_layers_for_model(collection_model, model_type)
        if args.probe_coverage == "narrow":
            probe_heads = default_probe_heads_for_model(collection_model)
        else:
            probe_heads = all_probe_heads_for_model(collection_model)
        target_layer_heads = tuple((int(layer), int(head)) for layer in probe_layers for head in probe_heads)

        bundle = collect_teacher_forced_boundary_collection(
            sample,
            collection_config,
            model=collection_model,
            tokenizer=collection_tokenizer,
            probe_layers=probe_layers,
            probe_heads=probe_heads,
            retain_runtime_cache=True,
        )
        token_ids, _ = materialize_long_context_ids(sample, collection_tokenizer)
        prefix_token_ids = token_ids[: sample.boundary.prefix_token_count]
        tail_token_ids = token_ids[sample.boundary.prefix_token_count :]
        prefix_runtime_cache = bundle.runtime_cache

        state = None
        sketch_source = None
        control_query_source = None
        sketch_layers = None
        control_layers = None
        sketch_base_cache = None
        sketch_base_position = None
        control_base_cache = None
        control_base_position = None

        if any(path in selected_paths for path in ("sketch", "control")):
            state = build_state_from_observations(collection_config, bundle.harvest.observations)
            sketch_source = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, collection_config)
            control_query_source = extract_teacher_forced_subsample_control(
                bundle.query_bank,
                max_entries=len(sketch_source.selected_entries),
            )

        reference_base_cache = None
        reference_base_position = None
        if any(path in selected_paths for path in ("reference", "sketch", "control")):
            reference_base_cache, reference_base_position = _build_base_cache(
                model=collection_model,
                device=collection_config.model.device,
                prefix_cache=prefix_runtime_cache,
                tail_token_ids=tail_token_ids,
                prefix_token_count=sample.boundary.prefix_token_count,
                compacted_layers=None,
            )
        chunked_reference_base_cache = None
        chunked_reference_base_position = None
        if "reference" in selected_paths and args.reference_mode == "compare_sources":
            chunked_prefix_cache = _build_chunked_prefix_cache(
                model=collection_model,
                device=collection_config.model.device,
                prefix_token_ids=prefix_token_ids,
                chunk_size=collection_config.model.prefill_chunk_size,
            )
            chunked_reference_base_cache, chunked_reference_base_position = _build_base_cache(
                model=collection_model,
                device=collection_config.model.device,
                prefix_cache=chunked_prefix_cache,
                tail_token_ids=tail_token_ids,
                prefix_token_count=sample.boundary.prefix_token_count,
                compacted_layers=None,
            )
            chunked_prefix_cache = None
            capture_schedule_prefix_cache = _build_capture_schedule_prefix_cache(
                model=collection_model,
                device=collection_config.model.device,
                prefix_token_ids=prefix_token_ids,
                prefill_chunk_size=collection_config.model.prefill_chunk_size,
            )
            capture_schedule_reference_base_cache, capture_schedule_reference_base_position = _build_base_cache(
                model=collection_model,
                device=collection_config.model.device,
                prefix_cache=capture_schedule_prefix_cache,
                tail_token_ids=tail_token_ids,
                prefix_token_count=sample.boundary.prefix_token_count,
                compacted_layers=None,
            )
            capture_schedule_prefix_cache = None

        if "sketch" in selected_paths:
            _, sketch_layers = build_path_runtime(
                sample.sample_id,
                sample.boundary.boundary_id,
                sketch_source.source,
                args.keys_per_head,
                bundle,
                sketch_source,
                target_layers=probe_layers,
                target_heads=probe_heads,
                target_layer_heads=target_layer_heads,
                compute_device=collection_config.model.device,
                key_selection_method=config.compaction.key_selection,
            )
            sketch_base_cache, sketch_base_position = _build_base_cache(
                model=collection_model,
                device=collection_config.model.device,
                prefix_cache=prefix_runtime_cache,
                tail_token_ids=tail_token_ids,
                prefix_token_count=sample.boundary.prefix_token_count,
                compacted_layers=sketch_layers,
            )

        if "control" in selected_paths:
            _, control_layers = build_path_runtime(
                sample.sample_id,
                sample.boundary.boundary_id,
                control_query_source.source,
                args.keys_per_head,
                bundle,
                control_query_source,
                target_layers=probe_layers,
                target_heads=probe_heads,
                target_layer_heads=target_layer_heads,
                compute_device=collection_config.model.device,
                key_selection_method=config.compaction.key_selection,
            )
            control_base_cache, control_base_position = _build_base_cache(
                model=collection_model,
                device=collection_config.model.device,
                prefix_cache=prefix_runtime_cache,
                tail_token_ids=tail_token_ids,
                prefix_token_count=sample.boundary.prefix_token_count,
                compacted_layers=control_layers,
            )

        bundle.runtime_cache = None
        bundle.boundary_keys = {}
        bundle.boundary_values = {}
        bundle.boundary_projected_values = {}
        bundle.output_targets = {}
        _clear_cuda_memory()

        reference_runs = None
        reference_total_runtime = None
        if any(path in selected_paths for path in ("reference", "sketch", "control")):
            reference_runs, reference_total_runtime = _run_prompt_path(
                model=collection_model,
                tokenizer=collection_tokenizer,
                prompts=prompts,
                prefix_token_ids=prefix_token_ids,
                tail_token_ids=tail_token_ids,
                device=collection_config.model.device,
                compacted_layers=None,
                prefix_token_count=sample.boundary.prefix_token_count,
                enable_thinking=collection_config.model.enable_thinking,
                max_new_tokens=args.max_new_tokens,
                reference_runs=None,
                base_cache=reference_base_cache,
                base_position=reference_base_position,
            )

        output = {
            "prompt_labels": [prompt.label for prompt in prompts],
            "selected_paths": selected_paths,
            "reference": None,
            "sketch": None,
            "control": None,
        }

        if "reference" in selected_paths:
            output["reference"] = {
                "runtime_seconds": reference_total_runtime,
                "runs": _serializable_runs(reference_runs),
                "mode": "cached",
            }
            if args.reference_mode == "uncached":
                uncached_runs, uncached_runtime = _run_prompt_path(
                    model=collection_model,
                    tokenizer=collection_tokenizer,
                    prompts=prompts,
                    prefix_token_ids=prefix_token_ids,
                    tail_token_ids=tail_token_ids,
                    device=collection_config.model.device,
                    compacted_layers=None,
                    prefix_token_count=sample.boundary.prefix_token_count,
                    enable_thinking=collection_config.model.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    reference_runs=None,
                    base_cache=None,
                    base_position=None,
                )
                output["reference"] = {
                    "runtime_seconds": uncached_runtime,
                    "runs": _serializable_runs(uncached_runs),
                    "mode": "uncached",
                }
            elif args.reference_mode == "both":
                uncached_runs, uncached_runtime = _run_prompt_path(
                    model=collection_model,
                    tokenizer=collection_tokenizer,
                    prompts=prompts,
                    prefix_token_ids=prefix_token_ids,
                    tail_token_ids=tail_token_ids,
                    device=collection_config.model.device,
                    compacted_layers=None,
                    prefix_token_count=sample.boundary.prefix_token_count,
                    enable_thinking=collection_config.model.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    reference_runs=None,
                    base_cache=None,
                    base_position=None,
                )
                output["reference"] = {
                    "cached": {
                        "runtime_seconds": reference_total_runtime,
                        "runs": _serializable_runs(reference_runs),
                    },
                    "uncached": {
                        "runtime_seconds": uncached_runtime,
                        "runs": _serializable_runs(uncached_runs),
                    },
                    "mode": "both",
                }
            elif args.reference_mode == "compare_sources":
                uncached_runs, uncached_runtime = _run_prompt_path(
                    model=collection_model,
                    tokenizer=collection_tokenizer,
                    prompts=prompts,
                    prefix_token_ids=prefix_token_ids,
                    tail_token_ids=tail_token_ids,
                    device=collection_config.model.device,
                    compacted_layers=None,
                    prefix_token_count=sample.boundary.prefix_token_count,
                    enable_thinking=collection_config.model.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    reference_runs=None,
                    base_cache=None,
                    base_position=None,
                )
                chunked_runs, chunked_runtime = _run_prompt_path(
                    model=collection_model,
                    tokenizer=collection_tokenizer,
                    prompts=prompts,
                    prefix_token_ids=prefix_token_ids,
                    tail_token_ids=tail_token_ids,
                    device=collection_config.model.device,
                    compacted_layers=None,
                    prefix_token_count=sample.boundary.prefix_token_count,
                    enable_thinking=collection_config.model.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    reference_runs=None,
                    base_cache=chunked_reference_base_cache,
                    base_position=chunked_reference_base_position,
                )
                capture_schedule_runs, capture_schedule_runtime = _run_prompt_path(
                    model=collection_model,
                    tokenizer=collection_tokenizer,
                    prompts=prompts,
                    prefix_token_ids=prefix_token_ids,
                    tail_token_ids=tail_token_ids,
                    device=collection_config.model.device,
                    compacted_layers=None,
                    prefix_token_count=sample.boundary.prefix_token_count,
                    enable_thinking=collection_config.model.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    reference_runs=None,
                    base_cache=capture_schedule_reference_base_cache,
                    base_position=capture_schedule_reference_base_position,
                )
                output["reference"] = {
                    "traced_collection": {
                        "runtime_seconds": reference_total_runtime,
                        "runs": _serializable_runs(reference_runs),
                    },
                    "chunked_no_trace": {
                        "runtime_seconds": chunked_runtime,
                        "runs": _serializable_runs(chunked_runs),
                    },
                    "capture_chunks_no_trace": {
                        "runtime_seconds": capture_schedule_runtime,
                        "runs": _serializable_runs(capture_schedule_runs),
                    },
                    "uncached": {
                        "runtime_seconds": uncached_runtime,
                        "runs": _serializable_runs(uncached_runs),
                    },
                    "mode": "compare_sources",
                }
        if "sketch" in selected_paths:
            sketch_runs, sketch_total_runtime = _run_prompt_path(
                model=collection_model,
                tokenizer=collection_tokenizer,
                prompts=prompts,
                prefix_token_ids=prefix_token_ids,
                tail_token_ids=tail_token_ids,
                device=collection_config.model.device,
                compacted_layers=sketch_layers,
                prefix_token_count=sample.boundary.prefix_token_count,
                enable_thinking=collection_config.model.enable_thinking,
                max_new_tokens=args.max_new_tokens,
                reference_runs=reference_runs,
                base_cache=sketch_base_cache,
                base_position=sketch_base_position,
            )
            output["sketch"] = {
                "runtime_seconds": sketch_total_runtime,
                "runs": _serializable_runs(sketch_runs),
            }
        if "control" in selected_paths:
            control_runs, control_total_runtime = _run_prompt_path(
                model=collection_model,
                tokenizer=collection_tokenizer,
                prompts=prompts,
                prefix_token_ids=prefix_token_ids,
                tail_token_ids=tail_token_ids,
                device=collection_config.model.device,
                compacted_layers=control_layers,
                prefix_token_count=sample.boundary.prefix_token_count,
                enable_thinking=collection_config.model.enable_thinking,
                max_new_tokens=args.max_new_tokens,
                reference_runs=reference_runs,
                base_cache=control_base_cache,
                base_position=control_base_position,
            )
            output["control"] = {
                "runtime_seconds": control_total_runtime,
                "runs": _serializable_runs(control_runs),
            }

        print(json.dumps(output, indent=2))
    finally:
        unload_qwen35_bundle(collection_model)


if __name__ == "__main__":
    main()
