from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, replace

from kv_compaction_qwen35_clean.behavioral_eval import (
    _build_base_cache,
    _clear_cuda_memory,
    _feed_tokens_with_cache,
    _run_prompt_path,
    select_prompt_subset,
)
from kv_compaction_qwen35_clean.boundary_collection import (
    LONG_CONTEXT_TOKEN_STRIDE,
    collect_teacher_forced_boundary_collection,
    resolve_replay_checkpoint_start,
    select_long_context_capture_indices,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-set", default="qwen35_calibration_v3")
    parser.add_argument("--keys-per-head", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument(
        "--token-budget",
        type=int,
        nargs="+",
        default=[2048, 1024, 512, 256],
        help="Near-boundary replay token budgets to test.",
    )
    return parser.parse_args()


def _serializable_runs(runs):
    return [asdict(run) for run in runs]


def _select_token_budget_capture_indices(
    prefix_token_count: int,
    turn_spans: list[tuple[int, int, str, str]],
    *,
    token_budget: int,
    stride: int = LONG_CONTEXT_TOKEN_STRIDE,
) -> list[int]:
    if prefix_token_count <= 1:
        return []
    window_start = max(0, prefix_token_count - max(1, int(token_budget)))
    indices = {
        index
        for index in select_long_context_capture_indices(prefix_token_count, stride=stride)
        if int(index) >= window_start
    }
    for start, end, _turn_id, _speaker in turn_spans:
        start = int(start)
        end = min(prefix_token_count, int(end))
        if end <= start or end <= window_start:
            continue
        indices.add(end - 1)
    return sorted(index for index in indices if 0 <= index < prefix_token_count)


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


def _build_checkpoint_cache(
    *,
    model,
    token_ids: list[int],
    checkpoint_start: int,
    turn_spans: list[tuple[int, int, str, str]],
    device: str,
):
    checkpoint_cache = None
    for start, end, _turn_id, _speaker in turn_spans:
        start = int(start)
        end = min(int(end), int(checkpoint_start))
        if end <= start:
            continue
        checkpoint_cache, _ = _feed_tokens_with_cache(
            model,
            token_ids[start:end],
            device=device,
            past_key_values=checkpoint_cache,
            start_position=start,
        )
    return checkpoint_cache


def _bank_summary(state) -> dict[str, object]:
    turn_counts = Counter(entry.source_turn_id for entry in state.entries)
    layer_counts = Counter(int(entry.layer) for entry in state.entries)
    head_counts = Counter(int(entry.head) for entry in state.entries)
    return {
        "slot_count": len(state.entries),
        "turn_count": len(turn_counts),
        "layer_count": len(layer_counts),
        "head_count": len(head_counts),
        "turn_counts": dict(sorted(turn_counts.items())),
        "layer_counts": dict(sorted(layer_counts.items())),
        "head_counts": dict(sorted(head_counts.items())),
    }


def _run_sketch_surface(
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
    keys_per_head: int,
    max_new_tokens: int,
):
    state = build_state_from_observations(collection_config, bundle.harvest.observations)
    sketch_source = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, collection_config)
    _, sketch_layers = build_path_runtime(
        sample.sample_id,
        sample.boundary.boundary_id,
        sketch_source.source,
        keys_per_head,
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
        max_new_tokens=max_new_tokens,
        reference_runs=None,
        base_cache=sketch_base_cache,
        base_position=sketch_base_position,
    )
    bundle.runtime_cache = None
    bundle.boundary_keys = {}
    bundle.boundary_values = {}
    bundle.boundary_projected_values = {}
    bundle.output_targets = {}
    return runs, runtime_seconds, state


def _surface_summary(
    *,
    label: str,
    collection_seconds: float,
    sketch_runtime_seconds: float,
    runs,
    bank_state,
    capture_indices: list[int] | None,
    checkpoint: dict[str, object] | None,
    replay_window_tokens: int | None,
) -> dict[str, object]:
    preserved_count = sum(int(run.central_detail_preserved) for run in runs)
    hallucination_run_count = sum(1 for run in runs if run.hallucination_flags)
    return {
        "label": label,
        "collection_seconds": round(collection_seconds, 6),
        "sketch_runtime_seconds": round(sketch_runtime_seconds, 6),
        "preserved_count": preserved_count,
        "prompt_count": len(runs),
        "hallucination_run_count": hallucination_run_count,
        "capture_indices": capture_indices,
        "checkpoint": checkpoint,
        "replay_window_tokens": replay_window_tokens,
        "bank": _bank_summary(bank_state),
        "runs": _serializable_runs(runs),
    }


def main() -> None:
    args = _parse_args()
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)
    prompts = select_prompt_subset(args.prompt_set, sample.prompt_family)
    collection_config = replace(config, model=replace(config.model, attn_implementation="eager"))
    model, tokenizer, model_type = load_qwen35_bundle(collection_config)
    try:
        probe_layers = default_probe_layers_for_model(model, model_type)
        probe_heads = default_probe_heads_for_model(model)
        target_layer_heads = tuple((int(layer), int(head)) for layer in probe_layers for head in probe_heads)
        token_ids, turn_spans = materialize_long_context_ids(sample, tokenizer)

        rows: list[dict[str, object]] = []

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
        full_runs, full_runtime_seconds, full_state = _run_sketch_surface(
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
            keys_per_head=args.keys_per_head,
            max_new_tokens=args.max_new_tokens,
        )
        rows.append(
            _surface_summary(
                label="full_collection",
                collection_seconds=full_collection_seconds,
                sketch_runtime_seconds=full_runtime_seconds,
                runs=full_runs,
                bank_state=full_state,
                capture_indices=None,
                checkpoint=None,
                replay_window_tokens=None,
            )
        )
        _clear_cuda_memory()

        for token_budget in args.token_budget:
            capture_indices = _select_token_budget_capture_indices(
                sample.boundary.prefix_token_count,
                turn_spans,
                token_budget=int(token_budget),
            )
            checkpoint = _checkpoint_for_capture_indices(capture_indices, turn_spans)
            if checkpoint is None:
                continue
            checkpoint_cache = _build_checkpoint_cache(
                model=model,
                token_ids=token_ids,
                checkpoint_start=int(checkpoint["start"]),
                turn_spans=turn_spans,
                device=collection_config.model.device,
            )
            replay_start = time.perf_counter()
            replay_bundle = collect_teacher_forced_boundary_collection(
                sample,
                collection_config,
                model=model,
                tokenizer=tokenizer,
                probe_layers=probe_layers,
                probe_heads=probe_heads,
                capture_indices=capture_indices,
                retain_runtime_cache=True,
                initial_past_key_values=checkpoint_cache,
                replay_start_position=int(checkpoint["start"]),
            )
            replay_collection_seconds = time.perf_counter() - replay_start
            replay_runs, replay_runtime_seconds, replay_state = _run_sketch_surface(
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
                keys_per_head=args.keys_per_head,
                max_new_tokens=args.max_new_tokens,
            )
            rows.append(
                _surface_summary(
                    label=f"replay_{int(token_budget)}",
                    collection_seconds=replay_collection_seconds,
                    sketch_runtime_seconds=replay_runtime_seconds,
                    runs=replay_runs,
                    bank_state=replay_state,
                    capture_indices=capture_indices,
                    checkpoint=checkpoint,
                    replay_window_tokens=sample.boundary.prefix_token_count - int(checkpoint["start"]),
                )
            )
            _clear_cuda_memory()

        print("label         collect_s  sketch_s  score  hall  slots  turns  layers  heads  replay_tokens")
        for row in rows:
            bank = row["bank"]
            print(
                f"{row['label']:<12} "
                f"{row['collection_seconds']:>9.3f} "
                f"{row['sketch_runtime_seconds']:>9.3f} "
                f"{row['preserved_count']:>5}/{row['prompt_count']:<2} "
                f"{row['hallucination_run_count']:>5} "
                f"{bank['slot_count']:>6} "
                f"{bank['turn_count']:>6} "
                f"{bank['layer_count']:>7} "
                f"{bank['head_count']:>6} "
                f"{str(row['replay_window_tokens'] or '-'):>13}"
            )
        print()
        print(json.dumps(rows, indent=2))
    finally:
        unload_qwen35_bundle(model)


if __name__ == "__main__":
    main()
