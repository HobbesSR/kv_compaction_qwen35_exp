from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

from kv_compaction_qwen35_clean.behavioral_eval import (
    _build_base_cache,
    _clear_cuda_memory,
    _run_prompt_path,
    select_prompt_subset,
)
from kv_compaction_qwen35_clean.boundary_collection import collect_teacher_forced_boundary_collection
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
from kv_compaction_qwen35_clean.runtime_compaction import build_path_runtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
    )
    parser.add_argument("--keys-per-head", type=int, default=8)
    parser.add_argument("--prompt-set", default="qwen35_calibration_v3")
    parser.add_argument("--prompt-label", default="qwen35_branch_switch_harness_note")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--probe-coverage", choices=("narrow", "all_heads"), default="narrow")
    return parser.parse_args()


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _format_gb(num_bytes: int) -> str:
    return f"{(float(num_bytes) / (1024 ** 3)):.2f}"


def _print_header() -> None:
    print(
        f"{'chunk':>6}  {'collect_s':>10}  {'peak_gb':>8}  "
        f"{'sketch_s':>9}  {'recall':>6}  {'ok':>4}  status"
    )


def _print_row(
    *,
    chunk_size: int,
    collection_seconds: float | None,
    peak_bytes: int | None,
    sketch_seconds: float | None,
    keyword_recall: float | None,
    success: bool | None,
    status: str,
) -> None:
    collect_text = f"{collection_seconds:.3f}" if collection_seconds is not None else "-"
    peak_text = _format_gb(peak_bytes) if peak_bytes is not None else "-"
    sketch_text = f"{sketch_seconds:.3f}" if sketch_seconds is not None else "-"
    recall_text = f"{keyword_recall:.3f}" if keyword_recall is not None else "-"
    success_text = (
        "yes"
        if success is True
        else "no"
        if success is False
        else "-"
    )
    print(
        f"{chunk_size:>6}  {collect_text:>10}  {peak_text:>8}  "
        f"{sketch_text:>9}  {recall_text:>6}  {success_text:>4}  {status}"
    )


def main() -> None:
    args = _parse_args()
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)
    prompts = select_prompt_subset(
        args.prompt_set,
        sample.prompt_family,
        prompt_labels=[args.prompt_label],
    )
    collection_base_config = replace(config, model=replace(config.model, attn_implementation="eager"))
    collection_model, collection_tokenizer, model_type = load_qwen35_bundle(collection_base_config)
    try:
        probe_layers = default_probe_layers_for_model(collection_model, model_type)
        if args.probe_coverage == "narrow":
            probe_heads = default_probe_heads_for_model(collection_model)
        else:
            probe_heads = all_probe_heads_for_model(collection_model)
        target_layer_heads = tuple((int(layer), int(head)) for layer in probe_layers for head in probe_heads)
        token_ids, _ = materialize_long_context_ids(sample, collection_tokenizer)
        prefix_token_ids = token_ids[: sample.boundary.prefix_token_count]
        tail_token_ids = token_ids[sample.boundary.prefix_token_count :]

        _print_header()
        for chunk_size in args.chunk_sizes:
            bundle = None
            sketch_base_cache = None
            sketch_layers = None
            try:
                run_config = replace(
                    collection_base_config,
                    model=replace(collection_base_config.model, prefill_chunk_size=int(chunk_size)),
                )
                _clear_cuda_memory()
                peak_bytes = None
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                start_time = time.perf_counter()
                bundle = collect_teacher_forced_boundary_collection(
                    sample,
                    run_config,
                    model=collection_model,
                    tokenizer=collection_tokenizer,
                    probe_layers=probe_layers,
                    probe_heads=probe_heads,
                    retain_runtime_cache=True,
                )
                collection_seconds = time.perf_counter() - start_time
                if torch.cuda.is_available():
                    peak_bytes = int(torch.cuda.max_memory_allocated())

                state = build_state_from_observations(run_config, bundle.harvest.observations)
                sketch_source = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, run_config)
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
                    compute_device=run_config.model.device,
                    key_selection_method=config.compaction.key_selection,
                )
                sketch_base_cache, sketch_base_position = _build_base_cache(
                    model=collection_model,
                    device=run_config.model.device,
                    prefix_cache=bundle.runtime_cache,
                    tail_token_ids=tail_token_ids,
                    prefix_token_count=sample.boundary.prefix_token_count,
                    compacted_layers=sketch_layers,
                )

                bundle.runtime_cache = None
                bundle.boundary_keys = {}
                bundle.boundary_values = {}
                bundle.boundary_projected_values = {}
                bundle.output_targets = {}
                _clear_cuda_memory()

                sketch_runs, sketch_runtime = _run_prompt_path(
                    model=collection_model,
                    tokenizer=collection_tokenizer,
                    prompts=prompts,
                    prefix_token_ids=prefix_token_ids,
                    tail_token_ids=tail_token_ids,
                    device=run_config.model.device,
                    compacted_layers=sketch_layers,
                    prefix_token_count=sample.boundary.prefix_token_count,
                    enable_thinking=run_config.model.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    reference_runs=None,
                    base_cache=sketch_base_cache,
                    base_position=sketch_base_position,
                )
                sketch_run = sketch_runs[0]
                _print_row(
                    chunk_size=int(chunk_size),
                    collection_seconds=collection_seconds,
                    peak_bytes=peak_bytes,
                    sketch_seconds=sketch_runtime,
                    keyword_recall=sketch_run.keyword_recall,
                    success=sketch_run.central_detail_preserved,
                    status="ok",
                )
            except Exception as exc:
                peak_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                _print_row(
                    chunk_size=int(chunk_size),
                    collection_seconds=None,
                    peak_bytes=peak_bytes,
                    sketch_seconds=None,
                    keyword_recall=None,
                    success=None,
                    status="oom" if _is_oom_error(exc) else f"error:{type(exc).__name__}",
                )
                _clear_cuda_memory()
                if _is_oom_error(exc):
                    break
                raise
            finally:
                bundle = None
                sketch_base_cache = None
                sketch_layers = None
                _clear_cuda_memory()
    finally:
        unload_qwen35_bundle(collection_model)


if __name__ == "__main__":
    main()
