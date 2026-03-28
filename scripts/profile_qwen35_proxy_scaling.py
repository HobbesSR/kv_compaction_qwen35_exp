#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.data_types import CompactHeadRuntime
from kv_compaction_qwen35_clean.model_runtime import resolve_qwen35_model_type
from kv_compaction_qwen35_clean.openai_chat_canonicalization import (
    canonicalize_openai_chat_messages,
    find_cached_prefix_for_openai_messages,
    find_cached_prefix_metadata_for_openai_messages,
)
from kv_compaction_qwen35_clean.qwen35_openai_proxy import Qwen35OpenAIProxyService
from kv_compaction_qwen35_clean.segment_compaction_cache import (
    build_config_fingerprint,
    build_segment_bundle,
    build_turn_segment_lineage,
    find_cached_prefix,
    find_cached_prefix_metadata,
    write_segment_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile OpenAI-proxy scaling as context grows.")
    parser.add_argument(
        "--config",
        default="configs/qwen35_smoke/qwen3_5_9b.yaml",
        help="Config used for tokenizer/model loading.",
    )
    parser.add_argument(
        "--token-targets",
        default="1024,2048,4096,7168",
        help="Comma-separated logical-token targets for the synthetic chat transcript.",
    )
    parser.add_argument(
        "--lookup-repeats",
        type=int,
        default=5,
        help="Repeat count for canonicalization and prefix lookup timings.",
    )
    parser.add_argument(
        "--live-token-targets",
        default="1024,4096,7168",
        help="Comma-separated logical-token targets for the live model latency sweep.",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip the live model latency sweep and run metadata/tokenization timings only.",
    )
    return parser.parse_args()


def _parse_targets(text: str) -> list[int]:
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def _build_messages_for_target(
    tokenizer,
    *,
    model_type: str,
    enable_thinking: bool,
    target_tokens: int,
) -> tuple[list[dict[str, object]], object]:
    messages: list[dict[str, object]] = [{"role": "system", "content": "You are a concise assistant."}]
    turn_index = 0
    transcript = canonicalize_openai_chat_messages(
        tokenizer,
        model_type=model_type,
        messages=messages + [{"role": "user", "content": "seed"}],
        enable_thinking=enable_thinking,
    )
    while len(transcript.message_token_ids) < target_tokens:
        user_content = (
            f"User turn {turn_index}. "
            f"Status recap: alpha beta gamma delta epsilon zeta eta theta. "
            f"Please keep track of task switch state {turn_index % 7}. "
        ) * 3
        messages.append({"role": "user", "content": user_content})
        transcript = canonicalize_openai_chat_messages(
            tokenizer,
            model_type=model_type,
            messages=messages,
            enable_thinking=enable_thinking,
        )
        if len(transcript.message_token_ids) >= target_tokens:
            break
        assistant_content = (
            f"Assistant turn {turn_index}. "
            f"Acknowledged task state {turn_index % 7} and preserved prior details. "
            f"Checklist items remain open and tracked. "
        ) * 2
        messages.append({"role": "assistant", "content": assistant_content})
        transcript = canonicalize_openai_chat_messages(
            tokenizer,
            model_type=model_type,
            messages=messages,
            enable_thinking=enable_thinking,
        )
        turn_index += 1
    return messages, transcript


def _sample_compacted_layers(
    *,
    target_layer_heads: tuple[tuple[int, int], ...],
    segment_start_token: int,
    segment_end_token: int,
    keys_per_head: int,
) -> dict[int, dict[int, CompactHeadRuntime]]:
    keep_count = max(1, min(int(keys_per_head), int(segment_end_token) - int(segment_start_token)))
    selected_indices = list(range(int(segment_end_token) - keep_count, int(segment_end_token)))
    compacted_layers: dict[int, dict[int, CompactHeadRuntime]] = {}
    for layer, head in target_layer_heads:
        compacted_layers.setdefault(int(layer), {})[int(head)] = CompactHeadRuntime(
            layer=int(layer),
            head=int(head),
            selected_indices=selected_indices,
            compact_keys=torch.zeros((keep_count, 256), dtype=torch.float16),
            compact_values=torch.zeros((keep_count, 256), dtype=torch.float16),
            beta=torch.zeros((keep_count,), dtype=torch.float16),
        )
    return compacted_layers


def _populate_cache(
    *,
    cache_root: Path,
    transcript,
    config_fingerprint: str,
    target_layer_heads: tuple[tuple[int, int], ...],
    keys_per_head: int,
    model_name: str,
    huggingface_id: str,
    tokenizer_name: str,
    tokenizer_fingerprint: str,
) -> int:
    lineage = build_turn_segment_lineage(
        token_ids=transcript.message_token_ids,
        turn_spans=transcript.turn_spans,
        config_fingerprint=config_fingerprint,
        min_segment_tokens=1,
    )
    for node in lineage:
        compacted_layers = _sample_compacted_layers(
            target_layer_heads=target_layer_heads,
            segment_start_token=node.boundary.segment_start_token,
            segment_end_token=node.boundary.segment_end_token,
            keys_per_head=keys_per_head,
        )
        bundle = build_segment_bundle(
            parent_hash=node.parent_hash,
            segment_token_ids=transcript.message_token_ids[node.boundary.segment_start_token : node.boundary.segment_end_token],
            boundary_turn_index=node.boundary.boundary_turn_index,
            segment_start_token=node.boundary.segment_start_token,
            segment_end_token=node.boundary.segment_end_token,
            logical_token_count_before=node.logical_token_count_before,
            logical_token_count_after=node.logical_token_count_after,
            model_name=model_name,
            huggingface_id=huggingface_id,
            tokenizer_name=tokenizer_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
            config_fingerprint=config_fingerprint,
            target_layer_heads=target_layer_heads,
            compacted_layers=compacted_layers,
        )
        write_segment_bundle(bundle, cache_root)
    return len(lineage)


def _median_seconds(fn, *, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _run_lookup_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    config = load_config(args.config)
    model_type = resolve_qwen35_model_type(
        config.model.huggingface_id,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    enable_thinking = config.model.enable_thinking if config.model.enable_thinking is not None else False

    with tempfile.TemporaryDirectory(prefix="qwen35_proxy_scale_cache_") as tmp_dir:
        cache_root = Path(tmp_dir)
        target_layer_heads = tuple(
            (int(layer), int(head))
            for layer in (3, 7, 11, 15, 19, 23, 27, 31)
            for head in (0, 3, 7)
        )
        tokenizer_fingerprint = f"{config.model.tokenizer_name}:{enable_thinking}"
        config_fingerprint = build_config_fingerprint(
            model_name=config.model.name,
            huggingface_id=config.model.huggingface_id,
            tokenizer_name=config.model.tokenizer_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
            target_layer_heads=target_layer_heads,
            keys_per_head=8,
            key_selection_method=config.compaction.key_selection,
            beta_solver="service_baseline",
            beta_regularization_strength=0.0,
            value_regularization_strength=0.0,
        )
        rows: list[dict[str, object]] = []
        for target_tokens in _parse_targets(args.token_targets):
            shutil.rmtree(cache_root, ignore_errors=True)
            cache_root.mkdir(parents=True, exist_ok=True)
            messages, transcript = _build_messages_for_target(
                tokenizer,
                model_type=model_type,
                enable_thinking=enable_thinking,
                target_tokens=target_tokens,
            )
            segment_count = _populate_cache(
                cache_root=cache_root,
                transcript=transcript,
                config_fingerprint=config_fingerprint,
                target_layer_heads=target_layer_heads,
                keys_per_head=8,
                model_name=config.model.name,
                huggingface_id=config.model.huggingface_id,
                tokenizer_name=config.model.tokenizer_name,
                tokenizer_fingerprint=tokenizer_fingerprint,
            )
            canonicalize_seconds = _median_seconds(
                lambda: canonicalize_openai_chat_messages(
                    tokenizer,
                    model_type=model_type,
                    messages=messages,
                    enable_thinking=enable_thinking,
                ),
                repeats=args.lookup_repeats,
            )
            metadata_lookup_seconds = _median_seconds(
                lambda: find_cached_prefix_metadata_for_openai_messages(
                    tokenizer,
                    model_type=model_type,
                    messages=messages,
                    config_fingerprint=config_fingerprint,
                    cache_root=cache_root,
                    min_segment_tokens=1,
                    enable_thinking=enable_thinking,
                ),
                repeats=args.lookup_repeats,
            )
            full_lookup_seconds = _median_seconds(
                lambda: find_cached_prefix_for_openai_messages(
                    tokenizer,
                    model_type=model_type,
                    messages=messages,
                    config_fingerprint=config_fingerprint,
                    cache_root=cache_root,
                    min_segment_tokens=1,
                    enable_thinking=enable_thinking,
                    device="cpu",
                ),
                repeats=max(1, min(args.lookup_repeats, 3)),
            )
            pure_metadata_lookup_seconds = _median_seconds(
                lambda: find_cached_prefix_metadata(
                    token_ids=transcript.message_token_ids,
                    turn_spans=transcript.turn_spans,
                    config_fingerprint=config_fingerprint,
                    cache_root=cache_root,
                    min_segment_tokens=1,
                ),
                repeats=args.lookup_repeats,
            )
            pure_full_lookup_seconds = _median_seconds(
                lambda: find_cached_prefix(
                    token_ids=transcript.message_token_ids,
                    turn_spans=transcript.turn_spans,
                    config_fingerprint=config_fingerprint,
                    cache_root=cache_root,
                    min_segment_tokens=1,
                    device="cpu",
                ),
                repeats=max(1, min(args.lookup_repeats, 3)),
            )
            rows.append(
                {
                    "logical_tokens": len(transcript.message_token_ids),
                    "turns": len(transcript.turn_spans),
                    "segments": segment_count,
                    "canonicalize_s": round(canonicalize_seconds, 6),
                    "lookup_meta_s": round(metadata_lookup_seconds, 6),
                    "lookup_full_s": round(full_lookup_seconds, 6),
                    "pure_lookup_meta_s": round(pure_metadata_lookup_seconds, 6),
                    "pure_lookup_full_s": round(pure_full_lookup_seconds, 6),
                }
            )
    return rows


def _run_live_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="qwen35_proxy_live_cache_") as tmp_dir:
        service = Qwen35OpenAIProxyService(config_path=args.config, cache_root=tmp_dir)
        try:
            for target_tokens in _parse_targets(args.live_token_targets):
                shutil.rmtree(service.cache_root, ignore_errors=True)
                service.cache_root.mkdir(parents=True, exist_ok=True)
                messages, transcript = _build_messages_for_target(
                    service.tokenizer,
                    model_type=service.model_type,
                    enable_thinking=bool(service.enable_thinking),
                    target_tokens=target_tokens,
                )
                if torch.cuda.is_available() and str(service.config.model.device).startswith("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(service.config.model.device)
                _response, metrics = service.complete(
                    messages=messages,
                    tools=None,
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                )
                peak_gb = None
                if torch.cuda.is_available() and str(service.config.model.device).startswith("cuda"):
                    peak_bytes = torch.cuda.max_memory_allocated(service.config.model.device)
                    peak_gb = round(peak_bytes / (1024**3), 2)
                rows.append(
                    {
                        "logical_tokens": len(transcript.message_token_ids),
                        "turns": len(transcript.turn_spans),
                        "prompt_tokens": metrics.prompt_tokens,
                        "completion_tokens": metrics.completion_tokens,
                        "generation_s": metrics.generation_seconds,
                        "peak_allocated_gb": peak_gb,
                    }
                )
        finally:
            service.close()
    return rows


def _print_table(title: str, rows: list[dict[str, object]]) -> None:
    print(title)
    if not rows:
        print("  <no rows>")
        return
    columns = list(rows[0].keys())
    widths = {
        column: max(len(column), max(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    print("  " + "  ".join(column.rjust(widths[column]) for column in columns))
    for row in rows:
        print("  " + "  ".join(str(row.get(column, "")).rjust(widths[column]) for column in columns))


def main() -> None:
    args = parse_args()
    lookup_rows = _run_lookup_sweep(args)
    _print_table("lookup sweep", lookup_rows)
    if not args.skip_live:
        live_rows = _run_live_sweep(args)
        print()
        _print_table("live sweep", live_rows)
    print()
    print(
        json.dumps(
            {
                "lookup_rows": lookup_rows,
                "live_rows": [] if args.skip_live else live_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
