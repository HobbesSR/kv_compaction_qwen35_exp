from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from kv_compaction_qwen35_clean.data_types import (
    ContextChunk,
    ContextTurn,
    LoadedContextSample,
    PromptBoundary,
    SmokeTestConfig,
)


QWEN35_REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATASET_PATHS = {
    "local_placeholder_qwen35": QWEN35_REPO_ROOT / "data/smoke_test/local_placeholder_qwen35.json",
}


def _resolve_dataset_path(dataset: str) -> Path:
    if dataset in LOCAL_DATASET_PATHS:
        return LOCAL_DATASET_PATHS[dataset]
    candidate = Path(dataset)
    if not candidate.is_absolute():
        candidate = QWEN35_REPO_ROOT / candidate
    return candidate


def _turn_spans(turns: list[ContextTurn]) -> list[tuple[ContextTurn, int, int]]:
    spans: list[tuple[ContextTurn, int, int]] = []
    cursor = 0
    for turn in turns:
        next_cursor = cursor + turn.token_count
        spans.append((turn, cursor, next_cursor))
        cursor = next_cursor
    return spans


def _build_chunks(turns: list[ContextTurn], chunk_size: int) -> list[ContextChunk]:
    total_tokens = sum(turn.token_count for turn in turns)
    spans = _turn_spans(turns)
    chunks: list[ContextChunk] = []
    for chunk_index, start_token in enumerate(range(0, total_tokens, chunk_size)):
        end_token = min(total_tokens, start_token + chunk_size)
        turn_ids = [
            turn.turn_id
            for turn, turn_start, turn_end in spans
            if turn_start < end_token and turn_end > start_token
        ]
        chunks.append(
            ContextChunk(
                chunk_id=f"chunk_{chunk_index}",
                start_token=start_token,
                end_token=end_token,
                turn_ids=turn_ids,
            )
        )
    return chunks


def _load_raw_sample(dataset_path: Path, sample_id: str | None) -> dict[str, object]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = payload["samples"]
    if not samples:
        raise ValueError(f"Dataset {dataset_path} does not contain any samples.")
    if sample_id is None:
        return samples[0]
    for sample in samples:
        if sample["sample_id"] == sample_id:
            return sample
    raise ValueError(f"Sample {sample_id!r} was not found in dataset {dataset_path}.")


def load_context_sample(config: SmokeTestConfig, sample_id: str | None = None) -> LoadedContextSample:
    dataset_path = _resolve_dataset_path(config.data.dataset)
    raw_sample = _load_raw_sample(dataset_path, sample_id)
    turns = [ContextTurn(**turn) for turn in raw_sample["turns"]]
    total_tokens = sum(turn.token_count for turn in turns)
    if total_tokens != config.data.context_tokens:
        raise ValueError(
            "Configured context_tokens does not match the loaded sample: "
            f"{config.data.context_tokens} != {total_tokens}."
        )
    if total_tokens <= config.compaction.preserved_tail_tokens:
        raise ValueError("preserved_tail_tokens must be smaller than the loaded context length.")

    if config.data.chunking.enabled:
        chunks = _build_chunks(turns, config.data.chunking.chunk_size)
    else:
        chunks = [
            ContextChunk(
                chunk_id="chunk_0",
                start_token=0,
                end_token=total_tokens,
                turn_ids=[turn.turn_id for turn in turns],
            )
        ]

    prefix_token_count = total_tokens - config.compaction.preserved_tail_tokens
    target_prefix_tokens = math.ceil(prefix_token_count / config.compaction.target_compression_ratio)
    compaction_chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.start_token < prefix_token_count]
    boundary = PromptBoundary(
        boundary_id=f"{raw_sample['sample_id']}:prompt_boundary_0",
        boundary_type=config.compaction.boundary,
        prefix_token_count=prefix_token_count,
        preserved_tail_tokens=config.compaction.preserved_tail_tokens,
        logical_context_tokens=total_tokens,
        physical_context_tokens=total_tokens,
        target_context_tokens_after_compaction=target_prefix_tokens + config.compaction.preserved_tail_tokens,
        compaction_chunk_ids=compaction_chunk_ids,
        primary_prompt_label=config.data.branch_switch.primary_prompt_label,
        primary_prompt_text=config.data.branch_switch.primary_prompt_template,
        alternate_prompt_label=config.data.branch_switch.alternate_prompt_label,
        alternate_prompt_text=config.data.branch_switch.alternate_prompt_template,
    )

    return LoadedContextSample(
        sample_id=raw_sample["sample_id"],
        dataset=config.data.dataset,
        source=raw_sample["source"],
        task_label=raw_sample["task_label"],
        prompt_family=str(raw_sample.get("prompt_family", "warehouse_migration_qwen35")),
        turns=turns,
        chunks=chunks,
        logical_context_tokens=total_tokens,
        physical_context_tokens=total_tokens,
        boundary=boundary,
    )


def write_context_summary(sample: LoadedContextSample, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(sample), indent=2) + "\n", encoding="utf-8")
    return output_path
