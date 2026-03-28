from pathlib import Path

from kv_compaction_qwen35_clean.segment_compaction_cache import build_segment_bundle, write_segment_bundle
from kv_compaction_qwen35_clean.openai_chat_canonicalization import (
    canonicalize_openai_chat_messages,
    find_cached_prefix_for_openai_messages,
    find_cached_prefix_metadata_for_openai_messages,
)
from kv_compaction_qwen35_clean.data_types import CompactHeadRuntime

import torch


class _TemplateTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        tokens = []
        for message in conversation:
            role = str(message["role"])
            content = str(message.get("content", ""))
            tokens.extend([len(role), len(content)])
        if kwargs.get("add_generation_prompt"):
            tokens.append(999)
        return tokens


class _BatchEncodingTemplateTokenizer(_TemplateTokenizer):
    def apply_chat_template(self, conversation, **kwargs):
        return {"input_ids": super().apply_chat_template(conversation, **kwargs)}


class _Qwen35LikeTemplateTokenizer(_BatchEncodingTemplateTokenizer):
    def apply_chat_template(self, conversation, **kwargs):
        if conversation and all(str(message["role"]) == "system" for message in conversation):
            raise RuntimeError("No user query found in messages.")
        return super().apply_chat_template(conversation, **kwargs)


class _FallbackTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def _sample_layers():
    return {
        4: {
            0: CompactHeadRuntime(
                layer=4,
                head=0,
                selected_indices=[1, 2],
                compact_keys=torch.tensor([[1.0, 2.0]]),
                compact_values=torch.tensor([[3.0, 4.0]]),
                beta=torch.tensor([[1.0], [0.0]]),
            )
        }
    }


def test_canonicalize_openai_chat_messages_uses_incremental_chat_template() -> None:
    tokenizer = _TemplateTokenizer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    transcript = canonicalize_openai_chat_messages(
        tokenizer,
        model_type="qwen3_5",
        messages=[
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "status?"},
            {"role": "assistant", "content": "working"},
        ],
        tools=tools,
        enable_thinking=False,
    )

    assert transcript.message_token_ids == [6, 10, 4, 7, 9, 7]
    assert transcript.turn_spans == [
        (0, 2, "turn_0", "system"),
        (2, 4, "turn_1", "user"),
        (4, 6, "turn_2", "assistant"),
    ]
    assert transcript.generation_prompt_start == 6
    assert transcript.generation_token_ids == [6, 10, 4, 7, 9, 7, 999]
    assert tokenizer.calls[0][1]["enable_thinking"] is False
    assert tokenizer.calls[0][1]["tools"] == tools


def test_canonicalize_openai_chat_messages_accepts_batchencoding_shape() -> None:
    tokenizer = _BatchEncodingTemplateTokenizer()
    transcript = canonicalize_openai_chat_messages(
        tokenizer,
        model_type="qwen3_5",
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
        enable_thinking=False,
    )

    assert transcript.message_token_ids == [4, 5, 9, 5]
    assert transcript.generation_token_ids == [4, 5, 9, 5, 999]


def test_canonicalize_openai_chat_messages_merges_leading_system_only_prefix_for_qwen35() -> None:
    tokenizer = _Qwen35LikeTemplateTokenizer()
    transcript = canonicalize_openai_chat_messages(
        tokenizer,
        model_type="qwen3_5",
        messages=[
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hello"},
        ],
        enable_thinking=False,
    )

    assert transcript.message_token_ids == [6, 5, 4, 5]
    assert transcript.turn_spans == [(0, 4, "turn_1", "user")]
    assert transcript.generation_prompt_start == 4


def test_canonicalize_openai_chat_messages_falls_back_to_plain_turn_rendering() -> None:
    tokenizer = _FallbackTokenizer()
    transcript = canonicalize_openai_chat_messages(
        tokenizer,
        model_type="qwen2",
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ],
    )

    assert transcript.turn_spans[0][0] == 0
    assert transcript.turn_spans[0][2:] == ("turn_0", "user")
    assert transcript.turn_spans[1][2:] == ("turn_1", "assistant")
    assert transcript.generation_prompt_start == len(transcript.message_token_ids)
    assert len(transcript.generation_token_ids) > len(transcript.message_token_ids)


def test_find_cached_prefix_for_openai_messages_reuses_cached_segments(tmp_path: Path) -> None:
    tokenizer = _TemplateTokenizer()
    messages = [
        {"role": "system", "content": "be precise"},
        {"role": "user", "content": "status?"},
        {"role": "assistant", "content": "working"},
    ]
    transcript = canonicalize_openai_chat_messages(
        tokenizer,
        model_type="qwen3_5",
        messages=messages,
        enable_thinking=False,
    )

    bundle = build_segment_bundle(
        parent_hash=None,
        segment_token_ids=transcript.message_token_ids[:4],
        boundary_turn_index=1,
        segment_start_token=0,
        segment_end_token=4,
        logical_token_count_before=0,
        logical_token_count_after=4,
        model_name="Qwen3.5-9B",
        huggingface_id="Qwen/Qwen3.5-9B",
        tokenizer_name="Qwen/Qwen3.5-9B",
        tokenizer_fingerprint="tok-v1",
        config_fingerprint="cfg-v1",
        target_layer_heads=((4, 0),),
        compacted_layers=_sample_layers(),
    )
    write_segment_bundle(bundle, tmp_path)

    transcript, lookup = find_cached_prefix_for_openai_messages(
        tokenizer,
        model_type="qwen3_5",
        messages=messages,
        config_fingerprint="cfg-v1",
        cache_root=tmp_path,
        min_segment_tokens=4,
        enable_thinking=False,
    )

    assert transcript.turn_spans == [
        (0, 2, "turn_0", "system"),
        (2, 4, "turn_1", "user"),
        (4, 6, "turn_2", "assistant"),
    ]
    assert len(lookup.bundles) == 1
    assert lookup.first_uncached_turn_index == 2
    assert lookup.bundles[0].metadata.logical_token_count_after == 4


def test_find_cached_prefix_metadata_for_openai_messages_reuses_cached_segments(tmp_path: Path) -> None:
    tokenizer = _TemplateTokenizer()
    messages = [
        {"role": "system", "content": "be precise"},
        {"role": "user", "content": "status?"},
        {"role": "assistant", "content": "working"},
    ]
    transcript = canonicalize_openai_chat_messages(
        tokenizer,
        model_type="qwen3_5",
        messages=messages,
        enable_thinking=False,
    )

    bundle = build_segment_bundle(
        parent_hash=None,
        segment_token_ids=transcript.message_token_ids[:4],
        boundary_turn_index=1,
        segment_start_token=0,
        segment_end_token=4,
        logical_token_count_before=0,
        logical_token_count_after=4,
        model_name="Qwen3.5-9B",
        huggingface_id="Qwen/Qwen3.5-9B",
        tokenizer_name="Qwen/Qwen3.5-9B",
        tokenizer_fingerprint="tok-v1",
        config_fingerprint="cfg-v1",
        target_layer_heads=((4, 0),),
        compacted_layers=_sample_layers(),
    )
    write_segment_bundle(bundle, tmp_path)

    transcript, lookup = find_cached_prefix_metadata_for_openai_messages(
        tokenizer,
        model_type="qwen3_5",
        messages=messages,
        config_fingerprint="cfg-v1",
        cache_root=tmp_path,
        min_segment_tokens=4,
        enable_thinking=False,
    )

    assert transcript.turn_spans == [
        (0, 2, "turn_0", "system"),
        (2, 4, "turn_1", "user"),
        (4, 6, "turn_2", "assistant"),
    ]
    assert len(lookup.bundle_metadata) == 1
    assert lookup.first_uncached_turn_index == 2
    assert lookup.bundle_metadata[0].logical_token_count_after == 4
