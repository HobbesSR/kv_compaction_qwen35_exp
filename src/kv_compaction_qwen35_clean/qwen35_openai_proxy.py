from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.data_types import CompactHeadRuntime
from kv_compaction_qwen35_clean.model_runtime import (
    default_probe_heads_for_model,
    default_probe_layers_for_model,
    load_qwen35_bundle,
    unload_qwen35_bundle,
)
from kv_compaction_qwen35_clean.openai_chat_canonicalization import (
    CanonicalChatTranscript,
    canonicalize_openai_chat_messages,
    find_cached_prefix_metadata_for_openai_messages,
)
from kv_compaction_qwen35_clean.segment_compaction_cache import (
    CachedPrefixMetadataLookup,
    build_config_fingerprint,
    build_segment_bundle,
    build_turn_segment_lineage,
    write_segment_bundle,
)


DEFAULT_PROXY_KEYS_PER_HEAD = 8
DEFAULT_PROXY_MIN_SEGMENT_TOKENS = 1
DEFAULT_PROXY_MAX_TOKENS = 4096
EAGER_PREFILL_ATTENTION_BUDGET_BYTES = 64 * 1024 * 1024
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_TOOL_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _feed_tokens_with_cache(
    model,
    token_ids: list[int],
    *,
    device: str,
    past_key_values=None,
    start_position: int,
    chunk_size: int,
):
    import torch

    if not token_ids:
        return past_key_values, None

    input_tensor = torch.tensor([token_ids], device=device, dtype=torch.long)
    processed = 0
    outputs = None
    while processed < len(token_ids):
        chunk_end = min(len(token_ids), processed + chunk_size)
        chunk_ids = input_tensor[:, processed:chunk_end]
        cache_position = torch.arange(start_position + processed, start_position + chunk_end, device=device)
        with torch.inference_mode():
            outputs = model(
                input_ids=chunk_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                cache_position=cache_position,
            )
        past_key_values = outputs.past_key_values
        processed = chunk_end
    return past_key_values, outputs


def _torch_dtype_bytes(dtype) -> int:
    import torch

    if dtype in {torch.float16, torch.bfloat16, torch.int16, torch.uint16}:
        return 2
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float64, torch.int64}:
        return 8
    return 4


def _bounded_eager_prefill_chunk_size(
    *,
    requested_chunk_size: int,
    context_tokens: int,
    num_attention_heads: int,
    bytes_per_attention_element: int,
) -> int:
    if requested_chunk_size <= 0:
        raise ValueError("requested_chunk_size must be positive.")
    if context_tokens <= 0 or num_attention_heads <= 0 or bytes_per_attention_element <= 0:
        return min(requested_chunk_size, 128)

    budget_elements = EAGER_PREFILL_ATTENTION_BUDGET_BYTES // bytes_per_attention_element
    bounded_chunk_size = budget_elements // max(1, context_tokens * num_attention_heads)
    bounded_chunk_size = max(16, (bounded_chunk_size // 16) * 16)
    return min(requested_chunk_size, bounded_chunk_size)


def _effective_prefill_chunk_size(
    model,
    *,
    requested_chunk_size: int,
    context_tokens: int,
) -> int:
    attn_implementation = getattr(model.config, "_attn_implementation", None) or getattr(
        model.config,
        "attn_implementation",
        None,
    )
    if attn_implementation != "eager":
        return requested_chunk_size
    return _bounded_eager_prefill_chunk_size(
        requested_chunk_size=requested_chunk_size,
        context_tokens=context_tokens,
        num_attention_heads=int(getattr(model.config, "num_attention_heads", 0) or 0),
        bytes_per_attention_element=_torch_dtype_bytes(model.dtype),
    )


def _sample_next_token(logits, *, temperature: float, top_p: float) -> int:
    import torch

    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    scaled_logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(scaled_logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
        probs = probs / torch.clamp_min(probs.sum(dim=-1, keepdim=True), 1e-12)
    return int(torch.multinomial(probs, num_samples=1).item())


def _cleanup_generated_text(text: str) -> str:
    return text.strip()


def _finish_reason_for_generation(
    *,
    hit_eos: bool,
    generated_token_count: int,
    max_tokens: int,
) -> str:
    if hit_eos:
        return "stop"
    if max_tokens > 0 and generated_token_count >= max_tokens:
        return "length"
    return "stop"


def _strip_thinking_content(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return text


def _coerce_tool_argument_value(raw_value: str):
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _parse_qwen_tool_calls(text: str) -> tuple[str | None, list[dict[str, object]]]:
    cleaned = _strip_thinking_content(text).strip()
    matches = list(_TOOL_CALL_BLOCK_RE.finditer(cleaned))
    if not matches:
        return cleaned or None, []

    prefix = cleaned[: matches[0].start()].strip() or None
    tool_calls: list[dict[str, object]] = []
    for match in matches:
        function_name = match.group(1).strip()
        body = match.group(2)
        arguments: dict[str, object] = {}
        for parameter_match in _TOOL_PARAMETER_RE.finditer(body):
            parameter_name = parameter_match.group(1).strip()
            raw_value = parameter_match.group(2).strip()
            arguments[parameter_name] = _coerce_tool_argument_value(raw_value)
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments),
                },
            }
        )
    return prefix, tool_calls


def _tool_calls_for_template(tool_calls: list[dict[str, object]]) -> list[dict[str, object]]:
    template_calls: list[dict[str, object]] = []
    for tool_call in tool_calls:
        function_block = tool_call.get("function")
        if not isinstance(function_block, dict):
            continue
        arguments = function_block.get("arguments", {})
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {"raw_arguments": arguments}
        else:
            parsed_arguments = arguments
        if not isinstance(parsed_arguments, dict):
            parsed_arguments = {"value": parsed_arguments}
        template_calls.append(
            {
                "id": str(tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}")),
                "type": "function",
                "function": {
                    "name": str(function_block.get("name", "")),
                    "arguments": parsed_arguments,
                },
            }
        )
    return template_calls


@dataclass(frozen=True)
class ProxyRequestMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    logical_tokens: int
    cached_segments: int
    first_uncached_turn_index: int
    cached_kv_slots: int
    cached_unique_tokens: int
    deepest_cached_segment_hash: str | None
    uncached_segment_active: bool
    uncached_segment_start_turn_index: int | None
    uncached_segment_start_token: int
    uncached_segment_end_token: int
    bundles_written: int
    bundle_write_status: str
    generation_seconds: float
    finish_reason: str


@dataclass(frozen=True)
class UncachedSegmentRange:
    active: bool
    start_turn_index: int | None
    start_token: int
    end_token: int
    generation_prompt_start: int


@dataclass(frozen=True)
class SegmentKVSlice:
    layer: int
    key_states: object
    value_states: object


def split_completed_transcript_trailer(
    completed_token_ids: list[int],
    generated_path_token_ids: list[int],
) -> tuple[bool, list[int]]:
    if completed_token_ids == generated_path_token_ids:
        return True, []
    if len(completed_token_ids) >= len(generated_path_token_ids) and completed_token_ids[: len(generated_path_token_ids)] == generated_path_token_ids:
        return True, completed_token_ids[len(generated_path_token_ids) :]
    return False, []


def extract_full_attention_segment_kv(
    past_key_values,
    *,
    layer_indices: tuple[int, ...],
    start_token: int,
    end_token: int,
) -> dict[int, SegmentKVSlice]:
    if not hasattr(past_key_values, "key_cache") or not hasattr(past_key_values, "value_cache"):
        raise TypeError("past_key_values does not expose key_cache/value_cache.")
    if start_token < 0 or end_token < start_token:
        raise ValueError("Invalid segment token range.")

    slices: dict[int, SegmentKVSlice] = {}
    for layer_idx in layer_indices:
        key_entry = past_key_values.key_cache[int(layer_idx)]
        value_entry = past_key_values.value_cache[int(layer_idx)]
        if key_entry is None or value_entry is None:
            continue
        if not hasattr(key_entry, "shape") or not hasattr(value_entry, "shape"):
            continue
        if len(key_entry.shape) != 4 or len(value_entry.shape) != 4:
            continue
        seq_len = int(key_entry.shape[2])
        slice_end = min(int(end_token), seq_len)
        slice_start = min(int(start_token), slice_end)
        slices[int(layer_idx)] = SegmentKVSlice(
            layer=int(layer_idx),
            key_states=key_entry[:, :, slice_start:slice_end, :].clone(),
            value_states=value_entry[:, :, slice_start:slice_end, :].clone(),
        )
    return slices


def build_recency_compacted_layers(
    *,
    segment_kv_slices: dict[int, SegmentKVSlice],
    target_layer_heads: tuple[tuple[int, int], ...],
    segment_start_token: int,
    keys_per_head: int,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> dict[int, dict[int, CompactHeadRuntime]]:
    import torch

    repeat_factor = max(1, int(num_attention_heads) // max(1, int(num_key_value_heads)))
    compacted_layers: dict[int, dict[int, CompactHeadRuntime]] = {}
    for layer_idx, head_idx in target_layer_heads:
        kv_slice = segment_kv_slices.get(int(layer_idx))
        if kv_slice is None:
            continue
        key_states = kv_slice.key_states
        value_states = kv_slice.value_states
        if not hasattr(key_states, "shape") or len(key_states.shape) != 4:
            continue
        seq_len = int(key_states.shape[2])
        if seq_len <= 0:
            continue
        keep_count = min(int(keys_per_head), seq_len)
        rel_start = seq_len - keep_count
        kv_head_index = min(int(num_key_value_heads) - 1, int(head_idx) // repeat_factor)
        compact_keys = key_states[0, kv_head_index, rel_start:seq_len, :].detach()
        compact_values = value_states[0, kv_head_index, rel_start:seq_len, :].detach()
        beta = torch.zeros((keep_count,), device=compact_keys.device, dtype=compact_keys.dtype)
        selected_indices = list(range(int(segment_start_token) + rel_start, int(segment_start_token) + seq_len))
        compacted_layers.setdefault(int(layer_idx), {})[int(head_idx)] = CompactHeadRuntime(
            layer=int(layer_idx),
            head=int(head_idx),
            selected_indices=selected_indices,
            compact_keys=compact_keys,
            compact_values=compact_values,
            beta=beta.detach(),
        )
    return compacted_layers


def _first_uncached_segment_index(lineage, first_uncached_turn_index: int) -> int:
    for index, node in enumerate(lineage):
        if node.last_turn_index >= int(first_uncached_turn_index):
            return index
    return len(lineage)


def resolve_uncached_segment_range(
    transcript: CanonicalChatTranscript,
    lookup: CachedPrefixMetadataLookup,
) -> UncachedSegmentRange:
    generation_prompt_start = int(transcript.generation_prompt_start)
    if not transcript.turn_spans:
        return UncachedSegmentRange(
            active=False,
            start_turn_index=None,
            start_token=0,
            end_token=generation_prompt_start,
            generation_prompt_start=generation_prompt_start,
        )
    if lookup.first_uncached_turn_index >= len(transcript.turn_spans):
        return UncachedSegmentRange(
            active=False,
            start_turn_index=None,
            start_token=generation_prompt_start,
            end_token=generation_prompt_start,
            generation_prompt_start=generation_prompt_start,
        )
    start_token = int(transcript.turn_spans[lookup.first_uncached_turn_index][0])
    return UncachedSegmentRange(
        active=True,
        start_turn_index=int(lookup.first_uncached_turn_index),
        start_token=start_token,
        end_token=generation_prompt_start,
        generation_prompt_start=generation_prompt_start,
    )


class Qwen35OpenAIProxyService:
    def __init__(
        self,
        *,
        config_path: str | Path,
        cache_root: str | Path,
        request_log_path: str | Path | None = None,
        inference_log_path: str | Path | None = None,
    ) -> None:
        self.config_path = str(config_path)
        self.config = load_config(config_path)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.request_log_path = Path(request_log_path) if request_log_path is not None else None
        if inference_log_path is None and self.request_log_path is not None:
            inference_log_path = self.request_log_path.with_name(
                f"{self.request_log_path.stem}_inference{self.request_log_path.suffix}"
            )
        self.inference_log_path = Path(inference_log_path) if inference_log_path is not None else None
        self._request_log_lock = Lock()
        self._inference_log_lock = Lock()
        if self.request_log_path is not None:
            self.request_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.inference_log_path is not None:
            self.inference_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.keys_per_head = DEFAULT_PROXY_KEYS_PER_HEAD
        self.min_segment_tokens = DEFAULT_PROXY_MIN_SEGMENT_TOKENS
        self.model, self.tokenizer, self.model_type = load_qwen35_bundle(self.config)
        self._request_lock = Lock()
        if self.model_type != "qwen3_5":
            self.close()
            raise ValueError("Qwen35OpenAIProxyService currently supports qwen3_5 models only.")
        self.target_layer_heads = tuple(
            (int(layer), int(head))
            for layer in default_probe_layers_for_model(self.model, self.model_type)
            for head in default_probe_heads_for_model(self.model)
        )
        self.full_attention_layers = tuple(sorted({layer for layer, _ in self.target_layer_heads}))
        self.num_attention_heads = int(getattr(self.model.config, "num_attention_heads", 0))
        self.num_key_value_heads = int(getattr(self.model.config, "num_key_value_heads", self.num_attention_heads))
        enable_thinking = self.config.model.enable_thinking
        if enable_thinking is None:
            enable_thinking = False
        self.enable_thinking = enable_thinking
        self.config_fingerprint = build_config_fingerprint(
            model_name=self.config.model.name,
            huggingface_id=self.config.model.huggingface_id,
            tokenizer_name=self.config.model.tokenizer_name,
            tokenizer_fingerprint=f"{self.config.model.tokenizer_name}:{self.enable_thinking}",
            target_layer_heads=self.target_layer_heads,
            keys_per_head=self.keys_per_head,
            key_selection_method=self.config.compaction.key_selection,
            beta_solver="service_baseline",
            beta_regularization_strength=0.0,
            value_regularization_strength=0.0,
        )

    def close(self) -> None:
        unload_qwen35_bundle(self.model)

    def log_raw_request(
        self,
        *,
        request_path: str,
        headers: dict[str, str],
        body: bytes,
        client_host: str | None,
    ) -> None:
        if self.request_log_path is None:
            return
        entry = {
            "ts": round(time.time(), 6),
            "path": request_path,
            "client_host": client_host,
            "headers": headers,
            "body": body.decode("utf-8", errors="replace"),
        }
        with self._request_log_lock:
            with self.request_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

    def log_inference_round(
        self,
        *,
        messages: list[dict[str, object]],
        transcript,
        tools: list[dict[str, object]] | None,
        generated_token_ids: list[int],
        generated_text: str,
        assistant_content: str | None,
        parsed_tool_calls: list[dict[str, object]],
        finish_reason: str,
        max_tokens: int,
    ) -> None:
        if self.inference_log_path is None:
            return
        entry = {
            "ts": round(time.time(), 6),
            "message_count": len(messages),
            "message_roles": [str(message.get("role")) for message in messages],
            "assistant_message_count": sum(1 for message in messages if message.get("role") == "assistant"),
            "tool_message_count": sum(1 for message in messages if message.get("role") == "tool"),
            "tool_schema_count": len(tools or []),
            "max_tokens": max_tokens,
            "message_token_ids": list(transcript.message_token_ids),
            "generation_token_ids": list(transcript.generation_token_ids),
            "generation_prompt_start": int(transcript.generation_prompt_start),
            "generation_prompt_text": self.tokenizer.decode(
                transcript.generation_token_ids,
                skip_special_tokens=False,
            ),
            "generated_token_ids": list(generated_token_ids),
            "generated_text": generated_text,
            "assistant_content": assistant_content,
            "parsed_tool_calls": parsed_tool_calls,
            "finish_reason": finish_reason,
        }
        with self._inference_log_lock:
            with self.inference_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

    def list_models_payload(self) -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.config.model.huggingface_id,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    def complete(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[dict[str, object], ProxyRequestMetrics]:
        import torch

        with self._request_lock:
            print(
                f"[proxy] request: {len(messages)} messages, "
                f"last_role={messages[-1].get('role') if messages else None}, "
                f"max_tokens={max_tokens}",
                flush=True,
            )
            transcript, lookup = find_cached_prefix_metadata_for_openai_messages(
                self.tokenizer,
                model_type=self.model_type,
                messages=messages,
                tools=tools,
                config_fingerprint=self.config_fingerprint,
                cache_root=self.cache_root,
                min_segment_tokens=self.min_segment_tokens,
                enable_thinking=self.enable_thinking,
            )
            uncached_segment = resolve_uncached_segment_range(transcript, lookup)
            prompt_ids = transcript.generation_token_ids
            generation_start = time.perf_counter()
            effective_chunk_size = _effective_prefill_chunk_size(
                self.model,
                requested_chunk_size=self.config.model.prefill_chunk_size,
                context_tokens=len(prompt_ids),
            )
            past_key_values, outputs = _feed_tokens_with_cache(
                self.model,
                prompt_ids,
                device=self.config.model.device,
                start_position=0,
                chunk_size=effective_chunk_size,
            )
            if outputs is None:
                raise ValueError("Prompt processing did not yield logits.")
            logits = outputs.logits[:, -1, :]
            logical_position = len(prompt_ids)
            generated_token_ids: list[int] = []
            hit_eos = False
            for _ in range(max_tokens):
                next_token = _sample_next_token(logits, temperature=temperature, top_p=top_p)
                if self.tokenizer.eos_token_id is not None and next_token == self.tokenizer.eos_token_id:
                    hit_eos = True
                    break
                generated_token_ids.append(next_token)
                token_tensor = torch.tensor([[next_token]], device=self.config.model.device, dtype=torch.long)
                cache_position = torch.tensor([logical_position], device=self.config.model.device, dtype=torch.long)
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=token_tensor,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                        cache_position=cache_position,
                    )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                logical_position += 1
            generated_text = _cleanup_generated_text(self.tokenizer.decode(generated_token_ids, skip_special_tokens=True))
            assistant_content, parsed_tool_calls = (
                _parse_qwen_tool_calls(generated_text)
                if tools
                else (_strip_thinking_content(generated_text).strip() or None, [])
            )
            bundles_written = 0
            bundle_write_status = "no_uncached_segment"
            if uncached_segment.active:
                completed_assistant_message: dict[str, object] = {
                    "role": "assistant",
                    "content": assistant_content or "",
                }
                if parsed_tool_calls:
                    completed_assistant_message["tool_calls"] = _tool_calls_for_template(parsed_tool_calls)
                completed_messages = list(messages) + [completed_assistant_message]
                completed_transcript = canonicalize_openai_chat_messages(
                    self.tokenizer,
                    model_type=self.model_type,
                    messages=completed_messages,
                    tools=tools,
                    enable_thinking=self.enable_thinking,
                )
                expected_completed_token_ids = prompt_ids + generated_token_ids
                completed_prefix_matches, completed_trailer = split_completed_transcript_trailer(
                    completed_transcript.message_token_ids,
                    expected_completed_token_ids,
                )
                if not completed_prefix_matches:
                    bundle_write_status = "skipped_template_mismatch"
                else:
                    if completed_trailer:
                        past_key_values, _ = _feed_tokens_with_cache(
                            self.model,
                            completed_trailer,
                            device=self.config.model.device,
                            past_key_values=past_key_values,
                            start_position=len(expected_completed_token_ids),
                            chunk_size=max(1, min(effective_chunk_size, 64)),
                        )
                    completed_lineage = build_turn_segment_lineage(
                        token_ids=completed_transcript.message_token_ids,
                        turn_spans=completed_transcript.turn_spans,
                        config_fingerprint=self.config_fingerprint,
                        min_segment_tokens=self.min_segment_tokens,
                    )
                    start_segment_index = _first_uncached_segment_index(
                        completed_lineage,
                        lookup.first_uncached_turn_index,
                    )
                    for node in completed_lineage[start_segment_index:]:
                        segment_kv_slices = extract_full_attention_segment_kv(
                            past_key_values,
                            layer_indices=self.full_attention_layers,
                            start_token=node.boundary.segment_start_token,
                            end_token=node.boundary.segment_end_token,
                        )
                        compacted_layers = build_recency_compacted_layers(
                            segment_kv_slices=segment_kv_slices,
                            target_layer_heads=self.target_layer_heads,
                            segment_start_token=node.boundary.segment_start_token,
                            keys_per_head=self.keys_per_head,
                            num_attention_heads=self.num_attention_heads,
                            num_key_value_heads=self.num_key_value_heads,
                        )
                        if not compacted_layers:
                            continue
                        bundle = build_segment_bundle(
                            parent_hash=node.parent_hash,
                            segment_token_ids=completed_transcript.message_token_ids[
                                node.boundary.segment_start_token : node.boundary.segment_end_token
                            ],
                            boundary_turn_index=node.boundary.boundary_turn_index,
                            segment_start_token=node.boundary.segment_start_token,
                            segment_end_token=node.boundary.segment_end_token,
                            logical_token_count_before=node.logical_token_count_before,
                            logical_token_count_after=node.logical_token_count_after,
                            model_name=self.config.model.name,
                            huggingface_id=self.config.model.huggingface_id,
                            tokenizer_name=self.config.model.tokenizer_name,
                            tokenizer_fingerprint=f"{self.config.model.tokenizer_name}:{self.enable_thinking}",
                            config_fingerprint=self.config_fingerprint,
                            target_layer_heads=self.target_layer_heads,
                            compacted_layers=compacted_layers,
                        )
                        write_segment_bundle(bundle, self.cache_root)
                        bundles_written += 1
                    if bundles_written > 0:
                        bundle_write_status = "bundles_written_with_trailer" if completed_trailer else "bundles_written"
                    else:
                        bundle_write_status = "no_full_attention_rows"

        generation_seconds = round(time.perf_counter() - generation_start, 6)
        cached_kv_slots = sum(metadata.physical_compacted_kv_slot_count for metadata in lookup.bundle_metadata)
        cached_unique_tokens = sum(
            metadata.physical_compacted_unique_token_count for metadata in lookup.bundle_metadata
        )
        deepest_hash = lookup.bundle_metadata[-1].segment_hash if lookup.bundle_metadata else None
        metrics = ProxyRequestMetrics(
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(generated_token_ids),
            total_tokens=len(prompt_ids) + len(generated_token_ids),
            logical_tokens=len(transcript.message_token_ids),
            cached_segments=len(lookup.bundle_metadata),
            first_uncached_turn_index=lookup.first_uncached_turn_index,
            cached_kv_slots=cached_kv_slots,
            cached_unique_tokens=cached_unique_tokens,
            deepest_cached_segment_hash=deepest_hash,
            uncached_segment_active=uncached_segment.active,
            uncached_segment_start_turn_index=uncached_segment.start_turn_index,
            uncached_segment_start_token=uncached_segment.start_token,
            uncached_segment_end_token=uncached_segment.end_token,
            bundles_written=bundles_written,
            bundle_write_status=bundle_write_status,
            generation_seconds=generation_seconds,
            finish_reason="stop",
        )
        finish_reason = _finish_reason_for_generation(
            hit_eos=hit_eos,
            generated_token_count=len(generated_token_ids),
            max_tokens=max_tokens,
        )
        if parsed_tool_calls:
            finish_reason = "tool_calls"
        metrics = ProxyRequestMetrics(
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            total_tokens=metrics.total_tokens,
            logical_tokens=metrics.logical_tokens,
            cached_segments=metrics.cached_segments,
            first_uncached_turn_index=metrics.first_uncached_turn_index,
            cached_kv_slots=metrics.cached_kv_slots,
            cached_unique_tokens=metrics.cached_unique_tokens,
            deepest_cached_segment_hash=metrics.deepest_cached_segment_hash,
            uncached_segment_active=metrics.uncached_segment_active,
            uncached_segment_start_turn_index=metrics.uncached_segment_start_turn_index,
            uncached_segment_start_token=metrics.uncached_segment_start_token,
            uncached_segment_end_token=metrics.uncached_segment_end_token,
            bundles_written=metrics.bundles_written,
            bundle_write_status=metrics.bundle_write_status,
            generation_seconds=metrics.generation_seconds,
            finish_reason=finish_reason,
        )
        print(
            f"[proxy] response: finish_reason={finish_reason}, "
            f"completion_tokens={len(generated_token_ids)}",
            flush=True,
        )
        self.log_inference_round(
            messages=messages,
            transcript=transcript,
            tools=tools,
            generated_token_ids=generated_token_ids,
            generated_text=generated_text,
            assistant_content=assistant_content,
            parsed_tool_calls=parsed_tool_calls,
            finish_reason=finish_reason,
            max_tokens=max_tokens,
        )
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model.huggingface_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_content,
                        **({"tool_calls": parsed_tool_calls} if parsed_tool_calls else {}),
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_tokens": metrics.total_tokens,
            },
            "kv_compaction": {
                "mode": "observe_only_baseline",
                "logical_tokens": metrics.logical_tokens,
                "cached_segments": metrics.cached_segments,
                "first_uncached_turn_index": metrics.first_uncached_turn_index,
                "cached_kv_slots": metrics.cached_kv_slots,
                "cached_unique_tokens": metrics.cached_unique_tokens,
                "deepest_cached_segment_hash": metrics.deepest_cached_segment_hash,
                "uncached_segment_active": metrics.uncached_segment_active,
                "uncached_segment_start_turn_index": metrics.uncached_segment_start_turn_index,
                "uncached_segment_start_token": metrics.uncached_segment_start_token,
                "uncached_segment_end_token": metrics.uncached_segment_end_token,
                "bundles_written": metrics.bundles_written,
                "bundle_write_status": metrics.bundle_write_status,
                "generation_seconds": metrics.generation_seconds,
            },
        }
        return payload, metrics


def create_proxy_handler(service: Qwen35OpenAIProxyService):
    class ProxyHandler(BaseHTTPRequestHandler):
        server_version = "Qwen35OpenAIProxy/0.1"

        @staticmethod
        def _metrics_headers(metrics: ProxyRequestMetrics) -> dict[str, str]:
            return {
                "X-KV-Logical-Tokens": str(metrics.logical_tokens),
                "X-KV-Cached-Segments": str(metrics.cached_segments),
                "X-KV-First-Uncached-Turn": str(metrics.first_uncached_turn_index),
                "X-KV-Cached-KV-Slots": str(metrics.cached_kv_slots),
                "X-KV-Cached-Unique-Tokens": str(metrics.cached_unique_tokens),
            }

        def _write_json(self, status: HTTPStatus, payload: dict[str, object], *, extra_headers: dict[str, str] | None = None) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _write_sse_completion(
            self,
            *,
            response_payload: dict[str, object],
            metrics: ProxyRequestMetrics,
        ) -> None:
            choice = dict(response_payload["choices"][0])
            message = dict(choice.get("message", {}))
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            shared = {
                "id": str(response_payload["id"]),
                "object": "chat.completion.chunk",
                "created": int(response_payload["created"]),
                "model": str(response_payload["model"]),
            }
            delta: dict[str, object] = {"role": "assistant"}
            if content not in (None, ""):
                delta["content"] = str(content)
            if isinstance(tool_calls, list) and tool_calls:
                delta["tool_calls"] = [
                    {
                        "index": index,
                        "id": str(tool_call.get("id")),
                        "type": "function",
                        "function": dict(tool_call.get("function", {})),
                    }
                    for index, tool_call in enumerate(tool_calls)
                ]
            first_chunk = {
                **shared,
                "choices": [
                    {
                        "index": int(choice.get("index", 0)),
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }
            final_chunk = {
                **shared,
                "choices": [
                    {
                        "index": int(choice.get("index", 0)),
                        "delta": {},
                        "finish_reason": str(choice.get("finish_reason", metrics.finish_reason)),
                    }
                ],
            }
            body = (
                f"data: {json.dumps(first_chunk)}\n\n"
                f"data: {json.dumps(final_chunk)}\n\n"
                "data: [DONE]\n\n"
            ).encode("utf-8")
            self.send_response(int(HTTPStatus.OK))
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(body)))
            for key, value in self._metrics_headers(metrics).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._write_json(HTTPStatus.OK, {"ok": True, "model": service.config.model.huggingface_id})
                return
            if self.path == "/v1/models":
                self._write_json(HTTPStatus.OK, service.list_models_payload())
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": {"message": "invalid Content-Length"}})
                return
            raw_body = self.rfile.read(content_length) or b"{}"
            service.log_raw_request(
                request_path=self.path,
                headers=dict(self.headers.items()),
                body=raw_body,
                client_host=self.client_address[0] if self.client_address else None,
            )
            payload = json.loads(raw_body)
            wants_stream = bool(payload.get("stream"))
            messages = payload.get("messages")
            tools = payload.get("tools")
            if not isinstance(messages, list) or not messages:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": {"message": "messages must be a non-empty list"}})
                return
            if tools is not None and not isinstance(tools, list):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": {"message": "tools must be a list when provided"}})
                return
            if payload.get("model") not in (None, service.config.model.huggingface_id, service.config.model.name):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": {"message": "unknown model"}})
                return
            max_tokens = int(payload.get("max_completion_tokens", payload.get("max_tokens", DEFAULT_PROXY_MAX_TOKENS)))
            temperature = float(payload.get("temperature", 0.0))
            top_p = float(payload.get("top_p", 1.0))
            try:
                response_payload, metrics = service.complete(
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as exc:  # pragma: no cover
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": {"message": f"{type(exc).__name__}: {exc}"}},
                )
                return
            if wants_stream:
                self._write_sse_completion(
                    response_payload=response_payload,
                    metrics=metrics,
                )
                return
            self._write_json(
                HTTPStatus.OK,
                response_payload,
                extra_headers=self._metrics_headers(metrics),
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return ProxyHandler


def run_proxy(
    *,
    config_path: str | Path,
    cache_root: str | Path,
    host: str,
    port: int,
    request_log_path: str | Path | None = None,
    inference_log_path: str | Path | None = None,
) -> None:
    service = Qwen35OpenAIProxyService(
        config_path=config_path,
        cache_root=cache_root,
        request_log_path=request_log_path,
        inference_log_path=inference_log_path,
    )
    server = ThreadingHTTPServer((host, port), create_proxy_handler(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


def _parse_proxy_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal OpenAI-compatible Qwen3.5 local proxy.")
    parser.add_argument(
        "--config",
        default="configs/qwen35_smoke/qwen3_5_9b.yaml",
        help="Experiment config used to load the dense Qwen3.5 model.",
    )
    parser.add_argument(
        "--cache-root",
        default="artifacts/qwen35_proxy_cache",
        help="Directory for segment bundle cache lookup/storage.",
    )
    parser.add_argument(
        "--request-log",
        default=None,
        help="Optional JSONL file to append exact raw request payloads to before parsing.",
    )
    parser.add_argument(
        "--inference-log",
        default=None,
        help="Optional JSONL file to append exact prompt/output token traces for each inference round.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    return parser.parse_args()


def main() -> None:
    args = _parse_proxy_args()
    run_proxy(
        config_path=args.config,
        cache_root=args.cache_root,
        request_log_path=args.request_log,
        inference_log_path=args.inference_log,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
