from __future__ import annotations

from dataclasses import dataclass
import json

from kv_compaction_qwen35_clean.segment_compaction_cache import (
    CachedPrefixLookup,
    CachedPrefixMetadataLookup,
    find_cached_prefix,
    find_cached_prefix_metadata,
)


@dataclass(frozen=True)
class CanonicalChatTranscript:
    message_token_ids: list[int]
    generation_token_ids: list[int]
    turn_spans: list[tuple[int, int, str, str]]
    generation_prompt_start: int


def _fallback_turn_text(role: str, turn_id: str, content: str) -> str:
    return f"{role.upper()} [{turn_id}]\n{content}\n\n"


def _message_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, sort_keys=True, separators=(",", ":"))


def _normalize_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "user"))
        entry = dict(message)
        entry.setdefault("content", "")
        entry["_segment_turn_id"] = str(message.get("turn_id", f"turn_{index}"))
        entry["role"] = role
        normalized.append(entry)
    return normalized


def _normalize_qwen35_tool_calls(message: dict[str, object]) -> dict[str, object]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message

    normalized_calls: list[dict[str, object]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized_calls.append(tool_call)
            continue
        function_block = tool_call.get("function")
        if not isinstance(function_block, dict):
            normalized_calls.append(tool_call)
            continue
        normalized_function = dict(function_block)
        arguments = normalized_function.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"raw_arguments": arguments}
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
            normalized_function["arguments"] = parsed
        normalized_call = dict(tool_call)
        normalized_call["function"] = normalized_function
        normalized_calls.append(normalized_call)

    normalized_message = dict(message)
    normalized_message["tool_calls"] = normalized_calls
    return normalized_message


def _token_ids_from_template_output(rendered) -> list[int]:
    if isinstance(rendered, list):
        return [int(token_id) for token_id in rendered]
    if hasattr(rendered, "get"):
        input_ids = rendered.get("input_ids")
        if isinstance(input_ids, list):
            return [int(token_id) for token_id in input_ids]
    raise TypeError("Expected apply_chat_template(..., tokenize=True) to return token ids.")


def _render_chat_template_token_ids(tokenizer, conversation, *, add_generation_prompt: bool, enable_thinking: bool | None):
    try:
        rendered = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=None,
        )
    except Exception as exc:
        if "No user query found in messages." in str(exc):
            return None
        raise
    return _token_ids_from_template_output(rendered)


def canonicalize_openai_chat_messages(
    tokenizer,
    *,
    model_type: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
    enable_thinking: bool | None = None,
) -> CanonicalChatTranscript:
    normalized = _normalize_messages(messages)
    spans: list[tuple[int, int, str, str]] = []

    if model_type == "qwen3_5" and hasattr(tokenizer, "apply_chat_template"):
        stripped_messages = [
            _normalize_qwen35_tool_calls({k: v for k, v in message.items() if k != "_segment_turn_id"})
            for message in normalized
        ]
        def render_prefix(conversation, *, add_generation_prompt: bool):
            try:
                rendered = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                    tools=tools,
                )
            except Exception as exc:
                if "No user query found in messages." in str(exc):
                    return None
                raise
            return _token_ids_from_template_output(rendered)

        message_only_ids = render_prefix(
            stripped_messages,
            add_generation_prompt=False,
        )
        if message_only_ids is None:
            raise ValueError("Qwen3.5 chat canonicalization requires at least one user message.")
        cursor = 0
        for index in range(len(normalized)):
            prefix_ids = render_prefix(
                stripped_messages[: index + 1],
                add_generation_prompt=False,
            )
            if prefix_ids is None:
                continue
            turn_id = str(normalized[index]["_segment_turn_id"])
            role = str(normalized[index]["role"])
            spans.append((cursor, len(prefix_ids), turn_id, role))
            cursor = len(prefix_ids)
        if not spans and normalized:
            spans.append((0, len(message_only_ids), str(normalized[-1]["_segment_turn_id"]), str(normalized[-1]["role"])))
        generation_ids = render_prefix(
            stripped_messages,
            add_generation_prompt=True,
        )
        if generation_ids is None:
            raise ValueError("Qwen3.5 generation prompt requires at least one user message.")
        return CanonicalChatTranscript(
            message_token_ids=message_only_ids,
            generation_token_ids=generation_ids,
            turn_spans=spans,
            generation_prompt_start=len(message_only_ids),
        )

    token_ids: list[int] = []
    for message in normalized:
        rendered = _fallback_turn_text(
            str(message["role"]),
            str(message["_segment_turn_id"]),
            _message_content_text(message.get("content")),
        )
        block_ids = tokenizer.encode(rendered, add_special_tokens=False)
        start = len(token_ids)
        token_ids.extend(int(token_id) for token_id in block_ids)
        spans.append((start, len(token_ids), str(message["_segment_turn_id"]), str(message["role"])))

    generation_ids = list(token_ids)
    if normalized:
        assistant_prefix = tokenizer.encode(
            f"ASSISTANT [turn_{len(normalized)}]\n",
            add_special_tokens=False,
        )
        generation_ids.extend(int(token_id) for token_id in assistant_prefix)

    return CanonicalChatTranscript(
        message_token_ids=token_ids,
        generation_token_ids=generation_ids,
        turn_spans=spans,
        generation_prompt_start=len(token_ids),
    )


def find_cached_prefix_for_openai_messages(
    tokenizer,
    *,
    model_type: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
    config_fingerprint: str,
    cache_root,
    min_segment_tokens: int = 0,
    enable_thinking: bool | None = None,
    device: str = "cpu",
) -> tuple[CanonicalChatTranscript, CachedPrefixLookup]:
    transcript = canonicalize_openai_chat_messages(
        tokenizer,
        model_type=model_type,
        messages=messages,
        tools=tools,
        enable_thinking=enable_thinking,
    )
    lookup = find_cached_prefix(
        token_ids=transcript.message_token_ids,
        turn_spans=transcript.turn_spans,
        config_fingerprint=config_fingerprint,
        cache_root=cache_root,
        min_segment_tokens=min_segment_tokens,
        device=device,
    )
    return transcript, lookup


def find_cached_prefix_metadata_for_openai_messages(
    tokenizer,
    *,
    model_type: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
    config_fingerprint: str,
    cache_root,
    min_segment_tokens: int = 0,
    enable_thinking: bool | None = None,
) -> tuple[CanonicalChatTranscript, CachedPrefixMetadataLookup]:
    transcript = canonicalize_openai_chat_messages(
        tokenizer,
        model_type=model_type,
        messages=messages,
        tools=tools,
        enable_thinking=enable_thinking,
    )
    lookup = find_cached_prefix_metadata(
        token_ids=transcript.message_token_ids,
        turn_spans=transcript.turn_spans,
        config_fingerprint=config_fingerprint,
        cache_root=cache_root,
        min_segment_tokens=min_segment_tokens,
    )
    return transcript, lookup
