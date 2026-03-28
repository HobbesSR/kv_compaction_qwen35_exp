from __future__ import annotations

from http.server import ThreadingHTTPServer
from types import SimpleNamespace
import json
import threading
import urllib.request

from kv_compaction_qwen35_clean.openai_chat_canonicalization import CanonicalChatTranscript
from kv_compaction_qwen35_clean.qwen35_openai_proxy import (
    _finish_reason_for_generation,
    _parse_qwen_tool_calls,
    ProxyRequestMetrics,
    Qwen35OpenAIProxyService,
    build_recency_compacted_layers,
    create_proxy_handler,
    SegmentKVSlice,
    extract_full_attention_segment_kv,
    resolve_uncached_segment_range,
    split_completed_transcript_trailer,
)
from kv_compaction_qwen35_clean.segment_compaction_cache import CachedPrefixLookup
import torch


class _FakeService:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model=SimpleNamespace(huggingface_id="Qwen/Qwen3.5-9B", name="qwen3.5-9b"))
        self.logged_requests = []

    def list_models_payload(self) -> dict[str, object]:
        return {"object": "list", "data": [{"id": self.config.model.huggingface_id, "object": "model"}]}

    def complete(self, *, messages, tools, max_tokens: int, temperature: float, top_p: float):
        assert messages
        assert tools is None or isinstance(tools, list)
        assert max_tokens == 8
        assert temperature == 0.0
        assert top_p == 1.0
        return (
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": self.config.model.huggingface_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                "kv_compaction": {
                    "mode": "observe_only_baseline",
                    "uncached_segment_active": True,
                    "uncached_segment_start_turn_index": 3,
                    "uncached_segment_start_token": 42,
                    "uncached_segment_end_token": 99,
                    "bundles_written": 1,
                    "bundle_write_status": "bundles_written",
                },
            },
            ProxyRequestMetrics(
                prompt_tokens=5,
                completion_tokens=1,
                total_tokens=6,
                logical_tokens=4,
                cached_segments=2,
                first_uncached_turn_index=3,
                cached_kv_slots=10,
                cached_unique_tokens=6,
                deepest_cached_segment_hash="abc123",
                uncached_segment_active=True,
                uncached_segment_start_turn_index=3,
                uncached_segment_start_token=42,
                uncached_segment_end_token=99,
                bundles_written=1,
                bundle_write_status="bundles_written",
                generation_seconds=0.123,
                finish_reason="stop",
            ),
        )

    def log_raw_request(self, *, request_path: str, headers: dict[str, str], body: bytes, client_host: str | None) -> None:
        self.logged_requests.append(
            {
                "path": request_path,
                "headers": headers,
                "body": body.decode("utf-8"),
                "client_host": client_host,
            }
        )


def _start_server(service=None):
    if service is None:
        service = _FakeService()
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_proxy_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, service


def test_proxy_health_and_models_endpoints() -> None:
    server, thread, _service = _start_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(f"{base}/healthz") as response:
            payload = json.loads(response.read())
        assert payload["ok"] is True
        with urllib.request.urlopen(f"{base}/v1/models") as response:
            payload = json.loads(response.read())
        assert payload["data"][0]["id"] == "Qwen/Qwen3.5-9B"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_proxy_chat_completion_endpoint_returns_openai_shape_and_headers() -> None:
    server, thread, _service = _start_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        request = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "Qwen/Qwen3.5-9B",
                    "messages": [{"role": "user", "content": "Say ok"}],
                    "max_tokens": 8,
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read())
            headers = dict(response.headers.items())
        assert payload["choices"][0]["message"]["content"] == "ok"
        assert payload["usage"]["total_tokens"] == 6
        assert headers["X-KV-Cached-Segments"] == "2"
        assert headers["X-KV-First-Uncached-Turn"] == "3"
        assert payload["kv_compaction"]["uncached_segment_active"] is True
        assert payload["kv_compaction"]["uncached_segment_start_turn_index"] == 3
        assert payload["kv_compaction"]["uncached_segment_start_token"] == 42
        assert payload["kv_compaction"]["uncached_segment_end_token"] == 99
        assert payload["kv_compaction"]["bundles_written"] == 1
        assert payload["kv_compaction"]["bundle_write_status"] == "bundles_written"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_proxy_streaming_requests_return_sse_completion_chunks() -> None:
    server, thread, _service = _start_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        request = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "Qwen/Qwen3.5-9B",
                    "messages": [{"role": "user", "content": "Say ok"}],
                    "max_tokens": 8,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
            headers = dict(response.headers.items())
        assert headers["Content-Type"].startswith("text/event-stream")
        assert headers["X-KV-Cached-Segments"] == "2"
        assert 'data: {"id": "chatcmpl-test", "object": "chat.completion.chunk"' in body
        assert '"delta": {"role": "assistant", "content": "ok"}' in body
        assert '"finish_reason": "stop"' in body
        assert body.endswith("data: [DONE]\n\n")
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_proxy_logs_raw_request_before_parsing() -> None:
    service = _FakeService()
    server, thread, service = _start_server(service)
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        request_body = {
            "model": "Qwen/Qwen3.5-9B",
            "messages": [{"role": "user", "content": "Say ok"}],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        request = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Test": "yes"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            json.loads(response.read())
        assert len(service.logged_requests) == 1
        logged = service.logged_requests[0]
        assert logged["path"] == "/v1/chat/completions"
        assert '"messages": [{"role": "user", "content": "Say ok"}]' in logged["body"]
        assert logged["headers"]["Content-Type"] == "application/json"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_log_inference_round_writes_prompt_and_output_trace(tmp_path) -> None:
    service = object.__new__(Qwen35OpenAIProxyService)
    service.inference_log_path = tmp_path / "inference.jsonl"
    service._inference_log_lock = threading.Lock()
    service.tokenizer = SimpleNamespace(
        decode=lambda token_ids, skip_special_tokens=False: "|".join(str(token_id) for token_id in token_ids)
    )
    transcript = CanonicalChatTranscript(
        message_token_ids=[10, 11, 12],
        generation_token_ids=[10, 11, 12, 13],
        turn_spans=[(0, 3, "turn_0", "user")],
        generation_prompt_start=3,
    )

    service.log_inference_round(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "prior"},
            {"role": "tool", "content": "tool result"},
        ],
        transcript=transcript,
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        generated_token_ids=[21, 22],
        generated_text="done",
        assistant_content="done",
        parsed_tool_calls=[],
        finish_reason="stop",
        max_tokens=128,
    )

    lines = service.inference_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["message_count"] == 4
    assert payload["message_roles"] == ["system", "user", "assistant", "tool"]
    assert payload["assistant_message_count"] == 1
    assert payload["tool_message_count"] == 1
    assert payload["tool_schema_count"] == 1
    assert payload["generation_token_ids"] == [10, 11, 12, 13]
    assert payload["generation_prompt_text"] == "10|11|12|13"
    assert payload["generated_token_ids"] == [21, 22]
    assert payload["generated_text"] == "done"
    assert payload["finish_reason"] == "stop"


def test_parse_qwen_tool_calls_extracts_function_calls() -> None:
    text = (
        "I will inspect the repo.\n\n"
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\nREADME.md\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    content, tool_calls = _parse_qwen_tool_calls(text)

    assert content == "I will inspect the repo."
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "README.md"}


def test_resolve_uncached_segment_range_returns_active_range_from_first_uncached_turn() -> None:
    transcript = CanonicalChatTranscript(
        message_token_ids=list(range(18)),
        generation_token_ids=list(range(25)),
        turn_spans=[
            (0, 6, "turn_0", "system"),
            (6, 12, "turn_1", "user"),
            (12, 18, "turn_2", "assistant"),
        ],
        generation_prompt_start=18,
    )
    lookup = CachedPrefixLookup(lineage=[], bundles=[], first_uncached_turn_index=1)

    result = resolve_uncached_segment_range(transcript, lookup)

    assert result.active is True
    assert result.start_turn_index == 1
    assert result.start_token == 6
    assert result.end_token == 18
    assert result.generation_prompt_start == 18


def test_resolve_uncached_segment_range_reports_no_active_range_when_all_turns_cached() -> None:
    transcript = CanonicalChatTranscript(
        message_token_ids=list(range(10)),
        generation_token_ids=list(range(14)),
        turn_spans=[
            (0, 4, "turn_0", "user"),
            (4, 10, "turn_1", "assistant"),
        ],
        generation_prompt_start=10,
    )
    lookup = CachedPrefixLookup(lineage=[], bundles=[], first_uncached_turn_index=2)

    result = resolve_uncached_segment_range(transcript, lookup)

    assert result.active is False
    assert result.start_turn_index is None
    assert result.start_token == 10
    assert result.end_token == 10


def test_extract_full_attention_segment_kv_slices_only_tensor_backed_layers() -> None:
    past_key_values = SimpleNamespace(
        key_cache=[
            torch.arange(1 * 4 * 10 * 2, dtype=torch.float32).reshape(1, 4, 10, 2),
            None,
            torch.arange(1000, 1000 + 1 * 4 * 10 * 2, dtype=torch.float32).reshape(1, 4, 10, 2),
        ],
        value_cache=[
            torch.arange(2000, 2000 + 1 * 4 * 10 * 2, dtype=torch.float32).reshape(1, 4, 10, 2),
            None,
            torch.arange(3000, 3000 + 1 * 4 * 10 * 2, dtype=torch.float32).reshape(1, 4, 10, 2),
        ],
    )

    slices = extract_full_attention_segment_kv(
        past_key_values,
        layer_indices=(0, 1, 2),
        start_token=3,
        end_token=7,
    )

    assert sorted(slices) == [0, 2]
    assert tuple(slices[0].key_states.shape) == (1, 4, 4, 2)
    assert tuple(slices[0].value_states.shape) == (1, 4, 4, 2)
    assert torch.equal(slices[0].key_states, past_key_values.key_cache[0][:, :, 3:7, :])
    assert torch.equal(slices[2].value_states, past_key_values.value_cache[2][:, :, 3:7, :])


def test_extract_full_attention_segment_kv_clamps_end_to_sequence_length() -> None:
    past_key_values = SimpleNamespace(
        key_cache=[torch.zeros((1, 4, 5, 2))],
        value_cache=[torch.ones((1, 4, 5, 2))],
    )

    slices = extract_full_attention_segment_kv(
        past_key_values,
        layer_indices=(0,),
        start_token=4,
        end_token=99,
    )

    assert tuple(slices[0].key_states.shape) == (1, 4, 1, 2)
    assert tuple(slices[0].value_states.shape) == (1, 4, 1, 2)


def test_build_recency_compacted_layers_uses_latest_positions_per_head() -> None:
    segment_kv_slices = {
        3: SegmentKVSlice(
            layer=3,
            key_states=torch.arange(1 * 4 * 6 * 2, dtype=torch.float32).reshape(1, 4, 6, 2),
            value_states=torch.arange(100, 100 + 1 * 4 * 6 * 2, dtype=torch.float32).reshape(1, 4, 6, 2),
        )
    }

    compacted_layers = build_recency_compacted_layers(
        segment_kv_slices=segment_kv_slices,
        target_layer_heads=((3, 0), (3, 7)),
        segment_start_token=20,
        keys_per_head=3,
        num_attention_heads=16,
        num_key_value_heads=4,
    )

    assert sorted(compacted_layers[3]) == [0, 7]
    runtime0 = compacted_layers[3][0]
    runtime7 = compacted_layers[3][7]
    assert runtime0.selected_indices == [23, 24, 25]
    assert runtime7.selected_indices == [23, 24, 25]
    assert tuple(runtime0.compact_keys.shape) == (3, 2)
    assert tuple(runtime7.compact_values.shape) == (3, 2)
    assert torch.equal(runtime0.beta, torch.zeros(3))


def test_split_completed_transcript_trailer_detects_template_suffix() -> None:
    ok, trailer = split_completed_transcript_trailer(
        completed_token_ids=[1, 2, 3, 4, 5],
        generated_path_token_ids=[1, 2, 3],
    )

    assert ok is True
    assert trailer == [4, 5]


def test_split_completed_transcript_trailer_rejects_prefix_mismatch() -> None:
    ok, trailer = split_completed_transcript_trailer(
        completed_token_ids=[1, 9, 3],
        generated_path_token_ids=[1, 2, 3],
    )

    assert ok is False
    assert trailer == []


def test_finish_reason_reports_length_when_generation_hits_max_tokens() -> None:
    assert _finish_reason_for_generation(hit_eos=False, generated_token_count=8, max_tokens=8) == "length"
    assert _finish_reason_for_generation(hit_eos=True, generated_token_count=3, max_tokens=8) == "stop"
    assert _finish_reason_for_generation(hit_eos=False, generated_token_count=0, max_tokens=8) == "stop"
