from __future__ import annotations

from kv_compaction_qwen35_clean.service_demo import ServiceDemoSummary, format_progress_event


def test_format_progress_event_capture() -> None:
    event = {
        "stage": "capture",
        "processed_token_count": 128,
        "prefix_token_count": 512,
        "monitored_observation_count": 24,
        "monitored_query_sample_count": 12,
    }
    text = format_progress_event(event)
    assert "[capture]" in text
    assert "128/512" in text
    assert "obs=24" in text
    assert "queries=12" in text


def test_service_demo_summary_serialization() -> None:
    summary = ServiceDemoSummary(
        sample_id="sample",
        boundary_id="boundary",
        keys_per_head=6,
        compacted_head_count=4,
        effective_compact_tokens=96,
        prefix_token_count=6144,
        preserved_tail_tokens=1024,
        capture_token_count=28,
        monitored_observation_count=336,
        monitored_query_sample_count=336,
    )
    payload = summary.to_serializable()
    assert payload["effective_compact_tokens"] == 96
    assert payload["sample_id"] == "sample"
