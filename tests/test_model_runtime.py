from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.context_loader import load_context_sample
from kv_compaction_qwen35_clean.model_runtime import build_runtime_plan, detect_runtime_dependencies


def test_build_qwen35_runtime_plan() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)

    plan = build_runtime_plan(sample, config)

    assert plan.model_name == "qwen3.5-9b"
    assert plan.huggingface_id == "Qwen/Qwen3.5-9B"
    assert plan.model_type == "qwen3_5"
    assert "warehouse migration" in plan.transcript_preview.lower()


def test_detect_runtime_dependencies_shape() -> None:
    status = detect_runtime_dependencies()

    assert isinstance(status.torch_available, bool)
    assert isinstance(status.qwen3_5_available, bool)
