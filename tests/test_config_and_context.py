from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.context_loader import load_context_sample


def test_load_qwen35_config_and_context_sample() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)

    assert config.model.huggingface_id == "Qwen/Qwen3.5-9B"
    assert config.model.enable_thinking is False
    assert sample.sample_id == "local_qwen35_smoke_v0"
    assert sample.prompt_family == "warehouse_migration_qwen35"
    assert sample.boundary.prefix_token_count == 7168
    assert sample.boundary.target_context_tokens_after_compaction == 1741
