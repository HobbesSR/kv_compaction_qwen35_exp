from __future__ import annotations

from pathlib import Path

from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.context_loader import load_context_sample
from kv_compaction_qwen35_clean.model_runtime import build_runtime_plan, write_runtime_plan


def main() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)
    plan = build_runtime_plan(sample, config)
    output_path = Path("artifacts/qwen35_smoke/runtime_plan.json")
    write_runtime_plan(plan, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
