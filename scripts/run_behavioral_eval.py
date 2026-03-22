from __future__ import annotations

from pathlib import Path

from kv_compaction_qwen35_clean.behavioral_eval import run_behavioral_evaluation, write_behavioral_result
from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.context_loader import load_context_sample


def main() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)
    result = run_behavioral_evaluation(
        sample,
        config,
        keys_per_head=6,
        prompt_set="qwen35_calibration_v0",
        key_selection_method=config.compaction.key_selection,
    )
    output_path = Path("artifacts/qwen35_smoke/behavioral_eval_qwen35_calibration_v0_k6.json")
    write_behavioral_result(result, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
