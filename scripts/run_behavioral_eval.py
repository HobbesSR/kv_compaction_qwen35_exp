from __future__ import annotations

import argparse
from pathlib import Path

from kv_compaction_qwen35_clean.behavioral_eval import run_behavioral_evaluation, write_behavioral_result
from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.context_loader import load_context_sample


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys-per-head", type=int, default=6)
    parser.add_argument("--prompt-set", default="qwen35_calibration_v0")
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)
    result = run_behavioral_evaluation(
        sample,
        config,
        keys_per_head=args.keys_per_head,
        prompt_set=args.prompt_set,
        key_selection_method=config.compaction.key_selection,
        prompt_limit=args.prompt_limit,
        max_new_tokens=args.max_new_tokens,
    )
    suffix = ""
    if args.prompt_limit is not None:
        suffix += f"_p{args.prompt_limit}"
    if args.max_new_tokens != 64:
        suffix += f"_t{args.max_new_tokens}"
    output_path = Path(
        f"artifacts/qwen35_smoke/behavioral_eval_{args.prompt_set}_k{args.keys_per_head}{suffix}.json"
    )
    write_behavioral_result(result, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
