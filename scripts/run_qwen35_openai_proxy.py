#!/usr/bin/env python
from __future__ import annotations

import argparse

from kv_compaction_qwen35_clean.qwen35_openai_proxy import run_proxy


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
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
