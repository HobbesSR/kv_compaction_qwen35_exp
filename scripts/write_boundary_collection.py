from __future__ import annotations

from pathlib import Path

from kv_compaction_qwen35_clean.boundary_collection import (
    collect_teacher_forced_boundary_collection,
    write_boundary_collection,
)
from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.context_loader import load_context_sample


def main() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)
    bundle = collect_teacher_forced_boundary_collection(
        sample,
        config,
        materialize_boundary_kv=False,
    )
    output_path = Path("artifacts/qwen35_smoke/boundary_collection.json")
    write_boundary_collection(bundle, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
