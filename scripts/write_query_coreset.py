from __future__ import annotations

from pathlib import Path

from kv_compaction_qwen35_clean.boundary_collection import (
    collect_teacher_forced_boundary_collection,
)
from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.context_loader import load_context_sample
from kv_compaction_qwen35_clean.coreset import extract_query_coreset, write_query_coreset
from kv_compaction_qwen35_clean.prototype_bank import build_state_from_observations, write_state


def main() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)
    bundle = collect_teacher_forced_boundary_collection(
        sample,
        config,
        materialize_boundary_kv=False,
    )
    state = build_state_from_observations(config, bundle.harvest.observations)
    coreset = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, config)

    state_path = Path("artifacts/qwen35_smoke/prototype_bank.json")
    coreset_path = Path("artifacts/qwen35_smoke/query_coreset.json")
    write_state(state, state_path)
    write_query_coreset(coreset, coreset_path)
    print(state_path)
    print(coreset_path)


if __name__ == "__main__":
    main()
