from __future__ import annotations

from pathlib import Path

import yaml

from kv_compaction_qwen35_clean.data_types import (
    BranchSwitchConfig,
    ChunkingConfig,
    CompactionConfig,
    DataConfig,
    ExperimentConfig,
    FeatureSchemaConfig,
    ModelConfig,
    ReferenceQueryConfig,
    SketchConfig,
    SmokeTestConfig,
)


def load_config(path: str | Path) -> SmokeTestConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return SmokeTestConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(
            dataset=raw["data"]["dataset"],
            context_tokens=raw["data"]["context_tokens"],
            branch_switch_probe=raw["data"]["branch_switch_probe"],
            chunking=ChunkingConfig(**raw["data"]["chunking"]),
            branch_switch=BranchSwitchConfig(**raw["data"]["branch_switch"]),
        ),
        compaction=CompactionConfig(**raw["compaction"]),
        feature_schema=FeatureSchemaConfig(**raw["feature_schema"]),
        sketch=SketchConfig(**raw["sketch"]),
        reference_queries=ReferenceQueryConfig(**raw["reference_queries"]),
        baselines=list(raw["baselines"]),
        metrics=list(raw["metrics"]),
    )
