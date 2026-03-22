from __future__ import annotations

import json
from pathlib import Path

from kv_compaction_qwen35_clean.data_types import QueryCoreset, QueryCoresetEntry, QuerySampleBank


def extract_teacher_forced_subsample_control(
    query_bank: QuerySampleBank,
    max_entries: int,
) -> QueryCoreset:
    ranked = sorted(
        query_bank.samples,
        key=lambda sample: (sample.raw_prefix_mass, sample.prefix_mass_share, sample.token_index),
        reverse=True,
    )
    selected = ranked[:max_entries]
    entries = [
        QueryCoresetEntry(
            coreset_id=f"ctrl{index}",
            prototype_id=sample.query_id,
            layer=sample.layer,
            head=sample.head,
            weight=round(sample.prefix_mass_share, 6),
            avg_prefix_mass_share=round(sample.prefix_mass_share, 6),
            avg_raw_prefix_mass=round(sample.raw_prefix_mass, 6),
            query_projection=sample.query_projection,
            output_projection_hint=[],
            last_token_index=sample.token_index,
        )
        for index, sample in enumerate(selected)
    ]
    return QueryCoreset(
        sample_id=query_bank.sample_id,
        boundary_id=query_bank.boundary_id,
        source="teacher_forced_subsample",
        max_entries=max_entries,
        selected_entries=entries,
        total_weight=round(sum(sample.prefix_mass_share for sample in selected), 6),
    )


def write_query_source(query_source: QueryCoreset, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(query_source.to_serializable(), indent=2) + "\n", encoding="utf-8")
    return output_path
