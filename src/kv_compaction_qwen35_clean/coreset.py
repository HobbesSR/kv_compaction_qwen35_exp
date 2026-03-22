from __future__ import annotations

import json
from pathlib import Path

from kv_compaction_qwen35_clean.data_types import QueryCoreset, QueryCoresetEntry, SmokeTestConfig
from kv_compaction_qwen35_clean.prototype_bank import PrototypeBankState


def extract_query_coreset(
    sample_id: str,
    boundary_id: str,
    state: PrototypeBankState,
    config: SmokeTestConfig,
    max_entries: int | None = None,
) -> QueryCoreset:
    limit = max_entries if max_entries is not None else min(len(state.entries), config.sketch.max_prototypes)
    ranked_entries = sorted(
        state.entries,
        key=lambda entry: (
            entry.weight * entry.avg_prefix_mass_share,
            entry.weight,
            entry.avg_raw_prefix_mass,
        ),
        reverse=True,
    )
    selected = ranked_entries[:limit]
    coreset_entries = [
        QueryCoresetEntry(
            coreset_id=f"q{index}",
            prototype_id=entry.prototype_id,
            layer=entry.layer,
            head=entry.head,
            weight=round(entry.weight, 6),
            avg_prefix_mass_share=round(entry.avg_prefix_mass_share, 6),
            avg_raw_prefix_mass=round(entry.avg_raw_prefix_mass, 6),
            query_projection=entry.center_query_projection,
            output_projection_hint=entry.center_output_projection,
            last_token_index=entry.last_token_index,
        )
        for index, entry in enumerate(selected)
    ]
    return QueryCoreset(
        sample_id=sample_id,
        boundary_id=boundary_id,
        source="prototype_bank",
        max_entries=limit,
        selected_entries=coreset_entries,
        total_weight=round(sum(entry.weight for entry in selected), 6),
    )


def write_query_coreset(coreset: QueryCoreset, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coreset.to_serializable(), indent=2) + "\n", encoding="utf-8")
    return output_path
