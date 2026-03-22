"""
Diagnose sketch coverage: trace source_turn_id through boundary collection,
prototype bank, coreset, and key selection to find where early-turn content
drops out.

Run in the dedicated Qwen3.5 env:

    PYTHONPATH=src python scripts/diagnose_sketch_coverage.py

Writes a boundary bundle to artifacts/qwen35_smoke/diag_boundary_collection.json
(without KV tensors, so the file is small), then prints the turn distribution
at every stage.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from kv_compaction_qwen35_clean.boundary_collection import (
    collect_teacher_forced_boundary_collection,
    write_boundary_collection,
)
from kv_compaction_qwen35_clean.config import load_config
from kv_compaction_qwen35_clean.context_loader import load_context_sample
from kv_compaction_qwen35_clean.coreset import extract_query_coreset
from kv_compaction_qwen35_clean.key_selection import match_coreset_to_query_samples, select_keys
from kv_compaction_qwen35_clean.prototype_bank import build_state_from_observations
from kv_compaction_qwen35_clean.query_controls import extract_teacher_forced_subsample_control


BUNDLE_PATH = Path("artifacts/qwen35_smoke/diag_boundary_collection.json")
KEYS_PER_HEAD = 6


def _turn_dist(items, turn_id_fn) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        counts[str(turn_id_fn(item))] += 1
    return dict(sorted(counts.items()))


def _print_dist(label: str, dist: dict[str, int], total: int) -> None:
    print(f"\n{label} (n={total}):")
    for turn_id, count in dist.items():
        bar = "#" * count
        pct = 100.0 * count / max(1, total)
        print(f"  {turn_id:12s}  {count:4d}  {pct:5.1f}%  {bar}")


def main() -> None:
    config = load_config("configs/qwen35_smoke/qwen3_5_9b.yaml")
    sample = load_context_sample(config)
    collection_config = replace(config, model=replace(config.model, attn_implementation="eager"))

    # --- stage 1: boundary collection ---
    print("collecting boundary bundle (this loads the model)...")
    bundle = collect_teacher_forced_boundary_collection(
        sample,
        collection_config,
        materialize_boundary_kv=False,
    )
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_boundary_collection(replace(bundle, runtime_cache=None), BUNDLE_PATH)
    print(f"bundle written to {BUNDLE_PATH}")

    observations = bundle.harvest.observations
    query_samples = bundle.query_bank.samples
    print(f"\nprefix_token_count : {sample.boundary.prefix_token_count}")
    print(f"capture positions  : {len(bundle.capture_token_indices or [])}")
    print(f"observations       : {len(observations)}")
    print(f"query_bank samples : {len(query_samples)}")

    # --- stage 2: observation turn distribution ---
    obs_dist = _turn_dist(observations, lambda o: o.source_turn_id)
    _print_dist("observations by source_turn_id", obs_dist, len(observations))

    # --- stage 3: prototype bank ---
    state = build_state_from_observations(collection_config, observations)
    entries = state.entries
    entry_dist = _turn_dist(entries, lambda e: e.source_turn_id)
    _print_dist("prototype bank entries by source_turn_id", entry_dist, len(entries))
    print(f"\nbank capacity: {config.sketch.max_prototypes}  |  survived: {len(entries)}")

    # layer/head distribution inside bank
    layer_head_dist: dict[str, int] = defaultdict(int)
    for e in entries:
        layer_head_dist[f"L{e.layer}H{e.head}"] += 1
    print("\nbank entries by (layer, head):", dict(sorted(layer_head_dist.items())))

    # --- stage 4: coreset ---
    sketch_source = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, collection_config)
    coreset_entries = sketch_source.selected_entries
    print(f"\ncoreset selected: {len(coreset_entries)} / {len(entries)} bank entries")
    for ce in coreset_entries:
        print(
            f"  {ce.coreset_id}  L{ce.layer}H{ce.head}  "
            f"last_tok={ce.last_token_index}  w={ce.weight:.4f}  mass={ce.avg_prefix_mass_share:.4f}"
        )

    # --- stage 5: match coreset → query_bank and select keys ---
    matched = match_coreset_to_query_samples(sketch_source, query_samples)
    sketch_match_dist = _turn_dist(matched, lambda pair: pair[1].source_turn_id)
    _print_dist("sketch matched query_bank samples by source_turn_id", sketch_match_dist, len(matched))

    # key selection requires boundary_keys — skip if absent
    if any(bundle.boundary_keys.values() if bundle.boundary_keys else []):
        selection = select_keys(
            sample.sample_id,
            sample.boundary.boundary_id,
            "prototype_bank",
            matched,
            bundle.boundary_keys,
            KEYS_PER_HEAD,
        )
        sketch_selected: list[int] = []
        for group in selection.groups:
            sketch_selected.extend(group.selected_indices)
        print(f"\nsketch selected positions ({len(sketch_selected)}): {sorted(sketch_selected)[:40]}")
    else:
        print("\n(boundary_keys not materialised — skipping key selection stage)")

    # --- stage 6: control for comparison ---
    control_source = extract_teacher_forced_subsample_control(
        bundle.query_bank,
        max_entries=len(coreset_entries),
    )
    control_dist = _turn_dist(
        control_source.selected_entries,
        lambda ce: next(
            (s.source_turn_id for s in query_samples if s.layer == ce.layer and s.head == ce.head and s.token_index == ce.last_token_index),
            "unknown",
        ),
    )
    _print_dist("control selected entries by source_turn_id", control_dist, len(control_source.selected_entries))


if __name__ == "__main__":
    main()
