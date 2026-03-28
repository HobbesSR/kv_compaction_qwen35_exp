import torch

from kv_compaction_qwen35_clean.data_types import CompactHeadRuntime
from kv_compaction_qwen35_clean.segment_compaction_cache import (
    build_config_fingerprint,
    build_segment_bundle,
    build_segment_hash,
    build_turn_segment_lineage,
    find_cached_prefix,
    find_cached_prefix_metadata,
    load_segment_bundle,
    load_segment_bundle_metadata,
    write_segment_bundle,
)


def _sample_compacted_layers() -> dict[int, dict[int, CompactHeadRuntime]]:
    return {
        4: {
            0: CompactHeadRuntime(
                layer=4,
                head=0,
                selected_indices=[3, 7, 11],
                compact_keys=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                compact_values=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                beta=torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
            ),
        },
        12: {
            3: CompactHeadRuntime(
                layer=12,
                head=3,
                selected_indices=[13, 21],
                compact_keys=torch.tensor([[9.0, 10.0]]),
                compact_values=torch.tensor([[11.0, 12.0]]),
                beta=torch.tensor([[0.1], [0.9]]),
            ),
        },
    }


def test_segment_hash_changes_when_parent_or_boundary_changes() -> None:
    token_ids = [101, 102, 103]
    config_fingerprint = "cfg"
    base = build_segment_hash(
        parent_hash=None,
        segment_token_ids=token_ids,
        config_fingerprint=config_fingerprint,
        boundary_turn_index=4,
        segment_start_token=128,
        segment_end_token=256,
    )
    parent_changed = build_segment_hash(
        parent_hash="parent",
        segment_token_ids=token_ids,
        config_fingerprint=config_fingerprint,
        boundary_turn_index=4,
        segment_start_token=128,
        segment_end_token=256,
    )
    boundary_changed = build_segment_hash(
        parent_hash=None,
        segment_token_ids=token_ids,
        config_fingerprint=config_fingerprint,
        boundary_turn_index=5,
        segment_start_token=128,
        segment_end_token=256,
    )
    assert parent_changed != base
    assert boundary_changed != base


def test_config_fingerprint_changes_with_target_layer_heads() -> None:
    base = build_config_fingerprint(
        model_name="Qwen3.5-9B",
        huggingface_id="Qwen/Qwen3.5-9B",
        tokenizer_name="Qwen3.5Tokenizer",
        tokenizer_fingerprint="tok-v1",
        target_layer_heads=((3, 0), (7, 7)),
        keys_per_head=8,
        key_selection_method="fast",
        beta_solver="ridge",
        beta_regularization_strength=0.01,
        value_regularization_strength=0.01,
    )
    changed = build_config_fingerprint(
        model_name="Qwen3.5-9B",
        huggingface_id="Qwen/Qwen3.5-9B",
        tokenizer_name="Qwen3.5Tokenizer",
        tokenizer_fingerprint="tok-v1",
        target_layer_heads=((3, 0), (7, 3)),
        keys_per_head=8,
        key_selection_method="fast",
        beta_solver="ridge",
        beta_regularization_strength=0.01,
        value_regularization_strength=0.01,
    )
    assert changed != base


def test_segment_bundle_roundtrip_preserves_metadata_and_tensors(tmp_path) -> None:
    compacted_layers = _sample_compacted_layers()
    target_layer_heads = ((4, 0), (12, 3))
    bundle = build_segment_bundle(
        parent_hash="root",
        segment_token_ids=[11, 12, 13, 14],
        boundary_turn_index=6,
        segment_start_token=2048,
        segment_end_token=3072,
        logical_token_count_before=2048,
        logical_token_count_after=3072,
        model_name="Qwen3.5-9B",
        huggingface_id="Qwen/Qwen3.5-9B",
        tokenizer_name="Qwen3.5Tokenizer",
        tokenizer_fingerprint="tok-v1",
        config_fingerprint="cfg-v1",
        target_layer_heads=target_layer_heads,
        compacted_layers=compacted_layers,
    )

    bundle_dir = write_segment_bundle(bundle, tmp_path)
    loaded = load_segment_bundle(bundle_dir, device="cpu")

    assert loaded.metadata.segment_hash == bundle.metadata.segment_hash
    assert loaded.metadata.parent_hash == "root"
    assert loaded.metadata.segment_token_count == 4
    assert loaded.metadata.logical_token_count_before == 2048
    assert loaded.metadata.logical_token_count_after == 3072
    assert loaded.metadata.physical_compacted_kv_slot_count == 5
    assert loaded.metadata.physical_compacted_unique_token_count == 5
    assert loaded.metadata.target_layer_heads == ["4:0", "12:3"]
    assert loaded.metadata.boundary.boundary_turn_index == 6
    assert loaded.metadata.boundary.segment_start_token == 2048
    assert loaded.metadata.boundary.segment_end_token == 3072

    loaded_runtime = loaded.compacted_layers[4][0]
    assert loaded_runtime.selected_indices == [3, 7, 11]
    assert torch.equal(loaded_runtime.compact_keys, compacted_layers[4][0].compact_keys)
    assert torch.equal(loaded_runtime.compact_values, compacted_layers[4][0].compact_values)
    assert torch.equal(loaded_runtime.beta, compacted_layers[4][0].beta)

    second_runtime = loaded.compacted_layers[12][3]
    assert second_runtime.selected_indices == [13, 21]
    assert torch.equal(second_runtime.compact_keys, compacted_layers[12][3].compact_keys)
    assert torch.equal(second_runtime.compact_values, compacted_layers[12][3].compact_values)
    assert torch.equal(second_runtime.beta, compacted_layers[12][3].beta)


def test_load_segment_bundle_metadata_skips_runtime_tensor_inflation(tmp_path) -> None:
    bundle = build_segment_bundle(
        parent_hash="root",
        segment_token_ids=[11, 12, 13, 14],
        boundary_turn_index=6,
        segment_start_token=2048,
        segment_end_token=3072,
        logical_token_count_before=2048,
        logical_token_count_after=3072,
        model_name="Qwen3.5-9B",
        huggingface_id="Qwen/Qwen3.5-9B",
        tokenizer_name="Qwen3.5Tokenizer",
        tokenizer_fingerprint="tok-v1",
        config_fingerprint="cfg-v1",
        target_layer_heads=((4, 0), (12, 3)),
        compacted_layers=_sample_compacted_layers(),
    )

    bundle_dir = write_segment_bundle(bundle, tmp_path)
    metadata = load_segment_bundle_metadata(bundle_dir)

    assert metadata.segment_hash == bundle.metadata.segment_hash
    assert metadata.runtime_tensors_path == "runtime.safetensors"
    assert metadata.physical_compacted_kv_slot_count == 5
    assert metadata.physical_compacted_unique_token_count == 5
    assert metadata.target_layer_heads == ["4:0", "12:3"]


def test_write_segment_bundle_uses_segment_hash_directory(tmp_path) -> None:
    bundle = build_segment_bundle(
        parent_hash=None,
        segment_token_ids=[1, 2, 3],
        boundary_turn_index=0,
        segment_start_token=0,
        segment_end_token=3,
        logical_token_count_before=0,
        logical_token_count_after=3,
        model_name="Qwen3.5-9B",
        huggingface_id="Qwen/Qwen3.5-9B",
        tokenizer_name="Qwen3.5Tokenizer",
        tokenizer_fingerprint="tok-v1",
        config_fingerprint="cfg-v1",
        target_layer_heads=((4, 0),),
        compacted_layers=_sample_compacted_layers(),
    )

    bundle_dir = write_segment_bundle(bundle, tmp_path)

    assert bundle_dir.name == bundle.metadata.segment_hash
    assert (bundle_dir / "bundle.json").exists()
    assert (bundle_dir / bundle.metadata.runtime_tensors_path).exists()


def test_turn_segment_lineage_builds_parent_chained_segments() -> None:
    nodes = build_turn_segment_lineage(
        token_ids=[10, 11, 12, 13, 14, 15],
        turn_spans=[
            (0, 2, "turn_0", "user"),
            (2, 4, "turn_1", "assistant"),
            (4, 6, "turn_2", "user"),
        ],
        config_fingerprint="cfg-v1",
    )

    assert [node.turn_ids for node in nodes] == [["turn_0"], ["turn_1"], ["turn_2"]]
    assert nodes[0].first_turn_index == 0
    assert nodes[0].last_turn_index == 0
    assert nodes[1].first_turn_index == 1
    assert nodes[1].last_turn_index == 1
    assert nodes[0].parent_hash is None
    assert nodes[1].parent_hash == nodes[0].segment_hash
    assert nodes[2].parent_hash == nodes[1].segment_hash
    assert nodes[1].boundary.segment_start_token == 2
    assert nodes[1].boundary.segment_end_token == 4


def test_turn_segment_lineage_merges_short_turns_until_minimum_budget() -> None:
    nodes = build_turn_segment_lineage(
        token_ids=list(range(10)),
        turn_spans=[
            (0, 2, "turn_0", "user"),
            (2, 4, "turn_1", "assistant"),
            (4, 9, "turn_2", "tool"),
            (9, 10, "turn_3", "assistant"),
        ],
        config_fingerprint="cfg-v1",
        min_segment_tokens=5,
    )

    assert [node.turn_ids for node in nodes] == [["turn_0", "turn_1", "turn_2"], ["turn_3"]]
    assert nodes[0].first_turn_index == 0
    assert nodes[0].last_turn_index == 2
    assert nodes[1].first_turn_index == 3
    assert nodes[1].last_turn_index == 3
    assert nodes[0].segment_token_count == 9
    assert nodes[0].logical_token_count_before == 0
    assert nodes[0].logical_token_count_after == 9
    assert nodes[1].segment_token_count == 1
    assert nodes[1].parent_hash == nodes[0].segment_hash


def test_find_cached_prefix_returns_longest_cached_prefix_and_first_uncached_turn(tmp_path) -> None:
    token_ids = list(range(12))
    turn_spans = [
        (0, 2, "turn_0", "user"),
        (2, 4, "turn_1", "assistant"),
        (4, 8, "turn_2", "tool"),
        (8, 12, "turn_3", "assistant"),
    ]
    lineage = build_turn_segment_lineage(
        token_ids=token_ids,
        turn_spans=turn_spans,
        config_fingerprint="cfg-v1",
        min_segment_tokens=5,
    )
    first_node = lineage[0]
    bundle = build_segment_bundle(
        parent_hash=first_node.parent_hash,
        segment_token_ids=token_ids[first_node.boundary.segment_start_token:first_node.boundary.segment_end_token],
        boundary_turn_index=first_node.boundary.boundary_turn_index,
        segment_start_token=first_node.boundary.segment_start_token,
        segment_end_token=first_node.boundary.segment_end_token,
        logical_token_count_before=first_node.logical_token_count_before,
        logical_token_count_after=first_node.logical_token_count_after,
        model_name="Qwen3.5-9B",
        huggingface_id="Qwen/Qwen3.5-9B",
        tokenizer_name="Qwen3.5Tokenizer",
        tokenizer_fingerprint="tok-v1",
        config_fingerprint="cfg-v1",
        target_layer_heads=((4, 0),),
        compacted_layers=_sample_compacted_layers(),
    )
    write_segment_bundle(bundle, tmp_path)

    lookup = find_cached_prefix(
        token_ids=token_ids,
        turn_spans=turn_spans,
        config_fingerprint="cfg-v1",
        cache_root=tmp_path,
        min_segment_tokens=5,
    )

    assert len(lookup.lineage) == 2
    assert len(lookup.bundles) == 1
    assert lookup.bundles[0].metadata.segment_hash == first_node.segment_hash
    assert lookup.first_uncached_turn_index == 3


def test_find_cached_prefix_metadata_returns_longest_cached_prefix_without_runtime_tensors(tmp_path) -> None:
    token_ids = list(range(12))
    turn_spans = [
        (0, 2, "turn_0", "user"),
        (2, 4, "turn_1", "assistant"),
        (4, 8, "turn_2", "tool"),
        (8, 12, "turn_3", "assistant"),
    ]
    lineage = build_turn_segment_lineage(
        token_ids=token_ids,
        turn_spans=turn_spans,
        config_fingerprint="cfg-v1",
        min_segment_tokens=5,
    )
    first_node = lineage[0]
    bundle = build_segment_bundle(
        parent_hash=first_node.parent_hash,
        segment_token_ids=token_ids[first_node.boundary.segment_start_token:first_node.boundary.segment_end_token],
        boundary_turn_index=first_node.boundary.boundary_turn_index,
        segment_start_token=first_node.boundary.segment_start_token,
        segment_end_token=first_node.boundary.segment_end_token,
        logical_token_count_before=first_node.logical_token_count_before,
        logical_token_count_after=first_node.logical_token_count_after,
        model_name="Qwen3.5-9B",
        huggingface_id="Qwen/Qwen3.5-9B",
        tokenizer_name="Qwen3.5Tokenizer",
        tokenizer_fingerprint="tok-v1",
        config_fingerprint="cfg-v1",
        target_layer_heads=((4, 0),),
        compacted_layers=_sample_compacted_layers(),
    )
    write_segment_bundle(bundle, tmp_path)

    lookup = find_cached_prefix_metadata(
        token_ids=token_ids,
        turn_spans=turn_spans,
        config_fingerprint="cfg-v1",
        cache_root=tmp_path,
        min_segment_tokens=5,
    )

    assert len(lookup.lineage) == 2
    assert len(lookup.bundle_metadata) == 1
    assert lookup.bundle_metadata[0].segment_hash == first_node.segment_hash
    assert lookup.first_uncached_turn_index == 3


def test_find_cached_prefix_returns_zero_when_first_segment_is_uncached(tmp_path) -> None:
    lookup = find_cached_prefix(
        token_ids=list(range(6)),
        turn_spans=[
            (0, 2, "turn_0", "user"),
            (2, 4, "turn_1", "assistant"),
            (4, 6, "turn_2", "user"),
        ],
        config_fingerprint="cfg-v1",
        cache_root=tmp_path,
        min_segment_tokens=5,
    )

    assert len(lookup.bundles) == 0
    assert lookup.first_uncached_turn_index == 0


def test_find_cached_prefix_treats_config_mismatch_as_uncached(tmp_path) -> None:
    token_ids = list(range(6))
    turn_spans = [
        (0, 2, "turn_0", "user"),
        (2, 4, "turn_1", "assistant"),
        (4, 6, "turn_2", "user"),
    ]
    lineage = build_turn_segment_lineage(
        token_ids=token_ids,
        turn_spans=turn_spans,
        config_fingerprint="cfg-v1",
    )
    node = lineage[0]
    bundle = build_segment_bundle(
        parent_hash=node.parent_hash,
        segment_token_ids=token_ids[node.boundary.segment_start_token:node.boundary.segment_end_token],
        boundary_turn_index=node.boundary.boundary_turn_index,
        segment_start_token=node.boundary.segment_start_token,
        segment_end_token=node.boundary.segment_end_token,
        logical_token_count_before=node.logical_token_count_before,
        logical_token_count_after=node.logical_token_count_after,
        model_name="Qwen3.5-9B",
        huggingface_id="Qwen/Qwen3.5-9B",
        tokenizer_name="Qwen3.5Tokenizer",
        tokenizer_fingerprint="tok-v1",
        config_fingerprint="cfg-v1",
        target_layer_heads=((4, 0),),
        compacted_layers=_sample_compacted_layers(),
    )
    write_segment_bundle(bundle, tmp_path)

    lookup = find_cached_prefix(
        token_ids=token_ids,
        turn_spans=turn_spans,
        config_fingerprint="cfg-v2",
        cache_root=tmp_path,
    )

    assert len(lookup.bundles) == 0
    assert lookup.first_uncached_turn_index == 0
