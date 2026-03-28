from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from kv_compaction_qwen35_clean.data_types import CompactHeadRuntime


def _hash_json_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_token_ids(token_ids: list[int]) -> str:
    encoded = ",".join(str(int(token_id)) for token_id in token_ids).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SegmentBoundaryIdentity:
    boundary_turn_index: int
    segment_start_token: int
    segment_end_token: int


@dataclass(frozen=True)
class SegmentLineageNode:
    segment_index: int
    first_turn_index: int
    last_turn_index: int
    segment_hash: str
    parent_hash: str | None
    turn_ids: list[str]
    segment_token_hash: str
    segment_token_count: int
    logical_token_count_before: int
    logical_token_count_after: int
    boundary: SegmentBoundaryIdentity


@dataclass(frozen=True)
class SegmentBundleMetadata:
    segment_hash: str
    parent_hash: str | None
    config_fingerprint: str
    model_name: str
    huggingface_id: str
    tokenizer_name: str
    tokenizer_fingerprint: str
    segment_token_hash: str
    segment_token_count: int
    logical_token_count_before: int
    logical_token_count_after: int
    physical_compacted_kv_slot_count: int
    physical_compacted_unique_token_count: int
    target_layers: list[int]
    target_heads: list[int]
    target_layer_heads: list[str]
    boundary: SegmentBoundaryIdentity
    runtime_tensors_path: str
    created_by: str = "segment_compaction_cache_v1"

    def to_serializable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["boundary"] = asdict(self.boundary)
        return payload


@dataclass(frozen=True)
class SegmentCompactionBundle:
    metadata: SegmentBundleMetadata
    compacted_layers: dict[int, dict[int, CompactHeadRuntime]]


@dataclass(frozen=True)
class CachedPrefixLookup:
    lineage: list[SegmentLineageNode]
    bundles: list[SegmentCompactionBundle]
    first_uncached_turn_index: int


@dataclass(frozen=True)
class CachedPrefixMetadataLookup:
    lineage: list[SegmentLineageNode]
    bundle_metadata: list[SegmentBundleMetadata]
    first_uncached_turn_index: int


def build_config_fingerprint(
    *,
    model_name: str,
    huggingface_id: str,
    tokenizer_name: str,
    tokenizer_fingerprint: str,
    target_layer_heads: tuple[tuple[int, int], ...],
    keys_per_head: int,
    key_selection_method: str,
    beta_solver: str,
    beta_regularization_strength: float,
    value_regularization_strength: float,
) -> str:
    return _hash_json_payload(
        {
            "model_name": model_name,
            "huggingface_id": huggingface_id,
            "tokenizer_name": tokenizer_name,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "target_layer_heads": [[int(layer), int(head)] for layer, head in target_layer_heads],
            "keys_per_head": int(keys_per_head),
            "key_selection_method": str(key_selection_method),
            "beta_solver": str(beta_solver),
            "beta_regularization_strength": float(beta_regularization_strength),
            "value_regularization_strength": float(value_regularization_strength),
        }
    )


def build_segment_hash(
    *,
    parent_hash: str | None,
    segment_token_ids: list[int],
    config_fingerprint: str,
    boundary_turn_index: int,
    segment_start_token: int,
    segment_end_token: int,
) -> str:
    return _hash_json_payload(
        {
            "parent_hash": parent_hash,
            "segment_token_hash": _hash_token_ids(segment_token_ids),
            "config_fingerprint": config_fingerprint,
            "boundary_turn_index": int(boundary_turn_index),
            "segment_start_token": int(segment_start_token),
            "segment_end_token": int(segment_end_token),
        }
    )


def build_turn_segment_lineage(
    *,
    token_ids: list[int],
    turn_spans: list[tuple[int, int, str, str]],
    config_fingerprint: str,
    min_segment_tokens: int = 0,
) -> list[SegmentLineageNode]:
    if min_segment_tokens < 0:
        raise ValueError("min_segment_tokens must be non-negative.")

    nodes: list[SegmentLineageNode] = []
    pending_turn_ids: list[str] = []
    pending_start: int | None = None
    pending_end: int | None = None
    pending_first_turn_index: int | None = None
    parent_hash: str | None = None

    for turn_index, (start, end, turn_id, _speaker) in enumerate(turn_spans):
        if pending_start is None:
            pending_start = int(start)
            pending_first_turn_index = turn_index
        pending_end = int(end)
        pending_turn_ids.append(str(turn_id))
        current_count = pending_end - pending_start

        is_last_turn = turn_index == len(turn_spans) - 1
        if not is_last_turn and current_count < min_segment_tokens:
            continue

        segment_token_ids = token_ids[pending_start:pending_end]
        segment_hash = build_segment_hash(
            parent_hash=parent_hash,
            segment_token_ids=segment_token_ids,
            config_fingerprint=config_fingerprint,
            boundary_turn_index=turn_index,
            segment_start_token=pending_start,
            segment_end_token=pending_end,
        )
        nodes.append(
            SegmentLineageNode(
                segment_index=len(nodes),
                first_turn_index=int(pending_first_turn_index),
                last_turn_index=turn_index,
                segment_hash=segment_hash,
                parent_hash=parent_hash,
                turn_ids=list(pending_turn_ids),
                segment_token_hash=_hash_token_ids(segment_token_ids),
                segment_token_count=len(segment_token_ids),
                logical_token_count_before=pending_start,
                logical_token_count_after=pending_end,
                boundary=SegmentBoundaryIdentity(
                    boundary_turn_index=turn_index,
                    segment_start_token=pending_start,
                    segment_end_token=pending_end,
                ),
            )
        )
        parent_hash = segment_hash
        pending_turn_ids = []
        pending_start = None
        pending_end = None
        pending_first_turn_index = None

    if pending_turn_ids:  # pragma: no cover
        raise AssertionError("Unflushed pending segment state remained after lineage build.")

    return nodes


def find_cached_prefix(
    *,
    token_ids: list[int],
    turn_spans: list[tuple[int, int, str, str]],
    config_fingerprint: str,
    cache_root: Path,
    min_segment_tokens: int = 0,
    device: str = "cpu",
) -> CachedPrefixLookup:
    lineage = build_turn_segment_lineage(
        token_ids=token_ids,
        turn_spans=turn_spans,
        config_fingerprint=config_fingerprint,
        min_segment_tokens=min_segment_tokens,
    )
    bundles: list[SegmentCompactionBundle] = []
    first_uncached_turn_index = 0

    for node in lineage:
        bundle_dir = cache_root / node.segment_hash
        if not bundle_dir.is_dir():
            first_uncached_turn_index = node.first_turn_index
            break
        bundle = load_segment_bundle(bundle_dir, device=device)
        if bundle.metadata.segment_hash != node.segment_hash:
            first_uncached_turn_index = node.first_turn_index
            break
        if bundle.metadata.segment_token_hash != node.segment_token_hash:
            first_uncached_turn_index = node.first_turn_index
            break
        if bundle.metadata.config_fingerprint != config_fingerprint:
            first_uncached_turn_index = node.first_turn_index
            break
        bundles.append(bundle)
        first_uncached_turn_index = node.last_turn_index + 1
    else:
        first_uncached_turn_index = len(turn_spans)

    return CachedPrefixLookup(
        lineage=lineage,
        bundles=bundles,
        first_uncached_turn_index=first_uncached_turn_index,
    )


def load_segment_bundle_metadata(bundle_dir: Path) -> SegmentBundleMetadata:
    payload = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    legacy_kv_slot_count = int(payload.get("physical_compacted_token_count", 0))
    return SegmentBundleMetadata(
        segment_hash=str(payload["segment_hash"]),
        parent_hash=str(payload["parent_hash"]) if payload.get("parent_hash") is not None else None,
        config_fingerprint=str(payload["config_fingerprint"]),
        model_name=str(payload["model_name"]),
        huggingface_id=str(payload["huggingface_id"]),
        tokenizer_name=str(payload["tokenizer_name"]),
        tokenizer_fingerprint=str(payload["tokenizer_fingerprint"]),
        segment_token_hash=str(payload["segment_token_hash"]),
        segment_token_count=int(payload["segment_token_count"]),
        logical_token_count_before=int(payload["logical_token_count_before"]),
        logical_token_count_after=int(payload["logical_token_count_after"]),
        physical_compacted_kv_slot_count=int(payload.get("physical_compacted_kv_slot_count", legacy_kv_slot_count)),
        physical_compacted_unique_token_count=int(
            payload.get("physical_compacted_unique_token_count", legacy_kv_slot_count)
        ),
        target_layers=[int(layer) for layer in payload["target_layers"]],
        target_heads=[int(head) for head in payload["target_heads"]],
        target_layer_heads=[str(label) for label in payload["target_layer_heads"]],
        boundary=SegmentBoundaryIdentity(**payload["boundary"]),
        runtime_tensors_path=str(payload["runtime_tensors_path"]),
        created_by=str(payload.get("created_by", "segment_compaction_cache_v1")),
    )


def find_cached_prefix_metadata(
    *,
    token_ids: list[int],
    turn_spans: list[tuple[int, int, str, str]],
    config_fingerprint: str,
    cache_root: Path,
    min_segment_tokens: int = 0,
) -> CachedPrefixMetadataLookup:
    lineage = build_turn_segment_lineage(
        token_ids=token_ids,
        turn_spans=turn_spans,
        config_fingerprint=config_fingerprint,
        min_segment_tokens=min_segment_tokens,
    )
    bundle_metadata: list[SegmentBundleMetadata] = []
    first_uncached_turn_index = 0

    for node in lineage:
        bundle_dir = cache_root / node.segment_hash
        if not bundle_dir.is_dir():
            first_uncached_turn_index = node.first_turn_index
            break
        metadata = load_segment_bundle_metadata(bundle_dir)
        if metadata.segment_hash != node.segment_hash:
            first_uncached_turn_index = node.first_turn_index
            break
        if metadata.segment_token_hash != node.segment_token_hash:
            first_uncached_turn_index = node.first_turn_index
            break
        if metadata.config_fingerprint != config_fingerprint:
            first_uncached_turn_index = node.first_turn_index
            break
        bundle_metadata.append(metadata)
        first_uncached_turn_index = node.last_turn_index + 1
    else:
        first_uncached_turn_index = len(turn_spans)

    return CachedPrefixMetadataLookup(
        lineage=lineage,
        bundle_metadata=bundle_metadata,
        first_uncached_turn_index=first_uncached_turn_index,
    )


def _runtime_tensor_name(*, layer: int, head: int, kind: str) -> str:
    return f"layer_{int(layer):02d}.head_{int(head):02d}.{kind}"


def _flatten_compacted_layers(compacted_layers: dict[int, dict[int, CompactHeadRuntime]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    tensors: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for layer in sorted(compacted_layers):
        for head in sorted(compacted_layers[layer]):
            runtime = compacted_layers[layer][head]
            key_name = _runtime_tensor_name(layer=layer, head=head, kind="compact_keys")
            value_name = _runtime_tensor_name(layer=layer, head=head, kind="compact_values")
            beta_name = _runtime_tensor_name(layer=layer, head=head, kind="beta")
            tensors[key_name] = runtime.compact_keys.detach().cpu()
            tensors[value_name] = runtime.compact_values.detach().cpu()
            tensors[beta_name] = runtime.beta.detach().cpu()
            rows.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "selected_indices": [int(index) for index in runtime.selected_indices],
                    "compact_keys_tensor": key_name,
                    "compact_values_tensor": value_name,
                    "beta_tensor": beta_name,
                }
            )
    return tensors, rows


def _inflate_compacted_layers(
    tensor_path: Path,
    runtime_rows: list[dict[str, object]],
    *,
    device: str,
) -> dict[int, dict[int, CompactHeadRuntime]]:
    tensors = load_file(str(tensor_path), device=device)
    compacted_layers: dict[int, dict[int, CompactHeadRuntime]] = {}
    for row in runtime_rows:
        layer = int(row["layer"])
        head = int(row["head"])
        compacted_layers.setdefault(layer, {})[head] = CompactHeadRuntime(
            layer=layer,
            head=head,
            selected_indices=[int(index) for index in row["selected_indices"]],
            compact_keys=tensors[str(row["compact_keys_tensor"])],
            compact_values=tensors[str(row["compact_values_tensor"])],
            beta=tensors[str(row["beta_tensor"])],
        )
    return compacted_layers


def build_segment_bundle(
    *,
    parent_hash: str | None,
    segment_token_ids: list[int],
    boundary_turn_index: int,
    segment_start_token: int,
    segment_end_token: int,
    logical_token_count_before: int,
    logical_token_count_after: int,
    model_name: str,
    huggingface_id: str,
    tokenizer_name: str,
    tokenizer_fingerprint: str,
    config_fingerprint: str,
    target_layer_heads: tuple[tuple[int, int], ...],
    compacted_layers: dict[int, dict[int, CompactHeadRuntime]],
    runtime_tensors_path: str = "runtime.safetensors",
) -> SegmentCompactionBundle:
    segment_hash = build_segment_hash(
        parent_hash=parent_hash,
        segment_token_ids=segment_token_ids,
        config_fingerprint=config_fingerprint,
        boundary_turn_index=boundary_turn_index,
        segment_start_token=segment_start_token,
        segment_end_token=segment_end_token,
    )
    target_layers = sorted({int(layer) for layer, _ in target_layer_heads})
    target_heads = sorted({int(head) for _, head in target_layer_heads})
    physical_compacted_kv_slot_count = sum(
        len(runtime.selected_indices)
        for layer_rows in compacted_layers.values()
        for runtime in layer_rows.values()
    )
    physical_compacted_unique_token_count = len(
        {
            int(index)
            for layer_rows in compacted_layers.values()
            for runtime in layer_rows.values()
            for index in runtime.selected_indices
        }
    )
    metadata = SegmentBundleMetadata(
        segment_hash=segment_hash,
        parent_hash=parent_hash,
        config_fingerprint=config_fingerprint,
        model_name=model_name,
        huggingface_id=huggingface_id,
        tokenizer_name=tokenizer_name,
        tokenizer_fingerprint=tokenizer_fingerprint,
        segment_token_hash=_hash_token_ids(segment_token_ids),
        segment_token_count=len(segment_token_ids),
        logical_token_count_before=int(logical_token_count_before),
        logical_token_count_after=int(logical_token_count_after),
        physical_compacted_kv_slot_count=physical_compacted_kv_slot_count,
        physical_compacted_unique_token_count=physical_compacted_unique_token_count,
        target_layers=target_layers,
        target_heads=target_heads,
        target_layer_heads=[f"{layer}:{head}" for layer, head in target_layer_heads],
        boundary=SegmentBoundaryIdentity(
            boundary_turn_index=int(boundary_turn_index),
            segment_start_token=int(segment_start_token),
            segment_end_token=int(segment_end_token),
        ),
        runtime_tensors_path=runtime_tensors_path,
    )
    return SegmentCompactionBundle(metadata=metadata, compacted_layers=compacted_layers)


def write_segment_bundle(bundle: SegmentCompactionBundle, root_dir: Path) -> Path:
    bundle_dir = root_dir / bundle.metadata.segment_hash
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = bundle_dir / bundle.metadata.runtime_tensors_path
    tensors, runtime_rows = _flatten_compacted_layers(bundle.compacted_layers)
    save_file(tensors, str(tensor_path))
    payload = bundle.metadata.to_serializable()
    payload["runtime_rows"] = runtime_rows
    (bundle_dir / "bundle.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return bundle_dir


def load_segment_bundle(bundle_dir: Path, *, device: str = "cpu") -> SegmentCompactionBundle:
    payload = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    metadata = load_segment_bundle_metadata(bundle_dir)
    compacted_layers = _inflate_compacted_layers(
        bundle_dir / metadata.runtime_tensors_path,
        payload["runtime_rows"],
        device=device,
    )
    return SegmentCompactionBundle(metadata=metadata, compacted_layers=compacted_layers)
