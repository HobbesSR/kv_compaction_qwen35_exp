from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from importlib import import_module
import json
from pathlib import Path

from kv_compaction_qwen35_clean.data_types import (
    BoundaryCollection,
    FeatureHarvest,
    FeatureObservation,
    LoadedContextSample,
    QuerySample,
    QuerySampleBank,
    SmokeTestConfig,
)
from kv_compaction_qwen35_clean.model_runtime import (
    default_probe_heads_for_model,
    default_probe_layers_for_model,
    load_qwen35_bundle,
    materialize_long_context_ids,
    unload_qwen35_bundle,
)


LONG_CONTEXT_TOKEN_STRIDE = 256
CAPTURE_ATTENTION_CHUNK_SIZE = 32


@dataclass
class AttentionTraceChunkBuffer:
    capacity: int
    query_length: int
    tracked_absolute_positions: set[int] | None = None
    count: int = 0
    layer_indices: object | None = None
    head_indices: object | None = None
    prefix_mass_shares: object | None = None
    raw_query_vectors: object | None = None
    raw_outputs: object | None = None
    query_segments: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    def add_query_position(
        self,
        *,
        query_position: int,
        absolute_query_position: int | None = None,
        layer_indices,
        head_indices,
        prefix_mass_shares,
        raw_query_vectors,
        raw_outputs,
    ) -> None:
        absolute_query_position = int(query_position if absolute_query_position is None else absolute_query_position)
        if (
            self.tracked_absolute_positions is not None
            and absolute_query_position not in self.tracked_absolute_positions
        ):
            return
        row_count = int(layer_indices.shape[0])
        start_index = self.count
        end_index = start_index + row_count
        if end_index > self.capacity:
            raise ValueError(
                f"AttentionTraceChunkBuffer overflow: {end_index} rows exceeds capacity {self.capacity}."
            )
        if row_count > 0 and self.layer_indices is None:
            self.layer_indices = layer_indices.detach().new_empty((self.capacity,))
            self.head_indices = head_indices.detach().new_empty((self.capacity,))
            self.prefix_mass_shares = prefix_mass_shares.detach().new_empty((self.capacity,))
            self.raw_query_vectors = raw_query_vectors.detach().new_empty((self.capacity, raw_query_vectors.shape[-1]))
            self.raw_outputs = raw_outputs.detach().new_empty((self.capacity, raw_outputs.shape[-1]))
        if row_count > 0:
            self.layer_indices[start_index:end_index].copy_(layer_indices.detach())
            self.head_indices[start_index:end_index].copy_(head_indices.detach())
            self.prefix_mass_shares[start_index:end_index].copy_(prefix_mass_shares.detach())
            self.raw_query_vectors[start_index:end_index].copy_(raw_query_vectors.detach())
            self.raw_outputs[start_index:end_index].copy_(raw_outputs.detach())
        self.query_segments.setdefault(absolute_query_position, []).append((start_index, end_index))
        self.count = end_index

    def snapshot_for_query_position(self, query_position: int):
        if self.count <= 0 or self.layer_indices is None:
            return None
        query_position = int(query_position)
        segments = [
            (start_index, end_index)
            for start_index, end_index in self.query_segments.get(query_position, [])
            if end_index > start_index
        ]
        if not segments:
            return None
        if len(segments) == 1:
            start_index, end_index = segments[0]
            return {
                "layer_indices": self.layer_indices[start_index:end_index],
                "head_indices": self.head_indices[start_index:end_index],
                "prefix_mass_shares": self.prefix_mass_shares[start_index:end_index],
                "raw_query_vectors": self.raw_query_vectors[start_index:end_index],
                "raw_outputs": self.raw_outputs[start_index:end_index],
            }

        import torch

        return {
            "layer_indices": torch.cat(
                [self.layer_indices[start_index:end_index] for start_index, end_index in segments],
                dim=0,
            ),
            "head_indices": torch.cat(
                [self.head_indices[start_index:end_index] for start_index, end_index in segments],
                dim=0,
            ),
            "prefix_mass_shares": torch.cat(
                [self.prefix_mass_shares[start_index:end_index] for start_index, end_index in segments],
                dim=0,
            ),
            "raw_query_vectors": torch.cat(
                [self.raw_query_vectors[start_index:end_index] for start_index, end_index in segments],
                dim=0,
            ),
            "raw_outputs": torch.cat(
                [self.raw_outputs[start_index:end_index] for start_index, end_index in segments],
                dim=0,
            ),
        }


def _serialize_pair_map(mapping: dict[tuple[int, int], list[list[float]]]) -> list[dict[str, object]]:
    return [
        {"layer": int(layer), "head": int(head), "rows": rows}
        for (layer, head), rows in sorted(mapping.items())
    ]


def _deserialize_pair_map(rows: list[dict[str, object]]) -> dict[tuple[int, int], list[list[float]]]:
    return {
        (int(row["layer"]), int(row["head"])): [list(vector) for vector in row["rows"]]
        for row in rows
    }


def _serialize_output_targets(
    output_targets: dict[tuple[int, int, int], list[float]],
) -> list[dict[str, object]]:
    return [
        {
            "layer": int(layer),
            "head": int(head),
            "token_index": int(token_index),
            "value": value,
        }
        for (layer, head, token_index), value in sorted(output_targets.items())
    ]


def _deserialize_output_targets(rows: list[dict[str, object]]) -> dict[tuple[int, int, int], list[float]]:
    return {
        (int(row["layer"]), int(row["head"]), int(row["token_index"])): list(row["value"])
        for row in rows
    }


def write_boundary_collection(bundle: BoundaryCollection, output_path: Path) -> Path:
    if bundle.runtime_cache is not None:
        raise ValueError("BoundaryCollection with runtime_cache cannot be serialized.")
    payload = {
        "harvest": bundle.harvest.to_serializable(),
        "query_bank": bundle.query_bank.to_serializable(),
        "boundary_keys": _serialize_pair_map(bundle.boundary_keys),
        "boundary_values": _serialize_pair_map(bundle.boundary_values),
        "boundary_projected_values": _serialize_pair_map(bundle.boundary_projected_values),
        "output_targets": _serialize_output_targets(bundle.output_targets),
        "runtime_cache_retained": False,
        "capture_token_indices": list(bundle.capture_token_indices or []),
        "monitored_observation_count": bundle.monitored_observation_count,
        "monitored_query_sample_count": bundle.monitored_query_sample_count,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_boundary_collection(path: Path) -> BoundaryCollection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("runtime_cache_retained"):
        raise ValueError("Serialized boundary collection unexpectedly retained a runtime cache.")
    harvest_payload = payload["harvest"]
    query_bank_payload = payload["query_bank"]
    return BoundaryCollection(
        harvest=FeatureHarvest(
            sample_id=str(harvest_payload["sample_id"]),
            boundary_id=str(harvest_payload["boundary_id"]),
            logical_context_tokens=int(harvest_payload["logical_context_tokens"]),
            physical_context_tokens=int(harvest_payload["physical_context_tokens"]),
            feature_granularity=str(harvest_payload["feature_granularity"]),
            tap_point=str(harvest_payload["tap_point"]),
            query_projection_dim=int(harvest_payload["query_projection_dim"]),
            output_projection_dim=int(harvest_payload["output_projection_dim"]),
            observed_layers=[int(layer) for layer in harvest_payload["observed_layers"]],
            observed_heads=[int(head) for head in harvest_payload["observed_heads"]],
            observation_count=int(harvest_payload["observation_count"]),
            observations=[FeatureObservation(**row) for row in harvest_payload["observations"]],
        ),
        query_bank=QuerySampleBank(
            sample_id=str(query_bank_payload["sample_id"]),
            boundary_id=str(query_bank_payload["boundary_id"]),
            query_dim=int(query_bank_payload["query_dim"]),
            sample_count=int(query_bank_payload["sample_count"]),
            samples=[QuerySample(**row) for row in query_bank_payload["samples"]],
        ),
        boundary_keys=_deserialize_pair_map(payload["boundary_keys"]),
        boundary_values=_deserialize_pair_map(payload["boundary_values"]),
        boundary_projected_values=_deserialize_pair_map(payload["boundary_projected_values"]),
        output_targets=_deserialize_output_targets(payload["output_targets"]),
        runtime_cache=None,
        capture_token_indices=[int(index) for index in payload.get("capture_token_indices", [])],
        monitored_observation_count=(
            int(payload["monitored_observation_count"])
            if payload.get("monitored_observation_count") is not None
            else None
        ),
        monitored_query_sample_count=(
            int(payload["monitored_query_sample_count"])
            if payload.get("monitored_query_sample_count") is not None
            else None
        ),
    )


def select_long_context_capture_indices(prefix_token_count: int, stride: int = LONG_CONTEXT_TOKEN_STRIDE) -> list[int]:
    if prefix_token_count <= 1:
        return []
    indices = list(range(stride - 1, prefix_token_count, stride))
    last_prefix_index = prefix_token_count - 1
    if not indices or indices[-1] != last_prefix_index:
        indices.append(last_prefix_index)
    return indices


def select_boundary_biased_capture_indices(
    prefix_token_count: int,
    turn_spans: list[tuple[int, int, str, str]],
    *,
    lookback_turns: int = 3,
    stride: int = LONG_CONTEXT_TOKEN_STRIDE,
) -> list[int]:
    if prefix_token_count <= 1:
        return []
    if not turn_spans:
        return select_long_context_capture_indices(prefix_token_count, stride=stride)

    eligible_turn_spans: list[tuple[int, int]] = []
    for start, end, _turn_id, _speaker in turn_spans:
        start = max(0, int(start))
        end = min(prefix_token_count, int(end))
        if start >= prefix_token_count:
            break
        if end <= start:
            continue
        eligible_turn_spans.append((start, end))
    if not eligible_turn_spans:
        return select_long_context_capture_indices(prefix_token_count, stride=stride)

    selected_turn_spans = eligible_turn_spans[-max(1, int(lookback_turns)) :]
    indices: set[int] = set()
    for turn_start, turn_end in selected_turn_spans:
        local_indices = list(range(turn_start + stride - 1, turn_end, stride))
        if not local_indices or local_indices[-1] != turn_end - 1:
            local_indices.append(turn_end - 1)
        indices.update(index for index in local_indices if 0 <= index < prefix_token_count)
    return sorted(indices)


def _capture_chunks(capture_indices: list[int], *, max_chunk_size: int) -> list[tuple[int, int]]:
    if not capture_indices:
        return []
    chunks: list[tuple[int, int]] = []
    chunk_start = int(capture_indices[0])
    previous_index = int(capture_indices[0])
    chunk_size = 1
    for capture_index in capture_indices[1:]:
        capture_index = int(capture_index)
        if capture_index == previous_index + 1 and chunk_size < max_chunk_size:
            previous_index = capture_index
            chunk_size += 1
            continue
        chunks.append((chunk_start, previous_index + 1))
        chunk_start = capture_index
        previous_index = capture_index
        chunk_size = 1
    chunks.append((chunk_start, previous_index + 1))
    return chunks


def _resolve_probe_layer_heads(
    probe_layers: tuple[int, ...],
    probe_heads: tuple[int, ...],
    probe_layer_heads: tuple[tuple[int, int], ...] | None,
) -> dict[int, tuple[int, ...]]:
    if probe_layer_heads is None:
        return {int(layer): tuple(int(head) for head in probe_heads) for layer in probe_layers}
    layer_to_heads: dict[int, list[int]] = {}
    for layer, head in probe_layer_heads:
        layer_to_heads.setdefault(int(layer), []).append(int(head))
    return {layer: tuple(dict.fromkeys(heads)) for layer, heads in layer_to_heads.items()}


def _clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    try:
        import torch
    except ImportError:
        torch = None

    def _clone_value(value):
        if torch is not None and torch.is_tensor(value):
            return value.clone()
        if isinstance(value, list):
            return [_clone_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_clone_value(item) for item in value)
        if isinstance(value, dict):
            return {key: _clone_value(item) for key, item in value.items()}
        if isinstance(value, (int, float, bool, str, type(None))):
            return value
        if hasattr(value, "__dict__"):
            cloned = object.__new__(type(value))
            for key, item in value.__dict__.items():
                setattr(cloned, key, _clone_value(item))
            return cloned
        return copy.deepcopy(value)

    return _clone_value(past_key_values)


def _attention_block_for_layer(model, layer_index: int):
    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "layers"):
        if not 0 <= int(layer_index) < len(base_model.layers):
            return None
        layer = base_model.layers[layer_index]
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        if hasattr(layer, "attn"):
            return layer.attn
        return None
    raise ValueError("Unable to discover attention blocks for the loaded model.")


def _attention_tensor_for_layer(model, attentions, layer_index: int):
    if not attentions:
        return None
    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "layers"):
        full_attention_layers = [
            idx for idx, layer in enumerate(base_model.layers) if getattr(layer, "layer_type", None) == "full_attention"
        ]
        if len(attentions) == len(full_attention_layers):
            if layer_index not in full_attention_layers:
                return None
            return attentions[full_attention_layers.index(layer_index)]
        if layer_index < len(attentions):
            return attentions[layer_index]
    if layer_index >= len(attentions):
        return None
    return attentions[layer_index]


def _cache_layer_count(cache) -> int:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return len(layers)
    key_cache = getattr(cache, "key_cache", None)
    if key_cache is not None:
        return len(key_cache)
    try:
        return len(cache)
    except TypeError:
        return 0


def _cache_layer_key_value(cache, layer_index: int):
    if cache is None:
        return None
    layers = getattr(cache, "layers", None)
    if layers is not None:
        if not 0 <= int(layer_index) < len(layers):
            return None
        layer = layers[layer_index]
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            return None
        return keys[0], values[0]
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        if not 0 <= int(layer_index) < len(key_cache):
            return None
        layer_key_cache = key_cache[layer_index]
        layer_value_cache = value_cache[layer_index]
        if layer_key_cache is None or layer_value_cache is None:
            return None
        return layer_key_cache[0], layer_value_cache[0]
    return None


def _projection_matrix(input_dim: int, output_dim: int, seed: int, device):
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn((input_dim, output_dim), generator=generator, device=device, dtype=torch.float32)


def _rounded_tensor_rows_to_lists(tensor) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in tensor.detach().cpu().tolist()]


def _rounded_tensor_to_list(tensor) -> list[float]:
    return [round(float(value), 6) for value in tensor.detach().cpu().tolist()]


def _project_rows(vectors, output_dim: int, seed: int) -> list[list[float]]:
    projected = vectors.float() @ _projection_matrix(vectors.shape[-1], output_dim, seed, vectors.device)
    return _rounded_tensor_rows_to_lists(projected)


def _project_vector(vector, output_dim: int, seed: int) -> list[float]:
    projected = vector.float() @ _projection_matrix(vector.shape[-1], output_dim, seed, vector.device)
    return _rounded_tensor_rows_to_lists(projected.unsqueeze(0))[0]


def _build_capture_rows_from_trace_payload(
    *,
    trace_payload,
    token_index: int,
    config: SmokeTestConfig,
):
    if not trace_payload:
        return []

    layer_indices = trace_payload["layer_indices"]
    head_indices = trace_payload["head_indices"]
    prefix_mass_shares = trace_payload["prefix_mass_shares"]
    raw_query_vectors = trace_payload["raw_query_vectors"]
    raw_outputs = trace_payload["raw_outputs"]
    if layer_indices.numel() == 0:
        return []

    rows = []
    for row_index in range(int(layer_indices.shape[0])):
        layer = int(layer_indices[row_index].item())
        head = int(head_indices[row_index].item())
        raw_query = raw_query_vectors[row_index]
        raw_output = raw_outputs[row_index]
        prefix_mass_share = round(float(prefix_mass_shares[row_index].item()), 6)
        if prefix_mass_share <= 0.0:
            continue
        rows.append(
            {
                "layer": layer,
                "head": head,
                "prefix_mass_share": prefix_mass_share,
                "raw_prefix_mass": round(float(token_index * prefix_mass_share), 6),
                "query_projection": _project_vector(
                    raw_query,
                    config.feature_schema.query_projection_dim,
                    config.experiment.seed + layer,
                ),
                "raw_query_vector": _rounded_tensor_to_list(raw_query),
                "output_projection": _project_vector(
                    raw_output,
                    config.feature_schema.output_projection_dim,
                    config.experiment.seed + 10_000 + layer,
                ),
                "raw_output": _rounded_tensor_to_list(raw_output),
            }
        )
    return rows


def _attention_query_states(attention_block, layer_inputs, num_heads: int):
    import torch

    projected_queries = attention_block.q_proj(layer_inputs)
    head_dim = int(getattr(attention_block, "head_dim", 0) or 0)
    if head_dim > 0 and projected_queries.shape[-1] == num_heads * head_dim * 2:
        query_states, _ = torch.chunk(
            projected_queries.view(*projected_queries.shape[:-1], num_heads, head_dim * 2),
            2,
            dim=-1,
        )
        query_norm = getattr(attention_block, "q_norm", None)
        if query_norm is not None:
            query_states = query_norm(query_states)
        return query_states
    return projected_queries.view(*projected_queries.shape[:-1], num_heads, -1)


def _turn_for_token_index(token_index: int, turn_spans: list[tuple[int, int, str, str]]) -> tuple[str, str]:
    for start, end, turn_id, speaker in turn_spans:
        if int(start) <= int(token_index) < int(end):
            return str(turn_id), str(speaker)
    if not turn_spans:
        return "", ""
    _, _, turn_id, speaker = turn_spans[-1]
    return str(turn_id), str(speaker)


def _build_capture_rows(
    *,
    model,
    outputs,
    token_index: int,
    query_position: int,
    config: SmokeTestConfig,
    probe_head_map: dict[int, tuple[int, ...]],
):
    hidden_states = outputs.hidden_states or ()
    attentions = outputs.attentions or ()
    if not hidden_states or not attentions:
        raise ValueError("Boundary capture outputs did not include hidden states and attentions.")
    current_cache = outputs.past_key_values
    if current_cache is None:
        raise ValueError("Boundary capture outputs did not include an updated cache.")

    import torch

    num_heads = int(model.config.num_attention_heads)
    num_key_value_heads = int(getattr(model.config, "num_key_value_heads", num_heads))
    repeat_factor = max(1, num_heads // num_key_value_heads)
    rows = []
    for layer_index, layer_probe_heads in probe_head_map.items():
        attention_block = _attention_block_for_layer(model, layer_index)
        layer_attention_tensor = _attention_tensor_for_layer(model, attentions, layer_index)
        if attention_block is None or layer_attention_tensor is None:
            continue
        max_head_index = layer_attention_tensor.shape[1] - 1
        valid_heads = [head for head in layer_probe_heads if head <= max_head_index]
        if not valid_heads:
            continue

        layer_inputs = hidden_states[layer_index][0, query_position].clone()
        query_states = _attention_query_states(attention_block, layer_inputs, num_heads)
        layer_attentions = layer_attention_tensor[0, :, query_position, :token_index]
        cache_pair = _cache_layer_key_value(current_cache, layer_index)
        if cache_pair is None:
            continue
        _, value_cache = cache_pair

        head_index_tensor = torch.tensor(valid_heads, device=query_states.device, dtype=torch.long)
        selected_query_states = query_states.index_select(0, head_index_tensor)
        selected_attentions = layer_attentions.index_select(0, head_index_tensor)
        prefix_mass_shares = selected_attentions.sum(dim=1)
        valid_mask = prefix_mass_shares > 0.0
        if not torch.any(valid_mask):
            continue

        head_index_tensor = head_index_tensor[valid_mask]
        selected_query_states = selected_query_states[valid_mask]
        selected_attentions = selected_attentions[valid_mask]
        prefix_mass_shares = prefix_mass_shares[valid_mask]
        kv_head_indices = torch.clamp(head_index_tensor // repeat_factor, max=num_key_value_heads - 1)
        prefix_values = value_cache.index_select(0, kv_head_indices)[:, :token_index, :]
        prefix_outputs = torch.bmm(selected_attentions.unsqueeze(1), prefix_values).squeeze(1)

        for row_index in range(head_index_tensor.shape[0]):
            layer = int(layer_index)
            head = int(head_index_tensor[row_index].item())
            raw_query = selected_query_states[row_index]
            raw_output = prefix_outputs[row_index]
            prefix_mass_share = round(float(prefix_mass_shares[row_index].item()), 6)
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "prefix_mass_share": prefix_mass_share,
                    "raw_prefix_mass": round(float(token_index * prefix_mass_share), 6),
                    "query_projection": _project_vector(
                        raw_query,
                        config.feature_schema.query_projection_dim,
                        config.experiment.seed + layer,
                    ),
                    "raw_query_vector": _rounded_tensor_to_list(raw_query),
                    "output_projection": _project_vector(
                        raw_output,
                        config.feature_schema.output_projection_dim,
                        config.experiment.seed + 10_000 + layer,
                    ),
                    "raw_output": _rounded_tensor_to_list(raw_output),
                }
            )
    return rows


def _can_use_qwen35_trace_prompt_capture(model) -> bool:
    base_model = getattr(model, "model", None)
    attn_impl = str(getattr(model.config, "_attn_implementation", "") or "")
    return base_model is not None and hasattr(base_model, "layers") and attn_impl == "eager"


@contextmanager
def _patched_qwen35_attention_trace_chunk(
    trace_layer_heads: tuple[tuple[int, int], ...],
    trace_buffer: AttentionTraceChunkBuffer,
    *,
    chunk_start_position: int,
):
    import torch

    modeling_qwen35 = import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    original_attention = modeling_qwen35.eager_attention_forward
    layer_head_map: dict[int, tuple[int, ...]] = {}
    for layer, head in trace_layer_heads:
        layer_head_map.setdefault(int(layer), []).append(int(head))

    head_index_cache: dict[tuple[int, str], torch.Tensor] = {}

    def traced_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        attn_output, attn_weights = original_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        if attn_weights is None:
            return attn_output, attn_weights
        layer_probe_heads = layer_head_map.get(int(module.layer_idx))
        if not layer_probe_heads:
            return attn_output, attn_weights

        cache_key = (int(module.layer_idx), str(query.device))
        head_index_tensor = head_index_cache.get(cache_key)
        if head_index_tensor is None:
            head_index_tensor = torch.tensor(layer_probe_heads, device=query.device, dtype=torch.long)
            head_index_cache[cache_key] = head_index_tensor

        value_states = modeling_qwen35.repeat_kv(value, module.num_key_value_groups)
        query_length = int(query.shape[2])
        total_key_length = int(value_states.shape[2])
        prefix_before_chunk = max(0, total_key_length - query_length)

        selected_queries = query[0].index_select(0, head_index_tensor)
        selected_weights = attn_weights[0].index_select(0, head_index_tensor)
        selected_values = value_states[0].index_select(0, head_index_tensor)
        layer_tensor = torch.full_like(head_index_tensor, int(module.layer_idx))

        for query_position in range(query_length):
            absolute_query_position = int(chunk_start_position) + query_position
            prefix_length = prefix_before_chunk + query_position
            if prefix_length <= 0:
                trace_buffer.add_query_position(
                    query_position=query_position,
                    absolute_query_position=absolute_query_position,
                    layer_indices=layer_tensor[:0],
                    head_indices=head_index_tensor[:0],
                    prefix_mass_shares=selected_queries.new_empty((0,)),
                    raw_query_vectors=selected_queries[:, query_position, :][:0],
                    raw_outputs=selected_queries[:, query_position, :][:0],
                )
                continue
            query_rows = selected_queries[:, query_position, :]
            weight_rows = selected_weights[:, query_position, :prefix_length]
            value_rows = selected_values[:, :prefix_length, :]
            prefix_mass_shares = weight_rows.sum(dim=1)
            prefix_outputs = torch.bmm(weight_rows.unsqueeze(1), value_rows).squeeze(1)
            trace_buffer.add_query_position(
                query_position=query_position,
                absolute_query_position=absolute_query_position,
                layer_indices=layer_tensor,
                head_indices=head_index_tensor,
                prefix_mass_shares=prefix_mass_shares,
                raw_query_vectors=query_rows,
                raw_outputs=prefix_outputs,
            )
        return attn_output, attn_weights

    modeling_qwen35.eager_attention_forward = traced_attention_forward
    try:
        yield
    finally:
        modeling_qwen35.eager_attention_forward = original_attention


def collect_teacher_forced_boundary_collection(
    sample: LoadedContextSample,
    config: SmokeTestConfig,
    *,
    model=None,
    tokenizer=None,
    probe_layers: tuple[int, ...] | None = None,
    probe_heads: tuple[int, ...] | None = None,
    probe_layer_heads: tuple[tuple[int, int], ...] | None = None,
    capture_indices: list[int] | None = None,
    retain_runtime_cache: bool = False,
    materialize_boundary_kv: bool = True,
    progress_callback=None,
    initial_past_key_values=None,
    replay_start_position: int = 0,
) -> BoundaryCollection:
    created_model = False
    model_type = "qwen3_5"
    if model is None or tokenizer is None:
        eager_config = config
        model, tokenizer, model_type = load_qwen35_bundle(eager_config)
        created_model = True
    try:
        layers = probe_layers or default_probe_layers_for_model(model, model_type)
        heads = probe_heads or default_probe_heads_for_model(model)
        return _collect_boundary_collection_with_model(
            sample=sample,
            config=config,
            model=model,
            tokenizer=tokenizer,
            probe_layers=layers,
            probe_heads=heads,
            probe_layer_heads=probe_layer_heads,
            capture_indices=capture_indices,
            retain_runtime_cache=retain_runtime_cache,
            materialize_boundary_kv=materialize_boundary_kv,
            progress_callback=progress_callback,
            initial_past_key_values=initial_past_key_values,
            replay_start_position=replay_start_position,
        )
    finally:
        if created_model:
            unload_qwen35_bundle(model)


def _collect_boundary_collection_with_model(
    *,
    sample: LoadedContextSample,
    config: SmokeTestConfig,
    model,
    tokenizer,
    probe_layers: tuple[int, ...],
    probe_heads: tuple[int, ...],
    probe_layer_heads: tuple[tuple[int, int], ...] | None,
    capture_indices: list[int] | None,
    retain_runtime_cache: bool,
    materialize_boundary_kv: bool,
    progress_callback,
    initial_past_key_values,
    replay_start_position: int,
) -> BoundaryCollection:
    import torch

    token_ids, turn_spans = materialize_long_context_ids(sample, tokenizer)
    prefix_token_count = int(sample.boundary.prefix_token_count)
    replay_start_position = int(replay_start_position)
    if replay_start_position < 0 or replay_start_position > prefix_token_count:
        raise ValueError(
            f"replay_start_position must be within [0, {prefix_token_count}], got {replay_start_position}."
        )
    if replay_start_position > 0 and initial_past_key_values is None:
        raise ValueError("initial_past_key_values is required when replay_start_position > 0.")
    replay_token_ids = token_ids[replay_start_position:prefix_token_count]
    full_input_ids = torch.tensor([replay_token_ids], device=config.model.device, dtype=torch.long)
    if capture_indices is None:
        capture_indices = select_long_context_capture_indices(prefix_token_count)
    capture_indices = sorted({int(index) for index in capture_indices if 0 <= int(index) < prefix_token_count})
    if replay_start_position > 0 and any(index < replay_start_position for index in capture_indices):
        raise ValueError("capture_indices before replay_start_position cannot be collected from a replay start cache.")

    observations: list[FeatureObservation] = []
    query_samples: list[QuerySample] = []
    output_targets: dict[tuple[int, int, int], list[float]] = {}
    probe_head_map = _resolve_probe_layer_heads(probe_layers, probe_heads, probe_layer_heads)
    use_trace_prompt_capture = _can_use_qwen35_trace_prompt_capture(model)
    trace_layer_heads = tuple(
        (int(layer_index), int(head_index))
        for layer_index, layer_probe_heads in sorted(probe_head_map.items())
        for head_index in layer_probe_heads
    )
    capture_index_set = set(capture_indices)
    past_key_values = _clone_past_key_values(initial_past_key_values)
    processed_token_count = replay_start_position
    while processed_token_count < prefix_token_count:
        chunk_end = min(prefix_token_count, processed_token_count + int(config.model.prefill_chunk_size))
        local_start = processed_token_count - replay_start_position
        local_end = chunk_end - replay_start_position
        chunk_input_ids = full_input_ids[:, local_start:local_end]
        chunk_indices = list(range(processed_token_count, chunk_end))
        overlapping_capture_indices = [index for index in chunk_indices if index in capture_index_set]
        trace_buffer = None
        patch_context = nullcontext()
        capture_kwargs = {
            "input_ids": chunk_input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
            "cache_position": torch.arange(
                processed_token_count,
                chunk_end,
                device=config.model.device,
                dtype=torch.long,
            ),
        }
        if overlapping_capture_indices:
            if use_trace_prompt_capture:
                trace_buffer = AttentionTraceChunkBuffer(
                    capacity=max(1, len(trace_layer_heads) * max(1, len(overlapping_capture_indices))),
                    query_length=max(1, chunk_end - processed_token_count),
                    tracked_absolute_positions=set(overlapping_capture_indices),
                )
                patch_context = _patched_qwen35_attention_trace_chunk(
                    trace_layer_heads,
                    trace_buffer,
                    chunk_start_position=processed_token_count,
                )
            else:
                capture_kwargs["output_hidden_states"] = True
                capture_kwargs["output_attentions"] = True
        with torch.inference_mode():
            with patch_context:
                chunk_outputs = model(**capture_kwargs)

        if overlapping_capture_indices:
            for capture_index in overlapping_capture_indices:
                query_position = capture_index - processed_token_count
                source_turn_id, source_speaker = _turn_for_token_index(capture_index, turn_spans)
                rows = (
                    _build_capture_rows_from_trace_payload(
                        trace_payload=(
                            trace_buffer.snapshot_for_query_position(capture_index)
                            if trace_buffer is not None
                            else None
                        ),
                        token_index=capture_index,
                        config=config,
                    )
                    if use_trace_prompt_capture
                    else _build_capture_rows(
                        model=model,
                        outputs=chunk_outputs,
                        token_index=capture_index,
                        query_position=query_position,
                        config=config,
                        probe_head_map=probe_head_map,
                    )
                )
                for row in rows:
                    observations.append(
                        FeatureObservation(
                            token_index=capture_index,
                            layer=int(row["layer"]),
                            head=int(row["head"]),
                            tap_point=config.feature_schema.tap_point,
                            query_projection=list(row["query_projection"]),
                            prefix_mass_share=float(row["prefix_mass_share"]),
                            raw_prefix_mass=float(row["raw_prefix_mass"]),
                            output_projection=list(row["output_projection"]),
                            source_turn_id=source_turn_id,
                            source_speaker=source_speaker,
                        )
                    )
                    query_samples.append(
                        QuerySample(
                            query_id=f"{sample.sample_id}:{row['layer']}:{row['head']}:{capture_index}",
                            layer=int(row["layer"]),
                            head=int(row["head"]),
                            token_index=capture_index,
                            prefix_mass_share=float(row["prefix_mass_share"]),
                            raw_prefix_mass=float(row["raw_prefix_mass"]),
                            query_projection=list(row["query_projection"]),
                            raw_query_vector=list(row["raw_query_vector"]),
                            source_turn_id=source_turn_id,
                            source_speaker=source_speaker,
                        )
                    )
                    output_targets[(int(row["layer"]), int(row["head"]), int(capture_index))] = list(row["raw_output"])

        past_key_values = chunk_outputs.past_key_values
        processed_token_count = chunk_end
        if progress_callback is not None:
            payload = {
                "stage": "capture" if overlapping_capture_indices else "prefill",
                "processed_token_count": processed_token_count,
                "prefix_token_count": prefix_token_count,
            }
            if overlapping_capture_indices:
                payload["monitored_observation_count"] = len(observations)
                payload["monitored_query_sample_count"] = len(query_samples)
            progress_callback(payload)

    if past_key_values is None:
        raise ValueError("Boundary collection did not produce a final cache.")

    num_heads = int(model.config.num_attention_heads)
    num_key_value_heads = int(getattr(model.config, "num_key_value_heads", num_heads))
    repeat_factor = max(1, num_heads // num_key_value_heads)
    boundary_keys: dict[tuple[int, int], list[list[float]]] = {}
    boundary_values: dict[tuple[int, int], list[list[float]]] = {}
    boundary_projected_values: dict[tuple[int, int], list[list[float]]] = {}
    if materialize_boundary_kv:
        for layer_index, layer_probe_heads in probe_head_map.items():
            if layer_index >= _cache_layer_count(past_key_values):
                continue
            cache_pair = _cache_layer_key_value(past_key_values, layer_index)
            if cache_pair is None:
                continue
            key_cache, value_cache = cache_pair
            for head_index in layer_probe_heads:
                kv_head_index = min(num_key_value_heads - 1, int(head_index) // repeat_factor)
                prefix_key_cache = key_cache[kv_head_index, :prefix_token_count, :]
                prefix_value_cache = value_cache[kv_head_index, :prefix_token_count, :]
                boundary_keys[(int(layer_index), int(head_index))] = _rounded_tensor_rows_to_lists(prefix_key_cache)
                boundary_values[(int(layer_index), int(head_index))] = _rounded_tensor_rows_to_lists(prefix_value_cache)
                boundary_projected_values[(int(layer_index), int(head_index))] = _project_rows(
                    prefix_value_cache,
                    config.feature_schema.output_projection_dim,
                    config.experiment.seed + 10_000 + int(layer_index),
                )

    harvest = FeatureHarvest(
        sample_id=sample.sample_id,
        boundary_id=sample.boundary.boundary_id,
        logical_context_tokens=prefix_token_count,
        physical_context_tokens=prefix_token_count,
        feature_granularity=config.feature_schema.granularity,
        tap_point=config.feature_schema.tap_point,
        query_projection_dim=config.feature_schema.query_projection_dim,
        output_projection_dim=config.feature_schema.output_projection_dim,
        observed_layers=sorted({int(observation.layer) for observation in observations}),
        observed_heads=sorted({int(observation.head) for observation in observations}),
        observation_count=len(observations),
        observations=observations,
    )
    query_bank = QuerySampleBank(
        sample_id=sample.sample_id,
        boundary_id=sample.boundary.boundary_id,
        query_dim=len(query_samples[0].raw_query_vector) if query_samples else 0,
        sample_count=len(query_samples),
        samples=query_samples,
    )
    return BoundaryCollection(
        harvest=harvest,
        query_bank=query_bank,
        boundary_keys=boundary_keys,
        boundary_values=boundary_values,
        boundary_projected_values=boundary_projected_values,
        output_targets=output_targets,
        runtime_cache=past_key_values if retain_runtime_cache or not materialize_boundary_kv else None,
        capture_token_indices=list(capture_indices),
        monitored_observation_count=len(observations),
        monitored_query_sample_count=len(query_samples),
    )
