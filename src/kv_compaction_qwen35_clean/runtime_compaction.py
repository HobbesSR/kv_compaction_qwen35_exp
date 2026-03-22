from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
import math

from kv_compaction_qwen35_clean.beta_fit import (
    _fit_nonnegative_ridge_scale_matrix,
    _fit_nonnegative_scale_matrix,
    split_query_bank_train_eval,
)
from kv_compaction_qwen35_clean.data_types import BoundaryCollection, CompactHeadRuntime, QuerySample
from kv_compaction_qwen35_clean.key_selection import match_coreset_to_query_samples, select_keys
from kv_compaction_qwen35_clean.model_runtime import QWEN35_MODELING_MODULES


BETA_SOLVER = "clamped_ridge"
BETA_REGULARIZATION = 0.01
VALUE_REGULARIZATION = 0.01
TRAIN_FRACTION = 0.75


def _fit_scale(design, target, solver: str, regularization_strength: float):
    if solver == "clamped_lstsq":
        return _fit_nonnegative_scale_matrix(design, target)
    if solver == "clamped_ridge":
        return _fit_nonnegative_ridge_scale_matrix(
            design,
            target,
            regularization_strength=regularization_strength,
        )
    raise ValueError(f"Unsupported beta solver {solver!r}.")


def _fit_head_runtime(
    group,
    train_samples: list[QuerySample],
    bundle: BoundaryCollection,
    full_key_tensor,
    full_value_tensor,
    compute_device: str,
) -> CompactHeadRuntime:
    import torch

    if not train_samples:
        raise ValueError(f"No training samples available for layer={group.layer}, head={group.head}.")

    selected_indices = group.selected_indices
    selected_key_tensor = full_key_tensor[selected_indices]
    train_query_tensor = torch.tensor(
        [sample.raw_query_vector for sample in train_samples],
        dtype=torch.float32,
        device=compute_device,
    )
    scale_norm = math.sqrt(train_query_tensor.shape[1])

    train_full_logits = (train_query_tensor @ full_key_tensor.T) / scale_norm
    train_selected_logits = (train_query_tensor @ selected_key_tensor.T) / scale_norm
    train_reference_max = train_full_logits.max(dim=1, keepdim=True).values
    train_target_mass = torch.exp(train_full_logits - train_reference_max).sum(dim=1)
    train_design = torch.exp(train_selected_logits - train_reference_max)
    scale = _fit_scale(
        train_design,
        train_target_mass,
        solver=BETA_SOLVER,
        regularization_strength=BETA_REGULARIZATION,
    )

    train_weight_matrix = (train_design * scale.unsqueeze(0)) / torch.clamp_min(
        train_target_mass.unsqueeze(1),
        1e-12,
    )
    train_target_output = torch.tensor(
        [bundle.output_targets[(sample.layer, sample.head, sample.token_index)] for sample in train_samples],
        dtype=torch.float32,
        device=compute_device,
    )

    gram = train_weight_matrix.T @ train_weight_matrix
    rhs = train_weight_matrix.T @ train_target_output
    diag_mean = torch.clamp_min(torch.diagonal(gram).mean(), 1e-12)
    ridge = torch.eye(
        gram.shape[0],
        dtype=train_weight_matrix.dtype,
        device=train_weight_matrix.device,
    )
    compact_values = torch.linalg.solve(gram + ridge * (diag_mean * VALUE_REGULARIZATION), rhs)

    return CompactHeadRuntime(
        layer=group.layer,
        head=group.head,
        selected_indices=selected_indices,
        compact_keys=selected_key_tensor.detach(),
        compact_values=compact_values.detach(),
        beta=torch.log(scale).detach(),
    )


def build_path_runtime(
    sample_id: str,
    boundary_id: str,
    source: str,
    keys_per_head: int,
    bundle: BoundaryCollection,
    query_source,
    target_layers: tuple[int, ...],
    target_heads: tuple[int, ...],
    target_layer_heads: tuple[tuple[int, int], ...] | None = None,
    compute_device: str = "cpu",
    key_selection_method: str = "highest_attention",
    head_budget_proportions: dict[tuple[int, int], float] | None = None,
    min_keys_per_head: int = 1,
):
    import torch

    matches = match_coreset_to_query_samples(query_source, bundle.query_bank.samples)
    selection = select_keys(
        sample_id,
        boundary_id,
        source,
        matches,
        bundle.boundary_keys,
        keys_per_head,
        selection_method=key_selection_method,
        head_budget_proportions=head_budget_proportions,
        min_keys_per_head=min_keys_per_head,
    )

    runtimes: dict[int, dict[int, CompactHeadRuntime]] = {}
    allowed_pairs = (
        set(target_layer_heads)
        if target_layer_heads is not None
        else {(layer, head) for layer in target_layers for head in target_heads}
    )
    train_groups, _ = split_query_bank_train_eval(bundle.query_bank.samples, train_fraction=TRAIN_FRACTION)
    tensor_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for group in selection.groups:
        group_key = (group.layer, group.head)
        if group_key not in allowed_pairs:
            continue
        train_samples = train_groups.get(group_key, [])
        if not train_samples:
            continue
        cached_tensors = tensor_cache.get(group_key)
        if cached_tensors is None:
            cached_tensors = (
                torch.tensor(bundle.boundary_keys[group_key], dtype=torch.float32, device=compute_device),
                torch.tensor(bundle.boundary_values[group_key], dtype=torch.float32, device=compute_device),
            )
            tensor_cache[group_key] = cached_tensors
        runtimes.setdefault(group.layer, {})[group.head] = _fit_head_runtime(
            group,
            train_samples,
            bundle,
            full_key_tensor=cached_tensors[0],
            full_value_tensor=cached_tensors[1],
            compute_device=compute_device,
        )
    return selection, runtimes


def _load_qwen35_modeling_module(model_type: str):
    module_name = QWEN35_MODELING_MODULES.get(model_type)
    if module_name is None:
        raise RuntimeError(f"Unsupported model_type {model_type!r} for compaction attention patching.")
    try:
        return import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            "The current environment does not expose the Qwen3.5 modeling module needed "
            "for compacted continuation. Use the dedicated Qwen3.5 environment."
        ) from exc


@contextmanager
def patched_compaction_attention(
    compacted_layers: dict[int, dict[int, CompactHeadRuntime]],
    prefix_token_count: int,
    *,
    model_type: str = "qwen3_5",
):
    import torch

    modeling_qwen35 = _load_qwen35_modeling_module(model_type)
    original_attention = modeling_qwen35.eager_attention_forward

    def compacted_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        base_output, _ = original_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        layer_runtimes = compacted_layers.get(module.layer_idx)
        if not layer_runtimes:
            return base_output, None

        key_states = modeling_qwen35.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_qwen35.repeat_kv(value, module.num_key_value_groups)
        attn_output = base_output.clone()
        batch_size = query.shape[0]
        query_length = query.shape[2]

        for head_index, runtime in layer_runtimes.items():
            tail_keys = key_states[:, head_index : head_index + 1, prefix_token_count:, :]
            tail_values = value_states[:, head_index : head_index + 1, prefix_token_count:, :]
            compact_keys = runtime.compact_keys.to(device=query.device, dtype=query.dtype)
            compact_values = runtime.compact_values.to(device=query.device, dtype=query.dtype)
            beta = runtime.beta.to(device=query.device, dtype=query.dtype)

            compact_key_tensor = compact_keys.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
            compact_value_tensor = compact_values.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
            custom_keys = torch.cat([compact_key_tensor, tail_keys], dim=2)
            custom_values = torch.cat([compact_value_tensor, tail_values], dim=2)

            head_query = query[:, head_index : head_index + 1, :, :]
            head_logits = torch.matmul(head_query, custom_keys.transpose(2, 3)) * scaling
            prefix_bias = beta.view(1, 1, 1, -1).expand(batch_size, 1, query_length, -1)
            if tail_keys.shape[2] > 0:
                tail_bias = torch.zeros(
                    (batch_size, 1, query_length, tail_keys.shape[2]),
                    device=query.device,
                    dtype=query.dtype,
                )
                head_logits = head_logits + torch.cat([prefix_bias, tail_bias], dim=-1)
            else:
                head_logits = head_logits + prefix_bias

            if attention_mask is not None:
                full_mask = attention_mask[:, :, :, : key_states.shape[2]]
                tail_mask = full_mask[:, :, :, prefix_token_count:]
                if tail_mask.shape[-1] > 0:
                    prefix_mask = torch.zeros(
                        (batch_size, 1, query_length, compact_keys.shape[0]),
                        device=query.device,
                        dtype=attention_mask.dtype,
                    )
                    head_logits = head_logits + torch.cat([prefix_mask, tail_mask], dim=-1)

            head_weights = torch.softmax(head_logits, dim=-1, dtype=torch.float32).to(query.dtype)
            head_output = torch.matmul(head_weights, custom_values).transpose(1, 2)
            attn_output[:, :, head_index : head_index + 1, :] = head_output

        return attn_output, None

    modeling_qwen35.eager_attention_forward = compacted_attention_forward
    try:
        yield
    finally:
        modeling_qwen35.eager_attention_forward = original_attention
