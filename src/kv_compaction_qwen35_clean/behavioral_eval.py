from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import re
import time
from pathlib import Path

from kv_compaction_qwen35_clean.boundary_collection import collect_teacher_forced_boundary_collection
from kv_compaction_qwen35_clean.coreset import extract_query_coreset
from kv_compaction_qwen35_clean.data_types import (
    BehavioralEvalResult,
    BehavioralPathResult,
    BehavioralPrompt,
    BehavioralRunResult,
    FactExpectation,
)
from kv_compaction_qwen35_clean.model_runtime import (
    all_probe_heads_for_model,
    build_qwen35_prompt_ids,
    default_probe_heads_for_model,
    default_probe_layers_for_model,
    load_qwen35_bundle,
    materialize_long_context_ids,
    unload_qwen35_bundle,
)
from kv_compaction_qwen35_clean.prototype_bank import build_state_from_observations
from kv_compaction_qwen35_clean.query_controls import extract_teacher_forced_subsample_control
from kv_compaction_qwen35_clean.runtime_compaction import (
    BETA_REGULARIZATION,
    BETA_SOLVER,
    TRAIN_FRACTION,
    VALUE_REGULARIZATION,
    build_path_runtime,
    patched_compaction_attention,
)


MAX_NEW_TOKENS = 64
DEFAULT_PROMPT_SET = "qwen35_calibration_v3"


def _build_qwen35_warehouse_prompts() -> list[BehavioralPrompt]:
    return [
        BehavioralPrompt(
            label="qwen35_same_task_status_triplet",
            category="same_task",
            prompt_text=(
                "Answer with exactly three short bullets. "
                "Give: (1) the cutover window, (2) the check required before live traffic, and "
                "(3) the current blocker."
            ),
            required_facts=[
                FactExpectation("saturday_cutover", ["saturday cutover", "saturday"]),
                FactExpectation("firmware_validation_pass", ["firmware validation pass", "firmware validation"]),
                FactExpectation("harness_certification", ["delayed harness certification", "harness certification"]),
            ],
            forbidden_markers=[],
            target_head_labels=["4:0", "12:3"],
        ),
        BehavioralPrompt(
            label="qwen35_same_task_handoff_rollback",
            category="same_task",
            prompt_text=(
                "Answer with exactly two short bullets. "
                "Bullet 1: the operator handoff focus. "
                "Bullet 2: what happens if only dock three must roll back."
            ),
            required_facts=[
                FactExpectation("handoff_checklist", ["handoff checklist", "operator handoff checklist", "checklist"]),
                FactExpectation("dock_three", ["dock three", "dock 3"]),
                FactExpectation("rollback_ordering", ["rollback ordering", "rollback order"]),
            ],
            forbidden_markers=[],
            target_head_labels=["12:3", "20:0"],
        ),
        BehavioralPrompt(
            label="qwen35_branch_switch_harness_note",
            category="branch_switch",
            prompt_text=(
                "Answer directly in one sentence. "
                "Which dock needs a different relay harness, and where is that note easy to miss?"
            ),
            required_facts=[
                FactExpectation("dock_three", ["dock three", "dock 3"]),
                FactExpectation("relay_harness", ["relay harness"]),
                FactExpectation("late_appendix", ["late appendix", "appendix"]),
            ],
            forbidden_markers=[],
            target_head_labels=["12:7", "20:0"],
        ),
        BehavioralPrompt(
            label="qwen35_branch_switch_appendix_details",
            category="branch_switch",
            prompt_text=(
                "Answer with exactly three short bullets. "
                "List the late appendix details besides the relay harness note."
            ),
            required_facts=[
                FactExpectation("supplier_phones", ["supplier phone numbers", "supplier phone"]),
                FactExpectation("cage_inventory", ["cage inventory"]),
                FactExpectation("shift_lead_names", ["shift lead names", "shift leads"]),
            ],
            forbidden_markers=[],
            target_head_labels=["12:7", "28:7"],
        ),
    ]


PROMPT_SET_LABELS = {
    "qwen35_calibration_v0": [
        "qwen35_same_task_status_triplet",
        "qwen35_same_task_handoff_rollback",
        "qwen35_branch_switch_harness_note",
        "qwen35_branch_switch_appendix_details",
    ],
    "qwen35_calibration_v1": [
        "qwen35_same_task_status_triplet",
        "qwen35_same_task_handoff_rollback",
        "qwen35_branch_switch_harness_note",
        "qwen35_branch_switch_appendix_details",
    ],
    "qwen35_calibration_v2": [
        "qwen35_same_task_status_triplet",
        "qwen35_same_task_handoff_rollback",
        "qwen35_branch_switch_harness_note",
        "qwen35_branch_switch_appendix_details",
    ],
    "qwen35_calibration_v3": [
        "qwen35_same_task_status_triplet",
        "qwen35_same_task_handoff_rollback",
        "qwen35_branch_switch_harness_note",
        "qwen35_branch_switch_appendix_details",
    ],
}

QWEN35_PROMPT_PARAPHRASES = {
    "qwen35_same_task_status_triplet": (
        "Respond with exactly these three lines and nothing else.\n"
        "cutover_window: <time window>\n"
        "live_traffic_check: <required check>\n"
        "current_blocker: <blocker>\n"
        "Use phrases from the context."
    ),
    "qwen35_same_task_handoff_rollback": (
        "Respond with exactly these two lines and nothing else.\n"
        "handoff_focus: <operator handoff focus>\n"
        "dock_three_rollback: <rollback order if only dock three rolls back>\n"
        "Include the specific rollback order from the context."
    ),
    "qwen35_branch_switch_harness_note": (
        "Respond with exactly this one line and nothing else.\n"
        "dock_and_note: <dock> ; <where the note is easy to miss>"
    ),
    "qwen35_branch_switch_appendix_details": (
        "Respond with exactly these three lines and nothing else.\n"
        "detail_1: <late appendix detail>\n"
        "detail_2: <late appendix detail>\n"
        "detail_3: <late appendix detail>\n"
        "Use the three late appendix details besides the relay harness note."
    ),
}

QWEN35_PROMPT_PARAPHRASES_V2 = {
    "qwen35_same_task_status_triplet": (
        "Answer with exactly three short bullets and nothing else. "
        "Bullet 1: the cutover window. "
        "Bullet 2: the check required before live traffic, using the phrase firmware validation pass. "
        "Bullet 3: the current blocker, including delayed harness certification for dock three."
    ),
    "qwen35_same_task_handoff_rollback": (
        "Answer with exactly two short bullets and nothing else. "
        "Bullet 1: the operator handoff focus, including the handoff checklist. "
        "Bullet 2: if only dock three rolls back, give the rollback order through dock two and then dock one."
    ),
}

QWEN35_PROMPT_PARAPHRASES_V3 = {
    "qwen35_same_task_status_triplet": (
        "Answer with exactly three short bullets and nothing else. "
        "Use wording copied from the context where possible. "
        "Bullet 1: the cutover window. "
        "Bullet 2: the required check before live traffic. "
        "Bullet 3: the current blocker."
    ),
    "qwen35_same_task_handoff_rollback": (
        "Answer with exactly two short bullets and nothing else. "
        "Use wording copied from the context where possible. "
        "Bullet 1: the operator handoff focus. "
        "Bullet 2: the rollback sequence if only dock three rolls back."
    ),
}


def build_prompt_set(prompt_set: str = DEFAULT_PROMPT_SET, prompt_family: str = "warehouse_migration_qwen35") -> list[BehavioralPrompt]:
    if prompt_family != "warehouse_migration_qwen35":
        raise ValueError(f"Unsupported prompt family {prompt_family!r}.")
    prompts_by_label = {prompt.label: prompt for prompt in _build_qwen35_warehouse_prompts()}
    try:
        labels = PROMPT_SET_LABELS[prompt_set]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt set {prompt_set!r}.") from exc
    prompts: list[BehavioralPrompt] = []
    for label in labels:
        prompt = prompts_by_label[label]
        if prompt_set == "qwen35_calibration_v1":
            prompt = replace(prompt, prompt_text=QWEN35_PROMPT_PARAPHRASES[label])
        elif prompt_set == "qwen35_calibration_v2" and label in QWEN35_PROMPT_PARAPHRASES_V2:
            prompt = replace(prompt, prompt_text=QWEN35_PROMPT_PARAPHRASES_V2[label])
        elif prompt_set == "qwen35_calibration_v3" and label in QWEN35_PROMPT_PARAPHRASES_V3:
            prompt = replace(prompt, prompt_text=QWEN35_PROMPT_PARAPHRASES_V3[label])
        prompts.append(prompt)
    return prompts


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _cleanup_generated_text(text: str) -> str:
    cleaned = text.strip()
    role_marker = re.search(r"\b(?:USER|ASSISTANT|SYSTEM|TOOL) \[[^\]]+\]", cleaned)
    if role_marker is not None:
        cleaned = cleaned[: role_marker.start()].strip()
    return cleaned


def _keyword_recall(text: str, keyword_groups: list[list[str]]) -> tuple[int, int, float]:
    lowered = _normalise_text(text)
    hits = 0
    for group in keyword_groups:
        if any(keyword in lowered for keyword in group):
            hits += 1
    total = len(keyword_groups)
    recall = round((hits / total) if total else 0.0, 6)
    return hits, total, recall


def _token_counts(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9']+", _normalise_text(text)))


def _unigram_f1(reference: str, candidate: str) -> float:
    reference_counts = _token_counts(reference)
    candidate_counts = _token_counts(candidate)
    overlap = sum(min(reference_counts[token], candidate_counts[token]) for token in reference_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(candidate_counts.values()), 1)
    recall = overlap / max(sum(reference_counts.values()), 1)
    return round((2.0 * precision * recall) / max(precision + recall, 1e-12), 6)


def _fact_labels_hit(text: str, facts: list[FactExpectation]) -> list[str]:
    lowered = _normalise_text(text)
    return [fact.label for fact in facts if any(keyword in lowered for keyword in fact.keywords)]


def _hallucination_flags(text: str, forbidden_markers: list[str]) -> list[str]:
    lowered = _normalise_text(text)
    return sorted(marker for marker in forbidden_markers if marker in lowered)


def _divergence_summary(
    reference_hits: list[str],
    candidate_hits: list[str],
    hallucination_flags: list[str],
    reference_unigram_f1: float | None,
) -> str:
    if reference_unigram_f1 is None:
        return "reference run"
    missing_vs_reference = [label for label in reference_hits if label not in candidate_hits]
    extra_vs_reference = [label for label in candidate_hits if label not in reference_hits]
    parts = []
    if missing_vs_reference:
        parts.append(f"misses reference-hit facts: {', '.join(missing_vs_reference)}")
    if extra_vs_reference:
        parts.append(f"adds non-reference facts: {', '.join(extra_vs_reference)}")
    if hallucination_flags:
        parts.append(f"unsupported markers: {', '.join(hallucination_flags)}")
    if not parts:
        if reference_unigram_f1 >= 0.95:
            return "matches reference closely"
        return "preserves required facts with wording differences"
    return "; ".join(parts)


def evaluate_run(
    prompt: BehavioralPrompt,
    generated_text: str,
    runtime_seconds: float,
    reference_text: str | None = None,
    reference_hits: list[str] | None = None,
) -> BehavioralRunResult:
    required_fact_labels_hit = sorted(_fact_labels_hit(generated_text, prompt.required_facts))
    required_fact_labels_hit_set = set(required_fact_labels_hit)
    required_fact_labels = [fact.label for fact in prompt.required_facts]
    missing_required_fact_labels = [label for label in required_fact_labels if label not in required_fact_labels_hit_set]

    central_labels = [fact.label for fact in prompt.required_facts if fact.central]
    central_fact_labels_hit = [label for label in required_fact_labels_hit if label in set(central_labels)]
    missing_central_fact_labels = [label for label in central_labels if label not in required_fact_labels_hit_set]
    central_detail_preserved = not missing_central_fact_labels
    hallucination_flags = _hallucination_flags(generated_text, prompt.forbidden_markers)

    keyword_groups = [fact.keywords for fact in prompt.required_facts]
    keyword_hits, keyword_total, keyword_recall = _keyword_recall(generated_text, keyword_groups)
    reference_unigram = _unigram_f1(reference_text, generated_text) if reference_text is not None else None
    reference_hits = reference_hits or []
    reference_missing_fact_labels = [label for label in reference_hits if label not in required_fact_labels_hit_set]
    reference_extra_fact_labels = [label for label in required_fact_labels_hit if label not in set(reference_hits)]

    return BehavioralRunResult(
        label=prompt.label,
        category=prompt.category,
        prompt_text=prompt.prompt_text,
        target_head_labels=list(prompt.target_head_labels),
        generated_text=generated_text,
        success=True,
        runtime_seconds=runtime_seconds,
        keyword_hits=keyword_hits,
        keyword_total=keyword_total,
        keyword_recall=keyword_recall,
        required_fact_labels_hit=required_fact_labels_hit,
        missing_required_fact_labels=missing_required_fact_labels,
        central_fact_labels_hit=central_fact_labels_hit,
        missing_central_fact_labels=missing_central_fact_labels,
        central_detail_preserved=central_detail_preserved,
        omitted_central_detail=not central_detail_preserved,
        hallucination_flags=hallucination_flags,
        reference_missing_fact_labels=reference_missing_fact_labels,
        reference_extra_fact_labels=reference_extra_fact_labels,
        divergence_summary=_divergence_summary(
            reference_hits=reference_hits,
            candidate_hits=required_fact_labels_hit,
            hallucination_flags=hallucination_flags,
            reference_unigram_f1=reference_unigram,
        ),
        reference_unigram_f1=reference_unigram,
    )


def _build_path_result(
    path: str,
    keys_per_head: int,
    compacted_layers,
    prompt_results: list[BehavioralRunResult],
    runtime_seconds: float,
    prefix_token_count: int,
) -> BehavioralPathResult:
    compacted_heads: list[dict[str, object]] = []
    effective_compact_tokens = 0
    if compacted_layers:
        for layer in sorted(compacted_layers):
            for head in sorted(compacted_layers[layer]):
                runtime = compacted_layers[layer][head]
                effective_compact_tokens += len(runtime.selected_indices)
                compacted_heads.append(
                    {
                        "layer": layer,
                        "head": head,
                        "selected_indices": runtime.selected_indices,
                        "selected_key_count": len(runtime.selected_indices),
                    }
                )

    return BehavioralPathResult(
        path=path,
        keys_per_head=keys_per_head,
        compaction_succeeded=True,
        compacted_head_count=len(compacted_heads),
        compacted_prefix_tokens=prefix_token_count,
        effective_compact_tokens=effective_compact_tokens,
        runtime_seconds=round(runtime_seconds, 6),
        preserved_central_detail_count=sum(run.central_detail_preserved for run in prompt_results),
        omitted_central_detail_count=sum(run.omitted_central_detail for run in prompt_results),
        hallucination_run_count=sum(bool(run.hallucination_flags) for run in prompt_results),
        runs=prompt_results,
        compacted_heads=compacted_heads,
    )


def write_behavioral_result(result: BehavioralEvalResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return output_path


def _feed_tokens_with_cache(
    model,
    token_ids: list[int],
    device: str,
    *,
    past_key_values=None,
    start_position: int = 0,
):
    import torch

    if not token_ids:
        return past_key_values, None

    input_tensor = torch.tensor([token_ids], device=device, dtype=torch.long)
    cache_position = torch.arange(start_position, start_position + len(token_ids), device=device)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_tensor,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
        )
    return outputs.past_key_values, outputs.logits[:, -1, :]


def _continue_with_prompt(
    *,
    model,
    tokenizer,
    prefix_token_ids: list[int],
    tail_token_ids: list[int],
    prompt: BehavioralPrompt,
    device: str,
    compacted_layers,
    prefix_token_count: int,
    enable_thinking: bool | None,
    max_new_tokens: int,
) -> tuple[str, float]:
    import torch

    start_time = time.perf_counter()
    patch_context = (
        patched_compaction_attention(compacted_layers, prefix_token_count, model_type="qwen3_5")
        if compacted_layers is not None
        else nullcontext()
    )

    with patch_context:
        prefix_cache, _ = _feed_tokens_with_cache(model, prefix_token_ids, device=device)
        logical_position = prefix_token_count
        prefix_cache, _ = _feed_tokens_with_cache(
            model,
            tail_token_ids,
            device=device,
            past_key_values=prefix_cache,
            start_position=logical_position,
        )
        logical_position += len(tail_token_ids)

        prompt_token_ids = build_qwen35_prompt_ids(
            tokenizer,
            prompt_text=prompt.prompt_text,
            enable_thinking=enable_thinking,
        )
        prefix_cache, logits = _feed_tokens_with_cache(
            model,
            prompt_token_ids,
            device=device,
            past_key_values=prefix_cache,
            start_position=logical_position,
        )
        logical_position += len(prompt_token_ids)

        generated_token_ids: list[int] = []
        next_token = int(torch.argmax(logits, dim=-1).item())
        for _ in range(max_new_tokens):
            if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
                break
            generated_token_ids.append(next_token)
            token_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
            cache_position = torch.tensor([logical_position], device=device, dtype=torch.long)
            with torch.inference_mode():
                outputs = model(
                    input_ids=token_tensor,
                    past_key_values=prefix_cache,
                    use_cache=True,
                    return_dict=True,
                    cache_position=cache_position,
                )
            prefix_cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = int(torch.argmax(logits, dim=-1).item())
            logical_position += 1

    generated_text = _cleanup_generated_text(tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip())
    return generated_text, round(time.perf_counter() - start_time, 6)


def _run_prompt_path(
    *,
    model,
    tokenizer,
    prompts: list[BehavioralPrompt],
    prefix_token_ids: list[int],
    tail_token_ids: list[int],
    device: str,
    compacted_layers,
    prefix_token_count: int,
    enable_thinking: bool | None,
    max_new_tokens: int,
    reference_runs: list[BehavioralRunResult] | None = None,
) -> tuple[list[BehavioralRunResult], float]:
    runs: list[BehavioralRunResult] = []
    total_runtime = 0.0
    for index, prompt in enumerate(prompts):
        generated_text, runtime = _continue_with_prompt(
            model=model,
            tokenizer=tokenizer,
            prefix_token_ids=prefix_token_ids,
            tail_token_ids=tail_token_ids,
            prompt=prompt,
            device=device,
            compacted_layers=compacted_layers,
            prefix_token_count=prefix_token_count,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
        )
        total_runtime += runtime
        if reference_runs is None:
            run = evaluate_run(
                prompt=prompt,
                generated_text=generated_text,
                runtime_seconds=runtime,
            )
        else:
            reference_run = reference_runs[index]
            run = evaluate_run(
                prompt=prompt,
                generated_text=generated_text,
                runtime_seconds=runtime,
                reference_text=reference_run.generated_text,
                reference_hits=reference_run.required_fact_labels_hit,
            )
        runs.append(run)
    return runs, total_runtime


def run_behavioral_evaluation(
    sample,
    config,
    *,
    keys_per_head: int,
    prompt_set: str = DEFAULT_PROMPT_SET,
    key_selection_method: str = "highest_attention",
    prompt_limit: int | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    probe_coverage: str = "narrow",
) -> BehavioralEvalResult:
    prompts = build_prompt_set(prompt_set, sample.prompt_family)
    if prompt_limit is not None:
        prompts = prompts[: max(0, int(prompt_limit))]
        if not prompts:
            raise ValueError("prompt_limit produced an empty prompt set.")
    collection_config = replace(config, model=replace(config.model, attn_implementation="eager"))
    collection_model, collection_tokenizer, model_type = load_qwen35_bundle(collection_config)
    try:
        probe_layers = default_probe_layers_for_model(collection_model, model_type)
        if probe_coverage == "narrow":
            probe_heads = default_probe_heads_for_model(collection_model)
        elif probe_coverage == "all_heads":
            probe_heads = all_probe_heads_for_model(collection_model)
        else:
            raise ValueError(f"Unsupported probe_coverage {probe_coverage!r}.")
        target_layer_heads = tuple((int(layer), int(head)) for layer in probe_layers for head in probe_heads)

        bundle = collect_teacher_forced_boundary_collection(
            sample,
            collection_config,
            model=collection_model,
            tokenizer=collection_tokenizer,
            probe_layers=probe_layers,
            probe_heads=probe_heads,
        )
        state = build_state_from_observations(collection_config, bundle.harvest.observations)
        sketch_source = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, collection_config)
        control_query_source = extract_teacher_forced_subsample_control(
            bundle.query_bank,
            max_entries=len(sketch_source.selected_entries),
        )
        token_ids, _ = materialize_long_context_ids(sample, collection_tokenizer)
    finally:
        unload_qwen35_bundle(collection_model)

    prefix_token_ids = token_ids[: sample.boundary.prefix_token_count]
    tail_token_ids = token_ids[sample.boundary.prefix_token_count :]

    reference_model, reference_tokenizer, _ = load_qwen35_bundle(config)
    try:
        reference_runs, reference_total_runtime = _run_prompt_path(
            model=reference_model,
            tokenizer=reference_tokenizer,
            prompts=prompts,
            prefix_token_ids=prefix_token_ids,
            tail_token_ids=tail_token_ids,
            device=config.model.device,
            compacted_layers=None,
            prefix_token_count=sample.boundary.prefix_token_count,
            enable_thinking=config.model.enable_thinking,
            max_new_tokens=max_new_tokens,
            reference_runs=None,
        )
    finally:
        unload_qwen35_bundle(reference_model)

    sketch_selection, sketch_layers = build_path_runtime(
        sample.sample_id,
        sample.boundary.boundary_id,
        sketch_source.source,
        keys_per_head,
        bundle,
        sketch_source,
        target_layers=probe_layers,
        target_heads=probe_heads,
        target_layer_heads=target_layer_heads,
        compute_device=collection_config.model.device,
        key_selection_method=key_selection_method,
    )
    control_selection, control_layers = build_path_runtime(
        sample.sample_id,
        sample.boundary.boundary_id,
        control_query_source.source,
        keys_per_head,
        bundle,
        control_query_source,
        target_layers=probe_layers,
        target_heads=probe_heads,
        target_layer_heads=target_layer_heads,
        compute_device=collection_config.model.device,
        key_selection_method=key_selection_method,
    )

    continuation_model, continuation_tokenizer, _ = load_qwen35_bundle(collection_config)
    try:
        sketch_runs, sketch_total_runtime = _run_prompt_path(
            model=continuation_model,
            tokenizer=continuation_tokenizer,
            prompts=prompts,
            prefix_token_ids=prefix_token_ids,
            tail_token_ids=tail_token_ids,
            device=collection_config.model.device,
            compacted_layers=sketch_layers,
            prefix_token_count=sample.boundary.prefix_token_count,
            enable_thinking=collection_config.model.enable_thinking,
            max_new_tokens=max_new_tokens,
            reference_runs=reference_runs,
        )
        control_runs, control_total_runtime = _run_prompt_path(
            model=continuation_model,
            tokenizer=continuation_tokenizer,
            prompts=prompts,
            prefix_token_ids=prefix_token_ids,
            tail_token_ids=tail_token_ids,
            device=collection_config.model.device,
            compacted_layers=control_layers,
            prefix_token_count=sample.boundary.prefix_token_count,
            enable_thinking=collection_config.model.enable_thinking,
            max_new_tokens=max_new_tokens,
            reference_runs=reference_runs,
        )
    finally:
        unload_qwen35_bundle(continuation_model)

    return BehavioralEvalResult(
        sample_id=sample.sample_id,
        boundary_id=sample.boundary.boundary_id,
        prompt_set=prompt_set,
        keys_per_head=keys_per_head,
        key_selection_method=key_selection_method,
        train_fraction=TRAIN_FRACTION,
        beta_solver=BETA_SOLVER,
        beta_regularization_strength=BETA_REGULARIZATION,
        value_regularization_strength=VALUE_REGULARIZATION,
        prompt_labels=[prompt.label for prompt in prompts],
        reference=_build_path_result(
            path="full_cache_reference",
            keys_per_head=keys_per_head,
            compacted_layers=None,
            prompt_results=reference_runs,
            runtime_seconds=reference_total_runtime,
            prefix_token_count=sample.boundary.prefix_token_count,
        ),
        sketch=_build_path_result(
            path=sketch_selection.source,
            keys_per_head=keys_per_head,
            compacted_layers=sketch_layers,
            prompt_results=sketch_runs,
            runtime_seconds=sketch_total_runtime,
            prefix_token_count=sample.boundary.prefix_token_count,
        ),
        control=_build_path_result(
            path=control_selection.source,
            keys_per_head=keys_per_head,
            compacted_layers=control_layers,
            prompt_results=control_runs,
            runtime_seconds=control_total_runtime,
            prefix_token_count=sample.boundary.prefix_token_count,
        ),
    )
