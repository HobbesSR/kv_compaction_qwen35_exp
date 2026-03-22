from __future__ import annotations

from kv_compaction_qwen35_clean.behavioral_eval import build_prompt_set, evaluate_run


def test_build_prompt_set_for_qwen35_surface() -> None:
    prompts = build_prompt_set("qwen35_calibration_v0", "warehouse_migration_qwen35")

    assert [prompt.label for prompt in prompts] == [
        "qwen35_same_task_status_triplet",
        "qwen35_same_task_handoff_rollback",
        "qwen35_branch_switch_harness_note",
        "qwen35_branch_switch_appendix_details",
    ]


def test_evaluate_run_marks_required_facts_and_reference_overlap() -> None:
    prompt = build_prompt_set("qwen35_calibration_v0", "warehouse_migration_qwen35")[0]

    result = evaluate_run(
        prompt=prompt,
        generated_text="- saturday cutover\n- firmware validation pass\n- delayed harness certification",
        runtime_seconds=1.0,
        reference_text="- saturday cutover\n- firmware validation pass\n- delayed harness certification",
        reference_hits=["saturday_cutover", "firmware_validation_pass", "harness_certification"],
    )

    assert result.central_detail_preserved is True
    assert result.required_fact_labels_hit == [
        "firmware_validation_pass",
        "harness_certification",
        "saturday_cutover",
    ]
    assert result.reference_missing_fact_labels == []
    assert result.reference_unigram_f1 == 1.0
