from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from importlib import import_module
from importlib.util import find_spec
import json
from pathlib import Path
from typing import Any

from kv_compaction_qwen35_clean.data_types import LoadedContextSample, SmokeTestConfig


QWEN35_MODEL_CLASS_MAP: dict[str, tuple[str, str]] = {
    "qwen3_5": ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5ForCausalLM"),
}


@dataclass(frozen=True)
class RuntimeDependencyStatus:
    torch_available: bool
    transformers_available: bool
    accelerate_available: bool
    qwen3_5_available: bool

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Qwen35RuntimePlan:
    sample_id: str
    model_name: str
    huggingface_id: str
    tokenizer_name: str
    device: str
    dtype: str
    attn_implementation: str
    trust_remote_code: bool
    local_files_only: bool
    model_type: str
    transcript_preview: str
    dependency_status: RuntimeDependencyStatus

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


def detect_runtime_dependencies() -> RuntimeDependencyStatus:
    return RuntimeDependencyStatus(
        torch_available=find_spec("torch") is not None,
        transformers_available=find_spec("transformers") is not None,
        accelerate_available=find_spec("accelerate") is not None,
        qwen3_5_available=find_spec("transformers.models.qwen3_5") is not None,
    )


def _normalize_role(speaker: str) -> str:
    if speaker in {"system", "user", "assistant"}:
        return speaker
    return "tool"


def build_teacher_forced_transcript(sample: LoadedContextSample) -> str:
    blocks = []
    for turn in sample.turns:
        role = _normalize_role(turn.speaker).upper()
        blocks.append(f"{role} [{turn.turn_id}]\n{turn.content}")
    return "\n\n".join(blocks)


def _build_load_kwargs(config: SmokeTestConfig) -> dict[str, Any]:
    load_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": config.model.local_files_only,
        "attn_implementation": config.model.attn_implementation,
        "dtype": config.model.dtype,
    }
    if detect_runtime_dependencies().accelerate_available:
        load_kwargs["low_cpu_mem_usage"] = True
    return load_kwargs


def resolve_qwen35_model_type(
    huggingface_id: str,
    *,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> str:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        huggingface_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model_type = str(getattr(config, "model_type", "") or "")
    if not model_type:
        raise ValueError(f"Unable to determine model_type for {huggingface_id!r}.")
    if model_type not in QWEN35_MODEL_CLASS_MAP:
        raise ValueError(f"Unsupported model_type {model_type!r} for qwen35_clean.")
    return model_type


def _fallback_model_type_from_huggingface_id(huggingface_id: str) -> str:
    lowered = huggingface_id.lower()
    if "qwen3.5" in lowered:
        return "qwen3_5"
    raise ValueError(f"Unable to infer qwen3.5 model_type from {huggingface_id!r}.")


def build_runtime_plan(sample: LoadedContextSample, config: SmokeTestConfig) -> Qwen35RuntimePlan:
    try:
        model_type = resolve_qwen35_model_type(
            config.model.huggingface_id,
            trust_remote_code=config.model.trust_remote_code,
            local_files_only=config.model.local_files_only,
        )
    except Exception:
        model_type = _fallback_model_type_from_huggingface_id(config.model.huggingface_id)
    return Qwen35RuntimePlan(
        sample_id=sample.sample_id,
        model_name=config.model.name,
        huggingface_id=config.model.huggingface_id,
        tokenizer_name=config.model.tokenizer_name,
        device=config.model.device,
        dtype=config.model.dtype,
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
        model_type=model_type,
        transcript_preview=build_teacher_forced_transcript(sample)[:240],
        dependency_status=detect_runtime_dependencies(),
    )


def load_qwen35_bundle(config: SmokeTestConfig):
    import torch
    from transformers import AutoTokenizer

    load_kwargs = _build_load_kwargs(config)
    dtype = getattr(torch, str(load_kwargs.pop("dtype")))
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    model_type = resolve_qwen35_model_type(
        config.model.huggingface_id,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    module_name, class_name = QWEN35_MODEL_CLASS_MAP[model_type]
    model_cls = getattr(import_module(module_name), class_name)
    model = model_cls.from_pretrained(
        config.model.huggingface_id,
        dtype=dtype,
        **load_kwargs,
    )
    model.eval()
    model = model.to(config.model.device)
    return model, tokenizer, model_type


def unload_qwen35_bundle(model) -> None:
    try:
        import torch
    except ImportError:
        return
    if model is None:
        return
    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def write_runtime_plan(plan: Qwen35RuntimePlan, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan.to_serializable(), indent=2) + "\n", encoding="utf-8")
    return output_path
