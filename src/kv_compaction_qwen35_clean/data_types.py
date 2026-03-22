from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    paper: str


@dataclass
class ModelConfig:
    name: str
    provider: str
    huggingface_id: str
    tokenizer_name: str
    device: str
    dtype: str
    attn_implementation: str
    trust_remote_code: bool
    local_files_only: bool
    max_context_tokens: int
    prefill_chunk_size: int
    probe_max_tokens: int
    logical_kv_length: str
    enable_thinking: bool | None = None


@dataclass
class ChunkingConfig:
    enabled: bool
    mode: str
    chunk_size: int


@dataclass
class BranchSwitchConfig:
    primary_prompt_label: str
    primary_prompt_template: str
    alternate_prompt_label: str
    alternate_prompt_template: str
    description: str


@dataclass
class DataConfig:
    dataset: str
    context_tokens: int
    branch_switch_probe: bool
    chunking: ChunkingConfig
    branch_switch: BranchSwitchConfig


@dataclass
class CompactionConfig:
    boundary: str
    strategy: str
    target_compression_ratio: float
    preserved_tail_tokens: int
    fit_beta: bool
    fit_values: bool
    key_selection: str
    head_budget: str


@dataclass
class SketchConfig:
    kind: str
    max_prototypes: int
    update_rule: str
    similarity_metric: str
    merge_threshold: float
    forgetting_factor: float
    min_prefix_mass: float
    novelty_weight: float
    residual_weight: float


@dataclass
class FeatureSchemaConfig:
    granularity: str
    projection_source: str
    query_projection_dim: int
    output_projection_dim: int
    mass_measure: str
    auxiliary_mass_metric: str
    output_summary: str
    tap_point: str


@dataclass
class ReferenceQueryConfig:
    primary_source: str
    compare_sources: list[str]
    max_queries_per_head: int


@dataclass
class SmokeTestConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    compaction: CompactionConfig
    feature_schema: FeatureSchemaConfig
    sketch: SketchConfig
    reference_queries: ReferenceQueryConfig
    baselines: list[str]
    metrics: list[str]


@dataclass
class ContextTurn:
    turn_id: str
    speaker: str
    token_count: int
    content: str


@dataclass
class ContextChunk:
    chunk_id: str
    start_token: int
    end_token: int
    turn_ids: list[str]


@dataclass
class PromptBoundary:
    boundary_id: str
    boundary_type: str
    prefix_token_count: int
    preserved_tail_tokens: int
    logical_context_tokens: int
    physical_context_tokens: int
    target_context_tokens_after_compaction: int
    compaction_chunk_ids: list[str]
    primary_prompt_label: str
    primary_prompt_text: str
    alternate_prompt_label: str
    alternate_prompt_text: str


@dataclass
class LoadedContextSample:
    sample_id: str
    dataset: str
    source: str
    task_label: str
    turns: list[ContextTurn]
    chunks: list[ContextChunk]
    logical_context_tokens: int
    physical_context_tokens: int
    boundary: PromptBoundary
    prompt_family: str = "warehouse_migration_qwen35"

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class FeatureObservation:
    token_index: int
    layer: int
    head: int
    tap_point: str
    query_projection: list[float]
    prefix_mass_share: float
    raw_prefix_mass: float
    output_projection: list[float]
    source_turn_id: str = ""
    source_speaker: str = ""


@dataclass
class FeatureHarvest:
    sample_id: str
    boundary_id: str
    logical_context_tokens: int
    physical_context_tokens: int
    feature_granularity: str
    tap_point: str
    query_projection_dim: int
    output_projection_dim: int
    observed_layers: list[int]
    observed_heads: list[int]
    observation_count: int
    observations: list[FeatureObservation]

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class QuerySample:
    query_id: str
    layer: int
    head: int
    token_index: int
    prefix_mass_share: float
    raw_prefix_mass: float
    query_projection: list[float]
    raw_query_vector: list[float]
    source_turn_id: str
    source_speaker: str


@dataclass
class QuerySampleBank:
    sample_id: str
    boundary_id: str
    query_dim: int
    sample_count: int
    samples: list[QuerySample]

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BoundaryCollection:
    harvest: FeatureHarvest
    query_bank: QuerySampleBank
    boundary_keys: dict[tuple[int, int], list[list[float]]]
    boundary_values: dict[tuple[int, int], list[list[float]]]
    boundary_projected_values: dict[tuple[int, int], list[list[float]]]
    output_targets: dict[tuple[int, int, int], list[float]]
    runtime_cache: object | None = None
    capture_token_indices: list[int] | None = None
    monitored_observation_count: int | None = None
    monitored_query_sample_count: int | None = None
