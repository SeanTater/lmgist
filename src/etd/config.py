from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str
    run_name: str
    seed: int


@dataclass
class PathsConfig:
    data_dir: Path
    embeddings_dir: Path
    outputs_dir: Path


@dataclass
class HardwareConfig:
    device: str


@dataclass
class DatasetConfig:
    name: str
    config: str | None
    split: str
    validation_split: str
    test_split: str
    text_field: str
    id_field: str | None
    max_tokens: int
    train_limit: int | None
    validation_limit: int | None
    test_limit: int | None
    shuffle_seed: int


@dataclass
class EmbeddingConfig:
    model: str
    pooling: str
    batch_size: int
    precompute: str
    storage_format: str
    max_precompute_gb: int


@dataclass
class ModelConfig:
    decoder_model: str
    prefix_tokens: int
    adapter_hidden_dim: int
    adapter_layers: int
    dropout: float


@dataclass
class LoraConfig:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


@dataclass
class PromptConfig:
    system: str
    user_prefix: str


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    log_every_steps: int
    save_every_steps: int
    eval_every_steps: int
    adapter_checkpoint: str | None


@dataclass
class EvaluationConfig:
    split: str
    max_samples: int
    batch_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    adapter_checkpoint: str


@dataclass
class Config:
    project: ProjectConfig
    paths: PathsConfig
    hardware: HardwareConfig
    dataset: DatasetConfig
    embeddings: EmbeddingConfig
    model: ModelConfig
    lora: LoraConfig
    prompt: PromptConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def _to_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def load_config(path: Path) -> Config:
    payload = yaml.safe_load(path.read_text())
    embeddings = payload["embeddings"]
    if "max_precompute_gb" not in embeddings:
        embeddings["max_precompute_gb"] = 250

    training = payload["training"]
    if "adapter_checkpoint" not in training:
        training["adapter_checkpoint"] = None

    return Config(
        project=ProjectConfig(**payload["project"]),
        paths=PathsConfig(
            **{
                **payload["paths"],
                "data_dir": _to_path(payload["paths"]["data_dir"]),
                "embeddings_dir": _to_path(payload["paths"]["embeddings_dir"]),
                "outputs_dir": _to_path(payload["paths"]["outputs_dir"]),
            }
        ),
        hardware=HardwareConfig(device=payload["hardware"]["device"]),
        dataset=DatasetConfig(**payload["dataset"]),
        embeddings=EmbeddingConfig(**embeddings),
        model=ModelConfig(**payload["model"]),
        lora=LoraConfig(**payload["lora"]),
        prompt=PromptConfig(**payload["prompt"]),
        training=TrainingConfig(**training),
        evaluation=EvaluationConfig(**payload["evaluation"]),
    )


def ensure_dirs(cfg: Config) -> None:
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.embeddings_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
