from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from typing import Any

from tqdm import tqdm

from etd.config import EmbeddingConfig


@dataclass
class EmbeddingEstimate:
    num_rows: int
    dims: int
    bytes_per_row: int
    total_bytes: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)


def estimate_embedding_storage(
    num_rows: int, dims: int = 768, dtype_bytes: int = 4
) -> EmbeddingEstimate:
    bytes_per_row = dims * dtype_bytes
    total_bytes = num_rows * bytes_per_row
    return EmbeddingEstimate(
        num_rows=num_rows,
        dims=dims,
        bytes_per_row=bytes_per_row,
        total_bytes=total_bytes,
    )


def load_embedding_model(cfg: EmbeddingConfig, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(cfg.model, device=device)


def compute_embeddings(texts: list[str], model: Any, cfg: EmbeddingConfig) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=cfg.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    if cfg.pooling == "max":
        return embeddings
    raise ValueError(f"Unsupported pooling {cfg.pooling}")


def precompute_embeddings(dataset: Any, model: Any, cfg: EmbeddingConfig, path: Path) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    embed_dim = model.get_sentence_embedding_dimension()
    memmap = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(dataset), embed_dim)
    )

    total = len(dataset)
    for start in tqdm(
        range(0, total, cfg.batch_size),
        desc="Precomputing embeddings",
        unit="batch",
        total=(total + cfg.batch_size - 1) // cfg.batch_size,
    ):
        batch = dataset[start : start + cfg.batch_size]
        texts = batch["text"]
        embeddings = compute_embeddings(texts, model, cfg)
        memmap[start : start + len(embeddings)] = embeddings

    memmap.flush()
    return memmap


def load_precomputed_embeddings(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")
