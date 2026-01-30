from __future__ import annotations

from dataclasses import dataclass
import json
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


@dataclass
class EmbeddingProgress:
    next_index: int
    total: int
    embed_dim: int


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
    return model.encode(
        texts,
        batch_size=cfg.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )


def _progress_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".progress.json")


def embeddings_incomplete(path: Path) -> bool:
    return _progress_path(path).exists()


def _load_progress(path: Path) -> EmbeddingProgress | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return EmbeddingProgress(
        next_index=int(payload["next_index"]),
        total=int(payload["total"]),
        embed_dim=int(payload["embed_dim"]),
    )


def _write_progress(path: Path, progress: EmbeddingProgress) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(
            {
                "next_index": progress.next_index,
                "total": progress.total,
                "embed_dim": progress.embed_dim,
            }
        )
    )
    tmp_path.replace(path)


def precompute_embeddings(
    dataset: Any, model: Any, cfg: EmbeddingConfig, path: Path, resume: bool = True
) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    embed_dim = model.get_sentence_embedding_dimension()
    total = len(dataset)
    progress_path = _progress_path(path)

    memmap: np.ndarray
    start_idx = 0
    if path.exists() and resume:
        try:
            memmap = np.lib.format.open_memmap(path, mode="r+")
            if memmap.dtype != np.float32 or memmap.shape != (total, embed_dim):
                raise ValueError("Existing embeddings file shape/dtype mismatch")
            progress = _load_progress(progress_path)
            if progress and progress.total == total and progress.embed_dim == embed_dim:
                start_idx = max(0, min(progress.next_index, total))
        except Exception:
            memmap = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(total, embed_dim)
            )
    else:
        memmap = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(total, embed_dim)
        )

    if start_idx >= total:
        memmap.flush()
        progress_path.unlink(missing_ok=True)
        return memmap

    if not progress_path.exists():
        _write_progress(
            progress_path,
            EmbeddingProgress(next_index=start_idx, total=total, embed_dim=embed_dim),
        )

    start_batch = start_idx // cfg.batch_size
    total_batches = (total + cfg.batch_size - 1) // cfg.batch_size
    for start in tqdm(
        range(start_idx, total, cfg.batch_size),
        desc="Precomputing embeddings",
        unit="batch",
        total=total_batches,
        initial=start_batch,
    ):
        batch = dataset[start : start + cfg.batch_size]
        texts = batch["text"]
        embeddings = compute_embeddings(texts, model, cfg)
        memmap[start : start + len(embeddings)] = embeddings
        memmap.flush()
        _write_progress(
            progress_path,
            EmbeddingProgress(
                next_index=start + len(embeddings), total=total, embed_dim=embed_dim
            ),
        )

    memmap.flush()
    progress_path.unlink(missing_ok=True)
    return memmap


def load_precomputed_embeddings(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")
