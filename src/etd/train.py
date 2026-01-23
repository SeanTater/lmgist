from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from etd.config import Config
from etd.datasets import load_splits, prepare_text
from etd.embedding import (
    compute_embeddings,
    estimate_embedding_storage,
    load_embedding_model,
    load_precomputed_embeddings,
    precompute_embeddings,
)
from etd.models import build_decoder
from etd.utils import ensure_dir, set_seed


@dataclass
class Batch:
    embeddings: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prefix_lengths: torch.Tensor


class IndexedTextDataset(Dataset):
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.dataset[int(idx)]
        return {"idx": int(idx), "text": record["text"]}


def _estimate_precompute(cfg: Config, dataset: Any, embedding_dim: int) -> bool:
    num_rows = len(dataset)
    estimate = estimate_embedding_storage(num_rows, dims=embedding_dim)
    precompute = str(cfg.embeddings.precompute).lower()

    if precompute == "auto":
        if estimate.total_gb > cfg.embeddings.max_precompute_gb:
            print(
                "Precompute disabled: estimated "
                f"{estimate.total_gb:.2f} GB > {cfg.embeddings.max_precompute_gb} GB"
            )
            return False
        print(f"Precompute enabled: estimated {estimate.total_gb:.2f} GB")
        return True
    if precompute == "true":
        print(f"Precompute forced on: estimated {estimate.total_gb:.2f} GB")
        return True
    if precompute == "false":
        print(f"Precompute forced off: estimated {estimate.total_gb:.2f} GB")
        return False

    raise ValueError(f"Unsupported precompute setting: {cfg.embeddings.precompute}")


def _build_prompt(prompt_cfg: Any, text: str, include_text: bool = True) -> str:
    base = f"{prompt_cfg.system}\n\n{prompt_cfg.user_prefix}\n"
    return f"{base}{text}" if include_text else base


def _build_collate_fn(
    cfg: Config,
    tokenizer: Any,
    embeddings: np.ndarray | None,
    embedding_model: Any,
) -> Callable[[list[dict[str, Any]]], Batch]:
    def _collate(batch: list[dict[str, Any]]) -> Batch:
        indices = [item["idx"] for item in batch]
        texts = [item["text"] for item in batch]
        prompts = [_build_prompt(cfg.prompt, text) for text in texts]
        tokens = tokenizer(
            prompts,
            truncation=True,
            max_length=cfg.dataset.max_tokens,
            padding=True,
            return_tensors="pt",
        )
        prompt_only = [_build_prompt(cfg.prompt, "", include_text=False) for _ in prompts]
        prompt_lengths = torch.tensor(
            [len(tokenizer(text).input_ids) for text in prompt_only],
            dtype=torch.long,
        )

        if embeddings is None:
            batch_embeddings = compute_embeddings(texts, embedding_model, cfg.embeddings)
        else:
            batch_embeddings = np.asarray(embeddings[indices])

        return Batch(
            embeddings=torch.tensor(batch_embeddings, dtype=torch.float32),
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            prefix_lengths=prompt_lengths,
        )

    return _collate


def train(cfg: Config) -> None:
    set_seed(cfg.project.seed)
    ensure_dir(cfg.paths.outputs_dir)
    ensure_dir(cfg.paths.embeddings_dir)

    datasets = load_splits(cfg.dataset)
    train_ds = prepare_text(datasets.train, cfg.dataset)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.decoder_model)
    tokenizer.pad_token = tokenizer.eos_token

    embedding_model = load_embedding_model(cfg.embeddings, cfg.hardware.device)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    precompute = _estimate_precompute(cfg, train_ds, embedding_dim)

    embeddings = None
    if precompute:
        if cfg.embeddings.storage_format != "npy":
            raise ValueError("Only npy storage is supported for precompute")
        embeddings_path = cfg.paths.embeddings_dir / "train.npy"
        if embeddings_path.exists():
            embeddings = load_precomputed_embeddings(embeddings_path)
            if len(embeddings) != len(train_ds):
                print(
                    "Precomputed embeddings size mismatch; rebuilding "
                    f"({len(embeddings)} vs {len(train_ds)})"
                )
                embeddings = precompute_embeddings(
                    train_ds,
                    embedding_model,
                    cfg.embeddings,
                    embeddings_path,
                )
        else:
            embeddings = precompute_embeddings(
                train_ds,
                embedding_model,
                cfg.embeddings,
                embeddings_path,
            )

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    model = build_decoder(cfg.model, cfg.hardware.device)
    model.model = torch.compile(model.model)
    model.adapter = torch.compile(model.adapter)
    model.model.requires_grad_(False)
    model.model.eval()
    model.adapter.train()

    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=cfg.training.learning_rate)
    device = torch.device(cfg.hardware.device)
    mean_loss = 0.0

    dataset = IndexedTextDataset(train_ds)
    generator = torch.Generator().manual_seed(cfg.dataset.shuffle_seed)
    collate_fn = _build_collate_fn(cfg, tokenizer, embeddings, embedding_model)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
    )

    optimizer.zero_grad(set_to_none=True)
    global_step = 0

    total_steps = cfg.training.epochs * len(dataloader)
    progress = tqdm(total=total_steps, desc="Training", unit="step")

    for _ in range(cfg.training.epochs):
        for batch in dataloader:
            global_step += 1
            batch_embeddings = batch.embeddings.to(device)
            batch_input_ids = batch.input_ids.to(device)
            batch_attention = batch.attention_mask.to(device)

            prefix = model.adapter(batch_embeddings)
            inputs = model.model.get_input_embeddings()(batch_input_ids)
            prefix_inputs = torch.cat([prefix, inputs], dim=1)

            attention = torch.cat(
                [
                    torch.ones((batch_input_ids.shape[0], prefix.shape[1]), device=device),
                    batch_attention,
                ],
                dim=1,
            )

            prefix_labels = torch.full(
                (batch_input_ids.shape[0], prefix.shape[1]),
                -100,
                dtype=torch.long,
                device=device,
            )
            labels = torch.cat([prefix_labels, batch_input_ids], dim=1)

            prompt_lengths = batch.prefix_lengths.to(device)
            prompt_mask = torch.zeros_like(labels, dtype=torch.bool)
            for i, length in enumerate(prompt_lengths):
                prompt_mask[i, prefix.shape[1] : prefix.shape[1] + length] = True
            labels = labels.masked_fill(prompt_mask, -100)
            labels = labels.masked_fill(attention == 0, -100)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model.model(
                    inputs_embeds=prefix_inputs,
                    attention_mask=attention,
                    labels=labels,
                )
                loss = outputs.loss / cfg.training.grad_accum_steps
            loss.backward()
            mean_loss += loss.item() * cfg.training.grad_accum_steps

            if global_step % cfg.training.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.adapter.parameters(), cfg.training.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % cfg.training.log_every_steps == 0:
                avg_loss = mean_loss / cfg.training.log_every_steps
                mean_loss = 0.0
                print(f"step={global_step} avg_loss={avg_loss:.4f}")

            if global_step % cfg.training.save_every_steps == 0:
                save_path = cfg.paths.outputs_dir / f"adapter-step{global_step}.pt"
                torch.save(model.adapter.state_dict(), save_path)

            progress.update(1)

    progress.close()

    final_path = cfg.paths.outputs_dir / "adapter-final.pt"
    torch.save(model.adapter.state_dict(), final_path)
