from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

import numpy as np
import sacrebleu
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from etd.config import Config
from etd.datasets import load_splits, prepare_text
from etd.embedding import (
    estimate_embedding_storage,
    load_embedding_model,
    load_precomputed_embeddings,
    precompute_embeddings,
)
from etd.models import build_decoder
from etd.train import IndexedTextDataset, _build_collate_fn, _build_prompt


@dataclass
class EvalBatch:
    idx: list[int]
    text: list[str]
    embeddings: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def _collect_split(cfg: Config) -> Any:
    datasets = load_splits(cfg.dataset)
    split = cfg.evaluation.split

    if split == "validation":
        return prepare_text(datasets.validation, cfg.dataset)
    if split == "test":
        return prepare_text(datasets.test, cfg.dataset)

    raise ValueError(f"Unsupported evaluation split: {split}")


def _load_adapter(model: Any, checkpoint: str, device: torch.device) -> None:
    state = torch.load(checkpoint, map_location=device)
    model.adapter.load_state_dict(state)


def _build_eval_collate(
    cfg: Config,
    tokenizer: Any,
    embeddings: np.ndarray | None,
    embedding_model: Any,
) -> Any:
    base_collate = _build_collate_fn(cfg, tokenizer, embeddings, embedding_model)

    def _collate(batch: list[dict[str, Any]]) -> EvalBatch:
        indices = [item["idx"] for item in batch]
        texts = [item["text"] for item in batch]
        base = base_collate(batch)
        return EvalBatch(
            idx=indices,
            text=texts,
            embeddings=base.embeddings,
            input_ids=base.input_ids,
            attention_mask=base.attention_mask,
        )

    return _collate


def _decode(tokens: torch.Tensor, tokenizer: Any) -> list[str]:
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)


def evaluate_model(cfg: Config) -> dict[str, float]:
    eval_ds = _collect_split(cfg)
    if cfg.evaluation.max_samples:
        eval_ds = eval_ds.select(range(min(cfg.evaluation.max_samples, len(eval_ds))))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.decoder_model)
    tokenizer.pad_token = tokenizer.eos_token

    embedding_model = load_embedding_model(cfg.embeddings, cfg.hardware.device)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()

    embeddings = None
    embeddings_path = cfg.paths.embeddings_dir / f"{cfg.evaluation.split}.npy"
    if embeddings_path.exists():
        embeddings = load_precomputed_embeddings(embeddings_path)
        if len(embeddings) != len(eval_ds):
            print(
                "Precomputed embeddings size mismatch; rebuilding "
                f"({len(embeddings)} vs {len(eval_ds)})"
            )
            embeddings = precompute_embeddings(
                eval_ds,
                embedding_model,
                cfg.embeddings,
                embeddings_path,
            )
    else:
        estimate = estimate_embedding_storage(len(eval_ds), dims=embedding_dim)
        if estimate.total_gb <= cfg.embeddings.max_precompute_gb:
            embeddings = precompute_embeddings(
                eval_ds,
                embedding_model,
                cfg.embeddings,
                embeddings_path,
            )

    model = build_decoder(cfg.model, cfg.hardware.device)
    device = torch.device(cfg.hardware.device)
    _load_adapter(model, cfg.evaluation.adapter_checkpoint, device)
    model.adapter.eval()
    model.model.eval()

    dataset = IndexedTextDataset(eval_ds)
    collate_fn = _build_eval_collate(cfg, tokenizer, embeddings, embedding_model)
    dataloader = DataLoader(dataset, batch_size=cfg.evaluation.batch_size, collate_fn=collate_fn)

    references = []
    predictions = []
    prompts_log = []

    with torch.no_grad():
        for batch in dataloader:
            batch_embeddings = batch.embeddings.to(device)
            prefix = model.adapter(batch_embeddings)
            prompts = [_build_prompt(cfg.prompt, text) for text in batch.text]
            prompts_log.extend(prompts)
            prompt_tokens = tokenizer(
                prompts,
                truncation=True,
                max_length=cfg.dataset.max_tokens,
                padding=True,
                return_tensors="pt",
            )
            prompt_ids = prompt_tokens["input_ids"].to(device)
            prompt_embeds = model.model.get_input_embeddings()(prompt_ids)
            prefix_embeds = torch.cat([prefix, prompt_embeds], dim=1)

            attention = torch.cat(
                [
                    torch.ones((prompt_ids.shape[0], prefix.shape[1]), device=device),
                    prompt_tokens["attention_mask"].to(device),
                ],
                dim=1,
            )

            generated = model.model.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=attention,
                max_new_tokens=cfg.evaluation.max_new_tokens,
                do_sample=True,
                temperature=cfg.evaluation.temperature,
                top_p=cfg.evaluation.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

            decoded = _decode(generated, tokenizer)
            predictions.extend(decoded)
            references.extend([[text] for text in batch.text])

    bleu = sacrebleu.corpus_bleu(predictions, list(zip(*references)))
    token_f1 = _token_f1(predictions, [ref[0] for ref in references])

    metrics = {
        "bleu": float(bleu.score),
        "token_f1": float(token_f1),
        "samples": len(predictions),
    }

    report_path = cfg.paths.outputs_dir / "eval-report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    results_path = cfg.paths.outputs_dir / "eval-results.parquet"
    _write_results(results_path, prompts_log, predictions, [ref[0] for ref in references])

    return metrics


def _write_results(
    path: Any, prompts: list[str], predictions: list[str], references: list[str]
) -> None:
    import polars as pl

    table = pl.DataFrame(
        {
            "prompt": prompts,
            "prediction": predictions,
            "reference": references,
        }
    )
    table.write_parquet(path)


def _token_f1(predictions: list[str], references: list[str]) -> float:
    scores = []
    for pred, ref in zip(predictions, references, strict=False):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        precision = len(pred_set & ref_set) / len(pred_set)
        recall = len(pred_set & ref_set) / len(ref_set)
        if precision + recall == 0:
            scores.append(0.0)
            continue
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))
