from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

import numpy as np
import sacrebleu
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from etd.config import Config
from etd.datasets import load_splits, prepare_text
from etd.embedding import (
    embeddings_incomplete,
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
    if isinstance(state, dict) and "adapter_state" in state:
        state = state["adapter_state"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
    model.adapter.load_state_dict(state)


def _build_eval_collate(
    cfg: Config,
    tokenizer: Any,
    embeddings: np.ndarray | None,
    embedding_model: Any,
) -> Any:
    base_collate = _build_collate_fn(
        cfg,
        tokenizer,
        embeddings,
        embedding_model,
        include_text=False,
    )

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


def _bucket_indices(lengths: list[int], buckets: int) -> list[list[int]]:
    if not lengths or buckets <= 0:
        return []
    count = len(lengths)
    buckets = min(buckets, count)
    order = sorted(range(count), key=lambda i: lengths[i])
    indices = []
    for i in range(buckets):
        start = i * count // buckets
        end = (i + 1) * count // buckets
        indices.append(order[start:end])
    return indices


def _length_bucket_metrics(
    predictions: list[str],
    references: list[str],
    lengths: list[int],
    buckets: int = 4,
) -> list[dict[str, Any]]:
    bucket_indices = _bucket_indices(lengths, buckets)
    metrics = []
    for bucket in bucket_indices:
        if not bucket:
            metrics.append(
                {
                    "min_tokens": None,
                    "max_tokens": None,
                    "avg_ref_tokens": 0.0,
                    "bleu": None,
                    "token_f1": None,
                    "samples": 0,
                }
            )
            continue
        bucket_refs = [references[i] for i in bucket]
        bucket_preds = [predictions[i] for i in bucket]
        bucket_lengths = [lengths[i] for i in bucket]
        bleu = sacrebleu.corpus_bleu(bucket_preds, [bucket_refs])
        token_f1 = _token_f1(bucket_preds, bucket_refs)
        metrics.append(
            {
                "min_tokens": min(bucket_lengths),
                "max_tokens": max(bucket_lengths),
                "avg_ref_tokens": float(np.mean(bucket_lengths)),
                "bleu": float(bleu.score),
                "token_f1": float(token_f1),
                "samples": len(bucket),
            }
        )
    return metrics


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
        if embeddings_incomplete(embeddings_path):
            print("Resuming incomplete embeddings precompute.")
            embeddings = precompute_embeddings(
                eval_ds,
                embedding_model,
                cfg.embeddings,
                embeddings_path,
                resume=True,
            )
        else:
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
                    resume=False,
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

    model = build_decoder(cfg.model, cfg.lora, cfg.hardware.device, embedding_dim)
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
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            batch_embeddings = batch.embeddings.to(device)
            prefix = model.adapter(batch_embeddings)
            prompts = [
                _build_prompt(
                    tokenizer,
                    cfg.prompt,
                    text,
                    include_assistant_text=False,
                    add_generation_prompt=True,
                )
                for text in batch.text
            ]
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
                do_sample=cfg.evaluation.do_sample,
                temperature=cfg.evaluation.temperature,
                top_p=cfg.evaluation.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

            decoded = _decode(generated, tokenizer)
            predictions.extend(decoded)
            references.extend([[text] for text in batch.text])

    ref_texts = [ref[0] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, list(zip(*references)))
    token_f1 = _token_f1(predictions, ref_texts)
    ref_token_lengths = [
        len(tokenizer(text, add_special_tokens=False).input_ids) for text in ref_texts
    ]
    avg_ref_tokens = float(np.mean(ref_token_lengths)) if ref_token_lengths else 0.0
    compression_ratio = avg_ref_tokens / cfg.model.prefix_tokens if cfg.model.prefix_tokens else 0.0

    def _bleu_metric(preds: list[str], refs: list[str]) -> float:
        return float(sacrebleu.corpus_bleu(preds, [refs]).score)

    bleu_ci_lower, bleu_ci_upper = _bootstrap_ci(predictions, ref_texts, _bleu_metric)
    f1_ci_lower, f1_ci_upper = _bootstrap_ci(predictions, ref_texts, _token_f1)

    metrics = {
        "bleu": float(bleu.score),
        "bleu_ci_lower": bleu_ci_lower,
        "bleu_ci_upper": bleu_ci_upper,
        "token_f1": float(token_f1),
        "token_f1_ci_lower": f1_ci_lower,
        "token_f1_ci_upper": f1_ci_upper,
        "avg_ref_tokens": avg_ref_tokens,
        "token_compression_ratio": compression_ratio,
        "samples": len(predictions),
        "length_buckets": _length_bucket_metrics(
            predictions,
            ref_texts,
            ref_token_lengths,
        ),
    }

    report_path = cfg.paths.outputs_dir / "eval-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    results_path = cfg.paths.outputs_dir / "eval-results.parquet"
    _write_results(results_path, prompts_log, predictions, ref_texts)

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


def _bootstrap_ci(
    predictions: list[str],
    references: list[str],
    metric_fn: Any,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    scores = []
    n = len(predictions)
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        resampled_preds = [predictions[i] for i in indices]
        resampled_refs = [references[i] for i in indices]
        scores.append(metric_fn(resampled_preds, resampled_refs))
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(lower), float(upper)
