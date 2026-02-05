from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable
import json
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from etd.config import Config
from etd.datasets import load_splits, prepare_text
from etd.embedding import (
    compute_embeddings,
    embeddings_incomplete,
    estimate_embedding_storage,
    load_embedding_model,
    load_precomputed_embeddings,
    precompute_embeddings,
)
from etd.models import build_decoder
from etd.utils import ensure_dir, find_latest_checkpoint, parse_checkpoint_step, set_seed


@dataclass
class SelectionConfig:
    min_docs: int = 2
    max_docs: int = 8
    train_samples: int = 100000
    eval_samples: int = 2000
    content_query_rate: float = 0.5
    refusal_rate: float = 0.2
    refusal_oob_rate: float = 0.5
    min_phrase_words: int = 3
    max_phrase_words: int = 6
    refusal_text: str = "Sorry, I couldn't find a matching document in this set."


@dataclass
class SelectionExample:
    doc_indices: list[int]
    query: str
    target_text: str
    doc_count: int
    is_refusal: bool


def load_selection_config(path: Path) -> SelectionConfig:
    payload = yaml.safe_load(path.read_text())
    selection = payload.get("selection", {}) if isinstance(payload, dict) else {}
    defaults = SelectionConfig()
    values = {field.name: getattr(defaults, field.name) for field in fields(SelectionConfig)}
    values.update(selection)
    return SelectionConfig(**values)


def _collect_split(cfg: Config, split: str) -> Any:
    datasets = load_splits(cfg.dataset)
    if split == cfg.dataset.split:
        return prepare_text(datasets.train, cfg.dataset)
    if split == cfg.dataset.validation_split:
        return prepare_text(datasets.validation, cfg.dataset)
    if split == cfg.dataset.test_split:
        return prepare_text(datasets.test, cfg.dataset)
    raise ValueError(f"Unsupported selection split: {split}")


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


def _sample_phrase(text: str, rng: random.Random, min_words: int, max_words: int) -> str:
    words = text.split()
    if not words:
        return ""
    if len(words) <= min_words:
        return " ".join(words)
    max_len = min(max_words, len(words))
    length = rng.randint(min_words, max_len)
    start = rng.randint(0, len(words) - length)
    return " ".join(words[start : start + length])


def _phrase_in_docs(phrase: str, docs: list[str], exclude_index: int | None = None) -> bool:
    if not phrase:
        return True
    for i, text in enumerate(docs):
        if exclude_index is not None and i == exclude_index:
            continue
        if phrase in text:
            return True
    return False


def _index_query(doc_count: int, index: int, refusal_text: str) -> str:
    return (
        f"There are {doc_count} documents. Return document #{index} exactly. "
        "If no document matches, reply with: "
        f'"{refusal_text}"'
    )


def _content_query(doc_count: int, phrase: str, refusal_text: str) -> str:
    return (
        f"There are {doc_count} documents. Return the document that contains the phrase "
        f'"{phrase}" exactly. If no document matches, reply with: "{refusal_text}"'
    )


def _select_decoy_index(total: int, chosen: set[int], rng: random.Random) -> int:
    while True:
        candidate = rng.randrange(total)
        if candidate not in chosen:
            return candidate


def _build_prompt(
    tokenizer: Any,
    prompt_cfg: Any,
    query: str,
    assistant_text: str | None,
    add_generation_prompt: bool,
) -> str:
    user_content = f"{prompt_cfg.user_prefix}\n{query}" if prompt_cfg.user_prefix else query
    messages = [
        {"role": "system", "content": prompt_cfg.system},
        {"role": "user", "content": user_content},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})

    if hasattr(tokenizer, "apply_chat_template"):
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "Tokenizer has no chat template. Use an instruct/chat model or set "
                "a tokenizer.chat_template."
            )
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""
    parts = [f"{bos}{m['role']}\n{m['content']}{eos}\n" for m in messages]
    if add_generation_prompt:
        parts.append(f"{bos}assistant\n")
    return "".join(parts)


class SelectionDataset(Dataset):
    def __init__(
        self,
        dataset: Any,
        cfg: SelectionConfig,
        samples: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.cfg = cfg
        self.samples = samples
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> SelectionExample:
        rng = random.Random(self.seed + idx)
        total = len(self.dataset)
        if total == 0:
            raise ValueError("Selection dataset is empty")
        doc_count = rng.randint(self.cfg.min_docs, self.cfg.max_docs)
        doc_count = max(1, min(doc_count, total))
        doc_indices = rng.sample(range(total), doc_count)
        doc_texts = [self.dataset[int(i)]["text"] for i in doc_indices]

        is_refusal = rng.random() < self.cfg.refusal_rate
        content_query = rng.random() < self.cfg.content_query_rate

        if not is_refusal:
            target_idx = rng.randrange(doc_count)
            target_text = doc_texts[target_idx]
            if content_query:
                phrase = _sample_phrase(
                    target_text,
                    rng,
                    self.cfg.min_phrase_words,
                    self.cfg.max_phrase_words,
                )
                if phrase and not _phrase_in_docs(phrase, doc_texts, exclude_index=target_idx):
                    query = _content_query(doc_count, phrase, self.cfg.refusal_text)
                else:
                    query = _index_query(doc_count, target_idx + 1, self.cfg.refusal_text)
            else:
                query = _index_query(doc_count, target_idx + 1, self.cfg.refusal_text)
            return SelectionExample(
                doc_indices=doc_indices,
                query=query,
                target_text=target_text,
                doc_count=doc_count,
                is_refusal=False,
            )

        if rng.random() < self.cfg.refusal_oob_rate or total <= doc_count:
            out_of_bounds = doc_count + rng.randint(1, max(2, doc_count // 2))
            query = _index_query(doc_count, out_of_bounds, self.cfg.refusal_text)
        else:
            decoy_idx = _select_decoy_index(total, set(doc_indices), rng)
            decoy_text = self.dataset[int(decoy_idx)]["text"]
            phrase = _sample_phrase(
                decoy_text,
                rng,
                self.cfg.min_phrase_words,
                self.cfg.max_phrase_words,
            )
            if phrase and not _phrase_in_docs(phrase, doc_texts, exclude_index=None):
                query = _content_query(doc_count, phrase, self.cfg.refusal_text)
            else:
                out_of_bounds = doc_count + rng.randint(1, max(2, doc_count // 2))
                query = _index_query(doc_count, out_of_bounds, self.cfg.refusal_text)

        return SelectionExample(
            doc_indices=doc_indices,
            query=query,
            target_text=self.cfg.refusal_text,
            doc_count=doc_count,
            is_refusal=True,
        )


def _build_collate_fn(
    cfg: Config,
    selection_cfg: SelectionConfig,
    tokenizer: Any,
    dataset: Any,
    embeddings: np.ndarray | None,
    embedding_model: Any,
    include_assistant_text: bool,
    add_generation_prompt: bool,
) -> Callable[[list[SelectionExample]], dict[str, Any]]:
    max_docs = selection_cfg.max_docs
    embedding_dim = embedding_model.get_sentence_embedding_dimension()

    def _collate(batch: list[SelectionExample]) -> dict[str, Any]:
        prompts = [
            _build_prompt(
                tokenizer,
                cfg.prompt,
                item.query,
                item.target_text if include_assistant_text else None,
                add_generation_prompt=add_generation_prompt,
            )
            for item in batch
        ]
        tokens = tokenizer(
            prompts,
            truncation=True,
            max_length=cfg.dataset.max_tokens,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_only = [
            _build_prompt(
                tokenizer,
                cfg.prompt,
                item.query,
                None,
                add_generation_prompt=False,
            )
            for item in batch
        ]
        prompt_only_tokens = tokenizer(
            prompt_only,
            truncation=True,
            max_length=cfg.dataset.max_tokens,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_lengths = prompt_only_tokens["attention_mask"].sum(dim=1).to(torch.long)

        doc_mask = torch.zeros((len(batch), max_docs), dtype=torch.bool)
        doc_embeddings = np.zeros((len(batch), max_docs, embedding_dim), dtype=np.float32)
        doc_counts = torch.zeros(len(batch), dtype=torch.long)

        for i, item in enumerate(batch):
            doc_count = item.doc_count
            doc_counts[i] = doc_count
            doc_mask[i, :doc_count] = True
            if embeddings is None:
                texts = [dataset[int(idx)]["text"] for idx in item.doc_indices]
                batch_embeddings = compute_embeddings(texts, embedding_model, cfg.embeddings)
            else:
                batch_embeddings = np.asarray(embeddings[item.doc_indices])
            doc_embeddings[i, :doc_count] = batch_embeddings

        return {
            "embeddings": torch.tensor(doc_embeddings, dtype=torch.float32),
            "doc_mask": doc_mask,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "prefix_lengths": prompt_lengths,
            "targets": [item.target_text for item in batch],
            "queries": [item.query for item in batch],
            "is_refusal": torch.tensor([item.is_refusal for item in batch], dtype=torch.bool),
            "doc_counts": doc_counts,
        }

    return _collate


def _load_or_precompute_embeddings(
    cfg: Config,
    dataset: Any,
    embedding_model: Any,
    split: str,
) -> np.ndarray | None:
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    embeddings_path = cfg.paths.embeddings_dir / f"{split}.npy"
    if embeddings_path.exists():
        if embeddings_incomplete(embeddings_path):
            print("Resuming incomplete embeddings precompute.")
            return precompute_embeddings(
                dataset,
                embedding_model,
                cfg.embeddings,
                embeddings_path,
                resume=True,
            )
        embeddings = load_precomputed_embeddings(embeddings_path)
        if len(embeddings) != len(dataset):
            print(
                "Precomputed embeddings size mismatch; rebuilding "
                f"({len(embeddings)} vs {len(dataset)})"
            )
            return precompute_embeddings(
                dataset,
                embedding_model,
                cfg.embeddings,
                embeddings_path,
                resume=False,
            )
        return embeddings

    if _estimate_precompute(cfg, dataset, embedding_dim):
        return precompute_embeddings(dataset, embedding_model, cfg.embeddings, embeddings_path)
    return None


def train_selection(cfg: Config, selection_cfg: SelectionConfig) -> None:
    set_seed(cfg.project.seed)
    ensure_dir(cfg.paths.outputs_dir)
    ensure_dir(cfg.paths.embeddings_dir)

    train_ds = _collect_split(cfg, cfg.dataset.split)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.decoder_model)
    tokenizer.pad_token = tokenizer.eos_token

    embedding_model = load_embedding_model(cfg.embeddings, cfg.hardware.device)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    embeddings = _load_or_precompute_embeddings(
        cfg,
        train_ds,
        embedding_model,
        cfg.dataset.split,
    )

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    model = build_decoder(cfg.model, cfg.lora, cfg.hardware.device, embedding_dim)
    checkpoint_path = None
    resume_step = 0
    if cfg.training.adapter_checkpoint:
        checkpoint_path = Path(cfg.training.adapter_checkpoint)
    else:
        latest = find_latest_checkpoint(cfg.paths.outputs_dir)
        if latest:
            resume_step, checkpoint_path = latest

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=cfg.hardware.device)
        lora_state = None
        if isinstance(state, dict) and "adapter_state" in state:
            adapter_state = state["adapter_state"]
            lora_state = state.get("lora_state")
            optimizer_state = state.get("optimizer_state")
            resume_step = int(state.get("global_step", resume_step))
        else:
            adapter_state = state
            optimizer_state = None
            parsed_step = parse_checkpoint_step(checkpoint_path)
            resume_step = parsed_step if parsed_step is not None else resume_step
        if any(key.startswith("_orig_mod.") for key in adapter_state):
            adapter_state = {
                key.replace("_orig_mod.", ""): value for key, value in adapter_state.items()
            }
        model.adapter.load_state_dict(adapter_state)
        if lora_state is not None:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model.model, lora_state)
    else:
        optimizer_state = None

    model.model.requires_grad_(False)
    model.model.eval()
    model.adapter.train()

    trainable_params = list(model.adapter.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.learning_rate)
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    dataset = SelectionDataset(
        train_ds,
        selection_cfg,
        samples=selection_cfg.train_samples,
        seed=cfg.dataset.shuffle_seed,
    )
    generator = torch.Generator().manual_seed(cfg.dataset.shuffle_seed)
    collate_fn = _build_collate_fn(
        cfg,
        selection_cfg,
        tokenizer,
        train_ds,
        embeddings,
        embedding_model,
        include_assistant_text=True,
        add_generation_prompt=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
    )

    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError("Training dataset is empty")

    total_steps = cfg.training.epochs * steps_per_epoch
    if resume_step >= total_steps:
        print(f"Checkpoint at step={resume_step} already meets total steps; skipping.")
        return

    resume_epoch = resume_step // steps_per_epoch
    resume_batch = resume_step % steps_per_epoch

    optimizer.zero_grad(set_to_none=True)
    global_step = resume_step
    progress = tqdm(
        total=total_steps,
        desc="Training",
        unit="step",
        initial=global_step,
    )

    device = torch.device(cfg.hardware.device)
    mean_loss = 0.0

    for epoch in range(resume_epoch, cfg.training.epochs):
        for batch_index, batch in enumerate(dataloader):
            if epoch == resume_epoch and batch_index < resume_batch:
                continue
            global_step += 1
            if resume_batch:
                resume_batch = 0
            batch_embeddings = batch["embeddings"].to(device)
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention = batch["attention_mask"].to(device)
            doc_mask = batch["doc_mask"].to(device)

            batch_size, max_docs, _ = batch_embeddings.shape
            prefix = model.adapter(batch_embeddings.view(batch_size * max_docs, -1))
            prefix = prefix.view(batch_size, max_docs, -1, model.model_dim)
            prefix = prefix * doc_mask[:, :, None, None]
            prefix = prefix.reshape(batch_size, -1, model.model_dim)

            inputs = model.model.get_input_embeddings()(batch_input_ids)
            prefix_inputs = torch.cat([prefix, inputs], dim=1)

            prefix_attention = doc_mask.repeat_interleave(cfg.model.prefix_tokens, dim=1)
            attention = torch.cat([prefix_attention, batch_attention.bool()], dim=1)

            prefix_labels = torch.full(
                (batch_input_ids.shape[0], prefix.shape[1]),
                -100,
                dtype=torch.long,
                device=device,
            )
            labels = torch.cat([prefix_labels, batch_input_ids], dim=1)

            prompt_lengths = batch["prefix_lengths"].to(device)
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
                torch.save(
                    {
                        "adapter_state": model.adapter.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "global_step": global_step,
                    },
                    save_path,
                )

            progress.update(1)

    progress.close()
    final_path = cfg.paths.outputs_dir / "adapter-final.pt"
    torch.save(
        {
            "adapter_state": model.adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
        },
        final_path,
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def evaluate_selection(cfg: Config, selection_cfg: SelectionConfig) -> dict[str, Any]:
    eval_ds = _collect_split(cfg, cfg.evaluation.split)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.decoder_model)
    tokenizer.pad_token = tokenizer.eos_token

    embedding_model = load_embedding_model(cfg.embeddings, cfg.hardware.device)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    embeddings = _load_or_precompute_embeddings(
        cfg,
        eval_ds,
        embedding_model,
        cfg.evaluation.split,
    )

    model = build_decoder(cfg.model, cfg.lora, cfg.hardware.device, embedding_dim)
    device = torch.device(cfg.hardware.device)
    state = torch.load(cfg.evaluation.adapter_checkpoint, map_location=device)
    lora_state = None
    if isinstance(state, dict) and "adapter_state" in state:
        lora_state = state.get("lora_state")
        adapter_state = state["adapter_state"]
    else:
        adapter_state = state
    if any(key.startswith("_orig_mod.") for key in adapter_state):
        adapter_state = {
            key.replace("_orig_mod.", ""): value for key, value in adapter_state.items()
        }
    model.adapter.load_state_dict(adapter_state)
    if lora_state is not None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model.model, lora_state)
    model.adapter.eval()
    model.model.eval()

    dataset = SelectionDataset(
        eval_ds,
        selection_cfg,
        samples=selection_cfg.eval_samples,
        seed=cfg.dataset.shuffle_seed + 1,
    )
    collate_fn = _build_collate_fn(
        cfg,
        selection_cfg,
        tokenizer,
        eval_ds,
        embeddings,
        embedding_model,
        include_assistant_text=False,
        add_generation_prompt=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.evaluation.batch_size,
        collate_fn=collate_fn,
    )

    predictions = []
    references = []
    queries = []
    refusal_flags = []
    doc_counts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            batch_embeddings = batch["embeddings"].to(device)
            doc_mask = batch["doc_mask"].to(device)
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention = batch["attention_mask"].to(device)

            batch_size, max_docs, _ = batch_embeddings.shape
            prefix = model.adapter(batch_embeddings.view(batch_size * max_docs, -1))
            prefix = prefix.view(batch_size, max_docs, -1, model.model_dim)
            prefix = prefix * doc_mask[:, :, None, None]
            prefix = prefix.reshape(batch_size, -1, model.model_dim)

            prompt_embeds = model.model.get_input_embeddings()(batch_input_ids)
            prefix_embeds = torch.cat([prefix, prompt_embeds], dim=1)

            prefix_attention = doc_mask.repeat_interleave(cfg.model.prefix_tokens, dim=1)
            attention = torch.cat([prefix_attention, batch_attention.bool()], dim=1)

            generated = model.model.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=attention,
                max_new_tokens=cfg.evaluation.max_new_tokens,
                do_sample=cfg.evaluation.do_sample,
                temperature=cfg.evaluation.temperature,
                top_p=cfg.evaluation.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

            prefix_len = prefix.shape[1]
            prompt_lengths = batch_attention.sum(dim=1).tolist()
            trimmed = [
                generated[i, prefix_len + length :] for i, length in enumerate(prompt_lengths)
            ]
            decoded = tokenizer.batch_decode(trimmed, skip_special_tokens=True)
            predictions.extend(decoded)
            references.extend(batch["targets"])
            queries.extend(batch["queries"])
            refusal_flags.extend(batch["is_refusal"].tolist())
            doc_counts.extend(batch["doc_counts"].tolist())

    normalized_preds = [_normalize_text(text) for text in predictions]
    normalized_refs = [_normalize_text(text) for text in references]

    exact_matches = [pred == ref for pred, ref in zip(normalized_preds, normalized_refs)]
    positive_indices = [i for i, flag in enumerate(refusal_flags) if not flag]
    refusal_indices = [i for i, flag in enumerate(refusal_flags) if flag]

    positive_matches = [exact_matches[i] for i in positive_indices]
    refusal_matches = [exact_matches[i] for i in refusal_indices]

    token_f1 = _token_f1(predictions, references)

    metrics = {
        "samples": len(predictions),
        "positive_samples": len(positive_indices),
        "refusal_samples": len(refusal_indices),
        "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "positive_exact_match": float(np.mean(positive_matches)) if positive_matches else 0.0,
        "refusal_accuracy": float(np.mean(refusal_matches)) if refusal_matches else 0.0,
        "token_f1": float(token_f1),
    }

    report_path = cfg.paths.outputs_dir / "eval-report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    results_path = cfg.paths.outputs_dir / "eval-results.parquet"
    _write_results(results_path, queries, predictions, references, refusal_flags, doc_counts)

    return metrics


def _write_results(
    path: Any,
    queries: list[str],
    predictions: list[str],
    references: list[str],
    refusal_flags: list[bool],
    doc_counts: list[int],
) -> None:
    import polars as pl

    table = pl.DataFrame(
        {
            "query": queries,
            "prediction": predictions,
            "reference": references,
            "is_refusal": refusal_flags,
            "doc_count": doc_counts,
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
