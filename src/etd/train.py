from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
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
    embeddings_incomplete,
    estimate_embedding_storage,
    load_embedding_model,
    load_precomputed_embeddings,
    precompute_embeddings,
)
from etd.models import build_decoder
from etd.utils import ensure_dir, find_latest_checkpoint, parse_checkpoint_step, set_seed


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


def _build_prompt(
    tokenizer: Any,
    prompt_cfg: Any,
    text: str,
    include_assistant_text: bool,
    add_generation_prompt: bool,
) -> str:
    messages = [
        {"role": "system", "content": prompt_cfg.system},
        {"role": "user", "content": prompt_cfg.user_prefix},
    ]
    if include_assistant_text:
        messages.append({"role": "assistant", "content": text})

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


def _build_collate_fn(
    cfg: Config,
    tokenizer: Any,
    embeddings: np.ndarray | None,
    embedding_model: Any,
    include_text: bool = True,
) -> Callable[[list[dict[str, Any]]], Batch]:
    rng = random.Random(cfg.dataset.shuffle_seed)

    def _collate(batch: list[dict[str, Any]]) -> Batch:
        indices = [item["idx"] for item in batch]
        texts = [item["text"] for item in batch]

        if cfg.dataset.min_tokens is not None:
            max_length = rng.randint(cfg.dataset.min_tokens, cfg.dataset.max_tokens)
        else:
            max_length = cfg.dataset.max_tokens

        prompts = [
            _build_prompt(
                tokenizer,
                cfg.prompt,
                text,
                include_assistant_text=include_text,
                add_generation_prompt=not include_text,
            )
            for text in texts
        ]
        tokens = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        prompt_only = [
            _build_prompt(
                tokenizer,
                cfg.prompt,
                "",
                include_assistant_text=False,
                add_generation_prompt=False,
            )
            for _ in prompts
        ]
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
        embeddings_path = cfg.paths.embeddings_dir / "train.npy"
        if embeddings_path.exists():
            if embeddings_incomplete(embeddings_path):
                print("Resuming incomplete embeddings precompute.")
                embeddings = precompute_embeddings(
                    train_ds,
                    embedding_model,
                    cfg.embeddings,
                    embeddings_path,
                    resume=True,
                )
            else:
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
                        resume=False,
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

    model = build_decoder(cfg.model, cfg.lora, cfg.hardware.device, embedding_dim)

    if cfg.lora.enabled and not cfg.training.adapter_checkpoint:
        raise ValueError("LoRA enabled but training.adapter_checkpoint is not set")

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
        if isinstance(state, dict) and "adapter_state" in state:
            adapter_state = state["adapter_state"]
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
    else:
        optimizer_state = None

    model.model.requires_grad_(False)
    if cfg.lora.enabled and hasattr(model.model, "enable_adapter_layers"):
        model.model.enable_adapter_layers()

    import os
    if not os.environ.get("DISABLE_TORCH_COMPILE"):
        model.model = torch.compile(model.model)
        model.adapter = torch.compile(model.adapter)

    if cfg.lora.enabled:
        model.model.train()
    else:
        model.model.eval()
    model.adapter.train()

    if cfg.training.freeze_adapter:
        model.adapter.requires_grad_(False)

    trainable_params = [p for p in model.adapter.parameters() if p.requires_grad]
    if cfg.lora.enabled:
        trainable_params += [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.learning_rate)
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    device = torch.device(cfg.hardware.device)
    mean_loss = 0.0

    dataset = IndexedTextDataset(train_ds)
    generator = torch.Generator().manual_seed(cfg.dataset.shuffle_seed)
    collate_fn = _build_collate_fn(cfg, tokenizer, embeddings, embedding_model, include_text=True)
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

    for epoch in range(resume_epoch, cfg.training.epochs):
        for batch_index, batch in enumerate(dataloader):
            if epoch == resume_epoch and batch_index < resume_batch:
                continue
            global_step += 1
            if resume_batch:
                resume_batch = 0
            torch.compiler.cudagraph_mark_step_begin()
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
                torch.save(
                    {
                        "adapter_state": model.adapter.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
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
            "epoch": cfg.training.epochs,
        },
        final_path,
    )
