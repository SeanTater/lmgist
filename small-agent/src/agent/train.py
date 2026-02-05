"""LoRA SFT training loop."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatDataset(Dataset):
    """JSONL dataset of chat messages."""

    def __init__(self, path: str, tokenizer, max_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = [json.loads(line) for line in Path(path).read_text().strip().split("\n")]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]["messages"]

        # Tokenize full conversation
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = tokens["input_ids"][0]

        # Mask non-assistant tokens: tokenize incrementally to find boundaries
        labels = torch.full_like(input_ids, -100)
        for i, msg in enumerate(messages):
            # Tokenize up to and including this message
            prefix = self.tokenizer.apply_chat_template(messages[: i + 1], tokenize=False)
            end = len(self.tokenizer(prefix, truncation=True, max_length=self.max_len)["input_ids"])
            if msg["role"] == "assistant":
                # Tokenize up to previous message to get start
                prev = self.tokenizer.apply_chat_template(messages[:i], tokenize=False) if i > 0 else ""
                start = len(self.tokenizer(prev, truncation=True, max_length=self.max_len)["input_ids"]) if prev else 0
                labels[start:end] = input_ids[start:end]

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def train(config_path: str, data_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float32,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["dropout"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = ChatDataset(data_path, tokenizer, cfg["training"]["max_seq_len"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(loader) // cfg["training"]["grad_accum_steps"] * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    model.train()
    step = 0
    for epoch in range(cfg["training"]["epochs"]):
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / cfg["training"]["grad_accum_steps"]
            loss.backward()

            if (batch_idx + 1) % cfg["training"]["grad_accum_steps"] == 0:
                # Linear warmup then cosine decay
                step += 1
                if step <= warmup_steps:
                    lr = cfg["training"]["lr"] * step / warmup_steps
                else:
                    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                    lr = cfg["training"]["lr"] * 0.5 * (1 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()

                if step % 10 == 0:
                    print(f"step {step}/{total_steps} loss={loss.item() * cfg['training']['grad_accum_steps']:.4f} lr={lr:.2e}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"Saved to {out}")
