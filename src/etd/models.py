from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from etd.config import ModelConfig


class PrefixAdapter(nn.Module):
    def __init__(self, embedding_dim: int, model_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        layers = []
        input_dim = embedding_dim
        for _ in range(cfg.adapter_layers):
            layers.append(nn.Linear(input_dim, cfg.adapter_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(cfg.dropout))
            input_dim = cfg.adapter_hidden_dim
        layers.append(nn.Linear(input_dim, cfg.prefix_tokens * model_dim))
        self.net = nn.Sequential(*layers)
        self.prefix_tokens = cfg.prefix_tokens
        self.model_dim = model_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        output = self.net(embeddings)
        return output.view(output.shape[0], self.prefix_tokens, self.model_dim)


@dataclass
class DecoderWithPrefix:
    model: nn.Module
    adapter: PrefixAdapter
    tokenizer: object | None
    model_dim: int


def build_decoder(cfg: ModelConfig, device: str) -> DecoderWithPrefix:
    model_config = AutoConfig.from_pretrained(cfg.decoder_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.decoder_model)
    model.to(device)

    model_dim = model_config.hidden_size
    adapter = PrefixAdapter(embedding_dim=768, model_dim=model_dim, cfg=cfg).to(device)

    return DecoderWithPrefix(model=model, adapter=adapter, tokenizer=None, model_dim=model_dim)
