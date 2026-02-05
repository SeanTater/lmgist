# lmgist

Research monorepo for language model experiments.

## Projects

### [etd/](etd/) — Embedding-to-Text Decoding

Can we reverse a sentence embedding back into text? Trains small prefix adapters to decode E5/MiniLM embeddings through frozen LLM decoders. Key result: a 2-layer MLP with a single prefix token achieves BLEU 44.6 at 7.8:1 compression.

### [small-agent/](small-agent/) — Small Agent Training

Making 4B parameter models viable for real coding agent work. SFT on tool use with failure recovery, rejection sampling RL, and curriculum training for multi-step coding tasks.

## Structure

Each project has its own `pyproject.toml` and can be developed independently:

```
cd etd && uv sync
cd small-agent && uv sync
```
