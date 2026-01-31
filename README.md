# Embedding-to-Text Decoding

**Can we reverse a sentence embedding back into text?**

Sentence embeddings compress text into fixed-size vectors for retrieval and similarity tasks. This project asks: how much of the original text survives that compression? Can we decode it back?

## The Core Idea

Sentence encoders like E5 or MiniLM produce 384–768 dimensional vectors that capture semantic meaning. These embeddings are treated as lossy summaries—good for comparing similarity, but the original words are supposedly lost.

We test this assumption by training a small "prefix adapter" that converts an embedding back into tokens a language model can understand, then asking the LM to reconstruct the original text.

```
Original text: "what is a corporation?"
      ↓
   [Encoder]  (E5-base-v2, frozen)
      ↓
   Embedding: [0.23, -0.41, 0.87, ...]  (768 dims)
      ↓
   [Adapter]  (small MLP or linear, trained)
      ↓
   Prefix tokens: [tok1, tok2, ..., tokK]  (K synthetic tokens)
      ↓
   [Decoder]  (Llama-3.2-1B, frozen)
      ↓
   Output: "what is a corporation."
```

The adapter is the only trainable component—just a few million parameters that learn to "translate" from embedding space into the decoder's token embedding space.

## Why Prefix Tokens?

Language models attend to all previous tokens when generating. By injecting synthetic "prefix" tokens before the prompt, we give the model additional context it can attend to. The adapter's job is to pack the embedding's information into these K prefix tokens.

This is similar to prompt tuning or prefix tuning, but instead of learning task-specific soft prompts, we're learning to encode a specific embedding.

## The Surprising Result: Compression Beats Capacity

We tested two adapter architectures across different values of K (number of prefix tokens):

| Adapter | Best K | BLEU | Compression Ratio |
|---------|--------|------|-------------------|
| MLP (2-layer) | 1 | 44.6 | 7.8 tokens/prefix |
| Linear | 32 | 38.7 | 0.24 tokens/prefix |

**The MLP adapter works best with just ONE prefix token.** It achieves a 7.8:1 compression ratio—reconstructing ~8 tokens of text from a single synthetic token. Adding more prefix tokens actually hurts performance.

The linear adapter shows the opposite pattern: it needs 32 prefix tokens to approach MLP performance. Without nonlinear transformations, it can't compress as efficiently.

This suggests the MLP learns to exploit the decoder's attention mechanism in ways that simple linear projections cannot.

## What Works and What Doesn't

**Works well:**
- Short queries (5-10 tokens) reconstruct with BLEU ~45-60
- Semantic structure is preserved even when exact words aren't
- Rare words get paraphrased to common alternatives ("ricolas" → "snacks")

**Doesn't work:**
- Long passages (512 tokens) fail completely (BLEU ~0)
- Cross-domain transfer fails (adapter trained on queries can't decode paragraphs)
- Joint adapter + LoRA training destroys the adapter's learned representations

**Interesting edge case:**
- LoRA fine-tuning works IF you freeze the adapter first
- The adapter and decoder must be trained separately, not jointly

## What This Tells Us About Embeddings

1. **Embeddings preserve more surface form than expected.** We can recover exact punctuation and word order for short texts.

2. **There's a compression limit.** Around 8 reference tokens per embedding dimension seems to be the practical ceiling for reconstruction.

3. **Reconstruction is dataset-specific.** The adapter learns patterns specific to its training distribution, not general "embedding-to-text" mappings.

4. **Embedding dimension matters.** MiniLM (384d) underperforms E5 (768d) by 14 BLEU points—larger embeddings preserve more reconstructable information.

## Architecture Details

```
┌─────────────────────────────────────────────────────────────┐
│                     Prefix Adapter                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ Linear  │ →  │  ReLU   │ →  │ Linear  │ →  K prefix     │
│  │ 768→2048│    │         │    │ 2048→K*d│    tokens       │
│  └─────────┘    └─────────┘    └─────────┘                 │
│       ↑                                                     │
│   embedding (768d)                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Frozen Decoder                           │
│  [prefix₁, prefix₂, ..., prefixₖ, "Reconstruct:", ...]     │
│                              ↓                              │
│              Autoregressive generation                      │
└─────────────────────────────────────────────────────────────┘
```

The decoder sees the prefix tokens as if they were part of the prompt. It attends to them normally, extracting whatever information the adapter encoded.

## Key Findings Summary

1. **Nonlinear adapters compress better.** MLP at K=1 beats Linear at K=32.
2. **More prefix tokens can hurt.** MLP performance degrades from K=1 to K=32.
3. **Embeddings are dataset-specific.** No cross-domain transfer.
4. **Adapter and decoder must train separately.** Joint training fails.
5. **Embedding capacity predicts reconstruction quality.** 768d > 384d.

## Further Reading

A detailed NeurIPS-formatted report with figures and analysis is available at `report/main.typ`. Compile with `typst compile report/main.typ`.

For installation, configuration options, and running experiments, see [RUNBOOK.md](RUNBOOK.md).
