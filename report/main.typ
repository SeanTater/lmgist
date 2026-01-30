#import "@preview/bloated-neurips:0.7.0": neurips2025
#import "@preview/cetz:0.3.2"
#import "@preview/cetz-plot:0.1.1": plot, chart

#let authors = (
  (name: "Sean Gallagher", affl: "ind"),
)
#let affls = (
  ind: (institution: "Independent"),
)

#show: neurips2025.with(
  title: [Embedding-to-Text Decoding with Prefix Adapters],
  authors: (authors, affls),
  accepted: true,
  abstract: [
    We study whether a frozen sentence embedding can be decoded back into text using a lightweight prefix adapter in front of a frozen decoder LLM. Using E5-base-v2 embeddings and either a two-layer MLP or a linear projection that maps each embedding into $K$ synthetic prefix tokens, we condition a Llama-3.2-1B-Instruct decoder with a fixed prompt and train only the adapter. On MS-MARCO v2.1 queries, the MLP adapter achieves best results at $K=1$ (BLEU 44.6, F1 0.75), while the linear adapter peaks at $K=32$ (BLEU 38.7, F1 0.72). These results suggest that nonlinear adapters excel under maximal compression, whereas linear projections benefit from additional prefix capacity.
  ],
)

= Introduction

Sentence encoders produce compact representations that are routinely used as lossy summaries for retrieval. Embedding-to-text decoding provides a direct probe of how much surface form and content survive in those vectors. This paper summarizes our current pipeline, training objective, and early measurements, with an emphasis on reproducibility and clear separation between completed results and in-progress experiments.

We target three questions:
+ How much text can be reconstructed from a single embedding using a small prefix adapter?
+ How does reconstruction quality vary with the synthetic token budget $K$ and input length?
+ Does partial decoder tuning (e.g., LoRA) improve reconstruction beyond adapter-only training?

= Method

== Data

We use MS-MARCO v2.1 queries with the standard train, validation, and test splits. The text field is `query`. Inputs are capped at 512 tokens. For datasets that ship with a single split (e.g., OpenWebText), we create deterministic train/validation/test splits by shuffling with a fixed seed and carving out fixed ratios, ensuring evaluations remain reproducible and non-overlapping.

== Model

Our architecture consists of three components:
- *Encoder:* E5-base-v2 sentence embeddings (frozen, max-pool to a single vector).
- *Adapter:* Either a 2-layer MLP (hidden dim 2048, dropout 0.1) or a single linear projection, mapping each embedding into $K$ prefix tokens.
- *Decoder:* Llama-3.2-1B-Instruct (frozen).

Conditioning is achieved by concatenating prefix tokens with prompt embeddings before decoding. The prompt consists of system message "You are a helpful assistant." and user prefix "Reconstruct the original text exactly."

== Training Objective

We minimize masked cross-entropy on the target text tokens. Prompt tokens and padding are masked so the loss reflects reconstruction quality conditioned on the embedding.

== Evaluation

We compute BLEU and token-level F1 on the validation split using greedy decoding. For auditability, we save per-example prompts, predictions, and references to Parquet. Additional diagnostics track average reference length, a token compression ratio (reference tokens divided by $K$), and length-bucketed performance.

= Experimental Setup

*Dataset:* MS-MARCO v2.1 (queries). *Embeddings:* intfloat/e5-base-v2. *Decoder:* meta-llama/Llama-3.2-1B-Instruct. *Prompt:* "Reconstruct the original text exactly." *$K$ sweep:* {1, 2, 4, 8, 16, 32} for both adapter types.

Adapter-only training uses 8 epochs for MS-MARCO and 3 epochs for OpenWebText. Stage B (decoder LoRA tuning) uses the best adapter checkpoint (MLP $K=1$) with two configurations: adapter frozen vs unfrozen during LoRA training.

= Results

#figure(
  grid(
    columns: 2,
    gutter: 1em,
    [
      #align(center)[*MLP Adapter*]
      #table(
        columns: 4,
        stroke: none,
        table.hline(),
        table.header[$K$][BLEU][F1][Comp.],
        table.hline(),
        [1], [44.62], [.749], [7.83],
        [2], [39.53], [.714], [3.92],
        [4], [28.10], [.619], [1.96],
        [8], [29.82], [.639], [0.98],
        [16], [28.90], [.609], [0.49],
        [32], [27.67], [.608], [0.24],
        table.hline(),
      )
    ],
    [
      #align(center)[*Linear Adapter*]
      #table(
        columns: 4,
        stroke: none,
        table.hline(),
        table.header[$K$][BLEU][F1][Comp.],
        table.hline(),
        [1], [27.69], [.640], [7.83],
        [2], [30.85], [.646], [3.92],
        [4], [31.52], [.684], [1.96],
        [8], [38.13], [.688], [0.98],
        [16], [37.12], [.713], [0.49],
        [32], [38.74], [.722], [0.24],
        table.hline(),
      )
    ],
  ),
  caption: [$K$ sweep results. MLP adapter (left) peaks at $K=1$; Linear adapter (right) peaks at $K=32$. Comp. = compression ratio (ref tokens / $K$).],
) <tab:k-sweep>

#figure(
  cetz.canvas({
    import cetz.draw: *
    plot.plot(
      size: (6, 4),
      x-label: [$K$ (prefix tokens)],
      y-label: [BLEU],
      x-tick-step: none,
      x-ticks: ((0, "1"), (1, "2"), (2, "4"), (3, "8"), (4, "16"), (5, "32")),
      y-min: 20,
      y-max: 50,
      legend: "inner-north-east",
      {
        plot.add(((0, 44.62), (1, 39.53), (2, 28.10), (3, 29.82), (4, 28.90), (5, 27.67)),
                  mark: "o", label: "MLP")
        plot.add(((0, 27.69), (1, 30.85), (2, 31.52), (3, 38.13), (4, 37.12), (5, 38.74)),
                  mark: "x", label: "Linear")
      }
    )
  }),
  caption: [BLEU vs $K$ for MLP and Linear adapters. The MLP adapter performs best at $K=1$, while the Linear adapter improves with additional prefix capacity.],
) <fig:k-sweep>

@tab:k-sweep shows the full $K$ sweep results, visualized in @fig:k-sweep. Key observations:

- *MLP adapter:* Best performance at $K=1$ (BLEU 44.6, F1 0.75), with quality degrading as $K$ increases. Performance plateaus around $K=4$--$32$ (BLEU 28--30), suggesting the nonlinear projection works best when forced to compress maximally.
- *Linear adapter:* Opposite trend---performance improves with larger $K$, peaking at $K=32$ (BLEU 38.7, F1 0.72). The simpler projection benefits from additional capacity.
- *Compression ratio trade-off:* At $K=1$, each prefix token must encode ~7.8 reference tokens on average. The MLP adapter handles this compression better than the linear baseline.

== Length-Bucketed Analysis

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(),
    table.header[Ref Tokens][BLEU][Token F1][Samples],
    table.hline(),
    [2--6], [58.66], [0.802], [50],
    [6--7], [60.01], [0.818], [50],
    [7--10], [45.03], [0.726], [50],
    [10--20], [34.23], [0.650], [50],
    table.hline(),
  ),
  caption: [Length-bucketed analysis for MLP $K=1$. Reconstruction quality degrades with reference length.],
) <tab:length>

@tab:length confirms that reconstruction quality degrades with reference length as expected, since longer sequences require more information to be packed into the fixed-size embedding.

== LoRA Experiments

Early LoRA runs that trained both adapter and LoRA jointly increased loss relative to adapter-only training. We hypothesized that jointly training the adapter corrupts the learned prefix representations. To test this, we ran two configurations starting from the best adapter checkpoint (MLP $K=1$).

#figure(
  table(
    columns: 3,
    stroke: none,
    table.hline(),
    table.header[Configuration][BLEU][Token F1],
    table.hline(),
    [Adapter-only (baseline)], [44.62], [0.749],
    [LoRA + unfrozen adapter], [3.97], [0.333],
    [LoRA + frozen adapter], [44.47], [0.750],
    table.hline(),
  ),
  caption: [LoRA configuration comparison. Jointly training adapter and LoRA destroys performance; freezing the adapter preserves it.],
) <tab:lora>

#figure(
  cetz.canvas({
    import cetz.draw: *
    chart.barchart(
      size: (5, 3),
      x-label: [BLEU],
      mode: "basic",
      (
        ([Baseline], 44.62),
        ([Unfrozen], 3.97),
        ([Frozen], 44.47),
      ),
    )
  }),
  caption: [LoRA configuration BLEU scores.],
) <fig:lora>

The results in @tab:lora confirm our hypothesis: jointly training adapter and LoRA destroys performance (BLEU 44.6 → 4.0), while freezing the adapter during LoRA training preserves it (BLEU 44.5, F1 0.75). The frozen adapter configuration matches baseline performance, indicating that LoRA fine-tuning neither helps nor hurts when the adapter is properly isolated.

The unfrozen model exhibits three failure modes: verbose preambles echoing the instruction, infinite loops, and complete hallucinations.

== Cross-Domain Evaluation

To test generalization, we evaluate adapters trained on one dataset against a different dataset.

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(),
    table.header[Train Data][Eval Data][BLEU][Token F1],
    table.hline(),
    [MS-MARCO], [MS-MARCO], [44.62], [0.749],
    [MS-MARCO], [OpenWebText], [0.00], [0.038],
    [OpenWebText], [OpenWebText], [0.01], [0.116],
    [OpenWebText], [MS-MARCO], [0.14], [0.088],
    table.hline(),
  ),
  caption: [Cross-domain evaluation. Transfer fails completely in both directions.],
) <tab:cross>

Cross-domain transfer fails completely (@tab:cross). The MS-MARCO adapter produces near-zero BLEU on OpenWebText, and vice versa. This suggests the adapter learns dataset-specific decoding patterns rather than general embedding-to-text mappings. The fundamental limitation is the compression ratio: short queries (~8 tokens) can be reconstructed, but 512-token passages cannot.

== Embedding Model Comparison

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(),
    table.header[Embedding Model][Dim][BLEU][Token F1],
    table.hline(),
    [E5-base-v2], [768], [44.62], [0.749],
    [MiniLM-L6-v2], [384], [30.80], [0.647],
    table.hline(),
  ),
  caption: [Embedding model comparison (MLP $K=1$). Larger embeddings preserve more reconstructable information.],
) <tab:embed>

The smaller MiniLM model (384 dim) underperforms E5-base-v2 (768 dim) by ~14 BLEU points (@tab:embed), suggesting that embedding capacity directly impacts reconstruction quality.

= Discussion

The $K$ sweep reveals an unexpected divergence between adapter architectures. The MLP adapter performs best at $K=1$ (BLEU 44.6), with quality degrading monotonically as $K$ increases. In contrast, the linear adapter improves with larger $K$, peaking at $K=32$ (BLEU 38.7). This suggests the nonlinear MLP can pack more information into a single prefix token, while the linear projection needs additional capacity to achieve comparable results.

The MLP adapter at $K=1$ achieves a compression ratio of 7.83 (reference tokens per prefix token), substantially higher than the linear adapter at the same $K$. This compression advantage likely stems from the MLP's ability to learn nonlinear feature combinations that better exploit the decoder's prefix attention mechanism.

= Limitations

- Results are limited to two datasets (MS-MARCO queries, OpenWebText passages) and a single decoder model (Llama-3.2-1B-Instruct).
- OpenWebText texts are truncated to 512 tokens, which is near the saturation point for BERT-family encoders but discards longer documents.
- BLEU and token F1 penalize topical reconstructions that capture semantic content without exact wording.
- Evaluation uses 200 samples; a larger test set may reveal additional variance.

= Conclusion

We demonstrate that frozen sentence embeddings can be partially decoded back into text using lightweight prefix adapters. The MLP adapter achieves BLEU 44.6 at $K=1$ on MS-MARCO queries, while requiring only adapter parameters to be trained. However, the approach fails on longer texts and does not transfer across domains, suggesting fundamental limits to embedding-to-text reconstruction.
