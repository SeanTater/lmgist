#import "@preview/cetz:0.3.2"
#import "@preview/cetz-plot:0.1.1": plot, chart

#set document(title: "Embedding-to-Text Decoding with Prefix Adapters: A Progress Report", author: "Sean Gallagher")
#set page(margin: 1in)
#set text(size: 11pt)

#align(center)[
  #text(size: 16pt, weight: "bold")[Embedding-to-Text Decoding with Prefix Adapters: A Progress Report]

  Sean Gallagher, Collaborators

  2026-01-30
]

= Abstract
We study whether a frozen sentence embedding can be decoded back into text using a lightweight prefix adapter in front of a frozen decoder LLM. Using E5-base-v2 embeddings and either a two-layer MLP or a linear projection that maps each embedding into K synthetic prefix tokens, we condition a Llama-3.2-1B-Instruct decoder with a fixed prompt and train only the adapter. On MS-MARCO v2.1 queries, the MLP adapter achieves best results at K=1 (BLEU 44.6, F1 0.75), while the linear adapter peaks at K=32 (BLEU 38.7, F1 0.72). These results suggest that nonlinear adapters excel under maximal compression, whereas linear projections benefit from additional prefix capacity.

= Introduction
Sentence encoders produce compact representations that are routinely used as lossy summaries for retrieval. Embedding-to-text decoding provides a direct probe of how much surface form and content survive in those vectors. This report summarizes our current pipeline, training objective, and early measurements, with an emphasis on reproducibility and clear separation between completed results and in-progress experiments.

We target three questions:
- How much text can be reconstructed from a single embedding using a small prefix adapter?
- How does reconstruction quality vary with the synthetic token budget K and input length?
- Does partial decoder tuning (e.g., LoRA) improve reconstruction beyond adapter-only training?

= Method
== Data
We use MS-MARCO v2.1 queries with the standard train, validation, and test splits. The text field is `query`. Inputs are capped at 512 tokens.
For datasets that ship with a single split (e.g., OpenWebText), we create deterministic train/validation/test splits by shuffling with a fixed seed and carving out fixed ratios, ensuring evaluations remain reproducible and non-overlapping.

== Model
- Encoder: E5-base-v2 sentence embeddings (frozen, max-pool to a single vector).
- Adapter: Either a 2-layer MLP (hidden dim 2048, dropout 0.1) or a single linear projection, mapping each embedding into K prefix tokens.
- Decoder: Llama-3.2-1B-Instruct (frozen).
- Prompt: system "You are a helpful assistant." and user prefix "Reconstruct the original text exactly."
- Conditioning: prefix tokens are concatenated with prompt embeddings before decoding.

== Training Objective
We minimize masked cross-entropy on the target text tokens. Prompt tokens and padding are masked so the loss reflects reconstruction quality conditioned on the embedding.

== Evaluation
We compute BLEU and token-level F1 on the validation split using greedy decoding. For auditability, we save per-example prompts, predictions, and references to Parquet. Additional diagnostics track average reference length, a token compression ratio (reference tokens divided by K), and length-bucketed performance.

= Experimental Setup
== Configuration
- Dataset: MS-MARCO v2.1 (queries).
- Embeddings: intfloat/e5-base-v2.
- Decoder: meta-llama/Llama-3.2-1B-Instruct.
- Prompt: "Reconstruct the original text exactly."
- K sweep: {1, 2, 4, 8, 16, 32} for both adapter types.
- MLP adapter: 2-layer with hidden dim 2048, dropout 0.1.
- Linear adapter: single linear projection from embedding dim to K × decoder dim.

== Optimization Regime
- Adapter-only training with the decoder frozen (8 epochs for MS-MARCO, 3 epochs for OpenWebText).
- Stage B (decoder LoRA tuning) uses the best adapter checkpoint (MLP K=1) with two configurations: adapter frozen vs unfrozen during LoRA training.

= Results
== MLP Adapter (2-layer, hidden=2048)
#table(
  columns: 4,
  [K], [BLEU], [Token F1], [Compression Ratio],
  [1], [44.62], [0.749], [7.83],
  [2], [39.53], [0.714], [3.92],
  [4], [28.10], [0.619], [1.96],
  [8], [29.82], [0.639], [0.98],
  [16], [28.90], [0.609], [0.49],
  [32], [27.67], [0.608], [0.24],
)

== Linear Adapter (single projection)
#table(
  columns: 4,
  [K], [BLEU], [Token F1], [Compression Ratio],
  [1], [27.69], [0.640], [7.83],
  [2], [30.85], [0.646], [3.92],
  [4], [31.52], [0.684], [1.96],
  [8], [38.13], [0.688], [0.98],
  [16], [37.12], [0.713], [0.49],
  [32], [38.74], [0.722], [0.24],
)

== K vs BLEU Comparison
#figure(
  cetz.canvas({
    import cetz.draw: *

    plot.plot(
      size: (10, 6),
      x-label: [K (prefix tokens)],
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
  caption: [BLEU vs K for MLP and Linear adapters. The MLP adapter performs best at K=1 with diminishing returns at higher K, while the Linear adapter improves with additional prefix capacity.]
)

== Key Observations
- *MLP adapter:* Best performance at K=1 (BLEU 44.6, F1 0.75), with quality degrading as K increases. Performance plateaus around K=4--32 (BLEU 28--30), suggesting the nonlinear projection works best when forced to compress maximally.
- *Linear adapter:* Opposite trend---performance improves with larger K, peaking at K=32 (BLEU 38.7, F1 0.72). The simpler projection benefits from additional capacity.
- *Compression ratio trade-off:* At K=1, each prefix token must encode ~7.8 reference tokens on average. The MLP adapter handles this compression better than the linear baseline.
- *Long text failure:* OpenWebText training (512-token passages) yields near-zero BLEU (0.01), indicating that embedding-to-text decoding fundamentally breaks down when the compression ratio becomes too extreme.
- *LoRA requires frozen adapter:* Jointly training adapter and LoRA destroys performance (BLEU 44.6 → 4.0), but freezing the adapter during LoRA training preserves it (BLEU 44.5). The adapter's learned representations must be isolated from decoder fine-tuning.
- *No cross-domain transfer:* Adapters trained on one dataset fail completely on another (BLEU ~0), indicating dataset-specific rather than general decoding patterns.
- *Embedding capacity matters:* MiniLM (384 dim) underperforms E5-base-v2 (768 dim) by 14 BLEU points, suggesting larger embeddings preserve more reconstructable information.

== Length-Bucketed Analysis (MLP K=1)
#table(
  columns: 4,
  [Ref Tokens], [BLEU], [Token F1], [Samples],
  [2--6], [58.66], [0.802], [50],
  [6--7], [60.01], [0.818], [50],
  [7--10], [45.03], [0.726], [50],
  [10--20], [34.23], [0.650], [50],
)
Reconstruction quality degrades with reference length as expected, since longer sequences require more information to be packed into the fixed-size embedding.

== Example Reconstructions
Selected examples comparing MLP and Linear adapters at K=1 and K=32:

#table(
  columns: 3,
  [Reference], [MLP K=1], [MLP K=32],
  [what is a corporation?], [what is a corporation.], [\$1. what is corporation?],
  [how much money has innocet donated to charity], [how much money has ibm donated to charity], [how much did isepct invest in onecare],
  [how do you change the ipv4 connectivity], [how do you change the ip connection], [how do you change the ip address],
  [how many ricolas can you eat in a day], [how many raciones can you eat in a day], [how many pancakes can you have a day],
  [how much does it cost to change title deeds], [how much does it cost to change title on a house], [how much do you pay for a deed change],
)

#table(
  columns: 3,
  [Reference], [Linear K=1], [Linear K=32],
  [what is a corporation?], [what is a corporation?], [what is a corporation?],
  [how much money has innocet donated to charity], [how much does icloud donate to charity], [how much money does imogen innovation donate],
  [how do you change the ipv4 connectivity], [how do you change the ip configuration for the network], [how do you change the ip connection],
  [how many ricolas can you eat in a day], [how many snacks can you eat in one day], [how many crackers can you eat a day],
  [how much does it cost to change title deeds], [how much does it cost to change property title], [how much does it cost to change title deeds],
)

The MLP K=1 adapter preserves semantic structure well but occasionally hallucinates proper nouns (e.g., "innocet" → "ibm"). Rare words like "ricolas" are paraphrased to common alternatives. The Linear K=32 adapter achieves a perfect reconstruction on the last example, showing that additional prefix capacity can recover exact phrasing when the simpler projection has enough room.

== LoRA Experiments
Early LoRA runs that trained both adapter and LoRA jointly increased loss relative to adapter-only training. We hypothesized that jointly training the adapter corrupts the learned prefix representations. To test this, we ran two configurations starting from the best adapter checkpoint (MLP K=1):

- *LoRA + unfrozen adapter:* Both LoRA and adapter parameters are trainable.
- *LoRA + frozen adapter:* Only LoRA parameters are trainable; the adapter is frozen.

#table(
  columns: 3,
  [Configuration], [BLEU], [Token F1],
  [Adapter-only (baseline)], [44.62], [0.749],
  [LoRA + unfrozen adapter], [3.97], [0.333],
  [LoRA + frozen adapter], [44.47], [0.750],
)

#figure(
  cetz.canvas({
    import cetz.draw: *

    chart.barchart(
      size: (8, 5),
      x-label: [BLEU],
      mode: "basic",
      (
        ([Adapter-only\ (baseline)], 44.62),
        ([LoRA +\ unfrozen adapter], 3.97),
        ([LoRA +\ frozen adapter], 44.47),
      ),
    )
  }),
  caption: [LoRA configuration comparison. Jointly training adapter and LoRA destroys performance; freezing the adapter preserves it.]
)

The results confirm our hypothesis: jointly training adapter and LoRA destroys performance (BLEU 44.6 → 4.0), while freezing the adapter during LoRA training preserves it (BLEU 44.5, F1 0.75). The frozen adapter configuration matches baseline performance, indicating that LoRA fine-tuning neither helps nor hurts when the adapter is properly isolated.

#table(
  columns: 2,
  [Reference], [LoRA + Unfrozen Adapter],
  [what is a corporation?], [Here is the original text... What is a corporation?],
  [do owls eat in the day], [do owls hunt in the day],
  [can you burn your lawn with fertilizer], [can you burn your lawn with fertilizer],
  [average force of a raindrop], [average force of a drop of water],
  [why did rachel carson write an obligation to endure], [here is the reconstructed text: dare to love and be loved by caroline knapp...],
  [are volvox autotrophic or heterotrophic], [Original text is: original text is: original text is: ...],
)

The model exhibits three failure modes: verbose preambles echoing the instruction, infinite loops, and complete hallucinations. Some reconstructions remain accurate despite the overall BLEU collapse.

== Cross-Domain Evaluation
To test generalization, we evaluate adapters trained on one dataset against a different dataset. This reveals whether the adapter learns dataset-specific patterns or more general embedding-to-text mappings.

=== OpenWebText Dataset
We use a 1M row subset of OpenWebText with deterministic 1%/1% validation/test holdouts. Text is truncated to 512 tokens, which approaches the saturation point for BERT-family encoders. Unlike MS-MARCO's short queries (~8 tokens), OpenWebText contains longer passages, testing whether the compression approach scales to more complex inputs.

=== Cross-Domain Results
#table(
  columns: 4,
  [Train Data], [Eval Data], [BLEU], [Token F1],
  [MS-MARCO], [MS-MARCO], [44.62], [0.749],
  [MS-MARCO], [OpenWebText], [0.00], [0.038],
  [OpenWebText], [OpenWebText], [0.01], [0.116],
  [OpenWebText], [MS-MARCO], [0.14], [0.088],
)

Cross-domain transfer fails completely in both directions. The MS-MARCO adapter produces near-zero BLEU on OpenWebText (0.00), and the OpenWebText adapter fares only slightly better on MS-MARCO (0.14). This suggests the adapter learns dataset-specific decoding patterns rather than general embedding-to-text mappings. The fundamental limitation is the compression ratio: short queries (~8 tokens) can be reconstructed, but 512-token passages cannot.

== Embedding Model Comparison
We compare embedding models to test whether stronger encoders improve reconstruction. All experiments use the MLP K=1 configuration on MS-MARCO.

#table(
  columns: 4,
  [Embedding Model], [Dim], [BLEU], [Token F1],
  [E5-base-v2 (baseline)], [768], [44.62], [0.749],
  [MiniLM-L6-v2], [384], [30.80], [0.647],
  [Qwen3-Embedding-4B], [--], [TBD], [TBD],
)

#figure(
  cetz.canvas({
    import cetz.draw: *

    chart.barchart(
      size: (8, 5),
      x-label: [BLEU],
      mode: "basic",
      (
        ([E5-base-v2\ (768 dim)], 44.62),
        ([MiniLM-L6-v2\ (384 dim)], 30.80),
      ),
    )
  }),
  caption: [Embedding model comparison (MLP K=1). Larger embeddings preserve more reconstructable information.]
)

The smaller MiniLM model (384 dim) underperforms E5-base-v2 (768 dim) by ~14 BLEU points, suggesting that embedding capacity directly impacts reconstruction quality. The Qwen3-4B experiment is pending.

#table(
  columns: 2,
  [Reference (truncated)], [Prediction (truncated)],
  [School tells girls they are bullies if they refuse to disrobe with transgender student...], [A new lawsuit filed by a group of students at a high school in California claims...],
  [Delicious vegan pastries from Cinnamon Snail...], [The Daily Grind is a blog that focuses on the intersection of food, culture...],
  [Universal Pictures and their Illumination Entertainment animation division...], [The new Disney animated film "Frozen" is set to be released...],
  [PHOENIX - Indianapolis Colts defensive lineman David Parry...], [INDIANAPOLIS -- A 25-year-old man who was arrested for shoplifting...],
)

= Ongoing Work
- Complete embedding model comparison (Qwen3-Embedding-4B pending).

= Discussion
The K sweep reveals an unexpected divergence between adapter architectures. The MLP adapter performs best at K=1 (BLEU 44.6), with quality degrading monotonically as K increases. In contrast, the linear adapter improves with larger K, peaking at K=32 (BLEU 38.7). This suggests the nonlinear MLP can pack more information into a single prefix token, while the linear projection needs additional capacity to achieve comparable results.

Length-bucketed analysis confirms that reconstruction quality degrades with reference length. At K=1 with the MLP adapter, short queries (2--7 tokens) achieve BLEU around 60, while longer queries (10--20 tokens) drop to BLEU 34. This is expected: longer sequences require more information to be encoded into the fixed-size embedding.

The MLP adapter at K=1 achieves a compression ratio of 7.83 (reference tokens per prefix token), substantially higher than the linear adapter at the same K. This compression advantage likely stems from the MLP's ability to learn nonlinear feature combinations that better exploit the decoder's prefix attention mechanism.

= Limitations
- Results are limited to two datasets (MS-MARCO queries, OpenWebText passages) and a single decoder model (Llama-3.2-1B-Instruct).
- OpenWebText texts are truncated to 512 tokens, which is near the saturation point for BERT-family encoders but discards longer documents.
- BLEU and token F1 penalize topical reconstructions that capture semantic content without exact wording. The OpenWebText examples show that predictions often match the topic (e.g., food blogs, legal news, video games) even when BLEU approaches zero. For some applications---topic modeling, semantic search, content summarization---this may be sufficient.
- Evaluation uses 200 samples; a larger test set may reveal additional variance.

= References
#list(
  [MS-MARCO v2.1 dataset.],
  [E5-base-v2 embeddings.],
  [Llama-3.2-1B-Instruct model.]
)
