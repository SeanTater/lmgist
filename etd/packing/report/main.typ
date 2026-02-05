#import "@preview/cetz:0.3.2"
#import "@preview/cetz-plot:0.1.1": plot, chart

#set page(
  paper: "us-letter",
  margin: (top: 1.2cm, bottom: 1.2cm, left: 1.3cm, right: 1.3cm),
  columns: 2,
)
#set text(size: 9pt)
#set par(leading: 0.65em)
#set heading(numbering: "1.", outlined: false)
#set figure(gap: 0.3em)

#let title = [Packing Context Into Few Embeddings]
#let author = [Sean Gallagher]

#align(center)[
  #text(size: 12pt, weight: "bold")[#title]
  \n
  #text(size: 9pt)[#author]
]

#v(0.6em)

*Abstract* — We move from “how much text is recoverable from a single embedding?” to “how few embeddings can encode enough context to preserve downstream reconstruction quality?” We define a packing ratio (reference tokens per embedding) and trace a Pareto frontier between reconstruction quality (BLEU, token F1) and embedding budget. The goal is to identify the minimal number of embeddings needed to match a baseline reconstruction quality within a specified tolerance.

= Motivation

Prior results show a strong length dependence and a sharp compression ceiling when mapping a single embedding into prefix tokens. The natural next question is whether multiple embeddings can be *packed* to recover longer context with minimal embedding budget. This shifts focus from “maximum compression at $K=1$” to a trade-off curve between number of embeddings and reconstruction quality.

We treat exact reconstruction as an *upper bound* objective: if a given embedding budget can support verbatim reconstruction, it should be sufficient for most downstream tasks that tolerate paraphrase or partial recall. The packing curve therefore provides a conservative bound on how many embeddings are needed for context fidelity.

= Problem Statement

Given a document $x$ and an embedding function $E$, we split $x$ into $m$ chunks and compute embeddings $E(x_1), \dots, E(x_m)$. We then decode using a model conditioned on these embeddings and measure reconstruction quality. We seek the smallest $m$ that achieves a target quality $\tau$ (e.g., within 95% of baseline).

We will evaluate:
- *Quality metrics:* BLEU, token F1, and length-bucketed BLEU/F1.
- *Efficiency metrics:* embeddings per document, reference tokens per embedding, and token budget per embedding.
- *Pareto frontier:* the set of $(m, quality)$ points that are not dominated by any other configuration.

= Approach

== Packing Strategy

We start with uniform chunking:
- Split by token count into $m$ contiguous segments.
- Compute one embedding per segment.
- Decode using a fixed prefix-token budget per embedding (initially $K=1$).

We will later explore adaptive chunking:
- Length-aware chunking (shorter chunks for high-entropy text).
- Overlapping windows to reduce boundary loss.
- Learned pooling of multiple embeddings into fewer prefix tokens.

== Decoder Conditioning

We will test two conditioning schemes:
1. *Concatenated prefix:* one adapter per embedding, concatenated into the prompt.
2. *Pooled prefix:* a shared adapter that pools multiple embeddings into a fixed number of prefix tokens.

== Instruction-Conditioned Embeddings

Some embedding models support instruction steering. We will first measure the packing curve *without* conditioning. Then we will add an instruction such as “Reconstruct the original text exactly” to align the embedding space with the decoder’s task and compare deltas in the Pareto frontier.

== Baselines

We define the baseline as the best single-embedding model from prior work (MLP, $K=1$), evaluated on the same text length distribution. We compare packing curves against this baseline.

We will also include a “no-adapter” ablation *only if* the embedding dimensionality matches the decoder’s token embedding size, in which case we can feed embeddings directly as prefix tokens. If the dimensions do not match, we will skip this baseline (no meaningful alignment without a trainable map).

= Experimental Plan

== Data

- MS-MARCO (short text) for quick iteration.
- OpenWebText (long text) for packing behavior at scale.

== Sweep

We will sweep:
- $m$ in {1, 2, 4, 8, 16} embeddings per document.
- Fixed $K=1$ per embedding initially.
- Optional: vary $K$ at fixed $m$ once the basic curve is measured.

== Evaluation Outputs

- Per-example predictions and references.
- Eval report with overall metrics and length buckets.
- Pareto summary table: $(m, BLEU, token F1, tokens/embedding)$.

= Expected Outcomes

We hypothesize:
- A steep initial gain from $m=1$ to $m=2$.
- Diminishing returns after $m=4$ or $m=8$.
- A Pareto frontier that clarifies the minimal embedding budget required for near-baseline quality.

= Risks and Unknowns

- Chunk boundaries may cause discontinuities that reduce BLEU disproportionately.
- Concatenating many prefix tokens may dilute attention.
- Adapter capacity may need to grow with $m$.

= Next Steps

- Implement a packing evaluation pipeline (chunking + multi-embedding decoding).
- Collect a first Pareto curve on MS-MARCO.
- Extend to OpenWebText with longer contexts.
