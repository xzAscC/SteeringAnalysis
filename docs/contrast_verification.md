# Contrast Pair Verification Experiment

## Research Question

Do the positive and negative texts used to construct a steering vector show directional separation from that vector? Specifically, do positive text hidden states have higher cosine similarity with the steering vector than negative text hidden states?

## Background

Steering vectors are extracted from contrast pairs using the mean difference of last-token activations:

$$\Delta h = \frac{1}{N} \sum_{i=1}^{N} (h_i^{+} - h_i^{-})$$

The vector $\Delta h$ represents the averaged directional difference between positive and negative concepts. This experiment asks: if we directly measure the cosine similarity between $\Delta h$ and the individual positive/negative text hidden states, does $\Delta h$ separate them?

This is distinct from the standard Assumption 1 verification, which measures cosine similarity on **generated** steered/unsteered text. Here we measure on the **source** texts that built the vector.

## Hypothesis

**H1 (Contrast Separation)**: For at least one layer, positive text hidden states will have cos_sim > τ with the steering vector, while negative text hidden states will have cos_sim < τ at the same threshold. This would replicate Assumption 1's Experiment 2 using source texts instead of generated text.

## Protocol

For each model (pythia-70m, 160m, 410m) and concept (refusal, sentiment, polite):

1. **Load contrast pairs** from the same datasets used for steering vector extraction (same seed, same num_pairs).
2. **Extract steering vector** per layer using standard mean-difference aggregation.
3. **Extract hidden states** from positive and negative texts at all layers.
4. **Compute cosine similarity** between each text's hidden states and the steering vector at each layer.
5. **Replicate Experiment 1 (token-level)**: For each layer, does the positive text have cos_sim > τ at all/any token positions, while negative text stays below τ?
6. **Replicate Experiment 2 (layer-level)**: Does there exist a layer where positive text cos_sim > τ but negative text does not?

Thresholds tested: τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.

## Results Location

- `results/contrast_pairs/{concept}_{model}_cosine_matrices.pt`
- `results/contrast_pairs/{concept}_{model}_summary.json`
- `results/contrast_pairs/{concept}_{model}_full_results.json`

## Key Findings

### Positive/Negative texts are NOT strongly separated by the steering vector

| Model | Concept | Exp2 holds at τ=0.3? | Notes |
|-------|---------|---------------------|-------|
| 70m | refusal | L3, L4 | Weak separation only at high τ |
| 70m | sentiment | ✗ | No separation at any threshold |
| 70m | polite | ✗ | No separation at any threshold |
| 160m | refusal | ✗ | Both pos/neg above at τ=0.1 |
| 160m | sentiment | ✗ | Near-zero cos_sim at most layers |
| 160m | polite | ✗ | No separation |
| 410m | refusal | L16 | Partial separation at higher layers |
| 410m | sentiment | ✗ | Very weak |
| 410m | polite | L14 | Moderate separation |

### Interpretation

The steering vector does **not** strongly separate the positive/negative texts it was constructed from. This is expected for three reasons:

1. **Aggregation dilution**: The steering vector is a mean difference across many pairs. Individual text representations have high variance around this averaged direction.

2. **Token-level vs last-token**: The vector is extracted from last-token activations, but the experiment measures cosine similarity across all tokens in the sequence. Many tokens (e.g., function words) carry no concept signal.

3. **Intervention vs classification**: The steering vector's power lies in **intervention** — actively pushing hidden states toward a concept direction during generation. It is not designed as a **classifier** that separates source texts. The standard Assumption 1 experiments confirm this: when the vector is applied as an intervention, the generated text shows strong alignment (cos_sim > τ), even though the source texts do not.

### Conclusion

Assumption 1 holds for **steering interventions** (generated text shows cos_sim > τ while unsteered text stays below) but does **not** hold for **source text classification** (positive/negative texts are not separated by the steering vector at moderate thresholds). This distinction is important: steering vectors are operational tools for model manipulation, not faithful embeddings of the concept space.
