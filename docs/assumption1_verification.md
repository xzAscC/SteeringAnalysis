# Assumption 1 Verification: Cosine Similarity Steering Detection

## Research Question

Does steering a model with a concept vector produce hidden states whose cosine similarity with the concept vector exceeds a threshold $\tau$ at specific layers?

## Background

Let $c$ be a concept vector, $O$ be generated text, and $o_i^{(l)}$ be the hidden state at layer $l$ and token position $i$.

Assumption 1 states that output $O$ is steered toward concept $c$ at layer $l$ iff:

$$\cos(o_i^{(l)}, c) > \tau$$

for all or some token positions $i$, where $\tau$ is a predefined cosine similarity threshold.

We test this assumption by comparing steered outputs $(O^+)$ against unsteered outputs $(O^-)$ from the same model and prompt set. If the assumption holds, hidden states from steered generations should align with $c$ more strongly than hidden states from unsteered generations, especially at the layers where steering has the clearest effect.

## Hypothesis

**H1**: For steered outputs, there exists at least one layer where $\cos(h_i^{(l)}, c) > \tau$ for some or all token positions.

**H2**: For unsteered outputs, $\cos(h_i^{(l)}, c) \leq \tau$ at that same layer for most token positions.

**H3**: The separation between steered and unsteered cosine similarities is layer-dependent.

## Experimental Design

### Independent Variables

| Variable | Values |
|----------|--------|
| Model | pythia-70m, pythia-160m, pythia-410m |
| Concept | refusal, sentiment, polite |
| Threshold $\tau$ | 0.1, 0.3, 0.5, 0.7, 0.9 |

### Dependent Variables

- **Per-layer cosine similarity matrices**: cosine similarity between each hidden state and the concept vector for each layer and token position.
- **Fraction of tokens above threshold**: proportion of token positions where $\cos(h_i^{(l)}, c) > \tau$.
- **Empirical percentile thresholds**: thresholds computed from the unsteered baseline distribution.

### Protocol

1. Extract concept vectors using contrast pair extraction.
2. Generate steered and unsteered outputs for each model and concept pair.
3. Extract hidden states at all layers for complete sequences.
4. Compute cosine similarity between each hidden state and the concept vector.
5. Apply threshold analysis at the token level for Experiment 1 and the layer level for Experiment 2.
6. Compute empirical percentile thresholds from the unsteered baseline.

## Sub-experiments

### Experiment 1: Token-level analysis

For each layer $l$, check whether $\cos(h_i^{+(l)}, c) > \tau$ for some or all token positions while $\cos(h_i^{-(l)}, c) \leq \tau$ for most token positions. The experiment evaluates all layers independently and reports per-layer verdicts (ALL / SOME / NONE).

This experiment tests whether any layer can classify steered versus unsteered hidden states at the token level.

### Experiment 2: Layer-level existence

Check whether there exists a layer $l$ where $\cos(h_i^{(l)}, c) > \tau$ for steered outputs, while no such layer exists for unsteered outputs.

This experiment tests whether steering creates detectable alignment at any layer, rather than requiring a fixed layer to be known in advance.

## Parameters

```python
from steering_analysis.config import VerificationConfig

config = VerificationConfig(
    thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
    empirical_percentiles=[95.0, 99.0],
    extraction_layers=[0.4, 0.5, 0.6, 0.7, 0.8],
    extraction_method="mean",
    extraction_num_pairs=50,
    steering_multiplier=1.0,
    max_new_tokens=100,
    temperature=0.0,
    num_samples=5,
    seed=42,
)
```

## Expected Results

1. **Steered outputs should show higher cosine similarity with the concept vector** than unsteered outputs for at least some layers and token positions.
2. **Low thresholds should classify more tokens as aligned** but may include more unsteered baseline tokens.
3. **High thresholds should be more selective** but may miss weaker steering effects.
4. **Layer-level separation should vary by model and concept**, with some layers showing clearer steered versus unsteered differences than others.
5. **Empirical percentile thresholds should provide a baseline-adjusted comparison** against each model and concept pair.

## Reproduction

```bash
uv run python scripts/verify_assumption1.py \
  --model EleutherAI/pythia-70m-deduped \
  --concept refusal \
  --thresholds 0.1 0.3 0.5 0.7 0.9 \
  --output results/assumption1/
```

```bash
for model in EleutherAI/pythia-70m-deduped EleutherAI/pythia-160m-deduped EleutherAI/pythia-410m-deduped; do
  for concept in refusal sentiment polite; do
    uv run python scripts/verify_assumption1.py \
      --model "$model" \
      --concept "$concept" \
      --thresholds 0.1 0.3 0.5 0.7 0.9 \
      --output results/assumption1/
  done
done
```

## Output Format

Files are written under `results/assumption1/` for each `{concept}_{model}` combination.

| File | Description |
|------|-------------|
| `{concept}_{model}_cosine_matrices.pt` | Per-layer cosine similarity matrices for steered and unsteered outputs. |
| `{concept}_{model}_summary.json` | Summary statistics, threshold pass rates, and empirical percentile thresholds. |
| `{concept}_{model}_full_results.json` | Complete verification results for token-level and layer-level analyses. |
