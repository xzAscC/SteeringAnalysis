# Angular Steering Experiment

## Research Question

Does norm-preserving angular steering — which shifts hidden state direction while preserving magnitude — maintain concept expression while reducing off-manifold drift compared to additive steering?

## Background

Standard additive steering perturbs hidden states as:

$$\tilde{h}_l(m) = h_l + m \cdot \Delta h$$

This changes both the direction and magnitude of the activation. The norm increase can push activations away from the model's valid activation manifold, potentially causing incoherent output.

Angular steering preserves the original norm while shifting direction:

$$\hat{h}_l(m) = \frac{(h_l + m \cdot \Delta h)}{\|h_l + m \cdot \Delta h\|} \cdot \|h_l\|$$

This is equivalent to: (1) record the original hidden state norm $\|h_l\|$, (2) add the steering vector to shift direction, (3) renormalize to restore the original norm.

The key property is that $\|\hat{h}_l(m)\| = \|h_l\|$ by construction, regardless of the multiplier $m$. Only the angular position in activation space changes.

## Hypothesis

**H1**: Angular steering produces concept expression comparable to additive steering at the same multiplier, as measured by cosine similarity with the concept vector.

**H2**: Angular steering reduces off-manifold drift because the activation magnitude is preserved, leading to more coherent output at high multipliers.

**H3**: Angular steering satisfies Assumption 1 (cosine similarity threshold test) with at least comparable strength to additive steering.

## Experimental Design

### Independent Variables

| Variable | Values |
|----------|--------|
| Concept | refusal, sentiment, polite |
| Steering multiplier | 0.01, 0.1, 1.0, 10.0 (scaled by per-layer average activation norm) |
| Layer | All extractable layers (varies by model) |
| Steering method | additive, angular |

### Dependent Variables

- **Per-layer cosine similarity with concept vector**: measures directional alignment.
- **Hidden state norm**: should be preserved under angular steering (verification check).
- **Experiment 1 verdicts (token-level)**: ALL / SOME / NONE classification.
- **Experiment 2 verdicts (layer-level existence)**: binary existence test.

### Protocol

1. Extract steering vectors for each concept using contrast pair extraction.
2. Normalize vectors to unit norm.
3. Compute per-layer average activation norm $\bar{a}_l$ from the prompt set.
4. For each (concept, layer, multiplier) combination:
   - Apply angular steering: shift direction, preserve norm.
   - Compute cosine similarity between steered hidden states and concept vector.
   - Run Experiment 1 (token-level) and Experiment 2 (layer-level) against thresholds.
5. Compare angular steering results against additive (full) and prefix steering results.

### Parameters

```python
from steering_analysis import VerificationConfig

config = VerificationConfig(
    steering_method="angular",
    thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
    steering_multiplier=1.0,
    num_samples=5,
    max_new_tokens=100,
    temperature=0.0,
    seed=42,
)
```

## Comparison with Additive and Prefix Steering

| Property | Additive (Full) | Prefix | Angular |
|----------|----------------|--------|---------|
| Norm preservation | No (increases with m) | No (increases with m) | Yes (by construction) |
| Direction change | Yes | Yes | Yes |
| Off-manifold risk | Highest (norm + direction) | Moderate (limited steps) | Lower (norm preserved) |
| Concept expression | Strongest (unrestricted) | Moderate (prefix only) | Comparable (direction only) |

## Reproduction

```bash
uv run python scripts/verify_assumption1.py \
  --model EleutherAI/pythia-70m-deduped \
  --concept refusal \
  --steering-method angular \
  --output results/assumption1/
```

Full experiment grid:

```bash
for model in EleutherAI/pythia-70m-deduped EleutherAI/pythia-160m-deduped EleutherAI/pythia-410m-deduped; do
  for concept in refusal sentiment polite; do
    uv run python scripts/verify_assumption1.py \
      --model "$model" \
      --concept "$concept" \
      --steering-method angular \
      --output results/assumption1/
  done
done
```

## Output Format

Same as Assumption 1 verification, with `_angular` label suffix:

| File | Description |
|------|-------------|
| `{concept}_{model}_angular_cosine_matrices.pt` | Per-layer cosine matrices (steered + unsteered) |
| `{concept}_{model}_angular_summary.json` | Threshold analysis with empirical percentiles |
| `{concept}_{model}_angular_full_results.json` | Complete Experiment 1 + Experiment 2 verdicts |
