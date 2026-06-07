# Prefix Steering Experiment

## Research Question

Does steering only the first N generated tokens suffice to shift model behavior toward a target concept, while preserving task capability better than full-response steering?

## Background

Hidden states in transformer models concentrate on a low-rank activation manifold rather than being uniformly distributed in representation space. Steering pushes activations along a one-dimensional trajectory:

$$\tilde{h}_l(m) = h_l + m \cdot \Delta h$$

Moderate $m$ may move activations along a useful concept direction, while large $|m|$ can push them off the valid manifold. The key insight motivating prefix steering is that early generated tokens have an outsized effect on response direction. A prefix intervention may establish a latent generation plan that later tokens follow naturally, without requiring continuous perturbation at every step.

Formally, if early prefix tokens induce hidden states encoding the intended behavior ($h_t \to$ plan includes concept), then later tokens attending to these prefix-induced states should inherit the steering effect. This is the "latent plan" hypothesis: prefix steering does not merely add surface tokens, it moves the model into a generation trajectory that subsequent tokens sustain.

## Hypothesis

**H1**: Prefix steering (first N tokens) achieves measurable concept expression at lower capability cost than full steering (all tokens), because it perturbs fewer activations and leaves the model closer to the natural manifold for most of the generation.

**H2**: The steering effect is mediated by a latent plan encoded in early hidden states. If the linear hypothesis holds, this plan should be linearly decodable from hidden states before the model actually produces concept tokens.

## Experimental Design

### Independent Variables

| Variable | Values |
|----------|--------|
| Concept | refusal, sentiment, polite |
| Steering multiplier | 0.01, 0.1, 1.0, 10.0 (scaled by per-layer average activation norm) |
| Layer | All extractable layers (varies by model) |
| Prefix length (steer_tokens) | 1, 3, 5, 10 |

### Dependent Variables

- **Preference log-odds** (PrefOdds): measures concept expression. $\text{PrefOdds}(q) = \mathcal{L}_n - \mathcal{L}_p$, the log-likelihood ratio between concept-negative and concept-positive completions.
- **Utility log-odds** (UtilOdds): measures task capability preservation. $\text{UtilOdds}(q) = \log \frac{e^{-\mathcal{L}_p} + e^{-\mathcal{L}_n}}{1 - e^{-\mathcal{L}_p} - e^{-\mathcal{L}_n}}$.
- **Steered output text**: qualitative inspection of generated completions.

### Protocol

1. Extract steering vectors for each concept using contrast pair extraction (mean or PCA aggregation).
2. Normalize vectors to unit norm.
3. Compute per-layer average activation norm $\bar{a}_l$ from the prompt set.
4. For each (concept, layer, multiplier, prefix_length) combination:
   - Apply steering at scale $m \cdot \bar{a}_l$ for the first `steer_tokens` generation steps.
   - Record the generated text, multiplier, layer, avg_activation, and steer_tokens.
5. Compare prefix steering outputs against full steering outputs at matched multiplier values.

### Parameters

```python
from steering_analysis import SteeringConfig

config = SteeringConfig(
    steer_tokens=5,
    multipliers=[0.01, 0.1, 1.0, 10.0],
    num_samples=10,
    max_new_tokens=100,
    temperature=0.0,
)
```

## Expected Results

1. **Concept expression increases with multiplier** but saturates or degrades at very high multipliers where activations leave the manifold.
2. **Prefix steering with small steer_tokens (1-5) should achieve concept expression comparable to full steering** at the same multiplier, supporting the latent plan hypothesis.
3. **Utility should degrade less under prefix steering than full steering** at matched concept expression levels, because only early activations are perturbed.
4. **Failure modes** may appear for continuous-control concepts (e.g., "answer only in valid JSON") where token-level maintenance is required beyond the prefix window.

## When Prefix Steering May Fail

- **Continuous-control concepts**: behaviors requiring token-level maintenance rather than single early initialization. The prefix effect can decay over long generations.
- **Context-dependent replanning**: later context may force the model to revise its plan, overriding the prefix-induced trajectory.
- **Stronger models**: more capable models may be less sensitive to shallow prefix perturbations because later hidden states can recover the original task trajectory.

## Reproduction

```bash
uv run python scripts/apply_steering.py \
  --vector results/vectors/sentiment.pt \
  --concept sentiment \
  --model Qwen/Qwen3-1.7B \
  --steer-tokens 5 \
  --multipliers 0.01 0.1 1.0 10.0 \
  --output results/steering/
```

## Output Format

JSONL files under `results/steering/`, one per layer. Each row:

| Field | Description |
|-------|-------------|
| `prompt` | Input prompt |
| `generated_text` | Steered model output |
| `layer` | Layer index where steering was applied |
| `multiplier` | Steering multiplier (before activation scaling) |
| `avg_activation` | Per-layer average activation norm used for scaling |
| `steer_tokens` | Number of generation steps that received steering |
| `sample_index` | Sample number within the prompt set |
