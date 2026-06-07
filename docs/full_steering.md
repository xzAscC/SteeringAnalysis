# Full Steering Experiment

## Research Question

How does applying a steering vector at every generation step affect both concept expression and task capability, compared to prefix-limited steering?

## Background

Full steering applies the concept vector at every generation step without exception. The steering perturbation is:

$$\tilde{h}_l(m) = h_l + m \cdot \Delta h$$

applied at each forward pass through the steered layer. Unlike prefix steering, which stops after N steps, full steering continuously displaces activations from their natural trajectory. This creates the strongest possible intervention: the model never returns to an unsteered generation path during the response.

Theoretical analysis suggests this is a double-edged sword. On one hand, full steering maximizes concept expression by maintaining constant pressure. On the other hand, repeated perturbation at every step is more likely to push activations away from the model's valid activation manifold, causing capability degradation that scales with the multiplier.

The validity decay model from the steering geometry analysis captures this:

$$D(m) = \left(1 + \frac{(m - m_+)^2}{L_+}\right)^{-p_+}$$

where $D(m)$ represents the scalar validity factor as a function of steering strength. For full steering, the cumulative effect of repeatedly applying $m \cdot \Delta h$ amplifies the off-manifold drift relative to prefix steering.

## Hypothesis

**H1**: Full steering achieves higher concept expression (PrefOdds) than prefix steering at matched multipliers, but at greater cost to task utility (UtilOdds).

**H2**: The preference-utility tradeoff under full steering follows the validity decay model: concept expression initially increases with multiplier, then degrades as activations leave the manifold.

**H3**: Full steering causes measurable distributional shift even on unrelated tasks, measurable via $D_{\text{KL}}(p_{\text{steered}} \| p_{\text{base}})$.

## Experimental Design

### Independent Variables

| Variable | Values |
|----------|--------|
| Concept | refusal, sentiment, polite |
| Steering multiplier | 0.01, 0.1, 1.0, 10.0 (scaled by per-layer average activation norm) |
| Layer | All extractable layers (varies by model) |
| steer_tokens | None (full steering applies at every step) |

### Dependent Variables

- **Preference log-odds** (PrefOdds): concept expression. $\text{PrefOdds}(q) = \mathcal{L}_n - \mathcal{L}_p$.
- **Utility log-odds** (UtilOdds): task capability. $\text{UtilOdds}(q) = \log \frac{e^{-\mathcal{L}_p} + e^{-\mathcal{L}_n}}{1 - e^{-\mathcal{L}_p} - e^{-\mathcal{L}_n}}$.
- **Steered output text**: qualitative inspection.

### Protocol

1. Extract steering vectors for each concept using contrast pair extraction.
2. Normalize vectors to unit norm.
3. Compute per-layer average activation norm $\bar{a}_l$ from the prompt set.
4. For each (concept, layer, multiplier) combination:
   - Apply steering at scale $m \cdot \bar{a}_l$ at every generation step (steer_tokens=None).
   - Record the generated text, multiplier, layer, avg_activation.
5. Compare against prefix steering outputs at matched multiplier values to isolate the effect of steering duration.

### Parameters

```python
from steering_analysis import SteeringConfig

config = SteeringConfig(
    steer_tokens=None,
    multipliers=[0.01, 0.1, 1.0, 10.0],
    num_samples=10,
    max_new_tokens=100,
    temperature=0.0,
)
```

## Expected Results

1. **Concept expression increases monotonically with multiplier** at low-to-moderate values, then may saturate or decrease at very high multipliers where validity decay dominates.
2. **Utility degrades more under full steering than prefix steering** at matched concept expression, because the model is continuously displaced from its natural manifold.
3. **The preference-utility tradeoff curve** should show that prefix steering occupies a more favorable region (higher concept expression per unit of utility loss) than full steering.
4. **At very high multipliers** (e.g., 10.0), full steering may produce degenerate or incoherent output, consistent with activations being pushed far off the valid manifold.

## Comparison with Prefix Steering

| Property | Full Steering | Prefix Steering |
|----------|---------------|-----------------|
| Steering duration | Every step | First N steps only |
| Concept expression | Maximum (at matched multiplier) | May be sufficient via latent plan |
| Utility preservation | Lower (continuous perturbation) | Higher (model returns to natural trajectory) |
| Off-manifold risk | Higher (cumulative drift) | Lower (limited perturbation window) |
| Best for | Stress-testing concept directions, measuring upper bound | Practical steering with capability preservation |

## Reproduction

```bash
uv run python scripts/apply_steering.py \
  --vector results/vectors/sentiment.pt \
  --concept sentiment \
  --model Qwen/Qwen3-1.7B \
  --multipliers 0.01 0.1 1.0 10.0 \
  --output results/steering/
```

Note: omitting `--steer-tokens` defaults to full steering (steer_tokens=None).

## Output Format

JSONL files under `results/steering/`, one per layer. Each row:

| Field | Description |
|-------|-------------|
| `prompt` | Input prompt |
| `generated_text` | Steered model output |
| `layer` | Layer index where steering was applied |
| `multiplier` | Steering multiplier (before activation scaling) |
| `avg_activation` | Per-layer average activation norm used for scaling |
| `steer_tokens` | null (full steering applies at every step) |
| `sample_index` | Sample number within the prompt set |
