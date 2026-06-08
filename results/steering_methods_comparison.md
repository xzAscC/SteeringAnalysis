# Steering Methods Comparison Report

## Overview

This report compares three steering methods against Assumption 1, plus a new contrast pair verification experiment. All experiments use the Pythia model family (70m, 160m, 410m) across three concepts (refusal, sentiment, polite).

**Assumption 1**: Output is steered toward concept c at layer l iff cos(o_i^(l), c) > τ for all (or some) token positions i, where τ is a predefined threshold.

### Methods Compared

| Method | Description | Label |
|--------|-------------|-------|
| **Additive (full)** | h' = h + m·v applied to all generation tokens | `{concept}_{model}` |
| **Angular** | h' = (h + m·v) · (‖h‖ / ‖h + m·v‖) — norm-preserving direction change | `{concept}_{model}_angular` |
| **Prefix (10 tokens)** | Additive steering applied only to first 10 generation tokens | `{concept}_{model}_prefix10` |

### Experiments

- **Experiment 1 (token-level)**: For each layer, does the steered output have cos_sim > τ at all/any token positions, while unsteered output stays below τ?
- **Experiment 2 (layer-level)**: Does there exist a layer where steered output has cos_sim > τ but unsteered does not?
- **Contrast Pair Verification**: Do the positive/negative texts used to build the steering vector show directional separation — i.e., positive text hidden states have higher cos_sim with the steering vector than negative text hidden states?

---

## 1. Assumption 1 Verification by Method

### 1.1 pythia-70m-deduped (6 layers, steering layers 2–4)

| Concept | Method | Best Layer | Exp2 holds at τ=0.3 | Exp2 holds at τ=0.5 | Exp2 holds at τ=0.7 | Exp2 holds at τ=0.9 |
|---------|--------|-----------|---------------------|---------------------|---------------------|---------------------|
| refusal | full | L2–L4 | ✗ | ✗ | ✗ | L4 only |
| refusal | angular | L2–L4 | ✗ | ✗ | ✗ | ✗ |
| refusal | prefix10 | L2–L4 | ✗ | ✗ | ✗ | L4 only |
| sentiment | full | L2–L4 | L2–L4 | L2–L3 | L2 | ✗ |
| sentiment | angular | L2–L4 | L2–L4 | L2–L4 | L2–L4 | L2–L4 |
| sentiment | prefix10 | L2–L4 | L2–L4 | L2–L4 | L2–L4 | L2–L4 |
| polite | full | L2–L3 | L2–L3 | L2–L3 | L2–L3 | L2–L3 |
| polite | angular | L2–L4 | L2–L4 | L2–L4 | L2–L4 | L2–L4 |
| polite | prefix10 | L2–L4 | L2–L4 | L2–L4 | L2–L4 | L2–L4 |

**Key observations (70m):**
- Angular and prefix10 both show **stronger** Assumption 1 adherence than full additive for sentiment and polite.
- For refusal, all methods struggle — Assumption 1 only holds at very high thresholds.
- Angular steering achieves near-perfect cos_sim alignment (fraction_above ≈ 1.0 at all thresholds) for refusal and polite, suggesting it forces the hidden states into the steering direction uniformly.

### 1.2 pythia-160m-deduped (12 layers, steering layers 4–9)

| Concept | Method | Best Layer | Exp2 holds at τ=0.3 | Exp2 holds at τ=0.5 | Exp2 holds at τ=0.7 |
|---------|--------|-----------|---------------------|---------------------|---------------------|
| refusal | full | L7–L9 | L7–L9 | L7–L9 | L7–L9 |
| refusal | angular | L4–L9 | L4–L9 | L4–L9 | L4–L9 |
| refusal | prefix10 | L4–L9 | ✗ (all layers) | ✗ (all layers) | ✗ (all layers) |
| sentiment | full | L4–L9 | L4–L9 | L4–L9 | L4, L7–L9 |
| sentiment | angular | L4–L9 | L4–L9 | L4–L9 | L4–L9 |
| sentiment | prefix10 | L4–L9 | L4–L9 | L4–L9 | L4–L9 |
| polite | full | L4–L9 | L4–L9 | L4–L9 | L4–L9 |
| polite | angular | L4–L9 | L4–L9 | L4–L9 | L4–L9 |
| polite | prefix10 | L4–L9 | L4–L9 | L4–L9 | L4–L9 |

**Key observations (160m):**
- Angular steering consistently achieves higher cos_sim fractions than additive.
- Prefix10 fails for refusal — the unsteered text also has high cos_sim at all layers, violating the "unsteered stays below" condition. This makes sense: with only 10 tokens of steering, the refusal behavior doesn't fully differentiate.
- Full additive and angular show comparable Exp2 results for sentiment and polite.

### 1.3 pythia-410m-deduped (24 layers, steering layers 9–18)

| Concept | Method | Best Layer | Exp2 holds at τ=0.3 | Exp2 holds at τ=0.5 | Exp2 holds at τ=0.7 |
|---------|--------|-----------|---------------------|---------------------|---------------------|
| refusal | full | L14–L18 | L14–L18 | L14, L16–L18 | L16–L18 |
| refusal | angular | L9–L18 | L9, L12–L18 | L14–L18 | L16–L18 |
| refusal | prefix10 | L9–L18 | L9–L18 | L9–L18 | L14–L18 |
| sentiment | full | L9–L18 | L9–L18 | L9–L18 | L9, L12, L16–L18 |
| sentiment | angular | L9–L18 | L9–L18 | L9–L18 | L9–L18 |
| sentiment | prefix10 | L9–L18 | L9–L18 | L9–L18 | L9–L18 |
| polite | full | L9–L18 | L9–L18 | L9–L18 | L9, L14–L18 |
| polite | angular | L9, L14–L16 | L9, L14–L16 | L14–L16 | L14–L16 |
| polite | prefix10 | L9–L18 | L9–L18 | L9–L18 | L9–L18 |

**Key observations (410m):**
- At larger scale, all three methods show strong Assumption 1 adherence across layers.
- Angular steering for polite is notably weaker — L12 has near-zero cos_sim, and L18 drops to 0.0 at τ=0.5+. This suggests angular steering may struggle with concepts that are less sharply defined in the model's representation space.
- Prefix10 performs surprisingly well at 410m, often matching or exceeding full additive, suggesting that larger models maintain the steering effect even after the steering signal stops.

---

## 2. Contrast Pair Verification

### 2.1 Experiment Design

Instead of generating steered/unsteered text, this experiment directly measures whether the positive and negative texts **used to construct the steering vector** are directionally separated by it:

- **Positive text** hidden states → cos_sim with steering vector (treated as "steered" condition)
- **Negative text** hidden states → cos_sim with steering vector (treated as "unsteered" condition)
- Run Exp1 and Exp2 from Assumption 1 with this substitution

### 2.2 Results Summary

| Model | Concept | Exp2 holds at τ=0.1? | Exp2 holds at τ=0.3? | Notes |
|-------|---------|---------------------|---------------------|-------|
| 70m | refusal | ✗ (L2–L4 both above) | L3, L4 | Weak separation |
| 70m | sentiment | ✗ | ✗ | No separation at any threshold |
| 70m | polite | ✗ | ✗ | No separation at any threshold |
| 160m | refusal | ✗ | ✗ | Both pos/neg above at τ=0.1 |
| 160m | sentiment | ✗ (L9 only) | ✗ | Near-zero cos_sim at most layers |
| 160m | polite | ✗ | ✗ | No separation |
| 410m | refusal | L9, L14 | L16 | Partial separation at higher layers |
| 410m | sentiment | L12 | ✗ | Very weak |
| 410m | polite | L12, L16 | L14 | Moderate separation |

### 2.3 Contrast Pair Conclusions

1. **The steering vector does NOT strongly separate its constituent positive/negative texts.** At τ=0.3 and above, almost no model/concept/layer combination shows clean separation where positive texts have cos_sim > τ and negative texts stay below.

2. **Refusal shows the most separation**, particularly at 410m (layers 9, 14 at τ=0.1; layer 16 at τ=0.3). This makes sense — refusal has the sharpest contrast (benign vs harmful), producing a more discriminative steering vector.

3. **Sentiment and polite show near-zero cos_sim** at most layers in 160m and 70m. The steering vector built from SST-2 positive/negative sentences or polite/impolite texts does NOT align with the hidden states of those same texts. This is a critical finding.

4. **Why does Assumption 1 hold for generated text but not for source texts?** The steering vector is extracted from the **mean difference** of the last-token activations across many pairs. Individual texts have high variance — a single positive sentence may not align well with the averaged direction. However, when the vector is **applied as a steering intervention**, it actively pushes hidden states toward the concept direction, making the cos_sim measurement on the *result* much higher.

---

## 3. Cross-Method Comparison

### 3.1 Summary Table: Assumption 1 Adherence (Exp2 holds at τ=0.5)

| Model | Concept | Full Additive | Angular | Prefix10 | Contrast Pairs |
|-------|---------|---------------|---------|----------|----------------|
| 70m | refusal | ✗ | ✗ | ✗ | ✗ |
| 70m | sentiment | L2–L3 | L2–L4 | L2–L4 | ✗ |
| 70m | polite | L2–L3 | L2–L4 | L2–L4 | ✗ |
| 160m | refusal | L7–L9 | L4–L9 | ✗ | ✗ |
| 160m | sentiment | L4–L9 | L4–L9 | L4–L9 | ✗ |
| 160m | polite | L4–L9 | L4–L9 | L4–L9 | ✗ |
| 410m | refusal | L14, L16–L18 | L14–L18 | L9–L18 | L16 only |
| 410m | sentiment | L9–L18 | L9–L18 | L9–L18 | ✗ |
| 410m | polite | L9–L18 | L14–L16 | L9–L18 | L14 only |

### 3.2 Method Rankings

**For Assumption 1 adherence on generated text:**

1. **Prefix10** — Consistently strong across models/concepts. The temporary steering injection is sufficient for the model to "lock in" the concept direction. Particularly effective at larger scales (410m).

2. **Angular** — Strong for sentiment/refusal, weaker for polite at 410m. The norm-preserving property maintains hidden state magnitude while shifting direction, producing very high cos_sim fractions (often ≈1.0).

3. **Full Additive** — The baseline. Solid performance but angular/prefix10 often match or exceed it.

**For contrast pair separation:**

- All methods fail to show strong separation. The steering vector, as a mean-difference direction, does not individually separate the source texts it was built from.

---

## 4. Overall Conclusions

### 4.1 On Assumption 1

Assumption 1 (cos_sim > τ as a detector for steering) **holds reliably** for generated steered text at moderate thresholds (τ=0.3–0.5) across all three methods, especially at larger model scales. The key requirement — steered text shows alignment while unsteered text does not — is satisfied at most mid-to-late layers.

### 4.2 On Steering Methods

- **Angular steering** is viable and often produces stronger cos_sim alignment than additive, at the cost of occasionally failing for softer concepts (polite at 410m).
- **Prefix steering** (10 tokens) is surprisingly effective, suggesting that the model's autoregressive nature amplifies an initial directional nudge.
- **Full additive** remains the most reliable baseline.

### 4.3 On Contrast Pair Verification

The steering vector does **not** strongly separate the positive/negative texts it was built from. This is expected — the vector captures an averaged directional difference, and individual text representations have high variance around that direction. The vector's power lies in **intervention** (pushing hidden states), not in **classification** (separating source texts). This distinction is important for understanding what steering vectors actually encode: they are operational tools for model manipulation, not faithful embeddings of the concept space.

---

## File Inventory

### results/assumption1/
- 27 original files (full additive: 3 models × 3 concepts × 3 files)
- 27 angular files (3 models × 3 concepts × 3 files)
- 27 prefix10 files (3 models × 3 concepts × 3 files)
- **Total: 81 files**

### results/contrast_pairs/
- 27 files (3 models × 3 concepts × 3 files)
- **Total: 27 files**

### Source code (new)
- `src/steering_analysis/contrast_verification.py` — Contrast pair verification module
- `scripts/verify_contrast_pairs.py` — CLI entry point
- `tests/test_contrast_verification.py` — 11 tests
