# SteeringAnalysis

Concept vector extraction and activation steering analysis for LLMs.

Extracts steering vectors from contrast pairs, applies them to model generation, and verifies the assumptions underlying activation steering via controlled experiments.

## What It Does

Given a concept (e.g., "refusal", "sentiment", "polite"), the pipeline:

1. **Loads contrast pairs** from HuggingFace datasets — positive and negative examples of the concept.
2. **Extracts a steering vector** per layer using mean-difference or PCA aggregation of hidden state activations.
3. **Applies steering** during generation by injecting the vector at specified layers with configurable multipliers.
4. **Verifies Assumption 1** — that steered outputs have cosine similarity with the concept vector above a threshold while unsteered outputs stay below — at both token-level (Exp1) and layer-level (Exp2).

Supports three steering methods:
- **Additive** — `h' = h + m·v` (default)
- **Angular** — norm-preserving direction shift: `h' = (h + m·v) · (‖h‖ / ‖h + m·v‖)`
- **Prefix** — additive steering applied only to the first N generation tokens

## Quick Start

Requires Python ≥3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Run the test suite (CPU-only, no GPU needed):

```bash
uv run pytest
```

## Experiments

### Extract Steering Vectors

```bash
uv run python scripts/extract_vectors.py --concept sentiment --model Qwen/Qwen3-1.7B
uv run python scripts/extract_vectors.py --concept refusal --method pca --num-pairs 50
```

### Apply Steering to Generation

```bash
uv run python scripts/apply_steering.py \
  --vector results/vectors/sentiment.pt --concept sentiment

# Prefix steering (first 5 tokens only)
uv run python scripts/apply_steering.py \
  --vector results/vectors/refusal.pt --concept refusal --steer-tokens 5

# Angular steering
uv run python scripts/apply_steering.py \
  --vector results/vectors/polite.pt --concept polite --steering-method angular
```

### Verify Assumption 1

```bash
uv run python scripts/verify_assumption1.py \
  --model EleutherAI/pythia-70m-deduped --concept refusal

# With angular steering
uv run python scripts/verify_assumption1.py \
  --model EleutherAI/pythia-70m-deduped --concept sentiment --steering-method angular

# With prefix steering
uv run python scripts/verify_assumption1.py \
  --model EleutherAI/pythia-70m-deduped --concept refusal --steer-tokens 10
```

### Verify Contrast Pairs

```bash
uv run python scripts/verify_contrast_pairs.py \
  --model EleutherAI/pythia-70m-deduped --concept refusal
```

All scripts support `--help` for full option listing.

## Project Structure

```
src/steering_analysis/     # Library code
├── config.py              # Dataclass configurations
├── types.py               # ContrastPair, SteeringVector, ContrastPairMetadata
├── models.py              # HookedModel — HF model wrapper with forward hooks
├── extract.py             # Steering vector extraction + dataset loaders
├── steering.py            # apply_steering() — generation with steering
├── assumption_verification.py  # Exp1/Exp2 verification on generated text
├── contrast_verification.py    # Exp1/Exp2 on source contrast pairs
└── utils.py               # Helpers

scripts/                   # CLI entrypoints (require GPU)
tests/                     # Test suite (CPU-only with mock models)
docs/                      # Experiment descriptions
results/                   # Experiment outputs (.pt, .json, .jsonl)
```

## Concepts and Datasets

| Concept | Positive Source | Negative Source |
|---------|----------------|-----------------|
| `refusal` | LLM-LAT/benign-dataset | LLM-LAT/harmful-dataset |
| `sentiment` | glue/sst2 (label=1) | glue/sst2 (label=0) |
| `polite` | Intel/polite-guard (polite) | Intel/polite-guard (impolite) |

## Output Formats

- **`.pt`** — PyTorch tensors (cosine similarity matrices)
- **`.json`** — Summary statistics and full experiment results
- **`.jsonl`** — Per-sample steering generation records

## Development

```bash
uv run pytest                        # run tests
uv run ruff check src/ tests/        # lint
uv run ruff format src/ tests/       # format
```

This project follows TDD — write failing tests first, then implement.

## License

MIT
