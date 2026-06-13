# SteeringAnalysis

Concept vector extraction and activation steering analysis for LLMs. Python 3.13, managed with `uv`.

## Commands

```bash
uv sync                              # install dependencies
uv run pytest                        # run all tests
uv run pytest tests/test_extract.py  # run single test file
uv run pytest -x                     # stop on first failure
uv run ruff check src/ tests/        # lint
uv run ruff format src/ tests/       # format
```

## Architecture

**Package**: `steering_analysis` (import name), built by hatchling from `src/steering_analysis/`.

**Core pipeline**: load contrast pairs → extract steering vector → apply steering → verify assumptions.

| Module | Purpose |
|---|---|
| `config.py` | Dataclass configs: `ModelConfig`, `ExtractionConfig`, `SteeringConfig`, `VerificationConfig` |
| `types.py` | `ContrastPair`, `SteeringVector`, `ContrastPairMetadata` |
| `models.py` | `HookedModel` — wraps HF `AutoModelForCausalLM` with forward hooks for activation capture and steering |
| `extract.py` | Steering vector extraction (mean/PCA aggregators) + dataset loaders for 3 concepts |
| `steering.py` | `apply_steering()` — generate text with steering at configurable layers/multipliers |
| `assumption_verification.py` | Exp1 (token-level) and Exp2 (layer-level) verification of Assumption 1 |
| `contrast_verification.py` | Same experiments applied to source contrast pairs instead of generated text |

**Scripts** (`scripts/`): CLI entrypoints, run with `uv run python scripts/<name>.py`. Each has `--help`.

**Valid concepts**: `refusal`, `sentiment`, `polite`. Hardcoded in `config.VALID_CONCEPTS`.

**Steering methods**: `additive` (default), `angular` (norm-preserving). Passed via `--steering-method`.

**Dataset sources**: HF Hub datasets loaded at runtime (glue/sst2, LLM-LAT/benign+harmful, Intel/polite-guard). Tests mock these.

## Directory Layout

| Path | Purpose |
|---|---|
| `src/steering_analysis/` | All source code |
| `scripts/` | CLI entrypoints for experiments |
| `tests/` | Test suite (flat, not nested by module) |
| `docs/` | Experiment descriptions (markdown with LaTeX) |
| `results/` | Experiment output (`.pt`, `.json`, `.jsonl`) |
| `.ignore` | Gitignored dirs that OpenCode can still search (`results/`, `docs/`) |

## Testing

- **`conftest.py`** provides `FakeCausalLM`, `FakeTokenizer`, `mock_hooked_model` fixture — monkeypatches `transformers.AutoModelForCausalLM` and `AutoTokenizer` so tests run CPU-only without downloading models.
- Tests import from `steering_analysis.*` (not `src.steering_analysis`) because `pyproject.toml` sets `pythonpath = ["src"]`.
- Test files are flat in `tests/`, not nested. Naming: `test_{module}.py`.

## Development Workflow — TDD (MANDATORY)

1. **Write failing tests first.** Define expected behavior before implementation.
2. **Red → Green → Refactor.** Minimum implementation to pass, then clean up.
3. **Run tests after every change.** `uv run pytest`.
4. Never submit code without tests.

## Lint & Format

- **Ruff**: `target-version = "py313"`, `line-length = 120`
- Enabled rules: `E`, `F`, `I`, `N`, `W`, `UP`
- Run `uv run ruff check src/ tests/` and `uv run ruff format src/ tests/` before committing.

## Gotchas

- **Layer indices are relative**: `ExtractionConfig.layers` uses floats (0.0–1.0), resolved to absolute layer indices via `HookedModel.resolve_layers()`. Default: `[0.4, 0.5, 0.6, 0.7, 0.8]`.
- **`read_token_index=-1`** (default): extracts from the last token in each sequence, handling variable-length padding correctly.
- **`allenai/` models** auto-set `trust_remote_code=True` via `ModelConfig.__post_init__`.
- **Scripts require a GPU** for real model inference. Tests use fakes and run CPU-only.
- **`pad_token`**: `HookedModel` sets `pad_token = eos_token` when the tokenizer has none.
- **`.ignore`**: Do not remove entries — `results/` and `docs/` are gitignored but needed for OpenCode search.
