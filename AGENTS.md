# SteeringAnalysis

Python 3.13 analysis project. Managed with `uv`.

## Directory Layout

| Directory | Purpose |
|---|---|
| `src/` | All source code |
| `scripts/` | Run scripts and entrypoints |
| `tests/` | Test suite |
| `docs/` | Experiment descriptions and documentation |
| `results/` | Experiment output and results |
| `.ignore` | Files gitignored but searchable by OpenCode agents |

## Tooling

- **Package manager**: `uv` (not pip/poetry)
- **Python**: 3.13 (`pyproject.toml` → `requires-python = ">=3.13"`)
- **Build config**: `pyproject.toml` — currently minimal, no dependencies yet

## Conventions

- Each experiment has its description in `docs/` and output in `results/`.
- `.ignore` contains files excluded from git but included for OpenCode search — do not remove entries from it without understanding why they are there.
