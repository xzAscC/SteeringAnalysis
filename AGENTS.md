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
- **Build config**: `pyproject.toml` — project metadata, dependencies, and tooling configuration

## Conventions

- Each experiment has its description in `docs/` and output in `results/`.
- `.ignore` contains files excluded from git but included for OpenCode search — do not remove entries from it without understanding why they are there.

## Development Workflow — TDD (MANDATORY)

All code MUST be written following Test-Driven Development. No exceptions.

### Rules

1. **Write tests FIRST.** Before any implementation, write failing tests that define expected behavior.
2. **Red → Green → Refactor.** Follow the strict TDD cycle:
   - **Red**: Write a test that fails (defines the desired behavior).
   - **Green**: Write the minimum implementation to make the test pass.
   - **Refactor**: Clean up code while keeping tests green.
3. **Never write implementation without a corresponding test.** Every function, class, and module must have tests written before (or alongside) the implementation.
4. **Tests must fail first.** If a test passes before implementation exists, it's not a valid TDD test — re-examine it.
5. **Run tests after every change.** Use `uv run pytest` to verify.
6. **Test file convention**: `tests/` mirrors `src/` structure. For `src/foo/bar.py`, create `tests/test_foo/test_bar.py`.

### What this means for agents

- When asked to implement a feature → start by writing tests, then implement.
- When asked to fix a bug → write a test that reproduces the bug first, then fix.
- When asked to refactor → tests must already exist and stay green throughout.
- Never submit code without tests.
