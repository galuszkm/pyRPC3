# Full project review: pyRPC3 tooling and source code

A 3-module Python library (Channel, RPC3, writter — 703 lines total) that reads and writes RPC3 binary time-series files, recently migrated from setup.py to uv with modern tooling (ruff, ty, just, pre-commit, commitizen). The tooling infrastructure (AGENTS.md, two .kiro skills, 7 justfile task modules, CI matrix, dependabot, GitHub templates, detect-secrets) significantly outweighs the library itself. The documentation layer is well-written and internally consistent — the real question is whether the overhead is justified.

Watch for: The `writter.py` module name is misspelled (confirmed). The `normalize_int16` function has no dedicated unit test despite the testing skill explicitly calling it out (confirmed). Type annotations on `writter.py` public API are incomplete — `normalize_int16` has no return type annotation (confirmed). The `check` justfile task silently re-formats code before linting, which means `just check` is not idempotent and could mask CI failures (confirmed). The CI workflow does not run `detect-secrets` or pre-commit hooks, so the secrets check exists locally but has zero enforcement in the pipeline (confirmed).

## High-level view

The AGENTS.md and skill files are well-calibrated for AI consumption at 315 lines total. The risk is maintenance drift — the docs describe a codebase that partially doesn't match reality (missing type annotations, typo in module name), and there's no automated check that skill claims stay true.

The pyproject.toml is correctly configured for a hatchling-based src layout. The numpy/matplotlib lower bounds are pinned quite high (>=2.2.5, >=3.10.1) for a library claiming Python 3.11 support — these versions may not install on all 3.11 environments.

The CI workflow covers format, lint, type-check, and test across 4 Python versions and 2 OSes. It does not run pre-commit hooks or detect-secrets, creating a gap between local and pipeline enforcement. The `--exit-zero-on-warning` flag on ty means the 7 type warnings (including a real `None` arithmetic bug in test code) never block CI.

The justfile `check` recipe calls `format` as a dependency — meaning running the "check" modifies your working tree. CI correctly uses `ruff format --check` (read-only), so local and CI behavior diverge.

The source code is straightforward binary parsing. The biggest gaps: no return type on `normalize_int16`, the module spelling mistake, `Channel.number` setter accepting `None` despite the type hint saying `int`, and `RPC3.__init__` silently collecting errors instead of raising.

<details>
<summary>Issues (10)</summary>

1. **Module name typo** — `writter.py` should be `writer.py`. Rename the file and update all imports and documentation references.
2. **Missing return type on `normalize_int16`** — Add `-> tuple[np.ndarray, float]` to the signature. The skill docs promise typed public APIs.
3. **No unit test for `normalize_int16`** — The testing skill explicitly lists this as a required test target. Add tests for zero arrays, single-value arrays, and normal cases.
4. **`just check` mutates the working tree** — Remove the `format` dependency from the `check` recipe. Replace it with `uv run ruff format --check` to match CI behavior.
5. **CI does not run detect-secrets** — Add a pre-commit or dedicated step in CI to run detect-secrets. Currently secrets enforcement is local-only.
6. **`Channel.number` setter accepts `None`** — The setter has `if value is not None and not isinstance(value, int)` which silently allows `None`, but the type hint and docstring say `int`. Remove the None guard or change the type hint.
7. **ty warnings never block CI** — The `--exit-zero-on-warning` flag means all 7 current type warnings pass silently. Fix the warnings in `src/` and consider removing the flag.
8. **High dependency lower bounds** — `numpy>=2.2.5` and `matplotlib>=3.10.1` may not resolve on all Python 3.11 environments. Consider lowering to `numpy>=1.24` and `matplotlib>=3.7` if broad 3.11 support is intended.
9. **`write_rpc3` is a trivial wrapper** — It just calls `_write_file` with identical arguments. Either inline it or give it a meaningful role (input validation, path normalization).
10. **`RPC3._read_file` returns `bool` but nobody checks it** — The constructor calls `self._read_file()` but discards the return value. Errors are silently collected in `self.errors` — a user must remember to check `get_errors()` after construction.

</details>

<details>
<summary>Details</summary>

## Module naming and API surface

The file is named `writter.py` (double-t). Every reference in the skill docs, project map, and `__init__.py` imports from `.writter`. This is a legacy typo baked into the project. It should be fixed before more tooling references accumulate, because every AI agent reading the skill docs will perpetuate it. The rename is safe — there are no external consumers referencing the internal path.

## Type annotation gaps

The skill documentation says "every public function/method declares parameter and return types." This is violated in two places:

```python
# writter.py
def normalize_int16(array: np.ndarray):  # no return type
```

```python
# writter.py
def write_rpc3(filename: str, dt: float, channels: list[Channel]):  # no return type (implicitly None, but should be explicit)
```

The `Channel.number` setter has a logic gap: it guards `if value is not None and not isinstance(value, int)` — meaning `None` passes validation silently, creating an object where `self._number` is `None` despite the property being typed as `-> int`. The `__init__` signature has `number: int` (no `None`), so this can only be triggered via direct setter assignment, but it's still an inconsistency that violates the "explicit over implicit" principle.

## The `just check` recipe mutates state

```just
check: format check-code check-type check-hooks
```

The `format` dependency calls `ruff format` (which rewrites files) and `ruff check --select=I --fix` (which rewrites imports). A developer running `just check` to verify their code is clean will have their working tree silently modified. The fix:

```just
check: check-format check-code check-type check-hooks

[group("check")]
check-format:
    uv run ruff format --check {{SOURCES}} {{TESTS}}
```

## CI enforcement gap: detect-secrets

The pre-commit config runs `detect-secrets` on commit and push. The CI workflow does not run `pre-commit run --all-files` or any equivalent. A contributor who bypasses hooks (common with `--no-verify`) or uses the GitHub web editor will never trigger secrets scanning.

## Testing coverage: what's missing

The testing skill explicitly lists `normalize_int16` as requiring unit tests ("Unit test: known arrays → known results"). No such test exists. The round-trip test exercises it indirectly, but doesn't cover edge cases:

- All-zero array (factor would be 0, triggering the `if factor > 0` branch)
- Single-element array
- Array already within int16 range
- Array with very large values near float64 limits

The `RPC3` class has 80.7% coverage. Uncovered behavioral branches include the `DATA_TYPE` error path in `_read_data` and the `read_channels` filtering logic.

## Source code: `write_rpc3` as trivial indirection

```python
def write_rpc3(filename: str, dt: float, channels: list[Channel]):
    _write_file(filename, dt, channels)
```

This adds no validation, no path handling, no error wrapping. Either it should do something (validate that `channels` is non-empty, that `dt > 0`, that all channels have data) or `_write_file` should just be the public function.

## Source code: silent error collection vs exceptions

`RPC3.__init__` reads the file and collects errors in `self.errors`, returning `False` from internal methods. But the constructor discards the return value:

```python
def __init__(self, ...):
    ...
    self._read_file()  # return value ignored
```

A user who writes `rpc = RPC3("bad_file.rsp")` gets an object with an empty `channels` list and must remember to call `get_errors()`. The skill docs say "Fail with context" but the actual behavior is fail-silent. At minimum, the docstring should warn. Ideally, a `strict=True` flag would raise on errors.

## Over-engineering assessment

The tooling-to-source ratio: ~600 lines of "meta" (skills, pyproject, justfile, CI) for 703 lines of source.

What should be simplified:
- **7 separate `.just` task files** — a single justfile with inline recipes would be more discoverable. The `secrets.just` file (4 recipes for detect-secrets management) serves a project with zero actual secrets.
- **commitizen + version bumping** — for a library at v1.0.1 with a single maintainer, this adds ceremony with no audience. There's no PyPI publishing step to justify it.
- **Two dependency groups** (dev + commit) — the split saves ~3 seconds in CI. Not worth the cognitive overhead.
- **GitHub issue templates, PR template, SECURITY.md, CONTRIBUTING.md** — cargo-culting unless external contributors are expected.

## Pre-commit config: missing ruff, broken stage wiring

Ruff is not in the pre-commit config. A developer can commit code that fails format/lint checks and won't know until CI runs. Adding `ruff-pre-commit` is trivial and would catch this locally.

The hooks are installed for `pre-push` and `commit-msg` stages (from `install-hooks`), but the pre-commit config specifies `stages: [pre-commit, pre-push]` for detect-secrets. Since no hook is installed for the `pre-commit` stage, detect-secrets only actually runs on push, not on commit — despite the config suggesting otherwise.

</details>

<details>
<summary>File map</summary>

| File | What it does |
|------|-------------|
| `AGENTS.md` | Top-level agent routing: points to skills by area |
| `.kiro/skills/library-development/SKILL.md` | Development principles, module map, conventions for src/ |
| `.kiro/skills/library-testing/SKILL.md` | Testing doctrine, what to test, what not to test |
| `.kiro/skills/library-development/references/project-map.md` | Module responsibilities, public API, tooling summary |
| `pyproject.toml` | Project metadata, build config, all tool settings |
| `.github/workflows/ci.yml` | Format/lint/type check + test matrix (4 Pythons x 2 OSes) |
| `justfile` | Task runner entry point, imports 7 task modules |
| `tasks/check.just` | Format + lint + type + hooks (mutates working tree) |
| `tasks/format.just` | Import sorting + code formatting |
| `tasks/test.just` | pytest with coverage and configurable threshold |
| `tasks/clean.just` | Remove build artifacts and caches |
| `tasks/commit.just` | Commitizen bump/commit/info |
| `tasks/install.just` | uv sync + hook installation |
| `tasks/secrets.just` | detect-secrets baseline management |
| `.pre-commit-config.yaml` | Whitespace hooks + detect-secrets |
| `src/pyRPC3/__init__.py` | Public re-exports: Channel, RPC3, write_rpc3 |
| `src/pyRPC3/Channel.py` | Data container for one time-series channel |
| `src/pyRPC3/RPC3.py` | Binary file reader + writer delegation |
| `src/pyRPC3/writter.py` | Low-level binary serialization (misspelled filename) |

</details>
