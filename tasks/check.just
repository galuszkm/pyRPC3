# run all check tasks
[group("check")]
check: check-format check-code check-type check-hooks

# check code formatting (no auto-fix)
[group("check")]
check-format:
    uv run ruff check --select=I {{SOURCES}} {{TESTS}}
    uv run ruff format --check {{SOURCES}} {{TESTS}}

# check code quality
[group("check")]
check-code:
    uv run ruff check {{SOURCES}} {{TESTS}}

# check code typing with ty (astral type checker)
[group("check")]
check-type:
    uv run ty check --exit-zero-on-warning {{SOURCES}} {{TESTS}}

# check pre-commit hooks
[group("check")]
check-hooks:
    uv run pre-commit run --all-files
