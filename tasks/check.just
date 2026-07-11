# run all check tasks
[group("check")]
check: format check-code check-type check-hooks

# check code quality
[group("check")]
check-code:
    uv run ruff check {{SOURCES}} {{TESTS}}

# check code typing with ty (astral type checker)
[group("check")]
check-type:
    uv run ty check {{SOURCES}} {{TESTS}}

# check pre-commit hooks
[group("check")]
check-hooks:
    uv run pre-commit run --all-files
