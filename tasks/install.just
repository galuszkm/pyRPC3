# run install tasks
[group("install")]
install: install-project install-hooks

# install git hooks
[group("install")]
install-hooks:
    uv run pre-commit install --hook-type=pre-push
    uv run pre-commit install --hook-type=commit-msg

# install the project with all dependency groups
[group("install")]
install-project:
    uv sync --all-groups
