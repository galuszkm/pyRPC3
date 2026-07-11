# create secrets baseline for the project
[group("secrets")]
secrets-baseline:
    uv run detect-secrets scan > .secrets.baseline

# update existing secrets baseline (rescan and merge)
[group("secrets")]
secrets-update:
    uv run detect-secrets scan --baseline .secrets.baseline

# audit secrets in baseline interactively
[group("secrets")]
secrets-audit:
    uv run detect-secrets audit .secrets.baseline

# scan for secrets without modifying baseline (CI-friendly)
[group("secrets")]
secrets-check:
    uv run detect-secrets-hook --baseline .secrets.baseline $(git ls-files)
