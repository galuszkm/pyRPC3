# bump package version
[group("commit")]
commit-bump:
    uv run cz bump

# create conventional commit
[group("commit")]
commit-files:
    uv run cz commit

# get commit info
[group("commit")]
commit-info:
    uv run cz info
