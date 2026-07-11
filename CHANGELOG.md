# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.0.1 (2026-07-11)

### Refactor

- **build**: migrate from setup.py to uv + pyproject.toml with hatchling backend
- **tooling**: add ruff (linter + formatter), ty (type checker), pytest-cov
- **tasks**: add just task runner with modular task files
- **hooks**: add pre-commit with detect-secrets baseline
- **ci**: add GitHub Actions CI workflow (lint, type check, test matrix)
- **docs**: add CONTRIBUTING.md, SECURITY.md, CHANGELOG.md
- **structure**: reorganize source to `src/pyRPC3/` package layout

## v1.0.0 (2023-01-01)

### Added

- **Reading RPC3 files** — parse RPC3 (.rsp, .rpc, .tim) binary files to extract header info and channel data
- **Writing RPC3 files** — save processed channel data to valid RPC3 format
- **Data normalization** — normalize channel data to 16-bit integer ranges
- **Channel class** — represent individual data channels with metadata (name, units, dt, scale)
- **Interactive plotting** — visualize channel data using Matplotlib
- **Comprehensive tests** — pytest suite covering read, write, round-trip, and channel operations
