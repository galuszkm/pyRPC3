# Project Map

## Source Layout

```
src/pyRPC3/
├── __init__.py      # Public API exports: Channel, RPC3, write_rpc3
├── Channel.py       # Channel class — single time-series data channel
├── RPC3.py          # RPC3 class — binary file reader/writer
└── writter.py       # write_rpc3() + normalize_int16() — low-level serialization
```

## Public API

Consumers import from `pyRPC3` directly:

```python
from pyRPC3 import RPC3, Channel, write_rpc3
```

## Module Responsibilities

### Channel.py
- `Channel` class: holds number, name, units, dt, scale, values (np.ndarray)
- Properties with validation: `name` (str), `number` (int)
- Methods: `get_max()`, `get_min()`, `copy()`, `plot()`, `_apply_scale()`

### RPC3.py
- `RPC3` class: reads binary RPC3 files (.rsp, .rpc, .tim)
- Constructor reads file immediately: `RPC3(filename, debug=False)`
- Header parsing: format, num channels, dt, channel metadata
- Data reading: interleaved frame groups, int16/float unpacking
- Writing: `save()` delegates to `write_rpc3()`
- Error collection: `get_errors()` returns list of error strings

### writter.py
- `normalize_int16(array)`: scale ndarray to int16 range, return (normalized, factor)
- `write_rpc3(filename, dt, channels)`: serialize channels to binary RPC3 format

## Binary Format Notes

RPC3 files use:
- 128-byte header entries (32-byte key + 96-byte value)
- Interleaved channel data in frame groups
- Int16 normalized values with per-channel scale factors
- Little-endian byte order

## Stack

- Python >= 3.11
- NumPy (array operations, data storage)
- Matplotlib (channel plotting)
- No other runtime dependencies

## Tooling

- `uv` — package manager
- `ruff` — linter + formatter
- `ty` — type checker
- `pytest` + `pytest-cov` — testing
- `just` — task runner
- `pre-commit` + `detect-secrets` — git hooks
- `commitizen` — conventional commits + version bumping
