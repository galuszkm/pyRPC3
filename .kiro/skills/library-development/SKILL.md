---
name: library-development
description: Build and extend pyRPC3 — the Python library for reading and writing RPC3 binary time-series files in src/pyRPC3/. Use when adding or editing file parsing, channel handling, data writing, or normalization logic.
metadata:
  area: library
  stack: python,numpy,matplotlib
---

# Library Development

Rules for the **pyRPC3 library** in `src/pyRPC3/`
(Python >= 3.11 + NumPy + Matplotlib).

pyRPC3 does one thing: **read and write RPC3 binary files, exposing channel
data as NumPy arrays.** The library is small and focused — three modules, each
with a single responsibility.

---

## Project Map

```
src/pyRPC3/
├── __init__.py      # Public API: Channel, RPC3, write_rpc3
├── Channel.py       # Channel class — one time-series data channel
├── RPC3.py          # RPC3 class — read/write RPC3 binary files
└── writer.py        # write_rpc3() — low-level binary writer + normalize_int16()
```

| Module | Responsibility |
|--------|---------------|
| `Channel.py` | Data container for a single channel (number, name, units, dt, values). Plotting. |
| `RPC3.py` | Binary file parser (header + data reading), file writing delegation, exception raising on errors. |
| `writer.py` | Low-level binary serialization of channels to RPC3 format. Int16 normalization. |

---

## Core Principles — NON-NEGOTIABLE

1. **Simple over clever** — this is a small library. Keep it that way. No
   abstractions unless they earn their keep.
2. **Transparency over performance** — binary parsing is inherently tricky.
   Favour readable struct unpacking with clear variable names over clever
   bit-twiddling.
3. **Explicit over implicit** — no global state. An `RPC3` instance holds its
   own channels, headers, and errors. A `Channel` holds its own data.
4. **NumPy is the data layer** — channel values are always `np.ndarray`. Never
   convert to Python lists for processing.
5. **Fail with context** — when a file can't be read, collect the error with
   enough detail for the user to understand why. Never silently skip bad data.

---

## The Pipeline — Read and Write

**Reading:**
```
RPC3(filename)
├── _read_file()
│   ├── _read_header(file_handle)     → populates self.headers, self.channels metadata
│   └── _read_data(file_handle)       → populates channel.values arrays
└── channels: list[Channel]            ready for use
    (raises FileNotFoundError or ValueError on failure)
```

**Writing:**
```
rpc.save(filename)  OR  write_rpc3(filename, dt, channels)
├── normalize_int16(array)            → scaled to int16 range
└── binary header + data blocks       → written to file
```

---

## Python Conventions

- **Typed signatures** — every public function/method declares parameter and
  return types. Use `X | None` (not `Optional`).
- **Docstrings** on every public class and method with `Args:` / `Returns:` /
  `Raises:` sections.
- **Early returns** — handle error cases first, keep nesting shallow.
- **Raise specific exceptions** (`ValueError`, `TypeError`) with a contextual
  message.
- **Import order** — stdlib, third-party, local (enforced by ruff).
- **Naming:** `PascalCase` classes, `snake_case` functions, `UPPER_SNAKE_CASE`
  constants.

---

## Adding to the Project — Checklist

1. **Decide which module it belongs to.** Channel data handling → `Channel.py`.
   File format parsing → `RPC3.py`. Binary serialization → `writer.py`. New
   concern → new module (rare for this project's scope).
2. **Read the existing module first** and match its style.
3. **Add tests** for any new public API in `tests/`.
4. **Verify** before done:
   ```bash
   uv run just check    # ruff format + lint + ty type check + hooks
   uv run just test     # pytest with coverage
   ```

---

## Things NOT to Do

- Don't add unnecessary abstractions — this is a 3-module library, not a
  framework.
- Don't convert NumPy arrays to Python lists for processing.
- Don't silently swallow file read errors — raise `FileNotFoundError` or `ValueError`.
- Don't use `Optional[X]` / `Union` / `List` / `Dict` — use `X | None`,
  `list`, `dict`.
- Don't leave untyped signatures on public APIs.
- Don't add files or folders outside the scope of the task.
- Don't leave broken or commented-out code.
