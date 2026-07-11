---
name: library-testing
description: Write, repair, and reason about tests for pyRPC3 in tests/. Use when adding, fixing, or reviewing tests. Defines what is worth testing, what is not, and how.
metadata:
  area: testing
  stack: pytest,numpy
---

# Library Testing

The testing doctrine for **pyRPC3** (`src/pyRPC3/`).

One sentence: **a test exists to catch a real regression in behaviour — never to
mirror the code or freeze its wording.** If a test breaks when nothing a caller
depends on changed, it is wrong.

---

## Core Principles

1. **Test behaviour, not implementation.** Assert on what a caller observes:
   returned values, raised exceptions, file output. Never on private attributes
   or internal state.
2. **Determinism is mandatory.** No network, no wall-clock waits. Use `tmp_path`
   for file I/O.
3. **Each test reads as a small story.** Arrange-Act-Assert, visibly separated.
   One behaviour per test, one reason to fail.
4. **Smallest reasonable test at the lowest layer.** Pure function → unit test.
   File round-trip → integration test with `tmp_path`.
5. **Favour DAMP over DRY.** Each test is self-contained and readable
   top-to-bottom. Light duplication is fine.

---

## What to Test

| Concern | How to test |
|---------|-------------|
| Channel initialization and validation | Unit test: construct, assert attributes |
| Channel data operations (min, max, scale) | Unit test: set values, call method, assert result |
| Channel error cases (empty data, bad types) | Unit test: `pytest.raises` the right exception |
| RPC3 file reading | Integration: read real `.rsp` fixture, assert channels/headers |
| RPC3 file writing + round-trip | Integration: read → write → re-read, compare |
| normalize_int16 | Unit test: known arrays → known results |
| Error handling (missing file, corrupt data) | Unit test: assert error collected or raised |

---

## What NOT to Test

- Private methods or internal state directly — drive the public API.
- Matplotlib rendering output — test that `plot()` doesn't raise, not pixels.
- NumPy internals — trust your dependencies.
- Exact error message text — assert the exception type.
- Trivial attribute assignment.

---

## Test Structure

```
tests/
├── __init__.py
├── rsp/                    # Binary fixture files
│   └── test1.rsp
├── test_channel.py         # Channel class behaviour
├── test_rpc3.py            # RPC3 class (with dummy subclass for unit tests)
├── test_rpc3_reading.py    # RPC3 reading real fixtures (integration)
└── test_writer.py          # write_rpc3 round-trip (integration)
```

---

## Conventions

- **Name states behaviour:** `test_get_max_no_data_raises_value_error`, not
  `test_max`.
- **Use `tmp_path`** for any file writing tests.
- **Use `monkeypatch`** to override `plt.show()` in plot tests.
- **Use `pytest.raises`** for expected exceptions — match stable identifiers,
  not full messages.
- **Parametrize** equivalent cases over copy-pasting.
- **Run with** `uv run just test` (never bare `pytest`).

---

## Adding a Test — Checklist

1. Name the behaviour you're protecting in one sentence.
2. Pick the right file based on what you're testing (channel, rpc3, writer).
3. Read a sibling test and mirror its shape.
4. Assert on observable output — returned values, file content, raised
   exceptions.
5. Verify: `uv run just check && uv run just test`.
