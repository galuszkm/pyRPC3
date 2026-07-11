# pyRPC3 — Agent Instructions

This is **pyRPC3**: a Python library for reading, processing, and writing RPC3
(.rsp, .rpc, .tim) binary files containing time-series channel data.

---

## Read the Skill First

Before touching code, load the skill for the area you are working in:

| Area | Skill to load |
|------|---------------|
| **Library source** (`src/pyRPC3/`) | `.kiro/skills/library-development/SKILL.md` |
| **Library tests** (`tests/`) | `.kiro/skills/library-testing/SKILL.md` |

---

## Core Principles — Apply Everywhere

When in doubt, apply these in order.

1. **Simple over clever** — the dullest solution that correctly solves the
   problem is the right one. Readable and maintainable beats terse.
2. **Transparency over performance** — prefer code that clearly shows what it
   does. Optimize only with measured evidence that it matters.
3. **Explicit over implicit** — no hidden state, no magic. Make data flow
   obvious.
4. **Single responsibility** — each module and class does one thing.
5. **Composition over inheritance** — small, focused pieces that compose.

---

## Behaviour Rules — Apply Everywhere

- **Smallest reasonable change.** Don't refactor unrelated code to land a
  feature. Touch only what the task requires.
- **Read before writing.** Before editing a file, read it. Before creating
  something new, read a sibling that plays the same role and match its shape.
- **If you find something broken in the area you're working, fix it.** Don't
  leave broken or commented-out code behind.
- **Never add files or change code outside the scope of the task.**
- **Comments explain what and why, never when or how something changed.**
- **Verify before done** — `uv run just check` then `uv run just test`.
