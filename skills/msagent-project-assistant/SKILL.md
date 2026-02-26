---
name: msagent-project-assistant
description: Work on this msAgent repository with the existing engineering workflow. Use when modifying source code, tests, CLI behavior, configuration, or project docs in this repo.
---

# msagent Project Assistant

Follow this workflow when handling changes in this repository:

1. Read the target files first and understand current behavior before editing.
2. Keep changes minimal and consistent with the current code style.
3. Add or update tests when behavior changes.
4. Run focused tests first, then broader tests if needed.
5. Summarize changed files and verification results clearly.

Repository conventions:

- Source code is under `src/msagent/`.
- Tests are under `tests/`.
- Use `PYTHONPATH=src pytest ...` for local test execution.
