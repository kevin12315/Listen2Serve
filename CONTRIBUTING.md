# Contributing

Use Python 3.12 or newer and install the development dependencies with `pip install -e '.[dev]'`. Run `make check-data test smoke lint` before proposing a change. Tests use simulated services and require no commercial API keys.

Keep changes focused on the released multi-turn evaluation. Explain the concrete behavior changed and the validation performed. Preserve scenario IDs, content, voice assignments, and scoring definitions unless the change explicitly concerns a benchmark revision.

Prompts are exported from code with `python scripts/dump_prompts.py`. Re-export snapshots after an intentional prompt change. Do not commit credentials, runtime logs, or personal information; local `.env`, `.venv`, `runs/`, and `reports/` directories are ignored by Git.

Report a bug with the command, Python version, code revision, condition, and a minimal reproduction. Remove secrets and customer information from reports.
