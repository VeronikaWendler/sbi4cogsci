# Toy example — Bayesian inference for a one-parameter simulator

- **Instructor:** (example — copy this folder's pattern for your own tutorial)
- **When:** not a schedule slot
- **Stack:** Python 3.11+ · marimo + uv (PEP 723, no install needed beyond uv)

## Run

```bash
uvx marimo edit --sandbox toy_example.py
```

## Requirements

Only [uv](https://docs.astral.sh/uv/). The command above creates an isolated,
pinned environment from the notebook's inline PEP 723 header on first run.
No data downloads; runs in seconds on any laptop.

## Files

- `toy_example.py` — marimo notebook, **source of truth**
- `toy_example_page.ipynb` — baked export WITH outputs (the website page;
  regenerate after editing, see below). The `_page` suffix is required:
  Quarto silently skips an `.ipynb` whose stem matches a sibling `.py`.

## Regenerate the site artifact (maintainer)

```bash
uvx marimo export ipynb --sandbox --include-outputs \
  toy_example.py -o toy_example_page.ipynb
```
