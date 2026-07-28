# Day 2: The Bayesian toolkit — PyMC, simulation, and MCMC

- **Instructor:** Alexander Fengler
- **When:** Day 2, Tuesday July 28 — four sessions:
  - 12:00 — "A tutorial on PyMC" (60 min)
  - 14:00 — "Hands-on: simulating data from a cognitive model" (30 min)
  - 14:30 — "Hands-on: toy models for parameter estimation" (30 min)
  - 15:00 — "MCMC methods" (30 min)
- **Stack:** Python 3.12 · the shared `tutorials/` uv environment (PyMC, ArviZ,
  bambi, ssm-simulators)

## Run

```bash
cd tutorials && uv sync
```

Then open any notebook in this folder and select the `tutorials/.venv` kernel.

## Requirements

Only [uv](https://docs.astral.sh/uv/). `uv sync` builds the shared environment
from `tutorials/pyproject.toml` and the committed `tutorials/uv.lock`, so every
participant resolves **identical** versions — which matters here, because ArviZ
1.0 (May 2026) removed `plot_posterior`, `plot_ppc` and `waic` and changed the
default credible interval from 94% HDI to 89% ETI. Most PyMC material online
predates that cutover and will not run.

No GPU needed. No data is downloaded — everything is simulated in-notebook.

**Runtime for a full top-to-bottom run** (measured, Apple silicon, CPU only):

| notebook | run "all cells" |
|---|---|
| `pymc-intro.ipynb` | ~15 s |
| `simulating-cognitive-models.ipynb` | ~10 s |
| `toy-models.ipynb` | **~5 min** — fits three DDMs (4 chains each) and their posterior predictives |
| `mcmc-and-identifiability.ipynb` | **a few minutes** — fits two DDMs, one on a hard posterior |

The last two are slow deliberately: they fit the DDM repeatedly. If you are
following along live, run cells as you go rather than "Run All" at the start.

## Files

- `pymc-intro.ipynb` — session 1. What a PyMC distribution *is*, a first model,
  what `pm.sample()` actually returns (an `xarray.DataTree`), ArviZ 1.x, a
  regression by hand, then the same regression in bambi.
- `simulating-cognitive-models.ipynb` — session 2. `ssm-simulators`: call
  logic, the model zoo by introspection, and building your own model.
- `toy-models.ipynb` — session 3. The bridge from PyMC to cognitive models:
  the DDM likelihood used directly as a PyMC distribution, a design with
  three coherence levels crossed with a speed/accuracy instruction, and a
  model-comparison arc ending in the true generating model.
- `mcmc-and-identifiability.ipynb` — session 4. What MCMC is actually doing,
  a hand-written Metropolis sampler, and what makes a posterior hard — using
  a DDM whose parameters stop being separately identifiable.
- `ddm-explorer.qmd` — a standalone interactive figure (OJS) driven by the
  precomputed `ddm_grid.json`.
- `../sbi4cogsci_style.py` — shared plot style. Colours are semantic: one
  meaning per colour across all five Fengler sessions.

Exercises are inline, with solutions in collapsed `<details>` blocks. Poll
prompts are marked with **Poll** and are run live in the room.

## Editing these notebooks

**Do not edit the `.ipynb` files directly — your changes will be overwritten.**

Each notebook is generated from a percent-format Python source in `_src/`:

```
_src/pymc-intro.py     <- edit this
pymc-intro.ipynb       <- generated; committed with outputs because the
                          Pages CI has no Python and never executes anything
```

Rebuild after editing:

```bash
cd tutorials && ./build_notebooks.sh day2_bayes_toolkit/_src/pymc-intro.py
```

or with no arguments to rebuild every tutorial. The script converts the source
with jupytext and then executes it with nbconvert, so the committed notebook
always carries real output.

The split exists because a `.ipynb` diff is unreviewable — a one-line change to
a plot arrives as thousands of lines of re-encoded PNG. The `.py` is the
readable history. `_src/` is kept off the site twice over: the render globs in
`_quarto.yml` are one level deep, and Quarto ignores any path segment starting
with `_`.

Keep the source in `_src/` rather than beside its notebook. A same-stem `.py`
sitting *next to* a `.ipynb` makes Quarto drop the page — it still prints
`Output created` and writes no file.

`_src/precompute_ddm_grid.py` regenerates the JSON behind `ddm-explorer.qmd`;
that page is the one thing here you can edit directly, since it has no notebook
source.
