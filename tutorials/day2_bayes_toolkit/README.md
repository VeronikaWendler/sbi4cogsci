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
| `toy-models.ipynb` | **a few minutes** — fits three DDMs and their posterior predictives |
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
- `../sbi4cogsci_style.py` — shared plot style. Colours are semantic: one
  meaning per colour across all five Fengler sessions.

Exercises are inline, with solutions in collapsed `<details>` blocks. Poll
prompts are marked with **Poll** and are run live in the room.
