# Day 2: The Bayesian toolkit — PyMC, simulation, and MCMC

- **Instructor:** Alexander Fengler
- **When:** Day 2, Tuesday July 28 — three sessions:
  - 12:00 — "A tutorial on PyMC" (60 min)
  - 14:00 — "Hands-on: simulating data from a cognitive model" (30 min)
  - 14:30 — "Toy models for parameter estimation" + 15:00 "MCMC methods" (60 min, continuous)
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
The sampler-comparison section deliberately runs some chains that mix badly;
expect a couple of minutes of sampling in total, not more.

## Files

- `pymc-intro.ipynb` — session 1. What a PyMC distribution *is*, a first model,
  what `pm.sample()` actually returns (an `xarray.DataTree`), ArviZ 1.x, a
  regression by hand, then the same regression in bambi.
- `simulating-cognitive-models.ipynb` — session 2. `ssm-simulators`: call
  logic, the model zoo by introspection, and building your own model.
- `mcmc-and-identifiability.ipynb` — sessions 3+4, run continuously. Sampler
  behaviour on correlated posteriors, then a DDM that forks into a
  well-designed case and a badly-designed one.
- `../sbi4cogsci_style.py` — shared plot style. Colours are semantic: one
  meaning per colour across all five Fengler sessions.

Exercises are inline, with solutions in collapsed `<details>` blocks. Poll
prompts are marked with **Poll** and are run live in the room.
