# Day 3: Tools for SBI

Materials for [Day 3 — Tools for SBI](https://stefanradev93.github.io/sbi4cogsci/schedule.html#day-3-wednesday-july-29-tools-for-sbi).

This folder holds several sessions from the same day. Each is listed below with
its own instructor, stack, and run command.

---

## HSSM (09:30)

- **Instructor:** Alexander Fengler
- **Stack:** Python 3.12 · the shared `tutorials/` uv environment
- **Run:** `cd tutorials && uv sync`, then open `hssm-intro.ipynb`

The shortest working HSSM model, the two defaults that fail silently
(`[-1, +1]` response coding and a `p_outlier=0.05` lapse), the SSM-specific fit
checks, and how to plug in a likelihood you trained yourself.

## BayesFlow — amortized inference for the DMC (10:00)

- **Instructors:** Stefan T. Radev, Simon Schaefer
- **Stack:** Python 3.12 · the shared `tutorials/` uv environment
- **Run:** `cd tutorials && uv sync`, then open `dmc-bayesflow.ipynb`

## MCMC for hierarchical Bayesian models (11:00)

- **Instructors:** Alexander Fengler, Brandon Turner
- **Stack:** Python 3.12 · the shared `tutorials/` uv environment
- **Run:** `cd tutorials && uv sync`, then open `hierarchical-mcmc.ipynb`

Opens with [a slide deck](hierarchical-mcmc-slides.qmd); the rest is live in the
notebook. Funnel geometry, centered vs non-centered, where the crossover between
them actually falls, and per-parameter parameterization in HSSM.

## Recurrent networks for dynamic data (12:00)

- **Instructor:** Tianhao (Tim) Pan
- **Stack:** Google Colab — nothing to install
- **Run:** open the Colab link below

**LaseNet** &nbsp; [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d9y-Svzs2z8EBRb5sY26Ni-n4p4tmd_W?usp=sharing)

---

## Requirements

Everything except the LaseNet Colab notebook runs in the shared environment
defined by `tutorials/pyproject.toml` and pinned by `tutorials/uv.lock`. Install
[uv](https://docs.astral.sh/uv/), then:

```bash
cd tutorials && uv sync
```

Select the `tutorials/.venv` kernel in your notebook editor. The first non-DDM
HSSM model downloads an ONNX likelihood network from HuggingFace, so that step
needs internet access.

## Repo-only files

Files beginning with `_` are deliberately kept off the published website:
`_compare-models.ipynb` is an unfinished draft, and
`_lasenet_tutorial_solution.ipynb` is an answer key. Quarto skips `_`-prefixed
paths. Note the repository itself is public, so a `_` prefix hides a file from
the site but not from GitHub.
