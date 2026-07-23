# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Toy example — Bayesian inference for a one-parameter simulator

    This folder is the **copyable pattern** for SBI4CogSci tutorial folders
    (see `tutorials/CONTRIBUTING.md`). The science here is deliberately tiny:
    we simulate data from a one-parameter "cognitive model" (a Gaussian with
    unknown mean $\mu$ and known noise $\sigma$) and recover $\mu$ with its
    exact conjugate posterior — the simplest possible instance of the
    simulate → infer loop this summer school is about.

    The notebook is a [marimo](https://marimo.io) file: plain Python, runnable
    with a single command (see the README), with its dependencies declared
    inline (PEP 723). The committed `.ipynb` next to it is a baked export
    that the course website renders — nothing executes at site-build time.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    n_slider = mo.ui.slider(
        5, 200, value=30, step=5, label="Number of simulated observations $n$"
    )
    n_slider
    return (n_slider,)


@app.cell
def _(n_slider, np):
    TRUE_MU = 1.2  # ground-truth parameter the "simulator" uses
    SIGMA = 1.0  # known observation noise

    _rng = np.random.default_rng(42)
    observations = _rng.normal(loc=TRUE_MU, scale=SIGMA, size=n_slider.value)
    return SIGMA, TRUE_MU, observations


@app.cell
def _(SIGMA, TRUE_MU, np, observations, plt):
    # Conjugate normal–normal update: prior mu ~ N(0, prior_sd^2), sigma known.
    _prior_sd = 2.0
    _n = observations.size
    post_var = 1.0 / (1.0 / _prior_sd**2 + _n / SIGMA**2)
    post_mean = post_var * observations.sum() / SIGMA**2

    _grid = np.linspace(-2.0, 4.0, 400)
    _prior = np.exp(-0.5 * (_grid / _prior_sd) ** 2) / np.sqrt(
        2 * np.pi * _prior_sd**2
    )
    _posterior = np.exp(-0.5 * (_grid - post_mean) ** 2 / post_var) / np.sqrt(
        2 * np.pi * post_var
    )

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(_grid, _prior, ls="--", color="gray", label="prior")
    _ax.plot(_grid, _posterior, color="C0", label="posterior")
    _ax.axvline(TRUE_MU, color="C3", ls=":", label=r"true $\mu$")
    _ax.plot(
        observations,
        np.zeros_like(observations),
        "|",
        color="k",
        alpha=0.4,
        label="data",
    )
    _ax.set_xlabel(r"$\mu$")
    _ax.set_ylabel("density")
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return post_mean, post_var


@app.cell(hide_code=True)
def _(TRUE_MU, mo, n_slider, post_mean, post_var):
    mo.md(
        f"""
    With $n = {n_slider.value}$ observations the posterior is
    $\\mu \\mid x \\sim \\mathcal{{N}}({post_mean:.2f}, {post_var**0.5:.2f}^2)$,
    concentrating around the true value ${TRUE_MU}$ as you increase the slider —
    the whole notebook re-runs reactively.

    **To add your own tutorial:** copy this folder, replace the content, keep
    the README's structure (one run command), and add a row to
    `tutorials/index.qmd`. Every layer used here — marimo, uv, the baked
    `.ipynb` site page — is optional; see `tutorials/CONTRIBUTING.md`.
    """
    )
    return


if __name__ == "__main__":
    app.run()
