"""Figures for the hierarchical-modelling material (Day 3, 11:00).

Written once, used three ways:

1. imported by `day3_sbi_tools/_hierarchical-mcmc.py` (the notebook),
2. baked to PNG for `hierarchical-mcmc-slides.qmd` by `_bake_slide_figures.py`,
3. ready for marimo — see the shape rule below.

**The shape rule.** Expensive *computation* and cheap *plotting* are separate
functions. A `fig_*` function takes already-computed results plus a few scalar
display parameters and returns a Figure. That is what makes these drop into a
marimo cell with `mo.ui.slider` without re-running an MCMC fit on every drag:

    result = pooling_experiment(...)          # once, in its own cell
    fig_shrinkage(result, highlight_n=n.value)  # re-runs on every slider move

Colours come from `sbi4cogsci_style`, so a colour means the same thing here as
in every other session.
"""

from __future__ import annotations

import numpy as np

import sbi4cogsci_style as S

# --------------------------------------------------------------------------
# Neal's funnel — geometry, no data
# --------------------------------------------------------------------------


def funnel_draws(n: int = 40_000, sd_v: float = 3.0, seed: int = 0):
    """Prior draws from Neal's funnel: v ~ Normal(0, sd_v), x | v ~ Normal(0, e^{v/2})."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, sd_v, n)
    x = rng.normal(0.0, np.exp(v / 2))
    return x, v


def fig_funnel(x, v, *, xlim=25.0, neck_below=-3.0, ax=None, title="Neal's funnel"):
    """The funnel, with the neck marked. `neck_below` shades v < that value."""
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
    keep = np.abs(x) < xlim
    ax.plot(x[keep], v[keep], "o", color=S.PRIMARY, ms=1.5, alpha=0.15,
            ls="none", label="prior draws")
    ax.axhspan(-9, neck_below, color=S.DIVERGENT, alpha=0.10)
    S.annotate(ax, "the neck:\nwidth shrinks like $e^{v/2}$",
               xy=(0, -6), xytext=(0.42 * xlim, -7.5))
    ax.set(title=title, xlabel="$x_1$", ylabel="$v$  (log scale)",
           xlim=(-xlim, xlim), ylim=(-9, 9))
    ax.legend(loc="upper right")
    if fig is not None:
        fig.tight_layout()
    return fig if fig is not None else ax.figure


# --------------------------------------------------------------------------
# Partial pooling on an unbalanced panel — why you would want a hierarchy
# --------------------------------------------------------------------------

#: Deliberately brutal: half the participants have almost no data. This is the
#: regime where pooling earns its keep, and it is common in real cognitive data
#: (dropouts, short sessions, excluded trials).
DEFAULT_TRIAL_COUNTS = (5, 6, 7, 8, 10, 12, 15, 18, 22, 28,
                        60, 90, 130, 180, 240, 300, 380, 450, 520, 600)


def simulate_unbalanced_panel(trial_counts=DEFAULT_TRIAL_COUNTS,
                              mu_v=0.8, tau_v=0.30,
                              a=1.2, z=0.5, t=0.3, seed=0):
    """Simulate a DDM panel where each participant has a different drift AND a
    wildly different number of trials.

    Returns (observed (n, 2), participant_index (n,), v_true (n_participants,),
    trial_counts array).
    """
    from ssms.basic_simulators.simulator import simulator

    rng = np.random.default_rng(seed)
    counts = np.asarray(trial_counts, dtype=int)
    v_true = rng.normal(mu_v, tau_v, counts.size)

    obs_blocks, idx_blocks = [], []
    for j, (v_j, n_j) in enumerate(zip(v_true, counts)):
        out = simulator(theta=[float(v_j), a, z, t], model="ddm",
                        n_samples=int(n_j), random_state=seed + j)
        obs_blocks.append(np.column_stack([out["rts"].flatten(),
                                           out["choices"].flatten()]))
        idx_blocks.append(np.full(int(n_j), j))

    return (np.vstack(obs_blocks), np.concatenate(idx_blocks), v_true, counts)


def fit_pooling(observed, participant_idx, n_participants, *, pooling,
                a=1.2, z=0.5, t=0.3, draws=1000, tune=1000, chains=4, seed=0,
                log_likelihood=False):
    """Fit per-participant drift with `pooling` in {"none", "partial"}.

    `a`, `z` and `t` are held at their true values on purpose: the point of this
    experiment is what pooling does to `v`, and fixing the rest keeps the
    comparison about one thing.
    """
    import pymc as pm
    from hssm.likelihoods import DDM

    with pm.Model():
        if pooling == "none":
            v = pm.Normal("v", 0.0, 2.0, shape=n_participants)
        elif pooling == "partial":
            mu = pm.Normal("mu_v", 0.0, 2.0)
            tau = pm.HalfNormal("tau_v", 1.0)
            offset = pm.Normal("z_off", 0.0, 1.0, shape=n_participants)
            v = pm.Deterministic("v", mu + tau * offset)   # non-centered
        else:
            raise ValueError("pooling must be 'none' or 'partial'")
        DDM("obs", v=v[participant_idx], a=a, z=z, t=t, observed=observed)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          nuts_sampler="pymc", progressbar=False, random_seed=seed)
        if log_likelihood:
            # Not stored by default, and az.loo needs it.
            pm.compute_log_likelihood(idata, progressbar=False)
    return idata


def pooling_experiment(**kwargs):
    """Run the whole no-pooling-vs-partial-pooling comparison.

    Expensive (two MCMC fits, a few seconds). Call once; hand the result to
    `fig_shrinkage` / `fig_pooling_error` as often as you like.
    """
    seed = kwargs.pop("seed", 0)
    with_loo = kwargs.pop("with_loo", True)
    observed, idx, v_true, counts = simulate_unbalanced_panel(seed=seed, **kwargs)
    n_p = counts.size
    est, p_loo = {}, {}
    for pooling in ("none", "partial"):
        idata = fit_pooling(observed, idx, n_p, pooling=pooling, seed=seed,
                            log_likelihood=with_loo)
        est[pooling] = idata.posterior.dataset["v"].values.reshape(-1, n_p).mean(0)
        if with_loo:
            import arviz as az
            p_loo[pooling] = float(az.loo(idata).p)
    return {"v_true": v_true, "trial_counts": counts,
            "no_pooling": est["none"], "partial_pooling": est["partial"],
            "n_trials_total": int(counts.sum()),
            # Nominal counts: one drift per participant, plus mu and tau when pooled.
            "nominal": {"no_pooling": n_p, "partial_pooling": n_p + 2},
            "p_loo": {"no_pooling": p_loo.get("none"),
                      "partial_pooling": p_loo.get("partial")}}


def fig_shrinkage(result, *, split_at=30, ax=None):
    """Each participant as an arrow from its no-pooling to its partial-pooling
    estimate, against how many trials they contributed.

    The lesson is visual: participants with little data get pulled a long way
    toward the population; participants with a lot barely move.
    """
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))

    n = result["trial_counts"]
    for j in range(n.size):
        ax.annotate("", xy=(n[j], result["partial_pooling"][j]),
                    xytext=(n[j], result["no_pooling"][j]),
                    arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=1.1))
    ax.plot(n, result["no_pooling"], "o", color=S.NAIVE, ms=6,
            ls="none", label="no pooling")
    ax.plot(n, result["partial_pooling"], "o", color=S.PRIMARY, ms=6,
            ls="none", label="partial pooling")
    ax.plot(n, result["v_true"], "X", color=S.TRUTH, ms=7, ls="none",
            label="truth", zorder=5)
    ax.axvline(split_at, color=S.MUTED, ls=":", lw=1)
    ax.set(xscale="log", xlabel="trials contributed by this participant (log)",
           ylabel="estimated drift $v$",
           title="Shrinkage: who gets moved, and how far")
    ax.legend(loc="lower right", fontsize=9, ncol=3)
    if fig is not None:
        fig.tight_layout()
    return fig if fig is not None else ax.figure


def fig_pooling_error(result, *, split_at=30, ax=None):
    """Absolute error against trial count, for both fits."""
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
    n = result["trial_counts"]
    for key, colour, label in (("no_pooling", S.NAIVE, "no pooling"),
                               ("partial_pooling", S.PRIMARY, "partial pooling")):
        ax.plot(n, np.abs(result[key] - result["v_true"]), "o-", color=colour,
                ms=5, lw=1.4, label=label)
    ax.axvline(split_at, color=S.MUTED, ls=":", lw=1)
    ax.set(xscale="log", xlabel="trials contributed (log)",
           ylabel=r"$|\hat{v} - v_{\mathrm{true}}|$",
           title="The gain is concentrated where the data is thin")
    ax.legend(fontsize=9)
    if fig is not None:
        fig.tight_layout()
    return fig if fig is not None else ax.figure


# --------------------------------------------------------------------------
# The two parameterizations, and the fact that their advantage reverses
# --------------------------------------------------------------------------


def _hier_normal(y_bar, se, *, parameterization):
    """Hierarchical normal, centered or non-centered.

    A half-normal prior on tau, not a log-normal: a log-normal suppresses both
    zero and infinity and so hides the very geometry we are trying to show
    (Betancourt, *Hierarchical Modeling*, 2020, §4.1).
    """
    import pymc as pm

    n = y_bar.size
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 5.0)
        tau = pm.HalfNormal("tau", 5.0)
        if parameterization == "centered":
            theta = pm.Normal("theta", mu, tau, shape=n)
            pm.Deterministic("z", (theta - mu) / tau)
        else:
            z = pm.Normal("z", 0.0, 1.0, shape=n)
            theta = pm.Deterministic("theta", mu + tau * z)
        pm.Normal("y", theta, se, observed=y_bar)
    return model


def geometry_experiment(obs_per_group=(1, 300), n_groups=8, true_tau=1.0,
                        obs_sigma=1.0, draws=1500, tune=1500, chains=4, seed=0):
    """Fit a hierarchical normal both ways, at a weak and a strong data regime.

    Returns a dict keyed by (obs_per_group, parameterization) holding the draws
    needed to draw the geometry: the group parameter in *its own* coordinates,
    log tau, and the divergence mask.
    """
    import pymc as pm

    rng = np.random.default_rng(seed)
    theta_true = rng.normal(0.0, true_tau, n_groups)

    out = {}
    for n_obs in obs_per_group:
        se = obs_sigma / np.sqrt(n_obs)
        y_bar = rng.normal(theta_true, se)
        for par in ("centered", "non-centered"):
            with _hier_normal(y_bar, np.full(n_groups, se), parameterization=par):
                idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                                  nuts_sampler="pymc", progressbar=False,
                                  random_seed=seed)
            post = idata.posterior.dataset
            # Plot each parameterization in the coordinates the SAMPLER works in:
            # theta for centered, z for non-centered. That is the whole point —
            # the geometry is a property of the coordinates, not the model.
            coord = "theta" if par == "centered" else "z"
            out[(n_obs, par)] = {
                "coord_name": coord,
                "coord": post[coord].values.reshape(-1, n_groups)[:, 0],
                "log_tau": np.log(post["tau"].values.ravel()),
                "diverging": idata.sample_stats["diverging"].values.ravel(),
                "n_divergences": int(idata.sample_stats["diverging"].values.sum()),
            }
    return {"results": out, "obs_per_group": tuple(obs_per_group)}


def fig_geometry_grid(experiment, figsize=(10.5, 7.0)):
    """2x2: parameterization across columns, data strength down rows.

    Top-left funnels downward; bottom-right funnels *upward*. That inversion is
    the reversal, and it is why "always non-center" is wrong.
    """
    import matplotlib.pyplot as plt

    res, obs = experiment["results"], experiment["obs_per_group"]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for r, n_obs in enumerate(obs):
        for c, par in enumerate(("centered", "non-centered")):
            ax, d = axes[r, c], res[(n_obs, par)]
            ax.plot(d["coord"], d["log_tau"], "o", color=S.PRIMARY, ms=1.8,
                    alpha=0.20, ls="none")
            if d["diverging"].any():
                S.divergences(ax, d["coord"][d["diverging"]],
                              d["log_tau"][d["diverging"]], label=None)
            strength = "weak" if n_obs == min(obs) else "strong"
            symbol = r"\theta" if d["coord_name"] == "theta" else "z"
            ax.set(title=f"{par}, {strength} data "
                         f"({n_obs} obs/group, {d['n_divergences']} div.)",
                   xlabel=f"${symbol}_1$",
                   ylabel=r"$\log \tau$" if c == 0 else "")
    fig.suptitle("The pathology swaps corners as the data gets stronger", y=1.0)
    fig.tight_layout()
    return fig


def pooling_summary(result, split_at=30):
    """Numbers to quote alongside the figures."""
    n = result["trial_counts"]
    low = n < split_at
    out = {}
    for key in ("no_pooling", "partial_pooling"):
        err = np.abs(result[key] - result["v_true"])
        out[key] = {"mae_all": err.mean(), "mae_low": err[low].mean(),
                    "mae_high": err[~low].mean()}
    out["low_n_improvement_pct"] = 100 * (
        1 - out["partial_pooling"]["mae_low"] / out["no_pooling"]["mae_low"])
    out["split_at"] = split_at
    return out
