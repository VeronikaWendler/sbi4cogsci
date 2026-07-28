"""Figures for the hierarchical-modelling material (Day 3, 11:00).

Written once, used three ways:

1. imported by `day3_sbi_tools/_src/hierarchical-mcmc.py` (the notebook),
2. baked to PNG for `hierarchical-mcmc-slides.qmd` by
   `day3_sbi_tools/_src/bake_slide_figures.py`,
3. ready for marimo — see the shape rule below.

**The shape rule.** Expensive *computation* and cheap *plotting* are separate
functions. A `fig_*` function takes already-computed results plus a few scalar
display parameters and returns a Figure. That is what makes these drop into a
marimo cell with `mo.ui.slider` without re-running an MCMC fit on every drag:

    result = pooling_experiment(...)          # once, in its own cell
    fig_shrinkage(result, highlight_n=n.value)  # re-runs on every slider move

**Why these create a figure and immediately close it.** A figure left open in
pyplot's global list gets rendered *twice* by a Jupyter cell ending in
`fig_shrinkage(result)`: once because the inline backend flushes every open
figure at the end of the cell (`display_data`), and again because the returned
Figure is the cell's value (`execute_result`). Closing it removes the first,
leaving the return value as the only thing displayed.

The figure is still created *through* pyplot rather than as a bare
`Figure()`, and that part is not optional. IPython only registers a renderer
for `matplotlib.figure.Figure` once the inline backend has been initialised by
a first pyplot call. A bare Figure returned before that has ever happened
displays as the text `<Figure size 792x484 with 1 Axes>` and **no image at
all** — which depends on what other cells ran first, so it fails silently and
inconsistently. Going through `plt.subplots` guarantees the backend is live.

`savefig` is unaffected by the close, which is what the slide baker relies on.

Colours come from `sbi4cogsci_style`, so a colour means the same thing here as
in every other session.
"""

from __future__ import annotations

import numpy as np

import sbi4cogsci_style as S


def _new_figure(figsize, nrows=1, ncols=1):
    """A Figure plus its Axes, detached from pyplot's global figure list.

    Created through pyplot so the inline backend is initialised, then closed so
    the end-of-cell flush does not display it a second time. See the module
    docstring — both halves are load-bearing.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    plt.close(fig)
    return fig, axes


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
    fig = None
    if ax is None:
        fig, ax = _new_figure((6.4, 4.4))
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


def fig_shrinkage(result, *, split_at=30, ax=None, ylabel="estimated drift $v$",
                  title="Shrinkage: who gets moved, and how far"):
    """Each participant as an arrow from its no-pooling to its partial-pooling
    estimate, against how many trials they contributed.

    The lesson is visual: participants with little data get pulled a long way
    toward the population; participants with a lot barely move.

    Works on any result dict carrying `trial_counts`, `v_true`, `no_pooling` and
    `partial_pooling` — which is why the regression capstone reuses it rather
    than defining a second near-identical plot.
    """
    fig = None
    if ax is None:
        fig, ax = _new_figure((7.2, 4.4))

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
           ylabel=ylabel, title=title)
    ax.legend(loc="lower right", fontsize=9, ncol=3)
    if fig is not None:
        fig.tight_layout()
    return fig if fig is not None else ax.figure


def fig_pooling_error(result, *, split_at=30, ax=None):
    """Absolute error against trial count, for both fits."""
    fig = None
    if ax is None:
        fig, ax = _new_figure((7.2, 4.0))
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
# The capstone: a per-participant regression on a continuous covariate
# --------------------------------------------------------------------------


def simulate_regression_panel(trial_counts=DEFAULT_TRIAL_COUNTS,
                              mu_b0=0.9, tau_b0=0.25,
                              mu_b1=1.1, tau_b1=0.30,
                              a=1.2, z=0.5, t=0.3, seed=0):
    """An unbalanced panel where each participant has their own *slope*.

    Drift varies within participant with a continuous difficulty covariate:

        v_gi = b0_g + b1_g * difficulty_i,     b0_g, b1_g ~ Normal(mu, tau)

    This is the realistic shape of a cognitive experiment, and it is harder
    than estimating one drift per person: a slope needs both enough trials
    *and* spread in the covariate.
    """
    from ssms.basic_simulators.simulator import simulator

    rng = np.random.default_rng(seed)
    counts = np.asarray(trial_counts, dtype=int)
    b0_true = rng.normal(mu_b0, tau_b0, counts.size)
    b1_true = rng.normal(mu_b1, tau_b1, counts.size)

    obs, idx, diff = [], [], []
    for g, n_g in enumerate(counts):
        d = rng.uniform(-1.0, 1.0, int(n_g))
        v = b0_true[g] + b1_true[g] * d
        theta = np.column_stack([v, np.full(int(n_g), a),
                                 np.full(int(n_g), z), np.full(int(n_g), t)])
        # theta as a (n_trials, n_params) matrix with n_samples=1 is how
        # ssm-simulators does trial-varying parameters.
        out = simulator(theta=theta, model="ddm", n_samples=1, random_state=seed + g)
        obs.append(np.column_stack([out["rts"].flatten(), out["choices"].flatten()]))
        idx.append(np.full(int(n_g), g))
        diff.append(d)

    return {"observed": np.vstack(obs), "participant_idx": np.concatenate(idx),
            "difficulty": np.concatenate(diff), "trial_counts": counts,
            "b0_true": b0_true, "b1_true": b1_true}


def fit_regression(panel, *, pooling, a=1.2, z=0.5, t=0.3,
                   draws=1000, tune=1000, chains=4, seed=0):
    """Fit per-participant intercept and slope, with or without pooling."""
    import pymc as pm
    from hssm.likelihoods import DDM

    g_idx = panel["participant_idx"]
    n_p = panel["trial_counts"].size
    with pm.Model():
        if pooling == "none":
            b0 = pm.Normal("b0", 0.0, 2.0, shape=n_p)
            b1 = pm.Normal("b1", 0.0, 2.0, shape=n_p)
        elif pooling == "partial":
            mu0 = pm.Normal("mu_b0", 0.0, 2.0)
            tau0 = pm.HalfNormal("tau_b0", 1.0)
            mu1 = pm.Normal("mu_b1", 0.0, 2.0)
            tau1 = pm.HalfNormal("tau_b1", 1.0)
            b0 = pm.Deterministic("b0", mu0 + tau0 * pm.Normal("z0", 0, 1, shape=n_p))
            b1 = pm.Deterministic("b1", mu1 + tau1 * pm.Normal("z1", 0, 1, shape=n_p))
        else:
            raise ValueError("pooling must be 'none' or 'partial'")
        v = b0[g_idx] + b1[g_idx] * panel["difficulty"]
        DDM("obs", v=v, a=a, z=z, t=t, observed=panel["observed"])
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          nuts_sampler="pymc", progressbar=False, random_seed=seed)
    return idata


def regression_experiment(seed=0, **kwargs):
    """The capstone comparison.

    Returns two result dicts — one for the intercept, one for the slope — each
    in the same shape `fig_shrinkage` / `fig_pooling_error` already understand.
    """
    panel = simulate_regression_panel(seed=seed, **kwargs)
    n_p = panel["trial_counts"].size
    est = {}
    for pooling in ("none", "partial"):
        idata = fit_regression(panel, pooling=pooling, seed=seed)
        post = idata.posterior.dataset
        est[pooling] = {k: post[k].values.reshape(-1, n_p).mean(0) for k in ("b0", "b1")}

    def pack(key, truth):
        return {"v_true": panel[truth], "trial_counts": panel["trial_counts"],
                "no_pooling": est["none"][key],
                "partial_pooling": est["partial"][key],
                "n_trials_total": int(panel["trial_counts"].sum())}

    return {"intercept": pack("b0", "b0_true"), "slope": pack("b1", "b1_true"),
            "n_trials_total": int(panel["trial_counts"].sum())}


# --------------------------------------------------------------------------
# The two parameterizations, and the fact that their advantage reverses
# --------------------------------------------------------------------------


#: The eight-schools data (Rubin 1981). The canonical case where the funnel
#: actually *manifests*: the school effects are small relative to their standard
#: errors, so the posterior for tau extends right down to zero and the neck is
#: reachable. Simulated panels where tau is genuinely non-zero push tau upward
#: and the neck never appears.
EIGHT_SCHOOLS_Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
EIGHT_SCHOOLS_SE = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

#: More groups makes the funnel *worse*, and that is the point of using more
#: than eight. Every group contributes one theta that has to fit through the
#: same neck, so the centered parameterization has to thread N coordinates
#: simultaneously rather than eight. Measured on this panel, weak likelihood,
#: centered, as the group count rises 8 -> 16 -> 32 -> 64:
#:
#:     divergences   346 ->  333 ->  670 -> 1038
#:     ESS(tau)      390 ->  163 ->   67 ->   11
#:
#: while the non-centered fit stays healthy throughout (ESS(tau) 4577 -> 2689).
DEFAULT_N_GROUPS = 32


def simulate_school_panel(n_groups=DEFAULT_N_GROUPS, seed=0):
    """A panel with the *eight-schools character*, at any number of groups.

    The property that matters is not the specific numbers, it is that each
    group's effect is small next to its own standard error. That is what lets
    the posterior for tau reach zero, which is what makes the neck reachable.
    So: standard errors spanning the same range as Rubin's, and a true tau of
    zero.

    Returns the real Rubin (1981) data unchanged when `n_groups == 8`, so the
    canonical case is still exactly the canonical case.
    """
    if n_groups == 8:
        return EIGHT_SCHOOLS_Y.copy(), EIGHT_SCHOOLS_SE.copy()
    rng = np.random.default_rng(seed)
    se = rng.uniform(EIGHT_SCHOOLS_SE.min(), EIGHT_SCHOOLS_SE.max(), n_groups)
    return rng.normal(8.0, se), se


def _hier_normal(y, se, *, parameterization):
    """Hierarchical normal, centered or non-centered.

    A half-normal prior on tau, not a log-normal: a log-normal suppresses both
    zero and infinity and so hides the very geometry we are trying to show
    (Betancourt, *Hierarchical Modeling*, 2020, §4.1).
    """
    import pymc as pm

    n = y.size
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 5.0)
        tau = pm.HalfNormal("tau", 5.0)
        if parameterization == "centered":
            pm.Normal("theta", mu, tau, shape=n)
        else:
            z = pm.Normal("z", 0.0, 1.0, shape=n)
            pm.Deterministic("theta", mu + tau * z)
        pm.Normal("y", model["theta"], se, observed=y)
    return model


def geometry_experiment(se_scales=(1.0, 0.05), draws=2000, tune=2000,
                        chains=4, seed=0, n_groups=DEFAULT_N_GROUPS):
    """Fit a schools-like panel both ways, at two data strengths.

    `se_scales` multiplies the standard errors: 1.0 is the weak-likelihood case
    (prior dominates) and a small value makes each group's own estimate precise
    (likelihood dominates). Varying the observation scale is exactly the
    manipulation Betancourt & Girolami use for their Figure 8.

    `n_groups` defaults to more than eight because the funnel sharpens with the
    group count — see `DEFAULT_N_GROUPS`. Pass 8 for Rubin's original data.
    """
    import pymc as pm
    import arviz as az

    y, se_base = simulate_school_panel(n_groups, seed=seed)
    out = {}
    for scale in se_scales:
        se = se_base * scale
        for par in ("centered", "non-centered"):
            with _hier_normal(y, se, parameterization=par):
                idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                                  nuts_sampler="pymc", progressbar=False,
                                  random_seed=seed)
            post = idata.posterior.dataset
            # Plot each parameterization in the coordinates its own SAMPLER
            # works in — theta for centered, z for non-centered. The geometry is
            # a property of the coordinates, not of the model.
            coord = "theta" if par == "centered" else "z"
            log_tau = np.log(post["tau"].values.ravel())
            out[(scale, par)] = {
                "coord_name": coord,
                "coord": post[coord].values.reshape(log_tau.size, -1)[:, 0],
                "log_tau": log_tau,
                "diverging": idata.sample_stats["diverging"].values.ravel(),
                "n_divergences": int(idata.sample_stats["diverging"].values.sum()),
                "min_log_tau": float(log_tau.min()),
                # ESS(tau) is the number that exposes the weak-data failure most
                # sharply — divergences say something is wrong, this says how
                # much of the chain was actually worth having.
                "ess_tau": float(az.ess(idata, var_names=["tau"]).tau),
            }
    return {"results": out, "se_scales": tuple(se_scales), "n_groups": n_groups}


def fig_geometry_grid(experiment, figsize=(11.0, 7.2)):
    """2x2: parameterization across columns, likelihood strength down rows.

    The y-axis is **shared within each row**, which is the whole point: in the
    weak-data row the centered chain simply stops, while the non-centered one
    carries on orders of magnitude further down. You are looking for where the
    blue points *end*, not only for their shape.
    """
    res, scales = experiment["results"], experiment["se_scales"]
    fig, axes = _new_figure(figsize, 2, 2)
    for r, scale in enumerate(scales):
        row = [res[(scale, p)] for p in ("centered", "non-centered")]
        lo = min(d["log_tau"].min() for d in row) - 0.3
        hi = max(d["log_tau"].max() for d in row) + 0.3
        for c, par in enumerate(("centered", "non-centered")):
            ax, d = axes[r, c], res[(scale, par)]
            ax.plot(d["coord"], d["log_tau"], "o", color=S.PRIMARY, ms=1.8,
                    alpha=0.18, ls="none")
            if d["diverging"].any():
                S.divergences(ax, d["coord"][d["diverging"]],
                              d["log_tau"][d["diverging"]], label=None)
            ax.set_ylim(lo, hi)                      # shared within the row
            strength = "weak likelihood" if scale == max(scales) else "strong likelihood"
            symbol = r"\theta" if d["coord_name"] == "theta" else "z"
            ax.set(title=f"{par} — {strength}\n"
                         f"{d['n_divergences']} divergences, "
                         f"reaches $\\log\\tau$ = {d['min_log_tau']:.1f}",
                   xlabel=f"${symbol}_1$",
                   ylabel=r"$\log \tau$" if c == 0 else "")
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
