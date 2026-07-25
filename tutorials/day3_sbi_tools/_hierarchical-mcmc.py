# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MCMC for hierarchical Bayesian models
#
# **Day 3, 11:00 — 60 minutes.** Alexander Fengler (with Brandon Turner).
#
# The motivating slides are
# [here](hierarchical-mcmc-slides.qmd). This notebook is the hands-on half.
#
# Yesterday afternoon we saw posteriors that were hard because they were
# **correlated** — a long thin ridge, but the same shape everywhere. Hierarchical
# models bring a nastier relative: **curvature that changes as you move**. No
# single step size works everywhere, and the failure is not slowness. It is
# *bias*.

# %%
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # -> tutorials/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import sbi4cogsci_style as S

S.use()
warnings.filterwarnings("ignore")

RANDOM_SEED = sum(map(ord, "sbi4cogsci-hierarchy"))
rng = np.random.default_rng(RANDOM_SEED)

DRAWS, TUNE, CHAINS = 1000, 1000, 4

print("pymc", pm.__version__, "| arviz", az.__version__)

# %% [markdown]
# ## 1. Neal's funnel — the geometry, before any data
#
# Neal (2003) reduced the whole problem to two lines:
#
# $$v \sim \mathcal{N}(0, 3), \qquad x_i \mid v \sim \mathcal{N}\!\left(0, e^{v/2}\right)$$
#
# There is no data and no likelihood. This is a *prior* — the shape a
# hierarchical model has before the data says anything. `v` plays the role of
# a log population scale, and `x` the role of group-level parameters.

# %%
v_prior = rng.normal(0, 3, 40000)
x_prior = rng.normal(0, np.exp(v_prior / 2))

fig, ax = plt.subplots(figsize=(6.4, 4.4))
keep = np.abs(x_prior) < 25
ax.plot(x_prior[keep], v_prior[keep], "o", color=S.PRIMARY, ms=1.5, alpha=0.15,
        ls="none", label="prior draws")
ax.axhspan(-9, -3, color=S.DIVERGENT, alpha=0.10)
S.annotate(ax, "the neck:\nwidth shrinks like $e^{v/2}$", xy=(0, -6), xytext=(9, -7.5))
ax.set(title="Neal's funnel", xlabel="$x_1$", ylabel="$v$  (log scale)",
       xlim=(-25, 25), ylim=(-9, 9))
ax.legend(loc="upper right")
fig.tight_layout()

# %% [markdown]
# At $v = 2$ the conditional standard deviation of $x$ is $e^{1} \approx 2.7$.
# At $v = -6$ it is $e^{-3} \approx 0.05$ — **fifty times narrower**. A step size
# tuned for the mouth is wildly unstable in the neck; one tuned for the neck
# would take forever to cross the mouth.
#
# ::: {.callout-important}
# ## The failure mode is bias, not slowness
# A sampler that cannot enter the neck does not merely explore it *slowly* — it
# systematically **never goes there**, so every posterior expectation is wrong.
# And $\hat{R}$ cannot see this, because all the chains fail the same way.
# :::

# %% [markdown]
# ## 2. Sampling it, both ways
#
# **Centered** — sample $x$ directly, with its scale depending on $v$:
#
# $$v \sim \mathcal{N}(0,3), \qquad x \sim \mathcal{N}(0, e^{v/2})$$
#
# **Non-centered** — sample a standard normal and rescale it:
#
# $$v \sim \mathcal{N}(0,3), \qquad \tilde{x} \sim \mathcal{N}(0,1),
#   \qquad x = e^{v/2}\,\tilde{x}$$
#
# These describe the *same distribution*. They are different **coordinate
# systems** for it, and the sampler only ever sees the coordinates.

# %%
def funnel_centered(dim=1):
    with pm.Model() as m:
        v = pm.Normal("v", 0.0, 3.0)
        pm.Normal("x", 0.0, pm.math.exp(v / 2), shape=dim)
    return m


def funnel_noncentered(dim=1):
    with pm.Model() as m:
        v = pm.Normal("v", 0.0, 3.0)
        x_tilde = pm.Normal("x_tilde", 0.0, 1.0, shape=dim)
        pm.Deterministic("x", pm.math.exp(v / 2) * x_tilde)
    return m


def sample(model, seed=RANDOM_SEED, **kw):
    with model:
        return pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1,
                         nuts_sampler="pymc", progressbar=False,
                         random_seed=seed, **kw)


idata_c = sample(funnel_centered())
idata_n = sample(funnel_noncentered())

for name, idata in [("centered", idata_c), ("non-centered", idata_n)]:
    div = int(idata.sample_stats["diverging"].values.sum())
    v = idata.posterior.dataset["v"].values
    print(f"{name:13s} divergences {div:5d}   "
          f"min v reached {v.min():6.2f}   mean v {v.mean():+.3f}  (true mean 0)")

# %% [markdown]
# ### The plot that shows it

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True, sharey=True)
for ax, (name, idata) in zip(axes, [("centered", idata_c), ("non-centered", idata_n)]):
    post = idata.posterior.dataset
    v = post["v"].values.ravel()
    x = post["x"].values.reshape(v.size, -1)[:, 0]
    div = idata.sample_stats["diverging"].values.ravel()
    ax.plot(x, v, "o", color=S.PRIMARY, ms=2, alpha=0.25, ls="none", label="draws")
    if div.any():
        S.divergences(ax, x[div], v[div])
    ax.set(title=f"{name}  ({int(div.sum())} divergences)",
           xlabel="$x_1$", xlim=(-12, 12), ylim=(-9, 9))
    ax.legend(loc="upper right")
axes[0].set_ylabel("$v$")
fig.tight_layout()

# %% [markdown]
# The divergences are not scattered at random — they **cluster in the neck**,
# which is precisely the region the centered chain then under-samples.
#
# > **Poll.** The centered run reports $\hat{R} = 1.00$ for `v`. What does that
# > tell you?
# >
# > **A.** The chains converged; the divergences are a performance warning.
# > **B.** Nothing useful — all four chains can fail in the same way.
# > **C.** It means the model is correctly specified.
# > **D.** It means we need more draws.
#
# <details>
# <summary>Answer</summary>
#
# **B.** $\hat{R}$ compares chains to each other. If every chain stops at the
# same place for the same geometric reason, they agree beautifully — about the
# wrong answer. This is why divergences are a *separate* diagnostic and why
# "$\hat{R}$ is fine" is not a clean bill of health.
#
# </details>

# %% [markdown]
# ### Proving it is bias
#
# Efficiency problems shrink as you run longer. Bias does not. Track the running
# mean of `v` — we know the true value is 0.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for name, idata, colour in [("centered", idata_c, S.NAIVE),
                            ("non-centered", idata_n, S.PRIMARY)]:
    v = idata.posterior.dataset["v"].values[0]
    ax.plot(np.cumsum(v) / np.arange(1, v.size + 1), color=colour, label=name)
S.truth_line(ax, 0.0, label="true E[v] = 0")
ax.set(title="Running mean of $v$ (chain 0)", xlabel="draw", ylabel=r"$\hat{E}[v]$")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The centered chain settles at the wrong value and **stays** there. Running it
# ten times longer moves it ten times more slowly toward nothing in particular.

# %% [markdown]
# ### What about just raising `target_accept`?

# %%
rows = []
for ta in [0.8, 0.95, 0.99]:
    idata = sample(funnel_centered(), nuts={"target_accept": ta})
    v = idata.posterior.dataset["v"].values
    rows.append({"target_accept": ta,
                 "divergences": int(idata.sample_stats["diverging"].values.sum()),
                 "min v reached": v.min(),
                 "mean v (true 0)": v.mean()})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

# %% [markdown]
# Divergences drop. The estimate does not become correct. **Raising
# `target_accept` suppresses the symptom, not the geometry** — it is a
# diagnostic aid, not a fix. The fix is to change coordinates.

# %% [markdown]
# ## 3. Non-centered is not always better
#
# This is the part most tutorials get wrong. The choice depends on whether the
# **prior** or the **likelihood** dominates each group's posterior
# (Papaspiliopoulos, Roberts & Sköld 2007; Betancourt & Girolami 2013):
#
# - **little data per group** → prior dominates → **non-centered** wins
# - **lots of data per group** → likelihood dominates → **centered** wins
#
# Let us find the crossover instead of taking it on faith. A standard
# hierarchical normal, sweeping the number of observations per group.

# %%
N_GROUPS, TRUE_MU, TRUE_TAU, OBS_SIGMA = 8, 0.0, 1.0, 1.0


def make_groups(obs_per_group, seed):
    g = np.random.default_rng(seed)
    theta = g.normal(TRUE_MU, TRUE_TAU, N_GROUPS)
    y = g.normal(theta[:, None], OBS_SIGMA, (N_GROUPS, obs_per_group))
    return y.mean(axis=1), OBS_SIGMA / np.sqrt(obs_per_group)


def hier_centered(y_bar, se):
    with pm.Model() as m:
        mu = pm.Normal("mu", 0.0, 5.0)
        tau = pm.HalfNormal("tau", 5.0)
        theta = pm.Normal("theta", mu, tau, shape=N_GROUPS)
        pm.Normal("y", theta, se, observed=y_bar)
    return m


def hier_noncentered(y_bar, se):
    with pm.Model() as m:
        mu = pm.Normal("mu", 0.0, 5.0)
        tau = pm.HalfNormal("tau", 5.0)
        z = pm.Normal("z", 0.0, 1.0, shape=N_GROUPS)
        theta = pm.Deterministic("theta", mu + tau * z)
        pm.Normal("y", theta, se, observed=y_bar)
    return m


sweep = []
for obs in [1, 3, 10, 30, 100]:
    y_bar, se = make_groups(obs, seed=RANDOM_SEED + obs)
    for label, builder in [("centered", hier_centered),
                           ("non-centered", hier_noncentered)]:
        idata = sample(builder(y_bar, se))
        grads = idata.sample_stats["n_steps"].values.sum() \
            if "n_steps" in idata.sample_stats else np.nan
        ess = float(az.ess(idata, var_names=["tau"]).tau)
        sweep.append({"obs/group": obs, "param": label,
                      "divergences": int(idata.sample_stats["diverging"].values.sum()),
                      "ESS(tau)": ess,
                      "ESS per 1k grads": 1000 * ess / grads if grads == grads else np.nan})

sweep = pd.DataFrame(sweep)
print(sweep.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9))
for label, colour in [("centered", S.NAIVE), ("non-centered", S.PRIMARY)]:
    sub = sweep[sweep.param == label]
    ax1.plot(sub["obs/group"], sub["divergences"], "o-", color=colour, label=label)
    ax2.plot(sub["obs/group"], sub["ESS per 1k grads"], "o-", color=colour, label=label)
ax1.set(title="Divergences", xlabel="observations per group", xscale="log",
        ylabel="count")
ax1.legend()
ax2.set(title="Efficiency", xlabel="observations per group", xscale="log",
        ylabel=r"ESS($\tau$) per 1k gradients")
ax2.legend()
fig.tight_layout()

# %% [markdown]
# ::: {.callout-note}
# ## The rule, stated properly
# "Always non-center hierarchical models" is the single most widespread piece of
# wrong advice in applied Bayesian work. The correct statement is: **non-center
# the parameters whose groups are data-poor.** With enough data per group the
# centered form is better behaved, because then the likelihood — not the prior —
# is what shapes each group's posterior.
#
# Which is why you want the choice to be **per parameter**.
# :::

# %% [markdown]
# ## 4. Per-parameter parameterization in HSSM
#
# A real SSM has parameters of both kinds in the same model. Drift `v` is often
# estimated from plenty of trials per participant; a boundary or non-decision
# time may be far more weakly identified. HSSM lets you choose independently.

# %%
import hssm

data = hssm.load_data("cavanagh_theta")
print(data.head(3).to_string(index=False))
print("\nparticipants:", data["participant_id"].nunique(), " trials:", len(data))

# %% [markdown]
# `noncentered` is not an HSSM argument — it passes straight through to
# `bmb.Model`. It takes a bool **or a dict keyed by parameter**, and a
# per-prior setting overrides the dict:
#
# ```
# resolution order:  prior.noncentered  >  noncentered[param]  >  True
# ```

# %%
def build(noncentered):
    return hssm.HSSM(
        data=data,
        model="ddm",
        noncentered=noncentered,
        include=[
            {"name": "v", "formula": "v ~ 1 + (1|participant_id)"},
            {"name": "a", "formula": "a ~ 1 + (1|participant_id)"},
        ],
        p_outlier=0.05,          # explicit: this is the default, not "off"
        prior_settings="safe",
    )


mixed = build({"v": False, "a": True})
mixed.build()

offsets = sorted(n for n in mixed.pymc_model.named_vars if n.endswith("_offset"))
print("nodes carrying an _offset (i.e. non-centered):")
for n in offsets:
    print("   ", n)

# %% [markdown]
# The `_offset` nodes are the structural fingerprint of the non-centered
# parameterization: `u_g = z_g · σ`. `a` has one, `v` does not — exactly what we
# asked for. You can read the parameterization off the graph without sampling
# anything.

# %% [markdown]
# ::: {.callout-warning}
# ## Two ways this bites
# **Non-centering only works for `Normal` priors whose `sigma` is itself a
# random variable.** Anything else raises `NotImplementedError` when the model
# is built — loud, at least.
#
# **The quiet one:** a `Normal` group prior with a nested `mu` hyperprior under
# non-centering leaves `mu` as a **disconnected free variable** — sampled, but
# influencing nothing. HSSM 0.4.0 ships detectors for this
# (`check_user_priors_against_parameterization`, `find_disconnected_free_rvs`).
# Always check `print(model)` / `model.graph()` after changing parameterization,
# because term-prior keys differ between the two forms and a mismatched key is
# **silently dropped**.
# :::

# %% [markdown]
# ### Exercise
#
# Build the same model with `noncentered=True` and with `noncentered=False`, and
# confirm from the graph alone (no sampling) which nodes change. Then predict —
# before running anything — which setting you would want for `v` in the Cavanagh
# data, given its 3,988 trials spread over 14 participants.
#
# <details>
# <summary>Solution and reasoning</summary>
#
# ```python
# for setting in [True, False]:
#     m = build(setting); m.build()
#     offs = sorted(n for n in m.pymc_model.named_vars if n.endswith("_offset"))
#     print(setting, "->", offs)
# ```
#
# With roughly 285 trials per participant, the drift rate is well identified
# *within* each participant — the likelihood dominates, so **centered** is the
# better choice for `v`. Parameters that are weakly constrained per participant,
# or a group scale estimated from only 14 groups, are the ones that want
# non-centering. This is the crossover from section 3, in a real model.
#
# </details>

# %% [markdown]
# ## 5. Where this goes next
#
# Centered and non-centered are the two endpoints of a continuum. **VIP**
# (variationally inferred parametrization; Gorinova, Moore & Hoffman, ICML 2020)
# learns a per-variable $\lambda \in [0,1]$:
#
# $$\theta = \mu + \sigma^{1-\lambda}\left(\eta - \lambda\mu\right)$$
#
# so $\lambda = 0$ is centered, $\lambda = 1$ is non-centered, and the optimiser
# picks the point in between for **each** variable. It is available in PyMC as
# `pymc_extras.model.transforms.autoreparam.vip_reparametrize`, and the paper
# shows learned mixed parameterizations beating *both* fixed extremes.
#
# ## What to take away
#
# - A hierarchical posterior has **position-dependent curvature**. That is a
#   different problem from correlation, and it needs a different fix.
# - Divergences mean **bias**, not slowness. $\hat{R}$ and ESS cannot see it.
# - Raising `target_accept` removes the warning, not the cause.
# - **Non-centered is not universally better.** Data-poor groups want
#   non-centered; data-rich groups want centered.
# - In HSSM, choose per parameter with `noncentered={"v": False, "a": True}`,
#   and verify it from the `_offset` nodes in the graph.

# %% [markdown]
# ### References
#
# - Neal, R. (2003). Slice sampling. *Annals of Statistics* 31(3), 705–767.
# - Papaspiliopoulos, Roberts & Sköld (2007). A general framework for the
#   parametrization of hierarchical models. *Statistical Science* 22(1), 59–73.
# - Betancourt & Girolami (2013). Hamiltonian Monte Carlo for hierarchical
#   models. [arXiv:1312.0906](https://arxiv.org/abs/1312.0906)
# - Betancourt (2017). Diagnosing biased inference with divergences.
# - Gorinova, Moore & Hoffman (2020). Automatic reparameterisation of
#   probabilistic programs. *ICML*.
