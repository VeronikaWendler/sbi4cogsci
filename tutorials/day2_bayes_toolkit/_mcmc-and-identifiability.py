# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MCMC methods, and when inference falls apart
#
# **Day 2, 15:00 — 30 minutes.** Alexander Fengler.
#
# Everything you fitted this morning and at 14:30 converged quietly. That is
# not guaranteed, and the failures teach more than the successes. This session
# has two halves:
#
# 1. **what MCMC is actually doing** — including writing a sampler from scratch
#    in ten lines, so it stops being a black box;
# 2. **what makes a posterior hard**, on the cognitive model you just fitted.

# %%
import sys, pathlib, warnings, time
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # -> tutorials/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import sbi4cogsci_style as S

S.use()
warnings.filterwarnings("ignore")

RANDOM_SEED = sum(map(ord, "sbi4cogsci-mcmc"))
rng = np.random.default_rng(RANDOM_SEED)

print("pymc", pm.__version__, "| arviz", az.__version__)

# %% [markdown]
# ## 1. What is MCMC actually trying to do?
#
# We have a distribution $\pi(\theta)$ over some parameter space, and we want
# to compute expectations under it:
#
# $$
# \mathbb{E}_{\pi}[f] \;=\; \int f(\theta)\, \pi(\theta) \, d\theta .
# $$
#
# A posterior mean is $f(\theta) = \theta$; a credible interval is a couple of
# quantiles; a posterior predictive is an expectation of the likelihood. All of
# them are integrals against $\pi$.
#
# In more than two or three dimensions those integrals are hopeless
# analytically and hopeless on a grid. So we give up on computing them and
# instead **draw samples** $\theta^{(1)}, \dots, \theta^{(N)} \sim \pi$ and
# average:
#
# $$
# \mathbb{E}_{\pi}[f] \;\approx\; \frac{1}{N} \sum_{i=1}^{N} f\!\left(\theta^{(i)}\right).
# $$
#
# ::: {.callout-note}
# ## $\pi$ does not have to be a posterior
# Nothing above mentions Bayes. MCMC is a general recipe for sampling from *any*
# distribution you can evaluate up to a constant — it is used in statistical
# physics, combinatorial optimisation and rendering. We happen to point it at
# posteriors, and for the first examples below $\pi$ will just be a distribution
# we picked because we know the right answer.
# :::
#
# ### The normalizing constant, and why we can ignore it
#
# For a posterior, Bayes' rule says
#
# $$
# \pi(\theta) \;=\; p(\theta \mid y) \;=\;
# \frac{p(y \mid \theta)\, p(\theta)}{p(y)},
# \qquad p(y) = \int p(y \mid \theta) p(\theta)\, d\theta .
# $$
#
# The numerator is easy: it is the likelihood times the prior, both of which you
# wrote down. The denominator $p(y)$ — the **evidence** — is an integral over
# the whole parameter space, and it is exactly the kind of integral we just
# admitted we cannot do.
#
# The escape is that **every MCMC algorithm only ever looks at ratios**. Write
# the unnormalised density as $\tilde{\pi}(\theta) = p(y \mid \theta)p(\theta)$,
# so $\pi = \tilde{\pi} / p(y)$. Then for any two points,
#
# $$
# \frac{\pi(\theta')}{\pi(\theta)}
# \;=\; \frac{\tilde{\pi}(\theta') / p(y)}{\tilde{\pi}(\theta) / p(y)}
# \;=\; \frac{\tilde{\pi}(\theta')}{\tilde{\pi}(\theta)} .
# $$
#
# **$p(y)$ cancels.** That single cancellation is what makes Bayesian
# computation possible at all: you never need the evidence to *sample* the
# posterior — only to compare whole models against each other.

# %% [markdown]
# ## 2. A Metropolis sampler in ten lines
#
# The oldest MCMC algorithm, and still the clearest. From the current point
# $\theta$:
#
# 1. **propose** a nearby point, $\theta' = \theta + \varepsilon$ with
#    $\varepsilon \sim \text{Normal}(0, s^2)$;
# 2. **accept** it with probability
#    $\alpha = \min\!\left(1,\ \tilde{\pi}(\theta')/\tilde{\pi}(\theta)\right)$;
# 3. if accepted move there, otherwise **stay put and record the current point
#    again**.
#
# Because the proposal is symmetric, that ratio is the whole rule. Step 3 is the
# part people find strange: rejecting does not mean discarding the iteration, it
# means the chain repeats itself — which is how low-density regions end up
# visited proportionally *less*, rather than never.

# %%
def metropolis(log_target, start, n_steps=20_000, step_size=1.0, seed=0):
    """Random-walk Metropolis. `log_target` need only be correct up to a constant."""
    rng = np.random.default_rng(seed)
    theta = np.atleast_1d(np.asarray(start, dtype=float))
    logp = log_target(theta)
    chain, n_accept = np.empty((n_steps, theta.size)), 0

    for i in range(n_steps):
        proposal = theta + rng.normal(0.0, step_size, theta.size)
        logp_prop = log_target(proposal)
        # accept with probability min(1, pi(prop)/pi(theta)) — in logs
        if np.log(rng.uniform()) < logp_prop - logp:
            theta, logp = proposal, logp_prop
            n_accept += 1
        chain[i] = theta

    return chain, n_accept / n_steps


# %% [markdown]
# To show that the normalizing constant genuinely does not matter, target a
# distribution written **without** one. This is a Gaussian mixture whose true
# density we happen to know, so we can check the answer:

# %%
def log_target_mixture(theta):
    """log of an UNNORMALISED two-component mixture. No 1/sqrt(2*pi) anywhere."""
    x = theta[0]
    return np.log(np.exp(-0.5 * ((x - 2.0) / 0.7) ** 2)
                  + 0.6 * np.exp(-0.5 * ((x + 1.5) / 0.5) ** 2))


chain, acc = metropolis(log_target_mixture, start=[0.0], step_size=1.5,
                        n_steps=40_000, seed=RANDOM_SEED)
print(f"acceptance rate {acc:.2f}")

grid = np.linspace(-4, 5, 400)
dens = np.exp([log_target_mixture([g]) for g in grid])
dens /= np.trapezoid(dens, grid)          # normalise only for PLOTTING

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(chain[2000:, 0], bins=90, density=True, color=S.PRIMARY, alpha=0.75,
        label="Metropolis samples")
ax.plot(grid, dens, color=S.TRUTH, ls="--", lw=2, label="true density")
ax.set(title="Sampling a distribution we never normalised",
       xlabel=r"$\theta$", ylabel="density")
ax.legend()
fig.tight_layout()

# %% [markdown]
# Twelve lines of Python, no gradients, no library — and the histogram lands on
# the density. Note that we **never computed the normalizing constant**; the
# sampler only ever saw ratios.
#
# > **Poll.** Our proposal was symmetric: $\theta' = \theta + \text{Normal}(0, s^2)$.
# > What breaks if the proposal is *asymmetric* and we keep this same rule?
# >
# > - **A.** Nothing — the chain still targets $\pi$.
# > - **B.** The chain converges to the wrong distribution.
# > - **C.** The chain still works but mixes more slowly.
# > - **D.** The acceptance rate goes to zero.
#
# <details>
# <summary>Answer</summary>
#
# **B.** With an asymmetric proposal you must include the proposal ratio too —
# the Metropolis–**Hastings** correction,
# $\alpha = \min(1,\ [\tilde{\pi}(\theta')q(\theta \mid \theta')] /
# [\tilde{\pi}(\theta)q(\theta' \mid \theta)])$. Omit it and the chain converges
# happily to something that is not your target, with no warning. Our symmetric
# Gaussian proposal makes $q$ cancel, which is why we could leave it out.
#
# </details>

# %% [markdown]
# ### The knob that decides everything
#
# `step_size` is the whole art of a random-walk sampler. Too small and every
# proposal is accepted but the chain barely moves; too large and almost
# everything is rejected so the chain barely moves. Both failures look like
# "high" or "reasonable" acceptance rates.

# %%
rows = []
for s in [0.05, 0.5, 1.5, 5.0, 20.0]:
    ch, a = metropolis(log_target_mixture, start=[0.0], step_size=s,
                       n_steps=20_000, seed=RANDOM_SEED)
    x = ch[2000:, 0]
    ess = float(az.ess(az.convert_to_datatree({"x": x[None, :]}),
                       var_names=["x"]).x)
    rows.append({"step_size": s, "acceptance": a, "ESS": ess,
                 "ESS/draw": ess / x.size})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

# %% [markdown]
# ::: {.callout-important}
# ## Acceptance rate is not a measure of quality
# The smallest step size has by far the **highest** acceptance rate and among
# the **worst** ESS — it accepts everything because it proposes almost nothing.
# Judge a sampler by effective sample size, never by how often it says yes.
# :::
#
# This is also why gradient-based samplers exist. NUTS does not guess a
# direction and hope; it uses $\nabla \log \tilde{\pi}$ to move along the
# distribution. We rely on that tomorrow — today it is enough to know what it
# is replacing.

# %% [markdown]
# ## 3. When the problem is the *posterior*, not the sampler
#
# Now the cognitive model. At 14:30 everything recovered cleanly. Here are two
# datasets from the **same** DDM, differing only in drift rate — and therefore
# in how often the participant makes an error.

# %%
from hssm.likelihoods import DDM
from ssms import Simulator

DRAWS, TUNE, CHAINS = 700, 700, 2
TRUE = {"v_balanced": 0.5, "v_extreme": 3.0, "a": 1.2, "z": 0.5, "t": 0.3}
PARAMS = ["v", "a", "z", "t"]


def make(v_true, n=600):
    o = Simulator(model="ddm").simulate(theta=[v_true, TRUE["a"], TRUE["z"], TRUE["t"]],
                                        n_samples=n, random_state=RANDOM_SEED)
    return np.column_stack([o["rts"].flatten(), o["choices"].flatten()])


def fit_ddm(observed, seed=RANDOM_SEED):
    with pm.Model():
        v = pm.Normal("v", 0.0, 3.0)
        a = pm.HalfNormal("a", 2.0)
        z = pm.Beta("z", 5.0, 5.0)
        t = pm.HalfNormal("t", 0.5)
        DDM("obs", v=v, a=a, z=z, t=t, observed=observed)
        return pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1,
                         nuts_sampler="pymc", progressbar=False, random_seed=seed)


fits, posteriors = {}, {}
for label, v_true in [("balanced", TRUE["v_balanced"]), ("extreme", TRUE["v_extreme"])]:
    obs = make(v_true)
    err = (obs[:, 1] == -1).mean()
    idata = fit_ddm(obs)
    fits[label] = idata
    p = idata.posterior.dataset
    posteriors[label] = {k: p[k].values.ravel() for k in PARAMS}
    print(f"{label:9s} error rate {err:5.1%}   "
          + "  ".join(f"{k}={posteriors[label][k].mean():5.2f}" for k in PARAMS))
print(f"\ntruth: v=0.5 or 3.0, a={TRUE['a']}, z={TRUE['z']}, t={TRUE['t']}")

# %% [markdown]
# ### The correlation structure is the diagnosis

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, label in zip(axes, ["balanced", "extreme"]):
    d = posteriors[label]
    M = np.array([[np.corrcoef(d[i], d[j])[0, 1] for j in PARAMS] for i in PARAMS])
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set(xticks=range(4), yticks=range(4), title=label)
    ax.set_xticklabels(PARAMS); ax.set_yticklabels(PARAMS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(M[i, j]) > 0.55 else S.TRUTH)
fig.colorbar(im, ax=axes, shrink=0.8, label="posterior correlation")
fig.suptitle("Every parameter pair, both designs", y=1.02)

# %% [markdown]
# In the balanced design the correlations are moderate and *structured*: `a`
# with `t` (both push the RT distribution rightward) and `v` with `z` (both
# push toward one boundary).
#
# In the extreme design **everything correlates with everything**, and the
# strongest pair is not the one people usually name. It is not $v$ with $a$ —
# it is $v$ with $t$, the drift rate against the non-decision time. (Read the
# actual number off the matrix above; how extreme it gets varies from dataset
# to dataset, but the *pattern* does not.)
#
# The mechanism is simple once stated. With essentially no errors, the choices
# carry no information at all — every trial went the same way — so the *only*
# signal left is the response-time distribution. And
#
# $$
# \text{RT} \;=\; \underbrace{t}_{\text{non-decision}}
# \;+\; \underbrace{\text{decision time}}_{\approx\, a / v \text{ for large } v} .
# $$
#
# Raise $t$ and lower $v$ together and total RT is unchanged. That is a
# near-perfect one-dimensional ridge: the data cannot tell the two apart.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, label in zip(axes, ["balanced", "extreme"]):
    d = posteriors[label]
    r = np.corrcoef(d["v"], d["t"])[0, 1]
    ax.plot(d["v"], d["t"], "o", color=S.PRIMARY, ms=2.5, alpha=0.25, ls="none",
            label="posterior draws")
    S.truth_point(ax, TRUE[f"v_{label}"], TRUE["t"])
    ax.set(title=f"{label}:  corr(v, t) = {r:+.2f}",
           xlabel="drift $v$", ylabel="non-decision time $t$")
    ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()

# %% [markdown]
# On the left, a compact blob on the truth. On the right, a long thin ridge —
# and the truth is on it, but so is every other point along the line.
#
# The consequences are severe and easy to miss:

# %%
tab = []
for label in ["balanced", "extreme"]:
    d = posteriors[label]
    tab.append({"design": label,
                **{f"sd({k})": d[k].std() for k in PARAMS},
                "v_hat": d["v"].mean(),
                "corr(v,t)": np.corrcoef(d["v"], d["t"])[0, 1]})
print(pd.DataFrame(tab).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print(f"\ntrue v in the extreme design: {TRUE['v_extreme']}")

# %% [markdown]
# ::: {.callout-important}
# ## High accuracy is bad data for parameter estimation
# This is the counterintuitive headline. Compare the two rows above: the
# near-perfect dataset gives posteriors several times wider on every parameter,
# lying along a ridge rather than filling a blob. Nothing is wrong with the
# sampler and nothing is wrong with the model — **the experiment did not
# collect the information**.
#
# Whether the point estimates also come out *biased* varies from dataset to
# dataset, which is itself worth noticing: on a ridge, where the posterior mean
# lands depends on where the prior and the little remaining information happen
# to pull it. Sometimes you get lucky. You cannot tell from one fit which case
# you are in — that is the problem.
#
# Lüken, Heathcote, Haaf & Matzke (2025, *Psychonomic Bulletin & Review*
# 32(3):1411–1424) study this systematically and recommend designing for error
# rates between **15% and 35%**. Below that, parameters stop being separately
# identifiable — and collecting more trials does not rescue a design with no
# errors in it.
# :::
#
# ### Exercise
#
# We measured `corr(v, t)`. Which pair is strongest in the **balanced** design,
# and does the mechanism make sense to you?
#
# <details>
# <summary>Answer</summary>
#
# `a`–`t` and `v`–`z` — read the exact values off the matrix you just plotted.
#
# Both are interpretable. Boundary separation and non-decision time both make
# responses slower, so raising one and lowering the other keeps mean RT roughly
# fixed — they compete to explain the same feature of the data. Drift and start
# point both push the process toward the upper boundary, so they compete to
# explain the choice proportion.
#
# Notice how far down the list `v`–`a` sits — the pair people most often name is
# not the one doing the damage in either design. Check which parameters actually
# trade off in *your* fit rather than assuming.
#
# </details>

# %% [markdown]
# ## What to take away
#
# ::: {.callout-tip}
# ## The four things that matter
#
# 1. **MCMC turns integrals into averages.** You cannot integrate $\pi$, so you
#    sample from it and average.
# 2. **The normalizing constant cancels.** Every accept/reject decision is a
#    *ratio*, so the evidence $p(y)$ never has to be computed to sample a
#    posterior.
# 3. **Acceptance rate is not quality.** A tiny step size accepts nearly
#    everything and explores nearly nothing. Judge by ESS.
# 4. **Some posteriors are hard because of the data, not the sampler.** A
#    near-degenerate ridge is the experiment's fault, and no sampler fixes it.
# :::
#
# ### Quick reference
#
# | want to | do |
# |---|---|
# | see the trade-off structure | correlation matrix of posterior draws |
# | see a specific trade-off | scatter the two parameters, mark the truth |
# | judge a sampler | `az.ess`, never the acceptance rate |
# | design a study you can fit | aim for **15-35% errors** |
#
# **Next, tomorrow at 11:00:** today's difficulties came from correlation that
# is roughly the same everywhere in parameter space. Hierarchical models bring a
# nastier relative — curvature that *changes as you move* — where the failure
# stops being inefficiency and becomes bias.
