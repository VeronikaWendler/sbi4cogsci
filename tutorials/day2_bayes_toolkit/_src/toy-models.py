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
# # Toy models for parameter estimation
#
# **Day 2, 14:30 — 30 minutes.** Alexander Fengler.
#
# This session is the bridge. At 12:00 you built models in PyMC. At 14:00 you
# simulated from cognitive models. Now we put the two together: **a
# drift-diffusion likelihood dropped straight into a PyMC model**, fitted the
# way you already know how to fit things.
#
# We will not explain *how* the DDM likelihood works — that is tomorrow. Today
# it is simply a distribution you can use, exactly like `pm.Normal`.
#
# The shape of this session is a *miniature scientific workflow*.

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

from ssms import Simulator
from hssm.likelihoods import DDM        # <- the DDM as a PyMC distribution
from hssm.likelihoods import logp_ddm   # the same likelihood, callable directly

RANDOM_SEED = sum(map(ord, "sbi4cogsci-toy"))
rng = np.random.default_rng(RANDOM_SEED)

# Four chains, run in parallel. `r_hat` compares variance *between* chains to
# variance *within* them, so two chains make it a weak test — ArviZ warns about
# exactly this. Four chains on four cores also finishes sooner than two chains
# run one after another, so this is cheaper as well as better.
DRAWS, TUNE, CHAINS = 800, 800, 4
CORES = 4

print("pymc", pm.__version__, "| arviz", az.__version__)

# %% [markdown]
# ## 1. The data
#
# A two-alternative decision task. Every trial has two design features:
#
# - **coherence** — how much evidence the stimulus carries: `low`, `medium`, `high`
# - **emphasis** — the instruction given: respond `speed`-fast or `accuracy`-carefully
#
# 250 trials in each of the six cells, so 500 trials per coherence level and
# 1500 in total. This is a synthetic dataset: something specific generated it,
# and by the end of the session you will know what.

# %%
CONDITIONS = ["low", "medium", "high"]
EMPHASES = ["speed", "accuracy"]
N_PER_CELL = 250

# --- the generating process (do not peek at the values until section 5) ------
_V_BY_COHERENCE = {"low": 0.35, "medium": 0.85, "high": 1.5}
_A_BY_EMPHASIS = {"speed": 0.9, "accuracy": 1.6}
_Z_TRUE, _T_TRUE = 0.5, 0.30

rows = []
for i, coh in enumerate(CONDITIONS):
    for j, emp in enumerate(EMPHASES):
        theta = [_V_BY_COHERENCE[coh], _A_BY_EMPHASIS[emp], _Z_TRUE, _T_TRUE]
        # Seed from the cell's *position*, not `hash((coh, emp))`: Python salts
        # string hashes per process (PYTHONHASHSEED), so a hash-derived seed
        # silently draws a different dataset on every run.
        out = Simulator(model="ddm").simulate(
            theta=theta, n_samples=N_PER_CELL,
            random_state=RANDOM_SEED + 10 * i + j)
        rows.append(pd.DataFrame({
            "rt": out["rts"].flatten(),
            "response": out["choices"].flatten().astype(int),
            "coherence": coh,
            "emphasis": emp,
        }))

data = pd.concat(rows, ignore_index=True)
data["coherence"] = pd.Categorical(data["coherence"], categories=CONDITIONS, ordered=True)
data["emphasis"] = pd.Categorical(data["emphasis"], categories=EMPHASES)
print(data.head())
print(f"\n{len(data)} trials")

# %% [markdown]
# ### Look before you fit
#
# Two summaries answer most of the question: how *accurate* is each cell, and
# how *fast*?

# %%
summary = (data.assign(correct=lambda d: d["response"] == 1)
               .groupby(["coherence", "emphasis"], observed=True)
               .agg(accuracy=("correct", "mean"), mean_rt=("rt", "mean"))
               .round(3))
print(summary.to_string())

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(len(CONDITIONS))
for emp, colour in [("speed", S.PRIMARY), ("accuracy", S.NAIVE)]:
    sub = summary.xs(emp, level="emphasis")
    ax1.plot(x, sub["accuracy"], "o-", color=colour, label=emp)
    ax2.plot(x, sub["mean_rt"], "o-", color=colour, label=emp)
for ax, ylab, title in [(ax1, "P(correct)", "Accuracy"), (ax2, "mean RT (s)", "Speed")]:
    ax.set(xticks=x, xlabel="coherence", ylabel=ylab, title=title)
    ax.set_xticklabels(CONDITIONS)
    ax.legend()
S.truth_line(ax1, 0.5, label="chance")
fig.tight_layout()

# %% [markdown]
# Two patterns:
#
# - Accuracy climbs steeply with coherence, and barely moves with emphasis.
# - Mean RT is dominated by emphasis, and moves comparatively little with
#   coherence.
#
# A model with one drift rate and one boundary for the whole dataset cannot
# produce that. Something has to vary — the question is *what*.
#
# > **Poll.** Before fitting: which parameter should coherence affect?
# >
# > - **A.** Boundary separation $a$ — harder stimuli need more evidence.
# > - **B.** Drift rate $v$ — coherence is how fast evidence accumulates.
# > - **C.** Non-decision time $t$ — harder stimuli take longer to encode.
# > - **D.** Start point $z$ — coherence biases you toward one response.
#
# <details>
# <summary>Answer</summary>
#
# **B.** Drift rate is the *quality of evidence per unit time*, which is
# exactly what a coherence manipulation changes. Boundary separation is how
# much evidence you demand before committing — that is under the participant's
# strategic control, which is what a speed/accuracy instruction manipulates.
# Keep that mapping: **stimulus → drift, instruction → boundary.**
#
# </details>

# %% [markdown]
# ## 2. The DDM as a PyMC distribution
#
# Here is the whole bridge:
#
# ```python
# from hssm.likelihoods import DDM
# DDM("obs", v=..., a=..., z=..., t=..., observed=observed)
# ```
#
# It behaves like any other PyMC distribution, with two differences:
#
# 1. its `observed` data is **two columns**, `[rt, response]`, not one, and
# 2. `response` must be coded **`-1` / `+1`**.
#
# The model we fit first is the simplest possible one — a single set of
# parameters for all 1500 trials:
#
# $$
# v \sim \text{Normal}(0, 3), \quad a \sim \text{HalfNormal}(2), \quad
# z \sim \text{Beta}(5,5), \quad t \sim \text{HalfNormal}(0.5),
# $$
# $$
# (\text{rt}_i, \text{resp}_i) \sim \text{DDM}(v,\ a,\ z,\ t).
# $$

# %%
observed = np.column_stack([data["rt"].to_numpy(), data["response"].to_numpy()])
coh_idx = data["coherence"].cat.codes.to_numpy()
emp_idx = data["emphasis"].cat.codes.to_numpy()

COORDS = {"coherence": CONDITIONS, "emphasis": EMPHASES}
PARAMS = ["v", "a", "z", "t"]

# Where to start `t`. See "A trap worth knowing about" below for why this is
# not optional: PyMC's default starting value for this prior lands *above* the
# fastest response time in the dataset, where the likelihood is undefined.
T_INIT = 0.1


def fit(model, seed=RANDOM_SEED):
    with model:
        # `initvals` starts `t` somewhere the likelihood is actually defined —
        # see "A trap worth knowing about" below. Note it goes here, on
        # `sample`, and *not* as `initval=` on the distribution: the latter
        # marks the model as having non-default initial values, which makes
        # `log_likelihood=True` (and therefore `az.loo` in section 4) fail with
        # "Cannot convert models with non-default initial_values".
        return pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CORES,
                         nuts_sampler="pymc", progressbar=False, random_seed=seed,
                         initvals={"t": np.array(T_INIT)},
                         idata_kwargs={"log_likelihood": True})


with pm.Model(coords=COORDS) as m1_flat:
    # Priors
    v = pm.Normal("v", 0.0, 3.0)
    a = pm.HalfNormal("a", 2.0)
    z = pm.Beta("z", 5.0, 5.0)
    t = pm.HalfNormal("t", 0.5)
    # Likelihood
    DDM("obs", v=v, a=a, z=z, t=t, observed=observed)

idata_flat = fit(m1_flat)
print(az.summary(idata_flat, var_names=PARAMS, kind="all").to_string())

# %% [markdown]
# ### Always look at the chains, not just the summary
#
# `kind="all"` is deliberate: it prints `r_hat` and `ess_bulk` next to the
# estimates. Read those two **first**. `r_hat` compares the variance between
# chains to the variance within them, so anything above about `1.01` says the
# chains disagree and the mean beside it is not a posterior mean of anything.
# `ess_bulk` is how many independent draws your correlated ones are worth.
#
# But both collapse a whole distribution into one number, and neither shows you
# its *shape*. Two plots do, and we will run **the same two after every fit** in
# this notebook so they can be compared at a glance:
#
# 1. **Marginals and traces.** The trace should look like a fuzzy caterpillar
#    with no trend and no long flat stretches, and the chains should sit on top
#    of one another. Anything else means the sampler has not settled.
# 2. **The joint posterior.** This is the one people skip, and it is the one
#    that carries the news. Marginals hide correlation; the pair plot shows it.
#    **Divergent transitions are drawn in red** — if they cluster somewhere
#    rather than scattering, that region is what your sampler could not handle.

# %%
S.posterior_diagnostics(idata_flat, PARAMS, title="Model 1: flat")

# %% [markdown]
# Clean caterpillars, overlapping chains, no red. The sampler did its job — the
# model is a poor description of the data, which is a different problem and one
# no amount of sampling fixes.
#
# In the joint, notice `a` and `t` leaning against each other. Both push the RT
# distribution to the right, so the data constrains their *combination* better
# than either alone. That is the mild version of what Session 4 turns into a
# real failure.

# %% [markdown]
# ### A trap worth knowing about: where the likelihood goes flat
#
# `t` is non-decision time — the part of the response that was never about
# deciding. So `t` **cannot exceed the response time it is part of**. What does
# the likelihood do if you ask it anyway?
#
# It does not raise, and it does not return `-inf`. It returns a constant:

# %%
_probe = np.array([[0.5, 1.0]])                      # one trial, rt = 0.5 s
for _t in [0.10, 0.30, 0.49, 0.51, 0.70, 3.00]:
    # `logp_ddm` builds a *symbolic* pytensor expression rather than a number;
    # `.eval()` is what actually computes it.
    _lp = logp_ddm(_probe, v=1.0, a=1.2, z=0.5, t=_t).eval()  # ty: ignore[unresolved-attribute]
    print(f"t = {_t:4.2f}   log p = {float(np.ravel(_lp)[0]):9.3f}"
          + ("   <- impossible: t > rt" if _t > 0.5 else ""))

# %% [markdown]
# Past `rt` the log-likelihood pins to `-66.1` and **stays there**. That region
# is perfectly flat, and flat means *no gradient*. NUTS navigates by gradient,
# so a chain that starts out there has nothing telling it which way is back. It
# does not crash or warn — it wanders on the plateau for the entire run, and
# because every trajectory runs to maximum tree depth, it is also very slow.
#
# This is not hypothetical. Our fastest trial is:

# %%
# Ask the model where it *would* have started, rather than deriving it: PyMC
# picks a "support point" per distribution, and guessing which formula it uses
# is a good way to be confidently wrong.
_default_t = float(np.exp(m1_flat.initial_point()["t_log__"]))
print(f"fastest RT in the dataset : {data['rt'].min():.3f} s")
print(f"PyMC's default start for t: {_default_t:.3f} s   <- already past it")

# %% [markdown]
# The default start sits **above the fastest RT**, i.e. inside the flat region.
# Only the random jitter PyMC adds at initialisation rescues the chains that
# happen to get pushed downward; the rest strand there for the whole run. That
# is why `fit()` passes `initvals={"t": T_INIT}`.
#
# Note *where* that goes: on `pm.sample`, not as `initval=` on the distribution.
# They look interchangeable and are not — `initval=` marks the model as having
# non-default initial values, and PyMC then refuses to compute a pointwise
# log-likelihood for it, which breaks the `az.loo` comparison in section 4.
#
# The general lesson is the one this whole section is about. **A stranded chain
# produces a summary table that looks like a result.** You would catch it in
# `r_hat`, and you would *see* it immediately in the pair plot — as a second
# blob of draws, sitting somewhere no parameter value should be.
#
# > **Poll.** A colleague's DDM fit returns `t = 0.62` with a wide interval, on
# > data whose fastest response is 0.45 s. What is the single most likely
# > explanation?
# >
# > - The participant was genuinely slow to encode the stimulus
# > - Not enough draws — it needs a longer run
# > - A chain is stuck where the likelihood is flat
# > - The boundary `a` is too wide

# %% [markdown]
# It sampled, it converged, and the numbers look perfectly reasonable. That is
# the trap: **a model that fits nothing in particular still returns a tidy
# answer.** You only find out by asking it to reproduce the data.

# %%
def cell_predictions(idata, model, n_keep=200):
    """Posterior-predictive accuracy and mean RT, per design cell.

    `sample_posterior_predictive` has no `draws` argument — it uses every
    posterior draw — so we thin afterwards to keep this quick.
    """
    with model:
        ppc = pm.sample_posterior_predictive(idata, random_seed=RANDOM_SEED,
                                             progressbar=False)
    sim = ppc["posterior_predictive"].dataset["obs"].values   # (chain, draw, obs, 2)
    sim = sim.reshape(-1, sim.shape[-2], sim.shape[-1])
    if sim.shape[0] > n_keep:                       # thin for speed
        sel = np.linspace(0, sim.shape[0] - 1, n_keep).astype(int)
        sim = sim[sel]
    out = []
    for coh in CONDITIONS:
        for emp in EMPHASES:
            mask = (data["coherence"] == coh) & (data["emphasis"] == emp)
            mask = mask.to_numpy()
            out.append({"coherence": coh, "emphasis": emp,
                        "accuracy": float((sim[:, mask, 1] == 1).mean()),
                        "mean_rt": float(sim[:, mask, 0].mean())})
    return pd.DataFrame(out).set_index(["coherence", "emphasis"])


pred_flat = cell_predictions(idata_flat, m1_flat)


def compare_plot(preds, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    xx = np.arange(len(CONDITIONS))
    for emp, colour in [("speed", S.PRIMARY), ("accuracy", S.NAIVE)]:
        obs_s = summary.xs(emp, level="emphasis")
        pre_s = preds.xs(emp, level="emphasis")
        ax1.plot(xx, obs_s["accuracy"], "o", color=colour, ms=8, label=f"{emp} (data)")
        ax1.plot(xx, pre_s["accuracy"], "--", color=colour, label=f"{emp} (model)")
        ax2.plot(xx, obs_s["mean_rt"], "o", color=colour, ms=8, label=f"{emp} (data)")
        ax2.plot(xx, pre_s["mean_rt"], "--", color=colour, label=f"{emp} (model)")
    for ax, ylab, sub in [(ax1, "P(correct)", "Accuracy"), (ax2, "mean RT (s)", "Speed")]:
        ax.set(xticks=xx, xlabel="coherence", ylabel=ylab, title=sub)
        ax.set_xticklabels(CONDITIONS)
        ax.legend(fontsize=8)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()


compare_plot(pred_flat, "Model 1: one drift, one boundary")

# %% [markdown]
# Circles are the data, dashed lines are the model. The model predicts **one
# number per panel** — flat lines — because it has no way to know which cell a
# trial came from. It splits the difference and gets everything wrong.

# %% [markdown]
# ## 3. Your turn: propose a better model
#
# You have four parameters and two design features. Let coherence and/or
# emphasis act on whichever parameters you think should carry them.
#
# In raw PyMC that is an index into a vector of parameters:
#
# ```python
# with pm.Model(coords=COORDS) as m:
#     v = pm.Normal("v", 0.0, 3.0, dims="coherence")   # one drift per level
#     ...
#     DDM("obs", v=v[coh_idx], a=a, z=z, t=t, observed=observed)
# ```
#
# `coh_idx` is an integer per trial saying which coherence level it belongs to,
# so `v[coh_idx]` is a per-trial drift rate. Nothing else changes.
#
# **Spend five minutes.** Fit at least one alternative and look at its
# predictions with `compare_plot`. Below are the two we will carry forward.

# %%
# Model 2: drift varies by coherence; one boundary for everyone.
with pm.Model(coords=COORDS) as m2_drift:
    # Priors
    v = pm.Normal("v", 0.0, 3.0, dims="coherence")
    a = pm.HalfNormal("a", 2.0)
    z = pm.Beta("z", 5.0, 5.0)
    t = pm.HalfNormal("t", 0.5)
    # Likelihood
    DDM("obs", v=v[coh_idx], a=a, z=z, t=t, observed=observed)

idata_drift = fit(m2_drift)

# %%
# Check the fit *before* looking at what it predicts: predictions drawn from a
# sampler that never converged are not predictions of anything.
print(az.summary(idata_drift, var_names=PARAMS, kind="all").to_string())
S.posterior_diagnostics(idata_drift, PARAMS, title="Model 2")

# %%
pred_drift = cell_predictions(idata_drift, m2_drift)
compare_plot(pred_drift, "Model 2: drift by coherence")

# %%
# Model 3: drift varies by coherence AND boundary varies by emphasis.
with pm.Model(coords=COORDS) as m3_both:
    # Priors
    v = pm.Normal("v", 0.0, 3.0, dims="coherence")
    a = pm.HalfNormal("a", 2.0, dims="emphasis")
    z = pm.Beta("z", 5.0, 5.0)
    t = pm.HalfNormal("t", 0.5)
    # Likelihood
    DDM("obs", v=v[coh_idx], a=a[emp_idx], z=z, t=t, observed=observed)

idata_both = fit(m3_both)

# %%
print(az.summary(idata_both, var_names=PARAMS, kind="all").to_string())
S.posterior_diagnostics(idata_both, PARAMS, title="Model 3")

# %%
pred_both = cell_predictions(idata_both, m3_both)
compare_plot(pred_both, "Model 3: drift by coherence, boundary by emphasis")

# %% [markdown]
# Model 2 captures the accuracy pattern and still misses the RT pattern —
# it has no way to separate the two emphasis conditions. Model 3 tracks both.

# %% [markdown]
# ## 4. Comparing models properly
#
# Eyeballing predictions is necessary but not sufficient: a model with more
# parameters can always fit better. **Leave-one-out cross-validation** (LOO)
# estimates out-of-sample predictive accuracy, so it penalises complexity that
# does not pay for itself.

# %% [markdown]
# First, the transparent version: how far is each model's prediction from the
# data, summed over the six design cells?

# %%
def discrepancy(preds):
    """Mean absolute error in predicted accuracy and mean RT, across cells."""
    joined = summary.join(preds, rsuffix="_pred")
    return pd.Series({
        "accuracy MAE": (joined["accuracy"] - joined["accuracy_pred"]).abs().mean(),
        "mean RT MAE": (joined["mean_rt"] - joined["mean_rt_pred"]).abs().mean(),
    })


preds_all = {"1: flat": pred_flat,
             "2: drift by coherence": pred_drift,
             "3: drift + boundary": pred_both}
print(pd.DataFrame({k: discrepancy(v) for k, v in preds_all.items()}).T
        .to_string(float_format=lambda v: f"{v:.4f}"))

# %% [markdown]
# That ranks the models but says nothing about **overfitting** — a model with
# more parameters will generally track the data more closely whether or not the
# extra flexibility is real. For that we want an estimate of *out-of-sample*
# predictive accuracy. `az.loo` provides one via leave-one-out
# cross-validation.

# %%
for name, idata in [("1: flat", idata_flat),
                    ("2: drift by coherence", idata_drift),
                    ("3: drift + boundary", idata_both)]:
    try:
        loo = az.loo(idata)
        k = loo.pareto_k.values
        bad = int((k > 0.7).sum())
        print(f"{name:24s} elpd_loo = {float(loo.elpd):9.1f}   "
              f"p_loo = {float(loo.p):6.1f}   Pareto k > 0.7: {bad:3d}/{k.size}")
    except Exception as exc:
        # Not defensive programming for its own sake — see the callout below.
        print(f"{name:24s} LOO FAILED: {type(exc).__name__}: {exc}")

# %% [markdown]
# <details class="sbi-warn" open>
# <summary>⚠️ <b>Read the Pareto <i>k</i> column before the elpd column</b></summary>
#
# LOO does not refit the model 1500 times; it *reweights* the existing draws,
# and that shortcut only works when no single observation dominates the weights.
# The Pareto $k$ diagnostic detects when it fails, and values above about 0.7
# mean the estimate for that point is not to be trusted.
#
# Look at what model 1 did. Depending on the draw it either reports a `p_loo` —
# nominally the effective number of parameters — of several hundred for a model
# with **four**, or it fails outright with `All tail values are the same`.
# Neither is a discovery about the model; both are LOO telling you it could not
# do its job. A grossly misspecified model makes some observations so
# surprising that those points dominate the importance weights, and the Pareto
# fit that LOO relies on has nothing left to work with.
#
# So: use LOO to separate *plausible* models from each other, and use posterior
# predictive plots to reject the implausible ones. Do not ask LOO to rank a
# model that the plots already told you is wrong.
#
# </details>

# %% [markdown]
# ## 5. What actually generated the data
#
# Time to look behind the curtain.

# %%
print("TRUE generating parameters")
print("  drift by coherence :", _V_BY_COHERENCE)
print("  boundary by emphasis:", _A_BY_EMPHASIS)
print(f"  start point z = {_Z_TRUE},  non-decision time t = {_T_TRUE}\n")

post = idata_both.posterior.dataset
print("Model 3 recovery")
for i, coh in enumerate(CONDITIONS):
    est = post["v"].values[..., i]
    print(f"  v[{coh:<6}] {est.mean():5.2f} +/- {est.std():.2f}"
          f"   (true {_V_BY_COHERENCE[coh]})")
for i, emp in enumerate(EMPHASES):
    est = post["a"].values[..., i]
    print(f"  a[{emp:<8}] {est.mean():5.2f} +/- {est.std():.2f}"
          f" (true {_A_BY_EMPHASIS[emp]})")
print(f"  z          {post['z'].values.mean():5.2f} +/- {post['z'].values.std():.2f}"
      f"   (true {_Z_TRUE})")
print(f"  t          {post['t'].values.mean():5.2f} +/- {post['t'].values.std():.2f}"
      f"   (true {_T_TRUE})")

# %% [markdown]
# Model 3 **is** the generating model, and it recovers the parameters. That is
# the happy case, and it is worth being explicit about why it went well:
#
# - the design has enough trials per cell,
# - the manipulations act on **different** parameters, so they do not compete,
# - and the error rates are moderate, which we will discover in half an hour is
#   not a small thing.
#
# <details class="sbi-warn" open>
# <summary>⚠️ <b>The comparison found the right model. It could not have told you it was <i>correct</i>.</b></summary>
#
# LOO ranks the candidates **you** proposed. If the true model is not among
# them, comparison happily hands you the best of a bad set, with no hint that
# anything is missing. That is what posterior predictive plots are for — they
# compare a model against the *data*, not against its rivals. Use both.
#
# </details>

# %% [markdown]
# ### Exercise
#
# Fit a fourth model in which **coherence acts on the boundary** and
# **emphasis acts on the drift** — the mapping deliberately swapped. Add it to
# the comparison.
#
# Predict first: will it fit better or worse than model 1? Than model 3?
#
# <details>
# <summary>Solution and what to notice</summary>
#
# ```python
# with pm.Model(coords=COORDS) as m4_swapped:
#     v = pm.Normal("v", 0.0, 3.0, dims="emphasis")
#     a = pm.HalfNormal("a", 2.0, dims="coherence")
#     z = pm.Beta("z", 5.0, 5.0)
#     t = pm.HalfNormal("t", 0.5)
#     DDM("obs", v=v[emp_idx], a=a[coh_idx], z=z, t=t, observed=observed)
#
# idata_swapped = fit(m4_swapped)
# for nm, idt in [("3: drift + boundary", idata_both), ("4: swapped", idata_swapped)]:
#     loo = az.loo(idt)
#     print(f"{nm:22s} elpd_loo = {float(loo.elpd):9.1f}")
# compare_plot(cell_predictions(idata_swapped, m4_swapped), "Model 4: swapped")
# ```
#
# It fits **much better than model 1** — it has the same number of parameters
# as model 3 and can bend both curves, so it soaks up a lot of the structure.
# It fits **worse than model 3**, and the predictive plot shows why: it can
# make accuracy depend on emphasis and RT depend on coherence, which is the
# wrong way round, so it cannot reproduce both patterns at once.
#
# The lesson is that "more flexible" and "right" are different things. A model
# can have exactly the right number of parameters, fit far better than nothing,
# and still be telling you a false story about the mechanism.
#
# </details>

# %% [markdown]
#
# ### Where this goes
#
# Writing `v[coh_idx]` by hand gets old quickly. On **Day 3 at 09:30** the same
# model is one line:
#
# ```python
# hssm.HSSM(data=data, model="ddm",
#           include=[{"name": "v", "formula": "v ~ 0 + C(coherence)"},
#                    {"name": "a", "formula": "a ~ 0 + C(emphasis)"}])
# ```
#
# **Next, at 15:00:** we stop assuming the sampler works. Everything today
# converged quietly — that is not guaranteed, and the failures are more
# interesting than the successes.
