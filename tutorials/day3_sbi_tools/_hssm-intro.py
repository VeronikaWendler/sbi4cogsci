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
# # HSSM in thirty minutes
#
# **Day 3, 09:30 — 30 minutes.** Alexander Fengler.
#
# Yesterday you built a DDM likelihood into a PyMC model by hand. HSSM is the
# layer that does that for you — and then lets you put a **regression on any
# parameter**, swap the likelihood for a **neural approximation**, or bring your
# own network entirely.
#
# Thirty minutes buys three things:
#
# 1. the shortest possible working model, and the two defaults that will burn
#    you if you do not know them,
# 2. the SSM-specific fit checks,
# 3. how to plug in a likelihood **you** trained.

# %%
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # -> tutorials/

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import hssm
import sbi4cogsci_style as S

S.use()
warnings.filterwarnings("ignore")

RANDOM_SEED = sum(map(ord, "sbi4cogsci-hssm"))

print("hssm", hssm.__version__)
print("built-in models:", len(hssm.list_models()))

# %% [markdown]
# ## 1. The shortest model that works

# %%
data = hssm.load_data("cavanagh_theta")
print(data.head(5).to_string(index=False))
print(f"\n{len(data)} trials, {data['participant_id'].nunique()} participants")
print("response values:", sorted(data["response"].unique()))

# %% [markdown]
# ::: {.callout-warning}
# ## Two defaults that fail silently
#
# **1. Responses are coded `-1` / `+1`, not `0` / `1`.** A value outside the
# model's choice set raises. A *valid but mis-coded* `0`/`1` does not raise — it
# quietly mismodels the start-point bias and hands you a clean-looking, wrong
# posterior.
#
# **2. `p_outlier` defaults to `0.05`.** Every model you fit has a 5% lapse
# mixture in it unless you say otherwise. That is a defensible default and a
# terrible surprise. State it explicitly in your code and in your paper, or
# switch it off with `p_outlier=None`.
# :::

# %%
model = hssm.HSSM(data=data, model="ddm", p_outlier=0.05)
print(model)

# %% [markdown]
# `print(model)` is the single most useful HSSM command: it resolves and shows
# every prior, every bound, the link functions, and the lapse process. Read it
# before you sample, every time.

# %%
idata = model.sample(draws=500, tune=500, chains=2, cores=1,
                     random_seed=RANDOM_SEED, progressbar=False)
print(type(idata).__name__)

# %% [markdown]
# ::: {.callout-important}
# ## `model.summary()` and `model.plot_trace()` are gone in HSSM 0.4.0
# Both now raise `NotImplementedError`. Use ArviZ directly on `model.traces`,
# naming the variables you want — otherwise you get one deterministic node *per
# trial per parameter* and an unreadable table.
# :::

# %%
PARAMS = ["v", "a", "z", "t"]
print(az.summary(model.traces, var_names=PARAMS, kind="stats").to_string())

# %%
az.plot_trace_dist(model.traces, var_names=PARAMS, combined=True)
plt.gcf().set_size_inches(9, 5)
plt.tight_layout()

# %% [markdown]
# ## 2. The fit checks that are specific to SSMs
#
# A trace plot tells you the sampler behaved. It says nothing about whether the
# *model* describes the data. For sequential-sampling models there are two
# checks worth knowing.

# %%
model.sample_posterior_predictive(kind="response")
model.plot_predictive()
plt.gcf().set_size_inches(7.5, 4)
plt.tight_layout()

# %% [markdown]
# The **quantile probability plot** is the field standard: RT quantiles on the
# y-axis against choice proportion on the x-axis, so speed and accuracy are
# visible in one picture, split by condition.

# %%
model.plot_quantile_probability(cond="stim")
plt.gcf().set_size_inches(7.5, 4.5)
plt.tight_layout()

# %% [markdown]
# > **Poll.** Your posterior predictive RT densities match the data well, but
# > the quantile probability plot is off in the slowest quantile. What is the
# > most likely culprit?
# >
# > **A.** The sampler did not converge.
# > **B.** Too few posterior predictive draws.
# > **C.** The boundary is wrong — a constant bound cannot produce that tail shape.
# > **D.** The non-decision time prior is too tight.
#
# <details>
# <summary>Answer</summary>
#
# **C.** The slow tail is where boundary shape shows up. A constant-bound DDM
# produces a specific tail; if the data's tail is shorter, a **collapsing**
# bound (`angle`, `weibull`) is the usual fix. Aggregate densities can look fine
# while the conditional structure is wrong — which is exactly why the quantile
# probability plot exists.
#
# </details>

# %% [markdown]
# ## 3. Bring your own likelihood
#
# The analytic DDM likelihood exists in closed form. Most interesting models do
# not have one. HSSM's answer: **train a neural network to approximate the
# likelihood, export it to ONNX, and hand HSSM the file.**

# %%
lan_model = hssm.HSSM(
    data=data,
    model="ddm",
    loglik_kind="approx_differentiable",   # neural backend
    p_outlier=0.05,
)
print("bounds (= the network's training region):")
for param, bound in lan_model.model_config.bounds.items():
    print(f"   {param:6s} {bound}")

# %% [markdown]
# ::: {.callout-important}
# ## The single most important guardrail in the ecosystem
# For an `approx_differentiable` model, **the bounds *are* the region the
# network was trained on.** Push a parameter outside — directly, or through a
# regression whose linear predictor wanders out — and the network extrapolates.
# You get a finite, plausible, **wrong** log-likelihood. No warning, no error.
#
# Compare the numbers above with the analytic DDM's bounds: `v` is
# `(-inf, inf)` analytically but `(-3, 3)` for the network. Always check
# `model.model_config.bounds` before setting your own priors.
# :::
#
# Sampling a JAX-backed LAN has its own rule: use the numpyro sampler, and in a
# notebook keep `cores=1, chains=1`, or native NUTS can raise cloudpickle errors.

# %%
lan_idata = lan_model.sample(sampler="numpyro", draws=500, tune=500,
                             chains=1, cores=1, random_seed=RANDOM_SEED,
                             progressbar=False)
print(az.summary(lan_model.traces, var_names=PARAMS, kind="stats").to_string())

# %% [markdown]
# ### Your own network, from any framework
#
# The interop contract is **ONNX**, and it is deliberately narrow:
#
# > The graph takes **one** rank-1 input vector of shape `(n_params + n_data,)`
# > — parameters concatenated with that trial's `[rt, choice]` — and returns a
# > scalar log-likelihood. **No `dynamic_axes`.** HSSM batches over trials
# > itself with `jax.vmap`.
#
# HSSM enforces this at load time and refuses anything with a symbolic input
# dimension. That refusal is a feature: `jaxonnxruntime` bakes the
# construction-time shapes into the traced closure, so a batched graph with any
# batch-dependent intermediate (a normalizing flow's log-det accumulator, say)
# would silently return wrong numbers instead of failing.
#
# Once you have a `.onnx`, the gesture is identical no matter what trained it:
#
# ```python
# model = hssm.HSSM(
#     data=obs_data,
#     model="ddm",
#     loglik="path/to/my_network.onnx",       # <- the .onnx extension is detected
#     loglik_kind="approx_differentiable",
#     p_outlier=0,
# )
# ```
#
# For a model HSSM does not know, supply a `ModelConfig` giving `list_params`,
# `choices`, `backend`, and — critically — `bounds` set to the **training box**.
#
# ### From BayesFlow
#
# A network trained with BayesFlow (this afternoon's session, and Radev's Day 3
# tutorial) reaches HSSM through **LANfactory**:
#
# ```python
# import os
# os.environ["KERAS_BACKEND"] = "torch"     # BEFORE importing keras / bayesflow
#
# from lanfactory.onnx import transform_bayesflow_to_onnx
#
# transform_bayesflow_to_onnx(
#     approximator,                 # a trained ContinuousApproximator (NLE)
#     "ddm_nle.onnx",
#     mode="nle",                   # or "nre" for a RatioApproximator
#     example_theta_dim=4,          # len(list_params)
#     example_x_dim=2,              # [rt, choice]
# )
# ```
#
# Then `hssm.HSSM(loglik="ddm_nle.onnx", loglik_kind="approx_differentiable")`,
# exactly as above.
#
# ::: {.callout-note}
# ## Things that will cost you an afternoon
# - `KERAS_BACKEND` must be `torch` **before** keras is imported —
#   `torch.onnx.export` cannot trace a JAX-backed Keras model. The exporter
#   raises `RuntimeError` if you forget, so at least it fails loudly.
# - The CouplingFlow must be ONNX-friendly: `permutation=None`,
#   `AffineTransform(clamp=False)` passed as an **instance**, `activation="silu"`
#   (not `hard_silu`), and a trivial adapter.
# - Leave `floatX` at its default `float64`. Flow exports can carry an int64
#   sentinel that truncates to `-1` under float32 and corrupts the graph.
# - LANfactory pulls in **torch**, so it is not part of this workshop's shared
#   environment. Export offline; ship the `.onnx`.
# :::
#
# There is a lower-level escape hatch too: if you already hold a pure-JAX
# single-trial function, pass it as `loglik` with `backend="jax"` and skip ONNX
# entirely. That is the recommended path for BayesFlow likelihood-ratio
# estimators.

# %% [markdown]
# ## 4. Where this goes — regressions on parameters
#
# The reason to use HSSM rather than hand-rolling PyMC is this: **any parameter
# can carry a trial-level, hierarchical regression.**
#
# ```python
# model = hssm.HSSM(
#     data=data,
#     model="ddm",
#     include=[{
#         "name": "v",
#         "formula": "v ~ 1 + theta + (1|participant_id)",
#         "link": "identity",
#     }],
#     noncentered=True,
#     p_outlier=0.05,
#     prior_settings="safe",
# )
# ```
#
# That says: drift rate depends on the trial-level covariate `theta`, with a
# participant-specific intercept drawn from a population distribution. Which is
# a hierarchical model — and hierarchical models have a posterior geometry that
# will eat your sampler alive if you let it.
#
# **That is the 11:00 session.**

# %% [markdown]
# ## What to take away
#
# - `hssm.HSSM(data, model="ddm")` is a complete model. `print(model)` before
#   sampling, every time.
# - Responses are `[-1, +1]`. `p_outlier` is `0.05` unless you say otherwise.
#   Both fail quietly.
# - `model.summary()` / `model.plot_trace()` are gone — use `az.*` on
#   `model.traces` with explicit `var_names`.
# - `plot_quantile_probability` is the SSM fit check; aggregate densities hide
#   conditional misfit.
# - For neural likelihoods, **bounds are the training region**. Outside them the
#   answer is wrong rather than absent.
# - ONNX is the interop contract: single-trial, rank-1, no `dynamic_axes`.
