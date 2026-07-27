# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simulating data from a cognitive model
#
# **Day 2, 14:00 — 30 minutes.** Alexander Fengler.
#
# <img src="../../images/logos/ssm-simulators-logo.png" alt="ssm-simulators logo"
#      style="display:block; margin:0.5rem auto 1.5rem auto; width:240px">
#
# A cognitive model is, first and last, a **generative** story: parameters in,
# data out. Before you fit anything, you should be able to push parameters
# through the model and look at what comes back. That is what this session is
# about.
#
# We use [**ssm-simulators**](https://lnccbrown.github.io/ssm-simulators/), the
# simulation layer of the HSSM ecosystem. It ships fast compiled simulators for
# a large family of sequential-sampling models, and it is what generates the
# training data behind the neural likelihoods you will meet tomorrow.
#
# By the end you will have:
#
# 1. simulated from a model and understood the shape rules,
# 2. discovered the model zoo *by introspection* rather than by memorising names,
# 3. **built your own model** and simulated from it,
# 4. produced the dataset we dissect in the next session.

# %%
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # -> tutorials/

import numpy as np
import matplotlib.pyplot as plt
import sbi4cogsci_style as S

S.use()
warnings.filterwarnings("ignore", category=UserWarning)

import ssms
from ssms import Simulator
from ssms.config import model_config

RANDOM_SEED = sum(map(ord, "sbi4cogsci-simulate"))
rng = np.random.default_rng(RANDOM_SEED)

print("ssm-simulators", ssms.__version__ if hasattr(ssms, "__version__") else "0.13.2")

# %% [markdown]
# ## 1. One call, and the shape rules
#
# The drift-diffusion model has four parameters: drift `v`, boundary separation
# `a`, relative start point `z`, and non-decision time `t`.
#
# The main entry point is the **`Simulator` class**. You build one for a model,
# then call `.simulate()` as often as you like — the model configuration is
# resolved once, at construction, rather than on every call.

# %%
sim = Simulator(model="ddm")

out = sim.simulate(theta=[0.5, 1.2, 0.5, 0.3],
                   n_samples=500,
                   random_state=RANDOM_SEED,
                   )

print("returned keys:", sorted(out.keys()))
print("rts    ", out["rts"].shape)
print("choices", out["choices"].shape, "->", np.unique(out["choices"]))

# %% [markdown]
# Three things to notice, all of which cost people time:
#
# 1. **Choices are coded `-1` and `+1`**, not `0`/`1`. This convention runs
#    through the whole ecosystem — HSSM will reject other codings for 2-choice
#    models, and a *valid-but-wrong* `0`/`1` coding silently mismodels bias.
# 2. **The response key is `choices`**, and `rts` is 2-D — `(n_samples, 1)`,
#    not a flat vector.
# 3. `rts` **includes** non-decision time. `t` is *added* on simulation and
#    *subtracted* on fitting.

# %% [markdown]
# ### The shape rule
#
# There are three ways to ask for data, and mixing them up is the single most
# common error. What you get back depends on **both** how many parameter sets
# you hand in and how many samples you ask for:
#
# | You want | `theta` | `n_samples` | shape of `rts` |
# |---|---|---|---|
# | many trials, **one** parameter set | vector `(n_params,)` | `n` | `(n, 1)` |
# | **one** trial each, many parameter sets | matrix `(n_trials, n_params)` | `1` | `(n_trials, 1)` |
# | many samples **per** parameter set | matrix `(n_trials, n_params)` | `n` | `(n, n_trials, 1)` |
#
# The second form is how trial-varying parameters work — one row per trial —
# and is what you need when drift depends on a condition, a covariate, or a
# stimulus. The third is what you want for a parameter sweep, where you need a
# whole distribution at each setting rather than a single draw.

# %%
# Third form: 7 parameter settings, 20 samples each.
theta_sweep = np.column_stack([
    np.linspace(-2.0, 2.0, 7),   # v varies
    np.full(7, 1.2),             # a
    np.full(7, 0.5),             # z
    np.full(7, 0.3),             # t
])
swept = sim.simulate(theta=theta_sweep, n_samples=20, random_state=RANDOM_SEED)
print("theta", theta_sweep.shape, "with n_samples=20 ->",
      "rts", swept["rts"].shape, "  (n_samples, n_trials, 1)")

# %% [markdown]
# Now the trial-varying form, at a scale that also tells us something about
# speed: 100,000 trials, drift ramping from strongly negative to strongly
# positive.

# %%
n_trials = 100_000
V_LO, V_HI = -5.0, 5.0
v_by_trial = np.linspace(V_LO, V_HI, n_trials)
theta_matrix = np.column_stack([
    v_by_trial,
    np.full(n_trials, 1.2),   # a
    np.full(n_trials, 0.5),   # z
    np.full(n_trials, 0.3),   # t
])

trialwise = sim.simulate(theta=theta_matrix, n_samples=1, random_state=RANDOM_SEED)
rt = trialwise["rts"].flatten()
ch = trialwise["choices"].flatten()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
ax1.plot(v_by_trial, rt, "o", color=S.PRIMARY, ms=1, alpha=0.10, ls="none")
ax1.set(title="Drift controls speed", xlabel="drift $v$", ylabel="RT (s)",
        ylim=(0, 4))

# Choice proportion in drift bins — binned across the FULL simulated range.
bins = np.linspace(V_LO, V_HI, 41)
idx = np.digitize(v_by_trial, bins) - 1
prop = [np.mean(ch[idx == i] == 1) if (idx == i).sum() else np.nan
        for i in range(len(bins) - 1)]
ax2.plot(0.5 * (bins[:-1] + bins[1:]), prop, "-", color=S.PRIMARY, lw=2,
         label="simulated")
S.truth_line(ax2, 0.5, label="chance")
ax2.set(title="...and accuracy", xlabel="drift $v$", ylabel="P(choice = +1)",
        ylim=(0, 1))
ax2.legend()
fig.tight_layout()

# %% [markdown]
# Accuracy **saturates**: past roughly $|v| = 3$ every trial goes the same way,
# and pushing drift higher buys nothing observable. Hold on to that — it is the
# mechanism behind the identifiability failure in the next session.
#
# > **Poll.** Drift `v` is negative on a trial. Which is true?
# >
# > - **A.** The trial is faster and more likely to end at `+1`.
# > - **B.** The trial is slower and more likely to end at `-1`.
# > - **C.** The trial is equally fast either way; only accuracy changes.
# > - **D.** The trial drifts toward `-1`, and is *faster* the more negative `v` is.
#
# <details>
# <summary>Answer</summary>
#
# **D.** The sign of `v` picks the boundary; the *magnitude* sets how fast
# evidence accumulates. Strongly negative drift is both fast and reliably `-1`.
# The slowest trials are the ones with drift near zero — the tall spike in the
# middle of the left panel. Speed and accuracy are governed by the same
# quantity here, which is exactly why they trade off.
#
# </details>

# %% [markdown]
# ## 2. How fast is this, and does threading help?
#
# You just simulated 100,000 trials. Worth knowing what that cost, because the
# answer determines whether you ever need to think about performance again.

# %%
import time

t0 = time.time()
sim.simulate(theta=theta_matrix, n_samples=1, random_state=RANDOM_SEED)
elapsed = time.time() - t0
print(f"{n_trials:,} trials in {elapsed:.2f}s  "
      f"({n_trials / elapsed / 1000:.0f}k trials/second)")

# %% [markdown]
# The simulators are compiled Cython, so this is fast enough that simulation is
# rarely your bottleneck — fitting is.
#
# `simulate()` also takes an `n_threads` argument. Whether it does anything
# depends on how your copy was **built**, and `ssm-simulators` ships a
# diagnostic that tells you:

# %%
from cssm import _openmp_status

_openmp_status.print_status()

# %% [markdown]
# <details class="sbi-warn" open>
# <summary>⚠️ <b><code>n_threads</code> is a no-op unless the wheel was built with OpenMP</b></summary>
#
# If the report above says `OpenMP compiled: No`, then `n_threads` is silently
# ignored — you will see identical timings for `n_threads=1` and `n_threads=8`,
# with no error and no warning. The prebuilt macOS wheels are commonly in this
# state.
#
# To get real parallelism you have to install the OpenMP runtime and rebuild:
#
# ```bash
# brew install libomp          # macOS   (Linux: apt install build-essential)
# pip install --force-reinstall --no-binary ssm-simulators ssm-simulators
# ```
#
# For this workshop it does not matter — single-threaded is already fast enough.
# It matters when you generate **training data for a neural likelihood**, where
# you may want tens of millions of trials.
#
# </details>

# %% [markdown]
# ## 3. The model zoo, by introspection
#
# Do not memorise the model list — interrogate it. Every model carries its own
# parameter names and the box those parameters live in.

# %%
import textwrap

names = sorted(model_config.keys())
print(f"{len(names)} models registered. The first 30, alphabetically:\n")
print(textwrap.fill(", ".join(names[:30]), width=76,
                    initial_indent="   ", subsequent_indent="   "))
print(f"\n   ... and {len(names) - 30} more.")

# %% [markdown]
# Each entry is a plain dictionary. Look at one in full — this is the whole
# contract a model has to satisfy:

# %%
cfg = model_config["angle"]
for key, value in cfg.items():
    shown = value if not callable(value) else f"<callable {getattr(value, '__name__', '?')}>"
    print(f"  {key:<22} {str(shown)[:70]}")

# %% [markdown]
# The two fields you will reach for constantly are `params` (the names, **in
# the order the simulator expects them**) and `param_bounds` (the box they are
# valid in). Compare a few models:

# %%
for name in ["ddm", "angle", "weibull", "lba3"]:
    c = model_config[name]
    lo, hi = c["param_bounds"]
    box = ", ".join(f"{p} [{l:g}, {h:g}]" for p, l, h in zip(c["params"], lo, hi))
    print(f"{name:8s} choices={str(c['choices']):9s} {box}")

# %% [markdown]
# `angle` is the DDM plus one parameter, `theta`, which tilts the decision
# boundary inward over time — the evidence needed to commit *decreases* the
# longer you deliberate. Look at what that does, both to the boundary and to
# the RT distribution.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

# Left: the boundary itself
t_grid = np.linspace(0, 3, 200)
a0 = 1.5
ax1.plot(t_grid, np.full_like(t_grid, a0), color=S.PRIMARY, label="ddm (constant)")
for theta_val, ls in [(0.4, "--"), (0.8, ":")]:
    ax1.plot(t_grid, np.maximum(a0 - np.tan(theta_val) * t_grid, 0),
             color=S.NAIVE, linestyle=ls, label=f"angle, $\\theta$={theta_val}")
ax1.set(title="Decision boundary over time", xlabel="time (s)",
        ylabel="boundary", ylim=(0, 1.7))
ax1.legend()

# Right: what that does to RTs
for name, theta, colour in [("ddm", [0.5, 1.5, 0.5, 0.3], S.PRIMARY),
                            ("angle", [0.5, 1.5, 0.5, 0.3, 0.8], S.NAIVE)]:
    o = Simulator(model=name).simulate(theta=theta, n_samples=3000,
                                       random_state=RANDOM_SEED)
    ax2.hist(o["rts"].flatten(), bins=60, range=(0, 5), density=True,
             histtype="step", lw=2, color=colour, label=name)
ax2.set(title="Collapsing bounds shorten the tail", xlabel="RT (s)", ylabel="density")
ax2.legend()
fig.tight_layout()

# %% [markdown]
# That is the whole game: a parameter with a mechanistic meaning produces a
# specific, *visible* signature in the data. Model building is choosing which
# signatures you want to be able to express.

# %% [markdown]
# ### Turn the knobs yourself
#
# Two tools, both browser-based:
#
# - **[DDM explorer](https://stefanradev93.github.io/sbi4cogsci/tutorials/day2_bayes_toolkit/ddm-explorer.html)**
#   — a companion page for this session: sliders over drift and boundary
#   separation, showing what each does to the response-time distributions and
#   to accuracy.
# - **[ssms-gui](https://github.com/AlexanderFengler/ssms-gui)** — a dashboard
#   for playing with `ssm-simulators` output directly, across the model zoo
#   rather than just the DDM. Built for exactly this kind of build-intuition
#   exploration.

# %% [markdown]
# ### Exercise 1
#
# Pick any model from the zoo that we have not plotted, print its parameters
# and bounds, simulate 2000 trials at a setting of your choice, and plot the RT
# distribution split by choice.
#
# <details>
# <summary>One solution</summary>
#
# ```python
# name = "weibull"
# c = model_config[name]
# print(c["params"], c["param_bounds"])
#
# # v, a, z, t, alpha, beta  — alpha/beta shape the collapsing boundary
# o = Simulator(model=name).simulate(
#     theta=[0.6, 1.4, 0.5, 0.3, 2.0, 2.5], n_samples=2000, random_state=0)
# rt, ch = o["rts"].flatten(), o["choices"].flatten()
#
# fig, ax = plt.subplots(figsize=(6, 3.5))
# for c_val, colour, lbl in [(1, S.PRIMARY, "+1"), (-1, S.NAIVE, "-1")]:
#     ax.hist(rt[ch == c_val], bins=50, range=(0, 4), density=True,
#             histtype="step", lw=2, color=colour, label=f"choice {lbl}")
# ax.set(xlabel="RT (s)", ylabel="density", title=f"{name}: RT by choice")
# ax.legend()
# ```
#
# </details>

# %% [markdown]
# ## 4. Capstone — build your own model
#
# Two routes, depending on how far from a diffusion you want to go. Both are
# about fifteen lines.
#
# ### Route A — keep the diffusion machinery, change the boundary
#
# Register a boundary function and graft it onto the DDM. The function takes
# time `t` plus its own parameters and returns the **final boundary value**,
# `a` included — it is not an offset that `a` gets added to. `a` must be listed
# first in its parameter list.

# %%
from ssms.config import register_boundary
from ssms.config.model_config_builder import ModelConfigBuilder


def exp_collapse(t, a=1.0, rate=0.5):
    """Exponentially collapsing boundary: a * exp(-rate * t)."""
    import numpy as np          # import locally — the function gets pickled
    return a * np.exp(-rate * np.asarray(t))


register_boundary("exp_collapse", exp_collapse, ["a", "rate"])

cfg_custom = ModelConfigBuilder.from_model("ddm")
cfg_custom = ModelConfigBuilder.add_boundary(cfg_custom, "exp_collapse", ["a", "rate"])

my_sim = Simulator(model=cfg_custom)
o_slow = my_sim.simulate(theta={"v": 0.8, "a": 1.5, "z": 0.5, "t": 0.2, "rate": 0.15},
                         n_samples=3000, random_state=RANDOM_SEED)
o_fast = my_sim.simulate(theta={"v": 0.8, "a": 1.5, "z": 0.5, "t": 0.2, "rate": 1.2},
                         n_samples=3000, random_state=RANDOM_SEED)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
for o, colour, lbl in [(o_slow, S.PRIMARY, "rate = 0.15 (slow collapse)"),
                       (o_fast, S.NAIVE, "rate = 1.2 (fast collapse)")]:
    ax.hist(o["rts"].flatten(), bins=60, range=(0, 5), density=True,
            histtype="step", lw=2, color=colour, label=lbl)
ax.set(title="A boundary you invented, five minutes ago", xlabel="RT (s)",
       ylabel="density")
ax.legend()
fig.tight_layout()

# %% [markdown]
# <details class="sbi-key" open>
# <summary>🔑 <b>Registration is per-process</b></summary>
#
# `register_boundary` writes into an in-memory registry. Restart the kernel and
# your model is gone — re-run the cell. In a script, register at import time.
#
# </details>

# %% [markdown]
# ### Route B — write the whole simulator
#
# If your model is not a diffusion at all, hand over a function. It must accept
# the parameters you declare plus `n_samples`, and return a dict with `rts`,
# `choices` and `metadata`, where **`rts` and `choices` are 2-D**, shape
# `(n_samples, 1)`.

# %%
def racing_exponentials(v0, v1, t, n_samples=1000, max_t=20.0, delta_t=0.001,
                        random_state=None, **kwargs):
    """Two exponential accumulators race; the first to finish picks the choice.

    Not a diffusion at all — the interface is genuinely open.
    """
    import numpy as np
    rng = np.random.default_rng(random_state)
    finish_0 = rng.exponential(1.0 / v0, n_samples)
    finish_1 = rng.exponential(1.0 / v1, n_samples)
    rts = (np.minimum(finish_0, finish_1) + t).reshape(-1, 1)
    choices = np.where(finish_1 < finish_0, 1, -1).reshape(-1, 1)
    return {"rts": rts, "choices": choices,
            "metadata": {"model": "racing_exponentials", "n_samples": n_samples}}


race = Simulator(simulator_function=racing_exponentials,
                 params=["v0", "v1", "t"], nchoices=2)
o = race.simulate(theta={"v0": 1.0, "v1": 2.0, "t": 0.25},
                  n_samples=3000, random_state=RANDOM_SEED)

rt_r, ch_r = np.asarray(o["rts"]).flatten(), np.asarray(o["choices"]).flatten()
print(f"P(choice = +1) = {(ch_r == 1).mean():.3f}   "
      f"(closed form v1/(v0+v1) = {2/3:.3f})")

fig, ax = plt.subplots(figsize=(6.5, 3.4))
for c_val, colour, lbl in [(1, S.PRIMARY, "+1 (faster accumulator)"),
                           (-1, S.NAIVE, "-1")]:
    ax.hist(rt_r[ch_r == c_val], bins=50, range=(0, 3), density=True,
            histtype="step", lw=2, color=colour, label=lbl)
ax.set(title="A model that is not a diffusion", xlabel="RT (s)", ylabel="density")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The analytic check matters: for two racing exponentials,
# $P(\text{choice}=+1) = v_1/(v_0+v_1)$. We recovered it. **Always find one
# quantity your simulator must reproduce in closed form** — it is the cheapest
# bug detector you will ever write.

# %% [markdown]
# ### Exercise 2
#
# Real response-time data almost always contains a few responses that the model
# did not generate: button mashing, attention lapses, a sneeze. Extend
# `racing_exponentials` with a **contaminant process**: with probability
# $p_\text{lapse}$, a trial instead produces a uniform random RT in
# $[0, 3]$ s and a coin-flip choice.
#
# Simulate with $p_\text{lapse} = 0$ and $p_\text{lapse} = 0.1$ and plot both.
# Where in the distribution does the contamination show up, and which part of
# it would you expect to distort a fit the most?
#
# <details>
# <summary>Solution, and why this parameter matters</summary>
#
# ```python
# def racing_with_lapse(v0, v1, t, p_lapse=0.0, n_samples=1000, max_t=20.0,
#                       delta_t=0.001, random_state=None, **kwargs):
#     import numpy as np
#     rng = np.random.default_rng(random_state)
#     f0 = rng.exponential(1.0 / v0, n_samples)
#     f1 = rng.exponential(1.0 / v1, n_samples)
#     rts = np.minimum(f0, f1) + t
#     choices = np.where(f1 < f0, 1, -1)
#
#     lapse = rng.random(n_samples) < p_lapse          # which trials are junk
#     rts[lapse] = rng.uniform(0.0, 3.0, lapse.sum())
#     choices[lapse] = rng.choice([-1, 1], lapse.sum())
#
#     return {"rts": rts.reshape(-1, 1), "choices": choices.reshape(-1, 1),
#             "metadata": {"model": "racing_with_lapse", "n_samples": n_samples}}
#
# race_l = Simulator(simulator_function=racing_with_lapse,
#                    params=["v0", "v1", "t", "p_lapse"], nchoices=2)
#
# fig, ax = plt.subplots(figsize=(6.5, 3.6))
# for p, colour in [(0.0, S.PRIMARY), (0.1, S.NAIVE)]:
#     o = race_l.simulate(theta={"v0": 1.0, "v1": 2.0, "t": 0.25, "p_lapse": p},
#                         n_samples=5000, random_state=0)
#     ax.hist(np.asarray(o["rts"]).flatten(), bins=60, range=(0, 3), density=True,
#             histtype="step", lw=2, color=colour, label=f"p_lapse = {p}")
# ax.set(xlabel="RT (s)", ylabel="density"); ax.legend()
# ```
#
# The contamination is nearly invisible in the bulk and obvious in the
# **tails** — it puts mass at very short RTs that the race process essentially
# never produces, and it fattens the slow tail. Those are exactly the regions a
# likelihood cares most about: a single impossibly-fast response can have a
# huge influence on the fit, because the model assigns it almost zero density.
#
# This is not a toy concern. HSSM includes a lapse process **by default** —
# `p_outlier=0.05` — for precisely this reason, and you will meet it tomorrow
# morning. Now you know what it is protecting you from.
#
# </details>

# %% [markdown]
# ## 5. The dataset for the next session
#
# We hand two datasets forward. Same model, same number of trials — the only
# difference is the drift, and therefore the error rate.

# %%
import pandas as pd

def make_dataset(v, label, n=800, seed=RANDOM_SEED):
    o = sim.simulate(theta=[v, 1.2, 0.5, 0.3], n_samples=n, random_state=seed)
    return pd.DataFrame({"rt": o["rts"].flatten(),
                         "response": o["choices"].flatten().astype(int),
                         "design": label})

# v = 0.5 lands ~23% errors, inside the 15-35% band that Lüken et al. (2025)
# recommend for identifiability. v = 3.0 is the pathological case: ~0% errors.
balanced = make_dataset(0.5, "balanced")
extreme = make_dataset(3.0, "extreme")

for df in (balanced, extreme):
    err = (df["response"] == -1).mean()
    print(f"{df['design'][0]:9s} error rate {err:6.1%}   mean RT {df['rt'].mean():.3f}s")

outdir = pathlib.Path("data")
outdir.mkdir(exist_ok=True)
pd.concat([balanced, extreme]).to_csv(outdir / "ddm_two_designs.csv", index=False)
print(f"\nwrote {outdir / 'ddm_two_designs.csv'}")

# %% [markdown]
# <details class="sbi-note">
# <summary>📝 <b>Where this is going</b></summary>
#
# Both datasets come from the *same model* with the *same* `a`, `z`, `t`. Only
# the drift differs. One of them supports clean parameter recovery and the
# other does not — and it is not the one most people would guess. That is the
# next session.
#
# </details>
#
# `data/` is gitignored: it is regenerated by running this notebook, never
# committed.

# %% [markdown]
# ## 6. Where to go next
#
# We stayed inside the two-choice diffusion family. The package goes
# considerably further, and two directions are worth knowing about:
#
# **Attentional drift-diffusion (`addm`)** — drift depends on where the
# participant is *looking* on each moment of the trial, so fixation data enters
# the model directly. Use it when you have eye-tracking alongside choices and
# RTs.
#
# ```python
# Simulator(model="addm")   # takes per-trial fixation data via extra_fields
# ```
#
# **Reinforcement-learning SSMs (`ssms.rl`)** — the drift rate on each trial is
# produced by a learning process, so choice, response time *and* learning are
# modelled jointly. This is how you fit a bandit task where you care about both
# what was learned and how the decision was made.
#
# ```python
# import ssms.rl                # learning rules that feed an SSM
# ```
#
# Both plug into HSSM the same way the models here do. If your project involves
# eye-tracking or learning, start there rather than bolting it on later.

# %% [markdown]
# ## What to take away
#
# <details class="sbi-tip">
# <summary>💡 <b>The five things that matter</b></summary>
#
#
# 1. **The simulator is the model.** If you cannot generate from it, you do not
#    understand it yet.
# 2. **Interrogate, do not memorise.** `model_config[name]["params"]` and
#    `["param_bounds"]` tell you what any model in the zoo wants.
# 3. **Shapes:** `theta` vector + `n_samples=n` gives `(n, 1)`; `theta` matrix +
#    `n_samples=1` gives `(n_trials, 1)`; matrix + `n>1` gives
#    `(n, n_trials, 1)`. Choices are `[-1, +1]`.
# 4. **You can build your own model in ~15 lines** — a new boundary, or a whole
#    new simulator. Registration is per-process.
# 5. **Always find one closed-form quantity your simulator must reproduce.**
#
# </details>
#
# ### Quick reference
#
# | want to | call |
# |---|---|
# | build a simulator | `sim = Simulator(model="ddm")` |
# | simulate | `sim.simulate(theta=[...], n_samples=n)` |
# | list models | `model_config.keys()` |
# | inspect one | `model_config["angle"]` |
# | check parallel support | `from cssm import _openmp_status; _openmp_status.print_status()` |
# | add a boundary | `register_boundary(name, fn, ["a", ...])` |
# | bring your own simulator | `Simulator(simulator_function=fn, params=[...], nchoices=2)` |
#
# **Next:** we take the two datasets we just wrote and find out what happens
# when you try to recover the parameters that made them.
