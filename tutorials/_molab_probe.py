# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo",
#     "ssm-simulators>=0.13.2",
#     "hssm>=0.4",
#     "pymc>=6.2",
#     "arviz>=1.2",
#     "matplotlib",
# ]
# ///
"""Probe molab's container to settle what the docs do not state.

WHY THIS EXISTS
    molab is confirmed to run real server-side CPython (4 CPUs / 32 GB, CoreWeave
    containers, uv-managed deps) — https://docs.marimo.io/guides/molab/. What the
    docs do NOT state is whether the container can actually host our stack:
    architecture, glibc, a C/C++ toolchain for PyTensor's runtime compilation,
    and the OpenMP runtime that ssm-simulators' `cssm` extensions link against.

HOW TO RUN
    1. Sign in at https://molab.marimo.io (running on server compute requires an
       account; reading does not).
    2. New notebook -> paste this file -> run.
    3. ⚠️ Confirm the view mode is **Server**, not WebAssembly. Under WebAssembly
       this probe is expected to fail at the very first import, and that failure
       says nothing about the container.

WHAT TO REPORT BACK
    The whole output of the last cell.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("# molab container probe\nEverything below is measured, not assumed.")
    return (mo,)


@app.cell
def _():
    # --- 1. Is this actually a server, and what kind? ------------------------
    import platform, sys, os, shutil, subprocess, textwrap

    lines = []
    def say(k, v):
        lines.append(f"{k:<26} {v}")

    say("python", sys.version.split()[0])
    say("platform.machine()", platform.machine())
    say("platform.system()", platform.system())
    try:
        say("libc", " ".join(platform.libc_ver()))
    except Exception as e:
        say("libc", f"unavailable: {e}")
    say("cpu_count", os.cpu_count())

    # Emscripten/Pyodide would show up here and invalidate everything else.
    say("is WASM?", "YES — WRONG VIEW MODE" if sys.platform == "emscripten" else "no (good)")
    return lines, say, shutil, subprocess, textwrap


@app.cell
def _(lines, say, shutil, subprocess):
    # --- 2. Toolchain: does PyTensor have a C compiler if it wants one? ------
    for tool in ("gcc", "g++", "cc", "ld"):
        say(f"which {tool}", shutil.which(tool) or "ABSENT")

    # --- 3. OpenMP runtime, which cssm's compiled extensions link against ----
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30)
        say("libgomp present", "yes" if "libgomp" in out.stdout else "NOT FOUND in ldconfig")
    except Exception as e:
        say("libgomp present", f"could not check: {e}")
    return


@app.cell
def _(lines, say):
    # --- 4. The real test: does the stack import and RUN? -------------------
    import time

    t0 = time.time()
    try:
        import ssms
        from ssms.basic_simulators.simulator import simulator
        say("ssms import", f"OK ({time.time()-t0:.1f}s)")
        # cssm is the compiled half. If OpenMP or glibc were wrong, this is where
        # it shows up.
        out = simulator(theta=[0.5, 1.2, 0.5, 0.3], model="ddm",
                        n_samples=500, random_state=1)
        say("cssm simulate", f"OK — {out['rts'].shape} rts, keys ok")
    except Exception as e:
        say("ssms/cssm", f"FAILED: {type(e).__name__}: {e}")

    try:
        t1 = time.time()
        import pymc as pm, numpy as np
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            pm.Normal("y", mu, 1, observed=np.random.default_rng(0).normal(size=50))
            idata = pm.sample(draws=100, tune=100, chains=1, cores=1,
                              nuts_sampler="pymc", progressbar=False, random_seed=1)
        say("pymc sample", f"OK ({time.time()-t1:.1f}s) — this proves the "
                           "compile path works, whichever backend it used")
    except Exception as e:
        say("pymc sample", f"FAILED: {type(e).__name__}: {e}")

    try:
        import hssm
        d = hssm.load_data("cavanagh_theta")
        say("hssm + bundled data", f"OK — {len(d)} rows")
    except Exception as e:
        say("hssm", f"FAILED: {type(e).__name__}: {e}")
    return


@app.cell
def _(lines, say):
    # --- 5. Network + persistence -------------------------------------------
    # HSSM downloads ONNX likelihood networks from HuggingFace on first use;
    # if egress is blocked or nothing persists, that is a workshop problem.
    import os, pathlib, urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=15)
        say("egress to huggingface", "OK")
    except Exception as e:
        say("egress to huggingface", f"BLOCKED/failed: {type(e).__name__}")

    p = pathlib.Path.home() / "molab_persistence_probe.txt"
    say("persistence marker", "ALREADY EXISTED (persists!)" if p.exists() else "created now — re-run later to test")
    p.write_text("probe")
    try:
        st = os.statvfs(str(pathlib.Path.home()))
        say("home free space", f"{st.f_bavail * st.f_frsize / 1e9:.1f} GB")
    except Exception as e:
        say("home free space", f"unknown: {e}")
    return


@app.cell
def _(lines, mo):
    mo.md("## Results\n\n```\n" + "\n".join(lines) + "\n```")
    return


if __name__ == "__main__":
    app.run()
