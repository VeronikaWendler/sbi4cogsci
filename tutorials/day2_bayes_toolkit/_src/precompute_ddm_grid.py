"""Precompute a DDM parameter grid for the interactive figure in T2.

The published site never executes Python, and `ssm-simulators` is Cython
(`cssm` ships 11 compiled .so files), so it cannot run in the browser either.
The interactive figure therefore reads a committed grid of *precomputed*
simulations and the slider selects among them.

Run this whenever the grid needs regenerating:

    cd tutorials && uv run python day2_bayes_toolkit/_src/precompute_ddm_grid.py

Writes `day2_bayes_toolkit/ddm_grid.json`.
"""

import json
import pathlib

import numpy as np
from ssms.basic_simulators.simulator import simulator

SEED = sum(map(ord, "sbi4cogsci-grid"))
N_SAMPLES = 4000
RT_MAX, N_BINS = 4.0, 40

V_GRID = [round(v, 2) for v in np.arange(-2.0, 2.01, 0.25)]
A_GRID = [round(a, 2) for a in np.arange(0.6, 2.01, 0.2)]

edges = np.linspace(0.0, RT_MAX, N_BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])


def cell(v, a):
    out = simulator(theta=[v, a, 0.5, 0.3], model="ddm",
                    n_samples=N_SAMPLES, random_state=SEED)
    rt = out["rts"].flatten()
    ch = out["choices"].flatten()
    rec = {"v": v, "a": a, "p_upper": round(float((ch == 1).mean()), 4),
           "mean_rt": round(float(rt.mean()), 4)}
    for name, mask in (("upper", ch == 1), ("lower", ch == -1)):
        # Density scaled by the share of trials ending on this boundary, so the
        # two curves are comparable and their areas show the choice split.
        counts, _ = np.histogram(rt[mask], bins=edges)
        share = mask.mean()
        dens = counts / max(counts.sum(), 1) * share / (edges[1] - edges[0])
        rec[name] = [round(float(x), 5) for x in dens]
    return rec


def main():
    grid = [cell(v, a) for a in A_GRID for v in V_GRID]
    payload = {
        "meta": {"model": "ddm", "z": 0.5, "t": 0.3, "n_samples": N_SAMPLES,
                 "seed": SEED, "rt_centers": [round(float(c), 4) for c in centers]},
        "v_grid": V_GRID,
        "a_grid": A_GRID,
        "cells": grid,
    }
    # This file lives in `<day folder>/_src/`; the JSON belongs beside the
    # page that fetches it, one level up.
    out = pathlib.Path(__file__).resolve().parent.parent / "ddm_grid.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    kb = out.stat().st_size / 1024
    print(f"{len(grid)} cells ({len(V_GRID)} x {len(A_GRID)}) -> {out.name}  {kb:.0f} KB")


if __name__ == "__main__":
    main()
