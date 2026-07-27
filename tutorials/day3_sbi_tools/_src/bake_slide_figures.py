"""Bake the slide figures for `hierarchical-mcmc-slides.qmd`.

The deck inherits `execute: enabled: false` from `tutorials/_metadata.yml` and
the Pages CI has no Python, so slides cannot generate plots. They reference
committed PNGs, and this script produces them.

Every figure comes from `sbi4cogsci_figures`, the same module the notebook
imports — so a change to a figure updates both, and there is no second copy of
the plotting code to drift out of sync.

    cd tutorials && uv run python day3_sbi_tools/_src/bake_slide_figures.py
"""

import pathlib
import sys

# This file lives in `<day folder>/_src/`, so the day folder is one level up
# and `tutorials/` — where the shared style and figure modules live — is two.
_DAY_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DAY_DIR.parent))                 # -> tutorials/

import matplotlib
matplotlib.use("Agg")

import sbi4cogsci_figures as F
import sbi4cogsci_style as S

SEED = sum(map(ord, "sbi4cogsci-hierarchy"))
OUT = _DAY_DIR / "figures"          # beside the deck, not beside this script


def main():
    S.use()
    OUT.mkdir(exist_ok=True)

    def save(fig, name):
        path = OUT / name
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print(f"  {path.relative_to(OUT.parent)}  {path.stat().st_size // 1024} KB")

    print("funnel geometry ...")
    x, v = F.funnel_draws(seed=SEED)
    save(F.fig_funnel(x, v), "funnel.png")

    print("pooling experiment (two MCMC fits, ~8s) ...")
    result = F.pooling_experiment(seed=SEED)
    save(F.fig_shrinkage(result), "shrinkage.png")
    save(F.fig_pooling_error(result), "pooling_error.png")

    print("geometry experiment (four MCMC fits, ~25s) ...")
    geom = F.geometry_experiment(seed=SEED)
    save(F.fig_geometry_grid(geom), "geometry.png")
    print("  the reversal, in one table:")
    for (scale, par), d in geom["results"].items():
        print(f"    {par:13s} se x{scale:<5g} -> {d['n_divergences']:4d} divergences,"
              f"  reaches log tau {d['min_log_tau']:6.2f}")

    s = F.pooling_summary(result)
    print("\nnumbers quoted on the slides — keep them in sync:")
    print(f"  participants      {result['trial_counts'].size}"
          f" ({result['trial_counts'].min()}-{result['trial_counts'].max()} trials each)")
    print(f"  MAE, n < {s['split_at']:<3}     no pooling {s['no_pooling']['mae_low']:.3f}"
          f"  ->  partial {s['partial_pooling']['mae_low']:.3f}")
    print(f"  MAE, n >= {s['split_at']:<2}     no pooling {s['no_pooling']['mae_high']:.3f}"
          f"  ->  partial {s['partial_pooling']['mae_high']:.3f}")
    print(f"  low-n improvement {s['low_n_improvement_pct']:.0f}%")


if __name__ == "__main__":
    main()
