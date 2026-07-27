"""Shared plot style for the SBI4CogSci tutorials.

One visual language across every session, so a colour means the same thing on
Day 2 afternoon as it does on Day 3 morning.

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path.cwd().parent))   # -> tutorials/
    import sbi4cogsci_style as S
    S.use()

The palette is **semantic, not decorative** — pick by the role the series plays,
never by the order it happens to be plotted:

    PRIMARY    the method we are advocating   (NUTS, non-centered, well-designed)
    NAIVE      the comparator that struggles  (Metropolis, centered, degenerate)
    ALT        a third series when needed     (Slice, an intermediate setting)
    DIVERGENT  divergences, funnel necks, failure regions
    TRUTH      ground truth — always dashed black, via truth_line()/truth_point()
    MUTED      context, annotation, de-emphasised marks

Colours are validated steps, checked all-pairs on a light surface (#fcfcfb):
worst CVD deltaE 9.9 (>= 8 target), worst normal-vision deltaE 16.3 (>= 15 floor).
ALT sits at 2.74:1 contrast, below 3:1 — so **always draw a legend or direct
label**; that is the relief the low contrast requires, not an optional nicety.

Figures are baked into notebooks as static images and the site theme is light,
so this is a light-mode palette by design.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

PRIMARY = "#2a78d6"   # blue
NAIVE = "#4a3aa7"     # violet
ALT = "#1baf7a"       # aqua
DIVERGENT = "#d03b3b" # red (status: critical)
TRUTH = "#0b0b0b"     # primary ink
MUTED = "#898781"

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Cycle order matches the semantic order above, so an unstyled plot still
#: reads correctly if someone forgets to pass an explicit colour.
CYCLE = [PRIMARY, NAIVE, ALT, DIVERGENT]


def use():
    """Apply the shared style to all subsequent matplotlib figures."""
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        # "semibold" is not a matplotlib weight — it falls back to 700 and emits a
        # findfont warning in every cell. Ask for bold directly.
        "axes.titleweight": "bold",
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "font.size": 11,
    })


def truth_line(ax, value, *, axis="y", label="truth", **kw):
    """Draw a ground-truth reference. Always dashed black — never a series colour."""
    draw = ax.axhline if axis == "y" else ax.axvline
    return draw(value, color=TRUTH, linestyle="--", linewidth=1.5,
                zorder=1, label=label, **kw)


def truth_point(ax, x, y, *, label="truth", **kw):
    """Mark a ground-truth location in parameter space."""
    return ax.plot(x, y, marker="X", color=TRUTH, markersize=11, linestyle="none",
                   zorder=5, label=label, **kw)


def divergences(ax, x, y, *, label="divergences", **kw):
    """Overlay divergent transitions. Small, opaque, on top of everything."""
    return ax.plot(x, y, marker="o", color=DIVERGENT, markersize=4,
                   linestyle="none", alpha=0.85, zorder=4, label=label, **kw)


def signed_rt_hist(ax, rt, choices, *, bins=60, rt_max=4.0, color=None,
                   label=None, lw=2, fill=False, alpha=0.20):
    """Two-choice RT histogram, mirrored about zero.

    Trials that ended at the **lower** boundary (choice `-1`) are drawn to the
    left of zero, trials at the **upper** boundary (`+1`) to the right. This is
    the conventional way to show a two-choice RT distribution: the shape of both
    response types *and* the choice split are visible in one panel, because each
    side's area is that response's share of the trials.

    Returns (proportion_lower, proportion_upper).
    """
    import numpy as np

    rt = np.asarray(rt).ravel()
    ch = np.asarray(choices).ravel()
    edges = np.linspace(0.0, rt_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    colour = color if color is not None else PRIMARY

    props = {}
    for sign, side in ((-1, "lower"), (1, "upper")):
        mask = ch == sign
        share = mask.mean()
        props[side] = share
        counts, _ = np.histogram(rt[mask], bins=edges)
        # Scale so the two sides together integrate to 1: each side's AREA is
        # then that response's probability.
        dens = counts / max(counts.sum(), 1) * share / width
        x = sign * centers
        ax.step(x, dens, where="mid", color=colour, lw=lw,
                label=label if sign == 1 else None)
        if fill:
            ax.fill_between(x, dens, step="mid", color=colour, alpha=alpha)

    ax.axvline(0.0, color=AXIS, lw=1)
    return props["lower"], props["upper"]


def label_choice_axis(ax, rt_max=4.0, lower="lower boundary (-1)",
                      upper="upper boundary (+1)"):
    """Label a mirrored RT axis: |RT| with the side annotated."""
    import numpy as np

    ticks = np.array([-rt_max, -rt_max / 2, 0, rt_max / 2, rt_max])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):g}" for t in ticks])
    ax.set_xlim(-rt_max, rt_max)
    # The side labels go in the axis label itself. Placing them inside the axes
    # collides with either the legend (top) or the data (bottom), depending on
    # the distribution — the label strip is the one place that is always free.
    ax.set_xlabel(f"← {lower}     response time (s)     {upper} →")


def posterior_diagnostics(idata, var_names, *, title=None,
                          trace_size=(9.5, None), pair_size=(7.0, 7.0)):
    """The same two looks at every fit: traces, then the joint.

    Both come from ArviZ (`plot_trace_dist` and `plot_pair`) rather than being
    hand-rolled — this only fixes the sizing, the divergence styling and the
    titles, so that every model in a notebook is inspected identically and the
    plots can be compared across fits at a glance.

    **Divergent transitions are drawn on the pair plot** in the palette's
    DIVERGENT colour. If they cluster anywhere rather than scattering, that
    location is the part of the posterior your sampler could not handle.

    Returns (trace_figure, pair_figure).
    """
    import arviz as az
    import matplotlib.pyplot as plt

    az.plot_trace_dist(idata, var_names=var_names, combined=True)
    fig_trace = plt.gcf()
    w, h = trace_size
    fig_trace.set_size_inches(w, h if h is not None else 1.5 * len(var_names) + 0.5)
    if title:
        fig_trace.suptitle(f"{title} — marginals and traces", y=1.02)
    fig_trace.tight_layout()

    # `divergence` is drawn with matplotlib's `scatter`, so the size keyword is
    # `s`, not `markersize` (which raises).
    az.plot_pair(idata, var_names=var_names, marginal=True,
                 visuals={"divergence": {"color": DIVERGENT, "s": 14,
                                         "alpha": 0.9}})
    fig_pair = plt.gcf()
    fig_pair.set_size_inches(*pair_size)
    if title:
        fig_pair.suptitle(f"{title} — joint posterior", y=1.01)
    fig_pair.tight_layout()

    return fig_trace, fig_pair


def annotate(ax, text, xy, xytext, **kw):
    """A muted callout arrow, for pointing at the interesting part of a figure."""
    return ax.annotate(text, xy=xy, xytext=xytext, color=MUTED, fontsize=10,
                       arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2), **kw)
