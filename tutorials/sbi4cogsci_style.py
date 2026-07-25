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


def annotate(ax, text, xy, xytext, **kw):
    """A muted callout arrow, for pointing at the interesting part of a figure."""
    return ax.annotate(text, xy=xy, xytext=xytext, color=MUTED, fontsize=10,
                       arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2), **kw)
