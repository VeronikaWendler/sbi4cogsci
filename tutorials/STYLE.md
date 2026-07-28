# Plot style for the SBI4CogSci tutorials

`sbi4cogsci_style.py` gives every tutorial one visual language, so a colour
means the same thing on Day 2 afternoon as it does on Day 3 morning. Import it
from any tutorial folder:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))   # -> tutorials/
import sbi4cogsci_style as S
S.use()
```

## The palette is semantic, not decorative

Pick a colour by the **role the series plays**, never by the order it happens
to be plotted.

| name | hex | means |
|---|---|---|
| `S.PRIMARY` | `#2a78d6` blue | the method being advocated — NUTS, non-centered, the well-designed case |
| `S.NAIVE` | `#4a3aa7` violet | the comparator that struggles — Metropolis, centered, the degenerate case |
| `S.ALT` | `#1baf7a` aqua | a third series — Slice, an intermediate setting |
| `S.DIVERGENT` | `#d03b3b` red | divergences, funnel necks, failure regions |
| `S.TRUTH` | `#0b0b0b` ink | ground truth — **always dashed**, via the helpers below |
| `S.MUTED` | `#898781` grey | context, annotation, de-emphasised marks |

## Helpers

Use these rather than drawing truth by hand, so it can never be mistaken for a
series:

```python
S.truth_line(ax, value, axis="y")   # dashed black reference line
S.truth_point(ax, x, y)             # black X marker in parameter space
S.divergences(ax, x, y)             # red overlay, drawn on top
S.annotate(ax, "text", xy, xytext)  # muted callout arrow
```

## Why these colours

They are validated steps, checked **all-pairs** on the light surface
(`#fcfcfb`): worst CVD ΔE 9.9 (target ≥ 8), worst normal-vision ΔE 16.3
(floor ≥ 15).

The obvious assignment — orange for "the failing method" — was rejected. Orange
sits at normal-vision ΔE 10.8 against the red used for divergences, well under
the floor, and those two co-occur in the funnel scatter. Violet clears both
gates.

⚠️ **`S.ALT` sits at 2.74:1 contrast, below 3:1.** Any chart using it must
carry a legend or direct label — that is the relief the low contrast requires,
not an optional nicety.

## Scope

Light-mode only, by design: figures are baked into notebooks as static images
and the site theme is light, so there is no dark variant to switch to.

To re-validate after changing any colour:

```bash
node <dataviz-skill>/scripts/validate_palette.js "#2a78d6,#4a3aa7,#1baf7a,#d03b3b" \
  --mode light --pairs all
```
