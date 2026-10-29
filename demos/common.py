"""Shared pieces: the naive measures each demo breaks, and the figure style.

The measures here are deliberately ordinary. Percentile normalisation, a percentile
threshold, a mass centroid: the kind of code most hand-built image measures contain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
# The notan repository checked out beside this one (as "notan" from GitHub, or "parallax-notan").
NOTAN = next((p for p in (ROOT.parent / "notan", ROOT.parent / "parallax-notan") if p.exists()), None)
if NOTAN is None:
    sys.exit("demos need the notan repository checked out beside this folder")
sys.path.insert(0, str(NOTAN / "src"))
sys.path.insert(0, str(NOTAN / "tests"))

INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ACCENT, BLUE, PLANE = "#eb6834", "#2f6db5", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": PLANE, "axes.facecolor": PLANE, "savefig.facecolor": PLANE,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

NUMBERS: dict = {}


def robust01(x, lo=2.0, hi=98.0):
    """The usual percentile stretch. Returns zeros when the two percentiles coincide."""
    x = np.asarray(x, dtype=np.float64)
    a, b = np.percentile(x, [lo, hi])
    if b <= a + 1e-9:
        return np.zeros_like(x)
    return np.clip((x - a) / (b - a), 0, 1)


def minmax01(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def centroid(w):
    """Weighted centroid as (x, y) offsets from the frame centre, in frame fractions."""
    h, wd = w.shape
    tot = float(w.sum())
    if tot <= 0:
        return 0.0, 0.0
    yy, xx = np.mgrid[0:h, 0:wd]
    return float((xx * w).sum() / tot / (wd - 1) - 0.5), float((yy * w).sum() / tot / (h - 1) - 0.5)


def lum(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def show(ax, img, title="", cmap="gray", vmin=0, vmax=1):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(RULE)
    if title:
        ax.set_title(title)


def save(fig, name):
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  {path.name}")


def record(key, **vals):
    NUMBERS[key] = {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in vals.items()}


def write_numbers():
    (FIG / "numbers.json").write_text(json.dumps(NUMBERS, indent=2) + "\n")
