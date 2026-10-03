#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sector-phase map for Paper 1b (Fig. 4).

DETERMINISTIC: plots deposited, verified data only. No RNG, no computation --
every cell is one entry of s_delta_sweeps.json, read and drawn.

Data source (deposited in this record, concept DOI 10.5281/zenodo.19601352):
  - s_delta_sweeps.json   two encodings, five lengths each, 50 sector phases
                          per length; the optimal sequence of each length.

WHY STRIPS AND NOT A SURFACE. The sector phase delta is sampled on a full,
gapless period (50 points, step 2*pi/50). The length L is not: the deposit
holds L in {3, 6, 8, 10, 12}. Drawing a surface over (L, delta) would have to
invent the intermediate lengths -- by interpolation or by empty rows. Both are
claims the data does not support. L is therefore a CATEGORICAL axis: five
separated strips, with nothing between them. Every drawn cell corresponds to
exactly one deposited value.

The colour scale is shared by both panels and runs from the classical bound
2 to the Tsirelson bound 2*sqrt(2); the two panels are only comparable if the
scale is.

Output: fig4_sector_map.png (2100 px wide, 300 dpi)
"""
import sys

sys.dont_write_bytecode = True

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("P1B_ANALYSIS_DATA", _PKG)
OUT = os.path.dirname(os.path.abspath(__file__))

CLASSICAL = 2.0
TSIRELSON = 2.0 * math.sqrt(2.0)
PANELS = [("encoding_A_d1b_2gen", "(a) two-generator $d_{1b}$ circuit model"),
          ("encoding_B_5d_3gen", "(b) three-generator encoding")]

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
})


def load():
    with open(os.path.join(DATA, "s_delta_sweeps.json"), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    d = load()
    for enc, _ in PANELS:
        assert enc in d, "missing encoding %s" % enc
        for L, cell in d[enc].items():
            assert len(cell["S"]) == 50, "%s L=%s: %d phases" % (enc, L, len(cell["S"]))
            assert len(cell["delta"]) == 50
            assert all(v is not None for v in cell["S"]), "%s L=%s: None in S" % (enc, L)

    # No sharey: the two panels have DIFFERENT sequences per strip, and a
    # shared y axis would give both the labels of whichever is set last.
    # Height 2.45 in, not 2.6: at 2.6 the figure plus its caption overran the
    # text block by 5.1 pt (Overfull \vbox during \output).
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    norm = Normalize(vmin=CLASSICAL, vmax=TSIRELSON)
    im = None
    for ax, (enc, title) in zip(axes, PANELS):
        lengths = sorted(d[enc], key=int)
        rows = [d[enc][L]["S"] for L in lengths]
        # Each strip is drawn separately so that nothing is implied between them.
        for k, (L, row) in enumerate(zip(lengths, rows)):
            im = ax.imshow(np.array(row)[None, :], aspect="auto", norm=norm,
                           cmap="viridis", origin="lower",
                           extent=(0.0, 2.0, k + 0.18, k + 0.82))
        ax.set_yticks([k + 0.5 for k in range(len(lengths))])
        ax.set_yticklabels(["$L=%s$" % L for L in lengths])
        # Sequence of this panel, printed inside the strip (one per row).
        # White alone is not enough: the strips run from dark blue to yellow,
        # and at the bright end (S near the Tsirelson bound) white on viridis
        # falls to a contrast ratio of 1.26:1 -- exactly on the cells that carry
        # the strongest result. Black alone is worse (the dark strip starts).
        # A thin dark outline carries against both ends of the scale.
        for k, L in enumerate(lengths):
            ax.text(0.02, k + 0.5, d[enc][L]["seq"], transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7.0, color="white",
                    family="monospace",
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
        ax.set_ylim(-0.1, len(lengths))
        ax.set_xlim(0.0, 2.0)
        ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        ax.set_xlabel(r"sector phase $\delta$")
        ax.set_title(title)
        ax.tick_params(length=2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    cb = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.015)
    cb.set_label(r"$|S|$")
    cb.set_ticks([CLASSICAL, 2.2, 2.4, 2.6, TSIRELSON])
    cb.set_ticklabels(["2.0", "2.2", "2.4", "2.6", r"$2\sqrt{2}$"])

    out = os.path.join(OUT, "fig4_sector_map.png")
    tmp = out + ".raw.png"
    fig.savefig(tmp)
    plt.close(fig)
    # PIL re-save: strips the matplotlib Software tEXt chunk
    from PIL import Image
    with Image.open(tmp) as im2:
        clean = Image.new(im2.mode, im2.size)
        clean.putdata(list(im2.getdata()))
        clean.save(out, format="PNG", dpi=(300, 300))
    os.remove(tmp)
    with Image.open(out) as im3:
        print("wrote %s (%d x %d px)" % (os.path.basename(out), im3.size[0], im3.size[1]))


if __name__ == "__main__":
    main()
