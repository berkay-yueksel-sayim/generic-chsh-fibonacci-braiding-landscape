#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figures for Paper 1b (CHSH-Form Values Above 2 in Fibonacci
Anyon Braiding: A Complete Landscape of Braid Words up to Length 12).  DETERMINISTIC: plots pre-computed,
verified Zenodo-bound data only (no RNG, no AI-generated physics).

Data sources (all deposited in this record, concept-DOI 10.5281/zenodo.19601352):
  - landscape_validation.json        (Tab. III: S5_max/S4_max + violation fractions per L)
  - s_delta_sweeps.json              (S(delta) sector-phase sweeps, 3-gen encoding;
                                      per-L sweep maxima feed the Fig. 1a top curve)
  - topology_switch_results.json     (sigma_3 block on/off)

Output: fig1_landscape.png, fig2_sector_phase.png, fig3_sigma3_switch.png  (300 dpi)
"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data files default to the deposit package root (the parent directory of
# figures/); the environment variables remain available as overrides.
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("P1B_ANALYSIS_DATA", _PKG)
LANDSCAPE = os.environ.get("P1B_LANDSCAPE_JSON",
                           os.path.join(_PKG, "landscape_validation.json"))
OUT = os.path.dirname(os.path.abspath(__file__))

TSIRELSON = 2.0 * math.sqrt(2.0)   # 2.8284271...
CLASSICAL = 2.0

# Wong colorblind-safe palette
BLUE, ORANGE, GREEN, VERM, PURPLE, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#555555"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

def _bound_lines(ax, xlo, xhi, label=True):
    ax.hlines(TSIRELSON, xlo, xhi, color=GRAY, ls="--", lw=0.9,
              label=r"Tsirelson bound $2\sqrt{2}$" if label else None, zorder=1)
    ax.hlines(CLASSICAL, xlo, xhi, color=GRAY, ls=":", lw=0.9,
              label=r"classical bound $|S|=2$" if label else None, zorder=1)


def main():
    # ----------------------------------------------------------------- load
    land = json.load(open(LANDSCAPE, encoding="utf-8"))["results_per_length"]
    def Lkey(k):  # results keyed "L=3"
        return int(str(k).split("=")[-1])
    rows = sorted(((Lkey(k), v) for k, v in land.items()), key=lambda x: x[0])
    Ls   = [L for L, _ in rows]
    S5   = [v["S5_max"] for _, v in rows]
    S4   = [v["S4_max"] for _, v in rows]
    f5   = [v["bell_pct_5D"] for _, v in rows]
    f4   = [v["bell_pct_4D"] for _, v in rows]
    sweeps = json.load(open(os.path.join(DATA, "s_delta_sweeps.json"), encoding="utf-8"))
    tsw    = json.load(open(os.path.join(DATA, "topology_switch_results.json"), encoding="utf-8"))

    # 3-generator encoding at the sector-phase optimum: per-L maxima of the
    # deposited S(delta) sweeps (encoding B winning sequences; L <= 9 subset
    # to match the x-range of the protocol curves)
    gL = [L for L in sorted(int(k) for k in sweeps["encoding_B_5d_3gen"]) if L <= 9]
    gS = [max(sweeps["encoding_B_5d_3gen"][str(L)]["S"]) for L in gL]

    # ====================================================== FIG 1: landscape ladder
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    a1.plot(gL, gS, "-o", color=BLUE, ms=4,
            label=r"3-generator, sector-phase opt.")
    a1.plot(Ls, S4, "-s", color=ORANGE, ms=3.5, label=r"Protocol A, 4D projection (Tab. III)")
    a1.plot(Ls, S5, "-^", color=GREEN, ms=3.5, label=r"Protocol B, native 5D (Tab. III)")
    _bound_lines(a1, min(Ls), max(Ls))
    a1.set_xlabel(r"braid length $L$"); a1.set_ylabel(r"maximal CHSH value $|S|_{\max}$")
    a1.set_ylim(1.95, 2.90); a1.set_xticks(Ls)
    a1.legend(loc="lower right", framealpha=0.9, handlelength=1.6)
    a1.text(-0.02, 1.02, r"(a)", transform=a1.transAxes, fontweight="bold", va="bottom")

    a2.plot(Ls, f4, "-s", color=ORANGE, ms=3.5, label=r"4D projection (Protocol A)")
    a2.plot(Ls, f5, "-^", color=GREEN, ms=3.5, label=r"native 5D (Protocol B)")
    a2.set_xlabel(r"braid length $L$"); a2.set_ylabel(r"CHSH-violation fraction (%)")
    a2.set_xticks(Ls); a2.set_ylim(0, 100)
    a2.legend(loc="lower right", framealpha=0.9, handlelength=1.6)
    a2.text(-0.02, 1.02, r"(b)", transform=a2.transAxes, fontweight="bold", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_landscape.png"))
    plt.close(fig)

    # ====================================================== FIG 2: sector-phase S(delta)
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    encB = sweeps["encoding_B_5d_3gen"]
    style = {3: (GREEN, 1.4, 1.0), 8: (BLUE, 1.5, 1.0), 12: (VERM, 1.0, 0.8)}
    for L in [3, 8, 12]:
        e = encB[str(L)]
        col, lw, al = style[L]
        dp = [d / math.pi for d in e["delta"]]
        ax.plot(dp, e["S"], color=col, lw=lw, alpha=al, label=r"$L=%d$" % L)
    _bound_lines(ax, 0, 2, label=True)
    ax.set_xlabel(r"sector phase $\delta/\pi$"); ax.set_ylabel(r"CHSH value $|S|(\delta)$")
    ax.set_xlim(0, 2); ax.set_ylim(1.4, 2.92)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              framealpha=0.9, handlelength=1.5, columnspacing=1.1, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_sector_phase.png"))
    plt.close(fig)

    # ====================================================== FIG 3: sigma_3 switch
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ent = {(e["seq"], e["block"]): e for e in tsw["three_generator_sigma3_block"]}
    seq = "23444432"
    on  = ent[(seq, None)]; off = ent[(seq, "3")]
    dpon  = [d / math.pi for d in on["delta"]]
    dpoff = [d / math.pi for d in off["delta"]]
    ax.plot(dpon, on["S"],  color=BLUE, lw=1.5, label=r"$\sigma_3$ active")
    ax.plot(dpoff, off["S"], color=VERM, lw=1.5, ls="--", label=r"$\sigma_3$ blocked")
    ax.fill_between(dpoff, off["S"], CLASSICAL, color=VERM, alpha=0.08)
    _bound_lines(ax, 0, max(dpon))
    ax.set_xlabel(r"sector phase $\delta/\pi$"); ax.set_ylabel(r"CHSH value $|S|(\delta)$")
    ax.set_xlim(0, max(dpon)); ax.set_ylim(1.9, 2.92)
    ax.legend(loc="upper right", framealpha=0.9, handlelength=1.8)
    ax.set_title(r"3-gen sequence $\sigma_2\sigma_3\sigma_4^4\sigma_3\sigma_2$ ($L=8$)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_sigma3_switch.png"))
    plt.close(fig)

    # ----------------------------------------------------------------- provenance echo
    print("FIG1 ladder: 3gen L=%s |S|=%s ; S4(L=8)=%.4f S5(L=8)=%.4f ; f4(L=9)=%.1f f5(L=9)=%.1f"
          % (gL, gS, S4[Ls.index(8)], S5[Ls.index(8)], f4[-1], f5[-1]))
    print("FIG2 encB L=8 seq=%s S(0)=%.4f (=2sqrt2? %.4f)" % (encB["8"]["seq"], encB["8"]["S"][0], TSIRELSON))
    print("FIG3 switch seq=%s : sigma3-active S_max=%.4f ; sigma3-blocked S_max=%.4f C_max=%.2e"
          % (seq, on["S_max"], off["S_max"], off["C_max"]))
    print("OK -> fig1_landscape.png, fig2_sector_phase.png, fig3_sigma3_switch.png")


if __name__ == "__main__":
    main()
