#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-render 1b Fig.2 (sector-phase S(delta)) and Fig.3 (sigma_3
switch) at finer plot resolution. NO recomputation -- the PUBLISHED 50-point
S(delta) data is shape-preservingly (PCHIP, overshoot-free) interpolated to 400
points so the L=12 / sigma_3-active curves read smoothly. The interpolant passes
EXACTLY through every published sample (positive control asserted): same data,
finer plot resolution, no value changed.

Data (verified, concept-DOI 10.5281/zenodo.19601352):
  s_delta_sweeps.json (encoding_B 3-gen S(delta)) · topology_switch_results.json (sigma_3 on/off)
"""
import json, math, os
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data files default to the deposit package root (the parent directory of
# figures/); the environment variable remains available as an override.
DATA = os.environ.get("P1B_ANALYSIS_DATA",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.dirname(os.path.abspath(__file__))
TS = 2*math.sqrt(2); CL = 2.0
BLUE,ORANGE,GREEN,VERM,GRAY = "#0072B2","#E69F00","#009E73","#D55E00","#555555"

def dense(delta, S, n=400):
    """Shape-preserving PCHIP interpolation of evenly-spaced periodic samples.
    Wrap one period so the curve is continuous on [0, 2pi]; returns (x/pi, y)."""
    d = list(delta) + [2*math.pi]          # wrap point delta=2pi := delta=0 value
    y = list(S) + [S[0]]
    p = PchipInterpolator(d, y)
    xs = np.linspace(0, 2*math.pi, n)
    ys = p(xs)
    # positive control: exact at the published nodes
    err = float(np.max(np.abs(p(np.array(delta)) - np.array(S))))
    return xs/math.pi, ys, err

plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],
    "mathtext.fontset":"dejavuserif","font.size":9,"axes.labelsize":9,
    "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7,
    "axes.linewidth":0.8,"lines.linewidth":1.3,"savefig.dpi":300,
    "savefig.bbox":"tight","savefig.pad_inches":0.02,"axes.grid":True,
    "grid.alpha":0.25,"grid.linewidth":0.5})

def bounds(ax):
    ax.hlines(TS,0,2,color=GRAY,ls="--",lw=0.9,label=r"Tsirelson bound $2\sqrt{2}$")
    ax.hlines(CL,0,2,color=GRAY,ls=":",lw=0.9,label=r"classical bound $|S|=2$")

maxerr = 0.0

# ---------------- FIG 2 sector-phase (PCHIP-dense), L=3,8,12 ----------------
encB = json.load(open(os.path.join(DATA,"s_delta_sweeps.json"),encoding="utf-8"))["encoding_B_5d_3gen"]
fig,ax=plt.subplots(figsize=(3.5,2.9))
sty={"3":(GREEN,1.4,1.0),"8":(BLUE,1.5,1.0),"12":(VERM,1.1,0.85)}
for L in ["3","8","12"]:
    e=encB[L]; x,y,err=dense(e["delta"], e["S"]); maxerr=max(maxerr,err)
    col,lw,al=sty[L]
    ax.plot(x,y,color=col,lw=lw,alpha=al,label=r"$L=%s$"%L)
bounds(ax)
ax.set_xlabel(r"sector phase $\delta/\pi$"); ax.set_ylabel(r"CHSH value $|S|(\delta)$")
ax.set_xlim(0,2); ax.set_ylim(1.4,2.92)
ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=3,framealpha=0.9,handlelength=1.5,columnspacing=1.1,fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig2_sector_phase.png")); plt.close(fig)

# ---------------- FIG 3 sigma_3 switch (PCHIP-dense), L=8 seq 23444432 ----------------
tsw = json.load(open(os.path.join(DATA,"topology_switch_results.json"),encoding="utf-8"))
ent = {(e["seq"], e["block"]): e for e in tsw["aufgabe1_3gen_sigma3_block"]}
seq="23444432"; on=ent[(seq,None)]; off=ent[(seq,"3")]
xon,yon,e1=dense(on["delta"], on["S"]); xoff,yoff,e2=dense(off["delta"], off["S"])
maxerr=max(maxerr,e1,e2)
fig,ax=plt.subplots(figsize=(3.4,2.7))
ax.plot(xon,yon,color=BLUE,lw=1.5,label=r"$\sigma_3$ active")
ax.plot(xoff,yoff,color=VERM,lw=1.5,ls="--",label=r"$\sigma_3$ blocked")
ax.fill_between(xoff,yoff,CL,color=VERM,alpha=0.08)
bounds(ax)
ax.set_xlabel(r"sector phase $\delta/\pi$"); ax.set_ylabel(r"CHSH value $|S|(\delta)$")
ax.set_xlim(0,2); ax.set_ylim(1.9,2.92)
ax.legend(loc="upper right",framealpha=0.9,handlelength=1.8)
ax.set_title(r"3-gen sequence $\sigma_2\sigma_3\sigma_4^4\sigma_3\sigma_2$ ($L=8$)",fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig3_sigma3_switch.png")); plt.close(fig)

assert maxerr < 1e-9, f"PCHIP not interpolating at nodes! maxerr={maxerr}"
print(f"POSITIVE CONTROL OK: interpolant exact at all published nodes (max node err {maxerr:.1e})")
print(f"sigma3 blocked S(published)=const? max={max(off['S']):.4f} min={min(off['S']):.4f}")
print("OK -> fig2_sector_phase.png, fig3_sigma3_switch.png (PCHIP-dense, 400 pts; same data)")
