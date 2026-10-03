#!/usr/bin/env python3
"""
D11 -- Fourier fit of the sector-phase sweeps S(delta) (Appendix; Fig. 2 lengths)
==================================================================================
Reads s_delta_sweeps.json (encoding_B_5d_3gen, the three-generator winners for
L = 3, 8, 12 -- the lengths drawn in Fig. 2) and fits, by least squares,

    S(delta_k) ~ a_0 + sum_{n=1}^{N} (a_n cos(n delta_k) + b_n sin(n delta_k)),
    delta_k = 2 pi k / 50,  k = 0..49,

for N = 1..6. For every (L, N) it reports R^2 = 1 - SS_res/SS_tot, the
coefficients, the harmonic amplitudes sqrt(a_n^2 + b_n^2) and the leading
harmonic (largest amplitude).

Two independent routes are computed and must agree to 1e-10 (assert):
numpy.linalg.lstsq on the design matrix, and the normal equations
(X^T X) c = X^T y solved directly. A positive control on a synthetic signal
(a_0 + a_2 cos 2 delta) must return leading harmonic n = 2 and R^2 = 1 for
N >= 2. R^2 is checked to be non-decreasing in N (nested models).

Deterministic, numpy only. Output: d11_fourier_fit_sweeps_results.json
(keys fit_L{L}_N{N}).
"""
from __future__ import annotations
import json
import math
import sys
import io
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
LENGTHS = (3, 8, 12)
N_MAX = 6
ROUTE_TOL = 1e-10


def design(delta: np.ndarray, N: int) -> np.ndarray:
    cols = [np.ones_like(delta)]
    for n in range(1, N + 1):
        cols.append(np.cos(n * delta))
        cols.append(np.sin(n * delta))
    return np.column_stack(cols)


def fit(delta: np.ndarray, y: np.ndarray, N: int) -> dict:
    X = design(delta, N)
    c_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]                 # route 1
    c_normal = np.linalg.solve(X.T @ X, X.T @ y)                   # route 2
    route_diff = float(np.max(np.abs(c_lstsq - c_normal)))
    assert route_diff <= ROUTE_TOL, f"routes disagree: {route_diff:.3e}"
    c = c_lstsq
    yhat = X @ c
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    a = [float(c[2 * n - 1]) for n in range(1, N + 1)]
    b = [float(c[2 * n]) for n in range(1, N + 1)]
    amp = [math.hypot(a[n - 1], b[n - 1]) for n in range(1, N + 1)]
    lead = int(np.argmax(amp)) + 1
    return dict(N=N, a0=float(c[0]), a=a, b=b, amplitudes=amp,
                leading_harmonic=lead, leading_amplitude=float(amp[lead - 1]),
                R2=float(r2), SS_res=ss_res, SS_tot=ss_tot, route_max_diff=route_diff)


def positive_control() -> dict:
    delta = 2 * math.pi * np.arange(50) / 50
    y = 2.3 + 0.3 * np.cos(2 * delta)
    r1 = fit(delta, y, 1)
    r2 = fit(delta, y, 2)
    ok = (r2["leading_harmonic"] == 2 and abs(r2["R2"] - 1.0) < 1e-12
          and abs(r2["leading_amplitude"] - 0.3) < 1e-12 and abs(r2["a0"] - 2.3) < 1e-12
          and r1["R2"] < 1e-12)
    assert ok, "positive control failed (synthetic a0 + a2 cos 2 delta)"
    return dict(signal="2.3 + 0.3 cos(2 delta) on the 50-point grid",
                R2_N1=r1["R2"], R2_N2=r2["R2"], leading_N2=r2["leading_harmonic"], ok=ok)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sweeps = json.loads((HERE / "s_delta_sweeps.json").read_text(encoding="utf-8"))["encoding_B_5d_3gen"]
    print("=" * 72)
    print("  D11 -- Fourier fit of S(delta), encoding B (three generators), L = 3, 8, 12")
    print("=" * 72)
    pc = positive_control()
    print(f"\n  positive control (synthetic cos 2 delta): leading n = {pc['leading_N2']}, "
          f"R2(N=1) = {pc['R2_N1']:.1e}, R2(N=2) = {pc['R2_N2']:.12f}  ok = {pc['ok']}")
    out = dict(meta=dict(script="d11_fourier_fit_sweeps.py", source="s_delta_sweeps.json / encoding_B_5d_3gen",
                         lengths=list(LENGTHS), N_range=[1, N_MAX], grid="delta_k = 2 pi k / 50, k = 0..49",
                         model="a0 + sum_{n=1}^{N} (a_n cos n delta + b_n sin n delta)",
                         R2="1 - SS_res/SS_tot", routes="numpy.linalg.lstsq and normal equations, "
                         f"max |diff| asserted <= {ROUTE_TOL:g}", words={}),
               positive_control=pc)
    route_worst = 0.0
    print(f"\n  {'L':>2}  word            N   R2          leading n  amplitude   routes |diff|")
    for L in LENGTHS:
        e = sweeps[str(L)]
        out["meta"]["words"][str(L)] = e["seq"]
        delta = np.array(e["delta"]); y = np.array(e["S"])
        prev = -1.0
        for N in range(1, N_MAX + 1):
            r = fit(delta, y, N)
            assert r["R2"] >= prev - 1e-12, f"R2 decreased with N at L={L}, N={N}"
            prev = r["R2"]
            route_worst = max(route_worst, r["route_max_diff"])
            out[f"fit_L{L}_N{N}"] = dict(L=L, word=e["seq"], **r)
            print(f"  {L:>2}  {e['seq']:<14}  {N}   {r['R2']:.4f}      {r['leading_harmonic']}          "
                  f"{r['leading_amplitude']:.4f}      {r['route_max_diff']:.1e}")
    out["meta"]["route_max_diff_over_all_fits"] = route_worst
    (HERE / "d11_fourier_fit_sweeps_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  routes agree to {route_worst:.1e} over all fits")
    print(f"  WROTE {HERE / 'd11_fourier_fit_sweeps_results.json'}")


if __name__ == "__main__":
    main()
