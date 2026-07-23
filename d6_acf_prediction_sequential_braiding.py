#!/usr/bin/env python3
"""
D6 -- Sequential-braiding autocorrelation prediction (Sec. VI, Table tab:acf)
==============================================================================
Reproduces the paper's falsifiable prediction: modeling the source as a
sequence of braiding steps on the shared six-anyon fusion space (5D,
three generators sigma2/sigma3/sigma4, sector phase delta=0), the CHSH
time series |S|_n develops positive lag-1 autocorrelation under models
A (i.i.d. generator choice) and B (Markov-persistent generator choice),
while model C (local generators only, no cross-bipartition sigma3) gives
identically zero autocorrelation.

This is the source side of the prediction (Table tab:acf: A +0.343 at
10.85 sigma, B +0.361 at 11.24 sigma, C exactly 0). The independent
experimental null control on real loophole-free Bell-test data is in
d7_acf_hensen2015_null_control.py.

Deterministic: fixed seeds throughout.
"""
from __future__ import annotations
import json
import math
import sys
import io
from pathlib import Path
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).parent

PHI = (1 + math.sqrt(5)) / 2
F2 = np.array([[1/PHI, 1/math.sqrt(PHI)],
               [1/math.sqrt(PHI), -1/PHI]], dtype=complex)
R1 = np.exp(-4j * math.pi / 5)
RT = np.exp(3j * math.pi / 5)


def braid_block(delta: float) -> np.ndarray:
    R = np.diag([R1, RT * np.exp(1j * delta)])
    return F2 @ R @ F2


def sigma2_5d(delta: float = 0.0) -> np.ndarray:
    r_tau = RT * np.exp(1j * delta)
    B = braid_block(delta)
    M = np.zeros((5, 5), dtype=complex)
    M[0, 0] = B[0, 0]; M[0, 3] = B[0, 1]
    M[3, 0] = B[1, 0]; M[3, 3] = B[1, 1]
    M[1, 1] = B[0, 0]; M[1, 4] = B[0, 1]
    M[4, 1] = B[1, 0]; M[4, 4] = B[1, 1]
    M[2, 2] = r_tau
    return M


def sigma3_5d(delta: float = 0.0) -> np.ndarray:
    r_tau = RT * np.exp(1j * delta)
    B = braid_block(delta)
    M = np.zeros((5, 5), dtype=complex)
    M[0, 0] = R1
    M[1, 1] = r_tau
    M[3, 3] = r_tau
    M[2, 2] = B[0, 0]; M[2, 4] = B[0, 1]
    M[4, 2] = B[1, 0]; M[4, 4] = B[1, 1]
    return M


def sigma4_5d(delta: float = 0.0) -> np.ndarray:
    r_tau = RT * np.exp(1j * delta)
    B = braid_block(delta)
    M = np.zeros((5, 5), dtype=complex)
    M[0, 0] = B[0, 0]; M[0, 1] = B[0, 1]
    M[1, 0] = B[1, 0]; M[1, 1] = B[1, 1]
    M[3, 3] = B[0, 0]; M[3, 4] = B[0, 1]
    M[4, 3] = B[1, 0]; M[4, 4] = B[1, 1]
    M[2, 2] = r_tau
    return M


def project_5d_to_2qubit(psi5: np.ndarray) -> np.ndarray:
    psi4 = np.array([psi5[0], psi5[1], psi5[3], psi5[2] + psi5[4]], dtype=complex)
    n = np.linalg.norm(psi4)
    return psi4 / n if n > 1e-15 else psi4


def concurrence(psi: np.ndarray) -> float:
    return float(2 * abs(psi[0] * psi[3] - psi[1] * psi[2]))


def chsh_horodecki(C: float) -> float:
    return 2.0 * math.sqrt(1.0 + C * C)


GATES = {'2': sigma2_5d(0.0), '3': sigma3_5d(0.0), '4': sigma4_5d(0.0)}


def sample_generator(model: str, prev, rng: np.random.Generator) -> str:
    if model == 'A':
        return rng.choice(['2', '3', '4'])
    if model == 'B':
        if prev is None:
            return rng.choice(['2', '3', '4'])
        u = rng.random()
        if u < 0.6:
            return prev
        others = [g for g in '234' if g != prev]
        return others[0] if u < 0.8 else others[1]
    if model == 'C':
        return rng.choice(['2', '4'])
    raise ValueError(model)


def run_trajectory(model: str, N: int, seed: int, psi0=None):
    rng = np.random.default_rng(seed)
    psi = np.array([1, 0, 0, 0, 0], dtype=complex) if psi0 is None else psi0.copy()
    S_arr = np.zeros(N); C_arr = np.zeros(N)
    prev = None
    for n in range(N):
        g = sample_generator(model, prev, rng)
        psi = GATES[g] @ psi
        nm = np.linalg.norm(psi)
        if nm > 1e-15:
            psi = psi / nm
        psi4 = project_5d_to_2qubit(psi)
        C = concurrence(psi4)
        C_arr[n] = C
        S_arr[n] = chsh_horodecki(C)
        prev = g
    return S_arr, C_arr


def acf(x: np.ndarray, kmax: int) -> np.ndarray:
    x = x - x.mean()
    var = np.dot(x, x)
    if var < 1e-15:
        return np.zeros(kmax + 1)
    out = np.zeros(kmax + 1)
    out[0] = 1.0
    for k in range(1, kmax + 1):
        out[k] = np.dot(x[:-k], x[k:]) / var
    return out


def shuffle_band(x: np.ndarray, kmax: int, n_perm: int, seed: int):
    rng = np.random.default_rng(seed)
    accum = np.zeros((n_perm, kmax + 1))
    for i in range(n_perm):
        xs = rng.permutation(x)
        accum[i] = acf(xs, kmax)
    lo = np.percentile(accum, 2.5, axis=0)
    hi = np.percentile(accum, 97.5, axis=0)
    return lo, hi, accum.std(axis=0)


def run_model(model: str, N: int, M: int, kmax: int, base_seed: int) -> dict:
    S_runs, acf_runs = [], []
    for r in range(M):
        S, _ = run_trajectory(model, N, base_seed + r)
        S_runs.append(S)
        acf_runs.append(acf(S, kmax))
    S_runs = np.array(S_runs); acf_runs = np.array(acf_runs)
    acf_mean = acf_runs.mean(axis=0)
    lo, hi, _ = shuffle_band(S_runs[0], kmax, n_perm=1000, seed=base_seed + 999)
    acf1_mean = acf_mean[1]
    shuffle_std = (hi[1] - lo[1]) / (2 * 1.96)
    sigma_above = abs(acf1_mean) / shuffle_std if shuffle_std > 1e-12 else 0.0
    bell_fraction = float((S_runs > 2.0).mean())
    return dict(model=model, N=N, M=M,
                S_mean=float(S_runs.mean()), S_std=float(S_runs.std()),
                bell_fraction=bell_fraction,
                acf1_mean=float(acf1_mean), shuffle_std=float(shuffle_std),
                sigma_above_shuffle=float(sigma_above))


def main():
    print("=" * 72)
    print("  D6 -- Sequential-braiding ACF prediction (Table tab:acf)")
    print("=" * 72)
    N, M, kmax, base = 1000, 20, 50, 20260415

    results = {}
    for m in "ABC":
        r = run_model(m, N, M, kmax, base + ord(m))
        results[m] = r
        print(f"  Model {m}:  ACF(1)={r['acf1_mean']:+.4f}  "
              f"sigma_above_shuffle={r['sigma_above_shuffle']:.2f}  "
              f"CHSH%={r['bell_fraction']*100:.1f}")

    expected = {"A": (0.343, 10.85), "B": (0.361, 11.24), "C": (0.0, 0.0)}
    checks = {}
    for m, (exp_acf, exp_sig) in expected.items():
        r = results[m]
        checks[f"model_{m}_acf1_matches_table"] = abs(round(r["acf1_mean"], 3) - exp_acf) < 0.001
        checks[f"model_{m}_sigma_matches_table"] = abs(round(r["sigma_above_shuffle"], 2) - exp_sig) < 0.02
    checks["ALL_PASS"] = bool(all(checks.values()))

    out = dict(meta=dict(script="d6_acf_prediction_sequential_braiding.py", N=N, M=M, kmax=kmax,
                          base_seed=base, encoding="5D 3-gen", delta=0.0,
                          table_reference="Table tab:acf (Sec. VI, Pillar 3)"),
                results=results, checks=checks)
    (HERE / "d6_acf_prediction_results.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  ALL_PASS = {checks['ALL_PASS']}")
    print(f"  WROTE {HERE / 'd6_acf_prediction_results.json'}")


if __name__ == "__main__":
    main()
