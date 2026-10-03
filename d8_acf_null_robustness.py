#!/usr/bin/env python3
"""
D8 -- ACF null test and initialization robustness (Sec. VI, Pillars 2 + Robustness)
====================================================================================
Reproduces, with a documented seed, the two remaining ACF results of
Sec. VI that previously had no deposited generator:

(a) Null test (Pillar 2): drawing N = 10,000 i.i.d. sector phases
    delta_i ~ U[0, 2*pi) and evaluating the closed-form CHSH value of a
    fixed winning braid word at each phase, the lag-1 autocorrelation of
    the resulting series is consistent with zero (95% CI +-1.96/sqrt(N)).
    Two configurations are evaluated:
      - "Model A configuration": the five-dimensional three-generator
        encoding used by the sequential-braiding source models
        (sigma2/sigma3/sigma4), represented by the L = 12 winning word
        232222343322 -- the delta = 0 maximizer under Protocol A from
        d10_protocol_a_projection_audit.py (delta0_first_maximizer;
        s_delta_sweeps.json / Fig. 2) -- CHSH via the Protocol A
        projection and the Horodecki closed form;
      - "d1b configuration": the two-generator d1b encoding,
        represented by the deposited L = 12 winning word ABABABABABAB
        (Table I headline row), same closed form.
    Since the phases are i.i.d., the CHSH series is i.i.d. and any
    apparent autocorrelation is purely statistical.

(b) Initialization robustness: repeating the sequential-braiding runs
    of d6_acf_prediction_sequential_braiding.py (Models A and B,
    N = 1000 steps, M = 20 runs) with the vacuum start replaced by a
    seeded random normalized five-dimensional initial state (one
    independent draw per run), the positive lag-1 autocorrelation
    survives.

All randomness is seeded (BASE_SEED below); rerunning this script
reproduces every number bit-for-bit, which the script itself verifies
via an internal determinism recheck of part (a).
"""
from __future__ import annotations
import json
import math
import sys
import io
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent

BASE_SEED = 20260723
# No bare threshold: S > 2 + TOL counts as violating, |S - 2| <= TOL is a boundary
# class reported separately (same convention as landscape_validation.py).
TOL = 1e-10
N_NULL = 10_000
N_STEPS = 1000
M_RUNS = 20
KMAX = 50

PHI = (1 + math.sqrt(5)) / 2
F2 = np.array([[1/PHI, 1/math.sqrt(PHI)],
               [1/math.sqrt(PHI), -1/PHI]], dtype=complex)
R1 = np.exp(-4j * math.pi / 5)
RT = np.exp(3j * math.pi / 5)

WORD_5D = "232222343322"     # L = 12 winner, encoding B: delta = 0 maximizer under Protocol A (d10; Fig. 2)
WORD_D1B = "ABABABABABAB"    # deposited L = 12 winner, encoding A (Table I)


# ----------------------------------------------------------------------
# d1b encoding (two generators AM, MB) -- identical to d4
# ----------------------------------------------------------------------
_STATES_D1B = [(0, 0, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1)]


def gate_AM_d1b(delta: float) -> np.ndarray:
    Rm = np.diag([R1, RT * np.exp(1j * delta)])
    Fd = F2.conj().T
    G = np.zeros((4, 4), dtype=complex)
    for j, (a_in, m_in, b_in) in enumerate(_STATES_D1B):
        for i, (a_out, m_out, b_out) in enumerate(_STATES_D1B):
            if b_in != b_out:
                continue
            val = 0.0 + 0j
            for c in range(2):
                val += F2[a_in, c] * Rm[c, c] * Fd[c, a_out]
            G[i, j] = val
    return G


def gate_MB_d1b(delta: float) -> np.ndarray:
    Rm = np.diag([R1, RT * np.exp(1j * delta)])
    Fd = F2.conj().T
    G = np.zeros((4, 4), dtype=complex)
    for j, (a_in, m_in, b_in) in enumerate(_STATES_D1B):
        for i, (a_out, m_out, b_out) in enumerate(_STATES_D1B):
            if a_in != a_out:
                continue
            val = 0.0 + 0j
            for c in range(2):
                val += F2[m_in, c] * Rm[c, c] * Fd[c, m_out]
            G[i, j] = val
    return G


def s_d1b(word: str, delta: float) -> float:
    G_AM = gate_AM_d1b(delta)
    G_MB = gate_MB_d1b(delta)
    psi = np.array([1, 0, 0, 0], dtype=complex)
    for ch in word:
        psi = (G_AM if ch == 'A' else G_MB) @ psi
    n = np.linalg.norm(psi)
    if n > 1e-15:
        psi = psi / n
    C = 2 * abs(psi[0] * psi[3] - psi[1] * psi[2])
    return 2.0 * math.sqrt(1.0 + C * C)


# ----------------------------------------------------------------------
# 5D encoding (three generators sigma2/sigma3/sigma4) -- identical to d4/d6
# ----------------------------------------------------------------------

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
    """Protocol A: |11>_L <- (|2>+|4>)/sqrt(2), then renormalize (landscape_validation.project_4d)."""
    psi4 = np.array([psi5[0], psi5[1], psi5[3], (psi5[2] + psi5[4]) / math.sqrt(2)], dtype=complex)
    n = np.linalg.norm(psi4)
    return psi4 / n if n > 1e-15 else psi4


def concurrence(psi: np.ndarray) -> float:
    return float(2 * abs(psi[0] * psi[3] - psi[1] * psi[2]))


def chsh_horodecki(C: float) -> float:
    return 2.0 * math.sqrt(1.0 + C * C)


def s_5d3gen(word: str, delta: float) -> float:
    gates = {'2': sigma2_5d(delta), '3': sigma3_5d(delta), '4': sigma4_5d(delta)}
    psi = np.array([1, 0, 0, 0, 0], dtype=complex)
    for ch in word:
        psi = gates[ch] @ psi
    psi4 = project_5d_to_2qubit(psi)
    return chsh_horodecki(concurrence(psi4))


# ----------------------------------------------------------------------
# ACF machinery -- identical to d6
# ----------------------------------------------------------------------

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


GATES0 = {'2': sigma2_5d(0.0), '3': sigma3_5d(0.0), '4': sigma4_5d(0.0)}


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
    raise ValueError(model)


def run_trajectory(model: str, N: int, seed: int, psi0: np.ndarray):
    rng = np.random.default_rng(seed)
    psi = psi0.copy()
    S_arr = np.zeros(N)
    prev = None
    for n in range(N):
        g = sample_generator(model, prev, rng)
        psi = GATES0[g] @ psi
        nm = np.linalg.norm(psi)
        if nm > 1e-15:
            psi = psi / nm
        psi4 = project_5d_to_2qubit(psi)
        S_arr[n] = chsh_horodecki(concurrence(psi4))
        prev = g
    return S_arr


def random_psi0(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=5) + 1j * rng.normal(size=5)
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------
# Part (a): null test
# ----------------------------------------------------------------------

def null_test(config: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    deltas = rng.uniform(0.0, 2 * math.pi, N_NULL)
    if config == "5d3gen_modelA":
        S = np.array([s_5d3gen(WORD_5D, d) for d in deltas])
    elif config == "d1b":
        S = np.array([s_d1b(WORD_D1B, d) for d in deltas])
    else:
        raise ValueError(config)
    a1 = float(acf(S, 1)[1])
    ci = 1.96 / math.sqrt(N_NULL)
    return dict(config=config, N=N_NULL, seed=seed, acf1=a1,
                ci95=ci, within_ci=bool(abs(a1) < ci),
                S_mean=float(S.mean()), S_std=float(S.std()))


# ----------------------------------------------------------------------
# Part (b): initialization robustness
# ----------------------------------------------------------------------

def robustness(model: str, base_seed: int) -> dict:
    init_rng = np.random.default_rng(base_seed + 555)
    S_runs, acf_runs = [], []
    for r in range(M_RUNS):
        psi0 = random_psi0(init_rng)
        S = run_trajectory(model, N_STEPS, base_seed + r, psi0)
        S_runs.append(S)
        acf_runs.append(acf(S, KMAX))
    S_runs = np.array(S_runs); acf_runs = np.array(acf_runs)
    acf1_mean = float(acf_runs.mean(axis=0)[1])
    lo, hi, _ = shuffle_band(S_runs[0], KMAX, n_perm=1000, seed=base_seed + 999)
    shuffle_std = (hi[1] - lo[1]) / (2 * 1.96)
    sigma_above = abs(acf1_mean) / shuffle_std if shuffle_std > 1e-12 else 0.0
    return dict(model=model, N=N_STEPS, M=M_RUNS,
                init="seeded random normalized 5D state (one draw per run)",
                acf1_mean=acf1_mean, shuffle_std=float(shuffle_std),
                sigma_above_shuffle=float(sigma_above),
                bell_fraction=float((S_runs > 2.0 + TOL).mean()),
                at_bound_fraction=float((np.abs(S_runs - 2.0) <= TOL).mean()))


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("=" * 72)
    print("  D8 -- ACF null test + initialization robustness")
    print("=" * 72)

    print(f"\n[Part a: null test, N={N_NULL} i.i.d. sector phases, "
          f"CI95=+-{1.96/math.sqrt(N_NULL):.4f}]")
    nulls = {}
    for cfg, seed_off in (("5d3gen_modelA", 1), ("d1b", 2)):
        r = null_test(cfg, BASE_SEED + seed_off)
        nulls[cfg] = r
        print(f"  {cfg:>14s}: ACF(1)={r['acf1']:+.4f}  within_CI={r['within_ci']}")

    # determinism recheck: identical seed -> identical value
    recheck = null_test("d1b", BASE_SEED + 2)
    determinism_ok = bool(recheck["acf1"] == nulls["d1b"]["acf1"])
    print(f"  determinism recheck (d1b, same seed): identical={determinism_ok}")

    print(f"\n[Part b: robustness, random initial state, "
          f"N={N_STEPS}, M={M_RUNS}]")
    robust = {}
    for m in "AB":
        r = robustness(m, BASE_SEED + 10 * (ord(m) - ord('A') + 1))
        robust[m] = r
        print(f"  Model {m}: ACF(1)={r['acf1_mean']:+.4f}  "
              f"sigma_above_shuffle={r['sigma_above_shuffle']:.2f}")

    # setup range: vacuum-start values from the deposited d6 results plus
    # the two random-start values above
    d6 = json.loads((HERE / "d6_acf_prediction_results.json")
                    .read_text(encoding="utf-8"))["results"]
    setups = dict(
        vacuum_A=d6["A"]["acf1_mean"], vacuum_B=d6["B"]["acf1_mean"],
        random_init_A=robust["A"]["acf1_mean"],
        random_init_B=robust["B"]["acf1_mean"],
    )
    vals = list(setups.values())
    rng_lo = math.floor(min(vals) * 100) / 100
    rng_hi = math.ceil(max(vals) * 100) / 100
    print(f"\n[Setup range across the four reproduced setups]")
    for k, v in setups.items():
        print(f"  {k:>14s}: {v:+.4f}")
    print(f"  printed range -> [{rng_lo:.2f}, {rng_hi:.2f}]")

    checks = dict(
        null_5d3gen_within_ci=nulls["5d3gen_modelA"]["within_ci"],
        null_d1b_within_ci=nulls["d1b"]["within_ci"],
        determinism_recheck_identical=determinism_ok,
        robustness_A_positive=bool(robust["A"]["acf1_mean"] > 0),
        robustness_B_positive=bool(robust["B"]["acf1_mean"] > 0),
    )
    checks["ALL_PASS"] = bool(all(checks.values()))

    out = dict(
        meta=dict(script="d8_acf_null_robustness.py", base_seed=BASE_SEED,
                  N_null=N_NULL, N_steps=N_STEPS, M_runs=M_RUNS,
                  words=dict(config_5d3gen=WORD_5D, config_d1b=WORD_D1B),
                  note="Seeded re-anchor of the Sec. VI Pillar-2 null test and "
                       "the initialization-robustness ACF values; the setup "
                       "range is taken over the four reproduced setups "
                       "(vacuum and random initialization, Models A and B)."),
        null_test=nulls,
        robustness=robust,
        setup_range=dict(setups=setups, printed=[rng_lo, rng_hi]),
        checks=checks,
    )
    (HERE / "d8_acf_null_robustness_results.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  ALL_PASS = {checks['ALL_PASS']}")
    print(f"  WROTE {HERE / 'd8_acf_null_robustness_results.json'}")


if __name__ == "__main__":
    main()
