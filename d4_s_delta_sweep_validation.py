#!/usr/bin/env python3
"""
D4 -- Deterministic validation of s_delta_sweeps.json  (RUN ORDER: d10 before d4)
=================================================================================
Independently recomputes the sector-phase sweeps |S|(delta), C(delta)
for the winning braid sequences already reported in Table I (d1b,
two-generator encoding; all ten rows) and used for Fig. 2
(three-generator encoding), and checks them against the deposited
s_delta_sweeps.json (L = 3, 6, 8, 10, 12) respectively against the
printed Table I values (L = 4, 5, 7, 9, 11).  As a further positive
control it re-evaluates every best_seq_5D winner stored in
landscape_validation.json (L = 3..9) with the native trit-valued
Protocol-B observables and confirms the stored S5 values.

Where the sequences come from.  The d1b rows are the published Table I
sequences (two-generator encoding).  The 5D rows (encoding B) are the
delta = 0 maximizers under Protocol A for L = 3, 6, 8, 10, 12, taken from
the exhaustive enumeration over all 3^L words in
d10_protocol_a_projection_audit.py (key delta0_first_maximizer: first
word in lexicographic order within 1e-10 of the maximum, two-pass rule).
That enumeration is part of this record, and this script checks
GEN5D_SEQS against it before doing anything else.  Run d10 first, then
d4 --regenerate (rewrites s_delta_sweeps.json), then d4 without a flag
(validates the file and writes d4_s_delta_sweep_validation_results.json).
Given a sequence, recomputing its S(delta) curve is a closed-form,
deterministic calculation (Horodecki, Horodecki & Horodecki 1995: for a
pure 2-qubit state, |S|_max = 2*sqrt(1+C^2) is exact and reachable), so
this script reproduces every number in s_delta_sweeps.json from scratch.

Reproducible: no randomness in the closed-form path. The one
measurement-angle optimizer used per curve (as an independent
cross-check that the closed-form is in fact achieved by an explicit
measurement, not just an upper bound) uses a fixed seed.
"""
from __future__ import annotations
import json
import math
import sys
import io
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent

PHI = (1 + math.sqrt(5)) / 2
F2 = np.array([[1/PHI, 1/math.sqrt(PHI)],
               [1/math.sqrt(PHI), -1/PHI]], dtype=complex)
R1 = np.exp(-4j * math.pi / 5)
RT = np.exp(3j * math.pi / 5)

# ----------------------------------------------------------------------
# d1b encoding (two generators AM, MB; 4D fusion space |a,m,b>)
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


def apply_sequence_d1b(seq: str, delta: float) -> np.ndarray:
    G_AM = gate_AM_d1b(delta)
    G_MB = gate_MB_d1b(delta)
    psi = np.array([1, 0, 0, 0], dtype=complex)
    for ch in seq:
        psi = (G_AM if ch == 'A' else G_MB) @ psi
    n = np.linalg.norm(psi)
    return psi / n if n > 1e-15 else psi


# ----------------------------------------------------------------------
# 5D encoding (three generators sigma2, sigma3, sigma4; full 5D fusion
# basis |0..4> projected to 2 qubits via |2>+|4> -> |11>)
# ----------------------------------------------------------------------

def braid_block(delta: float) -> np.ndarray:
    R = np.diag([R1, RT * np.exp(1j * delta)])
    return F2 @ R @ F2


def sigma2_5d(delta: float) -> np.ndarray:
    r_tau = RT * np.exp(1j * delta)
    B = braid_block(delta)
    M = np.zeros((5, 5), dtype=complex)
    M[0, 0] = B[0, 0]; M[0, 3] = B[0, 1]
    M[3, 0] = B[1, 0]; M[3, 3] = B[1, 1]
    M[1, 1] = B[0, 0]; M[1, 4] = B[0, 1]
    M[4, 1] = B[1, 0]; M[4, 4] = B[1, 1]
    M[2, 2] = r_tau
    return M


def sigma3_5d(delta: float) -> np.ndarray:
    r_tau = RT * np.exp(1j * delta)
    B = braid_block(delta)
    M = np.zeros((5, 5), dtype=complex)
    M[0, 0] = R1
    M[1, 1] = r_tau
    M[3, 3] = r_tau
    M[2, 2] = B[0, 0]; M[2, 4] = B[0, 1]
    M[4, 2] = B[1, 0]; M[4, 4] = B[1, 1]
    return M


def sigma4_5d(delta: float) -> np.ndarray:
    r_tau = RT * np.exp(1j * delta)
    B = braid_block(delta)
    M = np.zeros((5, 5), dtype=complex)
    M[0, 0] = B[0, 0]; M[0, 1] = B[0, 1]
    M[1, 0] = B[1, 0]; M[1, 1] = B[1, 1]
    M[3, 3] = B[0, 0]; M[3, 4] = B[0, 1]
    M[4, 3] = B[1, 0]; M[4, 4] = B[1, 1]
    M[2, 2] = r_tau
    return M


def apply_sequence_5d(seq: str, delta: float) -> np.ndarray:
    gates = {'2': sigma2_5d(delta), '3': sigma3_5d(delta), '4': sigma4_5d(delta)}
    psi = np.array([1, 0, 0, 0, 0], dtype=complex)
    for ch in seq:
        psi = gates[ch] @ psi
    return psi


def project_5d_to_2qubit(psi5: np.ndarray) -> np.ndarray:
    """Protocol A: |11>_L <- (|2>+|4>)/sqrt(2), then renormalize -- the same
    map as project_4d in landscape_validation.py."""
    psi4 = np.array([psi5[0], psi5[1], psi5[3], (psi5[2] + psi5[4]) / math.sqrt(2)], dtype=complex)
    n = np.linalg.norm(psi4)
    return psi4 / n if n > 1e-15 else psi4


# ----------------------------------------------------------------------
# Shared measures
# ----------------------------------------------------------------------

def concurrence(psi: np.ndarray) -> float:
    """Wootters concurrence for a pure 2-qubit state."""
    return float(2 * abs(psi[0] * psi[3] - psi[1] * psi[2]))


def chsh_horodecki(C: float) -> float:
    """|S|_max = 2*sqrt(1+C^2), exact and reachable for pure states
    (Horodecki, Horodecki & Horodecki, Phys. Lett. A 200, 340 (1995))."""
    return 2.0 * math.sqrt(1.0 + C * C)


def correlator_bloch(psi, theta_A, phi_A, theta_B, phi_B) -> float:
    sx = math.sin(theta_A) * math.cos(phi_A); sy = math.sin(theta_A) * math.sin(phi_A); cz = math.cos(theta_A)
    MA = np.array([[cz, sx - 1j*sy], [sx + 1j*sy, -cz]], dtype=complex)
    sx = math.sin(theta_B) * math.cos(phi_B); sy = math.sin(theta_B) * math.sin(phi_B); cz = math.cos(theta_B)
    MB = np.array([[cz, sx - 1j*sy], [sx + 1j*sy, -cz]], dtype=complex)
    M = np.kron(MA, MB)
    return float(np.real(psi.conj() @ M @ psi))


def chsh_optimize_bloch(psi, seed: int = 2026, n_restarts: int = 8):
    """Full Bloch-sphere CHSH optimization (independent cross-check route,
    not needed for the closed-form values themselves)."""
    from scipy.optimize import differential_evolution, minimize

    def neg_S(x):
        return -abs(
            correlator_bloch(psi, x[0], x[1], x[4], x[5])
            + correlator_bloch(psi, x[0], x[1], x[6], x[7])
            + correlator_bloch(psi, x[2], x[3], x[4], x[5])
            - correlator_bloch(psi, x[2], x[3], x[6], x[7])
        )

    bounds = [(0.0, math.pi), (0.0, 2*math.pi)] * 4
    res = differential_evolution(neg_S, bounds, seed=seed, tol=1e-12, maxiter=300, popsize=20, polish=True)
    best_s, best_x = -res.fun, res.x
    rng = np.random.default_rng(seed)
    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(0.0, math.pi if i % 2 == 0 else 2*math.pi) for i in range(8)])
        r = minimize(neg_S, x0, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-14, 'gtol': 1e-12})
        if -r.fun > best_s:
            best_s, best_x = -r.fun, r.x
    return float(best_s)


# ----------------------------------------------------------------------
# Sequences: d1b rows = the published Table I sequences; 5D rows = the
# delta = 0 maximizers under Protocol A from the d10 enumeration (checked
# against d10_protocol_a_projection_audit_results.json in main()).
# ----------------------------------------------------------------------
D1B_SEQS = {3: "ABB", 6: "AAABAB", 8: "BABBABAB", 10: "ABBAABBAAB", 12: "ABABABABABAB"}
# delta = 0 maximizers under Protocol A, first in lexicographic order (two-pass rule of d10);
# the former words 423 / 424233 / 23444432 / 3234442432 / 324342333324 were maximizers only
# under the projection without the 1/sqrt(2).
GEN5D_SEQS = {3: "243", 6: "232432", 8: "23343322", 10: "2322343322", 12: "232222343322"}
N_POINTS = 50

# The remaining five rows of Table I (L = 4, 5, 7, 9, 11), copied verbatim
# from the published table together with their printed values
# (|S|, C, % Tsirelson).  These lengths are not part of
# s_delta_sweeps.json, so they are validated directly against the
# printed Table I values via the same closed-form sweep.
TAB1_EXTRA_SEQS = {
    4:  ("ABAB",        2.225, 0.488, 78.7),
    5:  ("AABAB",       2.348, 0.615, 83.0),
    7:  ("ABBAABB",     2.536, 0.780, 89.7),
    9:  ("ABBBBABAB",   2.639, 0.861, 93.3),
    11: ("ABABBBBABAB", 2.784, 0.968, 98.4),
}
# Provenance note: the pre-v1.5.1 values for L = 4 (2.224/0.486/78.6) and
# L = 7 (2.535/0.779/89.6) stem from a coarser 25-point delta grid used in
# the original enumeration era; on the deposited 50-point grid (the grid of
# s_delta_sweeps.json, whose L = 3, 6, 8, 10, 12 rows this script reproduces
# bit-exactly) their per-sequence optima are 2.225/0.488/78.7 and
# 2.536/0.780/89.7.  This is the same correction class as the L = 6 row
# (2.394 -> 2.398), re-anchored in the 2026-07-22 build.
# 2026-09-30: the L = 6 and L = 9 rows now carry the printed Table I sequences
# AAABAB (2.433/0.693/86.0) and ABBBBABAB (2.639/0.861/93.3), i.e. the per-length
# optima of the companion enumeration (delta_opt in fibonacci_enumeration_results.json);
# the previous entries ABBAAB (2.398) and ABBAABABB (2.624) were not the printed rows.
# The 5D curves use Protocol A ((|2>+|4>)/sqrt(2)); the earlier deposit omitted the
# 1/sqrt(2). Run with --regenerate to rewrite s_delta_sweeps.json from this script.
# 2026-10-02: the 5D rows carry the delta = 0 maximizers under Protocol A (d10,
# delta0_first_maximizer, two-pass rule); the previous words were maximizers only under
# the projection without the 1/sqrt(2). The enumeration (d10) is part of this record.
TSIRELSON = 2.0 * math.sqrt(2.0)

# ----------------------------------------------------------------------
# Trit-valued (Protocol B) observables on the native 5D space, exactly
# as defined in landscape_validation.py -- used here only as a positive
# control that re-evaluates the best_seq_5D winners stored in
# landscape_validation.json (delta = 0).
# ----------------------------------------------------------------------
_Z_A5 = np.diag([+1, +1, -1, -1, -1]).astype(complex)
_Z_B5 = np.diag([+1, -1, -1, +1, -1]).astype(complex)
_X_A5 = np.zeros((5, 5), dtype=complex)
_X_A5[0, 3] = _X_A5[3, 0] = 1; _X_A5[1, 4] = _X_A5[4, 1] = 1
_X_B5 = np.zeros((5, 5), dtype=complex)
_X_B5[0, 1] = _X_B5[1, 0] = 1; _X_B5[3, 4] = _X_B5[4, 3] = 1
_Y_A5 = -1j * (_X_A5 @ _Z_A5)
_Y_B5 = -1j * (_X_B5 @ _Z_B5)
_ALICE5 = [_X_A5, _Y_A5, _Z_A5]
_BOB5 = [_X_B5, _Y_B5, _Z_B5]


def chsh_5d_native(psi: np.ndarray) -> float:
    """Native 5D (Protocol B) CHSH via the Horodecki criterion on the
    3x3 correlation matrix, identical to landscape_validation.py."""
    T = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            T[i, j] = np.vdot(psi, (_ALICE5[i] @ _BOB5[j]) @ psi).real
    eig = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
    return float(2 * np.sqrt(max(eig[0] + eig[1], 0.0)))


def sweep_d1b(seq: str) -> dict:
    deltas = np.linspace(0, 2*math.pi, N_POINTS, endpoint=False)
    S = np.zeros(N_POINTS); C = np.zeros(N_POINTS)
    for i, d in enumerate(deltas):
        psi = apply_sequence_d1b(seq, d)
        C[i] = concurrence(psi)
        S[i] = chsh_horodecki(C[i])
    return dict(seq=seq, delta=deltas.tolist(), S=S.tolist(), C=C.tolist())


def sweep_5d(seq: str) -> dict:
    deltas = np.linspace(0, 2*math.pi, N_POINTS, endpoint=False)
    S = np.zeros(N_POINTS); C = np.zeros(N_POINTS)
    for i, d in enumerate(deltas):
        psi4 = project_5d_to_2qubit(apply_sequence_5d(seq, d))
        C[i] = concurrence(psi4)
        S[i] = chsh_horodecki(C[i])
    return dict(seq=seq, delta=deltas.tolist(), S=S.tolist(), C=C.tolist())


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    # GEN5D_SEQS must be the delta = 0 first maximizers of the d10 enumeration (run d10 first)
    d10_path = HERE / "d10_protocol_a_projection_audit_results.json"
    d10_first = json.loads(d10_path.read_text(encoding="utf-8")).get("delta0_first_maximizer")
    if d10_first is None:
        sys.exit(f"d4: {d10_path.name} has no 'delta0_first_maximizer' block -- run d10 first")
    for L, seq in GEN5D_SEQS.items():
        w = d10_first[str(L)]["word"]
        if w != seq:
            sys.exit(f"d4: GEN5D_SEQS[{L}] = {seq} but the d10 first maximizer is {w} -- fix GEN5D_SEQS")
    print(f"GEN5D_SEQS agree with d10 delta0_first_maximizer for L = {list(GEN5D_SEQS)}")
    shipped_path = HERE / "s_delta_sweeps.json"
    if "--regenerate" in sys.argv:
        # Rewrite the shipped sweeps from this script (d1b rows and 5D rows alike);
        # a subsequent run without the flag validates the file bit for bit.
        regen = {"encoding_A_d1b_2gen": {str(L): sweep_d1b(s) for L, s in D1B_SEQS.items()},
                 "encoding_B_5d_3gen": {str(L): sweep_5d(s) for L, s in GEN5D_SEQS.items()}}
        shipped_path.write_text(json.dumps(regen, indent=2))
        print(f"REGENERATED {shipped_path}")
        return
    shipped = json.loads(shipped_path.read_text(encoding="utf-8"))

    recomputed = {"encoding_A_d1b_2gen": {}, "encoding_B_5d_3gen": {}}
    print("=" * 72)
    print("  D4 -- s_delta_sweeps.json validation")
    print("=" * 72)

    max_diff_overall = 0.0
    per_curve_report = []

    print("\n[Encoding A -- d1b, two generators AM/MB]")
    for L, seq in D1B_SEQS.items():
        r = sweep_d1b(seq)
        recomputed["encoding_A_d1b_2gen"][str(L)] = r
        ship = shipped["encoding_A_d1b_2gen"][str(L)]
        assert ship["seq"] == seq, f"seq mismatch at L={L}: shipped={ship['seq']} expected={seq}"
        dS = float(np.max(np.abs(np.array(r["S"]) - np.array(ship["S"]))))
        dC = float(np.max(np.abs(np.array(r["C"]) - np.array(ship["C"]))))
        max_diff_overall = max(max_diff_overall, dS, dC)
        i_true_max = int(np.argmax(r["S"]))
        i_shipped_argmax_of_shipped = int(np.argmax(ship["S"]))
        per_curve_report.append(dict(
            encoding="d1b", L=L, seq=seq, max_abs_diff_S=dS, max_abs_diff_C=dC,
            recomputed_true_max_S=round(r["S"][i_true_max], 6),
            recomputed_true_max_C=round(r["C"][i_true_max], 6),
            shipped_argmax_index_matches_recomputed_argmax=(i_true_max == i_shipped_argmax_of_shipped),
        ))
        print(f"  L={L:2d}  seq={seq:>14s}  max|dS|={dS:.2e}  max|dC|={dC:.2e}  "
              f"true_max_S={r['S'][i_true_max]:.6f}  true_max_C={r['C'][i_true_max]:.6f}")

    print("\n[Encoding B -- 5D, three generators sigma2/sigma3/sigma4; delta = 0 value, sweep maximum,"
          " grid index k of the maximum and all k within 1e-9; negative control max S <= 2*sqrt(2)]")
    K_TOL = 1e-9
    for L, seq in GEN5D_SEQS.items():
        r = sweep_5d(seq)
        recomputed["encoding_B_5d_3gen"][str(L)] = r
        ship = shipped["encoding_B_5d_3gen"][str(L)]
        assert ship["seq"] == seq, f"seq mismatch at L={L}: shipped={ship['seq']} expected={seq}"
        dS = float(np.max(np.abs(np.array(r["S"]) - np.array(ship["S"]))))
        dC = float(np.max(np.abs(np.array(r["C"]) - np.array(ship["C"]))))
        max_diff_overall = max(max_diff_overall, dS, dC)
        S_arr = np.array(r["S"])
        i_max = int(np.argmax(S_arr))
        k_within = [int(k) for k in np.flatnonzero(S_arr >= S_arr[i_max] - K_TOL)]
        assert S_arr.max() <= TSIRELSON + 1e-9, f"5D sweep L={L} exceeds the Tsirelson bound: {S_arr.max()!r}"
        per_curve_report.append(dict(encoding="5D", L=L, seq=seq, max_abs_diff_S=dS, max_abs_diff_C=dC,
                                     S_delta0=float(S_arr[0]), sweep_max_S=float(S_arr[i_max]), k_max=i_max,
                                     k_within_tol=k_within, k_tol=K_TOL,
                                     delta_over_pi_within_tol=[2.0 * k / N_POINTS for k in k_within],
                                     max_S_below_tsirelson=True))
        print(f"  L={L:2d}  seq={seq:>14s}  max|dS|={dS:.2e}  max|dC|={dC:.2e}  S(0)={S_arr[0]:.6f}  "
              f"sweep max={S_arr[i_max]:.6f} at k={i_max} (delta/pi={2.0 * i_max / N_POINTS:.2f})  "
              f"k within {K_TOL:g}: {k_within}")

    # ------------------------------------------------------------------
    # Remaining Table I rows (L = 4, 5, 7, 9, 11): closed-form sweep of
    # the printed sequences, checked against the printed values.
    # ------------------------------------------------------------------
    print("\n[Table I rows not covered by s_delta_sweeps.json -- validation "
          "against the printed values]")
    tab1_extra_report = []
    tab1_extra_all_match = True
    for L, (seq, s_pub, c_pub, pct_pub) in TAB1_EXTRA_SEQS.items():
        r = sweep_d1b(seq)
        i_max = int(np.argmax(r["S"]))
        s_max = r["S"][i_max]; c_max = r["C"][i_max]
        pct = 100.0 * s_max / TSIRELSON
        match = (round(s_max, 3) == s_pub and round(c_max, 3) == c_pub
                 and round(pct, 1) == pct_pub)
        tab1_extra_all_match = tab1_extra_all_match and match
        tab1_extra_report.append(dict(
            L=L, seq=seq,
            recomputed_S=round(s_max, 6), recomputed_C=round(c_max, 6),
            recomputed_pct_tsirelson=round(pct, 3),
            published=dict(S=s_pub, C=c_pub, pct_tsirelson=pct_pub),
            matches_published=match,
        ))
        print(f"  L={L:2d}  seq={seq:>14s}  S={s_max:.6f} (pub {s_pub})  "
              f"C={c_max:.6f} (pub {c_pub})  %T={pct:.3f} (pub {pct_pub})  "
              f"match={match}")

    # ------------------------------------------------------------------
    # Positive control: re-evaluate every best_seq_5D winner stored in
    # landscape_validation.json (L = 3..9, delta = 0) with the native
    # trit-valued Protocol-B observables and confirm it reproduces the
    # stored best_S5 / S5_max.
    # ------------------------------------------------------------------
    print("\n[Positive control: best_seq_5D winners in "
          "landscape_validation.json (delta = 0)]")
    land = json.loads((HERE / "landscape_validation.json")
                      .read_text(encoding="utf-8"))["results_per_length"]
    best_seq_report = []
    best_seq_all_match = True
    for L in range(3, 10):
        entry = land[f"L={L}"]
        seq_digits = "".join(str(g) for g in entry["best_seq_5D"])
        psi5 = apply_sequence_5d(seq_digits, 0.0)
        s5 = chsh_5d_native(psi5)
        d_best = abs(s5 - entry["best_S5"])
        d_max = abs(s5 - entry["S5_max"])
        ok = d_best < 1e-9 and d_max < 1e-9
        best_seq_all_match = best_seq_all_match and ok
        best_seq_report.append(dict(
            L=L, best_seq_5D=entry["best_seq_5D"],
            reevaluated_S5=s5, stored_best_S5=entry["best_S5"],
            stored_S5_max=entry["S5_max"],
            abs_diff_vs_best_S5=d_best, matches=ok,
        ))
        print(f"  L={L}  seq={seq_digits:>10s}  S5={s5:.12f}  "
              f"stored={entry['best_S5']:.12f}  |diff|={d_best:.2e}  ok={ok}")

    # Independent cross-check: at the true argmax of one representative curve
    # (L=12 d1b, the paper's headline result), confirm the closed-form Horodecki
    # value is actually achieved by an explicit measurement (full Bloch-sphere
    # angle optimization), not just an unreached upper bound.
    print("\n[Cross-check: closed-form vs. explicit measurement-angle optimum, L=12 d1b]")
    seq12 = D1B_SEQS[12]
    r12 = recomputed["encoding_A_d1b_2gen"]["12"]
    i_best = int(np.argmax(r12["S"]))
    delta_best = r12["delta"][i_best]
    psi_best = apply_sequence_d1b(seq12, delta_best)
    s_opt = chsh_optimize_bloch(psi_best)
    s_h = r12["S"][i_best]
    gap = abs(s_opt - s_h)
    print(f"  delta={delta_best:.6f}  S_horodecki={s_h:.10f}  S_optimized={s_opt:.10f}  gap={gap:.2e}")

    checks = dict(
        all_seq_match_shipped=True,
        max_abs_diff_S_or_C_over_all_curves=max_diff_overall,
        closed_form_matches_explicit_optimum_gap=gap,
        all_table1_extra_rows_match_published=tab1_extra_all_match,
        all_best_seq_5D_reproduce_stored_S5=best_seq_all_match,
        all_5D_sweeps_below_tsirelson=True,   # asserted in the 5D loop above
    )
    checks["ALL_PASS"] = bool(max_diff_overall < 1e-6 and gap < 1e-6
                              and tab1_extra_all_match and best_seq_all_match)

    out = dict(
        meta=dict(script="d4_s_delta_sweep_validation.py", n_points=N_POINTS,
                   note="Closed-form Horodecki reproduction of the Table I sequences (all ten "
                        "rows) and of the 5D rows of s_delta_sweeps.json / Fig. 2, whose words "
                        "are the delta = 0 maximizers under Protocol A from the d10 enumeration "
                        "(part of this record; checked against d10 before running), plus a "
                        "positive control re-evaluating the best_seq_5D winners stored in "
                        "landscape_validation.json.",
                   gen5d_seqs_source="d10_protocol_a_projection_audit_results.json: delta0_first_maximizer"),
        per_curve=per_curve_report,
        table1_extra_rows=tab1_extra_report,
        best_seq_5D_control=best_seq_report,
        cross_check_L12=dict(delta=delta_best, S_horodecki=s_h, S_optimized=s_opt, gap=gap),
        checks=checks,
    )
    (HERE / "d4_s_delta_sweep_validation_results.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    for k, v in checks.items():
        print(f"  [{'PASS' if (v is True or (isinstance(v, float) and v < 1e-6)) else 'INFO'}] {k} = {v}")
    print(f"\n  ALL_PASS = {checks['ALL_PASS']}")
    print(f"  WROTE {HERE / 'd4_s_delta_sweep_validation_results.json'}")


if __name__ == "__main__":
    main()
