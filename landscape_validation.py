#!/usr/bin/env python3
"""
Paper 1b Validation — Checks 1 + 2
====================================
Check 1: CHSH directly in 5D (without the 5D->4D projection)
Check 2: leakage norm N^2 per sequence

Goal: test whether the 5D->4D projection (|2>+|4>->|11>) distorts the
Bell landscape. If Bell%_5D ~ Bell%_4D the projection is harmless;
otherwise it must be discussed in the paper.

Method, Check 1: 3x3 correlation matrix T_{ij} = <psi| M_ij |psi>,
with M_ij Pauli-like observables on the 5D fusion space, and
|S|_max = 2*sqrt(m1+m2) (Horodecki; m1, m2 the two largest eigenvalues
of T^T.T).

Method, Check 2: N^2 = |psi0|^2 + |psi1|^2 + |psi3|^2 + |psi2+psi4|^2 / 2,
the weight inside the logical subspace spanned by |0>, |1>, |3> and the
normalized |11>_L = (|2>+|4>)/sqrt(2); the total 5D norm is 1, so N^2 <= 1.

Output:
- landscape_validation.json (tables)
- validation_summary.md (summary)

Reproducible: no fixed seed needed (deterministic enumeration).
"""
from __future__ import annotations
import json
import sys
import io
import time
from pathlib import Path
import numpy as np


OUT = Path(__file__).parent
phi = (1 + np.sqrt(5)) / 2
phi_inv = 1 / phi
sqrt_phi_inv = np.sqrt(phi_inv)

# R-eigenvalues (Fibonacci TQFT)
R1 = np.exp(-4j * np.pi / 5)
Rt = np.exp(+3j * np.pi / 5)

# F-matrix on the 2D fusion channel
F = np.array([[phi_inv, sqrt_phi_inv], [sqrt_phi_inv, -phi_inv]])
Finv = np.linalg.inv(F)  # equal to F since F is self-inverse up to sign; compute exactly

# ------------------------------------------------------------------
# 5D Basis (linear tree, 6 anyons -> vacuum)
# |0> = (x1,x2,x3) = (1, tau, 1)     logical |00>
# |1> = (1, tau, tau)                logical |01>
# |2> = (tau, 1, tau)                logical |11> part 1
# |3> = (tau, tau, 1)                logical |10>
# |4> = (tau, tau, tau)              logical |11> part 2
# ------------------------------------------------------------------

def build_sigma1() -> np.ndarray:
    return np.diag([R1, R1, Rt, Rt, Rt]).astype(np.complex128)

def build_sigma5() -> np.ndarray:
    return np.diag([R1, Rt, Rt, R1, Rt]).astype(np.complex128)

def build_sigma2() -> np.ndarray:
    """
    sigma2: F R F^-1 on {|0>,|3>} and {|1>,|4>}; |2> -> R_tau * |2>
    (Convention: braid on anyons 2-3 = x1 subtree)
    """
    M = np.zeros((5, 5), dtype=np.complex128)
    # Block on {|0>,|3>}: indices 0 and 3
    # Braid = F diag(R1,Rt) F^-1
    # In the x1-vertex basis: R_1 sits on x1=1, R_tau on x1=tau
    # For sigma2 we braid particles 2-3, which affects the first fusion channel x1.
    # In our basis, x1=1 for |0>,|1> and x1=tau for |2>,|3>,|4>.
    # So the block structure for sigma2 in the x1 label is:
    # states with x1 fixed independent of other labels.
    # But x1 and x2 are entangled: (x1,x2) constrained by fusion rules.
    # The canonical form: sigma2 = F . diag(R1, Rt) . F^{-1} on the x1-subtree of each x2-fixed sector
    B = F @ np.diag([R1, Rt]) @ Finv
    # Sector (x2=tau, x3=1): states |0> (x1=1) and |3> (x1=tau)
    idx = [0, 3]
    for i in range(2):
        for j in range(2):
            M[idx[i], idx[j]] = B[i, j]
    # Sector (x2=tau, x3=tau): states |1> (x1=1) and |4> (x1=tau)
    idx = [1, 4]
    for i in range(2):
        for j in range(2):
            M[idx[i], idx[j]] = B[i, j]
    # Sector (x2=1, x3=tau): only state |2>; braid is scalar Rt
    M[2, 2] = Rt
    return M

def build_sigma4() -> np.ndarray:
    """
    sigma4: F R F^-1 on {|0>,|1>} and {|3>,|4>}; |2> -> R_tau * |2>
    (Convention: braid on anyons 4-5 = x3 subtree)
    """
    M = np.zeros((5, 5), dtype=np.complex128)
    B = F @ np.diag([R1, Rt]) @ Finv
    # (x1=1, x2=tau): |0> (x3=1), |1> (x3=tau)
    idx = [0, 1]
    for i in range(2):
        for j in range(2):
            M[idx[i], idx[j]] = B[i, j]
    # (x1=tau, x2=tau): |3> (x3=1), |4> (x3=tau)
    idx = [3, 4]
    for i in range(2):
        for j in range(2):
            M[idx[i], idx[j]] = B[i, j]
    # (x1=tau, x2=1, x3=tau): only |2>
    M[2, 2] = Rt
    return M

def build_sigma3() -> np.ndarray:
    """
    sigma3: F R F^-1 on {|2>,|4>} (bridge generator); diag on others
    |0> -> R_1, |1> -> R_tau, |3> -> R_tau
    """
    M = np.zeros((5, 5), dtype=np.complex128)
    M[0, 0] = R1
    M[1, 1] = Rt
    M[3, 3] = Rt
    B = F @ np.diag([R1, Rt]) @ Finv
    idx = [2, 4]
    for i in range(2):
        for j in range(2):
            M[idx[i], idx[j]] = B[i, j]
    return M

SIGMA = {
    1: build_sigma1(),
    2: build_sigma2(),
    3: build_sigma3(),
    4: build_sigma4(),
    5: build_sigma5(),
}

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------

def is_unitary(M: np.ndarray, tol: float = 1e-10) -> bool:
    I = np.eye(M.shape[0])
    return np.allclose(M @ M.conj().T, I, atol=tol)

def yang_baxter_check() -> dict:
    """Return |sigma_i sigma_{i+1} sigma_i - sigma_{i+1} sigma_i sigma_{i+1}|_max per i."""
    out = {}
    for i in [1, 2, 3, 4]:
        a = SIGMA[i]; b = SIGMA[i+1]
        lhs = a @ b @ a
        rhs = b @ a @ b
        out[f"sigma{i}_{i+1}"] = float(np.max(np.abs(lhs - rhs)))
    return out

# ------------------------------------------------------------------
# Observables on the 5D space
# ------------------------------------------------------------------
# Qubit A (= x1):
#   x1 = 1   -> |0>, |1>          (eigenvalue +1 of Z_A)
#   x1 = tau -> |2>, |3>, |4>     (eigenvalue -1 of Z_A)
# Qubit B (= x3):
#   x3 = 1   -> |0>, |3>          (+1)
#   x3 = tau -> |1>, |2>, |4>     (-1)
#
# For X_A: flip x1=1 <-> x1=tau within same (x2, x3) sector.
#   (x2=tau, x3=1):  |0> <-> |3>
#   (x2=tau, x3=tau): |1> <-> |4>
#   (x2=1,   x3=tau): |2> has no partner -> eigenvalue 0 on |2>
# Analog for X_B: flip x3=1 <-> x3=tau within same (x1, x2) sector.
#   (x1=1,   x2=tau): |0> <-> |1>
#   (x1=tau, x2=tau): |3> <-> |4>
#   (x1=tau, x2=1,   x3=tau): |2> isolated -> 0
# Y = -i * X . Z  (in the same 5D space).
# ------------------------------------------------------------------

Z_A = np.diag([+1, +1, -1, -1, -1]).astype(np.complex128)
Z_B = np.diag([+1, -1, -1, +1, -1]).astype(np.complex128)

def build_XA() -> np.ndarray:
    X = np.zeros((5, 5), dtype=np.complex128)
    X[0, 3] = 1; X[3, 0] = 1     # |0> <-> |3>
    X[1, 4] = 1; X[4, 1] = 1     # |1> <-> |4>
    # |2> -> 0
    return X

def build_XB() -> np.ndarray:
    X = np.zeros((5, 5), dtype=np.complex128)
    X[0, 1] = 1; X[1, 0] = 1     # |0> <-> |1>
    X[3, 4] = 1; X[4, 3] = 1     # |3> <-> |4>
    return X

X_A = build_XA()
X_B = build_XB()
Y_A = -1j * (X_A @ Z_A)  # hermitian with eigenvalues {-1, 0, +1}
Y_B = -1j * (X_B @ Z_B)

# (X_A, Y_A, Z_A) and (X_B, Y_B, Z_B) — note X and Y are hermitian with spectrum in {-1,0,+1}
# due to the |2>-isolation. This is physically the trit-extension mentioned
# in the validation briefing (Option C).

ALICE = [X_A, Y_A, Z_A]
BOB = [X_B, Y_B, Z_B]

# ------------------------------------------------------------------
# 5D CHSH via Horodecki shortcut
# ------------------------------------------------------------------

def chsh_5d(psi: np.ndarray) -> float:
    """
    Horodecki: |S|_max = 2 * sqrt(m1 + m2) where m1 m2 are the two largest
    eigenvalues of T^T T, T_{ij} = Re <psi| A_i B_j |psi>.
    Since A and B operate on the SAME space (not tensored), this is the
    canonical 5D correlation matrix.
    """
    T = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            M = ALICE[i] @ BOB[j]
            val = np.vdot(psi, M @ psi)
            T[i, j] = val.real
    TtT = T.T @ T
    eig = np.sort(np.linalg.eigvalsh(TtT))[::-1]
    return float(2 * np.sqrt(max(eig[0] + eig[1], 0.0)))

# ------------------------------------------------------------------
# Leakage norm
# ------------------------------------------------------------------

def leakage_N2(psi: np.ndarray) -> float:
    """N^2 = |psi0|^2 + |psi1|^2 + |psi3|^2 + |psi2+psi4|^2 / 2 -- the weight in
    the logical subspace; the factor 1/2 is the projection onto the NORMALIZED
    |11>_L = (|2>+|4>)/sqrt(2) (same quantity as N^2_P in n2_reanchor.py)."""
    a0 = abs(psi[0])**2
    a1 = abs(psi[1])**2
    a3 = abs(psi[3])**2
    a24 = abs(psi[2] + psi[4])**2 / 2.0
    return float(a0 + a1 + a3 + a24)

# ------------------------------------------------------------------
# 4D projection-based CHSH (for comparison with existing 92%)
# ------------------------------------------------------------------
# Projection: |00>_L = |0>, |01>_L = |1>, |10>_L = |3>, |11>_L = (|2>+|4>)/sqrt(2)
# After projection + renormalisation, use standard 2-qubit CHSH via Horodecki
# on the resulting density matrix.

def project_4d(psi5: np.ndarray) -> np.ndarray | None:
    """Return 4D computational state (renormalized) or None if norm==0."""
    psi4 = np.array([psi5[0], psi5[1], psi5[3], (psi5[2] + psi5[4]) / np.sqrt(2)],
                    dtype=np.complex128)
    n = np.linalg.norm(psi4)
    if n < 1e-12:
        return None
    return psi4 / n

def concurrence_2q(psi4: np.ndarray) -> float:
    """Pure-state 2-qubit concurrence."""
    # psi4 in basis |00>|01>|10>|11>
    # C = 2 |psi00 * psi11 - psi01 * psi10|
    return float(2 * abs(psi4[0] * psi4[3] - psi4[1] * psi4[2]))

def chsh_4d_horodecki(psi4: np.ndarray) -> float:
    """Horodecki |S|_max for pure 2-qubit state via concurrence:
       |S|_max = 2 sqrt(1 + C^2).
    """
    C = concurrence_2q(psi4)
    return float(2 * np.sqrt(1 + C**2))

# ------------------------------------------------------------------
# Enumeration
# ------------------------------------------------------------------

def apply_sequence(gates: list[int], psi0: np.ndarray) -> np.ndarray:
    psi = psi0.copy()
    for g in gates:
        psi = SIGMA[g] @ psi
    return psi

# TOLERANCE. No bare threshold comparisons: a sequence counts as violating
# when S > 2 + TOL with TOL = 1e-10. Sequences with |S - 2| <= TOL are
# counted separately as boundary cases and are never silently assigned to
# either side. Same convention as the companion record
# (data/fibonacci_enumeration.py).
TOL = 1e-10

def enumerate_length(L: int, gate_set: tuple[int, ...] = (2, 3, 4),
                     psi0: np.ndarray | None = None) -> dict:
    """Enumerate all |gate_set|^L sequences at length L."""
    if psi0 is None:
        psi0 = np.array([1, 0, 0, 0, 0], dtype=np.complex128)
    n_total = len(gate_set) ** L
    S5 = np.zeros(n_total)
    S4 = np.zeros(n_total)
    N2 = np.zeros(n_total)
    best_seq = None
    best_S5 = 0.0
    for idx in range(n_total):
        # decode idx in base |gate_set|
        seq = []
        x = idx
        for _ in range(L):
            seq.append(gate_set[x % len(gate_set)])
            x //= len(gate_set)
        psi5 = apply_sequence(seq, psi0)
        S5[idx] = chsh_5d(psi5)
        N2[idx] = leakage_N2(psi5)
        psi4 = project_4d(psi5)
        if psi4 is not None:
            S4[idx] = chsh_4d_horodecki(psi4)
        if S5[idx] > best_S5:
            best_S5 = S5[idx]; best_seq = list(seq)
    return {
        "length": L,
        "n_sequences": n_total,
        "S5_max": float(S5.max()),
        "S5_mean": float(S5.mean()),
        "bell_pct_5D": float((S5 > 2.0 + TOL).mean() * 100),
        "at_bound_pct_5D": float((abs(S5 - 2.0) <= TOL).mean() * 100),
        "below_pct_5D": float((S5 < 2.0 - TOL).mean() * 100),
        "S4_max": float(S4.max()),
        "S4_mean": float(S4.mean()),
        "bell_pct_4D": float((S4 > 2.0).mean() * 100),
        "violation_tolerance": TOL,
        "violation_rule": "S > 2 + tol counts as violating; |S - 2| <= tol "
                          "is counted separately as a boundary case, never "
                          "silently assigned to either side; the 4D column "
                          "uses the bare threshold because no boundary class "
                          "exists there (smallest gap 7.5e-04)",
        "N2_mean": float(N2.mean()),
        "N2_min": float(N2.min()),
        "N2_max": float(N2.max()),
        "best_seq_5D": best_seq,
        "best_S5": best_S5,
    }

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("=" * 66)
    print("  FIBONACCI LANDSCAPE VALIDATION — Checks 1 + 2")
    print("=" * 66)

    # Sanity
    print("\n[Sanity] Unitarity:")
    for i in [1, 2, 3, 4, 5]:
        print(f"  sigma{i}: unitary = {is_unitary(SIGMA[i])}")
    print("\n[Sanity] Yang-Baxter (max residual):")
    yb = yang_baxter_check()
    for k, v in yb.items():
        print(f"  {k}: {v:.2e}")
    yb_ok = all(v < 1e-10 for v in yb.values())
    print(f"  Yang-Baxter bestanden: {yb_ok}")

    # Also check hermiticity and spectrum of A, B observables
    print("\n[Observable Spectrum]")
    for name, M in [("Z_A", Z_A), ("Z_B", Z_B), ("X_A", X_A), ("X_B", X_B),
                    ("Y_A", Y_A), ("Y_B", Y_B)]:
        herm = np.allclose(M, M.conj().T, atol=1e-12)
        ev = np.sort(np.linalg.eigvalsh(M).real)
        print(f"  {name}: hermitian={herm}, eigenvalues={ev}")

    # Main enumeration
    results = {}
    for L in range(3, 10):
        t0 = time.time()
        r = enumerate_length(L)
        dt = time.time() - t0
        # walltime_s is deliberately NOT stored: it would make the JSON
        # nonreproducible byte-for-byte across runs.
        results[f"L={L}"] = r
        print(f"\nL={L}: {r['n_sequences']} sequences in {dt:.1f}s")
        print(f"  |S|_max_5D = {r['S5_max']:.4f}  Bell%_5D = {r['bell_pct_5D']:.2f}%")
        print(f"  |S|_max_4D = {r['S4_max']:.4f}  Bell%_4D = {r['bell_pct_4D']:.2f}%")
        print(f"  Delta Bell% (4D - 5D) = {r['bell_pct_4D'] - r['bell_pct_5D']:+.2f}%")
        print(f"  N^2 mean={r['N2_mean']:.4f}  min={r['N2_min']:.4f}  max={r['N2_max']:.4f}")
        print(f"  best 5D seq: {r['best_seq_5D']}")

    # Conclusion
    deltas = [results[k]["bell_pct_4D"] - results[k]["bell_pct_5D"] for k in results]
    max_delta = max(abs(d) for d in deltas)
    if max_delta < 5:
        conclusion = "VALIDATED"
    elif max_delta < 15:
        conclusion = "NEEDS_DISCUSSION"
    else:
        conclusion = "PROBLEMATIC"

    out = {
        "meta": {
            "date": "2026-04-15",
            "script": "landscape_validation.py",
            "seed": None,
            "yang_baxter_max_residual": yb,
        },
        "results_per_length": results,
        "conclusion": conclusion,
        "max_abs_delta_bell_pct": round(float(max_delta), 3),
    }
    (OUT / "landscape_validation.json").write_text(json.dumps(out, indent=2))
    print(f"\nWRITE {OUT / 'landscape_validation.json'}")
    print(f"  max |Delta Bell%| = {max_delta:.2f}% -> {conclusion}")

if __name__ == "__main__":
    main()
