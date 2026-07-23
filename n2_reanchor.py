#!/usr/bin/env python3
"""
Paper 1b — N^2 subspace-fidelity re-anchor (v1.5 correction)
==================================================================
The originally deposited landscape_validation.py computes the leakage
norm WITHOUT the 1/2 factor implied by the logical-basis projector
|11>_L = (|2>+|4>)/sqrt(2):

    N^2_old = |psi0|^2 + |psi1|^2 + |psi3|^2 + |psi2+psi4|^2   (can exceed 1)

The correct computational-subspace fidelity (projector onto the
orthonormal 4D logical basis {|0>,|1>,|3>,(|2>+|4>)/sqrt(2)}) is

    N^2_P  = |psi0|^2 + |psi1|^2 + |psi3|^2 + |psi2+psi4|^2 / 2   (in [0,1])

landscape_validation.json and landscape_validation.py are published
Zenodo-live artifacts (paper 1b v1.0-v1.4) and are NOT edited in place.
This script independently reproduces the SAME enumeration (same F/R
matrices, same sigma_2/sigma_3/sigma_4 generators, same gate_set=(2,3,4),
psi0=|0>, L=3..9) and recomputes the corrected metric via two
independent routes that must agree bit-exactly:

  Route (i)  — direct closed-form:      N^2_P = a0+a1+a3+a24/2
  Route (ii) — explicit projector:      N^2_P = <psi| P |psi>,
               P = |0><0| + |1><1| + |3><3| + |c><c|,  |c>=(|2>+|4>)/sqrt(2)

Positive controls (must all pass before the enumeration is trusted):
  - N^2_P(|11>_L) = N^2_P((|2>+|4>)/sqrt(2)) = 1   (fully inside subspace)
  - N^2_P((|2>-|4>)/sqrt(2))                 = 0   (fully orthogonal, leaks completely)
  - max(N^2_P) over ALL enumerated states, ALL L <= 1  (projector never exceeds unity)

Output: n2_reanchor.json (per-L N^2_P statistics + control results).
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent
phi = (1 + np.sqrt(5)) / 2
phi_inv = 1 / phi
sqrt_phi_inv = np.sqrt(phi_inv)

# R-eigenvalues (Fibonacci TQFT) -- identical to landscape_validation.py
R1 = np.exp(-4j * np.pi / 5)
Rt = np.exp(+3j * np.pi / 5)

# F-matrix on the 2D fusion channel -- identical to landscape_validation.py
F = np.array([[phi_inv, sqrt_phi_inv], [sqrt_phi_inv, -phi_inv]])
Finv = np.linalg.inv(F)

# ------------------------------------------------------------------
# 5D Basis (linear tree, 6 anyons -> vacuum) -- identical convention:
# |0> = (1,tau,1)  |1> = (1,tau,tau)  |2> = (tau,1,tau)
# |3> = (tau,tau,1)  |4> = (tau,tau,tau)
# ------------------------------------------------------------------

def build_sigma2() -> np.ndarray:
    M = np.zeros((5, 5), dtype=np.complex128)
    B = F @ np.diag([R1, Rt]) @ Finv
    idx = [0, 3]
    for i in range(2):
        for j in range(2):
            M[idx[i], idx[j]] = B[i, j]
    idx = [1, 4]
    for i in range(2):
        for j in range(2):
            M[idx[i], idx[j]] = B[i, j]
    M[2, 2] = Rt
    return M

def build_sigma4() -> np.ndarray:
    M = np.zeros((5, 5), dtype=np.complex128)
    B = F @ np.diag([R1, Rt]) @ Finv
    idx = [0, 1]
    for i in range(2):
        for j in range(2):
            M[idx[i], idx[j]] = B[i, j]
    idx = [3, 4]
    for i in range(2):
        for j in range(2):
            M[idx[i], idx[j]] = B[i, j]
    M[2, 2] = Rt
    return M

def build_sigma3() -> np.ndarray:
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

SIGMA = {2: build_sigma2(), 3: build_sigma3(), 4: build_sigma4()}

# Positive control: unitarity of the 3 generators used in the enumeration
for _k, _M in SIGMA.items():
    assert np.allclose(_M @ _M.conj().T, np.eye(5), atol=1e-10), f"sigma{_k} not unitary"

# ------------------------------------------------------------------
# Route (ii): explicit projector P = |0><0| + |1><1| + |3><3| + |c><c|
# ------------------------------------------------------------------
c_vec = np.zeros(5, dtype=np.complex128)
c_vec[2] = 1 / np.sqrt(2)
c_vec[4] = 1 / np.sqrt(2)

P = np.zeros((5, 5), dtype=np.complex128)
for _i in (0, 1, 3):
    e = np.zeros(5, dtype=np.complex128)
    e[_i] = 1.0
    P += np.outer(e, e.conj())
P += np.outer(c_vec, c_vec.conj())

# P must itself be a valid (idempotent, hermitian) rank-4 projector
assert np.allclose(P, P.conj().T, atol=1e-12), "P not Hermitian"
assert np.allclose(P @ P, P, atol=1e-10), "P not idempotent"
assert abs(np.trace(P).real - 4.0) < 1e-10, "P not rank 4"

def n2_direct(psi: np.ndarray) -> float:
    """Route (i): closed-form N^2_P = a0+a1+a3+a24/2."""
    a0 = abs(psi[0]) ** 2
    a1 = abs(psi[1]) ** 2
    a3 = abs(psi[3]) ** 2
    a24 = abs(psi[2] + psi[4]) ** 2
    return float(a0 + a1 + a3 + a24 / 2.0)

def n2_projector(psi: np.ndarray) -> float:
    """Route (ii): explicit projector <psi|P|psi>."""
    val = np.vdot(psi, P @ psi)
    return float(val.real)

def apply_sequence(gates: list[int], psi0: np.ndarray) -> np.ndarray:
    psi = psi0.copy()
    for g in gates:
        psi = SIGMA[g] @ psi
    return psi

# ------------------------------------------------------------------
# Positive controls (must pass BEFORE trusting the enumeration below)
# ------------------------------------------------------------------
controls = {}

# |11>_L = (|2>+|4>)/sqrt(2)  -> N^2_P must be exactly 1
psi_11 = c_vec.copy()
controls["N2_P(|11>_L) == 1"] = {
    "direct": n2_direct(psi_11), "projector": n2_projector(psi_11),
    "pass": abs(n2_direct(psi_11) - 1.0) < 1e-12 and abs(n2_projector(psi_11) - 1.0) < 1e-12,
}

# (|2>-|4>)/sqrt(2) -> orthogonal to the logical subspace -> N^2_P must be 0
psi_orth = np.zeros(5, dtype=np.complex128)
psi_orth[2] = 1 / np.sqrt(2)
psi_orth[4] = -1 / np.sqrt(2)
controls["N2_P((|2>-|4>)/sqrt2) == 0"] = {
    "direct": n2_direct(psi_orth), "projector": n2_projector(psi_orth),
    "pass": abs(n2_direct(psi_orth)) < 1e-12 and abs(n2_projector(psi_orth)) < 1e-12,
}

# |0>, |1>, |3> individually -> must be exactly 1 (already logical basis vectors)
for _i in (0, 1, 3):
    _psi = np.zeros(5, dtype=np.complex128)
    _psi[_i] = 1.0
    controls[f"N2_P(|{_i}>) == 1"] = {
        "direct": n2_direct(_psi), "projector": n2_projector(_psi),
        "pass": abs(n2_direct(_psi) - 1.0) < 1e-12 and abs(n2_projector(_psi) - 1.0) < 1e-12,
    }

assert all(c["pass"] for c in controls.values()), f"POSITIVE CONTROL FAILED: {controls}"

# ------------------------------------------------------------------
# Enumeration -- identical scheme to landscape_validation.py:
# gate_set=(2,3,4), psi0=|0>=[1,0,0,0,0], L=3..9, all 3^L sequences
# ------------------------------------------------------------------
gate_set = (2, 3, 4)
psi0 = np.array([1, 0, 0, 0, 0], dtype=np.complex128)

results = {}
route_mismatch_max = 0.0
global_max_n2p = 0.0

for L in range(3, 10):
    t0 = time.time()
    n_total = len(gate_set) ** L
    n2_dir = np.zeros(n_total)
    n2_prj = np.zeros(n_total)
    best_seq_min = None
    for idx in range(n_total):
        seq = []
        x = idx
        for _ in range(L):
            seq.append(gate_set[x % len(gate_set)])
            x //= len(gate_set)
        psi5 = apply_sequence(seq, psi0)
        d = n2_direct(psi5)
        p = n2_projector(psi5)
        n2_dir[idx] = d
        n2_prj[idx] = p
        route_mismatch_max = max(route_mismatch_max, abs(d - p))
    dt = time.time() - t0
    min_idx = int(np.argmin(n2_dir))
    # decode the argmin sequence for provenance
    x = min_idx
    min_seq = []
    for _ in range(L):
        min_seq.append(gate_set[x % len(gate_set)])
        x //= len(gate_set)
    global_max_n2p = max(global_max_n2p, float(n2_dir.max()))
    results[f"L={L}"] = {
        "n_sequences": n_total,
        "N2_P_min": float(n2_dir.min()),
        "N2_P_mean": float(n2_dir.mean()),
        "N2_P_max": float(n2_dir.max()),
        "argmin_seq": min_seq,
    }
    print(f"L={L}: N2_P min={n2_dir.min():.6f} mean={n2_dir.mean():.6f} "
          f"max={n2_dir.max():.6f}  (route mismatch so far: {route_mismatch_max:.2e}, "
          f"walltime {dt:.3f}s)")

assert route_mismatch_max < 1e-12, f"Route (i) and (ii) disagree: max diff {route_mismatch_max:.3e}"
assert global_max_n2p <= 1.0 + 1e-10, f"N2_P exceeded 1: {global_max_n2p}"

out = {
    "meta": {
        "script": "n2_reanchor.py",
        "purpose": "v1.5 re-anchor: corrected subspace-fidelity N2_P with 1/2 factor",
        "same_enumeration_as": "landscape_validation.py (gate_set=(2,3,4), psi0=|0>, L=3..9)",
        "route_max_mismatch": route_mismatch_max,
        "global_max_N2_P": global_max_n2p,
    },
    "positive_controls": controls,
    "results_per_length": results,
}
(OUT / "n2_reanchor.json").write_text(json.dumps(out, indent=2))
print(f"\nAll positive controls PASS. Route (i)/(ii) max mismatch: {route_mismatch_max:.2e}")
print(f"Global max N2_P over all L: {global_max_n2p:.6f} (<=1: OK)")
print(f"WRITE {OUT / 'n2_reanchor.json'}")
