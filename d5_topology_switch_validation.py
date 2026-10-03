#!/usr/bin/env python3
"""
D5 -- Generator and validation of topology_switch_results.json
================================================================
Computes |S|(delta), C(delta) for the sigma3-block test (three-generator
encoding: blocking the cross-bipartition generator sigma3 makes the
concurrence vanish exactly in the assignment |11> <- |4>)
and the AM/MB-block test (two-generator d1b encoding), for the fixed
representative sequences of the paper (SEQS_5D, SEQ_D1B below; they are
representative sequences, not winners of any search).

  --regenerate : writes topology_switch_results.json from this script
                 (every field of every entry is computed here; schema in
                 with_summary);
  no flag      : validates the file against a fresh recomputation and
                 writes d5_topology_switch_validation_results.json.

Open 5D configurations use the Protocol A map (project_protocol_a,
|11>_L <- (|2>+|4>)/sqrt(2)); blocked ones (sigma3 -> I) the natural
assignment |11>_L <- |4>, with the Protocol A values reported alongside.
Given a sequence and a choice of which generator is replaced by the
identity, the S(delta)/C(delta) curve is a closed-form, deterministic
calculation (Horodecki, Horodecki & Horodecki 1995).

Reproducible: no randomness in the closed-form path; the independent
measurement-angle cross-check uses a fixed seed.
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


def gates_d1b(delta: float, block: str | None) -> dict:
    I4 = np.eye(4, dtype=complex)
    g = {'A': gate_AM_d1b(delta), 'B': gate_MB_d1b(delta)}
    if block in g:
        g[block] = I4
    return g


def apply_d1b(seq: str, delta: float, block: str | None) -> np.ndarray:
    g = gates_d1b(delta, block)
    psi = np.array([1, 0, 0, 0], dtype=complex)
    for ch in seq:
        psi = g[ch] @ psi
    n = np.linalg.norm(psi)
    return psi / n if n > 1e-15 else psi


# ----------------------------------------------------------------------
# 5D encoding (three generators sigma2, sigma3, sigma4)
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


def gates_5d(delta: float, block: str | None) -> dict:
    I5 = np.eye(5, dtype=complex)
    g = {'2': sigma2_5d(delta), '3': sigma3_5d(delta), '4': sigma4_5d(delta)}
    if block in g:
        g[block] = I5
    return g


def apply_5d(seq: str, delta: float, block: str | None) -> np.ndarray:
    g = gates_5d(delta, block)
    psi = np.array([1, 0, 0, 0, 0], dtype=complex)
    for ch in seq:
        psi = g[ch] @ psi
    return psi


def project_natural(psi5: np.ndarray) -> np.ndarray:
    """Natural assignment |11>_L <- |4>: |2> is discarded and the rest renormalized.
    For words without sigma3 the weight on |2> is exactly zero, so nothing is discarded
    and the state is the product state the two local generators produce."""
    psi4 = np.array([psi5[0], psi5[1], psi5[3], psi5[4]], dtype=complex)
    n = np.linalg.norm(psi4)
    return psi4 / n if n > 1e-15 else psi4


def project_protocol_a(psi5: np.ndarray) -> np.ndarray:
    """Protocol A: |11>_L <- (|2>+|4>)/sqrt(2), then renormalize (landscape_validation.project_4d).
    A state without weight on |2> keeps only half of its |4> weight, so this map can
    report C > 0 for a product state."""
    psi4 = np.array([psi5[0], psi5[1], psi5[3], (psi5[2] + psi5[4]) / math.sqrt(2)], dtype=complex)
    n = np.linalg.norm(psi4)
    return psi4 / n if n > 1e-15 else psi4


# ----------------------------------------------------------------------
# Shared measures
# ----------------------------------------------------------------------

def concurrence(psi: np.ndarray) -> float:
    return float(2 * abs(psi[0] * psi[3] - psi[1] * psi[2]))


def chsh_horodecki(C: float) -> float:
    return 2.0 * math.sqrt(1.0 + C * C)


def correlator_bloch(psi, theta_A, phi_A, theta_B, phi_B) -> float:
    sx = math.sin(theta_A) * math.cos(phi_A); sy = math.sin(theta_A) * math.sin(phi_A); cz = math.cos(theta_A)
    MA = np.array([[cz, sx - 1j*sy], [sx + 1j*sy, -cz]], dtype=complex)
    sx = math.sin(theta_B) * math.cos(phi_B); sy = math.sin(theta_B) * math.sin(phi_B); cz = math.cos(theta_B)
    MB = np.array([[cz, sx - 1j*sy], [sx + 1j*sy, -cz]], dtype=complex)
    M = np.kron(MA, MB)
    return float(np.real(psi.conj() @ M @ psi))


def chsh_optimize_bloch(psi, seed: int = 2026, n_restarts: int = 8):
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


N_POINTS = 50
TOL = 1e-10          # violation rule S > 2 + TOL (same convention as landscape_validation.py)
SEQS_5D = ["423", "423333", "23444432"]   # representative 5D sequences of the paper (not winners)
SEQ_D1B = "ABABABABABAB"                   # Table I headline row (d1b)


def sweep_5d(seq: str, block: str | None) -> dict:
    """Blocked configurations (block == '3') are evaluated in the natural assignment
    |11>_L <- |4> (|2> carries no weight there); open configurations use the Protocol A
    map project_protocol_a. For blocked configurations the Protocol A values are
    returned alongside (C_protocol_A, S_protocol_A)."""
    deltas = np.linspace(0, 2*math.pi, N_POINTS, endpoint=False)
    S = np.zeros(N_POINTS); C = np.zeros(N_POINTS)
    S_pa = np.zeros(N_POINTS); C_pa = np.zeros(N_POINTS); leak = np.zeros(N_POINTS)
    for i, d in enumerate(deltas):
        psi5 = apply_5d(seq, d, block)
        psi4 = project_natural(psi5) if block == '3' else project_protocol_a(psi5)
        C[i] = concurrence(psi4)
        S[i] = chsh_horodecki(C[i])
        if block == '3':
            leak[i] = abs(psi5[2]) ** 2
            C_pa[i] = concurrence(project_protocol_a(psi5))
            S_pa[i] = chsh_horodecki(C_pa[i])
    out = dict(seq=seq, encoding="5D", block=block, n_points=N_POINTS,
               delta=deltas.tolist(), S=S.tolist(), C=C.tolist())
    if block == '3':
        out["assignment"] = "natural: |11>_L <- |4>"
        out["leakage_max"] = float(leak.max())
        out["C_protocol_A"] = C_pa.tolist()
        out["S_protocol_A"] = S_pa.tolist()
    return out


def sweep_d1b(seq: str, block: str | None) -> dict:
    deltas = np.linspace(0, 2*math.pi, N_POINTS, endpoint=False)
    S = np.zeros(N_POINTS); C = np.zeros(N_POINTS)
    for i, d in enumerate(deltas):
        psi = apply_d1b(seq, d, block)
        C[i] = concurrence(psi)
        S[i] = chsh_horodecki(C[i])
    return dict(seq=seq, encoding="d1b", block=block, n_points=N_POINTS,
                delta=deltas.tolist(), S=S.tolist(), C=C.tolist())


def compare(recomputed: dict, shipped: dict) -> tuple[float, float]:
    dS = float(np.max(np.abs(np.array(recomputed["S"]) - np.array(shipped["S"]))))
    dC = float(np.max(np.abs(np.array(recomputed["C"]) - np.array(shipped["C"]))))
    return dS, dC


def describe(r: dict) -> str:
    L = len(r["seq"])
    if r["encoding"] == "5D":
        tag = "[open]" if r["block"] is None else "[sigma3 -> I (blocked)]"
    else:
        tag = {None: "[open]", 'A': "[AM -> I (blocked)]", 'B': "[MB -> I (blocked)]"}[r["block"]]
    return f"L={L}, sequence {r['seq']} {tag}"


def with_summary(r: dict) -> dict:
    """Entry schema of topology_switch_results.json, written in full by this script:
    seq, encoding, block, n_points, delta, S, C, S_max, S_mean, C_max,
    frac_bell (fraction of grid points with S > 2 + TOL), description;
    blocked 5D entries add assignment, leakage_max, C_protocol_A, S_protocol_A."""
    S = np.array(r["S"]); C = np.array(r["C"])
    out = dict(seq=r["seq"], encoding=r["encoding"], block=r["block"], n_points=r["n_points"],
               delta=r["delta"], S=r["S"], C=r["C"],
               S_max=float(S.max()), S_mean=float(S.mean()), C_max=float(C.max()),
               frac_bell=float(np.mean(S > 2.0 + TOL)), description=describe(r))
    for k in ("assignment", "leakage_max", "C_protocol_A", "S_protocol_A"):
        if k in r:
            out[k] = r[k]
    return out


def regenerate(shipped_path: Path) -> None:
    """Write topology_switch_results.json from this script. Positive control against a
    previously shipped file, if present: the d1b curves involve no projection and must
    reproduce the shipped values to < 1e-12; blocked 5D curves give S = 2 exactly."""
    old = json.loads(shipped_path.read_text(encoding="utf-8")) if shipped_path.exists() else None
    new = {"three_generator_sigma3_block": [with_summary(sweep_5d(s, b)) for s in SEQS_5D for b in (None, '3')],
           "two_generator_AM_MB_block": [with_summary(sweep_d1b(SEQ_D1B, b)) for b in (None, 'A', 'B')]}
    if old is not None:
        old_a2 = {(r["seq"], r["block"]): r for r in old["two_generator_AM_MB_block"]}
        d_max = 0.0
        for r in new["two_generator_AM_MB_block"]:
            d_max = max(d_max, *compare(r, old_a2[(r["seq"], r["block"])]))
        print(f"  positive control, d1b curves vs previously shipped file: max|diff| = {d_max:.2e}"
              f"  (< 1e-12: {d_max < 1e-12})")
        old_fields = set().union(*(r.keys() for g in old.values() for r in g))
        new_fields = set().union(*(r.keys() for g in new.values() for r in g))
        print(f"  fields dropped: {sorted(old_fields - new_fields)}  fields added: {sorted(new_fields - old_fields)}")
    for r in new["three_generator_sigma3_block"]:
        if r["block"] == '3':
            print(f"  blocked {r['seq']:<10} S == 2 exactly at all {r['n_points']} points: "
                  f"{all(s == 2.0 for s in r['S'])}   C_max = {r['C_max']:.2e}")
    shipped_path.write_text(json.dumps(new, indent=2))
    print(f"REGENERATED {shipped_path}")


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    shipped_path = HERE / "topology_switch_results.json"
    if "--regenerate" in sys.argv:
        regenerate(shipped_path)
        return
    shipped = json.loads(shipped_path.read_text(encoding="utf-8"))
    shipped_a1 = {(r["seq"], r["block"]): r for r in shipped["three_generator_sigma3_block"]}
    shipped_a2 = {(r["seq"], r["block"]): r for r in shipped["two_generator_AM_MB_block"]}

    print("=" * 72)
    print("  D5 -- topology_switch_results.json validation")
    print("=" * 72)

    max_diff_overall = 0.0
    per_curve = []
    recomputed_a1 = []
    recomputed_a2 = []
    summary_ok = True

    def summary_matches(r, ship):
        s = with_summary(r)
        return all(s[k] == ship.get(k) for k in ("S_max", "S_mean", "C_max", "frac_bell", "description"))

    print("\n[Three-generator encoding (5D), sigma3 block]")
    for seq in SEQS_5D:
        for block in [None, '3']:
            r = sweep_5d(seq, block)
            recomputed_a1.append(r)
            ship = shipped_a1[(seq, block)]
            dS, dC = compare(r, ship)
            summary_ok = summary_ok and summary_matches(r, ship)
            max_diff_overall = max(max_diff_overall, dS, dC)
            per_curve.append(dict(seq=seq, block=block, max_abs_diff_S=dS, max_abs_diff_C=dC))
            tag = "open" if block is None else f"block sigma{block}->I"
            print(f"  seq={seq:<12} {tag:<16} max|dS|={dS:.2e}  max|dC|={dC:.2e}  "
                  f"C_max={max(r['C']):.6f}")

    print("\n[Two-generator circuit model (d1b), AM/MB block]")
    seq_d1b = SEQ_D1B
    for block in [None, 'A', 'B']:
        r = sweep_d1b(seq_d1b, block)
        recomputed_a2.append(r)
        ship = shipped_a2[(seq_d1b, block)]
        dS, dC = compare(r, ship)
        summary_ok = summary_ok and summary_matches(r, ship)
        max_diff_overall = max(max_diff_overall, dS, dC)
        per_curve.append(dict(seq=seq_d1b, block=block, max_abs_diff_S=dS, max_abs_diff_C=dC))
        tag = "open" if block is None else f"block {block}M->I"
        print(f"  seq={seq_d1b}  {tag:<10} max|dS|={dS:.2e}  max|dC|={dC:.2e}  "
              f"C_max={max(r['C']):.6f}")

    # sigma3-block check: blocking sigma3 (the cross-bipartition generator)
    # must collapse concurrence to exactly 0 for every 5D sequence, at
    # every sector phase -- this is the paper's central topological claim.
    print("\n[sigma3-block check: natural assignment -> C=0 exactly, all sequences/phases;"
          " Protocol A on the same product states reported alongside]")
    gate_ok = True
    for r in recomputed_a1:
        if r["block"] == '3':
            cmax = max(r["C"])
            ok = cmax < 1e-12
            gate_ok = gate_ok and ok
            print(f"  seq={r['seq']:<12} max C over 50 phases = {cmax:.3e}  [{'PASS' if ok else 'FAIL'}]"
                  f"  leakage_max={r['leakage_max']:.1e}  Protocol A: C_max={max(r['C_protocol_A']):.4f}"
                  f" S_max={max(r['S_protocol_A']):.4f}")

    # Independent cross-check: full Bloch-sphere measurement-angle
    # optimization at the argmax of one representative curve per
    # encoding, confirming the closed-form Horodecki value is achieved.
    print("\n[Cross-check: closed-form vs. explicit measurement optimum]")
    cross_checks = []
    for label, r in [("5D open, seq=23444432", next(x for x in recomputed_a1 if x["seq"] == "23444432" and x["block"] is None)),
                     ("d1b open, seq=ABABABABABAB", next(x for x in recomputed_a2 if x["block"] is None))]:
        i_best = int(np.argmax(r["S"]))
        d_best = r["delta"][i_best]
        if r["encoding"] == "5D":
            psi = project_protocol_a(apply_5d(r["seq"], d_best, r["block"]))
        else:
            psi = apply_d1b(r["seq"], d_best, r["block"])
        s_opt = chsh_optimize_bloch(psi)
        s_h = r["S"][i_best]
        gap = abs(s_opt - s_h)
        cross_checks.append(dict(label=label, delta=d_best, S_horodecki=s_h, S_optimized=s_opt, gap=gap))
        print(f"  {label:<28} delta={d_best:.6f}  S_H={s_h:.10f}  S_opt={s_opt:.10f}  gap={gap:.2e}")

    max_gap = max(c["gap"] for c in cross_checks)
    checks = dict(
        max_abs_diff_S_or_C_over_all_curves=max_diff_overall,
        summary_fields_match_shipped=summary_ok,
        sigma3_block_gives_exact_C_zero=gate_ok,
        closed_form_matches_explicit_optimum_max_gap=max_gap,
    )
    checks["ALL_PASS"] = bool(max_diff_overall < 1e-12 and summary_ok and gate_ok and max_gap < 1e-6)

    out = dict(
        meta=dict(script="d5_topology_switch_validation.py", n_points=N_POINTS,
                   note="Closed-form Horodecki sweeps of the representative sigma3-block and "
                        "AM/MB-block sequences (not winners of any search); open 5D curves under "
                        "Protocol A, blocked ones in the natural assignment. topology_switch_results.json "
                        "is written by this script (--regenerate) and validated here.",
                   schema=dict(generated_fields=["seq", "encoding", "block", "n_points", "delta", "S", "C",
                                                 "S_max", "S_mean", "C_max", "frac_bell", "description"],
                               blocked_5d_extra_fields=["assignment", "leakage_max", "C_protocol_A", "S_protocol_A"],
                               dropped_fields=["S_opt", "S_horodecki"],
                               frac_bell="fraction of grid points with S > 2 + 1e-10")),
        per_curve=per_curve,
        blocked_configurations_under_protocol_A=[
            dict(seq=r["seq"], block=r["block"], leakage_max=r["leakage_max"],
                 C_natural_max=float(max(r["C"])),
                 C_protocol_A_max=float(max(r["C_protocol_A"])), S_protocol_A_max=float(max(r["S_protocol_A"])),
                 C_protocol_A=r["C_protocol_A"], S_protocol_A=r["S_protocol_A"])
            for r in recomputed_a1 if r["block"] == '3'],
        cross_checks=cross_checks,
        checks=checks,
    )
    (HERE / "d5_topology_switch_validation_results.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    for k, v in checks.items():
        print(f"  [{'PASS' if (v is True or (isinstance(v, float) and v < 1e-6)) else 'INFO'}] {k} = {v}")
    print(f"\n  ALL_PASS = {checks['ALL_PASS']}")
    print(f"  WROTE {HERE / 'd5_topology_switch_validation_results.json'}")


if __name__ == "__main__":
    main()
