#!/usr/bin/env python3
"""
D9 -- Braiding alone on the two-generator fusion code, and the braid
relations of the deformed five-dimensional representation
=====================================================================
Part 1 (fusion code, delta = 0). The two-generator d_1b encoding uses the
braid generators sigma_2 (anyons 2-3, acts on Alice's charge x_1) and
sigma_4 (anyons 4-5, acts on Bob's charge x_3) of the five-dimensional
representation of landscape_validation.py. On the four basis states with
x_2 = tau,

    |00>_L <- |0>,  |01>_L <- |1>,  |10>_L <- |3>,  |11>_L <- |4>,

the two generators act as single-qubit gates on different anyons and
commute, and |2> = (tau, 1, tau) is never reached. Starting from |0>, ALL
braid words of length 1..12 over {sigma_2, sigma_4} (2 + 4 + ... + 4096 =
8190 words) are enumerated; for each word the weight on |2> (leakage), the
concurrence C of the logical two-qubit state and the Horodecki value
|S| = 2 sqrt(1 + C^2) are recorded. Expected: leakage 0, C at machine zero,
|S| = 2 exactly for every word. Positive control: the word sigma_2, sigma_4,
sigma_3 (the cross-bipartition generator) on |0> gives C = 1.

Part 2 (braid relations on the delta grid). With the deformation
R_tau -> R_tau e^{i delta} (everywhere R_tau enters) the Yang-Baxter
residua ||s2 s3 s2 - s3 s2 s3||_F and ||s3 s4 s3 - s4 s3 s4||_F, the
distant commutator ||[s2, s4]||_F and the mirror relation
conj(sigma_i(delta)) = e^{i phi} sigma_i(6 pi/5 - delta) are evaluated on
the 50-point grid delta_k = 2 pi k / 50 of s_delta_sweeps.json, and at the
delta values of Table tab:yb_under_delta.

The generators at delta = 0 are checked against landscape_validation.SIGMA
(imported read-only) before anything else is computed. Deterministic, no
random numbers, numpy only. Output: d9_fusion_code_braiding_only_results.json
(the runtime is printed, not stored, so the file reproduces byte for byte).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

import landscape_validation as lv

HERE = Path(__file__).parent
LMAX = 12
N_GRID = 50
TOL_EXACT = 1e-12

PHI = (1 + np.sqrt(5)) / 2
F2 = np.array([[1 / PHI, 1 / np.sqrt(PHI)], [1 / np.sqrt(PHI), -1 / PHI]], dtype=np.complex128)
R1 = np.exp(-4j * np.pi / 5)
RT = np.exp(+3j * np.pi / 5)


def generators(delta: float = 0.0) -> dict:
    """sigma_2, sigma_3, sigma_4 of the five-dimensional representation with
    R_tau -> R_tau e^{i delta} (block structure as in landscape_validation.py)."""
    r_tau = RT * np.exp(1j * delta)
    B = F2 @ np.diag([R1, r_tau]) @ F2
    s2 = np.zeros((5, 5), dtype=np.complex128)
    s4 = np.zeros((5, 5), dtype=np.complex128)
    s3 = np.zeros((5, 5), dtype=np.complex128)
    for idx in ([0, 3], [1, 4]):
        for i in range(2):
            for j in range(2):
                s2[idx[i], idx[j]] = B[i, j]
    s2[2, 2] = r_tau
    for idx in ([0, 1], [3, 4]):
        for i in range(2):
            for j in range(2):
                s4[idx[i], idx[j]] = B[i, j]
    s4[2, 2] = r_tau
    s3[0, 0] = R1
    s3[1, 1] = r_tau
    s3[3, 3] = r_tau
    for i in range(2):
        for j in range(2):
            s3[[2, 4][i], [2, 4][j]] = B[i, j]
    return {2: s2, 3: s3, 4: s4}


def fro(A: np.ndarray) -> float:
    return float(np.linalg.norm(A))


def logical_natural(psi5: np.ndarray) -> np.ndarray:
    """|11>_L <- |4>; |2> is discarded and the rest renormalized."""
    psi4 = np.array([psi5[0], psi5[1], psi5[3], psi5[4]], dtype=np.complex128)
    n = np.linalg.norm(psi4)
    return psi4 / n if n > 1e-15 else psi4


def logical_protocol_a(psi5: np.ndarray) -> np.ndarray:
    """Protocol A: |11>_L <- (|2>+|4>)/sqrt(2), then renormalize."""
    psi4 = np.array([psi5[0], psi5[1], psi5[3], (psi5[2] + psi5[4]) / np.sqrt(2)], dtype=np.complex128)
    n = np.linalg.norm(psi4)
    return psi4 / n if n > 1e-15 else psi4


def concurrence(psi4: np.ndarray) -> float:
    return float(2 * abs(psi4[0] * psi4[3] - psi4[1] * psi4[2]))


def horodecki(C: float) -> float:
    return float(2 * np.sqrt(1 + C * C))


def part1(G: dict) -> dict:
    s2, s3, s4 = G[2], G[3], G[4]
    psi0 = np.array([1, 0, 0, 0, 0], dtype=np.complex128)
    states = [psi0]
    n_words = 0
    leak_max = 0.0
    c_max, c_argmax = 0.0, None
    s_min, s_max = np.inf, -np.inf
    per_length = {}
    words = [()]
    for L in range(1, LMAX + 1):
        new_states, new_words = [], []
        for psi, w in zip(states, words):
            for g, M in ((2, s2), (4, s4)):
                new_states.append(M @ psi)
                new_words.append(w + (g,))
        states, words = new_states, new_words
        leak_L = 0.0
        c_L = 0.0
        for psi, w in zip(states, words):
            leak = float(abs(psi[2]) ** 2)
            C = concurrence(logical_natural(psi))
            S = horodecki(C)
            leak_L = max(leak_L, leak)
            if C > c_L:
                c_L = C
            if C > c_max:
                c_max, c_argmax = C, "".join(str(g) for g in w)
            s_min, s_max = min(s_min, S), max(s_max, S)
            n_words += 1
        leak_max = max(leak_max, leak_L)
        per_length[str(L)] = dict(n_words=len(words), leakage_max=leak_L, C_max=c_L)
    # positive control: sigma_2, sigma_4, sigma_3 on |0>
    psi = s3 @ (s4 @ (s2 @ psi0))
    pc = dict(word="243", leakage=float(abs(psi[2]) ** 2),
              C_natural=concurrence(logical_natural(psi)),
              C_protocol_A=concurrence(logical_protocol_a(psi)))
    return dict(n_words=n_words, leakage_max=leak_max, C_max=c_max, C_argmax_word=c_argmax,
                S_min=float(s_min), S_max=float(s_max),
                n_words_with_S_exactly_2=None,  # filled below
                per_length=per_length, positive_control_with_sigma3=pc)


def relations(G: dict) -> dict:
    s2, s3, s4 = G[2], G[3], G[4]
    return dict(yb_23=fro(s2 @ s3 @ s2 - s3 @ s2 @ s3),
                yb_34=fro(s3 @ s4 @ s3 - s4 @ s3 @ s4),
                comm_24=fro(s2 @ s4 - s4 @ s2))


def mirror_deviation(delta: float) -> float:
    """max over i of || conj(sigma_i(delta)) - e^{i phi} sigma_i(6 pi/5 - delta) ||_F,
    phi taken from the overlap (no square-root floor)."""
    Ga, Gb = generators(delta), generators(6 * np.pi / 5 - delta)
    worst = 0.0
    for i in (2, 3, 4):
        A, B = np.conj(Ga[i]), Gb[i]
        ov = np.vdot(B, A)                     # Tr(B^dagger A)
        ph = ov / abs(ov) if abs(ov) > 0 else 1.0
        worst = max(worst, fro(A - ph * B))
    return worst


def main():
    sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    t0 = time.time()
    print("=" * 72)
    print("  D9 -- braiding alone on the two-generator fusion code; braid relations on the delta grid")
    print("=" * 72)

    # generators at delta = 0 must equal landscape_validation.SIGMA (read-only import)
    G0 = generators(0.0)
    dev_lv = max(fro(G0[i] - lv.SIGMA[i]) for i in (2, 3, 4))
    print(f"\n[Control] generators(0) vs landscape_validation.SIGMA: max ||diff||_F = {dev_lv:.2e}")
    assert dev_lv < 1e-14, "generator convention drifted from landscape_validation.py"

    # ---- Part 1
    p1 = part1(G0)
    # count words with |S| = 2 within 1e-12 (recomputed cheaply from C_max: all words share C ~ 0)
    p1["n_words_with_S_exactly_2"] = p1["n_words"] if abs(p1["S_max"] - 2) <= TOL_EXACT and abs(p1["S_min"] - 2) <= TOL_EXACT else None
    rel0 = relations(G0)
    print(f"\n[Part 1] {p1['n_words']} words over {{sigma_2, sigma_4}}, L = 1..{LMAX}, from |0>:")
    print(f"  max leakage |psi_2|^2 = {p1['leakage_max']:.1e}")
    print(f"  max C = {p1['C_max']:.1e} (word {p1['C_argmax_word']})")
    print(f"  |S| in [{p1['S_min']:.12f}, {p1['S_max']:.12f}]  -- all {p1['n_words_with_S_exactly_2']} words at |S| = 2 within {TOL_EXACT:g}")
    print(f"  ||[sigma_2, sigma_4]||_F = {rel0['comm_24']:.1e}   Yang-Baxter (2,3) {rel0['yb_23']:.1e}  (3,4) {rel0['yb_34']:.1e}")
    pc = p1["positive_control_with_sigma3"]
    print(f"  positive control sigma_2 -> sigma_4 -> sigma_3 on |0>: leakage {pc['leakage']:.4f}, "
          f"C_natural = {pc['C_natural']:.4f}, C_protocol_A = {pc['C_protocol_A']:.4f}")

    # ---- Part 2: braid relations on the 50-point grid
    deltas = np.linspace(0, 2 * np.pi, N_GRID, endpoint=False)
    grid = []
    for k, d in enumerate(deltas):
        r = relations(generators(float(d)))
        r.update(k=k, delta_over_pi=float(d / np.pi), yb_max=max(r["yb_23"], r["yb_34"]),
                 mirror_deviation=mirror_deviation(float(d)))
        grid.append(r)
    exact = [g["k"] for g in grid if g["yb_max"] < TOL_EXACT]
    rest = [g for g in grid if g["yb_max"] >= TOL_EXACT]
    yb_rest_min = min(rest, key=lambda g: g["yb_max"])
    comm_max = max(g["comm_24"] for g in grid)
    mirror_max = max(g["mirror_deviation"] for g in grid)
    table_deltas = [0.0, 0.1, 0.25, 0.5, 1.0, 1.191, 1.5]
    table = []
    for dp in table_deltas:
        r = relations(generators(dp * np.pi))
        table.append(dict(delta_over_pi=dp, yb_23=r["yb_23"], yb_34=r["yb_34"], comm_24=r["comm_24"]))
    print(f"\n[Part 2] braid relations on the {N_GRID}-point grid delta_k = 2 pi k / {N_GRID}:")
    print(f"  Yang-Baxter < {TOL_EXACT:g} exactly at k = {exact} (delta/pi = {[round(2 * k / N_GRID, 3) for k in exact]})")
    print(f"  smallest Yang-Baxter residuum elsewhere: {yb_rest_min['yb_max']:.4f} at k = {yb_rest_min['k']} "
          f"(and mirrored: k = {[g['k'] for g in rest if abs(g['yb_max'] - yb_rest_min['yb_max']) < 1e-9]})")
    print(f"  largest distant commutator ||[sigma_2, sigma_4]||_F over the grid: {comm_max:.1e}")
    print(f"  mirror conj(sigma_i(delta)) = e^(i phi) sigma_i(6 pi/5 - delta): largest deviation {mirror_max:.1e}")
    print("  Table tab:yb_under_delta points (YB residuum, Frobenius):")
    for t in table:
        print(f"    delta/pi = {t['delta_over_pi']:<6} yb_23 = {t['yb_23']:.5f}  yb_34 = {t['yb_34']:.5f}  comm = {t['comm_24']:.1e}")

    out = dict(
        meta=dict(script="d9_fusion_code_braiding_only.py", encoding="5D three-generator representation of "
                  "landscape_validation.py; fusion code = {sigma_2, sigma_4} on x_2 = tau",
                  logical_assignment="|00>,|01>,|10>,|11> <- |0>,|1>,|3>,|4>", Lmax=LMAX, n_grid=N_GRID,
                  tolerance_exact=TOL_EXACT, norm="Frobenius", seed=None,
                  generator_control_vs_landscape_validation=dev_lv),
        part1_fusion_code=dict(**p1, relations_at_delta0=rel0),
        part2_braid_relations_on_grid=dict(
            exact_k=exact, exact_delta_over_pi=[2 * k / N_GRID for k in exact],
            smallest_residuum_elsewhere=dict(value=yb_rest_min["yb_max"], k=yb_rest_min["k"]),
            distant_commutator_max=comm_max, mirror_deviation_max=mirror_max,
            grid=grid, table_yb_under_delta=table),
    )
    (HERE / "d9_fusion_code_braiding_only_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  runtime {time.time() - t0:.1f} s (printed, not stored)")
    print(f"  WROTE {HERE / 'd9_fusion_code_braiding_only_results.json'}")


if __name__ == "__main__":
    main()
