#!/usr/bin/env python3
"""
D10 -- What the Protocol A projection contributes to the CHSH-form values
of the three-generator encoding
=======================================================================
Protocol A maps the five-dimensional fusion state to two logical qubits by

    |00>_L <- |0>,  |01>_L <- |1>,  |10>_L <- |3>,  |11>_L <- (|2> + |4>)/sqrt(2),

followed by renormalization (project_4d in landscape_validation.py). A state
with no weight on |2> keeps only half of its |4> weight under this map, so
the map can turn a product state into a non-product one. This script
quantifies that for every length L = 2..9 and all 3^L words over
{sigma_2, sigma_3, sigma_4} (the pipeline of landscape_validation.py):

  * the number of words with |S|_A > 2 + TOL under Protocol A;
  * how many of these contain no sigma_3 (gates local to Alice or Bob only);
  * how many of these have concurrence 0 in the natural assignment
    |11>_L <- |4> (|2> discarded and the rest renormalized);
  * the best word under Protocol A, its |S|_A and its natural concurrence.

In addition, at delta = 0 it determines for L = 3, 6, 8, 10, 12 (the lengths
of s_delta_sweeps.json / Fig. 2) the first maximizer under Protocol A by a
two-pass rule: pass 1 finds S* = max |S|_A over all 3^L words; pass 2 returns
the first word in lexicographic order with |S|_A >= S* - 1e-10, the number of
such words (ties) and its natural concurrence (key delta0_first_maximizer).
These words are the GEN5D_SEQS of d4_s_delta_sweep_validation.py -- RUN ORDER:
d10 before d4. The running-best logic of the L = 2..9 table (band 1e-9) is
unchanged.

Words are written as digit strings in application order (left to right).
Enumeration order is lexicographic in that string (22..., 23..., 24..., 32...);
the best word is the first maximum in that order, and ties are counted.
Positive control: the per-length maxima and violation fractions must
reproduce S4_max and bell_pct_4D of landscape_validation.json.

Deterministic, numpy only. Output: d10_protocol_a_projection_audit_results.json
(runtime printed, not stored).
"""
from __future__ import annotations
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np

import landscape_validation as lv

HERE = Path(__file__).parent
TOL = lv.TOL          # 1e-10, same violation rule as landscape_validation.py
C_ZERO = 1e-10
LENGTHS = range(2, 10)
FIRST_MAX_LENGTHS = (3, 6, 8, 10, 12)   # lengths of s_delta_sweeps.json / Fig. 2
FIRST_MAX_BAND = 1e-10                 # pass-2 band below the maximum S*


def logical_natural(psi5: np.ndarray) -> np.ndarray:
    psi4 = np.array([psi5[0], psi5[1], psi5[3], psi5[4]], dtype=np.complex128)
    n = np.linalg.norm(psi4)
    return psi4 / n if n > 1e-15 else psi4


def audit_length(L: int) -> dict:
    gens = {g: lv.SIGMA[g] for g in (2, 3, 4)}
    psi0 = np.array([1, 0, 0, 0, 0], dtype=np.complex128)
    n_total = 3 ** L
    n_viol = n_no3 = n_nat0 = 0
    best_S, best_word, best_Cnat, n_ties = -1.0, None, None, 0
    S_all = np.zeros(n_total)
    for idx, word in enumerate(itertools.product((2, 3, 4), repeat=L)):
        psi = psi0
        for g in word:
            psi = gens[g] @ psi
        psi4 = lv.project_4d(psi)
        S_A = lv.chsh_4d_horodecki(psi4) if psi4 is not None else 0.0
        S_all[idx] = S_A
        if S_A > 2.0 + TOL:
            n_viol += 1
            no3 = 3 not in word
            C_nat = lv.concurrence_2q(logical_natural(psi))
            if no3:
                n_no3 += 1
            if C_nat < C_ZERO:
                n_nat0 += 1
        if S_A > best_S + 1e-9:
            best_S, best_word, n_ties = S_A, word, 1
            best_Cnat = lv.concurrence_2q(logical_natural(psi))
        elif abs(S_A - best_S) <= 1e-9:
            n_ties += 1
    return dict(L=L, n_words=n_total, n_S_A_above_2=n_viol, n_of_these_without_sigma3=n_no3,
                n_of_these_natural_C_zero=n_nat0,
                best=dict(S_A=float(best_S), word="".join(map(str, best_word)),
                          natural_C=float(best_Cnat), n_ties=n_ties),
                bell_pct_4D=float(100.0 * n_viol / n_total),
                at_bound_pct_4D=float(100.0 * np.mean(np.abs(S_all - 2.0) <= TOL)),
                S_A_max=float(S_all.max()))


def first_maximizer_delta0(L: int) -> dict:
    """Two-pass rule at delta = 0 over all 3^L words, enumerated in lexicographic
    order of the application-order digit string (as in audit_length):
      pass 1: S* = max |S|_A over all words;
      pass 2: winner = first word with |S|_A >= S* - FIRST_MAX_BAND;
              ties = number of such words; natural_C = concurrence of the winner
              in the natural assignment |11>_L <- |4>.
    The running-best logic of audit_length (band 1e-9) is left untouched."""
    gens = {g: lv.SIGMA[g] for g in (2, 3, 4)}
    psi0 = np.array([1, 0, 0, 0, 0], dtype=np.complex128)
    S_all = np.zeros(3 ** L)
    for idx, word in enumerate(itertools.product((2, 3, 4), repeat=L)):      # pass 1
        psi = psi0
        for g in word:
            psi = gens[g] @ psi
        psi4 = lv.project_4d(psi)
        S_all[idx] = lv.chsh_4d_horodecki(psi4) if psi4 is not None else 0.0
    S_star = float(S_all.max())
    hits = np.flatnonzero(S_all >= S_star - FIRST_MAX_BAND)                   # pass 2
    first = int(hits[0])
    word = next(itertools.islice(itertools.product((2, 3, 4), repeat=L), first, None))
    psi = psi0
    for g in word:
        psi = gens[g] @ psi
    return dict(word="".join(map(str, word)), S_A=float(S_all[first]), n_ties=int(hits.size),
                natural_C=float(lv.concurrence_2q(logical_natural(psi))), S_star=S_star)


def main():
    sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    t0 = time.time()
    print("=" * 72)
    print("  D10 -- Protocol A projection audit (three-generator encoding)")
    print("=" * 72)
    stored = json.loads((HERE / "landscape_validation.json").read_text(encoding="utf-8"))["results_per_length"]
    rows, control = [], []
    print(f"\n  {'L':>2} {'words':>6} {'S_A>2':>6} {'no s3':>6} {'natC=0':>7}   best S_A  word          natural C  ties")
    for L in LENGTHS:
        r = audit_length(L)
        rows.append(r)
        b = r["best"]
        print(f"  {L:>2} {r['n_words']:>6} {r['n_S_A_above_2']:>6} {r['n_of_these_without_sigma3']:>6} "
              f"{r['n_of_these_natural_C_zero']:>7}   {b['S_A']:.6f}  {b['word']:<13} {b['natural_C']:.4f}    {b['n_ties']}")
        key = f"L={L}"
        if key in stored:
            ok = (abs(r["S_A_max"] - stored[key]["S4_max"]) < 1e-9
                  and abs(r["bell_pct_4D"] - stored[key]["bell_pct_4D"]) < 1e-9)
            control.append(dict(L=L, S4_max_stored=stored[key]["S4_max"], S_A_max=r["S_A_max"],
                                bell_pct_4D_stored=stored[key]["bell_pct_4D"], bell_pct_4D=r["bell_pct_4D"], ok=ok))
    ctrl_ok = all(c["ok"] for c in control)
    print(f"\n  positive control vs landscape_validation.json (S4_max, bell_pct_4D, L = 3..9): {ctrl_ok}")

    # delta = 0 first maximizers (two-pass rule) for the s_delta_sweeps.json / Fig. 2 lengths
    print(f"\n  delta = 0 first maximizer under Protocol A, two-pass rule (band {FIRST_MAX_BAND:g}):")
    print(f"  {'L':>2}  word            S_A        ties  natural C   (agrees with L<=9 table)")
    first_max = {}
    table_best = {r['L']: r['best']['word'] for r in rows}
    for L in FIRST_MAX_LENGTHS:
        r = first_maximizer_delta0(L)
        first_max[str(L)] = dict(word=r["word"], S_A=r["S_A"], n_ties=r["n_ties"], natural_C=r["natural_C"])
        same = (table_best.get(L) == r["word"]) if L in table_best else None
        print(f"  {L:>2}  {r['word']:<14}  {r['S_A']:.6f}  {r['n_ties']:>4}  {r['natural_C']:.4f}      {same}")

    out = dict(meta=dict(script="d10_protocol_a_projection_audit.py",
                         projection_A="|11>_L <- (|2>+|4>)/sqrt(2), renormalized (landscape_validation.project_4d)",
                         natural_assignment="|11>_L <- |4>, |2> discarded, renormalized",
                         violation_tolerance=TOL, natural_C_zero_tolerance=C_ZERO,
                         word_order="lexicographic in application order; best = first maximum", seed=None,
                         first_maximizer_rule="two-pass at delta = 0: S* = max over all 3^L words; "
                                              "winner = first word in lexicographic order with S >= S* - 1e-10; "
                                              "n_ties = number of such words",
                         first_maximizer_lengths=list(FIRST_MAX_LENGTHS)),
               per_length=rows, positive_control=dict(all_ok=ctrl_ok, rows=control),
               delta0_first_maximizer=first_max)
    (HERE / "d10_protocol_a_projection_audit_results.json").write_text(json.dumps(out, indent=2))
    print(f"  runtime {time.time() - t0:.1f} s (printed, not stored)")
    print(f"  WROTE {HERE / 'd10_protocol_a_projection_audit_results.json'}")


if __name__ == "__main__":
    main()
