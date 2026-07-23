# Generic CHSH Violation in Fibonacci Anyon Braiding: A Landscape Analysis

**Author:** Berkay Yüksel Sayim
**ORCID:** [0009-0004-4993-7352](https://orcid.org/0009-0004-4993-7352)

## Abstract

We report the first, to our knowledge, complete sequence-by-sequence numerical
landscape of CHSH-inequality violation across all Fibonacci anyon braiding
words of length *L* = 3–12 in two complementary encodings of six anyons. In
the two-generator *d*₁ᵦ encoding (native four-dimensional fusion space), the
optimal sequence at *L* = 12 reaches |S| = 2.811 (99.4% of the Tsirelson bound
2√2) with concurrence *C* = 0.988. In the three-generator encoding (full
five-dimensional fusion space), Tsirelson saturation |S| = 2√2 occurs already
at *L* = 8.

CHSH violation is generic: a majority of braiding sequences violate the CHSH
inequality under both measurement protocols considered. A single topological
mechanism controls the landscape — blocking the cross-bipartition generator σ₃
collapses concurrence to *C* = 0 exactly for all sequences and all sector
phases, so that topology acts as a gatekeeper for entanglement. Sequential
braiding on a shared fusion space produces positive lag-1 autocorrelation
ACF(1) = +0.34 at 10.85σ significance.

Throughout, "CHSH violation" denotes a CHSH value exceeding 2 on the Fibonacci
fusion space, which does not factorize into spatially separated Alice/Bob
subsystems; the reported values are signatures of topological nonseparability,
not Bell violations in the Einstein–Podolsky–Rosen sense.

## Contents

This record is compiled from `main_v1.5.tex` (RevTeX 4-2). The compiled
`main_v1.5.pdf` is included.

## Data and Code Availability

The deterministic scripts and JSON output files that validate the results
reported in this paper are archived on Zenodo under the concept DOI
[10.5281/zenodo.19601352](https://doi.org/10.5281/zenodo.19601352)
(which always resolves to the latest version) and released under CC BY 4.0.

- `landscape_validation.py`/`.json`, `n2_reanchor.py`/`.json` — three-generator
  (5D) landscape validation and leakage-norm reanchor.
- `d4_s_delta_sweep_validation.py` — closed-form sector-phase sweep validation
  covering all rows of Table I and the sequences of the three-generator table,
  reproducing `s_delta_sweeps.json` and the printed table values, plus a
  positive control re-evaluating the `best_seq_5D` winners stored in
  `landscape_validation.json`.
- `d5_topology_switch_validation.py` — topological-gatekeeper validation,
  including the exact sigma_3-block C=0 check, reproducing
  `topology_switch_results.json`.
- `d6_acf_prediction_sequential_braiding.py` — the sequential-braiding
  autocorrelation prediction (Table tab:acf, Models A/B/C).
- `d7_acf_hensen2015_null_control.py` + `hensen_data/` — independent
  null-control reanalysis of the Hensen et al. (2015) loophole-free Bell data,
  including the underlying public dataset.
- `d8_acf_null_robustness.py`/`.json` — seeded Pillar-2 null test
  (N = 10,000 i.i.d. sector phases, Model-A and d1b configurations) and
  initialization-robustness evaluation (random 5D initial state, Models A/B).
- `figures/` — the three publication figures (`fig1_landscape.png`,
  `fig2_sector_phase.png`, `fig3_sigma3_switch.png`) together with the
  deterministic plotting scripts that generate them from the deposited
  JSON files (`make_1b_figs.py`, `make_1b_figs_dense.py`).

The exhaustive sequence search that originally identified the winning braid
words reported in Table I and the three-generator table, and the verification
scripts for the sector-phase Yang-Baxter check of Appendix C
(`d2_delta_phase_consistency.py`, `d3_5d_braid_algebra_consistency.py`), are
not part of this record.

## License

This work is released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license.
