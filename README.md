# CHSH-Form Values Above 2 in Fibonacci Anyon Braiding: A Complete Landscape of Braid Words up to Length 12

**Author:** Berkay Yüksel Sayim
**ORCID:** [0009-0004-4993-7352](https://orcid.org/0009-0004-4993-7352)

## Abstract

We report a complete sequence-by-sequence landscape of CHSH-form values for
six Fibonacci anyons in two encodings, covering all words of length
*L* = 3–12 with two generators and *L* = 2–9 with three. In the two-generator
d1b encoding, braiding alone cannot exceed the classical value: the two
generators act on different anyons and commute, and all 8190 braid words of
length *L* = 1–12 give |S| = 2 exactly on the leakage-free fusion code. A
two-qubit circuit built from the same Fibonacci *F*- and *R*-matrices, which
does not satisfy the braid relations, reaches |S| = 2.733 at *L* = 12 with the
standard Fibonacci data (96.6% of the Tsirelson bound 2√2) and |S| = 2.811 at
the optimum over a phase deformation; this excess is therefore not a braiding
effect. Throughout, "CHSH violation" denotes a CHSH value exceeding 2 on the
Fibonacci fusion space, which does not factorize into spatially separated
Alice/Bob subsystems; the reported values are signatures of nonseparability
and single-algebra contextuality, not Bell violations in the
Einstein–Podolsky–Rosen sense. In the three-generator encoding, the generator
σ₃ that acts across the bipartition is the only source of entanglement:
without it, the state stays in the fusion code as a product state, for all
sequences and deformation phases. Under four-dimensional
computational-subspace projection the maximum is |S| = 2.824 at *L* = 8, and
91.8% of the words exceed 2 at *L* = 9, including 510 words without σ₃ that do
so only through the projection; a Horodecki-type score on the native
five-dimensional observables exceeds 2 for 42.8%. Sequential braiding on a
shared fusion space produces a positive lag-1 autocorrelation of the simulated
CHSH value, ACF(1) = +0.35 at 11.22σ, because the state is carried from step
to step; a null test with independently drawn phases shows none.

## Contents

This record is compiled from `main_v1.6.tex` (RevTeX 4-2). The compiled
`main_v1.6.pdf` is included.

## Data and Code Availability

The deterministic scripts and JSON output files that validate the results
reported in this paper are archived on Zenodo under the concept DOI
[10.5281/zenodo.19601352](https://doi.org/10.5281/zenodo.19601352)
(which always resolves to the latest version).

Run order: `d10` before `d4` (d4 checks its five-dimensional words against
the d10 result); `d4 --regenerate`, then `d4`; `d5 --regenerate`, then `d5`;
`d6` before `d8` (d8 reads the vacuum-start values from
`d6_acf_prediction_results.json`); `d11` and the figure scripts read the JSON
files written before them.

- `landscape_validation.py`/`.json` — three-generator (5D) landscape under
  both measurement protocols (Tables II and III).
- `n2_reanchor.py`/`.json` — leakage-norm reanchor.
- `d10_protocol_a_projection_audit.py`/`_results.json` — Protocol A
  projection audit (words that exceed 2 only through the projection) and the
  first δ = 0 maximizers of the three-generator encoding for
  *L* = 3, 6, 8, 10, 12 over all 3^*L* words.
- `d4_s_delta_sweep_validation.py`/`_results.json` — closed-form
  sector-phase sweep validation covering all rows of Table I and the
  five-dimensional sweeps of the d10 maximizers; reproduces
  `s_delta_sweeps.json`.
- `d5_topology_switch_validation.py`/`_results.json` — σ₃-blocking
  validation, including the exact σ₃-block C = 0 check; writes
  `topology_switch_results.json`.
- `d6_acf_prediction_sequential_braiding.py` + `d6_acf_prediction_results.json`
  — the sequential-braiding autocorrelation (Table IV, Models A/B/C).
- `d7_acf_hensen2015_null_control.py`/`_results.json` + `hensen_data/` — independent
  null-control reanalysis of the Hensen et al. (2015) loophole-free Bell
  data, including the underlying public dataset.
- `d8_acf_null_robustness.py`/`_results.json` — seeded Pillar-2 null test
  (N = 10,000 i.i.d. sector phases, Model-A and d1b configurations) and
  initialization-robustness evaluation (random 5D initial state, Models A/B).
- `d9_fusion_code_braiding_only.py`/`_results.json` — braiding alone on the
  leakage-free fusion code (all 8190 words of length 1–12 give |S| = 2
  exactly), and the Yang–Baxter residua and distant commutator on the
  50-point δ grid and at the δ values of Table V.
- `d11_fourier_fit_sweeps.py`/`_results.json` — Fourier fits of the
  five-dimensional sweeps (appendix on the Fourier structure).
- `figures/` — the four figures of the paper with the scripts that draw them
  from the JSON files: `make_1b_figs.py` writes `fig2_sector_phase.png`
  (Fig. 1), `fig3_sigma3_switch.png` (Fig. 3) and `fig1_landscape.png`
  (Fig. 4); `make_1b_fig4_map.py` writes `fig4_sector_map.png` (Fig. 2);
  `make_1b_figs_dense.py` draws denser variants into `figures/dense_out/`,
  which are not used in the paper.

Notes on the JSON files:

- `landscape_validation.json`: the field `conclusion` ("PROBLEMATIC") is a
  label written by the script. It compares the violating fractions of the
  two measurement protocols and reads PROBLEMATIC when they differ by more
  than 15 percentage points; that difference is the protocol dependence
  reported in Table III (largest at *L* = 5, +56.0 points). It does not flag
  an error.
- `topology_switch_results.json` is now written by `d5 --regenerate`. Its
  top-level keys are renamed: `aufgabe1_3gen_sigma3_block` →
  `three_generator_sigma3_block`, `aufgabe2_2gen_AM_MB_block` →
  `two_generator_AM_MB_block`. The fields `S_opt` and `S_horodecki` of the
  earlier file are no longer present; `S` is the Horodecki value.

The three-generator δ = 0 maximizers come from `d10`. The two-generator
sequences of Table I come from the full enumeration deposited with the
companion record (concept DOI
[10.5281/zenodo.19600752](https://doi.org/10.5281/zenodo.19600752)). The
checks of the remaining generator pairs in the appendix on the braid algebra
and the Z₅ phase spectrum were made with two further scripts
(`d2_delta_phase_consistency.py`, `d3_5d_braid_algebra_consistency.py`) that
are not part of this record.

## License

- Paper, figures, and data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see `LICENSE`
- Source code (`*.py`): [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) — see `LICENSE-CODE`
