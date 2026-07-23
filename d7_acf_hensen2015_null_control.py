#!/usr/bin/env python3
"""
D7 -- Null control: ACF(1) on the Hensen et al. (2015) loophole-free
Bell data (Sec. VI.C, "Applicability to existing experimental data")
=====================================================================
Data: Hensen et al., "Loophole-free Bell inequality violation using
electron spins separated by 1.3 kilometres", Nature 526, 682 (2015),
doi:10.1038/nature.15759. Openly deposited at 4TU.ResearchData
(bell_open_data.txt, included in hensen_data/ alongside this script).

This script:
  STEP 1  Ports the authors' own filter and CHSH definition (from the
          openly deposited bell_open_data_analysis_example.py) to
          reproduce S, k/n, p-value exactly. If this fails, the outcome
          extraction below is wrong (ground-truth lock).
  STEP 2  Extracts the chronologically ordered sequence of the 245
          valid Bell trials and forms the per-trial CHSH-outcome
          observable.
  STEP 3  ACF(1) with the 95% CI band and a 1000-permutation shuffle
          significance test; plus ACF(k>1), a drift/stationarity check,
          a runs test, and a within-run robustness variant.

HONEST FRAMING: this dataset consists of re-prepared, independently
heralded Bell trials (a fresh NV-NV entangled state per trial), so the
paper's sequential-braiding mechanism (Sec. VI, shared fusion-space
memory across consecutive braiding steps) is structurally absent here.
This is therefore a null-control / apparatus-systematics measurement on
an unrelated physical platform, not a confirmation or falsification of
the paper's own prediction (that requires a platform that physically
implements braiding sequentially on a shared fusion space; see Sec.
VI.C). The result reported in the paper (ACF(1) = -0.047, consistent
with zero) is reproduced below.

Reproducible: deterministic data load; fixed seed for the permutation
test.
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
import numpy as np
from scipy.stats import binom, norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).parent
DATA = HERE / "hensen_data" / "bell_open_data.txt"
RNG = np.random.default_rng(20260624)

# ----------------------------------------------------------------------
# STEP 1 -- load + reproduce the authors' CHSH (ground-truth lock)
# ----------------------------------------------------------------------
ts = np.loadtxt(DATA, delimiter=',', skiprows=0, usecols=[0], dtype=np.datetime64)
data = np.loadtxt(DATA, delimiter=',', skiprows=0, usecols=np.arange(1, 17), dtype=np.int64)

day_number = data[:, 0]
run_number = data[:, 1]
er_c1_time, er_c1_ch = data[:, 2], data[:, 3]
er_c2_time, er_c2_ch = data[:, 4], data[:, 5]
random_number_A = data[:, 6]
random_number_B = data[:, 7]
readout_click_A_time = data[:, 10]
readout_click_B_time = data[:, 11]
click_after_excite_A_time = data[:, 12]
click_after_excite_B_time = data[:, 13]
last_invalid_marker_A = data[:, 14]
last_invalid_marker_B = data[:, 15]

# event-ready window parameters, from the authors' own analysis script
w0_ch0, w0_ch1 = 5426350, 5425700
w_len = 55000 - 2550
w_sep = 250000
ro_start, ro_len = 10620, 3700
invalid_lookback = 250

f_w1c0 = (w0_ch0 <= er_c1_time) & (er_c1_time < w0_ch0 + w_len) & (er_c1_ch == 0)
f_w1c1 = (w0_ch1 <= er_c1_time) & (er_c1_time < w0_ch1 + w_len) & (er_c1_ch == 1)
f_w1 = f_w1c0 | f_w1c1
f_w2c0 = (w0_ch0 + w_sep <= er_c2_time) & (er_c2_time < w0_ch0 + w_sep + w_len) & (er_c2_ch == 0)
f_w2c1 = (w0_ch1 + w_sep <= er_c2_time) & (er_c2_time < w0_ch1 + w_sep + w_len) & (er_c2_ch == 1)
f_w2 = f_w2c0 | f_w2c1
f_psimin = (er_c1_ch != er_c2_ch)
f_erwin = f_w1 & f_w2 & f_psimin
f_noinvA = (last_invalid_marker_A == 0) | (last_invalid_marker_A > invalid_lookback)
f_noinvB = (last_invalid_marker_B == 0) | (last_invalid_marker_B > invalid_lookback)
f_noexc = (click_after_excite_A_time == 0) & (click_after_excite_B_time == 0)
bell_trial_filter = f_erwin & f_noinvA & f_noinvB & f_noexc

det_A = (readout_click_A_time > ro_start) & (readout_click_A_time <= ro_start + ro_len)
det_B = (readout_click_B_time > ro_start) & (readout_click_B_time <= ro_start + ro_len)

a_i = random_number_A
b_i = random_number_B
x_i = det_A.astype(np.int64) * 2 - 1
y_i = det_B.astype(np.int64) * 2 - 1
t_i = bell_trial_filter.astype(np.int64)

n = int(t_i.sum())
c_i = t_i * (((-1) ** (a_i * b_i)) * (x_i * y_i) + 1) // 2
k = int(c_i.sum())
tau = 5.4e-6 * 2
ksi = 3.0 / 4 + 3 * (tau + tau ** 2)
p_value = float(1 - binom.cdf(k - 1, n, ksi))

inputs_ab = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs_xy = [[+1, +1], [+1, -1], [-1, +1], [-1, -1]]
cm = np.zeros((4, 4), dtype=np.int64)
E = np.zeros(4); Eerr = np.zeros(4)
for ii, (a, b) in enumerate(inputs_ab):
    for jj, (x, y) in enumerate(outputs_xy):
        cm[ii, jj] = np.sum(t_i & (a_i == a) & (b_i == b) & (x_i == x) & (y_i == y))
    tot = cm[ii].sum()
    E[ii] = (cm[ii, 0] - cm[ii, 1] - cm[ii, 2] + cm[ii, 3]) / float(tot)
    Eerr[ii] = np.sqrt((1 - E[ii] ** 2) / float(tot))
S = float(E[0] + E[1] + E[2] - E[3])
S_err = float(np.sqrt((Eerr ** 2).sum()))

reproduce = dict(n=n, k=k, p_value=round(p_value, 4), S=round(S, 3), S_err=round(S_err, 3),
                 E=[round(float(e), 3) for e in E])
lock_repro = dict(
    n_is_245=(n == 245), k_is_196=(k == 196),
    S_is_2p42=(abs(S - 2.42) < 0.02), p_is_0p039=(abs(p_value - 0.039) < 0.002))

# ----------------------------------------------------------------------
# STEP 2 -- ordered valid-trial sequences
# ----------------------------------------------------------------------
idx = np.where(t_i == 1)[0]
order_ok = bool(np.all(np.diff(ts[idx].astype('datetime64[us]').astype(np.int64)) > 0))
g = (((-1) ** (a_i[idx] * b_i[idx])) * (x_i[idx] * y_i[idx])).astype(float)
w = ((g + 1) / 2)
prod = (x_i[idx] * y_i[idx]).astype(float)
day_seq, run_seq = day_number[idx], run_number[idx]


def acf_k(seriesz, kk):
    sx = np.asarray(seriesz, float); m = sx.mean(); d = sx - m
    den = float(np.dot(d, d))
    if den == 0: return float('nan')
    return float(np.dot(d[:-kk], d[kk:]) / den)


def shuffle_sig(seriesz, n_perm=1000, rng=RNG):
    obs = acf_k(seriesz, 1)
    if not np.isfinite(obs): return obs, float('nan'), float('nan'), float('nan')
    xx = np.array(seriesz, float); perms = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(xx); perms[i] = acf_k(xx, 1)
    mu, sd = perms.mean(), perms.std(ddof=1)
    sigma = (obs - mu) / sd if sd > 0 else float('nan')
    p_two = float((np.abs(perms - mu) >= abs(obs - mu)).mean())
    return obs, sigma, sd, p_two


band = 1.96 / np.sqrt(n)
observables = {"g_signed_chsh_term": g, "w_win_loss": w, "prod_raw_xy": prod}
acf_results = {}
for name, sER in observables.items():
    obs, sigma, sd, p_two = shuffle_sig(sER, 1000, RNG)
    acf_results[name] = dict(
        acf1=round(obs, 4), shuffle_sigma=round(sigma, 3) if np.isfinite(sigma) else None,
        shuffle_sd=round(sd, 4), shuffle_p_two=round(p_two, 4),
        ci95_band=round(band, 4), within_band=bool(abs(obs) < band),
        mean=round(float(np.mean(sER)), 4))

acf_spectrum = {f"lag_{kk}": round(acf_k(g, kk), 4) for kk in range(1, 16)}

# ----------------------------------------------------------------------
# STEP 3 -- drift / stationarity / runs
# ----------------------------------------------------------------------
half = n // 2
win_rate_1 = float(w[:half].mean()); win_rate_2 = float(w[half:].mean())
k1, n1 = int(w[:half].sum()), half
k2, n2 = int(w[half:].sum()), n - half
pbar = (k1 + k2) / (n1 + n2)
se = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
z_drift = float((win_rate_1 - win_rate_2) / se) if se > 0 else float('nan')
p_drift = float(2 * (1 - norm.cdf(abs(z_drift))))

wl = w.astype(int)
n_pos, n_neg = int(wl.sum()), int((1 - wl).sum())
runs = 1 + int(np.sum(wl[1:] != wl[:-1]))
mu_runs = 2 * n_pos * n_neg / n + 1
var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n ** 2 * (n - 1))
z_runs = float((runs - mu_runs) / np.sqrt(var_runs)) if var_runs > 0 else float('nan')
p_runs = float(2 * (1 - norm.cdf(abs(z_runs))))

same = (day_seq[1:] == day_seq[:-1]) & (run_seq[1:] == run_seq[:-1])
gd = g - g.mean()
num_wr = float(np.sum(gd[:-1][same] * gd[1:][same]))
den_wr = float(np.dot(gd, gd))
acf1_within_run = num_wr / den_wr if den_wr > 0 else float('nan')
n_within_pairs = int(same.sum())

drift = dict(
    win_rate_first_half=round(win_rate_1, 4), win_rate_second_half=round(win_rate_2, 4),
    z_drift=round(z_drift, 3), p_drift=round(p_drift, 4),
    runs_observed=runs, runs_expected_iid=round(mu_runs, 2), z_runs=round(z_runs, 3), p_runs=round(p_runs, 4),
    acf1_within_run_only=round(acf1_within_run, 4) if np.isfinite(acf1_within_run) else None,
    n_within_run_consecutive_pairs=n_within_pairs,
    n_cross_boundary_pairs=int((~same).sum()))

gaps_s = np.diff(ts[idx].astype('datetime64[s]').astype(np.int64))
gap_stats = dict(
    median_gap_s=float(np.median(gaps_s)), mean_gap_s=float(np.mean(gaps_s)),
    max_gap_s=float(np.max(gaps_s)), min_gap_s=float(np.min(gaps_s)),
    n_days=int(len(np.unique(day_seq))), n_runs=int(len(np.unique(list(zip(day_seq.tolist(), run_seq.tolist()))))))

# ----------------------------------------------------------------------
# Assemble + checks
# ----------------------------------------------------------------------
out = dict(
    dataset="Hensen et al. 2015, Nature 526, 682 (open data at 4TU.ResearchData)",
    reproduce_published=reproduce, lock_reproduce=lock_repro,
    chronological_order_verified=order_ok,
    acf_primary=acf_results, acf_spectrum_signed_term=acf_spectrum,
    drift_stationarity=drift, gap_structure=gap_stats,
    band_2sigma=round(band, 4),
    interpretation=(
        "ACF computed on the true chronological order of the 245 valid Bell trials. "
        "This dataset consists of re-prepared, independently heralded trials, so the "
        "paper's sequential-braiding mechanism (shared fusion-space memory across "
        "consecutive steps) is structurally absent here; this is a null-control / "
        "apparatus-systematics measurement, not a test of the paper's own prediction. "
        "Consecutive valid trials are seconds-to-minutes apart and span day/run "
        "boundaries (see gap_structure)."))

checks = {}
checks["reproduces_published_S_k_n_p"] = bool(all(lock_repro.values()))
checks["order_chronological"] = order_ok
checks["g_and_w_same_acf"] = bool(abs(acf_results["g_signed_chsh_term"]["acf1"]
                                     - acf_results["w_win_loss"]["acf1"]) < 1e-6)
checks["matches_paper_reported_acf1_minus_0p047"] = bool(
    abs(acf_results["g_signed_chsh_term"]["acf1"] - (-0.047)) < 0.001)
out["CHECKS"] = checks
out["ALL_PASS"] = bool(all(checks.values()))

(HERE / "d7_acf_hensen2015_null_control_results.json").write_text(json.dumps(out, indent=2))

print("=" * 72)
print("  D7 -- Null control: ACF on Hensen 2015 loophole-free Bell data")
print("=" * 72)
print("\n[STEP 1] Reproduce authors' CHSH (ground-truth lock):")
print(f"   n={n} (pub 245)  k={k} (pub 196)  S={S:.3f}+-{S_err:.3f} (pub 2.42+-0.20)  p={p_value:.3f} (pub 0.039)")
print(f"   LOCKS: {lock_repro}")
print(f"   chronological order of 245 valid trials verified strictly increasing: {order_ok}")
print(f"\n[STEP 3] ACF(1) on 245-trial chronological sequence (2sigma band = +-{band:.4f}):")
for name, r in acf_results.items():
    print(f"   {name:>20}: ACF(1)={r['acf1']:+.4f}  shuffle={r['shuffle_sigma']}sigma "
          f"p={r['shuffle_p_two']}  within_band={r['within_band']}")
print(f"\n   Drift (1st vs 2nd half win-rate): {drift['win_rate_first_half']} vs "
      f"{drift['win_rate_second_half']}  z={drift['z_drift']} p={drift['p_drift']}")
print(f"   Runs test: observed={drift['runs_observed']} expected_iid={drift['runs_expected_iid']} "
      f"z={drift['z_runs']} p={drift['p_runs']}")
print("\n" + "=" * 72)
for kk, vv in checks.items():
    print(f"   [{'PASS' if vv else 'FAIL'}] {kk}")
print(f"\n  ALL_PASS = {out['ALL_PASS']}")
print(f"  WROTE {HERE/'d7_acf_hensen2015_null_control_results.json'}")
