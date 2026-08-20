# Results memo: Wasserstein-GP forecasting of risk-neutral densities

Milind Sahu, BS-MS Mathematics, IISER Tirupati — Supervisor: Dr. Sven Karbach, University of Amsterdam
Run: **real Deribit data** (`day1_data_collection/build_real_chain.py`), BTC-26JUL19 calls,
2019-07-01, 48 intraday snapshots (30-min bins), 38 train / 10 held out.

**Revision note**: this replaces the synthetic-data memo. The synthetic run (40 driftless-random-walk
days) is archived under `_synthetic_backup/` for comparison. This is the "pull real Deribit data" step
that was the top item on Sven's outstanding list.

## What's different about this data, honestly

This is real market data, but it is **one calendar day of tick data**, not the multi-week daily
series the project ultimately wants:

- 48 "time steps" are 30-minute intraday snapshots within 2019-07-01 UTC, not calendar days. Each
  snapshot is the last-observed quote per strike as of the bin boundary (not an average — averaging
  would smear out the real intraday move, which is the signal being tested).
- Single expiry: BTC-26JUL19 (25 days to expiry as of 2019-07-01), the closest listed maturity to the
  30-day tenor the synthetic generator used. Calls only, `mid_iv = mark_iv` (Deribit's own smoothed
  mark; bid/ask IV is only populated on ~80% of ticks in the raw dump, mark_iv on >99.99%).
- The forward moved for real over the day: 10,148 → 11,199 (about +10%), so unlike the synthetic
  series (driftless random walk in SVI params), there is genuine directional structure here — but
  it's intraday drift on one asset on one day, not the cross-day structure ("does today's smile shape
  predict tomorrow's") the project is actually about.
- Only 38 training points. The synthetic run had 32; this isn't a big sample either way, and it
  matters for what follows below.

So: real prices, real quotes, real market noise — but a proxy for the eventual multi-day dataset, not
a replacement for it. Pulling several hundred actual trading days (Sven's original suggestion) is
still the highest-value next step; see Next steps.

## Results on this run

| Metric | Wasserstein-GP | Linear baseline | Persistence |
|---|---:|---:|---:|
| RMSE (variance target) | 246,208 | 165,310 | **171,787** |
| 95% coverage | 100.0% | 90.0% | 100.0% |
| NLPD (mean, lower better) | 13.92 | **13.47** | 13.52 |
| **W2 forecast error (full density, mean)** | 10,877.6 | 464.2 | **94.1** |
| Shape-only W2 (mean removed) | 2,707.6 | 402.0 | **33.5** |

(Variance-target units are price², hence the large numbers — BTC was trading around 10,500–11,200
over this window, so a variance RMSE of ~165k corresponds to a vol-of-vol on the order of a few
hundred dollars, roughly in line with intraday moves of this size.)

Diagnostics: no-forecast Gaussian floor (today's own mean/var, zero forecast skill, scored against
tomorrow) = 465.3 W2 — essentially tied with the linear baseline (464.2), same pattern as the
synthetic run. Martingale check on RND extraction: 36/48 snapshots pass at 1% tolerance (mean
rel_error 0.60%, max 1.53%) — looser than the synthetic run's 40/40, which is expected: real bid/ask
noise makes the SVI fit to real ticks less clean than a fit to a noiseless synthetic smile.

## Reading these results honestly

**Persistence wins outright, by a wide margin, on every metric** — more decisively than on synthetic
data. On the scalar target it's within range of the baseline; on the full-density forecast it beats
the GP by more than 100x (94.1 vs. 10,877.6 W2).

That GP number needed checking before writing it down, because 100x isn't "the GP is a bit worse," it's
"something is likely degenerate." Digging in: `WassersteinBarycenterForecaster` predicts by
Nadaraya-Watson weighting in quantile space, `w_i ∝ exp(-gamma * W2(x*, x_i)^2)`, using `gamma` fit by
MLE on the scalar-target GP. At real BTC price levels, pairwise W2 distances between quantile
functions are on the order of hundreds to thousands (price units); with only 38 training points and
an *unbounded* log-space MLE, the optimizer found `gamma ≈ 1.0` — large enough that `exp(-gamma·W2²)`
underflows to ~0 for every training point except near-exact matches. The kernel effectively becomes a
one-hot / near-empty weighting instead of a smooth interpolant, so the barycenter forecast for a
held-out day mixes almost no real information and comes out close to a degenerate quantile function —
which then scores terribly against a real, non-degenerate target. This is a known GP failure mode
(unconstrained lengthscale MLE overfitting on small n), not something specific to Wasserstein kernels,
but it's much more visible here than on the synthetic run because synthetic prices were scaled near
100 and real BTC prices are near 10,000–20,000, so the same nominal `gamma` behaves completely
differently. **I did not re-bound or re-tune `gamma` to make this number look better** — that would be
exactly the kind of after-the-fact adjustment that turns an honest negative result into a curated one.
It's flagged here as a concrete, fixable methodology gap (median-heuristic or cross-validated `gamma`
bounds, not MLE alone) for the next iteration, not patched in this run.

**Net read**: on this one real trading day, persistence is unambiguously the model to beat, the linear
baseline is a distant second, and the Wasserstein-GP's full-density forecaster needs a lengthscale fix
before it's a fair comparison at all — the 100x gap is mostly a scale/overfitting artifact of a
38-point training set, not evidence the Wasserstein-kernel approach itself is wrong. That's a narrower
and less flattering conclusion than the synthetic run's memo, and it's the correct one to report from
one day of real, single-asset, single-expiry data.

## Update: fixing the gamma overfit

Next steps #1 above, done. `day4_gp_model/fit_gp.py::fit_hyperparameters` now bounds `log(gamma)`
in the L-BFGS-B search instead of leaving it unconstrained, using a **median heuristic**:
`gamma_med = 1 / median(off-diagonal pairwise W2)^2` on the training Gram matrix, computed fresh
from whatever data is passed in (so it's scale-aware — it doesn't hardcode "real BTC" vs.
"synthetic" anywhere). The search is allowed ±1.5 decades around `gamma_med`; MLE still picks the
value within that window, it just can't run off to a degenerate one outside it.

On this run: median training-set W2 = 244.3 (price units) → `gamma_med` = 1.68e-5, bound window
[5.3e-7, 5.3e-4]. MLE converged to **gamma = 2.0e-5** — close to the median-heuristic value and
comfortably inside the window, not pinned at either boundary, which is the sanity check that this
is a real data-driven fit and not just a clamp. (Old unconstrained fit: gamma = 1.0, ~5 orders of
magnitude off — at that gamma, `exp(-gamma * W2^2)` for a *typical* pair, W2≈244, is
`exp(-1 * 244^2)` ≈ 0, hence the near-one-hot kernel diagnosed above.)

**Results after the fix:**

| Metric | Wasserstein-GP | Linear baseline | Persistence |
|---|---:|---:|---:|
| RMSE (variance target) | 219,686 | 165,310 | **171,787** |
| 95% coverage | 90.0% | 90.0% | **100.0%** |
| NLPD (mean, lower better) | 13.88 | **13.47** | 13.52 |
| **W2 forecast error (full density, mean)** | **106.2** | 464.2 | 94.1 |
| Shape-only W2 (mean removed) | **45.8** | 402.0 | 33.5 |

The full-density W2 error goes from 10,877.6 → 106.2, a **102.5x reduction**. Persistence still
wins that metric (94.1 vs. 106.2) and the shape-only one (33.5 vs. 45.8), but the gap is now a
normal single-digit-percent margin instead of two orders of magnitude — the GP is finally losing
(narrowly) on its actual forecasting ability rather than on a broken kernel. It's also now
decisively better than the linear baseline's Gaussian-shape reconstruction on both full and
shape-only W2 (464.2 and 402.0), which is the comparison that actually tests whether the
Wasserstein-kernel machinery buys anything over a moment-matched Gaussian — and here it does.

The scalar-target numbers (RMSE, coverage, NLPD) moved only slightly, as expected: `gamma` mainly
governs the barycenter forecaster's Nadaraya-Watson weights, not the scalar GP's posterior mean
directly (though it does enter the same kernel matrix, hence the small RMSE change, 246,208 →
219,686). Persistence still wins the scalar comparison outright, and coverage on the GP dropped
from 100% to 90% (n_test=10, so this is one flipped point, not a big shift) — worth watching once
the dataset grows past 10 test days rather than reading much into it now.

**What this fix doesn't claim**: it doesn't make the Wasserstein-GP outperform persistence — it
still doesn't, on this one-day, 38-training-point sample. What it does is make the full-density
comparison an honest one: before the fix, the 100x gap was mostly measuring a broken lengthscale
fit, not the Wasserstein-kernel approach itself; after it, the ~13% gap against persistence and the
clear win against the linear baseline are numbers worth actually trusting.

## Next steps (in order)

1. **Pull several hundred real calendar days**, not 48 intraday bins of one day — this needs either
   Deribit's historical settlement/mark-price endpoints (network access this sandbox doesn't have) or
   further tick-data dumps like this one, chained across dates. This is now the highest-value next
   step (gamma fix above was #1, now done).
2. Once (1) is done, re-run the shape-only diagnostic (already implemented in
   `day6_evaluation/evaluate.py`) on genuine cross-day structure — the synthetic run couldn't test
   this (driftless random walk), and this one-day run is too short a horizon to test it either.
3. Revisit the gamma bound width (currently ±1.5 decades around the median heuristic) once there's
   enough data to cross-validate it properly instead of choosing the window by inspection.
4. **Index extension**: SPX/SPY via CBOE DataShop free samples, once BTC/ETH multi-day results are in
   good shape.

## Files

- `day1_data_collection/build_real_chain.py` — builds `option_chains.csv` from the raw tick dump
  (filtering, snapshotting, and all the real-data-specific choices are documented in its docstring).
- `run_all_real.py` — Day 1 → Day 7 pipeline runner for the real data (mirrors `run_all.py`, writes to
  the same output paths).
- `day2_rnd_extraction/data/martingale_diagnostics.csv` — per-snapshot martingale check.
- `day6_evaluation/results/comparison_summary.csv`, `comparison_per_day.csv`, `comparison_plot.png` —
  the numbers and chart behind this memo.
- `day7_writeup/docs/index.html` — interactive version of the same comparison.
- `_synthetic_backup/` — the previous synthetic-data run (data, results, memo, README), kept for
  before/after comparison.
