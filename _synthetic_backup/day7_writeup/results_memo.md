# Results memo: Wasserstein-GP forecasting of risk-neutral densities

Milind Sahu, BS-MS Mathematics, IISER Tirupati — Supervisor: Dr. Sven Karbach, University of Amsterdam
Run: synthetic SVI-smile data (`day1_data_collection/simulate_data.py`), 40 trading days, 8 held out.

**Revision note**: this memo replaces the previous version after Sven's review identified
a calibration bug, a missing benchmark, and two RND-extraction correctness issues. All four
are fixed in this run (see "What changed" below); real Deribit data (Sven's next step) is
still outstanding — see Next steps.

## What was built (Days 1–6)

1. **Data + smoothing**: option-chain snapshots (real Deribit/yfinance clients, plus an
   offline synthetic generator used for this run) → SVI smile fit per day. The synthetic
   generator now targets an ATM annualized vol of ~18% (equity-index level, matching its
   own docstring) and widens the log-moneyness strike grid to ±4σ dynamically each day
   (was a fixed ±0.35, i.e. ±1.8σ at the old vol level).
2. **RND extraction**: Breeden–Litzenberger on the smoothed call-price curve, using the
   corrected Black-76 convention (`C = e^{-rτ}[F·N(d1) − K·N(d2)]`; the previous version
   mixed forward- and spot-measure terms). Extraction correctness is now checked via a
   martingale test (extracted mean vs. forward) rather than the previous "integrates to
   1.0" check, which the code's own renormalization step guaranteed would always pass.
3. **Wasserstein kernel**: closed-form 2-Wasserstein distance via quantile functions,
   `k(F,F') = exp(-gamma * W2(F,F')^2)`. Validated against the known closed form for
   Gaussians and checked positive semi-definite on synthetic batches.
4. **Wasserstein-GP**: distribution-input GP forecasting `Var(RND_t) → Var(RND_{t+1})`,
   hyperparameters fit by maximizing the log marginal likelihood via Cholesky. Predictive
   variance now correctly adds the observation-noise term `sigma_n^2` (see below).
5. **Linear baseline**: OLS on `(mean_t, var_t, skew_t) → Var(RND_{t+1})`, homoscedastic
   residual-based predictive std.
6. **Persistence benchmark (new)**: `y_hat_{t+1} = y_t` for the scalar target, and
   `Q_hat_{t+1} = Q_t` (today's quantile function used verbatim) for the full density.
   The random-walk null every daily forecasting claim has to clear first.
7. **Evaluation**: all three models scored on the same 8 held-out days: scalar-target
   accuracy/calibration, full-density Wasserstein forecast error, plus two diagnostics
   (a no-forecast Gaussian floor, and a shape-only W2 with each side's mean removed).

## What changed, and why

1. **GP calibration bug (fixed).** `predict()` returned `Var[f(x*)|data]` — the posterior
   variance of the *latent* function — but coverage/NLPD score against *realized*
   observations, which carry an extra `sigma_n^2` noise term. Adding it back is a one-line
   fix (`var = f_var + sigma_n2`).
2. **Persistence benchmark (added).** Neither model meant anything without it.
3. **RND extraction support problem (fixed).** The old ±0.35 log-moneyness grid was ±1.8σ
   at the simulator's original (crypto-level) vol, truncating ~7% of tail mass; the
   renormalization step then pushed the extracted mean off the forward, violating the
   martingale condition. Fixed by (a) calibrating the simulator to an actual equity-index
   vol level and (b) computing the strike grid as ±4σ dynamically each day.
4. **Black-Scholes convention bug (fixed).** `bs_call_price` mixed forward- and spot-measure
   terms (small at 30-day tenor, grows with maturity); now a clean Black-76 form.
5. **Real Deribit data — not yet done.** This sandbox's network egress doesn't reach
   `deribit.com`, so `fetch_deribit.py` couldn't be exercised here; it's unchanged and
   ready to run on a machine with network access. This is the highest-value remaining
   step (see Next steps).

## Results on this run

| Metric | Wasserstein-GP | Linear baseline | Persistence |
|---|---:|---:|---:|
| RMSE (variance target) | 1.632 | 1.151 | **0.884** |
| 95% coverage | 75.0% | 100.0% | 100.0% |
| NLPD (mean, lower better) | 2.083 | 1.571 | **1.327** |
| **W2 forecast error (full density, mean)** | **0.864** | 1.086 | 0.957 |

Diagnostics:

| | value |
|---|---:|
| Gaussian fit to *today* (zero forecast skill), W2 vs. tomorrow | 1.083 |
| Shape-only W2 (mean removed) — GP | 0.141 |
| Shape-only W2 (mean removed) — baseline | 0.420 |
| Shape-only W2 (mean removed) — persistence | 0.102 |

Martingale check on RND extraction: 40/40 days pass at 1% tolerance, max relative error
0.12% (down from ~2.98% before the strike-grid/vol fix).

## Reading these results honestly

- **On the scalar target**, persistence wins on every metric, and the GP is calibrated
  but the widest of the three (75% empirical coverage against a 95% nominal target, on
  only 8 test days — a single flipped day moves this by 12.5 points, so it's noisy but
  not obviously broken the way 25% was). The GP and linear baseline are both worse than
  simply predicting "tomorrow's variance = today's variance." On this synthetic data
  that's expected: `simulate_data.py` drives its SVI parameters with a driftless random
  walk, so day-to-day changes are unpredictable by construction and the only learnable
  signal is level persistence — which is exactly what persistence exploits directly and
  what a barycenter-style forecaster approximates indirectly. The synthetic data cannot
  distinguish "the GP is capturing something persistence can't" from "the GP is doing
  slightly worse than the thing it's implicitly approximating." That question needs real
  data with genuine (non-random-walk) structure.
- **On the full-density forecast**, the GP now beats *both* the linear baseline and
  persistence on W2 (0.864 vs. 1.086 vs. 0.957). That's a real result, but two caveats
  before it's over-read: (a) the no-forecast Gaussian floor is 1.083 — nearly identical
  to the baseline's 1.086 — so almost the entire baseline-vs-GP gap is the cost of
  forcing a Gaussian shape, not a forecasting difference; and (b) the GP beats
  persistence by a much smaller margin (0.864 vs. 0.957) than it beats the baseline, and
  the shape-only W2 (mean removed) actually favors persistence over the GP (0.102 vs.
  0.141). So the GP's edge on the full metric is coming substantially from the mean/level
  part of the forecast, not from shape — which is the part of the Wasserstein kernel this
  project is actually meant to test. Removing the mean-shift component and re-scoring
  shape directly, on real data, is the next honest step (Sven's step 5).

Net read: the calibration bug is fixed and the GP's intervals are no longer badly
miscalibrated; the full-density result survives the persistence check but the shape-only
diagnostic shows the win isn't yet clearly a shape-awareness win. That's a narrower, more
defensible claim than the previous memo made, and it points directly at what real data
needs to settle.

## Next steps (in order, per Sven's review)

1. ~~Add `sigma_n^2` to the predictive variance.~~ Done.
2. ~~Add persistence as the primary benchmark.~~ Done.
3. ~~Fix the RND extraction~~ (strike grid, Black-Scholes convention, martingale check). Done.
4. **Pull real Deribit data.** BTC/ETH history is free and deep; aim for several hundred
   days in one maturity bucket rather than 40. Requires running `fetch_deribit.py` on a
   machine with network access to `deribit.com` — the highest-value remaining step, and
   the only way to resolve the ambiguity above (random-walk synthetic data can't).
5. **Re-ask the shape question in a metric that can see shape**, on real data: report W2
   after removing the mean shift (implemented here as a diagnostic — `w2_rowwise_shape_only`
   in `day6_evaluation/evaluate.py`) or score skew/tail quantiles directly. This is where
   the project's actual contribution would live.
6. **Longer horizon / event-conditional forecasting** (e.g. RND shift around FOMC/CPI
   prints), once 1–5 are settled on real data.
7. **Index extension**: SPX/SPY via CBOE DataShop free samples, once BTC/ETH results
   look promising.

## Files

- `day2_rnd_extraction/data/martingale_diagnostics.csv` — per-day martingale check
  (extracted mean vs. forward).
- `day5_baseline/results/persistence_predictions.csv` — persistence benchmark predictions.
- `day6_evaluation/results/comparison_summary.csv`, `comparison_per_day.csv`,
  `comparison_plot.png` — the numbers and chart behind this memo (now three-way,
  GP/baseline/persistence).
- `day7_writeup/docs/index.html` — interactive version of the same comparison
  (Chart.js, single file, ready for GitHub Pages).
