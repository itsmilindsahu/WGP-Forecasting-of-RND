# SPX 20-Day Wasserstein-RND Pilot

## Dataset

- Underlying: SPX
- Pilot window: 2026-09-02 to 2026-09-30
- Chain dates used in the 20-day pilot: 20
- Dates with usable 30D smiles: 16
- Clean option rows: 35,466
- 30D smile points: 1,296
- RND points: 1,264

The raw collection stage covered 25 available trading dates in the wider collection window. The final pilot uses the latest 20 available trading dates.

## Option-chain schema

The cleaned chain contains:

- valuation date
- expiry
- strike
- option type
- bid
- ask
- midpoint
- volume
- time to maturity
- tau in years

Forward, interest rate and implied-volatility fields are derived in the subsequent processing stage.

Raw market-data responses are kept separately from the cleaned research dataset.

## 30D construction

For each valuation date, expiries bracketing 30 calendar days were interpolated in total variance at fixed log-forward moneyness.

Days without sufficient overlapping expiry/moneyness coverage were excluded rather than extrapolated.

### Missing 30D dates

Four of the 20 pilot dates did not produce a 30D smile:

- 2026-09-14: insufficient overlapping expiry/moneyness coverage.
- 2026-09-15: insufficient overlapping expiry/moneyness coverage.
- 2026-09-17: insufficient overlapping expiry/moneyness coverage.
- 2026-09-18: insufficient overlapping expiry/moneyness coverage.

Therefore, 16 of the 20 pilot dates produced usable 30D smiles.

## SVI fitting

An SVI curve was fitted independently to each available 30D smile.

- SVI dates: 16
- Curve points: 1,296
- Parameter sets: 16

The fitted curves were then used to construct smooth call-price curves for RND extraction.

## RND extraction

Risk-neutral densities were extracted using the Breeden-Litzenberger second derivative of the smoothed call-price curve.

The resulting densities were normalized and checked for:

- total probability mass
- martingale consistency
- non-negativity
- call-price convexity

## RND diagnostics

- Mass approximately 1: 16/16
- Martingale pass: 16/16
- Non-negative density: 16/16
- Call convexity pass: 15/16
- Maximum martingale relative error: 0.24%

### Tiny numerical convexity violation

The only failed call-convexity diagnostic occurred on:

- Date: 2026-09-11
- Minimum raw density: -7.193283e-08
- Negative grid fraction: 6.33%
- Martingale relative error: 0.18%

The negative curvature is at numerical scale. The density is clipped to zero before normalization, and the day is retained rather than silently removed.

## Wasserstein representation

RNDs are represented in forward-normalized coordinates:

`S_T / F_t`

This makes the Wasserstein distance primarily sensitive to distributional shape rather than the absolute SPX level.

The Gaussian Wasserstein kernel uses a median-heuristic initialization with stabilized kernel-weight normalization and an explicit nearest-neighbour fallback.

## Forecast evaluation

The chronological evaluation uses 12 training days and a common 3-day held-out test set.

All three distributional forecasting approaches are evaluated using the same Wasserstein-2 error:

1. Persistence baseline
2. Linear moment baseline
3. Wasserstein GP

### Model comparison

| Model | Mean W2 | Median W2 |
|---|---:|---:|
| Persistence | 0.001062 | 0.000741 |
| Wasserstein GP | 0.001680 | 0.001687 |
| Linear moment baseline | 0.007257 | 0.007133 |

Relative to persistence on the common held-out dates:

- Wasserstein GP mean W2 difference: +0.000617
- Wasserstein GP relative change: +58.1%
- Linear baseline mean W2 difference: +0.006195
- Linear baseline relative change: +583.1%

The scalar GP was evaluated separately for the predicted RND variance:

- RMSE: 0.000138451
- Empirical 95% coverage: 100%

### Interpretation

This is a small pilot with only three common held-out forecast dates. The results should therefore be treated as pipeline-validation results rather than evidence of general predictive superiority.

On this pilot test set, persistence provides the lowest observed W2 error. The Wasserstein GP does not outperform persistence in this experiment, while the simple linear moment baseline has a substantially larger W2 error.

A larger historical sample and a longer held-out evaluation period are required before drawing conclusions about predictive performance.

## Reproducibility and data handling

The project separates:

- raw ThetaData responses
- cleaned option-chain data
- derived IV and forward data
- constant-maturity smiles
- SVI curves
- RNDs
- Wasserstein forecasts
- baseline comparisons
- diagnostic and visualization outputs

ThetaData API credentials and raw market data should not be committed to the public repository.

## Next research stage

The pilot validates the end-to-end workflow:

ThetaData option chain
-> cleaned chain
-> forward and implied volatility
-> 30D total-variance interpolation
-> SVI smoothing
-> Breeden-Litzenberger RND
-> forward normalization
-> Wasserstein distance/kernel
-> GP forecasting
-> baseline comparison.

The next stage should expand the historical sample, increase the number of held-out forecast dates, and compare the Wasserstein approach against additional baselines such as persistence, Wasserstein autoregression, and conventional GP models using moments or SVI parameters.