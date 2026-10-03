# Wasserstein-GP Forecasting of Risk-Neutral Densities

**Milind Sahu — BS-MS Mathematics, IISER Tirupati**
Working with **Dr. Sven Karbach (Amsterdam)** on this research project.

---

## Overview

This project investigates whether **risk-neutral probability distributions (RNDs)** extracted from option markets can be forecast directly in **distribution space** using a Gaussian Process with a **Wasserstein-distance kernel**.

Instead of representing an implied distribution only through a few moments such as mean, variance, or skewness, the approach treats the entire RND as the object being forecast.

The current research pipeline is:

```text
SPX Option Chains
       ↓
Data Cleaning
       ↓
Forward Price + Risk-Free Rate
       ↓
Black-76 Implied Volatility
       ↓
30-Day Constant-Maturity IV Smile
       ↓
SVI Smoothing
       ↓
Breeden–Litzenberger
       ↓
Risk-Neutral Density
       ↓
Forward Normalization
       ↓
Wasserstein-2 Distance
       ↓
Wasserstein Kernel
       ↓
Gaussian Process Forecast
       ↓
Persistence + Moment-Based Baselines
       ↓
Out-of-Sample W₂ Evaluation
```

The repository has evolved from an initial synthetic-data prototype into a real-data **SPX 20-trading-day pilot**.

---

# Research Objective

The central question is:

> **Can the evolution of option-implied risk-neutral distributions be forecast directly using a Wasserstein-kernel Gaussian Process?**

The project therefore focuses on three related problems:

1. Extracting reliable RNDs from option-market data.
2. Defining a meaningful distance/kernel between RNDs.
3. Forecasting future RNDs and evaluating the forecast in distribution space.

The main distributional metric is the **2-Wasserstein distance**:

$$
W_2(F,G)
=
\left(
\int_0^1
\left|
F^{-1}(u)-G^{-1}(u)
\right|^2du
\right)^{1/2}.
$$

---

# Project Development

The project has gone through several stages.

## Stage 1 — Synthetic Prototype

The initial implementation was built using synthetic option smiles.

The first complete pipeline implemented:

1. Synthetic option-chain generation.
2. IV smile construction.
3. SVI smoothing.
4. Breeden–Litzenberger RND extraction.
5. Wasserstein-2 distance.
6. Wasserstein kernel.
7. Gaussian Process regression.
8. Moment-based linear baseline.
9. Persistence baseline.
10. Forecast evaluation.
11. Interactive results visualization.

This stage was used to validate the overall mathematical and computational architecture before moving to real market data.

---

# Stage 2 — Sven's Review and Bug Fixes

The initial prototype exposed several implementation issues.

### 1. GP uncertainty calibration

The GP initially used latent-function posterior variance when calculating predictive coverage/NLPD for observed targets.

The observation-noise term

$$
\sigma_n^2
$$

was missing from the predictive variance used for evaluation.

This was corrected.

---

### 2. Persistence benchmark

A persistence benchmark was added:

$$
\hat F_{t+1}=F_t.
$$

This is essential because a learned model should be compared against the simple assumption that tomorrow's distribution is approximately today's distribution.

---

### 3. RND extraction support

The original synthetic strike grid was too narrow relative to the distribution tails.

The extraction grid was widened and the simulator was recalibrated to a more realistic equity-index volatility level.

This improved the martingale consistency of the extracted RNDs.

---

### 4. Black-76 convention

The original option-pricing implementation mixed forward- and spot-measure conventions.

This was replaced by a consistent **Black-76** formulation.

---

# Stage 3 — Real Deribit Experiment

The next stage moved from synthetic data to a real historical Deribit option dataset.

The earlier experiment used a historical tick-level export containing BTC and ETH option data. It was filtered to a BTC option contract and converted into intraday snapshots.

This experiment was primarily a **methodological and debugging stage**.

It revealed a major numerical issue in the Wasserstein kernel.

---

# Stage 4 — Gamma / Kernel Stabilization

The Wasserstein kernel is

$$
k(F,G)
=
\exp\{-\gamma W_2(F,G)^2\}.
$$

If \(\gamma\) becomes too large, almost every off-diagonal kernel value approaches zero:

$$
k(F,G)\approx0.
$$

The resulting kernel weights become nearly one-hot, causing the full-density forecast to collapse toward a single training distribution.

This happened during the earlier real-data experiment.

The solution was to stabilize \(\gamma\) using a **median-distance heuristic**:

$$
\gamma_{\mathrm{med}}
=
\frac{1}
{\operatorname{median}(W_2^2)}.
$$

The optimization of \(\gamma\) was then bounded around this scale instead of allowing unconstrained maximum-likelihood optimization to move to a degenerate region.

This reduced the earlier full-density W₂ error by more than two orders of magnitude in the Deribit experiment.

The lesson from this stage directly informed the current SPX implementation.

---

# Stage 5 — Current SPX / ThetaData Pilot

The main research workflow was subsequently restructured around **S&P 500 Index (SPX) options**.

The current implementation uses real option-chain data and a multi-day historical workflow rather than a single-day intraday experiment.

The pilot targets approximately:

* **7–60 DTE** option contracts
* SPX calls and puts
* bid / ask / midpoint prices
* strike
* expiry
* volume
* valuation date
* timestamps where available
* time to maturity

The raw market-data response is kept separate from the cleaned research schema.

Raw market data and generated datasets are intentionally excluded from GitHub.

---

# Current SPX Pilot Dataset

The current pilot uses the latest **20 available SPX trading-day observations** in the selected historical window.

### Dataset summary

| Quantity                       |      Value |
| ------------------------------ | ---------: |
| Trading-day chain observations |     **20** |
| Clean option rows              | **35,466** |
| Call rows                      | **17,733** |
| Put rows                       | **17,733** |
| Usable 30-day smiles           |     **16** |
| 30-day smile points            |  **1,296** |
| RND grid points                |  **1,264** |
| Training observations          |     **12** |
| Common held-out observations   |      **3** |

The pilot covers valuation dates from:

**2026-09-02 → 2026-09-30**

subject to available market data.

---

# 1. SPX Option-Chain Collection

### File

```text
01_data_collection/build_spx_chain.py
```

The collector converts the raw option response into a standardized research schema.

### Research schema

```text
date
expiry
expiry_date
strike
right
option_type
bid
ask
mid
volume
created
last_trade
timestamp
dte
tau_years
```

Where timestamps are unavailable in the source response, they remain missing rather than being artificially generated.

The pipeline also stores raw responses separately from the cleaned dataset.

---

# 2. Forward Price and Implied Volatility

### File

```text
01_data_collection/build_spx_iv.py
```

For each valuation date, the pipeline derives:

* risk-free rate
* implied forward
* Black-76 implied volatility
* log-forward moneyness
* total variance

The implied forward is obtained using put-call parity.

The resulting representation is suitable for constructing a constant-maturity implied-volatility smile.

---

# 3. 30-Day Constant-Maturity Smile

### File

```text
01_data_collection/build_30d_smiles.py
```

The target maturity is approximately **30 calendar days**.

For each valuation date, expiries bracketing the 30-day target are used.

The interpolation is performed in **total variance**, not directly in volatility:

$$
w(k,\tau)
=
\sigma_{IV}^2(k,\tau)
$$

where

$$
k=\log(K/F_t).
$$

The result is a common 30-day smile expressed on a fixed log-forward-moneyness grid.

---

## Missing 30-Day Dates

Four dates did not have sufficient expiry/moneyness overlap to construct a reliable 30-day smile:

```text
2026-09-14
2026-09-15
2026-09-17
2026-09-18
```

These dates are explicitly recorded in:

```text
07_writeup/spx_pilot_results/missing_30d_days.csv
```

rather than being silently interpolated or filled.

---

# 4. SVI Smile Fitting

### File

```text
01_data_collection/fit_iv_curve.py
```

The 30-day IV smiles are fitted using an SVI parameterization.

Current pilot:

* **16 fitted curves**
* **1,296 smile points**
* **16 parameter sets**

SVI provides a smooth representation of the smile before numerical differentiation is used for RND extraction.

---

# 5. Risk-Neutral Density Extraction

### File

```text
02_rnd_extraction/breeden_litzenberger.py
```

The RND is extracted using the Breeden–Litzenberger relationship:

$$
q(K)
=
\frac{\partial^2 C(K)}
{\partial K^2}.
$$

The smoothed option-price representation is differentiated numerically to obtain the density.

The resulting densities are then checked for:

* total mass
* non-negativity
* martingale consistency
* call-price convexity

---

# RND Diagnostics

The current pilot gives:

| Diagnostic                        |      Result |
| --------------------------------- | ----------: |
| Probability mass                  | **16 / 16** |
| Martingale check                  | **16 / 16** |
| Non-negative density              | **16 / 16** |
| Call-price convexity              | **15 / 16** |
| Maximum martingale relative error |   **0.24%** |

One very small numerical convexity violation occurred on:

```text
2026-09-11
```

with minimum raw density approximately:

$$
-7.19\times10^{-8}.
$$

This was treated as a numerical discretization effect and clipped in the final density representation.

The detailed diagnostic table is available at:

```text
07_writeup/spx_pilot_results/rnd_diagnostic_summary.csv
```

---

# 6. Forward-Normalized RND Representation

The extracted RNDs are represented using **forward-normalized coordinates**.

For example:

$$
X=\frac{S_T}{F_t}.
$$

This prevents the Wasserstein metric from being dominated simply by changes in the absolute SPX level.

The objective is for \(W_2\) to primarily capture changes in the **shape of the risk-neutral distribution**.

---

# 7. Wasserstein Distance

For two RNDs \(F\) and \(G\):

$$
W_2(F,G)
=
\left[
\int_0^1
\left(
F^{-1}(u)-G^{-1}(u)
\right)^2du
\right]^{1/2}.
$$

Because the RNDs are one-dimensional, the Wasserstein distance can be computed directly from their quantile representations.

---

# 8. Wasserstein Kernel

### File

```text
03_wasserstein_kernel/wasserstein_kernel.py
```

The kernel is:

$$
k(F,G)
=
\exp
\left[
-\gamma W_2(F,G)^2
\right].
$$

The implementation includes:

* Wasserstein-2 computation
* median-heuristic \(\gamma\)
* Gaussian Wasserstein kernel
* Gram matrix construction
* PSD diagnostics
* stable kernel-weight normalization
* nearest-neighbour fallback

The final weighting procedure explicitly normalizes weights so that:

$$
\sum_i w_i=1.
$$

If numerical underflow causes the kernel weights to become unusable, the implementation falls back to the nearest historical distribution rather than producing a near-zero or invalid forecast.

---

# Wasserstein Kernel Diagnostics

Current pilot:

| Quantity                |           Value |
| ----------------------- | --------------: |
| RND days                |          **16** |
| Minimum W₂              |    **0.000552** |
| Maximum W₂              |    **0.006247** |
| Median-heuristic γ      |  **177,223.82** |
| Kernel diagonal         |           **1** |
| Minimum Gram eigenvalue | **1.02 × 10⁻³** |

These diagnostics indicate that the current kernel construction is numerically well behaved on the pilot sample.

---

# 9. Wasserstein Gaussian Process

### File

```text
04_gp_model/fit_gp.py
```

The GP uses the Wasserstein kernel to define similarity between historical RNDs.

The implementation includes a full-density forecasting procedure based on a Wasserstein barycenter / kernel-weighted distribution forecast.

The model is evaluated directly in distribution space.

---

# 10. Baselines

Two baselines are used.

## Linear moment baseline

```text
05_baselines/linear_baseline.py
```

Uses distributional moments to construct a conventional forecast.

This provides a comparison against a model that compresses the distribution into a small number of summary statistics.

---

## Persistence baseline

```text
05_baselines/persistence_baseline.py
```

The persistence forecast is:

$$
\hat F_{t+1}=F_t.
$$

This is an important benchmark because RNDs may exhibit substantial short-term persistence.

---

# Current Forecast Evaluation

The current pilot uses a chronological split:

```text
Training: 12 observations
Common held-out observations: 3
```

The primary distributional metric is the **full-density Wasserstein-2 error**.

---

# Current SPX Results

## Full-density W₂ comparison

| Model                  |      Mean W₂ |    Median W₂ |
| ---------------------- | -----------: | -----------: |
| **Persistence**        | **0.001062** | **0.000741** |
| Wasserstein GP         |     0.001680 |     0.001687 |
| Linear moment baseline |     0.007257 |     0.007133 |

### Relative to persistence

The Wasserstein GP has a mean W₂ error approximately:

$$
58.1\%
$$

higher than persistence on this small common holdout.

The linear moment baseline has a mean W₂ error approximately:

$$
583.1\%
$$

higher than persistence.

These relative differences are descriptive statistics for the pilot and should not be interpreted as final generalization claims.

---

# Per-Day Forecast Results

The common held-out dates are:

| Date       |      Wasserstein GP W₂ |         Persistence W₂ |
| ---------- | ---------------------: | ---------------------: |
| 2026-09-28 | approximately 0.001687 | approximately 0.000741 |
| 2026-09-29 | approximately 0.001532 |   approximately 0.000? |
| 2026-09-30 | approximately 0.001821 |   approximately 0.001? |

For the authoritative per-day values, see:

```text
06_evaluation/results/model_comparison.csv
```

The repository intentionally keeps generated evaluation CSVs out of Git because they are reproducible outputs rather than source code.

---

# Scalar GP Diagnostic

A separate scalar-output GP evaluation produced:

| Metric       |          Result |
| ------------ | --------------: |
| RMSE         | **0.000138451** |
| 95% coverage |        **100%** |

This is a separate scalar-target diagnostic and should not be confused with the full-density Wasserstein forecast metric.

---

# Interpretation of Current Results

The current SPX pilot is primarily an **end-to-end methodological validation**.

The observed results show:

1. The option-chain data can be transformed into a common 30-day IV representation.
2. SVI provides a smooth representation for subsequent density extraction.
3. The extracted RNDs pass the main mass and martingale diagnostics.
4. Forward normalization provides a common coordinate system for Wasserstein comparison.
5. The Wasserstein kernel is numerically stable under the current median-heuristic treatment.
6. A full-density Wasserstein GP forecast can be generated and compared directly against persistence and moment-based baselines.

However:

> **Persistence has the lowest observed W₂ error in this pilot.**

The Wasserstein GP does not outperform persistence on the three common held-out dates.

The linear moment baseline has substantially larger observed W₂ error.

Because the common holdout contains only **three dates**, these results are preliminary and should not be interpreted as a final conclusion about the forecasting ability of the Wasserstein GP.

The next experiment therefore needs a substantially larger historical sample and longer out-of-sample period.

---

# Repository Structure

```text
WGP-Forecasting-of-RND/
│
├── README.md
├── requirements.txt
├── CODEBASE_GUIDE.md
├── run_all.py
├── run_all_real.py
│
├── 01_data_collection/
│   ├── build_spx_chain.py
│   ├── make_20day_spx.py
│   ├── build_spx_iv.py
│   ├── build_30d_smiles.py
│   ├── fit_iv_curve.py
│   ├── build_real_chain.py
│   ├── fetch_deribit.py
│   ├── fetch_yfinance.py
│   ├── simulate_data.py
│   └── data/
│       └── # generated/raw data — gitignored
│
├── 02_rnd_extraction/
│   ├── breeden_litzenberger.py
│   └── data/
│       └── # generated RND data — gitignored
│
├── 03_wasserstein_kernel/
│   ├── wasserstein_kernel.py
│   └── test_synthetic.py
│
├── 04_gp_model/
│   └── fit_gp.py
│
├── 05_baselines/
│   ├── linear_baseline.py
│   └── persistence_baseline.py
│
├── 06_evaluation/
│   ├── evaluate.py
│   └── model_comparison.py
│
└── 07_writeup/
    ├── build_demo.py
    ├── finalize_spx_pilot.py
    ├── results_memo.md
    ├── docs/
    │   └── index.html
    │
    └── spx_pilot_results/
        ├── SPX_PILOT_REPORT.md
        ├── missing_30d_days.csv
        ├── rnd_diagnostic_summary.csv
        └── plots/
```

---

# Generated Data Policy

The repository deliberately does **not** commit:

* raw ThetaData responses
* raw option-chain Parquet files
* generated option-chain CSVs
* generated IV datasets
* generated RND datasets
* model result directories
* API credentials
* archived synthetic datasets

These files are excluded through `.gitignore`.

This keeps the public repository focused on the reproducible research code and documentation without exposing credentials or raw licensed market data.

---

# Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

The real-data workflow requires an appropriately authorized market-data account and local API configuration.

The API key should remain in a local environment file and must never be committed to GitHub.

The project can then be run through the individual research stages or the available orchestration scripts.

---

# Research Outputs

The current pilot produces the following outputs.

### IV / Smile

```text
30-day IV smiles
SVI fitted curves
```

### RND

```text
Daily RND curves
Mass diagnostics
Martingale diagnostics
Non-negativity diagnostics
Call-convexity diagnostics
```

### Wasserstein

```text
Wasserstein distance matrix
Kernel matrix
Kernel PSD diagnostics
```

### Forecasting

```text
Wasserstein GP forecasts
Persistence forecasts
Linear moment forecasts
W₂ forecast errors
```

### Visualization

The pilot includes plots for:

* option coverage
* 30-day IV smiles
* SVI fits
* extracted RNDs
* RND diagnostics
* all-day RND comparison
* forecast W₂ comparison

The detailed report is:

```text
07_writeup/spx_pilot_results/SPX_PILOT_REPORT.md
```

The interactive page is:

```text
07_writeup/docs/index.html
```

---

# Data and Licensing

The current market-data workflow uses **ThetaData**.

The repository does not contain raw ThetaData responses or generated market-data datasets.

Anyone reproducing the data-collection stage should obtain their own authorized ThetaData access and comply with the applicable ThetaData terms.

The VolForge software used in the workflow and the underlying market data are separate licensing matters.

Any publication or distribution of derived market-data products should be checked against the applicable data-provider permissions before release.

---

# What Changed from the Earlier Version

The most important change is that this repository is no longer centered on the earlier single-day Deribit experiment.

### Earlier version
