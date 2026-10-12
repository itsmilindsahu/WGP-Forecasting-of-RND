# Wasserstein Forecasting of Risk-Neutral Densities

This repository contains research code for extracting option-implied risk-neutral distributions (RNDs) and forecasting their evolution with a Gaussian process using a Wasserstein-distance kernel.

## Research Pipeline

```text
SPX option quotes
  -> forward and rate estimation
  -> Black-76 implied volatility
  -> 30-calendar-day total-variance interpolation
  -> SVI smile fitting
  -> arbitrage-constrained call-price smoothing
  -> Breeden-Litzenberger density extraction
  -> forward-normalized quantile representation
  -> Wasserstein-kernel GP and benchmark forecasts
  -> held-out Wasserstein-2 evaluation
```

The distributional distance is computed from quantile functions:

$$
W_2(F,G) = \left(\int_0^1 |F^{-1}(u)-G^{-1}(u)|^2\,du\right)^{1/2}.
$$

For the kernel $k(F,G)=\exp(-\gamma W_2(F,G)^2)$, $\gamma$ is an inverse squared length-scale parameter. A small value gives a broad, nearly constant kernel; a large value suppresses off-diagonal similarities. The implementation bounds its estimate around a median-distance heuristic and normalizes forecast weights, with a nearest-neighbour fallback for numerical collapse.

## Models

- Wasserstein-kernel Gaussian process with a quantile-barycenter full-distribution forecast
- Persistence forecast
- Linear baseline using distribution moments
- Wasserstein autoregression on quantile coordinates
- Conventional RBF Gaussian process using distribution moments

The distribution-valued models are compared with a common chronological holdout using full-density W2 error. Results should be interpreted in light of the small pilot sample and short holdout.

## Data and Reproducibility

Market-data responses, generated chains, IV/RND datasets, and pilot-specific result artifacts are kept local and excluded from Git. Reproduction requires an appropriately authorized data subscription, a locally configured API key, the VolForge dependency, and compliance with the data provider's terms. Do not commit credentials, raw market data, or derived outputs without permission.

The SPX stages are run individually using the scripts under `01_data_collection/`, followed by `02_rnd_extraction/breeden_litzenberger.py`, `04_gp_model/fit_gp.py`, `05_baselines/`, and `06_evaluation/`. `run_all.py` is the separate synthetic-data demonstration; it does not fetch or process the SPX ThetaData pilot.

Install the general dependencies with:

```powershell
pip install -r requirements.txt
```

Configure the ThetaData/VolForge client locally before running the collection stage. Keep API credentials in environment variables or a local `.env` file, never in the repository.

## Repository Map

- `01_data_collection/`: synthetic data, option-chain adapters, 30-day smile construction, and SVI fitting
- `02_rnd_extraction/`: call-price projection, Breeden-Litzenberger extraction, and diagnostics
- `03_wasserstein_kernel/`: quantile conversion, W2 distances, kernels, and numerical checks
- `04_gp_model/`: Wasserstein-kernel GP and full-distribution forecasting
- `05_baselines/`: moment, persistence, Wasserstein autoregression, and moment-GP forecasts
- `06_evaluation/`: shared held-out comparison and evaluation utilities
- `07_writeup/`: report-generation source; data-backed outputs are local-only

See [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) for key modules and execution order.
