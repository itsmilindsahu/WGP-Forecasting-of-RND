# Codebase Guide

The code is organized by research stage. Generated market-data and pilot outputs are local-only and ignored by Git.

| Path | Purpose |
|---|---|
| `01_data_collection/build_spx_chain.py` | Fetch SPX EOD option quotes and write a cleaned chain plus collection diagnostics. |
| `01_data_collection/build_spx_iv.py` | Add expiry-level forward/rate and bid, midpoint, and ask Black-76 IV fields. |
| `01_data_collection/build_30d_smiles.py` | Interpolate total variance at fixed log-forward moneyness to a 30-calendar-day tenor. |
| `01_data_collection/fit_iv_curve.py` | Fit an SVI total-variance curve to each constructed smile. |
| `02_rnd_extraction/breeden_litzenberger.py` | Project call prices onto discrete no-arbitrage constraints, extract RNDs, and write diagnostics. |
| `03_wasserstein_kernel/wasserstein_kernel.py` | Convert RNDs to quantiles and compute W2 distances, Gaussian kernels, and stable normalized weights. |
| `04_gp_model/fit_gp.py` | Fit the Wasserstein-kernel GP and produce a quantile-barycenter distribution forecast. |
| `05_baselines/` | Linear moment, persistence, Wasserstein autoregression, and moment-based RBF GP baselines. |
| `06_evaluation/model_comparison.py` | Compare all distribution forecasts on common held-out dates. |
| `run_all.py` | Run the separate synthetic-data example; it does not fetch SPX data. |

The ThetaData client and its permissions are separate from the VolForge code. Use only an authorized account, keep credentials local, and do not publish raw or derived market data without explicit permission.
