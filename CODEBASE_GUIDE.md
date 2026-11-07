# Codebase Guide

The code is organized by research stage. Generated market-data and pilot outputs are local-only and ignored by Git.

| Path | Purpose |
|---|---|
| `01_data_collection/build_spx_chain.py` | Fetch SPX and SPXW EOD option quotes, tag/cache responses per date/root, and retry failed multi-session requests as single-day requests. |
| `01_data_collection/build_spx_iv.py` | Add root/expiry-level forward/rate and bid, midpoint, and ask Black-76 IV fields. |
| `01_data_collection/build_30d_smiles.py` | Interpolate total variance at fixed log-forward moneyness to 30 calendar days and write per-date expiry-bracketing diagnostics. |
| `01_data_collection/fit_iv_curve.py` | Fit an SVI total-variance curve to each constructed smile. |
| `02_rnd_extraction/breeden_litzenberger.py` | Project call prices onto discrete no-arbitrage constraints, extract RNDs, and write diagnostics. |
| `03_wasserstein_kernel/wasserstein_kernel.py` | Convert RNDs to quantiles and compute W2 distances, Gaussian kernels, and stable normalized weights. |
| `04_gp_model/fit_gp.py` | Fit the Wasserstein-kernel GP and produce a quantile-barycenter distribution forecast. |
| `05_baselines/` | Linear moment, persistence, Wasserstein autoregression, and moment-based RBF GP baselines. |
| `06_evaluation/model_comparison.py` | Compare all distribution forecasts on common held-out dates. |
| `06_evaluation/trading_calendar_pairs.py` | Generate exact h-session pairs from the XNYS calendar, then retain only pairs with both RNDs. |
| `06_evaluation/run_horizon_evaluation.py` | Evaluate h=1 and h=5 distribution forecasts on shared chronological splits and write metrics/diagnostics. |
| `06_evaluation/run_inference_robustness.py` | Compute paired Newey-West and moving-block-bootstrap inference, plus expanding-window and volatility-regime robustness checks. |
| `06_evaluation/quantile_change_models.py` | Normalize and project forward-scaled quantile functions, build fold-local features, and provide PCA, kernel-ridge, and forward-CV helpers. |
| `06_evaluation/run_phase4_quantile_change.py` | Run quantile-change models on shared h=1/h=5 pairs; use rolling-origin primary, fixed 80/20 secondary, blocked forward-chaining CV, April-2025 training sensitivity, and common projection. |
| `06_evaluation/test_quantile_change_models.py` | Test projection validity, exact persistence at zero change, fold-local PCA, and shared model evaluation pairs. |
| `run_all.py` | Run the separate synthetic-data example; it does not fetch SPX data. |

The standard SPX path sends the complete merged chain directly from `build_spx_chain.py` to `build_spx_iv.py`; `make_20day_spx.py` remains a pilot-only date selector. The ThetaData EOD `created` timestamp is preserved through the chain and IV stages; it identifies the EOD snapshot, not per-contract quote freshness. For expiries present under both roots, the 30-day smile uses SPX; SPXW supplies expiries unique to that root. The ThetaData client and its permissions are separate from the VolForge code. Use only an authorized account, keep credentials local, and do not publish raw or derived market data without explicit permission.

The horizon evaluator uses the NYSE/XNYS sessions before checking which RND dates are present. A missing RND removes pairs that use it as an endpoint; dates are never bridged. Both horizons use the same chronological 80/20 split across all included models. The Gaussian-shape moment baseline is labeled as a reference; the target-moments Gaussian is reported separately as a diagnostic-only oracle, not as a forecast model. Inference outputs include paired Newey-West and moving-block-bootstrap results and a refit sensitivity excluding martingale-miss dates. Local outputs include `06_evaluation/results/trading_calendar_pairs.csv`, `horizon_forecast_predictions.csv`, `horizon_forecast_summary.csv`, `forecast_inference_summary.csv`, `martingale_miss_diagnostics.csv`, `martingale_exclusion_sensitivity.csv`, `thetadata_history_probe.csv`, `failed_smile_quote_quality.csv`, and `pipeline_failure_report.txt`. The report aggregates collection, IV, smile, SVI, RND, optimizer, and quantile diagnostics.

The Phase 4 quantile-change evaluator is separate from the legacy horizon evaluator. It models target-minus-input quantile coefficients, projects every forecast to a nonnegative nondecreasing function with mean one, and selects GP kernel/noise settings from projected squared-W2 loss in forward-chaining folds inside each training window. Rolling-origin folds are primary; the fixed 80/20 holdout is secondary. Its local-only files are prefixed `phase4_` and include predictions, per-origin loss differentials, summary scores, the CV loss curve, selected hyperparameters, full-sweep sensitivity results, inference, and volatility regimes. It also reports full training alongside sensitivity excluding all April 2025 pairs from training; the full parameter sweep is sensitivity-only, never a selected test result.
