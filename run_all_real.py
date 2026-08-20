"""
Runs Day 1 -> Day 7 end-to-end using REAL Deribit data instead of
simulate_data.py's synthetic generator.

Data source: a tick-level Deribit options-chain export for 2019-07-01
(BTC-26JUL19 calls, ~545K raw ticks binned into 30-min intraday snapshots
by day1_data_collection/build_real_chain.py -> 48 snapshots x 26 strikes).
See that script's docstring for why intraday snapshots, why this expiry,
and why mark_iv.

Writes to the exact same output paths run_all.py does, so
day7_writeup/build_demo.py and the README numbers are the real-data run
after this executes. The synthetic-data outputs this overwrites are backed
up under _synthetic_backup/ before this script is run.

Usage:
    python run_all_real.py --tick-csv-gz /mnt/user-data/uploads/deribit_options_chain_2019-07-01_OPTIONS_csv.gz
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "day1_data_collection"))
sys.path.append(os.path.join(ROOT, "day2_rnd_extraction"))
sys.path.append(os.path.join(ROOT, "day3_wasserstein_kernel"))
sys.path.append(os.path.join(ROOT, "day4_gp_model"))
sys.path.append(os.path.join(ROOT, "day5_baseline"))
sys.path.append(os.path.join(ROOT, "day6_evaluation"))

from build_real_chain import load_filtered, bin_to_snapshots  # noqa: E402
from fit_iv_curve import smooth_all_days  # noqa: E402
from breeden_litzenberger import extract_all_days  # noqa: E402
from wasserstein_kernel import build_kernel_from_densities  # noqa: E402
from fit_gp import WassersteinGP, rnd_variance  # noqa: E402
from linear_baseline import MomentLinearBaseline, rnd_moments  # noqa: E402
from persistence_baseline import PersistenceForecaster  # noqa: E402
from evaluate import run_evaluation, print_comparison_table, make_comparison_plot  # noqa: E402


def main(tick_csv_gz: str, underlying: str, expiration_us: int, freq: str):
    for d in [
        "day1_data_collection/data", "day2_rnd_extraction/data",
        "day4_gp_model/results", "day5_baseline/results", "day6_evaluation/results",
    ]:
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)

    print("### DAY 1: real Deribit BTC-26JUL19 calls, 2019-07-01, 30-min snapshots ###")
    raw = load_filtered(tick_csv_gz, underlying, expiration_us)
    chain = bin_to_snapshots(raw, freq=freq)
    chain.to_csv(os.path.join(ROOT, "day1_data_collection/data/option_chains.csv"), index=False)
    curves, params = smooth_all_days(chain)
    curves.to_csv(os.path.join(ROOT, "day1_data_collection/data/iv_curves.csv"), index=False)
    params.to_csv(os.path.join(ROOT, "day1_data_collection/data/svi_params.csv"), index=False)
    print(f"  {chain['date'].nunique()} snapshots, {len(chain)} raw chain rows -> smoothed IV curves written.\n")

    print("### DAY 2: Breeden-Litzenberger RND extraction ###")
    rnd, mtg_diagnostics = extract_all_days(curves)
    rnd.to_csv(os.path.join(ROOT, "day2_rnd_extraction/data/rnd_curves.csv"), index=False)
    mtg_diagnostics.to_csv(os.path.join(ROOT, "day2_rnd_extraction/data/martingale_diagnostics.csv"), index=False)
    n_fail = (~mtg_diagnostics["passed"]).sum()
    print(f"  Extracted RNDs for {rnd['date'].nunique()} snapshots.")
    print(
        "  Martingale check (extracted mean vs forward, tol=1%): "
        f"{len(mtg_diagnostics) - n_fail}/{len(mtg_diagnostics)} pass, "
        f"max rel_error = {mtg_diagnostics['rel_error'].max():.4%}\n"
    )

    print("### DAY 3: Wasserstein kernel construction on real RND sequence ###")
    dates = sorted(rnd["date"].unique())
    grids, densities = [], []
    for d in dates:
        g = rnd[rnd["date"] == d].sort_values("strike")
        grids.append(g["strike"].to_numpy())
        densities.append(g["density"].to_numpy())

    Q, W2, K = build_kernel_from_densities(grids, densities, gamma=50.0)
    eigvals = np.linalg.eigvalsh(K)
    pd.DataFrame(K).to_csv(os.path.join(ROOT, "day3_wasserstein_kernel/gram_matrix.csv"), index=False)
    print(f"  Built {K.shape[0]}x{K.shape[0]} kernel matrix.")
    print(f"  min eigenvalue = {eigvals.min():.3e} (PSD check: {'OK' if eigvals.min() > -1e-6 else 'FAIL'})\n")

    print("### DAY 4: fit Wasserstein-kernel GP, RND_t -> Var(RND_{t+1}) ###")
    y_all = np.array([rnd_variance(g, dens) for g, dens in zip(grids, densities)])
    X_Q, y = Q[:-1], y_all[1:]

    n = len(y)
    n_train = int(n * 0.8)
    gp = WassersteinGP().fit(X_Q[:n_train], y[:n_train])
    mean_pred, std_pred = gp.predict(X_Q[n_train:])
    y_test = y[n_train:]

    rmse = np.sqrt(np.mean((mean_pred - y_test) ** 2))
    z = (y_test - mean_pred) / std_pred
    coverage_95 = np.mean(np.abs(z) < 1.96)

    print(f"  Fitted hyperparameters: sigma_f2={gp.sigma_f2:.5f}, gamma={gp.gamma:.5f}, sigma_n2={gp.sigma_n2:.2e}")
    print(f"  Held-out RMSE = {rmse:.6f}, 95% coverage = {coverage_95:.1%} (n_test={len(y_test)})")

    out = pd.DataFrame(
        {"date": dates[n_train + 1: n + 1], "y_true": y_test, "y_pred_mean": mean_pred, "y_pred_std": std_pred}
    )
    out.to_csv(os.path.join(ROOT, "day4_gp_model/results/predictions.csv"), index=False)
    with open(os.path.join(ROOT, "day4_gp_model/results/summary.txt"), "w") as f:
        f.write(f"sigma_f2={gp.sigma_f2}\ngamma={gp.gamma}\nsigma_n2={gp.sigma_n2}\n")
        f.write(f"rmse={rmse}\ncoverage_95={coverage_95}\n")

    print("\n### DAY 5: moment-based linear baseline ###")
    moments_all = np.array([rnd_moments(g, d) for g, d in zip(grids, densities)])
    X_moments = moments_all[:-1]
    baseline = MomentLinearBaseline().fit(X_moments[:n_train], y[:n_train])
    bl_mean_pred, bl_std_pred = baseline.predict(X_moments[n_train:])
    bl_rmse = np.sqrt(np.mean((bl_mean_pred - y_test) ** 2))
    print(f"  Baseline held-out RMSE = {bl_rmse:.6f} (GP RMSE was {rmse:.6f})")

    bl_out = pd.DataFrame(
        {"date": dates[n_train + 1: n + 1], "y_true": y_test, "y_pred_mean": bl_mean_pred, "y_pred_std": bl_std_pred}
    )
    bl_out.to_csv(os.path.join(ROOT, "day5_baseline/results/baseline_predictions.csv"), index=False)
    print("  Wrote day5_baseline/results/baseline_predictions.csv\n")

    print("### DAY 5b: persistence benchmark ###")
    y_persistence_all = y_all[:-1]
    y_persistence_train, y_persistence_test = y_persistence_all[:n_train], y_persistence_all[n_train:]
    persistence = PersistenceForecaster().fit(y_persistence_train, y_train=y[:n_train])
    pers_mean_pred, pers_std_pred = persistence.predict(y_persistence_test)
    pers_rmse = np.sqrt(np.mean((pers_mean_pred - y_test) ** 2))
    print(f"  Persistence held-out RMSE = {pers_rmse:.6f}  (GP RMSE = {rmse:.6f}, baseline RMSE = {bl_rmse:.6f})")

    pers_out = pd.DataFrame(
        {"date": dates[n_train + 1: n + 1], "y_true": y_test, "y_pred_mean": pers_mean_pred, "y_pred_std": pers_std_pred}
    )
    pers_out.to_csv(os.path.join(ROOT, "day5_baseline/results/persistence_predictions.csv"), index=False)
    print("  Wrote day5_baseline/results/persistence_predictions.csv\n")

    print("### DAY 6: evaluate GP vs baseline vs persistence ###")
    metrics, per_day = run_evaluation(os.path.join(ROOT, "day2_rnd_extraction/data/rnd_curves.csv"))
    print_comparison_table(metrics)
    per_day.to_csv(os.path.join(ROOT, "day6_evaluation/results/comparison_per_day.csv"), index=False)
    pd.Series(metrics).to_csv(os.path.join(ROOT, "day6_evaluation/results/comparison_summary.csv"))
    make_comparison_plot(metrics, os.path.join(ROOT, "day6_evaluation/results/comparison_plot.png"))
    print("  Wrote day6_evaluation/results/{comparison_per_day.csv, comparison_summary.csv, comparison_plot.png}")

    print("\n### DAY 7: build write-up demo (docs/index.html) ###")
    sys.path.append(os.path.join(ROOT, "day7_writeup"))
    from build_demo import build as build_demo  # noqa: E402
    build_demo()

    print("\nAll outputs written (real Deribit BTC-26JUL19 data, 2019-07-01).")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick-csv-gz", required=True)
    ap.add_argument("--underlying", default="BTC")
    ap.add_argument("--expiration-us", type=int, default=1564128000000000)
    ap.add_argument("--freq", default="30min")
    args = ap.parse_args()
    main(args.tick_csv_gz, args.underlying, args.expiration_us, args.freq)
