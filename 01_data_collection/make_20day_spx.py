from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "01_data_collection"
    / "data"
    / "option_chains_spx.csv"
)

OUTPUT = (
    ROOT
    / "01_data_collection"
    / "data"
    / "option_chains_spx_20d.csv"
)


df = pd.read_csv(INPUT, parse_dates=["date"])

dates = sorted(df["date"].unique())

# Keep the latest 20 available trading dates
selected_dates = dates[-20:]

df_20 = df[
    df["date"].isin(selected_dates)
].copy()

df_20 = df_20.sort_values(
    ["date", "expiry_date", "option_type", "strike"]
)

df_20.to_csv(
    OUTPUT,
    index=False
)

print("=" * 70)
print("20-DAY SPX PILOT DATASET")
print("=" * 70)

print(f"Dates selected : {len(selected_dates)}")
print(f"Rows           : {len(df_20):,}")

print()
print("Date range:")
print(selected_dates[0])
print("to")
print(selected_dates[-1])

print()
print("Dates:")
for d in selected_dates:
    print(d)

print()
print(f"Saved to:")
print(OUTPUT)
print("=" * 70)