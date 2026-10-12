from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta

from dotenv import load_dotenv

# Load VolForge / ThetaData configuration
load_dotenv()

import pandas as pd

from vol_forge.theta_data.theta_data_client import ThetaDataClientFactory


# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "SPX"

MIN_DTE = 7
MAX_DTE = 60

END_DATE = date(2026, 9, 30)
CALENDAR_DAYS = 35

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "01_data_collection" / "data"
RAW_DIR = OUT_DIR / "raw"
RAW_RESPONSE_DIR = RAW_DIR / "api_responses"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "option_chains_spx.csv"


# ============================================================
# THETADATA
# ============================================================

client = ThetaDataClientFactory.create_instance()


# ============================================================
# GENERATE CANDIDATE DATES
# ============================================================

start_date = END_DATE - timedelta(days=CALENDAR_DAYS)

dates = []

d = start_date

while d <= END_DATE:

    if d.weekday() < 5:
        dates.append(d)

    d += timedelta(days=1)


print("=" * 70)
print("SPX / THETADATA DATA COLLECTION")
print("=" * 70)

print(f"Candidate dates : {start_date} -> {END_DATE}")
print(f"DTE range       : {MIN_DTE} -> {MAX_DTE}")
print()


# ============================================================
# COLLECT DATA
# ============================================================

all_rows = []

successful_dates = []
filter_diagnostics = []


for value_date in dates:

    print(f"Fetching {value_date} ...")

    try:

        # ----------------------------------------------------
        # OPTION CHAIN
        # ----------------------------------------------------

        options = client.option_history_eod(
            value_date,
            value_date,
            SYMBOL,
            "*",
            max_dte=MAX_DTE,
        )

        if options is None or options.is_empty():

            print("  No option data.")
            filter_diagnostics.append({
                "date": value_date,
                "raw_rows": 0,
                "dte_rows": 0,
                "valid_quote_rows": 0,
                "clean_rows": 0,
                "missing_quote_timestamp_rows": 0,
            })
            continue

        raw_options = options.to_pandas()
        raw_rows = len(raw_options)


        # ----------------------------------------------------
        # INDEX DATA
        # ----------------------------------------------------

        index_data = client.index_history_eod(
            SYMBOL,
            value_date,
            value_date,
        )

        response_id = pd.Timestamp.now(
            tz="UTC"
        ).strftime("%Y%m%dT%H%M%S%fZ")

        raw_options.to_parquet(
            RAW_RESPONSE_DIR
            / f"SPX_options_{value_date}_{response_id}.parquet",
            index=False,
        )

        if index_data is not None:
            raw_index = (
                index_data.to_pandas()
                if hasattr(index_data, "to_pandas")
                else pd.DataFrame(index_data)
            )
            raw_index.to_parquet(
                RAW_RESPONSE_DIR
                / f"SPX_index_{value_date}_{response_id}.parquet",
                index=False,
            )


        # ----------------------------------------------------
        # CONVERT TO PANDAS
        # ----------------------------------------------------

        pdf = raw_options.copy()


        # ----------------------------------------------------
        # PRESERVE ORIGINAL THETADATA FIELDS
        # ----------------------------------------------------

        # Explicit research-standard names.
        pdf["expiry"] = pd.to_datetime(
            pdf["expiration"]
        ).dt.date

        pdf["right"] = (
            pdf["right"]
            .astype(str)
            .str.upper()
        )

        pdf["date"] = pd.Timestamp(value_date)


        # ----------------------------------------------------
        # DTE
        # ----------------------------------------------------

        pdf["dte"] = (
            pd.to_datetime(pdf["expiry"])
            - pd.Timestamp(value_date)
        ).dt.days


        pdf = pdf[
            (pdf["dte"] >= MIN_DTE)
            & (pdf["dte"] <= MAX_DTE)
        ].copy()


        if pdf.empty:

            print("  No contracts in DTE range.")
            filter_diagnostics.append({
                "date": value_date,
                "raw_rows": raw_rows,
                "dte_rows": 0,
                "valid_quote_rows": 0,
                "clean_rows": 0,
                "missing_quote_timestamp_rows": int(
                    raw_options.get("timestamp", pd.Series(index=raw_options.index, dtype=object)).isna().sum()
                ),
            })
            continue

        dte_rows = len(pdf)


        # ----------------------------------------------------
        # MID QUOTE
        # ----------------------------------------------------

        pdf["mid"] = (
            pdf["bid"] + pdf["ask"]
        ) / 2.0


        # ----------------------------------------------------
        # BASIC QUOTE FILTERS
        # ----------------------------------------------------

        pdf = pdf[
            pdf["bid"].notna()
            & pdf["ask"].notna()
            & (pdf["bid"] >= 0)
            & (pdf["ask"] >= 0)
            & (pdf["ask"] >= pdf["bid"])
        ].copy()

        valid_quote_rows = len(pdf)


        if pdf.empty:

            print("  No valid quotes.")
            timestamp = raw_options.get(
                "timestamp",
                pd.Series(index=raw_options.index, dtype=object),
            )
            filter_diagnostics.append({
                "date": value_date,
                "raw_rows": raw_rows,
                "dte_rows": dte_rows,
                "valid_quote_rows": 0,
                "clean_rows": 0,
                "missing_quote_timestamp_rows": int(timestamp.isna().sum()),
            })
            continue


        # ----------------------------------------------------
        # TIME TO MATURITY
        # ----------------------------------------------------

        pdf["tau_years"] = pdf["dte"] / 365.0


        # ----------------------------------------------------
        # COMPATIBILITY ALIASES
        #
        # Existing downstream scripts use these names.
        # ----------------------------------------------------

        pdf["expiry_date"] = pdf["expiry"]
        pdf["option_type"] = pdf["right"]


        # ----------------------------------------------------
        # OPTIONAL MARKET-DATA FIELDS
        #
        # Keep them if ThetaData provides them.
        # Otherwise create empty columns so the schema
        # remains explicit.
        # ----------------------------------------------------

        optional_fields = [
            "created",
            "last_trade",
            "timestamp",
        ]

        for field in optional_fields:

            if field not in pdf.columns:
                pdf[field] = pd.NaT

        filter_diagnostics.append({
            "date": value_date,
            "raw_rows": raw_rows,
            "dte_rows": dte_rows,
            "valid_quote_rows": valid_quote_rows,
            "clean_rows": len(pdf),
            "missing_quote_timestamp_rows": int(pdf["timestamp"].isna().sum()),
        })


        # ----------------------------------------------------
        # KEEP CLEAN RESEARCH SCHEMA
        # ----------------------------------------------------

        keep = [
            "date",
            "expiry",
            "expiry_date",

            "strike",

            "right",
            "option_type",

            "bid",
            "ask",
            "mid",

            "volume",

            "created",
            "last_trade",
            "timestamp",

            "dte",
            "tau_years",
        ]


        pdf = pdf[keep].copy()


        # ----------------------------------------------------
        # ADD TO MASTER CLEAN DATASET
        # ----------------------------------------------------

        all_rows.append(pdf)

        successful_dates.append(value_date)


        print(
            f"  {len(pdf):,} clean contracts"
        )


    except Exception as exc:

        print(
            f"  ERROR: {exc}"
        )


# ============================================================
# COMBINE
# ============================================================

if not all_rows:

    raise RuntimeError(
        "No SPX data was collected."
    )


chain = pd.concat(
    all_rows,
    ignore_index=True,
)


# ============================================================
# BASIC SUMMARY
# ============================================================

print()
print("=" * 70)
print("COLLECTION SUMMARY")
print("=" * 70)

print(
    f"Trading dates collected : "
    f"{chain['date'].nunique()}"
)

print(
    f"Total clean contracts   : "
    f"{len(chain):,}"
)

print()

print(
    chain.groupby("right")
    .size()
)

print()

print(
    chain.groupby("date")
    .size()
)


# ============================================================
# SAVE
# ============================================================

chain.to_csv(
    OUTPUT_FILE,
    index=False,
)

pd.DataFrame(filter_diagnostics).to_csv(
    OUT_DIR / "spx_collection_diagnostics.csv",
    index=False,
)


print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("Schema:")
print(
    ", ".join(chain.columns)
)

print("=" * 70)