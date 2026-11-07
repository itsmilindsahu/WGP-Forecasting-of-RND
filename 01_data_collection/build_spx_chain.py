from __future__ import annotations

from pathlib import Path
from datetime import date
import time

from dotenv import load_dotenv
import exchange_calendars

# Load VolForge / ThetaData configuration
load_dotenv(
    Path(r"C:\Work\UNIVERSITY OF AMSTERDAM\VolForge\.env")
)

import pandas as pd

from vol_forge.theta_data.theta_data_client import ThetaDataClientFactory


# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "SPX"
ROOTS = ("SPX", "SPXW")

MIN_DTE = 7
MAX_DTE = 60
REQUEST_BATCH_SESSIONS = 4

START_DATE = date(2024, 1, 2)
END_DATE = date(2026, 11, 4)
REQUEST_DELAY_SECONDS = 1.0

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


def write_parquet_cache(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def cache_root_date_chunks(
    dates: list[date],
) -> tuple[dict[tuple[date, str], str], list[dict[str, object]]]:
    failures = {}
    retry_diagnostics = []
    for offset in range(0, len(dates), REQUEST_BATCH_SESSIONS):
        date_chunk = dates[offset : offset + REQUEST_BATCH_SESSIONS]
        for root in ROOTS:
            missing_dates = [
                value_date
                for value_date in date_chunk
                if not (
                    (RAW_RESPONSE_DIR / f"{root}_options_{value_date}.parquet").exists()
                    or (RAW_RESPONSE_DIR / f"{root}_options_{value_date}.empty").exists()
                )
            ]
            if not missing_dates:
                continue

            try:
                time.sleep(REQUEST_DELAY_SECONDS)
                options = client.option_history_eod(
                    date_chunk[0],
                    date_chunk[-1],
                    root,
                    "*",
                    max_dte=MAX_DTE,
                )
                if options is None or options.is_empty():
                    for value_date in missing_dates:
                        (RAW_RESPONSE_DIR / f"{root}_options_{value_date}.empty").touch()
                    continue

                raw_options = options.to_pandas()
                date_column = next(
                    (
                        column
                        for column in ("created", "timestamp", "date")
                        if column in raw_options.columns
                    ),
                    None,
                )
                if date_column is None:
                    if len(date_chunk) != 1:
                        raise ValueError(
                            "Multi-date option response has no observation-date column"
                        )
                    observation_dates = pd.Series(date_chunk[0], index=raw_options.index)
                else:
                    observation_dates = pd.to_datetime(
                        raw_options[date_column],
                        errors="coerce",
                    ).dt.date

                for value_date in missing_dates:
                    day_options = raw_options.loc[
                        observation_dates == value_date
                    ].copy()
                    cache_path = RAW_RESPONSE_DIR / f"{root}_options_{value_date}.parquet"
                    empty_path = RAW_RESPONSE_DIR / f"{root}_options_{value_date}.empty"
                    if day_options.empty:
                        empty_path.touch()
                    else:
                        day_options["root"] = root
                        write_parquet_cache(day_options, cache_path)
            except Exception as exc:
                for value_date in missing_dates:
                    failures[(value_date, root)] = type(exc).__name__
                print(
                    f"ERROR option history {root} "
                    f"{date_chunk[0]}..{date_chunk[-1]}: {type(exc).__name__}"
                )
                for value_date in missing_dates:
                    try:
                        time.sleep(REQUEST_DELAY_SECONDS)
                        options = client.option_history_eod(
                            value_date,
                            value_date,
                            root,
                            "*",
                            max_dte=MAX_DTE,
                        )
                        if options is None or options.is_empty():
                            (RAW_RESPONSE_DIR / f"{root}_options_{value_date}.empty").touch()
                            retry_diagnostics.append({
                                "date": value_date,
                                "root": root,
                                "status": "empty_single_day_fallback",
                                "batch_error_type": type(exc).__name__,
                                "batch_error_message": str(exc)[:300],
                            })
                            failures.pop((value_date, root), None)
                            continue

                        day_options = options.to_pandas()
                        date_column = next(
                            (
                                column
                                for column in ("created", "timestamp", "date")
                                if column in day_options.columns
                            ),
                            None,
                        )
                        if date_column is not None:
                            observation_dates = pd.to_datetime(
                                day_options[date_column],
                                errors="coerce",
                            ).dt.date
                            day_options = day_options.loc[
                                observation_dates == value_date
                            ].copy()

                        if day_options.empty:
                            (RAW_RESPONSE_DIR / f"{root}_options_{value_date}.empty").touch()
                            retry_diagnostics.append({
                                "date": value_date,
                                "root": root,
                                "status": "empty_single_day_fallback",
                                "batch_error_type": type(exc).__name__,
                                "batch_error_message": str(exc)[:300],
                            })
                        else:
                            day_options["root"] = root
                            write_parquet_cache(
                                day_options,
                                RAW_RESPONSE_DIR / f"{root}_options_{value_date}.parquet",
                            )
                            retry_diagnostics.append({
                                "date": value_date,
                                "root": root,
                                "status": "single_day_fallback_succeeded",
                                "batch_error_type": type(exc).__name__,
                                "batch_error_message": str(exc)[:300],
                                "rows": len(day_options),
                            })
                        failures.pop((value_date, root), None)
                    except Exception as retry_exc:
                        failures[(value_date, root)] = (
                            f"{type(retry_exc).__name__}: {str(retry_exc)[:300]}"
                        )
                        retry_diagnostics.append({
                            "date": value_date,
                            "root": root,
                            "status": "single_day_fallback_failed",
                            "batch_error_type": type(exc).__name__,
                            "batch_error_message": str(exc)[:300],
                            "error_type": type(retry_exc).__name__,
                            "error_message": str(retry_exc)[:300],
                        })
    return failures, retry_diagnostics


# ============================================================
# GENERATE CANDIDATE DATES
# ============================================================

calendar = exchange_calendars.get_calendar("XNYS")
dates = [
    pd.Timestamp(session).date()
    for session in calendar.sessions_in_range(
        pd.Timestamp(START_DATE),
        pd.Timestamp(END_DATE),
    )
]


print("=" * 70)
print("SPX / THETADATA DATA COLLECTION")
print("=" * 70)

print(f"Candidate dates : {START_DATE} -> {END_DATE}")
print(f"Option roots    : {', '.join(ROOTS)}")
print(f"DTE range       : {MIN_DTE} -> {MAX_DTE}")
print()


# ============================================================
# COLLECT DATA
# ============================================================

all_rows = []

successful_dates = []
filter_diagnostics = []

batch_failures, retry_diagnostics = cache_root_date_chunks(dates)

for value_date in dates:

    print(f"Fetching {value_date} ...")
    current_stage = "option history"
    current_root = None

    try:

        root_frames = []
        root_row_counts = {}

        missing_root = None
        for root in ROOTS:
            current_root = root
            options_cache = RAW_RESPONSE_DIR / f"{root}_options_{value_date}.parquet"
            empty_cache = RAW_RESPONSE_DIR / f"{root}_options_{value_date}.empty"

            if options_cache.exists():
                root_options = pd.read_parquet(options_cache)
            elif empty_cache.exists():
                root_options = pd.DataFrame()
            else:
                missing_root = root
                break

            root_row_counts[root] = len(root_options)
            if not root_options.empty:
                root_options["root"] = root
                root_frames.append(root_options)

        if missing_root:
            error_type = batch_failures.get(
                (value_date, missing_root),
                "MissingDateCache",
            )
            print(f"  ERROR: no {missing_root} cache ({error_type})")
            filter_diagnostics.append({
                "date": value_date,
                "status": "error",
                "failed_stage": "option history",
                "failed_root": missing_root,
                "error_type": error_type,
            })
            continue

        if not root_frames:
            print("  No option data.")
            filter_diagnostics.append({
                "date": value_date,
                "raw_rows": 0,
                "dte_rows": 0,
                "valid_quote_rows": 0,
                "clean_rows": 0,
                "missing_quote_timestamp_rows": 0,
                "spx_raw_rows": root_row_counts["SPX"],
                "spxw_raw_rows": root_row_counts["SPXW"],
            })
            continue

        raw_options = pd.concat(root_frames, ignore_index=True)
        raw_rows = len(raw_options)

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
                    raw_options.get(
                        "created",
                        pd.Series(index=raw_options.index, dtype=object),
                    ).isna().sum()
                ),
                "spx_raw_rows": root_row_counts["SPX"],
                "spxw_raw_rows": root_row_counts["SPXW"],
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
            quote_timestamp = raw_options.get(
                "created",
                pd.Series(index=raw_options.index, dtype=object),
            )
            filter_diagnostics.append({
                "date": value_date,
                "raw_rows": raw_rows,
                "dte_rows": dte_rows,
                "valid_quote_rows": 0,
                "clean_rows": 0,
                "missing_quote_timestamp_rows": int(quote_timestamp.isna().sum()),
                "spx_raw_rows": root_row_counts["SPX"],
                "spxw_raw_rows": root_row_counts["SPXW"],
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
            "missing_quote_timestamp_rows": int(
                pdf.get(
                    "created",
                    pd.Series(index=pdf.index, dtype=object),
                ).isna().sum()
            ),
            "spx_raw_rows": root_row_counts["SPX"],
            "spxw_raw_rows": root_row_counts["SPXW"],
        })


        # ----------------------------------------------------
        # KEEP CLEAN RESEARCH SCHEMA
        # ----------------------------------------------------

        keep = [
            "date",
            "root",
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

        root_context = f" for {current_root}" if current_root else ""
        print(
            f"  ERROR during {current_stage}{root_context}: "
            f"{type(exc).__name__}"
        )
        filter_diagnostics.append({
            "date": value_date,
            "status": "error",
            "failed_stage": current_stage,
            "failed_root": current_root,
            "error_type": type(exc).__name__,
        })


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

pd.DataFrame(retry_diagnostics).to_csv(
    OUT_DIR / "spx_collection_retry_diagnostics.csv",
    index=False,
    columns=[
        "date",
        "root",
        "status",
        "batch_error_type",
        "batch_error_message",
        "rows",
        "error_type",
        "error_message",
    ],
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