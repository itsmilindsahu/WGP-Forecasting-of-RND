from __future__ import annotations

from datetime import date

import pandas as pd


def nyse_sessions(start_date: date, end_date: date) -> list[pd.Timestamp]:
    import exchange_calendars

    calendar = exchange_calendars.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
    )
    return [pd.Timestamp(session).tz_localize(None).normalize() for session in sessions]


def build_trading_pairs(
    calendar_sessions: list[pd.Timestamp],
    observed_dates: set[pd.Timestamp],
    horizon: int,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be a positive number of trading sessions")

    sessions = [
        pd.Timestamp(session).tz_localize(None).normalize()
        for session in calendar_sessions
    ]
    observed = {
        pd.Timestamp(value).tz_localize(None).normalize()
        for value in observed_dates
    }

    pairs = []
    for index in range(len(sessions) - horizon):
        input_date = sessions[index]
        target_date = sessions[index + horizon]
        if input_date in observed and target_date in observed:
            pairs.append(
                {
                    "input_date": input_date,
                    "target_date": target_date,
                    "horizon_sessions": horizon,
                }
            )

    return pd.DataFrame(
        pairs,
        columns=["input_date", "target_date", "horizon_sessions"],
    )
