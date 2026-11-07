from datetime import date
import unittest

import pandas as pd

from trading_calendar_pairs import build_trading_pairs, nyse_sessions


class TradingCalendarPairTests(unittest.TestCase):
    def test_missing_session_drops_only_pairs_with_missing_endpoint(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=5, freq="B")
        observed = set(sessions) - {pd.Timestamp("2024-01-03")}

        pairs = build_trading_pairs(list(sessions), observed, horizon=1)

        self.assertEqual(
            list(zip(pairs.input_date, pairs.target_date)),
            [
                (pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")),
                (pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")),
            ],
        )
        self.assertFalse(
            (
                (pairs.input_date == pd.Timestamp("2024-01-02"))
                & (pairs.target_date == pd.Timestamp("2024-01-04"))
            ).any()
        )

    def test_one_session_pair_across_weekend_and_exchange_holiday(self) -> None:
        sessions = nyse_sessions(date(2024, 3, 28), date(2024, 4, 1))

        self.assertEqual(
            sessions,
            [
                pd.Timestamp("2024-03-28"),
                pd.Timestamp("2024-04-01"),
            ],
        )
        pairs = build_trading_pairs(sessions, set(sessions), horizon=1)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.iloc[0].input_date, pd.Timestamp("2024-03-28"))
        self.assertEqual(pairs.iloc[0].target_date, pd.Timestamp("2024-04-01"))
