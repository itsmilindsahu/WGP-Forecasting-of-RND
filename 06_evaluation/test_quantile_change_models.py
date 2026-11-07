from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

EVALUATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_DIR))

from quantile_change_models import (  # noqa: E402
    normalize_quantile_matrix,
    pca_features,
    project_quantile_matrix,
)
from run_phase4_quantile_change import (  # noqa: E402
    _assert_shared_index_signatures,
    _assert_shared_pair_indices,
)


class QuantileChangeModelTests(unittest.TestCase):
    def test_projection_is_valid_for_random_input_changes(self):
        rng = np.random.default_rng(2026)
        input_quantiles = normalize_quantile_matrix(
            rng.normal(1.0, 0.3, size=(12, 80))
        )
        changes = rng.normal(0.0, 0.4, size=(12, 80))

        projected = project_quantile_matrix(input_quantiles, changes)

        self.assertTrue(np.all(projected >= 0))
        self.assertTrue(np.all(np.diff(projected, axis=1) >= -1e-14))
        self.assertTrue(np.allclose(projected.mean(axis=1), 1.0, atol=1e-14))

    def test_zero_change_reproduces_persistence_exactly(self):
        rng = np.random.default_rng(17)
        inputs = normalize_quantile_matrix(rng.normal(1.0, 0.2, size=(5, 40)))

        forecasts = project_quantile_matrix(inputs, np.zeros_like(inputs))

        np.testing.assert_array_equal(forecasts, inputs)

    def test_pca_training_features_do_not_depend_on_evaluation_values(self):
        rng = np.random.default_rng(91)
        training = rng.normal(size=(24, 30))
        evaluation = rng.normal(size=(4, 30))
        changed_evaluation = evaluation + 1e5

        training_scores, evaluation_scores, _ = pca_features(
            training, evaluation
        )
        repeated_training_scores, changed_scores, _ = pca_features(
            training, changed_evaluation
        )

        np.testing.assert_allclose(training_scores, repeated_training_scores)
        self.assertFalse(np.allclose(evaluation_scores, changed_scores))

    def test_all_models_must_share_pair_indices(self):
        shared = pd.DataFrame(
            [
                {
                    "evaluation": "rolling_origin",
                    "outer_split": "origin_50pct",
                    "training_condition": "full_training",
                    "horizon_sessions": 1,
                    "model": model,
                    "input_date": input_date,
                    "target_date": target_date,
                }
                for model, input_date, target_date in (
                    ("Persistence", "2024-01-02", "2024-01-03"),
                    ("Persistence", "2024-01-03", "2024-01-04"),
                    ("GP", "2024-01-02", "2024-01-03"),
                    ("GP", "2024-01-03", "2024-01-04"),
                )
            ]
        )
        _assert_shared_pair_indices(shared)

        shared.loc[
            (shared["model"] == "GP") & (shared["input_date"] == "2024-01-03"),
            "target_date",
        ] = "2024-01-05"
        with self.assertRaises(AssertionError):
            _assert_shared_pair_indices(shared)

    def test_train_and_test_pair_signatures_must_match_across_models(self):
        selections = pd.DataFrame(
            [
                {
                    "evaluation": "rolling_origin",
                    "outer_split": "origin_50pct",
                    "training_condition": "full_training",
                    "horizon_sessions": 1,
                    "model": model,
                    "train_pair_index_hash": "train-hash",
                    "test_pair_index_hash": "test-hash",
                }
                for model in ("Persistence", "GP")
            ]
        )
        _assert_shared_index_signatures(selections)

        selections.loc[selections["model"] == "GP", "train_pair_index_hash"] = (
            "different-train-hash"
        )
        with self.assertRaises(AssertionError):
            _assert_shared_index_signatures(selections)


if __name__ == "__main__":
    unittest.main()
