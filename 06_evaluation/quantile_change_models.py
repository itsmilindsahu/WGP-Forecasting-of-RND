from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GAMMA_MULTIPLIERS = (0.1, 0.3, 1.0, 3.0, 10.0)
REGULARIZATION_GRID = (1e-6, 1e-4, 1e-2, 1e-1, 1.0)
INNER_CV_BOUNDARIES = (0.50, 0.65, 0.80, 1.0)
PCA_EXPLAINED_VARIANCE = 0.95


def _isotonic_increasing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Isotonic input must be a finite one-dimensional array")

    levels: list[float] = []
    weights: list[int] = []
    for value in values:
        levels.append(float(value))
        weights.append(1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            combined_weight = weights[-2] + weights[-1]
            combined_level = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / combined_weight
            levels[-2:] = [combined_level]
            weights[-2:] = [combined_weight]

    result = np.empty_like(values)
    offset = 0
    for level, weight in zip(levels, weights):
        result[offset : offset + weight] = level
        offset += weight
    return result


def normalize_quantile(quantile: np.ndarray) -> np.ndarray:
    quantile = np.asarray(quantile, dtype=float)
    if quantile.ndim != 1 or not np.isfinite(quantile).all():
        raise ValueError("Quantile must be a finite one-dimensional array")
    projected = np.maximum(_isotonic_increasing(quantile), 0.0)
    mean = float(projected.mean())
    if mean <= 0:
        raise ValueError("Quantile must have positive mean")
    return projected / mean


def normalize_quantile_matrix(quantiles: np.ndarray) -> np.ndarray:
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.ndim != 2:
        raise ValueError("Quantiles must be a two-dimensional array")
    return np.stack([normalize_quantile(row) for row in quantiles])


def project_quantile(
    input_quantile: np.ndarray,
    predicted_change: np.ndarray,
) -> np.ndarray:
    input_quantile = np.asarray(input_quantile, dtype=float)
    predicted_change = np.asarray(predicted_change, dtype=float)
    if input_quantile.shape != predicted_change.shape:
        raise ValueError("Input quantile and predicted change shapes must match")
    if not np.isfinite(input_quantile).all() or not np.isfinite(predicted_change).all():
        raise ValueError("Input quantile and predicted change must be finite")
    if (
        np.all(input_quantile >= 0)
        and np.all(np.diff(input_quantile) >= 0)
        and np.isclose(input_quantile.mean(), 1.0, rtol=0.0, atol=1e-14)
        and np.all(predicted_change == 0)
    ):
        return input_quantile.copy()
    return normalize_quantile(input_quantile + predicted_change)


def project_quantile_matrix(
    input_quantiles: np.ndarray,
    predicted_changes: np.ndarray,
) -> np.ndarray:
    input_quantiles = np.asarray(input_quantiles, dtype=float)
    predicted_changes = np.asarray(predicted_changes, dtype=float)
    if input_quantiles.shape != predicted_changes.shape:
        raise ValueError("Input and change matrices must have matching shapes")
    return np.stack(
        [
            project_quantile(input_quantile, change)
            for input_quantile, change in zip(input_quantiles, predicted_changes)
        ]
    )


def squared_w2_per_pair(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError("Prediction and target matrices must have matching 2D shapes")
    probability_grid = np.linspace(1e-4, 1 - 1e-4, predictions.shape[1])
    return np.trapezoid(
        (predictions - targets) ** 2,
        probability_grid,
        axis=1,
    )


def moment_features(quantiles: np.ndarray) -> np.ndarray:
    quantiles = np.asarray(quantiles, dtype=float)
    probability_grid = np.linspace(1e-4, 1 - 1e-4, quantiles.shape[1])
    width = probability_grid[-1] - probability_grid[0]
    means = np.trapezoid(quantiles, probability_grid, axis=1) / width
    centered = quantiles - means[:, None]
    variances = np.trapezoid(centered**2, probability_grid, axis=1) / width
    stds = np.sqrt(np.maximum(variances, 1e-16))
    skewness = (
        np.trapezoid(centered**3, probability_grid, axis=1)
        / width
        / stds**3
    )
    kurtosis = (
        np.trapezoid(centered**4, probability_grid, axis=1)
        / width
        / stds**4
    )
    return np.column_stack((stds, skewness, kurtosis))


def pca_features(
    train_quantiles: np.ndarray,
    evaluation_quantiles: np.ndarray,
    explained_variance: float = PCA_EXPLAINED_VARIANCE,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_quantiles = np.asarray(train_quantiles, dtype=float)
    evaluation_quantiles = np.asarray(evaluation_quantiles, dtype=float)
    if train_quantiles.ndim != 2 or evaluation_quantiles.ndim != 2:
        raise ValueError("PCA inputs must be two-dimensional")
    if train_quantiles.shape[1] != evaluation_quantiles.shape[1]:
        raise ValueError("Training and evaluation quantile dimensions must match")
    if not 0 < explained_variance <= 1:
        raise ValueError("Explained variance threshold must be in (0, 1]")

    center = train_quantiles.mean(axis=0)
    _, singular_values, components = np.linalg.svd(
        train_quantiles - center,
        full_matrices=False,
    )
    variance = singular_values**2
    total_variance = float(variance.sum())
    if total_variance <= 0:
        component_count = 1
    else:
        component_count = int(
            np.searchsorted(
                np.cumsum(variance) / total_variance,
                explained_variance,
            )
            + 1
        )
    basis = components[:component_count]
    train_scores = (train_quantiles - center) @ basis.T
    evaluation_scores = (evaluation_quantiles - center) @ basis.T
    return train_scores, evaluation_scores, component_count


def standardize_features(
    train_features: np.ndarray,
    evaluation_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_features = np.asarray(train_features, dtype=float)
    evaluation_features = np.asarray(evaluation_features, dtype=float)
    if train_features.ndim != 2 or evaluation_features.ndim != 2:
        raise ValueError("Feature matrices must be two-dimensional")
    if train_features.shape[1] != evaluation_features.shape[1]:
        raise ValueError("Training and evaluation feature widths must match")
    if not np.isfinite(train_features).all() or not np.isfinite(
        evaluation_features
    ).all():
        raise ValueError("Feature matrices must contain only finite values")
    center = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train_features - center) / scale, (evaluation_features - center) / scale


def pairwise_squared_distances(
    left: np.ndarray,
    right: np.ndarray | None = None,
) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right_values = left if right is None else np.asarray(right, dtype=float)
    if left.ndim != 2 or right_values.ndim != 2:
        raise ValueError("Distance inputs must be two-dimensional")
    if left.shape[1] != right_values.shape[1]:
        raise ValueError("Distance input widths must match")
    distances = (
        np.sum(left**2, axis=1)[:, None]
        + np.sum(right_values**2, axis=1)[None, :]
        - 2 * left @ right_values.T
    )
    distances = np.maximum(distances, 0.0)
    if right is None:
        distances = 0.5 * (distances + distances.T)
        np.fill_diagonal(distances, 0.0)
    return distances


def median_distance_scale(squared_distances: np.ndarray) -> float:
    squared_distances = np.asarray(squared_distances, dtype=float)
    if squared_distances.ndim != 2 or squared_distances.shape[0] != squared_distances.shape[1]:
        raise ValueError("Median heuristic requires a square training distance matrix")
    upper = squared_distances[np.triu_indices(len(squared_distances), k=1)]
    positive = upper[np.isfinite(upper) & (upper > 0)]
    return float(np.median(positive)) if len(positive) else 1.0


def forward_chaining_folds(
    sample_count: int,
    boundaries: tuple[float, ...] = INNER_CV_BOUNDARIES,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if sample_count < 8:
        raise ValueError("At least 8 training pairs are required for inner CV")
    if not boundaries or boundaries[-1] != 1.0:
        raise ValueError("CV boundaries must end at 1.0")
    cuts = [int(np.floor(sample_count * value)) for value in boundaries]
    if len(cuts) < 2:
        raise ValueError("At least two CV boundaries are required")
    folds = []
    previous_end = cuts[0]
    for end in cuts[1:]:
        if end <= previous_end:
            raise ValueError("CV boundaries create an empty validation block")
        train_indices = np.arange(previous_end)
        validation_indices = np.arange(previous_end, end)
        folds.append((train_indices, validation_indices))
        previous_end = end
    if len(folds) != len(boundaries) - 1:
        raise ValueError("Training window is too small to form all forward CV folds")
    return folds


@dataclass(frozen=True)
class KernelFit:
    alpha: float
    output_mean: np.ndarray
    output_scale: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    standardized_targets: np.ndarray

    def predict(
        self,
        cross_kernel: np.ndarray,
        alpha: float | None = None,
    ) -> np.ndarray:
        selected_alpha = self.alpha if alpha is None else float(alpha)
        spectral_targets = self.eigenvectors.T @ self.standardized_targets
        coefficients = self.eigenvectors @ (
            spectral_targets / (self.eigenvalues[:, None] + selected_alpha)
        )
        return (cross_kernel @ coefficients) * self.output_scale + self.output_mean


def prepare_kernel_fit(
    training_kernel: np.ndarray,
    training_targets: np.ndarray,
    alpha: float,
) -> KernelFit:
    training_kernel = np.asarray(training_kernel, dtype=float)
    training_targets = np.asarray(training_targets, dtype=float)
    if training_kernel.ndim != 2 or training_kernel.shape[0] != training_kernel.shape[1]:
        raise ValueError("Training kernel must be square")
    if training_targets.ndim != 2 or training_targets.shape[0] != len(training_kernel):
        raise ValueError("Kernel targets must match the training rows")
    if alpha <= 0:
        raise ValueError("Kernel noise/regularization must be positive")
    output_mean = training_targets.mean(axis=0)
    output_scale = training_targets.std(axis=0)
    output_scale[output_scale < 1e-10] = 1.0
    standardized_targets = (training_targets - output_mean) / output_scale
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (training_kernel + training_kernel.T)
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return KernelFit(
        alpha=float(alpha),
        output_mean=output_mean,
        output_scale=output_scale,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        standardized_targets=standardized_targets,
    )


def ridge_predict(
    training_features: np.ndarray,
    training_targets: np.ndarray,
    evaluation_features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    training_features = np.asarray(training_features, dtype=float)
    training_targets = np.asarray(training_targets, dtype=float)
    evaluation_features = np.asarray(evaluation_features, dtype=float)
    x_mean = training_features.mean(axis=0)
    x_scale = training_features.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    train_x = (training_features - x_mean) / x_scale
    eval_x = (evaluation_features - x_mean) / x_scale
    y_mean = training_targets.mean(axis=0)
    y_scale = training_targets.std(axis=0)
    y_scale[y_scale < 1e-10] = 1.0
    train_y = (training_targets - y_mean) / y_scale
    gram = train_x.T @ train_x + alpha * np.eye(train_x.shape[1])
    coefficients = np.linalg.solve(gram, train_x.T @ train_y)
    return (eval_x @ coefficients) * y_scale + y_mean
