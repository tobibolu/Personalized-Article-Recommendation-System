"""Matrix factorization (SVD) recommendation engine.

Decomposes the user-item interaction matrix into latent factors to capture
hidden patterns in user preferences and article characteristics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_svd(user_item_matrix: pd.DataFrame) -> tuple:
    """Perform SVD on the user-item matrix.

    Returns:
        Tuple of (U, s, Vt) from numpy SVD.
    """
    U, s, Vt = np.linalg.svd(user_item_matrix.fillna(0))
    return U, s, Vt


def explained_variance_ratio(s: np.ndarray) -> np.ndarray:
    """Compute cumulative explained variance ratio from singular values."""
    return np.cumsum(s**2) / np.sum(s**2)


def variance_thresholds(s: np.ndarray) -> dict:
    """Find number of components needed for 50%, 80%, 90% variance."""
    evr = explained_variance_ratio(s)
    return {
        "50pct": int(np.argmax(evr >= 0.5) + 1),
        "80pct": int(np.argmax(evr >= 0.8) + 1),
        "90pct": int(np.argmax(evr >= 0.9) + 1),
    }


def reconstruct_matrix(U: np.ndarray, s: np.ndarray, Vt: np.ndarray,
                       k: int) -> np.ndarray:
    """Reconstruct the user-item matrix using k latent factors."""
    return np.dot(U[:, :k] * s[:k], Vt[:k, :])


def evaluate_svd(user_item_matrix: pd.DataFrame,
                 n_factors_list: list = None,
                 test_size: float = 0.2,
                 seed: int = 42) -> pd.DataFrame:
    """Evaluate SVD recommendations across different numbers of latent factors.

    Args:
        user_item_matrix: Binary user-item matrix.
        n_factors_list: List of k values to evaluate.
        test_size: Fraction of rows to use as test set.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: n_factors, rmse, coverage.
    """
    if n_factors_list is None:
        n_factors_list = [10, 20, 50, 100]

    rng = np.random.RandomState(seed)
    test_idx = rng.choice(user_item_matrix.index,
                          size=int(len(user_item_matrix) * test_size),
                          replace=False)

    U, s, Vt = compute_svd(user_item_matrix)
    test_positions = [user_item_matrix.index.get_loc(i) for i in test_idx]
    test_matrix = user_item_matrix.loc[test_idx].fillna(0).values

    results = []
    for k in n_factors_list:
        pred_full = reconstruct_matrix(U, s, Vt, k)
        test_pred = pred_full[test_positions]
        rmse = float(np.sqrt(np.mean((test_matrix - test_pred) ** 2)))
        coverage = float(np.mean(pred_full > 0))
        results.append({"n_factors": k, "rmse": round(rmse, 4), "coverage": round(coverage, 4)})

    return pd.DataFrame(results)


def get_svd_recommendations(user_id: str, user_item_matrix: pd.DataFrame,
                            U: np.ndarray, s: np.ndarray, Vt: np.ndarray,
                            k: int = 50, n: int = 10) -> list:
    """Generate article recommendations for a user using SVD.

    Args:
        user_id: Email hash of the target user.
        user_item_matrix: Original binary user-item matrix.
        U, s, Vt: SVD components.
        k: Number of latent factors to use.
        n: Number of recommendations.

    Returns:
        List of recommended article IDs.
    """
    user_pos = user_item_matrix.index.get_loc(user_id)
    pred_row = np.dot(U[user_pos, :k] * s[:k], Vt[:k, :])

    already_read = set(user_item_matrix.columns[user_item_matrix.loc[user_id] == 1])
    article_scores = pd.Series(pred_row, index=user_item_matrix.columns)
    article_scores = article_scores.drop(labels=list(already_read), errors="ignore")

    return list(article_scores.sort_values(ascending=False).head(n).index)
