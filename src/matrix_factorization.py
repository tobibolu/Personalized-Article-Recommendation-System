"""Matrix factorization (SVD) recommendation engine.

Decomposes the user-item interaction matrix into latent factors to capture
hidden patterns in user preferences and article characteristics.
"""

import numpy as np
import pandas as pd

from .evaluation import evaluate_holdout


def compute_svd(user_item_matrix: pd.DataFrame) -> tuple:
    """Perform SVD on the user-item matrix.

    Returns:
        Tuple of (U, s, Vt) from numpy SVD.
    """
    U, s, Vt = np.linalg.svd(user_item_matrix.fillna(0), full_matrices=False)
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


def evaluate_svd(train_matrix: pd.DataFrame, holdout: dict,
                 n_factors_list: list = None,
                 k_values: tuple = (5, 10)) -> pd.DataFrame:
    """Evaluate SVD on interactions hidden before factorisation.

    Reconstruction RMSE on the matrix used to fit SVD is not a recommender
    evaluation: it rewards reproducing already-observed zeros and ones. This
    function instead measures whether held-out articles appear in the ranked
    unseen recommendations for each user.
    """
    if n_factors_list is None:
        n_factors_list = [10, 20, 50, 100]

    U, s, Vt = compute_svd(train_matrix)

    results = []
    for n_factors in n_factors_list:
        if n_factors > len(s):
            continue
        metrics = evaluate_holdout(
            holdout,
            recommend_fn=lambda user_id, n: get_svd_recommendations(
                user_id,
                train_matrix,
                U,
                s,
                Vt,
                k=n_factors,
                n=n,
            ),
            k_values=k_values,
        )
        results.append({"n_factors": n_factors, **metrics})

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
