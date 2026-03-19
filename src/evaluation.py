"""Evaluation metrics for recommendation systems."""
from __future__ import annotations

import inspect
import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Precision@K: fraction of top-K recommendations that are relevant.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of actually relevant (interacted) item IDs.
        k: Cutoff.
    """
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Recall@K: fraction of relevant items that appear in top-K recommendations."""
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Rewards relevant items appearing earlier in the recommendation list.
    """
    rec_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(rec_k) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(user_item_matrix, recommend_fn,
                             k: int = 10, sample_size: int = 100,
                             seed: int = 42) -> dict:
    """Evaluate a recommendation function using leave-one-out protocol.

    For each sampled user, one interaction is held out and we check
    whether the recommender can recover it.

    Args:
        user_item_matrix: Binary user-item matrix.
        recommend_fn: Callable(user_id) -> list of recommended article IDs.
        k: Evaluation cutoff.
        sample_size: Number of users to evaluate.
        seed: Random seed.

    Returns:
        Dictionary with avg_precision_at_k, avg_recall_at_k, avg_ndcg_at_k.
    """
    def _supports_matrix_arg(func) -> bool:
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return False

        params = list(sig.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return True

        positional = [
            p for p in params
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(positional) >= 2

    def _get_recommendations(func, user_id, eval_matrix):
        if _supports_matrix_arg(func):
            return func(user_id, eval_matrix)
        return func(user_id)

    rng = np.random.RandomState(seed)

    # Filter to users with at least 2 interactions
    user_counts = user_item_matrix.sum(axis=1)
    eligible = user_counts[user_counts >= 2].index
    sample_users = rng.choice(eligible, min(sample_size, len(eligible)), replace=False)

    precisions, recalls, ndcgs = [], [], []

    for user in sample_users:
        user_seen = user_item_matrix.columns[user_item_matrix.loc[user] == 1].tolist()
        held_out = rng.choice(user_seen)
        relevant = {held_out}
        eval_matrix = user_item_matrix.copy()
        eval_matrix.loc[user, held_out] = 0

        try:
            recs = _get_recommendations(recommend_fn, user, eval_matrix)
        except Exception:
            continue

        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))

    return {
        f"avg_precision@{k}": round(np.mean(precisions), 4) if precisions else 0.0,
        f"avg_recall@{k}": round(np.mean(recalls), 4) if recalls else 0.0,
        f"avg_ndcg@{k}": round(np.mean(ndcgs), 4) if ndcgs else 0.0,
        "n_evaluated": len(precisions),
    }
