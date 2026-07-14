"""Leakage-safe ranking evaluation for recommendation systems.

The evaluation unit is an unseen user-item interaction. For every eligible
user, one article is removed from a copy of the interaction matrix. Models
must train and recommend from that reduced matrix, and are scored on whether
the hidden article appears near the top of the recommendation list.
"""

from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of the first ``k`` slots containing a relevant item."""
    if k <= 0:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items recovered in the first ``k`` slots."""
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Position-sensitive ranking quality, normalised to the range [0, 1]."""
    recommended_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(position + 2)
        for position, item in enumerate(recommended_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(position + 2) for position in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def leave_one_out_split(
    user_item_matrix: pd.DataFrame,
    min_interactions: int = 2,
    seed: int = 42,
    sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Hide one unique article interaction from every eligible user.

    Parameters
    ----------
    user_item_matrix:
        Binary user-item matrix.
    min_interactions:
        Minimum number of unique interacted articles required before hiding one.
    seed:
        Controls both optional user sampling and held-out article selection.
    sample_size:
        Optional number of eligible users to evaluate. ``None`` uses all.

    Returns
    -------
    train_matrix, holdout
        ``train_matrix`` is a copy with each hidden cell set to zero. ``holdout``
        maps each evaluated user ID to a one-item set containing the hidden article.
    """
    if min_interactions < 2:
        raise ValueError("min_interactions must be at least 2 for leave-one-out evaluation")

    values = user_item_matrix.to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError("user_item_matrix must contain binary 0/1 values")

    counts = user_item_matrix.sum(axis=1)
    eligible_users = counts[counts >= min_interactions].index.to_numpy()
    rng = np.random.RandomState(seed)

    if sample_size is not None and sample_size < len(eligible_users):
        eligible_users = rng.choice(eligible_users, size=sample_size, replace=False)

    train_matrix = user_item_matrix.copy()
    holdout = {}

    for user_id in eligible_users:
        interacted = user_item_matrix.columns[
            user_item_matrix.loc[user_id].to_numpy() == 1
        ].to_numpy()
        hidden_article = rng.choice(interacted)
        train_matrix.loc[user_id, hidden_article] = 0
        holdout[user_id] = {hidden_article}

    return train_matrix, holdout


def evaluate_holdout(
    holdout: dict,
    recommend_fn: Callable[[object, int], list],
    k_values: Iterable[int] = (5, 10),
) -> dict:
    """Evaluate recommendations against genuinely unseen interactions.

    ``recommend_fn`` must accept ``(user_id, n)`` and return an ordered list.
    With one hidden relevant item per user, Recall@K is also Hit Rate@K.
    """
    cutoffs = sorted(set(int(k) for k in k_values))
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("k_values must contain positive integers")

    totals = {
        metric: {k: [] for k in cutoffs}
        for metric in ("precision", "recall", "ndcg")
    }
    max_k = max(cutoffs)

    for user_id, relevant in holdout.items():
        recommendations = list(dict.fromkeys(recommend_fn(user_id, max_k)))
        for k in cutoffs:
            totals["precision"][k].append(precision_at_k(recommendations, relevant, k))
            totals["recall"][k].append(recall_at_k(recommendations, relevant, k))
            totals["ndcg"][k].append(ndcg_at_k(recommendations, relevant, k))

    result = {"n_evaluated": len(holdout)}
    for metric, by_cutoff in totals.items():
        for k, values_at_k in by_cutoff.items():
            result[f"{metric}@{k}"] = float(np.mean(values_at_k)) if values_at_k else 0.0
    return result
