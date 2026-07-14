"""User-user collaborative filtering recommendation engine.

Recommends articles based on what similar users have read.
"""

import numpy as np
import pandas as pd


def find_similar_users(user_id: str, user_item_matrix: pd.DataFrame) -> pd.Series:
    """Find users most similar to the given user based on dot-product similarity.

    Args:
        user_id: Email hash identifying the user.
        user_item_matrix: Binary user-item matrix.

    Returns:
        Series of similarity scores, sorted descending (excluding the target user).
    """
    interacted = user_item_matrix.columns[user_item_matrix.loc[user_id] == 1]
    similarities = user_item_matrix.loc[:, interacted].sum(axis=1)
    return similarities.sort_values(ascending=False).drop(user_id)


def get_user_articles(user_id: str, user_item_matrix: pd.DataFrame) -> list:
    """Return article IDs that a user has interacted with."""
    user_row = user_item_matrix.loc[user_id]
    return list(user_row[user_row == 1].index)


def user_user_recs(user_id: str, user_item_matrix: pd.DataFrame, m: int = 10) -> list:
    """Generate recommendations using user-user collaborative filtering.

    Args:
        user_id: Target user email hash.
        user_item_matrix: Binary user-item matrix.
        m: Number of recommendations to return.

    Returns:
        List of recommended article IDs.
    """
    similarities = find_similar_users(user_id, user_item_matrix)
    similarities = similarities[similarities > 0].head(100)
    if similarities.empty:
        return []

    seen = set(get_user_articles(user_id, user_item_matrix))
    neighbour_matrix = user_item_matrix.loc[similarities.index]

    # Articles read by more-similar neighbours receive more weight. Popularity
    # and article ID provide deterministic tie-breaks.
    weighted_score = neighbour_matrix.T.dot(similarities)
    support = neighbour_matrix.sum(axis=0)
    candidates = [
        article_id
        for article_id in user_item_matrix.columns
        if article_id not in seen and weighted_score.loc[article_id] > 0
    ]
    candidates.sort(
        key=lambda article_id: (
            -weighted_score.loc[article_id],
            -support.loc[article_id],
            article_id,
        )
    )
    return candidates[:m]


def evaluate_collaborative(user_item_matrix: pd.DataFrame,
                           sample_size: int = 100,
                           seed: int = 42) -> dict:
    """Evaluate user-user CF on a random sample of users.

    Returns:
        Dictionary with avg_recommendations, avg_similarity_score.
    """
    rng = np.random.RandomState(seed)
    sample_users = rng.choice(user_item_matrix.index, min(sample_size, len(user_item_matrix.index)), replace=False)

    coverage = []
    avg_similarity = []

    for user in sample_users:
        recs = user_user_recs(user, user_item_matrix, m=10)
        coverage.append(len(recs))
        if recs:
            similar_users = find_similar_users(user, user_item_matrix).index.tolist()
            sim_scores = []
            for rec in recs:
                if rec in user_item_matrix.columns:
                    count = sum(1 for u in similar_users[:50] if user_item_matrix.loc[u, rec] == 1)
                    sim_scores.append(count / min(50, len(similar_users)))
            if sim_scores:
                avg_similarity.append(np.mean(sim_scores))

    return {
        "avg_recommendations": round(np.mean(coverage), 2),
        "avg_similarity_score": round(np.mean(avg_similarity), 4) if avg_similarity else 0.0,
    }
