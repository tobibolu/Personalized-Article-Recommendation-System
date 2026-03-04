"""Rank-based (popularity) recommendation engine.

Recommends the most popular articles to all users. Serves as a baseline
and a cold-start fallback when no user history is available.
"""

import pandas as pd


def get_top_articles(df: pd.DataFrame, n: int = 10) -> list[str]:
    """Return titles of the top-N most popular articles by interaction count."""
    top_ids = df["article_id"].value_counts().head(n).index
    titles = []
    for article_id in top_ids:
        title = df.loc[df["article_id"] == article_id, "title"].iloc[0]
        titles.append(title)
    return titles


def get_top_article_ids(df: pd.DataFrame, n: int = 10) -> list:
    """Return article IDs of the top-N most popular articles."""
    return list(df["article_id"].value_counts().head(n).index)


def recommendation_coverage(df: pd.DataFrame, n: int = 10) -> dict:
    """Analyze how much of total engagement the top-N articles cover.

    Returns:
        Dictionary with n_recommended, total_articles, pct_articles, pct_interactions.
    """
    total_articles = df["article_id"].nunique()
    total_interactions = len(df)
    top_ids = get_top_article_ids(df, n)
    top_interactions = df[df["article_id"].isin(top_ids)].shape[0]

    return {
        "n_recommended": n,
        "total_articles": total_articles,
        "pct_articles": round(n / total_articles * 100, 2),
        "pct_interactions": round(top_interactions / total_interactions * 100, 2),
    }
