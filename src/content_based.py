"""Content-based recommendation engine using TF-IDF similarity.

Recommends articles similar to a given article based on textual features
(title + description). Helps mitigate the cold-start problem for new articles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_content_matrix(articles_df: pd.DataFrame,
                          max_features: int = 5000) -> tuple:
    """Build a TF-IDF matrix from article text.

    Args:
        articles_df: Article content DataFrame (must have doc_full_name, doc_description).
        max_features: Maximum number of TF-IDF features.

    Returns:
        Tuple of (tfidf_matrix, fitted TfidfVectorizer).
    """
    text = (
        articles_df["doc_full_name"].fillna("")
        + " "
        + articles_df["doc_description"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    matrix = tfidf.fit_transform(text)
    return matrix, tfidf


def find_similar_articles(article_id, articles_df: pd.DataFrame,
                          content_matrix, n: int = 5) -> pd.DataFrame:
    """Find the N most similar articles to a given article.

    Args:
        article_id: The source article ID.
        articles_df: Article metadata DataFrame (with reset index).
        content_matrix: Pre-computed TF-IDF matrix.
        n: Number of similar articles to return.

    Returns:
        DataFrame of similar articles with similarity scores.
    """
    matches = articles_df[articles_df["article_id"] == int(article_id)]
    if matches.empty:
        return pd.DataFrame(columns=list(articles_df.columns) + ["similarity_score"])
    idx = matches.index[0]
    sims = cosine_similarity(content_matrix[idx : idx + 1], content_matrix).flatten()
    similar_indices = sims.argsort()[::-1][1 : n + 1]

    result = articles_df.iloc[similar_indices].copy()
    result["similarity_score"] = sims[similar_indices]
    return result


def evaluate_content_recommendations(articles_df: pd.DataFrame,
                                     sample_size: int = 100,
                                     seed: int = 42) -> dict:
    """Evaluate content-based recommendations on a random sample of articles.

    Returns:
        Dictionary with avg_similarity, median_similarity.
    """
    content_matrix, _ = create_content_matrix(articles_df)
    rng = np.random.RandomState(seed)
    sample_articles = rng.choice(articles_df["article_id"].values, sample_size)

    similarities = []
    for aid in sample_articles:
        similar = find_similar_articles(aid, articles_df, content_matrix, n=5)
        similarities.extend(similar["similarity_score"].tolist())

    return {
        "avg_similarity": round(np.mean(similarities), 3),
        "median_similarity": round(np.median(similarities), 3),
    }
