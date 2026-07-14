"""Content-based recommendation engine using TF-IDF similarity.

Recommends articles similar to a given article based on textual features
(title + description). Helps mitigate the cold-start problem for new articles.
"""

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


def create_content_similarity(content_matrix) -> np.ndarray:
    """Create a dense article-to-article cosine-similarity matrix."""
    return cosine_similarity(content_matrix)


def get_user_content_recs(user_id, user_item_matrix: pd.DataFrame,
                          articles_df: pd.DataFrame, similarity_matrix: np.ndarray,
                          n: int = 10) -> list:
    """Recommend unseen articles similar to a user's training history.

    Candidate scores are the mean cosine similarity to the articles still
    visible in the leave-one-out training matrix.
    """
    article_ids = articles_df["article_id"].astype(int).tolist()
    position = {article_id: idx for idx, article_id in enumerate(article_ids)}
    seen = set(user_item_matrix.columns[user_item_matrix.loc[user_id] == 1])
    profile_positions = [position[int(article_id)] for article_id in seen if int(article_id) in position]
    if not profile_positions:
        return []

    scores = similarity_matrix[profile_positions].mean(axis=0)
    available = set(int(article_id) for article_id in user_item_matrix.columns)
    candidates = [
        article_id for article_id in article_ids
        if article_id in available and article_id not in seen
    ]
    candidates.sort(key=lambda article_id: (-scores[position[article_id]], article_id))
    return candidates[:n]


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
