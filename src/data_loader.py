"""Data loading, cleaning, and preprocessing utilities."""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_interactions(path: str = None) -> pd.DataFrame:
    """Load and clean user-article interaction data.

    Args:
        path: Path to CSV file. Defaults to data/user-item-interactions.csv.

    Returns:
        Cleaned DataFrame with columns: article_id, title, email.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "user-item-interactions.csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["article_id"] = df["article_id"].astype(int)
    return df


def load_articles(path: str = None) -> pd.DataFrame:
    """Load and deduplicate article content data.

    Args:
        path: Path to CSV file. Defaults to data/articles_community.csv.

    Returns:
        Deduplicated DataFrame with article metadata.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "articles_community.csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    return df


def compute_interaction_stats(df: pd.DataFrame) -> dict:
    """Compute key interaction metrics from the interactions DataFrame.

    Returns:
        Dictionary with total_interactions, unique_users, unique_articles,
        avg_interactions, median_interactions, max_interactions.
    """
    interactions_per_user = df.groupby("email")["article_id"].count()
    return {
        "total_interactions": len(df),
        "unique_users": df["email"].nunique(),
        "unique_articles": df["article_id"].nunique(),
        "avg_interactions": round(interactions_per_user.mean(), 2),
        "median_interactions": interactions_per_user.median(),
        "max_interactions": interactions_per_user.max(),
    }


def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary user-item interaction matrix.

    Args:
        df: Interactions DataFrame with email and article_id columns.

    Returns:
        Binary DataFrame where rows=users (email), columns=article_ids,
        values=1 if user interacted with article, 0 otherwise.
    """
    user_item = df.groupby(["email", "article_id"])["title"].max().unstack()
    user_item = user_item.notnull().astype(int)
    return user_item


def get_sparsity(user_item_matrix: pd.DataFrame) -> float:
    """Calculate sparsity percentage of the user-item matrix."""
    total = user_item_matrix.shape[0] * user_item_matrix.shape[1]
    nonzero = np.count_nonzero(user_item_matrix)
    return (total - nonzero) / total * 100
