"""Tests for data loading and preprocessing utilities."""

import os
import pytest
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import (
    load_interactions,
    load_articles,
    compute_interaction_stats,
    create_user_item_matrix,
    get_sparsity,
)


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "user-item-interactions.csv")
ARTICLES_PATH = os.path.join(DATA_DIR, "articles_community.csv")


@pytest.fixture
def interactions_df():
    return load_interactions(INTERACTIONS_PATH)


@pytest.fixture
def articles_df():
    return load_articles(ARTICLES_PATH)


class TestLoadInteractions:
    def test_loads_dataframe(self, interactions_df):
        assert isinstance(interactions_df, pd.DataFrame)

    def test_has_required_columns(self, interactions_df):
        assert set(interactions_df.columns) >= {"article_id", "title", "email"}

    def test_no_unnamed_columns(self, interactions_df):
        assert not any("Unnamed" in col for col in interactions_df.columns)

    def test_not_empty(self, interactions_df):
        assert len(interactions_df) > 0


class TestLoadArticles:
    def test_loads_dataframe(self, articles_df):
        assert isinstance(articles_df, pd.DataFrame)

    def test_no_duplicate_article_ids(self, articles_df):
        assert articles_df["article_id"].is_unique

    def test_has_required_columns(self, articles_df):
        required = {"article_id", "doc_full_name", "doc_description"}
        assert required <= set(articles_df.columns)


class TestInteractionStats:
    def test_returns_dict(self, interactions_df):
        stats = compute_interaction_stats(interactions_df)
        assert isinstance(stats, dict)

    def test_stats_keys(self, interactions_df):
        stats = compute_interaction_stats(interactions_df)
        expected_keys = {
            "total_interactions", "unique_users", "unique_articles",
            "avg_interactions", "median_interactions", "max_interactions",
        }
        assert expected_keys == set(stats.keys())

    def test_positive_values(self, interactions_df):
        stats = compute_interaction_stats(interactions_df)
        for v in stats.values():
            assert v > 0


class TestUserItemMatrix:
    def test_binary_values(self, interactions_df):
        matrix = create_user_item_matrix(interactions_df)
        unique_vals = set(matrix.values.flatten())
        assert unique_vals <= {0, 1}

    def test_sparsity_in_range(self, interactions_df):
        matrix = create_user_item_matrix(interactions_df)
        sparsity = get_sparsity(matrix)
        assert 0 < sparsity < 100
