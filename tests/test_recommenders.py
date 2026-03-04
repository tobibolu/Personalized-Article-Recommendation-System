"""Tests for recommendation engines."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_interactions, load_articles, create_user_item_matrix
from src.rank_recommender import get_top_articles, get_top_article_ids, recommendation_coverage
from src.content_based import create_content_matrix, find_similar_articles
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@pytest.fixture(scope="module")
def interactions_df():
    return load_interactions(os.path.join(DATA_DIR, "user-item-interactions.csv"))


@pytest.fixture(scope="module")
def articles_df():
    return load_articles(os.path.join(DATA_DIR, "articles_community.csv"))


class TestRankRecommender:
    def test_returns_correct_count(self, interactions_df):
        titles = get_top_articles(interactions_df, n=5)
        assert len(titles) == 5

    def test_returns_strings(self, interactions_df):
        titles = get_top_articles(interactions_df, n=3)
        assert all(isinstance(t, str) for t in titles)

    def test_ids_match_count(self, interactions_df):
        ids = get_top_article_ids(interactions_df, n=10)
        assert len(ids) == 10

    def test_coverage_returns_dict(self, interactions_df):
        cov = recommendation_coverage(interactions_df, n=10)
        assert "pct_articles" in cov
        assert "pct_interactions" in cov


class TestContentBased:
    def test_content_matrix_shape(self, articles_df):
        matrix, tfidf = create_content_matrix(articles_df)
        assert matrix.shape[0] == len(articles_df)
        assert matrix.shape[1] <= 5000

    def test_similar_articles_count(self, articles_df):
        matrix, _ = create_content_matrix(articles_df)
        first_id = articles_df["article_id"].iloc[0]
        similar = find_similar_articles(first_id, articles_df, matrix, n=5)
        assert len(similar) == 5

    def test_similarity_scores_range(self, articles_df):
        matrix, _ = create_content_matrix(articles_df)
        first_id = articles_df["article_id"].iloc[0]
        similar = find_similar_articles(first_id, articles_df, matrix, n=5)
        assert all(0 <= s <= 1 for s in similar["similarity_score"])


class TestEvaluationMetrics:
    def test_precision_perfect(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_precision_none(self):
        assert precision_at_k([1, 2, 3], {4, 5}, k=3) == 0.0

    def test_recall_perfect(self):
        assert recall_at_k([1, 2, 3, 4], {1, 2}, k=4) == 1.0

    def test_recall_partial(self):
        assert recall_at_k([1, 4, 5], {1, 2, 3}, k=3) == pytest.approx(1 / 3)

    def test_ndcg_perfect(self):
        assert ndcg_at_k([1, 2], {1, 2}, k=2) == 1.0

    def test_ndcg_zero(self):
        assert ndcg_at_k([3, 4], {1, 2}, k=2) == 0.0
