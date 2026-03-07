# Personalized Article Recommendation System

A multi-method recommendation engine that personalizes article suggestions for users on the IBM Watson Studio platform, improving content discovery through analysis of user behavior data.

## Key Results

| Method | Metric | Score |
|--------|--------|-------|
| Rank-Based (Baseline) | Coverage of interactions | 14.24% (top 10 articles) |
| User-User Collaborative Filtering | Avg similarity score | 0.13 |
| Content-Based (TF-IDF) | Avg cosine similarity | 0.319 |
| Matrix Factorization (SVD, k=100) | RMSE | 0.057 |

**Finding:** Matrix factorization outperforms all other methods, with RMSE decreasing from 0.086 (k=10) to 0.057 (k=100) and increasing coverage. A hybrid approach combining all methods is recommended for production deployment.

## Project Structure

```
├── app/
│   └── streamlit_app.py          # Interactive recommendation dashboard
├── data/
│   ├── README.md                  # Data dictionary
│   ├── user-item-interactions.csv # User interaction log (45,993 records)
│   └── articles_community.csv    # Article metadata (714 unique articles)
├── notebooks/
│   └── recommender.ipynb          # Analysis notebook with narrative
├── sql/
│   ├── create_tables.sql          # Database schema
│   └── queries.sql                # Analytical SQL queries (CTEs, window functions)
├── src/
│   ├── data_loader.py             # Data loading and preprocessing
│   ├── database.py                # SQLite integration and SQL analytics
│   ├── rank_recommender.py        # Popularity-based recommendations
│   ├── collaborative.py           # User-user collaborative filtering
│   ├── content_based.py           # TF-IDF content similarity
│   ├── matrix_factorization.py    # SVD-based recommendations
│   └── evaluation.py              # Precision@K, Recall@K, NDCG metrics
├── tests/
│   ├── test_data_loader.py        # Data validation tests
│   └── test_recommenders.py       # Recommendation engine tests
├── requirements.txt
├── Makefile
└── .github/workflows/ci.yml      # Automated linting and testing
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
make test

# Launch interactive dashboard
make run-app
```

## Methods

### 1. Rank-Based Recommendations (Baseline)
Recommends the most popular articles to all users. Simple but effective for cold-start users with no interaction history.

### 2. User-User Collaborative Filtering
Identifies users with similar reading patterns and recommends articles that similar users have read. Personalized but limited by the sparse user-item matrix (99%+ sparsity).

### 3. Content-Based Filtering (TF-IDF)
Uses article titles and descriptions to compute text similarity via TF-IDF vectorization and cosine similarity. Addresses the cold-start problem for new articles.

### 4. Matrix Factorization (SVD)
Decomposes the user-item matrix into latent factors capturing hidden patterns. Achieves the best predictive performance (lowest RMSE) and is the primary recommendation engine.

## SQL Analytics

The project includes a SQLite integration (`src/database.py`) with analytical queries demonstrating:
- **CTEs** for user engagement tiering
- **Window functions** (NTILE, LAG, LEAD) for interaction analysis
- **Self-joins** for article co-occurrence and user overlap
- **Aggregations** with HAVING for filtered group analysis

See [`sql/queries.sql`](sql/queries.sql) for the full query set.

## Evaluation

All four methods are compared using consistent metrics:
- **Precision@10 / Recall@10** for recommendation relevance
- **NDCG@10** for ranking quality
- **RMSE** for matrix factorization prediction accuracy
- **Coverage** for recommendation diversity

Evaluation uses a leave-one-out holdout protocol (one hidden interaction per user) to avoid train/evaluation leakage.

| Method | Precision@10 | Recall@10 | NDCG@10 |
|--------|-------------|-----------|---------|
| Rank-Based | 0.006 | 0.060 | 0.033 |
| Collaborative | 0.046 | 0.460 | 0.369 |
| SVD (k=50) | 0.078 | 0.780 | 0.684 |

## Dataset

- **45,993** user-article interactions from **5,148** unique users across **714** articles
- Median user reads 3 articles; long-tail distribution with power users reading 300+
- User emails are SHA-1 hashed for privacy

## Tech Stack

Python 3.11 | pandas | scikit-learn | NumPy | Matplotlib | Seaborn | Streamlit | SQLite | pytest

## Live Demo

**[Launch Interactive Dashboard](https://personalized-article-recommendation-system-vbounmhgyeszgstw62m.streamlit.app/)**

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
