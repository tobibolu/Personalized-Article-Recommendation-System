# Personalized Article Recommendation System

A multi-method recommendation engine for IBM Watson Studio articles, with popularity, user-user collaborative filtering, TF-IDF content similarity, and truncated SVD.

The project’s central technical result is a leakage-safe offline evaluation: one article is hidden from each sampled user **before** interaction-based recommendations are generated, and models are scored on whether they recover that unseen article near the top of the list.

## Verified offline results

All methods use the same reproducible leave-one-out split of 1,000 users sampled from 3,591 users with at least two unique article interactions.

| Method | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Popularity baseline | 0.061 | 0.034 | 0.104 | 0.048 |
| User-user collaborative filtering | 0.102 | 0.071 | 0.154 | 0.088 |
| TF-IDF content-based | 0.015 | 0.010 | 0.027 | 0.014 |
| SVD, 50 factors | **0.124** | **0.088** | **0.192** | **0.110** |

SVD with 50 factors recovers 19.2% of hidden articles within ten recommendations, compared with 10.4% for popularity, an **8.8 percentage-point offline Recall@10 improvement** on this fixed holdout.

This is evidence of better offline ranking, not proof of user-engagement or revenue lift. Absolute recall remains modest, so SVD is an experiment candidate rather than a production-deployment claim.

## Why the evaluation changed

The earlier project reported SVD reconstruction RMSE on the same sparse matrix used to fit the factorisation. That is not a prospective recommender test: most user-item cells are zero, and reproducing already-observed zeros and ones does not show that the system can rank a user’s next relevant article. The earlier “leave-one-out” helper also left every interaction visible.

The corrected protocol:

1. Builds a binary matrix from unique user-article interactions.
2. Keeps users with at least two unique articles.
3. Reproducibly samples 1,000 eligible users with seed 42.
4. Removes one article from each sampled user in a copy of the matrix.
5. Fits/generates interaction-based recommendations from that reduced matrix only.
6. Excludes the user’s visible training history from every ranked list.
7. Measures Precision@K, Recall@K, and NDCG@K against only the hidden article.

With one relevant hidden item per user, Recall@K is also Hit Rate@K. NDCG@K additionally rewards putting the hidden article closer to rank 1.

## Project structure

```text
├── app/
│   └── streamlit_app.py           # Interactive recommendation dashboard
├── data/
│   ├── README.md
│   ├── user-item-interactions.csv # 45,993 interaction rows
│   └── articles_community.csv
├── notebooks/
│   └── recommender.ipynb          # Executed analysis and evaluation
├── sql/
│   ├── create_tables.sql
│   └── queries.sql
├── src/
│   ├── data_loader.py
│   ├── database.py
│   ├── rank_recommender.py
│   ├── collaborative.py
│   ├── content_based.py
│   ├── matrix_factorization.py
│   └── evaluation.py
├── tests/
│   ├── test_data_loader.py
│   └── test_recommenders.py
├── evaluation_metrics.json        # Machine-readable notebook results
├── requirements.txt
├── Makefile
└── .github/workflows/ci.yml
```

## Recommendation methods

### Popularity baseline

Ranks articles by training interaction count and removes articles already visible in the target user’s history. It provides a simple cold-start floor.

### User-user collaborative filtering

Finds users with overlapping visible histories, then ranks unseen articles by similarity-weighted neighbour support. Deterministic popularity and article-ID tie-breaks make repeated runs reproducible.

### Content-based filtering

Builds TF-IDF vectors from article names and descriptions. A user’s candidates are scored by mean cosine similarity to the articles in their visible training history. Its weak holdout result shows that textual similarity alone is not a strong proxy for interaction relevance in this dataset.

### Matrix factorisation

Factorises the leave-one-out training matrix with SVD and ranks articles the user has not seen. Factor counts of 10, 20, 50, and 100 are compared using ranking metrics; 50 factors gives the best Recall@10 on the fixed evaluation split.

## SQL analytics

The SQLite component demonstrates:

- CTEs for user engagement tiers;
- `NTILE`, `LAG`, and `LEAD` window functions;
- self-joins for article co-occurrence and user overlap;
- filtered aggregations with `HAVING`.

See [`sql/queries.sql`](sql/queries.sql).

## Dataset

- 45,993 interaction rows
- 5,148 unique users
- 714 interacted articles
- 3,591 users eligible for leave-one-out evaluation
- Median user history: 3 unique articles
- User emails are SHA-1 hashed

## Reproduce

```bash
pip install -r requirements.txt
pytest -q
jupyter notebook notebooks/recommender.ipynb
```

Executing the notebook regenerates `evaluation_metrics.json`.

Launch the dashboard with:

```bash
streamlit run app/streamlit_app.py
```

## Production next steps

- Repeat the split across several seeds and report uncertainty intervals.
- Use an out-of-time split when timestamps are available.
- Tune hybrid weights on validation data while preserving a final test set.
- Add catalog coverage, novelty, and diversity to the relevance metrics.
- A/B test SVD against popularity for returning users with a pre-declared engagement metric.

## Live demo

[Launch the Streamlit dashboard](https://personalized-article-recommendation-system-vbounmhgyeszgstw62m.streamlit.app/)

## License

Licensed under the MIT License. See [`LICENSE`](LICENSE).
