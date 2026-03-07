"""SQLite database utilities for the recommendation system.

Demonstrates SQL proficiency by loading interaction data into a relational
database and running analytical queries.
"""
from __future__ import annotations

import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "recommendations.db")


def create_database(interactions_df: pd.DataFrame, articles_df: pd.DataFrame,
                    db_path: str = None) -> str:
    """Create SQLite database from DataFrames.

    Args:
        interactions_df: User-article interactions.
        articles_df: Article metadata.
        db_path: Path for the database file.

    Returns:
        Path to the created database.
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)

    conn.execute("DROP TABLE IF EXISTS interactions")
    conn.execute("DROP TABLE IF EXISTS articles")

    conn.execute("""
        CREATE TABLE articles (
            article_id INTEGER PRIMARY KEY,
            doc_full_name TEXT,
            doc_description TEXT,
            doc_status TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            title TEXT,
            email TEXT,
            FOREIGN KEY (article_id) REFERENCES articles(article_id)
        )
    """)

    articles_df[["article_id", "doc_full_name", "doc_description", "doc_status"]].to_sql(
        "articles", conn, if_exists="replace", index=False
    )
    interactions_df.to_sql("interactions", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_email ON interactions(email)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_article ON interactions(article_id)")

    conn.commit()
    conn.close()
    return db_path


def run_query(query: str, db_path: str = None) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Analytical SQL queries demonstrating SQL proficiency
# ---------------------------------------------------------------------------

QUERIES = {
    "top_articles_by_unique_users": """
        SELECT
            i.article_id,
            a.doc_full_name AS title,
            COUNT(DISTINCT i.email) AS unique_users,
            COUNT(*) AS total_interactions
        FROM interactions i
        LEFT JOIN articles a ON i.article_id = a.article_id
        GROUP BY i.article_id, a.doc_full_name
        ORDER BY unique_users DESC
        LIMIT 15
    """,

    "user_engagement_tiers": """
        WITH user_counts AS (
            SELECT email, COUNT(*) AS interaction_count
            FROM interactions
            GROUP BY email
        )
        SELECT
            CASE
                WHEN interaction_count = 1 THEN '1 (one-time)'
                WHEN interaction_count BETWEEN 2 AND 5 THEN '2-5 (casual)'
                WHEN interaction_count BETWEEN 6 AND 20 THEN '6-20 (regular)'
                WHEN interaction_count BETWEEN 21 AND 50 THEN '21-50 (active)'
                ELSE '50+ (power user)'
            END AS engagement_tier,
            COUNT(*) AS user_count,
            ROUND(AVG(interaction_count), 1) AS avg_interactions
        FROM user_counts
        GROUP BY engagement_tier
        ORDER BY MIN(interaction_count)
    """,

    "article_overlap_between_users": """
        WITH user_articles AS (
            SELECT email, article_id
            FROM interactions
            GROUP BY email, article_id
        )
        SELECT
            a.email AS user_a,
            b.email AS user_b,
            COUNT(*) AS shared_articles
        FROM user_articles a
        JOIN user_articles b
            ON a.article_id = b.article_id
            AND a.email < b.email
        GROUP BY a.email, b.email
        HAVING shared_articles >= 5
        ORDER BY shared_articles DESC
        LIMIT 20
    """,

    "article_co_occurrence": """
        WITH user_articles AS (
            SELECT DISTINCT email, article_id
            FROM interactions
        )
        SELECT
            a.article_id AS article_a,
            b.article_id AS article_b,
            COUNT(DISTINCT a.email) AS co_read_users
        FROM user_articles a
        JOIN user_articles b
            ON a.email = b.email
            AND a.article_id < b.article_id
        GROUP BY a.article_id, b.article_id
        HAVING co_read_users >= 10
        ORDER BY co_read_users DESC
        LIMIT 20
    """,

    "interaction_percentiles": """
        WITH user_counts AS (
            SELECT email, COUNT(*) AS cnt
            FROM interactions
            GROUP BY email
        ),
        ranked AS (
            SELECT
                cnt,
                NTILE(4) OVER (ORDER BY cnt) AS quartile
            FROM user_counts
        )
        SELECT
            quartile,
            MIN(cnt) AS min_interactions,
            MAX(cnt) AS max_interactions,
            ROUND(AVG(cnt), 1) AS avg_interactions,
            COUNT(*) AS user_count
        FROM ranked
        GROUP BY quartile
        ORDER BY quartile
    """,
}


def run_all_analytics(db_path: str = None) -> dict:
    """Run all analytical queries and return results as a dict of DataFrames."""
    results = {}
    for name, query in QUERIES.items():
        results[name] = run_query(query, db_path)
    return results
