-- =============================================================================
-- Analytical SQL Queries for the Article Recommendation System
-- Demonstrates: CTEs, window functions, aggregations, joins, subqueries
-- =============================================================================

-- 1. Top 15 articles by unique user reach
SELECT
    i.article_id,
    a.doc_full_name                    AS title,
    COUNT(DISTINCT i.email)            AS unique_users,
    COUNT(*)                           AS total_interactions
FROM interactions i
LEFT JOIN articles a ON i.article_id = a.article_id
GROUP BY i.article_id, a.doc_full_name
ORDER BY unique_users DESC
LIMIT 15;


-- 2. User engagement tiers using CTEs
WITH user_counts AS (
    SELECT email, COUNT(*) AS interaction_count
    FROM interactions
    GROUP BY email
)
SELECT
    CASE
        WHEN interaction_count = 1             THEN '1 (one-time)'
        WHEN interaction_count BETWEEN 2 AND 5 THEN '2-5 (casual)'
        WHEN interaction_count BETWEEN 6 AND 20  THEN '6-20 (regular)'
        WHEN interaction_count BETWEEN 21 AND 50 THEN '21-50 (active)'
        ELSE '50+ (power user)'
    END AS engagement_tier,
    COUNT(*)                   AS user_count,
    ROUND(AVG(interaction_count), 1) AS avg_interactions
FROM user_counts
GROUP BY engagement_tier
ORDER BY MIN(interaction_count);


-- 3. Article co-occurrence matrix (articles frequently read together)
WITH user_articles AS (
    SELECT DISTINCT email, article_id
    FROM interactions
)
SELECT
    a.article_id   AS article_a,
    b.article_id   AS article_b,
    COUNT(DISTINCT a.email) AS co_read_users
FROM user_articles a
JOIN user_articles b
    ON a.email = b.email
    AND a.article_id < b.article_id
GROUP BY a.article_id, b.article_id
HAVING co_read_users >= 10
ORDER BY co_read_users DESC
LIMIT 20;


-- 4. User interaction percentiles using window functions
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
    MIN(cnt)           AS min_interactions,
    MAX(cnt)           AS max_interactions,
    ROUND(AVG(cnt), 1) AS avg_interactions,
    COUNT(*)           AS user_count
FROM ranked
GROUP BY quartile
ORDER BY quartile;


-- 5. Rolling engagement: articles gaining traction (would use date if available)
-- Demonstrates LAG/LEAD window functions with interaction ordering
WITH article_rank AS (
    SELECT
        article_id,
        COUNT(*) AS interactions,
        ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS popularity_rank,
        LAG(COUNT(*)) OVER (ORDER BY COUNT(*) DESC) AS prev_article_count,
        LEAD(COUNT(*)) OVER (ORDER BY COUNT(*) DESC) AS next_article_count
    FROM interactions
    GROUP BY article_id
)
SELECT
    article_id,
    interactions,
    popularity_rank,
    prev_article_count,
    next_article_count,
    interactions - COALESCE(next_article_count, 0) AS gap_to_next
FROM article_rank
LIMIT 20;
