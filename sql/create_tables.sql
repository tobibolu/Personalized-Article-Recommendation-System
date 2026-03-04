-- Schema for the Article Recommendation System
-- Database: SQLite

CREATE TABLE IF NOT EXISTS articles (
    article_id   INTEGER PRIMARY KEY,
    doc_full_name   TEXT NOT NULL,
    doc_description TEXT,
    doc_status      TEXT DEFAULT 'Live'
);

CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id  INTEGER NOT NULL,
    title       TEXT,
    email       TEXT NOT NULL,
    FOREIGN KEY (article_id) REFERENCES articles(article_id)
);

CREATE INDEX IF NOT EXISTS idx_interactions_email    ON interactions(email);
CREATE INDEX IF NOT EXISTS idx_interactions_article  ON interactions(article_id);
