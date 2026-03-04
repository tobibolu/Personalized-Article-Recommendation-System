# Data

This directory contains the datasets used by the recommendation system.

## Files

| File | Description | Size | Rows |
|------|-------------|------|------|
| `user-item-interactions.csv` | User-article interaction log (article_id, title, email) | 4.4 MB | 46,811 |
| `articles_community.csv` | Article metadata and content (body, description, title, status) | 8.9 MB | ~169K |

## Data Dictionary

### user-item-interactions.csv

| Column | Type | Description |
|--------|------|-------------|
| `article_id` | float | Unique article identifier |
| `title` | string | Article title |
| `email` | string | SHA-1 hashed user email (privacy-protected) |

### articles_community.csv

| Column | Type | Description |
|--------|------|-------------|
| `doc_body` | string | Full article text/HTML content |
| `doc_description` | string | Short article description |
| `doc_full_name` | string | Article title |
| `doc_status` | string | Publication status (e.g., "Live") |
| `article_id` | integer | Unique article identifier |

## Data Source

Data originates from the IBM Watson Studio platform. It captures anonymized user
interactions with technical articles hosted on the platform.

## Notes

- 5 duplicate articles exist in the raw articles file and are removed during loading.
- User emails are SHA-1 hashed for privacy.
- The `recommendations.db` SQLite file is generated at runtime by `src/database.py`.
