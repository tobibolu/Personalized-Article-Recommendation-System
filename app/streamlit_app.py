"""Interactive Article Recommendation Dashboard.

Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_interactions, load_articles, create_user_item_matrix, compute_interaction_stats
from src.rank_recommender import get_top_articles, get_top_article_ids
from src.collaborative import user_user_recs
from src.content_based import create_content_matrix, find_similar_articles
from src.matrix_factorization import compute_svd, get_svd_recommendations

st.set_page_config(page_title="Article Recommender", page_icon="📚", layout="wide")


@st.cache_data
def load_data():
    interactions = load_interactions()
    articles = load_articles()
    return interactions, articles


@st.cache_data
def build_matrices(_interactions, _articles):
    user_item = create_user_item_matrix(_interactions)
    content_matrix, tfidf = create_content_matrix(_articles)
    U, s, Vt = compute_svd(user_item)
    return user_item, content_matrix, tfidf, U, s, Vt


def main():
    st.title("📚 Article Recommendation Engine")
    st.markdown(
        "Explore personalized article recommendations using multiple algorithms. "
        "Select a user or article to see recommendations from different methods."
    )

    interactions, articles = load_data()
    user_item, content_matrix, tfidf, U, s, Vt = build_matrices(interactions, articles)

    # Sidebar: stats
    st.sidebar.header("Dataset Overview")
    stats = compute_interaction_stats(interactions)
    for key, val in stats.items():
        st.sidebar.metric(key.replace("_", " ").title(), f"{val:,}" if isinstance(val, int) else f"{val}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔥 Popular Articles", "👤 User Recommendations", "📄 Similar Articles"])

    with tab1:
        st.subheader("Most Popular Articles (Rank-Based)")
        n = st.slider("Number of articles", 5, 30, 10, key="rank_n")
        top_titles = get_top_articles(interactions, n)
        top_ids = get_top_article_ids(interactions, n)
        for i, (title, aid) in enumerate(zip(top_titles, top_ids), 1):
            st.write(f"**{i}.** {title} (ID: {int(aid)})")

    with tab2:
        st.subheader("Personalized Recommendations")
        users = sorted(user_item.index.tolist())
        user_id = st.selectbox("Select a user (email hash)", users[:200],
                               help="Showing first 200 users for performance")
        n_recs = st.slider("Number of recommendations", 5, 20, 10, key="user_n")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Collaborative Filtering**")
            cf_recs = user_user_recs(user_id, user_item, m=n_recs)
            if cf_recs:
                for i, aid in enumerate(cf_recs, 1):
                    title_match = articles.loc[articles["article_id"] == aid, "doc_full_name"]
                    title = title_match.iloc[0] if len(title_match) > 0 else f"Article {int(aid)}"
                    st.write(f"{i}. {title}")
            else:
                st.info("No collaborative filtering recommendations available for this user.")

        with col2:
            st.markdown("**Matrix Factorization (SVD)**")
            svd_recs = get_svd_recommendations(user_id, user_item, U, s, Vt, k=50, n=n_recs)
            for i, aid in enumerate(svd_recs, 1):
                title_match = articles.loc[articles["article_id"] == aid, "doc_full_name"]
                title = title_match.iloc[0] if len(title_match) > 0 else f"Article {int(aid)}"
                st.write(f"{i}. {title}")

    with tab3:
        st.subheader("Content-Based Similarity")
        article_options = articles[["article_id", "doc_full_name"]].dropna()
        selected = st.selectbox(
            "Select an article",
            article_options["article_id"].tolist()[:200],
            format_func=lambda x: f"{articles.loc[articles['article_id']==x, 'doc_full_name'].iloc[0][:80]}..."
        )
        if selected is not None:
            similar = find_similar_articles(selected, articles, content_matrix, n=10)
            for _, row in similar.iterrows():
                st.write(f"- **{row['doc_full_name']}** (similarity: {row['similarity_score']:.3f})")


if __name__ == "__main__":
    main()
