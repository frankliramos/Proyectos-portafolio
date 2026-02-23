"""
Content-based filtering module using TF-IDF and cosine similarity.

Recommends products similar to those a user has already interacted with,
based on product metadata (name, description, category, brand).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import joblib

from .config import CONTENT_CONFIG, MODELS_DIR, ITEM_COL


class ContentFilter:
    """
    Content-based recommendation engine using TF-IDF feature vectors.

    Solves the cold-start problem for new users by relying on product
    metadata rather than historical interaction data.
    """

    def __init__(
        self,
        max_features: int = None,
        ngram_range: tuple = None,
        category_weight: float = 0.3,
        brand_weight: float = 0.2,
        text_weight: float = 0.5,
    ):
        cfg = CONTENT_CONFIG
        self.max_features = max_features or cfg["max_features"]
        self.ngram_range = ngram_range or cfg["ngram_range"]
        self.category_weight = category_weight
        self.brand_weight = brand_weight
        self.text_weight = text_weight

        self.vectorizer: TfidfVectorizer = None
        self.tfidf_matrix = None
        self.product_index: dict = {}
        self.products_df: pd.DataFrame = None

    def _build_corpus(self, products_df: pd.DataFrame) -> pd.Series:
        """Combine text fields into a single weighted document per product."""
        corpus = (
            products_df.get("product_name", "").fillna("") + " "
            + products_df.get("product_description", "").fillna("") + " "
            + (products_df.get("product_category", "").fillna("") + " ") * 3
            + (products_df.get("brand", "").fillna("") + " ") * 2
        )
        return corpus.str.lower().str.strip()

    def fit(self, products_df: pd.DataFrame) -> "ContentFilter":
        """
        Build TF-IDF vectors from product metadata.

        Args:
            products_df: DataFrame with product columns (product_id, product_name,
                         product_description, product_category, brand)

        Returns:
            self
        """
        self.products_df = products_df.reset_index(drop=True).copy()
        self.product_index = {
            pid: idx
            for idx, pid in enumerate(self.products_df[ITEM_COL])
        }

        corpus = self._build_corpus(self.products_df)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=CONTENT_CONFIG.get("min_df", 1),
            max_df=CONTENT_CONFIG.get("max_df", 0.8),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        return self

    def find_similar(
        self, product_id: str, n_similar: int = 10
    ) -> pd.DataFrame:
        """
        Find products similar to a given product.

        Args:
            product_id: Reference product identifier
            n_similar: Number of similar products to return

        Returns:
            DataFrame with columns [product_id, similarity, ...]
        """
        if product_id not in self.product_index:
            return pd.DataFrame(columns=[ITEM_COL, "similarity"])

        idx = self.product_index[product_id]
        product_vec = self.tfidf_matrix[idx]
        sim_scores = cosine_similarity(product_vec, self.tfidf_matrix).flatten()
        sim_scores[idx] = -1  # exclude the product itself

        top_indices = np.argsort(sim_scores)[::-1][:n_similar]
        result = self.products_df.iloc[top_indices].copy()
        result["similarity"] = sim_scores[top_indices]
        return result

    def recommend_from_history(
        self,
        interacted_product_ids: list,
        n_recommendations: int = 10,
        exclude_ids: set = None,
    ) -> pd.DataFrame:
        """
        Recommend products based on a user's interaction history.

        Computes the mean TF-IDF vector of interacted products and returns
        the most similar unseen items.

        Args:
            interacted_product_ids: List of product IDs the user has seen
            n_recommendations: Number of recommendations to return
            exclude_ids: Additional product IDs to exclude

        Returns:
            DataFrame with product details and similarity score
        """
        valid_indices = [
            self.product_index[pid]
            for pid in interacted_product_ids
            if pid in self.product_index
        ]
        if not valid_indices:
            return pd.DataFrame(columns=[ITEM_COL, "score"])

        user_profile = self.tfidf_matrix[valid_indices].mean(axis=0)
        sim_scores = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        exclude = set(interacted_product_ids) | (exclude_ids or set())
        for pid in exclude:
            if pid in self.product_index:
                sim_scores[self.product_index[pid]] = -1

        top_indices = np.argsort(sim_scores)[::-1][:n_recommendations]
        result = self.products_df.iloc[top_indices].copy()
        result["score"] = sim_scores[top_indices]
        return result

    def save(self, path: Path = None) -> None:
        """Persist the fitted model to disk."""
        path = path or MODELS_DIR / "content_tfidf.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"✅ Content filter saved to {path}")

    @classmethod
    def load(cls, path: Path = None) -> "ContentFilter":
        """Load a previously saved model."""
        path = path or MODELS_DIR / "content_tfidf.pkl"
        return joblib.load(path)
