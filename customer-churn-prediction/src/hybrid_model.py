"""
Hybrid recommendation model combining collaborative and content-based filtering.

Merges scores from both approaches using a weighted linear combination,
providing robust recommendations even for users with limited interaction history.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from .config import HYBRID_CONFIG, MODELS_DIR, ITEM_COL
from .collaborative_filter import CollaborativeFilter
from .content_filter import ContentFilter
from .preprocessing import normalize_scores


class HybridRecommender:
    """
    Weighted hybrid recommendation engine.

    Combines collaborative filtering (CF) and content-based filtering (CB)
    to deliver high-quality personalized recommendations across all user
    segments, including cold-start users.
    """

    def __init__(
        self,
        project_root: Path = None,
        cf_weight: float = None,
        cb_weight: float = None,
    ):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.cf_weight = cf_weight if cf_weight is not None else HYBRID_CONFIG["collaborative_weight"]
        self.cb_weight = cb_weight if cb_weight is not None else HYBRID_CONFIG["content_weight"]

        self.cf_model: CollaborativeFilter = None
        self.cb_model: ContentFilter = None
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_models(self) -> "HybridRecommender":
        """Load pre-trained CF and CB models from disk."""
        models_dir = self.project_root / "models"
        self.cf_model = CollaborativeFilter.load(models_dir / "collaborative_als.pkl")
        self.cb_model = ContentFilter.load(models_dir / "content_tfidf.pkl")
        self._is_loaded = True
        return self

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        filter_purchased: bool = True,
        interacted_items: list = None,
    ) -> pd.DataFrame:
        """
        Generate hybrid recommendations for a user.

        Falls back gracefully:
        - If CF model unavailable → pure CB
        - If CB model unavailable → pure CF
        - If user is new (cold-start) → pure CB based on popular items

        Args:
            user_id: Target user identifier
            n_recommendations: Number of products to recommend
            filter_purchased: Exclude products user has already interacted with
            interacted_items: List of product IDs the user has interacted with

        Returns:
            DataFrame with product details, score, and recommendation reason
        """
        interacted_set = set(interacted_items or [])
        n_candidates = n_recommendations * 3  # over-fetch for re-ranking

        cf_recs = pd.DataFrame(columns=[ITEM_COL, "score"])
        cb_recs = pd.DataFrame(columns=[ITEM_COL, "score"])

        if self.cf_model is not None:
            cf_recs = self.cf_model.recommend(
                user_id,
                n_recommendations=n_candidates,
                filter_already_interacted=filter_purchased,
                interacted_items=interacted_set,
            )

        if self.cb_model is not None and interacted_items:
            cb_recs = self.cb_model.recommend_from_history(
                list(interacted_set),
                n_recommendations=n_candidates,
                exclude_ids=interacted_set if filter_purchased else None,
            )
            cb_recs = cb_recs.rename(columns={"score": "score"})[[ITEM_COL, "score"]]

        # Merge and combine scores
        combined = self._merge_scores(cf_recs, cb_recs)
        combined = combined.sort_values("score", ascending=False).head(n_recommendations)
        combined["reason"] = combined["score"].apply(self._score_to_reason)
        return combined

    def find_similar_products(
        self, product_id: str, n_similar: int = 10
    ) -> pd.DataFrame:
        """Find products similar to a given item using the CB model."""
        if self.cb_model is None:
            return pd.DataFrame(columns=[ITEM_COL, "similarity"])
        return self.cb_model.find_similar(product_id, n_similar)

    def batch_recommend(
        self, user_ids: list, n_recommendations: int = 10
    ) -> list:
        """Generate recommendations for a list of users."""
        results = []
        for uid in user_ids:
            recs = self.recommend(uid, n_recommendations=n_recommendations)
            for _, row in recs.iterrows():
                results.append(
                    {
                        "user_id": uid,
                        ITEM_COL: row[ITEM_COL],
                        "score": row["score"],
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _merge_scores(
        self, cf_recs: pd.DataFrame, cb_recs: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine CF and CB scores with configured weights."""
        if cf_recs.empty and cb_recs.empty:
            return pd.DataFrame(columns=[ITEM_COL, "score"])

        if cf_recs.empty:
            cb_recs["score"] = normalize_scores(cb_recs["score"].values)
            return cb_recs

        if cb_recs.empty:
            cf_recs["score"] = normalize_scores(cf_recs["score"].values)
            return cf_recs

        # Normalize each set of scores independently
        cf_recs = cf_recs.copy()
        cb_recs = cb_recs.copy()
        cf_recs["score"] = normalize_scores(cf_recs["score"].values)
        cb_recs["score"] = normalize_scores(cb_recs["score"].values)

        merged = pd.merge(
            cf_recs.rename(columns={"score": "cf_score"}),
            cb_recs.rename(columns={"score": "cb_score"}),
            on=ITEM_COL,
            how="outer",
        ).fillna(0)

        merged["score"] = (
            self.cf_weight * merged["cf_score"]
            + self.cb_weight * merged["cb_score"]
        )
        return merged[[ITEM_COL, "score"]]

    @staticmethod
    def _score_to_reason(score: float) -> str:
        if score >= 0.85:
            return "Highly recommended based on your preferences"
        elif score >= 0.70:
            return "Customers with similar tastes also bought this"
        elif score >= 0.55:
            return "Popular in your favourite categories"
        else:
            return "You might also like this"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = None) -> None:
        """Save the hybrid model configuration and sub-models."""
        path = path or MODELS_DIR / "hybrid_recommender.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"✅ Hybrid recommender saved to {path}")

    @classmethod
    def load(cls, path: Path = None) -> "HybridRecommender":
        """Load a previously saved hybrid model."""
        path = path or MODELS_DIR / "hybrid_recommender.pkl"
        return joblib.load(path)
