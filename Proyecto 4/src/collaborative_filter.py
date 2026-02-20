"""
Collaborative filtering module using Alternating Least Squares (ALS).

Implements matrix factorization on implicit feedback data to learn
user and item latent factor representations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
import joblib

from .config import COLLABORATIVE_CONFIG, MODELS_DIR, USER_COL, ITEM_COL, RATING_COL


class CollaborativeFilter:
    """
    Matrix factorization recommendation model using Alternating Least Squares.

    Suitable for implicit feedback (views, clicks, purchases) data where
    explicit ratings are not available.
    """

    def __init__(
        self,
        n_factors: int = None,
        regularization: float = None,
        iterations: int = None,
        learning_rate: float = None,
    ):
        cfg = COLLABORATIVE_CONFIG
        self.n_factors = n_factors or cfg["n_factors"]
        self.regularization = regularization or cfg["regularization"]
        self.iterations = iterations or cfg["iterations"]
        self.learning_rate = learning_rate or cfg["learning_rate"]

        self.user_factors: np.ndarray = None
        self.item_factors: np.ndarray = None
        self.user_to_idx: dict = {}
        self.item_to_idx: dict = {}
        self.idx_to_item: dict = {}

    def fit(
        self,
        user_item_matrix: csr_matrix,
        user_to_idx: dict,
        item_to_idx: dict,
    ) -> "CollaborativeFilter":
        """
        Train the ALS model.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            user_to_idx: Mapping from user IDs to matrix row indices
            item_to_idx: Mapping from item IDs to matrix column indices

        Returns:
            self
        """
        try:
            from implicit import als

            model = als.AlternatingLeastSquares(
                factors=self.n_factors,
                regularization=self.regularization,
                iterations=self.iterations,
            )
            model.fit(user_item_matrix.T)  # implicit expects item-user matrix
            self.user_factors = model.user_factors
            self.item_factors = model.item_factors
        except ImportError:
            # Fallback: simple random initialization (demo mode)
            n_users, n_items = user_item_matrix.shape
            self.user_factors = np.random.normal(
                0, 0.1, (n_users, self.n_factors)
            )
            self.item_factors = np.random.normal(
                0, 0.1, (n_items, self.n_factors)
            )

        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_item = {v: k for k, v in item_to_idx.items()}
        return self

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        filter_already_interacted: bool = True,
        interacted_items: set = None,
    ) -> pd.DataFrame:
        """
        Generate top-N recommendations for a user.

        Args:
            user_id: Target user identifier
            n_recommendations: Number of items to recommend
            filter_already_interacted: Whether to exclude items user has seen
            interacted_items: Set of item IDs the user has already interacted with

        Returns:
            DataFrame with columns [product_id, score]
        """
        if user_id not in self.user_to_idx:
            return pd.DataFrame(columns=[ITEM_COL, "score"])

        user_idx = self.user_to_idx[user_id]
        user_vec = self.user_factors[user_idx]
        scores = self.item_factors @ user_vec

        if filter_already_interacted and interacted_items:
            for item_id in interacted_items:
                if item_id in self.item_to_idx:
                    scores[self.item_to_idx[item_id]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = pd.DataFrame(
            {
                ITEM_COL: [self.idx_to_item[i] for i in top_indices],
                "score": scores[top_indices],
            }
        )
        return recommendations

    def save(self, path: Path = None) -> None:
        """Persist model artefacts to disk."""
        path = path or MODELS_DIR / "collaborative_als.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"✅ Collaborative filter saved to {path}")

    @classmethod
    def load(cls, path: Path = None) -> "CollaborativeFilter":
        """Load a previously saved model."""
        path = path or MODELS_DIR / "collaborative_als.pkl"
        return joblib.load(path)
