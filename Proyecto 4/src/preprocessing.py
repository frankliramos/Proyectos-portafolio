"""
Data preprocessing module for the recommendation system.

Handles:
- User-item interaction matrix construction
- Feature scaling and normalization
- Train/test splitting
- Cold-start handling for new users/items
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from .config import (
    PROCESSED_DATA_DIR,
    USER_COL,
    ITEM_COL,
    RATING_COL,
    TIMESTAMP_COL,
    INTERACTION_WEIGHTS,
    MIN_INTERACTIONS,
    MIN_PRODUCT_INTERACTIONS,
)


def compute_interaction_scores(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw interaction types to weighted implicit feedback scores.

    Args:
        interactions_df: DataFrame with columns [user_id, product_id, interaction_type]

    Returns:
        DataFrame with an additional 'interaction_score' column
    """
    df = interactions_df.copy()
    df[RATING_COL] = df["interaction_type"].map(INTERACTION_WEIGHTS).fillna(1)
    # Aggregate multiple interactions per user-item pair
    df = (
        df.groupby([USER_COL, ITEM_COL])[RATING_COL]
        .sum()
        .reset_index()
    )
    return df


def filter_sparse_users_items(
    interactions_df: pd.DataFrame,
    min_user_interactions: int = MIN_INTERACTIONS,
    min_item_interactions: int = MIN_PRODUCT_INTERACTIONS,
) -> pd.DataFrame:
    """
    Remove users and items with too few interactions to reduce noise.

    Args:
        interactions_df: Scored interactions DataFrame
        min_user_interactions: Minimum interactions required per user
        min_item_interactions: Minimum interactions required per item

    Returns:
        Filtered DataFrame
    """
    df = interactions_df.copy()
    user_counts = df[USER_COL].value_counts()
    item_counts = df[ITEM_COL].value_counts()

    valid_users = user_counts[user_counts >= min_user_interactions].index
    valid_items = item_counts[item_counts >= min_item_interactions].index

    return df[df[USER_COL].isin(valid_users) & df[ITEM_COL].isin(valid_items)]


def build_user_item_matrix(interactions_df: pd.DataFrame):
    """
    Build a sparse user-item interaction matrix.

    Args:
        interactions_df: DataFrame with user_id, product_id, interaction_score

    Returns:
        Tuple of (sparse matrix, user_index mapping, item_index mapping)
    """
    users = interactions_df[USER_COL].unique()
    items = interactions_df[ITEM_COL].unique()

    user_to_idx = {user: idx for idx, user in enumerate(users)}
    item_to_idx = {item: idx for idx, item in enumerate(items)}

    row_indices = interactions_df[USER_COL].map(user_to_idx)
    col_indices = interactions_df[ITEM_COL].map(item_to_idx)
    scores = interactions_df[RATING_COL].values

    matrix = csr_matrix(
        (scores, (row_indices, col_indices)),
        shape=(len(users), len(items)),
    )
    return matrix, user_to_idx, item_to_idx


def temporal_train_test_split(
    interactions_df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple:
    """
    Split interactions by timestamp to avoid data leakage.

    Args:
        interactions_df: DataFrame with a timestamp column
        test_ratio: Fraction of the most recent interactions to use as test set

    Returns:
        Tuple of (train_df, test_df)
    """
    df = interactions_df.sort_values(TIMESTAMP_COL)
    cutoff = int(len(df) * (1 - test_ratio))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Apply min-max normalization to a 1-D score array."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores.reshape(-1, 1)).flatten()


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    product_features_df: pd.DataFrame,
) -> None:
    """Persist processed DataFrames as Parquet files."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_DATA_DIR / "train_interactions.parquet", index=False)
    test_df.to_parquet(PROCESSED_DATA_DIR / "test_interactions.parquet", index=False)
    product_features_df.to_parquet(
        PROCESSED_DATA_DIR / "product_features.parquet", index=False
    )
    print(f"✅ Processed data saved to {PROCESSED_DATA_DIR}")
