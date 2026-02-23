"""
Evaluation module for recommendation system metrics.

Implements standard information-retrieval metrics used in recommender systems:
  - Precision@K
  - Recall@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - MAP (Mean Average Precision)
  - Coverage
  - Serendipity / Novelty (optional)
"""

import numpy as np
import pandas as pd
from typing import List, Dict


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def precision_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Fraction of top-K recommendations that are relevant.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of ground-truth relevant item IDs
        k: Cut-off position

    Returns:
        Precision@K value in [0, 1]
    """
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Fraction of relevant items that appear in top-K recommendations.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of ground-truth relevant item IDs
        k: Cut-off position

    Returns:
        Recall@K value in [0, 1]
    """
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Rewards relevant items appearing earlier in the recommendation list.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of ground-truth relevant item IDs
        k: Cut-off position

    Returns:
        NDCG@K value in [0, 1]
    """
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(recommended: List, relevant: set) -> float:
    """
    Average Precision for a single user (used to compute MAP).

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of ground-truth relevant item IDs

    Returns:
        Average Precision value in [0, 1]
    """
    if not relevant:
        return 0.0
    hits = 0
    cumulative_precision = 0.0
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
            cumulative_precision += hits / rank
    return cumulative_precision / len(relevant)


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_recommendations(
    user_recommendations: Dict[str, List],
    user_ground_truth: Dict[str, set],
    k_values: List[int] = None,
) -> pd.DataFrame:
    """
    Compute Precision, Recall, NDCG, and MAP across all users.

    Args:
        user_recommendations: {user_id: [ranked list of item IDs]}
        user_ground_truth:    {user_id: set of relevant item IDs}
        k_values: List of K values to evaluate (default: [5, 10, 20])

    Returns:
        DataFrame with one row per K value containing mean metrics
    """
    if k_values is None:
        k_values = [5, 10, 20]

    results = []
    for k in k_values:
        metrics_per_user = {
            "precision": [],
            "recall": [],
            "ndcg": [],
            "ap": [],
        }
        for user_id, recs in user_recommendations.items():
            relevant = user_ground_truth.get(user_id, set())
            metrics_per_user["precision"].append(precision_at_k(recs, relevant, k))
            metrics_per_user["recall"].append(recall_at_k(recs, relevant, k))
            metrics_per_user["ndcg"].append(ndcg_at_k(recs, relevant, k))
            metrics_per_user["ap"].append(average_precision(recs, relevant))

        results.append(
            {
                "k": k,
                f"Precision@{k}": np.mean(metrics_per_user["precision"]),
                f"Recall@{k}": np.mean(metrics_per_user["recall"]),
                f"NDCG@{k}": np.mean(metrics_per_user["ndcg"]),
                "MAP": np.mean(metrics_per_user["ap"]),
            }
        )

    return pd.DataFrame(results).set_index("k")


def catalog_coverage(
    user_recommendations: Dict[str, List],
    total_items: int,
) -> float:
    """
    Fraction of the item catalogue appearing in at least one recommendation.

    Args:
        user_recommendations: {user_id: [list of recommended item IDs]}
        total_items: Total number of items in the catalogue

    Returns:
        Coverage value in [0, 1]
    """
    all_recommended = {
        item for recs in user_recommendations.values() for item in recs
    }
    return len(all_recommended) / total_items if total_items > 0 else 0.0


def print_evaluation_report(metrics_df: pd.DataFrame, coverage: float = None) -> None:
    """Pretty-print evaluation results."""
    print("=" * 60)
    print("RECOMMENDATION SYSTEM EVALUATION REPORT")
    print("=" * 60)
    print(metrics_df.round(4).to_string())
    if coverage is not None:
        print(f"\nCatalog Coverage: {coverage:.2%}")
    print("=" * 60)
