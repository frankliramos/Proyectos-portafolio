"""
Source code modules for Product Recommendation System
"""

__version__ = "1.0.0"
__author__ = "Franklin Ramos"

from .config import (
    COLLABORATIVE_CONFIG,
    CONTENT_CONFIG,
    HYBRID_CONFIG,
    DEFAULT_N_RECOMMENDATIONS,
)
from .data_loader import load_interactions, load_products, load_users
from .preprocessing import (
    compute_interaction_scores,
    filter_sparse_users_items,
    build_user_item_matrix,
    temporal_train_test_split,
    normalize_scores,
)
from .collaborative_filter import CollaborativeFilter
from .content_filter import ContentFilter
from .hybrid_model import HybridRecommender
from .evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision,
    evaluate_recommendations,
    catalog_coverage,
    print_evaluation_report,
)

__all__ = [
    "CollaborativeFilter",
    "ContentFilter",
    "HybridRecommender",
    "load_interactions",
    "load_products",
    "load_users",
    "compute_interaction_scores",
    "filter_sparse_users_items",
    "build_user_item_matrix",
    "temporal_train_test_split",
    "normalize_scores",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "average_precision",
    "evaluate_recommendations",
    "catalog_coverage",
    "print_evaluation_report",
    "COLLABORATIVE_CONFIG",
    "CONTENT_CONFIG",
    "HYBRID_CONFIG",
    "DEFAULT_N_RECOMMENDATIONS",
]
