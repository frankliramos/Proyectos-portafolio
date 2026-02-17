"""
Configuration settings for the recommendation system
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Model parameters
COLLABORATIVE_CONFIG = {
    'n_factors': 100,
    'regularization': 0.01,
    'iterations': 20,
    'learning_rate': 0.01
}

CONTENT_CONFIG = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.8
}

HYBRID_CONFIG = {
    'collaborative_weight': 0.6,
    'content_weight': 0.4
}

NEURAL_CONFIG = {
    'embedding_dim': 64,
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 50
}

# Recommendation settings
DEFAULT_N_RECOMMENDATIONS = 10
MIN_INTERACTIONS = 5
MIN_PRODUCT_INTERACTIONS = 3

# Evaluation metrics
EVALUATION_K_VALUES = [5, 10, 20]

# Data columns
USER_COL = 'user_id'
ITEM_COL = 'product_id'
RATING_COL = 'interaction_score'
TIMESTAMP_COL = 'timestamp'

# Interaction weights
INTERACTION_WEIGHTS = {
    'view': 1,
    'cart_add': 2,
    'purchase': 5,
    'rating': 3
}

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
