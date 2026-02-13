"""
Configuration and path management for the project
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

# Model configurations
MODEL_CONFIG = {
    'xgboost': {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'scale_pos_weight': 4.0,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'logistic_regression': {
        'C': 0.1,
        'penalty': 'l2',
        'class_weight': 'balanced',
        'random_state': 42,
        'max_iter': 1000
    }
}

# Feature configurations
NUMERIC_FEATURES = [
    'credit_score', 'age', 'tenure', 'balance',
    'num_of_products', 'estimated_salary'
]

CATEGORICAL_FEATURES = ['geography', 'gender']

BINARY_FEATURES = ['has_cr_card', 'is_active_member']

TARGET = 'exited'

# Risk thresholds
RISK_THRESHOLDS = {
    'high': 0.6,
    'medium': 0.3
}
