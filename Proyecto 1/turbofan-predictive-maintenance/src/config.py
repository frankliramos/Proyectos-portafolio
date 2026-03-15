# src/config.py
"""
Central project configuration.

This module centralizes paths, parameters, and basic configuration so they
can be reused across all components (data loading, training, evaluation,
dashboard, and more).

Author: Franklin Ramos
Date: 2026-02-03
"""

from pathlib import Path

# ===========================
# PROJECT PATHS
# ===========================

# Project root (assumes this file is in PROJECT 1/src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"

# Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ===========================
# FD001 DATA FILES
# ===========================

FD001_TRAIN_FILE = RAW_DATA_DIR / "train_FD001.txt"
FD001_TEST_FILE = RAW_DATA_DIR / "test_FD001.txt"
FD001_RUL_FILE = RAW_DATA_DIR / "RUL_FD001.txt"
FD001_PROCESSED_FILE = PROCESSED_DATA_DIR / "fd001_prepared.parquet"
FD001_TEST_PROCESSED_FILE = PROCESSED_DATA_DIR / "fd001_test_prepared.parquet"

# ===========================
# MODEL FILES
# ===========================

# Current model version
MODEL_VERSION = "v1"

# Model artifact paths
LSTM_MODEL_FILE = MODELS_DIR / f"lstm_model_{MODEL_VERSION}.pth"
SCALER_FILE = MODELS_DIR / f"scaler_{MODEL_VERSION}.pkl"
FEATURE_COLS_FILE = MODELS_DIR / f"feature_cols_{MODEL_VERSION}.pkl"

# ===========================
# MODEL HYPERPARAMETERS
# ===========================

# Time sequences
DEFAULT_SEQUENCE_LENGTH = 30

# LSTM architecture
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 256
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Preprocessing
MAX_RUL_CLIP = 125  # Max RUL value for clipping during training

# ===========================
# DASHBOARD CONFIGURATION
# ===========================

# Default health state thresholds
DEFAULT_CRITICAL_THRESHOLD = 30
DEFAULT_WARNING_THRESHOLD = 70

# Default sensors to visualize
DEFAULT_SENSORS = ["sensor_4", "sensor_11", "sensor_12"]

# ===========================
# PROJECT METADATA
# ===========================

PROJECT_NAME = "Turbofan RUL Prediction"
PROJECT_VERSION = "1.0.0"
AUTHOR = "Franklin Ramos"
DESCRIPTION = "Predictive Maintenance using LSTM for NASA CMAPSS Dataset"

# ===========================
# LOGGING
# ===========================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
