import os
from pathlib import Path
import torch

# Rutas Base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Dataset Config
DATASET_NAME = "financial_phrasebank"
DATASET_CONFIG = (
    "sentences_allagree"  # Opciones: sentences_allagree, sentences_75agree, etc.
)

# Model Config
MODEL_NAME = "ProsusAI/finbert"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# Device Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Labels
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
