import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco_customer_churn.xlsx"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

TARGET = "Churn Value"

GPU_CONFIG = {"tree_method": "gpu_hist", "device": "cuda", "predictor": "gpu_predictor"}
