"""
Data loading utilities for the recommendation system
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_interactions(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load user-product interaction data
    
    Args:
        file_path: Path to interactions file. If None, uses default location.
        
    Returns:
        DataFrame with interaction data
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "interactions.csv"
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} interactions from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Interactions file not found at {file_path}")
        return pd.DataFrame()


def load_products(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load product catalog data
    
    Args:
        file_path: Path to products file. If None, uses default location.
        
    Returns:
        DataFrame with product data
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "products.csv"
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} products from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Products file not found at {file_path}")
        return pd.DataFrame()


def load_users(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load user profile data
    
    Args:
        file_path: Path to users file. If None, uses default location.
        
    Returns:
        DataFrame with user data
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "users.csv"
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} users from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Users file not found at {file_path}")
        return pd.DataFrame()


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed training and test data
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train_path = PROCESSED_DATA_DIR / "train_data.parquet"
    test_path = PROCESSED_DATA_DIR / "test_data.parquet"
    
    try:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        logger.info(f"Loaded processed data: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df
    except FileNotFoundError as e:
        logger.error(f"Processed data not found: {e}")
        return pd.DataFrame(), pd.DataFrame()


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save preprocessed data to parquet files
    
    Args:
        train_df: Training data
        test_df: Test data
    """
    train_path = PROCESSED_DATA_DIR / "train_data.parquet"
    test_path = PROCESSED_DATA_DIR / "test_data.parquet"
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    logger.info(f"Saved processed data to {PROCESSED_DATA_DIR}")
