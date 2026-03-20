"""
Data loading utilities for customer churn dataset
"""

import pandas as pd
from pathlib import Path
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_data(filename='bank_churn.csv'):
    """
    Load raw customer churn data
    
    Args:
        filename: Name of the raw data file
        
    Returns:
        pandas.DataFrame: Raw customer data
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Please ensure the raw data is available in data/raw/"
        )
    
    df = pd.read_csv(filepath)
    return df


def load_processed_data(filename='churn_prepared.parquet'):
    """
    Load preprocessed customer data
    
    Args:
        filename: Name of the processed data file
        
    Returns:
        pandas.DataFrame: Preprocessed customer data
    """
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            "Please run data preparation script first."
        )
    
    df = pd.read_parquet(filepath)
    return df


def save_processed_data(df, filename='churn_prepared.parquet'):
    """
    Save processed data to parquet format
    
    Args:
        df: DataFrame to save
        filename: Output filename
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DATA_DIR / filename
    df.to_parquet(filepath, index=False)
    print(f"Processed data saved to: {filepath}")
