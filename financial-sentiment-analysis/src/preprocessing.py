"""
Data preprocessing functions for customer churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES, TARGET


def preprocess_data(df, fit_scaler=True, scaler=None):
    """
    Preprocess customer data for modeling
    
    Args:
        df: Input DataFrame
        fit_scaler: Whether to fit a new scaler or use existing
        scaler: Existing scaler object (if fit_scaler=False)
        
    Returns:
        tuple: (processed_df, scaler)
    """
    df = df.copy()
    
    # Handle missing values if any
    df = df.fillna(df.median(numeric_only=True))
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
    
    # Scale numeric features
    if fit_scaler:
        scaler = StandardScaler()
        df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit_scaler=False")
        df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    
    return df, scaler


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification
    
    Args:
        df: Input DataFrame
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in DataFrame")
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def handle_class_imbalance(X_train, y_train, method='smote'):
    """
    Handle class imbalance using SMOTE or other techniques
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: Method to use ('smote' or 'class_weights')
        
    Returns:
        tuple: (X_resampled, y_resampled) or (X_train, y_train, class_weights)
    """
    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        except ImportError:
            print("SMOTE not available. Using class weights instead.")
            method = 'class_weights'
    
    if method == 'class_weights':
        from sklearn.utils import class_weight
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight(
            'balanced', classes=classes, y=y_train
        )
        class_weights = dict(zip(classes, weights))
        return X_train, y_train, class_weights
    
    raise ValueError(f"Unknown method: {method}")
