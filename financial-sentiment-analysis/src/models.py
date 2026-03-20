"""
Model training and evaluation functions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from .config import MODEL_CONFIG
import joblib


def train_xgboost(X_train, y_train, params=None):
    """
    Train XGBoost classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model parameters (uses default if None)
        
    Returns:
        Trained XGBoost model
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    if params is None:
        params = MODEL_CONFIG['xgboost']
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


def train_random_forest(X_train, y_train, params=None):
    """
    Train Random Forest classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model parameters (uses default if None)
        
    Returns:
        Trained Random Forest model
    """
    if params is None:
        params = MODEL_CONFIG['random_forest']
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


def train_logistic_regression(X_train, y_train, params=None):
    """
    Train Logistic Regression classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model parameters (uses default if None)
        
    Returns:
        Trained Logistic Regression model
    """
    if params is None:
        params = MODEL_CONFIG['logistic_regression']
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    return model


def create_ensemble(models, voting='soft'):
    """
    Create ensemble model from multiple classifiers
    
    Args:
        models: List of tuples (name, model)
        voting: 'soft' or 'hard' voting
        
    Returns:
        Voting classifier ensemble
    """
    ensemble = VotingClassifier(estimators=models, voting=voting)
    return ensemble


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics


def save_model(model, filepath):
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath):
    """
    Load trained model from disk
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    return model
