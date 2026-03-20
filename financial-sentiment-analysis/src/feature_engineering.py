"""
Feature engineering functions for customer churn prediction
"""

import pandas as pd
import numpy as np


def create_derived_features(df):
    """
    Create new features from existing ones
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new features
    """
    df = df.copy()
    
    # Balance to salary ratio
    df['balance_to_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1)
    
    # Customer value score (composite metric)
    df['customer_value_score'] = (
        df['balance'] / 100000 * 0.4 +
        df['num_of_products'] / 4 * 0.3 +
        df['tenure'] / 10 * 0.2 +
        df['is_active_member'] * 0.1
    )
    
    # Age groups
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 40, 50, 100],
        labels=['young', 'middle', 'mature', 'senior']
    )
    
    # Tenure groups
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[-1, 2, 5, 10],
        labels=['new', 'regular', 'loyal']
    )
    
    # Balance groups
    df['balance_group'] = pd.cut(
        df['balance'],
        bins=[-1, 10000, 100000, 250000],
        labels=['low', 'medium', 'high']
    )
    
    # Product engagement score
    df['product_engagement'] = df['num_of_products'] * df['is_active_member']
    
    # Credit score groups
    df['credit_score_group'] = pd.cut(
        df['credit_score'],
        bins=[0, 600, 700, 850],
        labels=['poor', 'good', 'excellent']
    )
    
    return df


def select_features(df, feature_list=None):
    """
    Select specific features for modeling
    
    Args:
        df: Input DataFrame
        feature_list: List of features to keep (None = keep all)
        
    Returns:
        DataFrame with selected features
    """
    if feature_list is None:
        return df
    
    available_features = [f for f in feature_list if f in df.columns]
    missing_features = set(feature_list) - set(available_features)
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    return df[available_features]


def create_interaction_features(df, feature_pairs=None):
    """
    Create interaction features between specified pairs
    
    Args:
        df: Input DataFrame
        feature_pairs: List of tuples (feature1, feature2) to interact
        
    Returns:
        DataFrame with interaction features
    """
    if feature_pairs is None:
        # Default interactions
        feature_pairs = [
            ('age', 'balance'),
            ('tenure', 'num_of_products'),
            ('is_active_member', 'num_of_products')
        ]
    
    df = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df[interaction_name] = df[feat1] * df[feat2]
    
    return df
