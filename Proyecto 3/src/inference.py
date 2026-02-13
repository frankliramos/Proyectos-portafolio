"""
Inference engine for churn prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .models import load_model
from .config import MODELS_DIR, RISK_THRESHOLDS
import joblib


class ChurnPredictor:
    """
    Inference engine for customer churn prediction
    """
    
    def __init__(self, project_root=None, model_name='ensemble_model.pkl'):
        """
        Initialize the inference engine
        
        Args:
            project_root: Path to project root directory
            model_name: Name of the model file to load
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        self.project_root = Path(project_root)
        self.models_dir = self.project_root / 'models'
        
        # Load model and scaler
        self.model = self._load_model(model_name)
        self.scaler = self._load_scaler()
        self.feature_names = self._load_feature_names()
    
    def _load_model(self, model_name):
        """Load trained model"""
        model_path = self.models_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Please train the model first."
            )
        return load_model(model_path)
    
    def _load_scaler(self):
        """Load fitted scaler"""
        scaler_path = self.models_dir / 'scaler.pkl'
        if scaler_path.exists():
            return joblib.load(scaler_path)
        return None
    
    def _load_feature_names(self):
        """Load feature names"""
        features_path = self.models_dir / 'feature_names.pkl'
        if features_path.exists():
            return joblib.load(features_path)
        return None
    
    def predict_proba(self, customer_data):
        """
        Predict churn probability for a single customer
        
        Args:
            customer_data: Dictionary or DataFrame with customer features
            
        Returns:
            float: Churn probability (0-1)
        """
        # Convert to DataFrame if dict
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Preprocess if needed
        if self.scaler is not None:
            customer_data = self._preprocess(customer_data)
        
        # Ensure feature order matches training
        if self.feature_names is not None:
            customer_data = customer_data[self.feature_names]
        
        # Predict
        proba = self.model.predict_proba(customer_data)[0, 1]
        return proba
    
    def predict_batch(self, customers_df):
        """
        Predict churn probability for multiple customers
        
        Args:
            customers_df: DataFrame with customer features
            
        Returns:
            numpy.ndarray: Array of churn probabilities
        """
        # Preprocess if needed
        if self.scaler is not None:
            customers_df = self._preprocess(customers_df)
        
        # Ensure feature order
        if self.feature_names is not None:
            customers_df = customers_df[self.feature_names]
        
        # Predict
        probabilities = self.model.predict_proba(customers_df)[:, 1]
        return probabilities
    
    def classify_risk(self, probability):
        """
        Classify risk level based on churn probability
        
        Args:
            probability: Churn probability (0-1)
            
        Returns:
            str: Risk level ('High', 'Medium', or 'Low')
        """
        if probability >= RISK_THRESHOLDS['high']:
            return 'High'
        elif probability >= RISK_THRESHOLDS['medium']:
            return 'Medium'
        else:
            return 'Low'
    
    def _preprocess(self, df):
        """Preprocess customer data"""
        # This is a simplified version
        # In production, would match exact training preprocessing
        df = df.copy()
        
        # Apply scaler to numeric features if available
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if self.scaler is not None and len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def get_retention_strategy(self, probability):
        """
        Get recommended retention strategy based on churn probability
        
        Args:
            probability: Churn probability
            
        Returns:
            dict: Retention strategy recommendations
        """
        risk_level = self.classify_risk(probability)
        
        strategies = {
            'High': {
                'priority': 'URGENT',
                'actions': [
                    'Immediate call from retention team',
                    'Personalized retention offer (fee waiver + bonus)',
                    'Executive-level outreach',
                    'Premium customer service upgrade'
                ],
                'expected_roi': '3.5x',
                'budget': '$200-500 per customer'
            },
            'Medium': {
                'priority': 'HIGH',
                'actions': [
                    'Personalized email campaign',
                    'Special offer on additional products',
                    'Account manager check-in call',
                    'Loyalty rewards activation'
                ],
                'expected_roi': '2.8x',
                'budget': '$100-200 per customer'
            },
            'Low': {
                'priority': 'STANDARD',
                'actions': [
                    'Standard customer service',
                    'Quarterly satisfaction survey',
                    'Loyalty program enrollment',
                    'Product usage tips newsletter'
                ],
                'expected_roi': '1.5x',
                'budget': '$20-50 per customer'
            }
        }
        
        return strategies.get(risk_level, strategies['Low'])
