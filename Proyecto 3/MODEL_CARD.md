# Model Card: Bank Customer Churn Prediction

## Model Details
- **Model Name**: Ensemble Churn Classifier
- **Version**: 1.0.0
- **Type**: Binary Classification
- **Framework**: Python (XGBoost, scikit-learn)
- **Developer**: Franklin Ramos
- **Date**: 2025

## Intended Use
- **Primary Use**: Identify bank customers at high risk of churning to enable proactive retention campaigns
- **Intended Users**: Customer success teams, marketing departments, bank management
- **Out-of-Scope Uses**: Individual credit decisions, discriminatory profiling

## Training Data
- **Source**: Kaggle Bank Customer Churn Dataset
- **Size**: 10,000 customer records
- **Features**: 14 attributes (demographics, account activity, product usage)
- **Target**: Binary (Exited: 1 = Churned, 0 = Retained)
- **Class Distribution**: ~20% churn rate (imbalanced)
- **Preprocessing**: SMOTE oversampling, class weight adjustment, feature scaling

## Evaluation Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 86.5% |
| ROC-AUC | 0.91 |
| F1-Score | 0.84 |
| Precision | 0.82 |
| Recall | 0.86 |

## Model Architecture
- **Ensemble**: XGBoost + Random Forest + Logistic Regression
- **Optimization**: F1-Score and ROC-AUC
- **Class Imbalance Handling**: SMOTE oversampling + class weights
- **Feature Engineering**: 20+ attributes including derived interaction features
- **Output**: Churn probability (0-100%) with risk classification (Low/Medium/High)

## Ethical Considerations
- Model uses demographic features (age, gender, geography) — monitor for bias across protected groups
- Churn predictions should not be used to deny services or discriminate
- Regular fairness audits recommended across geography and gender segments
- Transparency: Customers flagged as high-risk should receive positive retention offers, not punitive actions

## Limitations & Caveats
- Trained on simulated (but realistic) data — production deployment requires real bank data
- Geographic scope limited to France, Spain, Germany
- Does not account for macroeconomic factors or competitive landscape
- Performance may degrade if customer behavior patterns shift significantly
- Model assumes static feature relationships — periodic retraining recommended

## Recommendations
- Retrain monthly with production data
- A/B test retention campaigns on model predictions
- Monitor for concept drift and demographic bias
- Integrate with CRM for automated retention workflows
- Validate on holdout data before each retraining cycle
