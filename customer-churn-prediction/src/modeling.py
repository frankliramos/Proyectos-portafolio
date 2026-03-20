import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from src.config import MODELS_DIR


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model (CPU).
    Configuration tuned for churn performance.
    """
    print("🚀 Starting XGBoost training (CPU)...")

    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,  # Use all available CPU cores
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Generate a detailed metrics report.
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE REPORT")
    print("=" * 60)
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    print(f"\n🎯 ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    print("=" * 60)

    return preds, probs


def save_model(model, filename="xgboost_churn_v1.json"):
    """
    Save the model in the models folder.
    """
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True)

    path = MODELS_DIR / filename
    model.save_model(str(path))
    print(f"✅ Model saved to: {path}")
