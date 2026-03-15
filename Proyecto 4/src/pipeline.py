import pandas as pd

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data, get_train_test_splits
from src.modeling import train_xgboost, evaluate_model
from src.business_metrics import calculate_business_impact


def run_pipeline():
    print("🚀 Starting Production Pipeline: Churn Prediction System")
    print("=" * 60)

    # 1) LOAD
    print("\n📂 Step 1: Loading data...")
    df_raw = load_raw_data()

    # 2) PREPROCESS
    print("\n🔧 Step 2: Preprocessing data...")
    df_ml = preprocess_data(df_raw)
    X_train, X_test, y_train, y_test = get_train_test_splits(df_ml)

    # Monthly Charges for ROI (not preprocessed)
    monthly_charges_test = df_raw.loc[X_test.index, "Monthly Charges"]

    # 3) TRAINING
    print("\n🧠 Step 3: Training model (XGBoost CPU)...")
    model = train_xgboost(X_train, y_train, X_test, y_test)

    # 4) EVALUATION
    print("\n📊 Step 4: Model evaluation...")
    preds, probs = evaluate_model(model, X_test, y_test)

    # 5) ROI / BUSINESS IMPACT
    print("\n💼 Step 5: Calculating Business Impact (ROI)...")
    impact = calculate_business_impact(y_test, preds, monthly_charges_test)

    print("\n" + "=" * 60)
    print("💰 FINANCIAL IMPACT REPORT (ROI)")
    print("=" * 60)
    print(
        f"💵 Monthly Revenue at Risk Identified: ${impact['revenue_at_risk_monthly']:,.2f}"
    )
    print(
        f"📉 Estimated Monthly Savings (40% retention): ${impact['potential_savings_monthly']:,.2f}"
    )
    print(
        f"🏆 PROJECTED ANNUAL SAVINGS:                 ${impact['potential_savings_annual']:,.2f}"
    )
    print("=" * 60)

    # 6) Risk segmentation
    print("\n🎯 Step 6: Segmenting customers by risk...")
    risk_segments = pd.Series(
        pd.cut(probs, bins=[0, 0.3, 0.7, 1.0], labels=["Bajo", "Medio", "Alto"]),
        name="SegmentoRiesgo",
    )

    print("\nCustomer distribution by segment:")
    print(risk_segments.value_counts().sort_index())

    print("\n✅ Pipeline executed successfully.")


if __name__ == "__main__":
    run_pipeline()
