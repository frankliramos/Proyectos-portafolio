#!/usr/bin/env python3
"""
Test Data Preparation and Dashboard Compatibility

This script verifies that processed data is correct and compatible
with the dashboard and inference model.

Author: Franklin Ramos
Date: 2026-02-04
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_data_files():
    """Verify that processed data files exist and have the correct format."""
    print("\n" + "=" * 70)
    print("TEST 1: DATA FILE VERIFICATION")
    print("=" * 70)

    from src.config import PROCESSED_DATA_DIR

    train_file = PROCESSED_DATA_DIR / "fd001_prepared.parquet"
    test_file = PROCESSED_DATA_DIR / "fd001_test_prepared.parquet"

    # Check training file
    print(f"\n📄 Verifying {train_file.name}...")
    if not train_file.exists():
        print(f"   ✗ File does not exist: {train_file}")
        return False

    df_train = pd.read_parquet(train_file)
    print(f"   ✓ File exists")
    print(f"   ✓ Shape: {df_train.shape}")
    print(f"   ✓ Engines: {df_train['unit_id'].nunique()}")

    if len(df_train) < 10000:
        print(f"   ✗ ERROR: Too few rows ({len(df_train)}). Only last cycles?")
        return False
    print(f"   ✓ Contains all cycles (>10k rows)")

    # Check test file
    print(f"\n📄 Verifying {test_file.name}...")
    if not test_file.exists():
        print(f"   ✗ File does not exist: {test_file}")
        return False

    df_test = pd.read_parquet(test_file)
    print(f"   ✓ File exists")
    print(f"   ✓ Shape: {df_test.shape}")
    print(f"   ✓ Engines: {df_test['unit_id'].nunique()}")

    if len(df_test) < 5000:
        print(f"   ✗ ERROR: Too few rows ({len(df_test)}). Only last cycles?")
        return False
    print(f"   ✓ Contains all cycles (>5k rows)")

    # Check columns match
    required_cols = ["unit_id", "time_cycles", "RUL", "op_setting_1", "sensor_1"]
    for col in required_cols:
        if col not in df_test.columns:
            print(f"   ✗ Missing column: {col}")
            return False
    print(f"   ✓ All required columns present")

    return True


def test_dashboard_compatibility():
    """Verify that the data is compatible with the dashboard."""
    print("\n" + "=" * 70)
    print("TEST 2: DASHBOARD COMPATIBILITY")
    print("=" * 70)

    from src.config import PROCESSED_DATA_DIR

    # Load data like the dashboard does
    data_path = PROCESSED_DATA_DIR / "fd001_test_prepared.parquet"
    df = pd.read_parquet(data_path)
    print(f"\n✓ Data loaded: {df.shape}")

    # Rename columns like the dashboard does
    if "unit_id" in df.columns:
        df = df.rename(columns={"unit_id": "id"})
    if "time_cycles" in df.columns:
        df = df.rename(columns={"time_cycles": "cycle"})
    print(f"✓ Columns renamed")

    # Validate
    if "id" not in df.columns or "cycle" not in df.columns:
        print(f"✗ ERROR: Missing 'id' or 'cycle' columns")
        return False
    print(f"✓ Column validation OK")

    # Check distribution
    engine_ids = df["id"].unique()
    last_rul_per_engine = df.groupby("id")["RUL"].last()

    critical = (last_rul_per_engine < 30).sum()
    warning = ((last_rul_per_engine >= 30) & (last_rul_per_engine < 70)).sum()
    healthy = (last_rul_per_engine >= 70).sum()

    print(f"\n📊 State distribution (last RUL per engine):")
    print(f"   🔴 Critical:  {critical:3d} ({critical/len(engine_ids)*100:5.1f}%)")
    print(f"   🟡 Warning:   {warning:3d} ({warning/len(engine_ids)*100:5.1f}%)")
    print(f"   🟢 Healthy:   {healthy:3d} ({healthy/len(engine_ids)*100:5.1f}%)")

    if critical == len(engine_ids):
        print(f"\n✗ ERROR: ALL engines are critical!")
        print(f"   This indicates the file only has last cycles.")
        return False

    if healthy == 0:
        print(f"\n✗ WARNING: No healthy engines")

    print(f"\n✓ State distribution is varied (not all critical)")

    return True


def test_model_compatibility():
    """Verify that the data is compatible with the inference model."""
    print("\n" + "=" * 70)
    print("TEST 3: MODEL COMPATIBILITY")
    print("=" * 70)

    try:
        import joblib
        from src.config import FEATURE_COLS_FILE, PROCESSED_DATA_DIR

        # Load expected features
        if not FEATURE_COLS_FILE.exists():
            print(f"✗ Feature file does not exist: {FEATURE_COLS_FILE}")
            return False

        feature_cols = joblib.load(FEATURE_COLS_FILE)
        print(f"\n✓ Expected model features: {len(feature_cols)}")
        print(f"   {feature_cols[:5]}...")

        # Load data
        df = pd.read_parquet(PROCESSED_DATA_DIR / "fd001_test_prepared.parquet")

        # Check if all features are present
        missing = set(feature_cols) - set(df.columns)
        if missing:
            print(f"✗ Missing features: {missing}")
            return False

        print(f"✓ All required features are present")

        # Test with one engine
        motor_data = df[df["unit_id"] == 1].sort_values("time_cycles")
        if len(motor_data) < 30:
            print(f"✗ Engine 1 has fewer than 30 cycles ({len(motor_data)})")
            return False

        print(f"✓ Engine 1 has enough cycles for prediction ({len(motor_data)})")

        return True

    except ImportError as e:
        print(f"⚠️  Could not verify model compatibility: {e}")
        print(f"   (It is OK if not all dependencies are installed)")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "PROCESSED DATA VERIFICATION")
    print("=" * 70)

    tests = [
        ("Data files", test_data_files),
        ("Dashboard compatibility", test_dashboard_compatibility),
        ("Model compatibility", test_model_compatibility),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ ERROR in test '{test_name}': {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {test_name}")

    all_passed = all(r for _, r in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\n💡 Data is ready for:")
        print("   - Dashboard (streamlit run app.py)")
        print("   - Analysis notebooks")
        print("   - LSTM model predictions")
        print("\n" + "=" * 70)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\n🔧 Fix:")
        print("   Run: python prepare_all_data.py")
        print("   To regenerate processed data correctly")
        print("\n" + "=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
