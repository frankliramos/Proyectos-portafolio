#!/usr/bin/env python3
"""
Prepare All Data for Dashboard and Analysis

This script runs the full preparation of training and test data.

Author: Franklin Ramos
Date: 2026-02-04
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "FULL DATA PREPARATION")
    print("=" * 70)

    print("\n📊 This script will prepare:")
    print("  1. Training data (train_FD001.txt) → fd001_prepared.parquet")
    print("  2. Test data (test_FD001.txt) → fd001_test_prepared.parquet")
    print("\n⚠️  IMPORTANT: Files will include ALL cycles for each engine.")
    print("    For model evaluation, use .groupby('unit_id').tail(1)")
    print("    to keep only the last cycle.\n")

    input("Press Enter to continue...")

    # Import here to avoid errors if modules not ready
    from prepare_train_data import prepare_train_data
    from prepare_test_data import prepare_test_data

    try:
        # Prepare training data
        print("\n" + "=" * 70)
        print("STEP 1/2: TRAINING DATA")
        print("=" * 70)
        df_train = prepare_train_data()

        # Prepare test data
        print("\n" + "=" * 70)
        print("STEP 2/2: TEST DATA")
        print("=" * 70)
        df_test = prepare_test_data()

        # Summary
        print("\n" + "=" * 70)
        print("✅ FULL PREPARATION COMPLETED")
        print("=" * 70)
        print(f"\n📁 Generated files:")
        print(
            f"  - fd001_prepared.parquet:      {len(df_train):6,} rows × {len(df_train.columns):2} columns"
        )
        print(
            f"  - fd001_test_prepared.parquet: {len(df_test):6,} rows × {len(df_test.columns):2} columns"
        )

        print(f"\n🔧 Engines available:")
        print(f"  - Training: {df_train['unit_id'].nunique()} engines")
        print(f"  - Test:     {df_test['unit_id'].nunique()} engines")

        print("\n✨ Data is ready for:")
        print("  → Monitoring dashboard (app.py)")
        print("  → Exploratory analysis (notebooks/)")
        print("  → Model training")

        print("\n💡 Next steps:")
        print("  1. Run the dashboard: streamlit run app.py")
        print("  2. Review notebooks in: notebooks/")
        print("  3. Train models with full data")

        print("\n" + "=" * 70 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Error during preparation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
