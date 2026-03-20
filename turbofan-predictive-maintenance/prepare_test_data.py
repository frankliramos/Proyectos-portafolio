#!/usr/bin/env python3
"""
Prepare Test Data for Dashboard

This script prepares test data (test_FD001.txt) for use in the dashboard.
Test data represents engines in different health states (not all failed),
which is more realistic for a production monitoring dashboard.

Process:
1. Load test data (test_FD001.txt)
2. Load true RUL values (RUL_FD001.txt)
3. Add RUL to test data
4. Rename s_N columns to sensor_N for consistency
5. Save as fd001_test_prepared.parquet

Author: Franklin Ramos
Date: 2026-02-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loading import load_fd001_test, add_true_rul_to_test
from src.config import PROCESSED_DATA_DIR


def prepare_test_data():
    """Prepare test data for the dashboard."""

    print("=" * 70)
    print("TEST DATA PREPARATION FOR DASHBOARD")
    print("=" * 70)

    # 1. Cargar datos de test
    print("\n1. Loading test data (test_FD001.txt)...")
    df_test = load_fd001_test()
    print(
        f"   ✓ Loaded {len(df_test)} records from {df_test['unit_id'].nunique()} engines"
    )

    # 2. Agregar RUL real a los datos de test
    print("\n2. Adding true RUL (RUL_FD001.txt)...")
    df_test = add_true_rul_to_test(df_test)
    print(f"   ✓ RUL added")

    # 3. Renombrar columnas s_N a sensor_N para consistencia con datos de entrenamiento
    print("\n3. Renaming sensor columns...")
    sensor_rename = {f"s_{i}": f"sensor_{i}" for i in range(1, 27)}
    df_test = df_test.rename(columns=sensor_rename)

    # Also rename op_N to op_setting_N for consistency
    op_rename = {f"op_{i}": f"op_setting_{i}" for i in range(1, 4)}
    df_test = df_test.rename(columns=op_rename)
    print(f"   ✓ Columns renamed")

    # 4. Verificar estadísticas de RUL
    print("\n4. RUL statistics for test data:")
    print(f"   - Total engines: {df_test['unit_id'].nunique()}")
    print(f"   - RUL min: {df_test['RUL'].min():.1f} cycles")
    print(f"   - RUL max: {df_test['RUL'].max():.1f} cycles")
    print(f"   - RUL mean: {df_test['RUL'].mean():.1f} cycles")
    print(f"   - RUL median: {df_test['RUL'].median():.1f} cycles")

    # Compute distribution by category (last values per engine)
    last_rul_per_engine = df_test.groupby("unit_id")["RUL"].last()
    critical = (last_rul_per_engine < 30).sum()
    warning = ((last_rul_per_engine >= 30) & (last_rul_per_engine < 70)).sum()
    healthy = (last_rul_per_engine >= 70).sum()

    print(f"\n   Engine distribution (last known RUL):")
    print(
        f"   - 🔴 Critical (RUL < 30):    {critical:3d} engines ({critical/len(last_rul_per_engine)*100:5.1f}%)"
    )
    print(
        f"   - 🟡 Warning (30-70):        {warning:3d} engines ({warning/len(last_rul_per_engine)*100:5.1f}%)"
    )
    print(
        f"   - 🟢 Healthy (RUL >= 70):    {healthy:3d} engines ({healthy/len(last_rul_per_engine)*100:5.1f}%)"
    )

    # 5. Guardar datos procesados
    output_path = PROCESSED_DATA_DIR / "fd001_test_prepared.parquet"
    print(f"\n5. Saving processed data...")
    print(f"   Path: {output_path}")

    df_test.to_parquet(output_path, index=False, compression="snappy")
    print(f"   ✓ File saved: {output_path.name}")
    print(f"   Size: {len(df_test)} rows × {len(df_test.columns)} columns")

    # 6. Verificar columnas
    print(f"\n6. Columns in processed file:")
    print(f"   {df_test.columns.tolist()}")

    print("\n" + "=" * 70)
    print("✅ TEST DATA PREPARED SUCCESSFULLY")
    print("=" * 70)
    print("\nNext step:")
    print("  → Update app.py to use 'fd001_test_prepared.parquet'")
    print("  → instead of 'fd001_prepared.parquet'")
    print("=" * 70)

    return df_test


if __name__ == "__main__":
    try:
        df = prepare_test_data()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
