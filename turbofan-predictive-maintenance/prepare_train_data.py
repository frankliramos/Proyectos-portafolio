#!/usr/bin/env python3
"""
Prepare Training Data for Analysis

This script prepares training data (train_FD001.txt) in a way that is
consistent with the test data for analysis and visualization.

Process:
1. Load train data (train_FD001.txt)
2. Compute RUL for each record
3. Rename s_N columns to sensor_N for consistency
4. Save as fd001_prepared.parquet

Author: Franklin Ramos
Date: 2026-02-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loading import load_fd001_train, add_rul_to_train
from src.config import PROCESSED_DATA_DIR


def prepare_train_data():
    """Prepare training data for analysis and the dashboard."""

    print("=" * 70)
    print("TRAINING DATA PREPARATION")
    print("=" * 70)

    # 1. Cargar datos de train
    print("\n1. Loading training data (train_FD001.txt)...")
    df_train = load_fd001_train()
    print(
        f"   ✓ Loaded {len(df_train)} records from {df_train['unit_id'].nunique()} engines"
    )

    # 2. Agregar RUL
    print("\n2. Computing RUL...")
    df_train = add_rul_to_train(df_train)
    print(f"   ✓ RUL computed")

    # 3. Renombrar columnas para consistencia
    print("\n3. Renaming sensor columns...")
    sensor_rename = {f"s_{i}": f"sensor_{i}" for i in range(1, 22)}
    df_train = df_train.rename(columns=sensor_rename)

    op_rename = {f"op_{i}": f"op_setting_{i}" for i in range(1, 4)}
    df_train = df_train.rename(columns=op_rename)
    print(f"   ✓ Columns renamed")

    # 4. Verificar estadísticas
    print("\n4. RUL statistics for training data:")
    print(f"   - Total engines: {df_train['unit_id'].nunique()}")
    print(f"   - Total records: {len(df_train)}")
    print(f"   - RUL min: {df_train['RUL'].min():.1f} cycles")
    print(f"   - RUL max: {df_train['RUL'].max():.1f} cycles")
    print(f"   - RUL mean: {df_train['RUL'].mean():.1f} cycles")
    print(f"   - RUL median: {df_train['RUL'].median():.1f} cycles")

    # Compute engine distribution by final state
    last_rul_per_engine = df_train.groupby("unit_id")["RUL"].last()
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

    # 5. Guardar
    output_path = PROCESSED_DATA_DIR / "fd001_prepared.parquet"
    print(f"\n5. Saving processed data...")
    print(f"   Path: {output_path}")

    df_train.to_parquet(output_path, index=False, compression="snappy")
    print(f"   ✓ File saved: {output_path.name}")
    print(f"   Size: {len(df_train)} rows × {len(df_train.columns)} columns")

    # 6. Verificar columnas
    print(f"\n6. Columns in processed file:")
    print(f"   {df_train.columns.tolist()}")

    print("\n" + "=" * 70)
    print("✅ TRAINING DATA PREPARED SUCCESSFULLY")
    print("=" * 70)
    print("\nNote: Data includes ALL cycles for each engine.")
    print("For model evaluation, use .groupby('unit_id').tail(1)")
    print("to keep only the last cycle per engine.")
    print("=" * 70)

    return df_train


if __name__ == "__main__":
    try:
        df = prepare_train_data()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
