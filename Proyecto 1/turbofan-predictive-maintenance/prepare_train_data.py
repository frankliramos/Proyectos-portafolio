#!/usr/bin/env python3
"""
Prepare Training Data for Analysis

Este script prepara los datos de entrenamiento (train_FD001.txt) de manera
consistente con los datos de prueba para análisis y visualización.

Proceso:
1. Cargar datos de train (train_FD001.txt)
2. Calcular RUL para cada registro
3. Renombrar columnas s_N a sensor_N para consistencia
4. Guardar como fd001_prepared.parquet

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
    """Prepara los datos de entrenamiento para análisis y dashboard."""
    
    print("="*70)
    print("PREPARACIÓN DE DATOS DE ENTRENAMIENTO")
    print("="*70)
    
    # 1. Cargar datos de train
    print("\n1. Cargando datos de entrenamiento (train_FD001.txt)...")
    df_train = load_fd001_train()
    print(f"   ✓ Cargados {len(df_train)} registros de {df_train['unit_id'].nunique()} motores")
    
    # 2. Agregar RUL
    print("\n2. Calculando RUL...")
    df_train = add_rul_to_train(df_train)
    print(f"   ✓ RUL calculado")
    
    # 3. Renombrar columnas para consistencia
    print("\n3. Renombrando columnas de sensores...")
    sensor_rename = {f's_{i}': f'sensor_{i}' for i in range(1, 22)}
    df_train = df_train.rename(columns=sensor_rename)
    
    op_rename = {f'op_{i}': f'op_setting_{i}' for i in range(1, 4)}
    df_train = df_train.rename(columns=op_rename)
    print(f"   ✓ Columnas renombradas")
    
    # 4. Verificar estadísticas
    print("\n4. Estadísticas de RUL en datos de entrenamiento:")
    print(f"   - Motores totales: {df_train['unit_id'].nunique()}")
    print(f"   - Registros totales: {len(df_train)}")
    print(f"   - RUL mínimo: {df_train['RUL'].min():.1f} ciclos")
    print(f"   - RUL máximo: {df_train['RUL'].max():.1f} ciclos")
    print(f"   - RUL promedio: {df_train['RUL'].mean():.1f} ciclos")
    print(f"   - RUL mediana: {df_train['RUL'].median():.1f} ciclos")
    
    # Calcular distribución de motores por estado final
    last_rul_per_engine = df_train.groupby('unit_id')['RUL'].last()
    critical = (last_rul_per_engine < 30).sum()
    warning = ((last_rul_per_engine >= 30) & (last_rul_per_engine < 70)).sum()
    healthy = (last_rul_per_engine >= 70).sum()
    
    print(f"\n   Distribución de motores (último RUL conocido):")
    print(f"   - 🔴 Críticos (RUL < 30):    {critical:3d} motores ({critical/len(last_rul_per_engine)*100:5.1f}%)")
    print(f"   - 🟡 Precaución (30-70):     {warning:3d} motores ({warning/len(last_rul_per_engine)*100:5.1f}%)")
    print(f"   - 🟢 Saludables (RUL >= 70): {healthy:3d} motores ({healthy/len(last_rul_per_engine)*100:5.1f}%)")
    
    # 5. Guardar
    output_path = PROCESSED_DATA_DIR / "fd001_prepared.parquet"
    print(f"\n5. Guardando datos procesados...")
    print(f"   Ruta: {output_path}")
    
    df_train.to_parquet(output_path, index=False, compression='snappy')
    print(f"   ✓ Archivo guardado: {output_path.name}")
    print(f"   Tamaño: {len(df_train)} filas × {len(df_train.columns)} columnas")
    
    # 6. Verificar columnas
    print(f"\n6. Columnas en archivo procesado:")
    print(f"   {df_train.columns.tolist()}")
    
    print("\n" + "="*70)
    print("✅ DATOS DE ENTRENAMIENTO PREPARADOS EXITOSAMENTE")
    print("="*70)
    print("\nNota: Los datos incluyen TODOS los ciclos para cada motor.")
    print("Para evaluación de modelos, usar .groupby('unit_id').tail(1)")
    print("para obtener solo el último ciclo de cada motor.")
    print("="*70)
    
    return df_train


if __name__ == "__main__":
    try:
        df = prepare_train_data()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
