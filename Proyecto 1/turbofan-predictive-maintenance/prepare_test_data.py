#!/usr/bin/env python3
"""
Prepare Test Data for Dashboard

Este script prepara los datos de prueba (test_FD001.txt) para ser usados
en el dashboard. Los datos de prueba representan motores en diferentes
estados de salud (no todos fallidos), lo cual es más realista para un
dashboard de monitoreo de producción.

Proceso:
1. Cargar datos de test (test_FD001.txt)
2. Cargar RUL reales (RUL_FD001.txt)
3. Agregar RUL a los datos de test
4. Renombrar columnas s_N a sensor_N para consistencia
5. Guardar como fd001_test_prepared.parquet

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
    """Prepara los datos de prueba para el dashboard."""
    
    print("="*70)
    print("PREPARACIÓN DE DATOS DE PRUEBA PARA DASHBOARD")
    print("="*70)
    
    # 1. Cargar datos de test
    print("\n1. Cargando datos de test (test_FD001.txt)...")
    df_test = load_fd001_test()
    print(f"   ✓ Cargados {len(df_test)} registros de {df_test['unit_id'].nunique()} motores")
    
    # 2. Agregar RUL real a los datos de test
    print("\n2. Agregando RUL real (RUL_FD001.txt)...")
    df_test = add_true_rul_to_test(df_test)
    print(f"   ✓ RUL agregado")
    
    # 3. Renombrar columnas s_N a sensor_N para consistencia con datos de entrenamiento
    print("\n3. Renombrando columnas de sensores...")
    sensor_rename = {f's_{i}': f'sensor_{i}' for i in range(1, 27)}
    df_test = df_test.rename(columns=sensor_rename)
    
    # También renombrar op_N a op_setting_N para consistencia
    op_rename = {f'op_{i}': f'op_setting_{i}' for i in range(1, 4)}
    df_test = df_test.rename(columns=op_rename)
    print(f"   ✓ Columnas renombradas")
    
    # 4. Verificar estadísticas de RUL
    print("\n4. Estadísticas de RUL en datos de test:")
    print(f"   - Motores totales: {df_test['unit_id'].nunique()}")
    print(f"   - RUL mínimo: {df_test['RUL'].min():.1f} ciclos")
    print(f"   - RUL máximo: {df_test['RUL'].max():.1f} ciclos")
    print(f"   - RUL promedio: {df_test['RUL'].mean():.1f} ciclos")
    print(f"   - RUL mediana: {df_test['RUL'].median():.1f} ciclos")
    
    # Calcular distribución por categorías (últimos valores por motor)
    last_rul_per_engine = df_test.groupby('unit_id')['RUL'].last()
    critical = (last_rul_per_engine < 30).sum()
    warning = ((last_rul_per_engine >= 30) & (last_rul_per_engine < 70)).sum()
    healthy = (last_rul_per_engine >= 70).sum()
    
    print(f"\n   Distribución de motores (último RUL conocido):")
    print(f"   - 🔴 Críticos (RUL < 30):    {critical:3d} motores ({critical/len(last_rul_per_engine)*100:5.1f}%)")
    print(f"   - 🟡 Precaución (30-70):     {warning:3d} motores ({warning/len(last_rul_per_engine)*100:5.1f}%)")
    print(f"   - 🟢 Saludables (RUL >= 70): {healthy:3d} motores ({healthy/len(last_rul_per_engine)*100:5.1f}%)")
    
    # 5. Guardar datos procesados
    output_path = PROCESSED_DATA_DIR / "fd001_test_prepared.parquet"
    print(f"\n5. Guardando datos procesados...")
    print(f"   Ruta: {output_path}")
    
    df_test.to_parquet(output_path, index=False, compression='snappy')
    print(f"   ✓ Archivo guardado: {output_path.name}")
    print(f"   Tamaño: {len(df_test)} filas × {len(df_test.columns)} columnas")
    
    # 6. Verificar columnas
    print(f"\n6. Columnas en archivo procesado:")
    print(f"   {df_test.columns.tolist()}")
    
    print("\n" + "="*70)
    print("✅ DATOS DE PRUEBA PREPARADOS EXITOSAMENTE")
    print("="*70)
    print("\nPróximo paso:")
    print("  → Actualizar app.py para usar 'fd001_test_prepared.parquet'")
    print("  → en lugar de 'fd001_prepared.parquet'")
    print("="*70)
    
    return df_test


if __name__ == "__main__":
    try:
        df = prepare_test_data()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
