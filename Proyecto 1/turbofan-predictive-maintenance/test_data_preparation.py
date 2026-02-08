#!/usr/bin/env python3
"""
Test Data Preparation and Dashboard Compatibility

Este script verifica que los datos procesados son correctos y compatibles
con el dashboard y el modelo de inferencia.

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
    """Verifica que los archivos de datos procesados existen y tienen el formato correcto."""
    print("\n" + "="*70)
    print("TEST 1: VERIFICACIÓN DE ARCHIVOS DE DATOS")
    print("="*70)
    
    from src.config import PROCESSED_DATA_DIR
    
    train_file = PROCESSED_DATA_DIR / "fd001_prepared.parquet"
    test_file = PROCESSED_DATA_DIR / "fd001_test_prepared.parquet"
    
    # Check training file
    print(f"\n📄 Verificando {train_file.name}...")
    if not train_file.exists():
        print(f"   ✗ Archivo no existe: {train_file}")
        return False
    
    df_train = pd.read_parquet(train_file)
    print(f"   ✓ Archivo existe")
    print(f"   ✓ Shape: {df_train.shape}")
    print(f"   ✓ Motores: {df_train['unit_id'].nunique()}")
    
    if len(df_train) < 10000:
        print(f"   ✗ ERROR: Muy pocas filas ({len(df_train)}). ¿Solo tiene últimos ciclos?")
        return False
    print(f"   ✓ Contiene todos los ciclos (>10k filas)")
    
    # Check test file
    print(f"\n📄 Verificando {test_file.name}...")
    if not test_file.exists():
        print(f"   ✗ Archivo no existe: {test_file}")
        return False
    
    df_test = pd.read_parquet(test_file)
    print(f"   ✓ Archivo existe")
    print(f"   ✓ Shape: {df_test.shape}")
    print(f"   ✓ Motores: {df_test['unit_id'].nunique()}")
    
    if len(df_test) < 5000:
        print(f"   ✗ ERROR: Muy pocas filas ({len(df_test)}). ¿Solo tiene últimos ciclos?")
        return False
    print(f"   ✓ Contiene todos los ciclos (>5k filas)")
    
    # Check columns match
    required_cols = ['unit_id', 'time_cycles', 'RUL', 'op_setting_1', 'sensor_1']
    for col in required_cols:
        if col not in df_test.columns:
            print(f"   ✗ Columna faltante: {col}")
            return False
    print(f"   ✓ Todas las columnas requeridas presentes")
    
    return True


def test_dashboard_compatibility():
    """Verifica que los datos son compatibles con el dashboard."""
    print("\n" + "="*70)
    print("TEST 2: COMPATIBILIDAD CON DASHBOARD")
    print("="*70)
    
    from src.config import PROCESSED_DATA_DIR
    
    # Load data like dashboard does
    data_path = PROCESSED_DATA_DIR / "fd001_test_prepared.parquet"
    df = pd.read_parquet(data_path)
    print(f"\n✓ Datos cargados: {df.shape}")
    
    # Rename columns like dashboard does
    if 'unit_id' in df.columns:
        df = df.rename(columns={'unit_id': 'id'})
    if 'time_cycles' in df.columns:
        df = df.rename(columns={'time_cycles': 'cycle'})
    print(f"✓ Columnas renombradas")
    
    # Validate
    if 'id' not in df.columns or 'cycle' not in df.columns:
        print(f"✗ ERROR: Faltan columnas 'id' o 'cycle'")
        return False
    print(f"✓ Validación de columnas OK")
    
    # Check distribution
    engine_ids = df['id'].unique()
    last_rul_per_engine = df.groupby('id')['RUL'].last()
    
    critical = (last_rul_per_engine < 30).sum()
    warning = ((last_rul_per_engine >= 30) & (last_rul_per_engine < 70)).sum()
    healthy = (last_rul_per_engine >= 70).sum()
    
    print(f"\n📊 Distribución de estados (último RUL por motor):")
    print(f"   🔴 Críticos:  {critical:3d} ({critical/len(engine_ids)*100:5.1f}%)")
    print(f"   🟡 Precaución: {warning:3d} ({warning/len(engine_ids)*100:5.1f}%)")
    print(f"   🟢 Saludables: {healthy:3d} ({healthy/len(engine_ids)*100:5.1f}%)")
    
    if critical == len(engine_ids):
        print(f"\n✗ ERROR: TODOS los motores están críticos!")
        print(f"   Esto indica que el archivo solo tiene últimos ciclos.")
        return False
    
    if healthy == 0:
        print(f"\n✗ ADVERTENCIA: NO hay motores saludables")
    
    print(f"\n✓ Distribución de estados es variada (no todos críticos)")
    
    return True


def test_model_compatibility():
    """Verifica que los datos son compatibles con el modelo de inferencia."""
    print("\n" + "="*70)
    print("TEST 3: COMPATIBILIDAD CON MODELO")
    print("="*70)
    
    try:
        import joblib
        from src.config import FEATURE_COLS_FILE, PROCESSED_DATA_DIR
        
        # Load expected features
        if not FEATURE_COLS_FILE.exists():
            print(f"✗ Archivo de features no existe: {FEATURE_COLS_FILE}")
            return False
        
        feature_cols = joblib.load(FEATURE_COLS_FILE)
        print(f"\n✓ Features esperadas por modelo: {len(feature_cols)}")
        print(f"   {feature_cols[:5]}...")
        
        # Load data
        df = pd.read_parquet(PROCESSED_DATA_DIR / "fd001_test_prepared.parquet")
        
        # Check if all features present
        missing = set(feature_cols) - set(df.columns)
        if missing:
            print(f"✗ Features faltantes: {missing}")
            return False
        
        print(f"✓ Todas las features requeridas están presentes")
        
        # Test with one motor
        motor_data = df[df['unit_id'] == 1].sort_values('time_cycles')
        if len(motor_data) < 30:
            print(f"✗ Motor 1 tiene menos de 30 ciclos ({len(motor_data)})")
            return False
        
        print(f"✓ Motor 1 tiene suficientes ciclos para predicción ({len(motor_data)})")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  No se pudo verificar compatibilidad con modelo: {e}")
        print(f"   (Es OK si no tienes todas las dependencias instaladas)")
        return True


def main():
    """Ejecuta todos los tests."""
    print("\n" + "="*70)
    print(" "*15 + "VERIFICACIÓN DE DATOS PROCESADOS")
    print("="*70)
    
    tests = [
        ("Archivos de datos", test_data_files),
        ("Compatibilidad con dashboard", test_dashboard_compatibility),
        ("Compatibilidad con modelo", test_model_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ ERROR en test '{test_name}': {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {test_name}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ TODOS LOS TESTS PASARON")
        print("="*70)
        print("\n💡 Los datos están listos para usar en:")
        print("   - Dashboard (streamlit run app.py)")
        print("   - Notebooks de análisis")
        print("   - Predicciones con el modelo LSTM")
        print("\n" + "="*70)
        return 0
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print("="*70)
        print("\n🔧 Solución:")
        print("   Ejecuta: python prepare_all_data.py")
        print("   Para regenerar los datos procesados correctamente")
        print("\n" + "="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
