#!/usr/bin/env python3
"""
Prepare All Data for Dashboard and Analysis

Este script ejecuta la preparación completa de datos de entrenamiento y prueba.

Author: Franklin Ramos
Date: 2026-02-04
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("\n" + "="*70)
    print(" "*20 + "PREPARACIÓN COMPLETA DE DATOS")
    print("="*70)
    
    print("\n📊 Este script preparará:")
    print("  1. Datos de entrenamiento (train_FD001.txt) → fd001_prepared.parquet")
    print("  2. Datos de prueba (test_FD001.txt) → fd001_test_prepared.parquet")
    print("\n⚠️  IMPORTANTE: Los archivos incluirán TODOS los ciclos de cada motor.")
    print("    Para evaluación de modelos, usar .groupby('unit_id').tail(1)")
    print("    para obtener solo el último ciclo.\n")
    
    input("Presiona Enter para continuar...")
    
    # Import here to avoid errors if modules not ready
    from prepare_train_data import prepare_train_data
    from prepare_test_data import prepare_test_data
    
    try:
        # Prepare training data
        print("\n" + "="*70)
        print("PASO 1/2: DATOS DE ENTRENAMIENTO")
        print("="*70)
        df_train = prepare_train_data()
        
        # Prepare test data
        print("\n" + "="*70)
        print("PASO 2/2: DATOS DE PRUEBA")
        print("="*70)
        df_test = prepare_test_data()
        
        # Summary
        print("\n" + "="*70)
        print("✅ PREPARACIÓN COMPLETA FINALIZADA")
        print("="*70)
        print(f"\n📁 Archivos generados:")
        print(f"  - fd001_prepared.parquet:      {len(df_train):6,} filas × {len(df_train.columns):2} columnas")
        print(f"  - fd001_test_prepared.parquet: {len(df_test):6,} filas × {len(df_test.columns):2} columnas")
        
        print(f"\n🔧 Motores disponibles:")
        print(f"  - Entrenamiento: {df_train['unit_id'].nunique()} motores")
        print(f"  - Prueba:        {df_test['unit_id'].nunique()} motores")
        
        print("\n✨ Los datos están listos para:")
        print("  → Dashboard de monitoreo (app.py)")
        print("  → Análisis exploratorio (notebooks/)")
        print("  → Entrenamiento de modelos")
        
        print("\n💡 Próximos pasos:")
        print("  1. Ejecutar dashboard: streamlit run app.py")
        print("  2. Ver notebooks en: notebooks/")
        print("  3. Entrenar modelos con datos completos")
        
        print("\n" + "="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error durante la preparación: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
