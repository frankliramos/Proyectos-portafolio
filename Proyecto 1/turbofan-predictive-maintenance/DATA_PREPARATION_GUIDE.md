# Guía de Preparación de Datos

## Descripción General

Este proyecto incluye scripts para preparar los datos del dataset NASA CMAPSS FD001 tanto para entrenamiento de modelos como para visualización en el dashboard.

## Archivos de Datos

### Datos Crudos (data/raw/)
- `train_FD001.txt`: Datos de entrenamiento (100 motores, ~20,631 ciclos totales)
- `test_FD001.txt`: Datos de prueba (100 motores, ~13,096 ciclos totales)
- `RUL_FD001.txt`: Valores reales de RUL para el conjunto de prueba

### Datos Procesados (data/processed/)
- `fd001_prepared.parquet`: Datos de entrenamiento preparados (TODOS los ciclos)
- `fd001_test_prepared.parquet`: Datos de prueba preparados (TODOS los ciclos)

## Scripts de Preparación

### 1. prepare_train_data.py
Prepara los datos de entrenamiento:
```bash
python prepare_train_data.py
```

Proceso:
1. Carga train_FD001.txt
2. Calcula RUL para cada ciclo: `RUL = max(time_cycles) - time_cycles`
3. Renombra columnas: `s_1` → `sensor_1`, `op_1` → `op_setting_1`
4. Guarda como fd001_prepared.parquet con **TODOS los ciclos**

### 2. prepare_test_data.py
Prepara los datos de prueba:
```bash
python prepare_test_data.py
```

Proceso:
1. Carga test_FD001.txt y RUL_FD001.txt
2. Asigna RUL real a cada ciclo usando el valor final conocido
3. Renombra columnas para consistencia
4. Guarda como fd001_test_prepared.parquet con **TODOS los ciclos**

### 3. prepare_all_data.py
Ejecuta ambos scripts en secuencia:
```bash
python prepare_all_data.py
```

## ⚠️ IMPORTANTE: Diferencia entre Dashboard y Evaluación de Modelos

### Para el Dashboard (app.py)
Los archivos procesados incluyen **TODOS los ciclos** de cada motor para:
- Visualizar la evolución temporal de sensores
- Mostrar el historial completo de cada motor
- Permitir análisis de tendencias
- Mostrar diferentes estados de salud a lo largo del tiempo

### Para Evaluación de Modelos (notebooks)
En los notebooks de entrenamiento, **debes filtrar al último ciclo** usando:
```python
# Para evaluación del modelo, usar solo el último ciclo de cada motor
test_last_cycles = test_data.groupby('unit_id').tail(1)
```

Esto es porque:
- En un escenario real, solo predices el RUL una vez al final
- Las métricas (MAE, RMSE, R²) se calculan sobre predicciones finales
- Evita inflar artificialmente las métricas con múltiples predicciones del mismo motor

## Estructura de Datos

### Columnas en Archivos Procesados
```
unit_id         : ID del motor (1-100)
time_cycles     : Número de ciclo (tiempo)
op_setting_1    : Configuración operacional 1
op_setting_2    : Configuración operacional 2
op_setting_3    : Configuración operacional 3
sensor_1        : Lectura del sensor 1
...
sensor_21       : Lectura del sensor 21
RUL             : Remaining Useful Life (ciclos restantes)
```

### Compatibilidad con Dashboard
El dashboard (`app.py`) espera columnas `id` y `cycle`, pero tiene código para renombrarlas automáticamente:
```python
if 'unit_id' in df.columns:
    df = df.rename(columns={'unit_id': 'id'})
if 'time_cycles' in df.columns:
    df = df.rename(columns={'time_cycles': 'cycle'})
```

## Estadísticas Esperadas

### Datos de Entrenamiento
- Total: ~20,631 registros
- Motores: 100
- RUL mínimo: 0 (al momento de falla)
- RUL máximo: ~361 ciclos

### Datos de Prueba
- Total: ~13,096 registros
- Motores: 100
- RUL mínimo: ~7 ciclos
- RUL máximo: ~340 ciclos
- Distribución de estados (último ciclo):
  - 🔴 Críticos (RUL < 30): ~25%
  - 🟡 Precaución (30-70): ~17%
  - 🟢 Saludables (RUL ≥ 70): ~58%

## Solución de Problemas

### Problema: Dashboard muestra 100% motores críticos
**Causa**: Archivo procesado contiene solo el último ciclo de cada motor (100 filas)

**Solución**: Ejecutar los scripts de preparación para regenerar los datos con todos los ciclos:
```bash
python prepare_all_data.py
```

### Problema: Columnas faltantes en datos procesados
**Causa**: Los datos no fueron procesados con los scripts actualizados

**Solución**: Regenerar los archivos procesados

### Problema: Predicciones del modelo fallan
**Causa**: Falta algún sensor o configuración operacional

**Verificación**:
```python
import joblib
feature_cols = joblib.load('models/feature_cols_v1.pkl')
print(feature_cols)  # Debe mostrar 24 columnas
```

## Notas Adicionales

1. Los archivos `.parquet` son más eficientes que CSV para datos grandes
2. La compresión `snappy` ofrece un buen balance entre velocidad y tamaño
3. Los datos procesados se pueden leer fácilmente con pandas:
   ```python
   import pandas as pd
   df = pd.read_parquet('data/processed/fd001_test_prepared.parquet')
   ```
4. NO es necesario feature engineering adicional para usar el modelo LSTM entrenado
5. El modelo espera exactamente 24 features: 3 op_settings + 21 sensores

## Referencias

- Dataset: [NASA CMAPSS Turbofan Engine Degradation Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- Paper: Saxena et al. "Damage propagation modeling for aircraft engine run-to-failure simulation"
