# NOTA IMPORTANTE SOBRE USO DE `.tail(1)` EN NOTEBOOKS

## Para Desarrolladores y Usuarios de los Notebooks

### ⚠️ Contexto Importante

En los notebooks de este proyecto, verás código como:
```python
test_last_cycles = test_features.groupby("unit_id").tail(1).copy()
```

### ✅ Esto es CORRECTO para Evaluación de Modelos

El uso de `.tail(1)` (seleccionar solo el último ciclo de cada motor) es **correcto y necesario** en los notebooks de entrenamiento/evaluación de modelos porque:

1. **Estándar de la industria**: En mantenimiento predictivo, se predice el RUL una sola vez cuando el motor llega al final de su ciclo de operación antes del mantenimiento.

2. **Métricas precisas**: Calcular MAE, RMSE, R² sobre el último ciclo evita inflar artificialmente las métricas con múltiples predicciones del mismo motor.

3. **Alineación con producción**: Simula el escenario real donde haces una predicción final para decidir si mantener o reemplazar el motor.

### 📊 PERO NO uses `.tail(1)` para Preparar Datos del Dashboard

Los scripts de preparación de datos (`prepare_train_data.py`, `prepare_test_data.py`) **NO** usan `.tail(1)` porque:

1. **Dashboard necesita historial completo**: Para visualizar la evolución temporal de sensores y RUL
2. **Mostrar estados variados**: Motores en diferentes estados de salud (saludable → precaución → crítico)
3. **Análisis de tendencias**: Detectar patrones de degradación a lo largo del tiempo

## Resumen

| Propósito | Usar `.tail(1)` | Archivo | Razón |
|-----------|----------------|---------|-------|
| Entrenamiento de modelo | ✅ Sí | `notebooks/02_model_baseline_fd001.ipynb` | Evitar data leakage |
| Evaluación de modelo | ✅ Sí | `notebooks/03_model_lstm_fd001.ipynb` | Métricas estándar |
| Preparación para dashboard | ❌ No | `prepare_test_data.py` | Visualización temporal |
| Preparación para dashboard | ❌ No | `prepare_train_data.py` | Análisis completo |
| Dashboard en producción | ❌ No | `app.py` | Monitoreo en tiempo real |

## Ejemplo Práctico

### ❌ INCORRECTO para Dashboard:
```python
# Esto resultaría en que el dashboard muestre TODOS los motores como críticos
# porque solo verías el estado final de cada motor (al borde de la falla)
test_data = load_test_data()
test_data = test_data.groupby('unit_id').tail(1)  # ❌ MAL para dashboard
test_data.to_parquet('fd001_test_prepared.parquet')
```

### ✅ CORRECTO para Dashboard:
```python
# Incluye TODOS los ciclos para visualización temporal
test_data = load_test_data()
# NO filtrar aquí - guardar todos los ciclos
test_data.to_parquet('fd001_test_prepared.parquet')  # ✅ BIEN para dashboard
```

### ✅ CORRECTO para Evaluación de Modelo:
```python
# En notebook de evaluación
test_data = pd.read_parquet('fd001_test_prepared.parquet')
# Filtrar solo para calcular métricas
test_last_cycles = test_data.groupby('unit_id').tail(1)  # ✅ BIEN para métricas
predictions = model.predict(test_last_cycles)
mae = mean_absolute_error(test_last_cycles['RUL'], predictions)
```

## Referencias

- Ver: `DATA_PREPARATION_GUIDE.md` para más detalles sobre preparación de datos
- Ver: `prepare_test_data.py` para ver cómo se preparan datos del dashboard
- Ver: `notebooks/02_model_baseline_fd001.ipynb` para ver uso correcto en evaluación

## Pregunta Frecuente

**P: ¿Por qué el dashboard mostraba 100% de motores críticos?**

R: El archivo procesado contenía solo el último ciclo de cada motor (100 filas en lugar de 13,096). Como los datos de prueba de NASA muestran motores que operaron hasta casi fallar, ese último ciclo siempre tiene RUL bajo (crítico). La solución fue regenerar los datos con TODOS los ciclos usando `prepare_test_data.py`.
