# Checklist del Portafolio de Data Science

Lista de tareas pendientes organizadas por prioridad para llevar el portafolio a nivel profesional senior.

---

## 🔴 CRÍTICO — Impacto directo en impresión del reclutador

- [ ] Agregar email real de contacto en todos los READMEs (reemplazar "Available upon request")
- [ ] Agregar URL del perfil de LinkedIn en todos los READMEs
- [ ] Generar y hacer commit de notebooks Jupyter para Proyecto 3 (EDA + Modelado de churn bancario)
- [ ] Generar y hacer commit de notebooks Jupyter para Proyecto 4 (EDA + Modelado de recomendaciones)
- [ ] Agregar archivos de modelos entrenados a `Proyecto 3/models/` (o script de descarga automática)
- [ ] Agregar archivos de modelos entrenados a `Proyecto 4/models/` (o script de descarga automática)
- [ ] Agregar dataset real a `Proyecto 3/data/` (Kaggle Bank Customer Churn: `Churn_Modelling.csv`)
- [ ] Agregar dataset real a `Proyecto 4/data/` (Amazon Reviews o MovieLens)
- [ ] Generar archivos de resultados para Proyecto 3: `results/metrics_comparison.csv`, `results/feature_importance.csv`, `results/confusion_matrix.png`
- [ ] Generar archivos de resultados para Proyecto 4: `results/metrics_comparison.csv`, `results/recommendation_samples.csv`

---

## 🟠 IMPORTANTE — Mejora significativa de la presentación

- [ ] Tomar capturas de pantalla de los dashboards de Proyecto 2, 3 y 4 y guardarlas en `assets/`
- [ ] Agregar `data_forecast.csv` o fuente de datos real para Proyecto 2
- [ ] Desplegar al menos un dashboard en Streamlit Cloud y agregar enlace en el README
- [ ] Agregar badge de cobertura de tests junto al badge de CI en los READMEs principales
- [ ] Revisar y limpiar notebooks con nombres incorrectos en Proyecto 3 (actualmente tienen nombres del Financial Phrase Bank)

---

## 🟡 MODERADO — Detalles de calidad profesional

- [ ] Verificar que `requirements.txt` esté presente y actualizado en Proyecto 3
- [ ] Limpiar el repositorio: eliminar `test_plot.png` si existe, agregar `catboost_info/` al `.gitignore`
- [ ] Evaluar si fusionar el contenido del repo `Portafolio` con `Proyecto 3` para consolidar trabajo
- [ ] Agregar README_ES.md a `Proyecto 2` si se requiere consistencia con otros proyectos (ya existe — revisar)
- [ ] Revisar que todos los README.md de proyectos tengan sección de "Resultados" con métricas concretas

---

## 🔵 MEJORAS DE NIVEL SENIOR — Para destacar sobre otros candidatos

- [ ] Agregar proyecto MLOps/Producción: API REST con FastAPI + Docker + model serving
- [ ] Agregar proyecto Big Data: procesamiento con PySpark o Dask
- [ ] Agregar proyecto NLP/LLM: clasificación de texto, RAG, o fine-tuning de modelo de lenguaje
- [ ] Escribir artículos técnicos (blog posts) sobre los proyectos y enlazarlos desde los READMEs
- [ ] Agregar proyecto de inferencia causal (A/B testing, uplift modeling)
- [ ] Configurar GitHub Actions para correr tests automáticos en todos los proyectos
- [ ] Agregar pre-commit hooks para linting y formateo automático de código

---

## ✅ COMPLETADO

- [x] Eliminar `faltante.txt` de la raíz (documento interno de evaluación)
- [x] Agregar badge de CI/CD en `README.md` y `README_ES.md` de la raíz
- [x] Corregir typo `pip install -r requirements` → `pip install -r requirements.txt` en `Proyecto 2/README_ES.md`
- [x] Actualizar `Proyecto 3/src/__init__.py` con imports explícitos de todos los módulos
- [x] Actualizar `Proyecto 4/src/__init__.py` con imports explícitos de todos los módulos
- [x] Corregir `Proyecto 3/data/raw/README.txt` con documentación correcta del Bank Customer Churn Dataset
- [x] Crear `Proyecto 2/MODEL_CARD.md` — Model Card profesional para el sistema de pronóstico de ventas
- [x] Crear `Proyecto 3/MODEL_CARD.md` — Model Card profesional para el clasificador de churn bancario
- [x] Crear `Proyecto 2/QUICKSTART.md` — Guía de inicio rápido para el dashboard de ventas
- [x] Crear `Proyecto 3/QUICKSTART.md` — Guía de inicio rápido para el dashboard de churn
- [x] Crear `Proyecto 4/QUICKSTART.md` — Guía de inicio rápido para el dashboard de recomendaciones
