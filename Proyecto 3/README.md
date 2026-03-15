# Financial Sentiment Analysis - FinBERT

## English

### Overview
This project delivers a **financial sentiment analysis** system using a fine-tuned
FinBERT model. It classifies financial sentences into **negative**, **neutral**, and
**positive** sentiment, with an interactive dashboard and a FastAPI inference service.

### Business Problem
Financial teams need fast, consistent interpretation of market news. Manual review
does not scale and creates delays in decision-making. The solution automates
sentiment scoring to support risk monitoring and trading workflows.

### Data
- Dataset: Financial PhraseBank (sentences_allagree)
- Size: 14,830 raw rows; cleaned and split into train/val/test
- Labels: negative, neutral, positive

### Modeling
- Baseline: Logistic Regression
- Fine-tuned: FinBERT (ProsusAI/finbert)
- Metrics tracked: accuracy, macro F1, classification report

### Results (from notebooks)
- Baseline accuracy: 0.75
- FinBERT accuracy: 0.88, macro F1: ~0.86

### Project Structure
```
Proyecto 3/
├── dashboard/                 # Streamlit demo UI
│   └── app.py
├── notebooks/                 # EDA, baselines, finetuning, interpretability
│   ├── 01_eda_financial_phrasebank.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_finbert_finetuning.ipynb
│   ├── 04_model_interpretability.ipynb
│   ├── 05_final_evaluation.ipynb
│   └── 06_fastapi_inference.ipynb
├── models/                    # Fine-tuned model artifacts
├── reports/                   # Metrics and figures
└── src/
	├── __init__.py
	└── config.py
```

### How to Run
1. Install dependencies.
2. Run notebooks in order for EDA, baseline, fine-tuning, and evaluation.
3. Launch the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

### API (FastAPI)
The notebook `06_fastapi_inference.ipynb` starts a local FastAPI server:
- Base URL: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs

---

## Espanol

### Resumen
Este proyecto implementa un **sistema de analisis de sentimiento financiero**
con un modelo FinBERT fine-tuned. Clasifica frases en **negative**, **neutral** y
**positive**, con un dashboard interactivo y un servicio de inferencia FastAPI.

### Problema de negocio
El analisis manual de noticias no escala y retrasa decisiones. El sistema
automatiza el scoring para apoyar monitoreo de riesgo y analisis de mercado.

### Datos
- Dataset: Financial PhraseBank (sentences_allagree)
- Tamano: 14,830 filas en crudo; split train/val/test
- Etiquetas: negative, neutral, positive

### Modelado
- Baseline: Logistic Regression
- Fine-tuned: FinBERT (ProsusAI/finbert)
- Metricas: accuracy, macro F1, classification report

### Resultados (segun notebooks)
- Baseline accuracy: 0.75
- FinBERT accuracy: 0.88, macro F1: ~0.86

### Ejecucion
1. Instalar dependencias.
2. Ejecutar notebooks en orden.
3. Lanzar el dashboard:
```bash
streamlit run dashboard/app.py
```

## 👤 Author

**Franklin Ramos**
- 📧 Email: franklin.ram.riv@gmail.com
- 💼 LinkedIn: [linkedin.com/in/franklin-ramos-riveros-62b70083](https://www.linkedin.com/in/franklin-ramos-riveros-62b70083/)
- 💼 GitHub: [github.com/frankliramos](https://github.com/frankliramos)
- Portfolio: [GitHub Portfolio](https://github.com/frankliramos/Proyectos-portafolio)
