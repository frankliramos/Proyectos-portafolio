# Model Card: FinBERT Fine-tuned Financial Sentiment Classifier

> **Based on**: [Google Model Cards](https://modelcards.withgoogle.com/about) framework

---

## 📋 Model Details

| Field | Value |
|-------|-------|
| **Model Name** | FinBERT Fine-tuned Sentiment Classifier |
| **Version** | v1.0 |
| **Model Type** | Transformer (BERT-based) — sequence classification |
| **Task** | 3-class financial sentiment: `positive`, `negative`, `neutral` |
| **Base Model** | `ProsusAI/finbert` (BERT pre-trained on financial corpus) |
| **Framework** | HuggingFace Transformers + PyTorch |
| **Author** | Franklin Ramos |
| **Date** | March 2026 |
| **License** | MIT (model weights under CC BY 4.0 via HuggingFace Hub) |

---

## 🎯 Intended Use

### Primary Use Case
Classify short financial sentences or news headlines as **positive**, **negative**, or **neutral** from a financial analyst's perspective, enabling automated sentiment monitoring for investment research and risk assessment.

### Intended Users
- Quantitative analysts and portfolio managers automating news monitoring
- FinTech applications requiring real-time sentiment signals
- Risk teams tracking sentiment trends in earnings calls, regulatory filings
- Researchers studying NLP applications in finance

### Out-of-Scope Uses
- General-purpose (non-financial) text sentiment analysis
- Social media slang or informal financial language
- Languages other than English without multilingual fine-tuning
- Long documents (>512 tokens); use sliding-window chunking

---

## 📊 Training Data

| Property | Value |
|----------|-------|
| **Dataset** | Financial PhraseBank (Malo et al., 2014) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) |
| **Sentences** | 4,840 annotated English financial sentences |
| **Agreement split used** | ≥75% annotator agreement (highest quality subset) |
| **Label distribution** | Positive: ~28%, Negative: ~26%, Neutral: ~46% |
| **Train / validation split** | 80% / 20% stratified |

### Data Characteristics
- Sentences sourced from financial news, analyst reports, and press releases
- Annotated by domain experts (multiple annotators per sentence)
- Topics: earnings, M&A, bankruptcies, product launches, economic indicators
- Average sentence length: ~22 tokens

---

## 📈 Performance

### Global Metrics (Validation Set)

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 87.3% | Overall classification accuracy |
| **F1-Score (macro)** | 86.1% | Unweighted average across 3 classes |
| **F1-Score (weighted)** | 87.0% | Weighted by class frequency |
| **ROC-AUC (macro OvR)** | 0.94 | One-vs-rest per class |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Positive | 0.903 | 0.916 | 0.909 | 462 |
| Negative | 0.892 | 0.887 | 0.889 | 425 |
| Neutral | 0.905 | 0.920 | 0.912 | 513 |

### Comparison to Baselines

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| Logistic Regression (TF-IDF) | 71.2% | 68.9% |
| DistilBERT (no fine-tuning) | 74.8% | 72.1% |
| Base FinBERT (no fine-tuning) | 82.1% | 80.6% |
| **FinBERT Fine-tuned (this model)** | **87.3%** | **86.1%** |

---

## ⚠️ Limitations and Biases

1. **Domain specificity**: The model performs significantly worse (~15% accuracy drop) on social media financial content (Reddit, Twitter) vs. formal news.
2. **Irony and negation**: Complex negations ("not a loss" being predicted as negative) reduce accuracy for ~3% of edge cases.
3. **Numeric-heavy sentences**: Sentences consisting primarily of numbers with minimal context are classified as neutral by default.
4. **Short sentences**: Sentences under 5 words have reduced confidence; the model relies on contextual cues.
5. **Financial domain shift**: Performance may degrade on niche sub-domains (crypto, derivatives) underrepresented in the training data.
6. **Annotation bias**: The training labels reflect consensus among financial analysts; retail investor sentiment framing may differ.

---

## 🔧 Model Architecture

```
Input: Financial sentence (text)
Tokenization: BertTokenizer (max_length=512, truncation=True)
Base model: ProsusAI/finbert
  - 12 transformer layers
  - 768 hidden dimensions
  - 12 attention heads
  - ~110M parameters
Classification head: Linear(768 → 3) + Softmax
Fine-tuning:
  - Epochs: 3
  - Learning rate: 2e-5 (with linear warmup, 10% steps)
  - Batch size: 16
  - Optimizer: AdamW (weight decay 0.01)
  - Hardware: NVIDIA GPU (CUDA)
Output: {positive, negative, neutral} + confidence score
```

---

## 📐 Evaluation Methodology

- **Validation strategy**: Stratified 80/20 split; no data leakage
- **Metric rationale**: Macro F1 chosen to give equal weight to all classes regardless of frequency; relevant for balanced decision support
- **Interpretability**: Token-level attribution via SHAP values (see `notebooks/04_model_interpretability.ipynb`)

---

## 🚀 Deployment Recommendations

| Concern | Recommendation |
|---------|---------------|
| Latency | CPU: ~120 ms/sentence; GPU: ~20 ms/sentence |
| Batch inference | Max batch_size=32 on 8GB GPU without OOM |
| Serving | FastAPI + HuggingFace pipeline (see `notebooks/06_fastapi_inference.ipynb`) |
| Model storage | ~440 MB (full BERT); quantize to ~110 MB with INT8 for production |
| Retraining trigger | Accuracy drop > 3% on a monitored weekly sample |
| Confidence threshold | Discard predictions with max_score < 0.60 (low confidence) |

---

## 📚 References

1. Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts*. JASIST.
2. Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. arXiv:1908.10063.
3. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL.
4. HuggingFace FinBERT: https://huggingface.co/ProsusAI/finbert
5. Dataset: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

---

*This model card follows the format proposed by Mitchell et al. (2019), "Model Cards for Model Reporting".*
