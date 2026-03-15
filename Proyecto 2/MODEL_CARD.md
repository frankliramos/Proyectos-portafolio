# Model Card: XGBoost Sales Forecasting Model

> **Based on**: [Google Model Cards](https://modelcards.withgoogle.com/about) framework

---

## 📋 Model Details

| Field | Value |
|-------|-------|
| **Model Name** | XGBoost Retail Sales Forecaster |
| **Version** | v1.0 |
| **Model Type** | Gradient Boosting (XGBoost) |
| **Task** | Time-series regression (15-day sales forecast) |
| **Framework** | XGBoost 2.x with GPU acceleration |
| **Author** | Franklin Ramos |
| **Date** | March 2026 |
| **License** | MIT |

---

## 🎯 Intended Use

### Primary Use Case
Forecast daily retail sales (unit volume) for the next 15 days, per store and product family, to support inventory planning and procurement decisions.

### Intended Users
- Inventory managers at multi-store retail chains
- Procurement and supply-chain teams
- Business intelligence analysts

### Out-of-Scope Uses
- Forecasting beyond a 15-day horizon (accuracy degrades significantly)
- Predicting individual customer purchase behavior
- Price optimization (the model does not predict revenue, only unit volume)
- Markets outside of Ecuador without recalibration

---

## 📊 Training Data

| Property | Value |
|----------|-------|
| **Dataset** | Corporación Favorita (Kaggle Store Sales competition) |
| **Time period** | January 2013 – August 2017 |
| **Records** | 2,947,428 training rows |
| **Stores** | 54 |
| **Product families** | 33 |
| **Target variable** | `sales` (daily unit sales) |
| **Train / validation split** | Temporal: last 15 days of training data as validation |

### Feature Engineering (27 features)

| Feature group | Features |
|--------------|---------|
| Lag (sales) | `sales_lag_16`, `sales_lag_21`, `sales_lag_30` |
| Rolling mean (sales) | `sales_roll_mean_7`, `sales_roll_mean_14`, `sales_roll_mean_30` |
| Lag (transactions) | `trans_lag_16`, `trans_lag_21` |
| Rolling mean (transactions) | `trans_roll_mean_7`, `trans_roll_mean_14`, `trans_roll_mean_28` |
| Temporal | `month`, `day_of_week`, `year`, `is_weekend` |
| External | `dcoilwtico` (oil price), `is_holiday` |
| Store metadata | `store_nbr`, `city`, `state`, `type`, `cluster` |
| Product | `family`, `onpromotion` |

---

## 📈 Performance

### Global Metrics (Validation Set — last 15 days)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSLE** | 0.40 | Primary competition metric; lower is better |
| **WAPE** | 16.9% | Weighted Absolute Percentage Error; ~83% accuracy |
| **RMSE (log-scale)** | 0.5925 | Training-set metric on log-transformed target |

### Segmented Performance

| Segment | WAPE | Notes |
|---------|------|-------|
| Grocery & Cleaning | ~11% | Most stable demand patterns |
| Beverages & Produce | ~15% | Moderate weekly seasonality |
| Automotive & Books | ~28% | High volatility; low-volume families |
| Holiday periods | ~22% | Promotional spikes harder to capture |

---

## ⚠️ Limitations and Biases

1. **Promotional underestimation**: The model uses `onpromotion` as a feature but tends to underestimate sales during promotional events (conservative bias ~+10% actual vs. predicted).
2. **Economic shocks**: Oil-price shocks not captured in recent lag features may cause significant forecast errors for correlated product families.
3. **New stores**: The model was trained on existing stores; predictions for new store locations require recalibration.
4. **Zero-sales days**: Families with intermittent demand (e.g., AUTOMOTIVE) can produce negative predictions (clipped to 0) and elevated WAPE.
5. **Temporal drift**: Model accuracy degrades if retrained data is stale by more than 3 months without retraining.

---

## 🔧 Model Architecture

```
Input: 27 engineered features (tabular)
Model: XGBoost (tree-based gradient boosting)
  - Objective: reg:squarederror (on log1p-transformed target)
  - Boosting rounds: 1,277 (with early stopping on validation RMSE)
  - Learning rate: default (0.3, tuned implicitly by early stopping)
  - GPU device: cuda (tree_method='hist')
Output: Predicted log1p(sales) → expm1 → clipped to [0, ∞)
```

---

## 📐 Evaluation Methodology

- **Validation strategy**: Walk-forward (temporal split); no data leakage
- **Baseline comparison**: Naive lag-7 predictor (WAPE ~31%) — model is ~45% better
- **Metric rationale**: RMSLE penalizes large relative errors equally for small and large values, suitable for sales data spanning several orders of magnitude

---

## 🚀 Deployment Recommendations

| Concern | Recommendation |
|---------|---------------|
| Latency | Batch inference (nightly); single inference < 50 ms |
| Retraining | Monthly or when WAPE drift > 5% on rolling window |
| Monitoring | Track WAPE on a 7-day rolling window per store/family |
| Serving | FastAPI REST endpoint or scheduled batch pipeline |
| Scaling | Single GPU instance sufficient for all 54×33 combinations |

---

## 📚 References

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Kaggle Store Sales competition: https://www.kaggle.com/c/store-sales-time-series-forecasting
- Corporación Favorita dataset: Kaggle public competition data

---

*This model card follows the format proposed by Mitchell et al. (2019), "Model Cards for Model Reporting".*
