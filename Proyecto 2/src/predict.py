import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path


def prepare_test_data(test_df, train_df, oil_df, stores_df, holidays_df):
    """Aplica la misma ingeniería de variables al set de test."""
    # 1. Merge con tiendas y petróleo
    df = test_df.merge(stores_df, on="store_nbr", how="left")
    df = df.merge(oil_df, on="date", how="left")

    # 2. Feriados
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])
    mask = (holidays_df["transferred"] == False) & (
        ~holidays_df["type"].isin(["Work Day", "Bridge"])
    )
    holidays_daily = (
        holidays_df[mask]
        .groupby("date")
        .agg(is_holiday=("type", lambda x: 1))
        .reset_index()
    )
    df = df.merge(holidays_daily, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    # 3. Características de fecha
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 4. Lags y Rolling (Usando los últimos datos de TRAIN)
    # Para el test, necesitamos los últimos días del train para calcular los lags
    full_df = pd.concat([train_df, df], axis=0).sort_values(
        ["store_nbr", "family", "date"]
    )

    for lag in [16, 21, 30]:
        full_df[f"sales_lag_{lag}"] = full_df.groupby(["store_nbr", "family"])[
            "sales"
        ].shift(lag)

    for window in [7, 14, 30]:
        full_df[f"sales_roll_mean_{window}"] = full_df.groupby(["store_nbr", "family"])[
            "sales"
        ].transform(lambda x: x.shift(16).rolling(window=window).mean())

    # Retornamos solo las filas que pertenecen al test
    return full_df[full_df["id"].isin(test_df["id"])].copy()


# --- EJECUCIÓN ---
base_path = Path(__file__).parent.parent
raw_path = base_path / "data" / "raw"

print("Cargando datos para predicción...")
test = pd.read_csv(raw_path / "test.csv")
train = pd.read_csv(raw_path / "train.csv")
oil = pd.read_csv(raw_path / "oil.csv")
stores = pd.read_csv(raw_path / "stores.csv")
holidays = pd.read_csv(raw_path / "holidays_events.csv")

# Limpieza rápida de petróleo (igual que en train)
oil["date"] = pd.to_datetime(oil["date"])
all_days = pd.date_range(start=oil["date"].min(), end=test["date"].max(), freq="D")
oil = (
    oil.set_index("date")
    .reindex(all_days)
    .ffill()
    .bfill()
    .reset_index()
    .rename(columns={"index": "date"})
)

print("Preparando variables de test...")
test_processed = prepare_test_data(test, train, oil, stores, holidays)

# Codificación de categorías (Debe ser igual al notebook)
for col in ["family", "city", "state", "type"]:
    test_processed[col] = test_processed[col].astype("category").cat.codes

features = [
    "store_nbr",
    "family",
    "onpromotion",
    "type",
    "cluster",
    "month",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "dcoilwtico",
    "sales_lag_16",
    "sales_lag_21",
    "sales_lag_30",
    "sales_roll_mean_7",
    "sales_roll_mean_14",
    "sales_roll_mean_30",
]

# --- AQUÍ NECESITAS EL MODELO ---
# Como el modelo está en la memoria del notebook, lo ideal es guardarlo allí primero.
