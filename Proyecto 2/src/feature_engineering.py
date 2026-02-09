import pandas as pd
import numpy as np
from pathlib import Path


def clean_oil_data(oil_df):
    """Limpia e imputa los datos del precio del petróleo."""
    oil_df["date"] = pd.to_datetime(oil_df["date"])
    all_days = pd.date_range(
        start=oil_df["date"].min(), end=oil_df["date"].max(), freq="D"
    )
    oil_df = oil_df.set_index("date").reindex(all_days)
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill().bfill()
    oil_df.index.name = "date"
    return oil_df.reset_index()


def create_date_features(df):
    """Crea características basadas en la fecha."""
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def create_holiday_features(holidays_df):
    """Crea un dataframe con una columna binaria de feriados por fecha."""
    holidays_df = holidays_df.copy()
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])

    # Filtramos solo feriados relevantes
    mask = (holidays_df["transferred"] == False) & (
        ~holidays_df["type"].isin(["Work Day", "Bridge"])
    )
    holidays_df = holidays_df[mask]

    # Marcamos 1 si es feriado
    holidays_daily = holidays_df.groupby("date", as_index=False).agg(
        is_holiday=("type", lambda x: 1)
    )

    return holidays_daily


def create_transaction_features(transactions_df):
    """Crea features de transacciones con lags y rolling windows."""
    transactions_df = transactions_df.copy()
    transactions_df["date"] = pd.to_datetime(transactions_df["date"])
    transactions_df = transactions_df.sort_values(["store_nbr", "date"])

    # Lags de transacciones (mínimo 16 para poder predecir 15 días adelante)
    for lag in [16, 21]:
        transactions_df[f"trans_lag_{lag}"] = transactions_df.groupby("store_nbr")[
            "transactions"
        ].shift(lag)

    # Rolling means de transacciones
    for window in [7, 14, 28]:
        transactions_df[f"trans_roll_mean_{window}"] = transactions_df.groupby(
            "store_nbr"
        )["transactions"].transform(lambda x: x.shift(16).rolling(window=window).mean())

    # Seleccionamos solo las columnas necesarias
    trans_features = transactions_df[
        [
            "date",
            "store_nbr",
            "trans_lag_16",
            "trans_lag_21",
            "trans_roll_mean_7",
            "trans_roll_mean_14",
            "trans_roll_mean_28",
        ]
    ]

    return trans_features


def generate_features():
    # Configuración de rutas
    base_path = Path(__file__).parent.parent
    raw_path = base_path / "data" / "raw"
    processed_path = base_path / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    train = pd.read_csv(raw_path / "train.csv", parse_dates=["date"])
    oil = pd.read_csv(raw_path / "oil.csv")
    stores = pd.read_csv(raw_path / "stores.csv")
    holidays = pd.read_csv(raw_path / "holidays_events.csv")
    transactions = pd.read_csv(raw_path / "transactions.csv", parse_dates=["date"])

    # 1. Limpieza de Petróleo
    print("Procesando datos de petróleo...")
    oil = clean_oil_data(oil)

    # 2. Feriados
    print("Procesando datos de feriados...")
    holidays_daily = create_holiday_features(holidays)

    # 3. Transacciones (NUEVO)
    print("Procesando datos de transacciones...")
    trans_features = create_transaction_features(transactions)

    # 4. Merge de datos básicos
    print("Uniendo datasets...")
    df = train.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil, on="date", how="left")
    df = df.merge(holidays_daily, on="date", how="left")
    df = df.merge(trans_features, on=["date", "store_nbr"], how="left")

    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    # 5. Características de Fecha
    df = create_date_features(df)

    # 6. Ingeniería de Lags de Ventas
    print("Creando variables de rezago (lags) y medias móviles de ventas...")
    df = df.sort_values(["store_nbr", "family", "date"])

    for lag in [16, 21, 30]:
        df[f"sales_lag_{lag}"] = df.groupby(["store_nbr", "family"])["sales"].shift(lag)

    for window in [7, 14, 30]:
        df[f"sales_roll_mean_{window}"] = df.groupby(["store_nbr", "family"])[
            "sales"
        ].transform(lambda x: x.shift(16).rolling(window=window).mean())

    # 7. Imputación de valores faltantes en transacciones
    print("Imputando valores faltantes en transacciones...")
    for col in [
        "trans_lag_16",
        "trans_lag_21",
        "trans_roll_mean_7",
        "trans_roll_mean_14",
        "trans_roll_mean_28",
    ]:
        df[col] = df.groupby("store_nbr")[col].transform(lambda x: x.fillna(x.mean()))

    print(f"Guardando datos procesados en {processed_path}...")
    df.dropna(subset=["sales_lag_30"], inplace=True)
    df.to_parquet(processed_path / "train_features.parquet", index=False)
    print("¡Proceso completado con éxito!")
    print(f"Forma final del dataset: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")


if __name__ == "__main__":
    generate_features()
