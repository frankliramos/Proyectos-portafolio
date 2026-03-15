import pandas as pd
import numpy as np
from pathlib import Path


def clean_oil_data(oil_df):
    """Clean and impute oil price data."""
    oil_df["date"] = pd.to_datetime(oil_df["date"])
    all_days = pd.date_range(
        start=oil_df["date"].min(), end=oil_df["date"].max(), freq="D"
    )
    oil_df = oil_df.set_index("date").reindex(all_days)
    oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill().bfill()
    oil_df.index.name = "date"
    return oil_df.reset_index()


def create_date_features(df):
    """Create date-based features."""
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def create_holiday_features(holidays_df):
    """Create a dataframe with a binary holiday flag per date."""
    holidays_df = holidays_df.copy()
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])

    # Filter only relevant holidays
    mask = (holidays_df["transferred"] == False) & (
        ~holidays_df["type"].isin(["Work Day", "Bridge"])
    )
    holidays_df = holidays_df[mask]

    # Mark as 1 if it is a holiday
    holidays_daily = holidays_df.groupby("date", as_index=False).agg(
        is_holiday=("type", lambda x: 1)
    )

    return holidays_daily


def create_transaction_features(transactions_df):
    """Create transaction features with lags and rolling windows."""
    transactions_df = transactions_df.copy()
    transactions_df["date"] = pd.to_datetime(transactions_df["date"])
    transactions_df = transactions_df.sort_values(["store_nbr", "date"])

    # Transaction lags (minimum 16 to predict 15 days ahead)
    for lag in [16, 21]:
        transactions_df[f"trans_lag_{lag}"] = transactions_df.groupby("store_nbr")[
            "transactions"
        ].shift(lag)

    # Rolling means for transactions
    for window in [7, 14, 28]:
        transactions_df[f"trans_roll_mean_{window}"] = transactions_df.groupby(
            "store_nbr"
        )["transactions"].transform(lambda x: x.shift(16).rolling(window=window).mean())

    # Select only required columns
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
    # Path configuration
    base_path = Path(__file__).parent.parent
    raw_path = base_path / "data" / "raw"
    processed_path = base_path / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train = pd.read_csv(raw_path / "train.csv", parse_dates=["date"])
    oil = pd.read_csv(raw_path / "oil.csv")
    stores = pd.read_csv(raw_path / "stores.csv")
    holidays = pd.read_csv(raw_path / "holidays_events.csv")
    transactions = pd.read_csv(raw_path / "transactions.csv", parse_dates=["date"])

    # 1. Oil data cleaning
    print("Processing oil data...")
    oil = clean_oil_data(oil)

    # 2. Holidays
    print("Processing holiday data...")
    holidays_daily = create_holiday_features(holidays)

    # 3. Transactions (new)
    print("Processing transaction data...")
    trans_features = create_transaction_features(transactions)

    # 4. Merge base datasets
    print("Merging datasets...")
    df = train.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil, on="date", how="left")
    df = df.merge(holidays_daily, on="date", how="left")
    df = df.merge(trans_features, on=["date", "store_nbr"], how="left")

    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    # 5. Date features
    df = create_date_features(df)

    # 6. Sales lag features
    print("Creating sales lag and rolling mean features...")
    df = df.sort_values(["store_nbr", "family", "date"])

    for lag in [16, 21, 30]:
        df[f"sales_lag_{lag}"] = df.groupby(["store_nbr", "family"])["sales"].shift(lag)

    for window in [7, 14, 30]:
        df[f"sales_roll_mean_{window}"] = df.groupby(["store_nbr", "family"])[
            "sales"
        ].transform(lambda x: x.shift(16).rolling(window=window).mean())

    # 7. Impute missing values in transactions
    print("Imputing missing transaction values...")
    for col in [
        "trans_lag_16",
        "trans_lag_21",
        "trans_roll_mean_7",
        "trans_roll_mean_14",
        "trans_roll_mean_28",
    ]:
        df[col] = df.groupby("store_nbr")[col].transform(lambda x: x.fillna(x.mean()))

    print(f"Saving processed data to {processed_path}...")
    df.dropna(subset=["sales_lag_30"], inplace=True)
    df.to_parquet(processed_path / "train_features.parquet", index=False)
    print("Process completed successfully!")
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    generate_features()
