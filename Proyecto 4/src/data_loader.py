import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH


def load_raw_data():
    """
    Load the churn dataset from the raw folder.
    Handles conversion of critical data types.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"File not found at: {RAW_DATA_PATH}")

    df = pd.read_excel(RAW_DATA_PATH)
    df.columns = [col.strip() for col in df.columns]

    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df["Total Charges"] = df["Total Charges"].fillna(0)

    return df


def get_data_summary(df):
    """
    Print an executive summary of the loaded data.
    """
    print("✅ Data loaded successfully.")
    print(f"📊 Total records: {df.shape[0]}")
    print(f"🧬 Total variables: {df.shape[1]}")
    print(f"🎯 Current churn rate: {(df['Churn Value'].mean() * 100):.2f}%")
    return df.info()


if __name__ == "__main__":
    data = load_raw_data()
    get_data_summary(data)
