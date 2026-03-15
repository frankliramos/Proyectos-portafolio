import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    """
    Clean and transform the dataframe for model training.
    Returns a numeric DataFrame ready for XGBoost/LightGBM.
    """
    # 1. Drop columns that should not be used to predict churn
    cols_to_drop = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Label",
        "Churn Reason",  # post-mortem info or duplicate of target
    ]
    df_ml = df.drop(columns=cols_to_drop)

    # 2. Identify categorical columns
    cat_cols = df_ml.select_dtypes(include=["object"]).columns

    # 3. Simple label encoding (acceptable for portfolio; can switch to one-hot later)
    le = LabelEncoder()
    for col in cat_cols:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

    return df_ml


def get_train_test_splits(df_ml, target="Churn Value", test_size=0.2, random_state=42):
    """
    Split data into train and test with stratification.
    """
    X = df_ml.drop(columns=[target])
    y = df_ml[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
