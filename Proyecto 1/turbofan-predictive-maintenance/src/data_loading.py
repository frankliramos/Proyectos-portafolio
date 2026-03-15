"""
Data Loading Utilities for NASA CMAPSS FD001

This module contains functions to load and prepare the NASA CMAPSS
dataset (FD001 subset) for predictive maintenance tasks (RUL prediction).

Author: Franklin Ramos
Date: 2026-02-03
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_DIR,
    FD001_TRAIN_FILE,
    FD001_TEST_FILE,
    FD001_RUL_FILE,
)


def _load_fd001_raw(path: Path) -> pd.DataFrame:
    """
    Load an FD001 file (train or test) from a txt file.

    Args:
        path (Path): Path to the FD001 .txt file.

    Returns:
        pd.DataFrame: DataFrame with named columns.
    """
    # Column definitions based on NASA CMAPSS specification
    col_names = [
        "unit_id",
        "time_cycles",
        "op_1",
        "op_2",
        "op_3",
    ]
    # 21 sensors
    sensor_cols = [f"s_{i}" for i in range(1, 22)]
    col_names.extend(sensor_cols)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=col_names,
        engine="python",
    )

    return df


def load_fd001_train() -> pd.DataFrame:
    """
    Load the FD001 training set from RAW_DATA_DIR.

    Returns:
        pd.DataFrame: Training data without the RUL column.
    """
    return _load_fd001_raw(FD001_TRAIN_FILE)


def load_fd001_test() -> pd.DataFrame:
    """
    Load the FD001 test set from RAW_DATA_DIR.

    Returns:
        pd.DataFrame: Test data without RUL (RUL comes in a separate file).
    """
    return _load_fd001_raw(FD001_TEST_FILE)


def load_fd001_rul() -> pd.DataFrame:
    """
    Load the RUL (Remaining Useful Life) file for FD001 (test).

    Returns:
        pd.DataFrame: DataFrame with a single 'RUL' column.
    """
    df_rul = pd.read_csv(
        FD001_RUL_FILE,
        sep=r"\s+",
        header=None,
        names=["RUL"],
        engine="python",
    )
    return df_rul


def add_rul_to_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and add the RUL column to the training DataFrame.

    For each engine (unit_id), RUL is defined as:
        RUL = max(time_cycles) - time_cycles

    Args:
        train_df (pd.DataFrame): FD001 training DataFrame.

    Returns:
        pd.DataFrame: DataFrame with an added 'RUL' column.
    """
    df = train_df.copy()

    # Max cycle per engine
    max_cycles = df.groupby("unit_id")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]

    # Merge
    df = df.merge(max_cycles, on="unit_id", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df = df.drop(columns=["max_cycle"])

    return df


def add_true_rul_to_test(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the true RUL column to the test set using RUL_FD001.

    NASA provides the remaining RUL for each engine at the last recorded
    cycle. This value is propagated backward for each row of that engine.

    Args:
        test_df (pd.DataFrame): FD001 test DataFrame (without RUL).

    Returns:
        pd.DataFrame: Test DataFrame with the 'RUL' column added.
    """
    df = test_df.copy()
    df_rul = load_fd001_rul().reset_index()
    df_rul.rename(columns={"index": "unit_id"}, inplace=True)
    df_rul["unit_id"] = df_rul["unit_id"] + 1  # unit_id starts at 1

    # Get the last cycle for each engine in test
    last_cycles = df.groupby("unit_id")["time_cycles"].max().reset_index()
    last_cycles.columns = ["unit_id", "last_cycle"]

    # Merge with provided RUL
    last_cycles = last_cycles.merge(df_rul, on="unit_id", how="left")

    # Map RUL at the last cycle
    df = df.merge(
        last_cycles[["unit_id", "last_cycle", "RUL"]], on="unit_id", how="left"
    )

    # Adjust RUL per cycle (counting backward from the last cycle)
    # RUL increases as we go backward in time
    df["RUL"] = df["RUL"] + (df["last_cycle"] - df["time_cycles"])
    df = df.drop(columns=["last_cycle"])

    return df


def load_fd001_prepared(save_processed=False):
    """
    Load and prepare both FD001 train and test datasets.

    Returns the training DataFrame with computed RUL and the test DataFrame
    with the true RUL values from the NASA-provided RUL file.

    Args:
        save_processed (bool): Reserved for future use. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df) with RUL columns.
    """
    train_df = add_rul_to_train(load_fd001_train())
    test_df = add_true_rul_to_test(load_fd001_test())
    return train_df, test_df
