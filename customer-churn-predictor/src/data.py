from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    RAW_DATA_FILE,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    PROCESSED_DATA_DIR,
)


def load_raw_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load raw churn dataset from a CSV file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    df = pd.read_csv(path)
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the target column to binary {0,1}.
    """
    df = df.copy()

    if df[TARGET_COL].dtype == "object":
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()
        mapping = {"yes": 1, "no": 0}
        df[TARGET_COL] = df[TARGET_COL].map(mapping)

    df = df[df[TARGET_COL].isin([0, 1])]
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean/convert feature columns so they are ready for the pipeline.

    - Convert TotalCharges to numeric (handles ' ' as NaN).
    """
    df = df.copy()

    # Telco dataset specific: TotalCharges has spaces -> convert to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def train_val_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets.
    """
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )

    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    df_train, df_val = train_test_split(
        df_train,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=df_train[TARGET_COL],
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    df_val.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    df_test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    return df_train, df_val, df_test
