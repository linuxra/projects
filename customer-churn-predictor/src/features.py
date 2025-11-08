"""
Feature engineering and preprocessing pipeline for churn prediction.
"""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor(
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """
    Create a ColumnTransformer that:
    - Imputes and scales numeric features.
    - Imputes and one-hot encodes categorical features.

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer to be used inside a Pipeline.
    """
    numeric_transformer = make_numeric_pipeline()
    categorical_transformer = make_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def make_numeric_pipeline():
    """Pipeline for numeric features."""
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def make_categorical_pipeline():
    """Pipeline for categorical features."""
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
