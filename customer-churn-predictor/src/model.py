"""
Model definition for churn prediction.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def build_model() -> Pipeline:
    """
    Build a full Scikit-learn Pipeline:
    [Preprocessor] -> [RandomForestClassifier]

    Returns
    -------
    Pipeline
        Configured model pipeline.
    """
    preprocessor = build_preprocessor()

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )
    return model
