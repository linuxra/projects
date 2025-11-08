"""
Evaluate the saved churn model on the test set.

Usage:
    python -m src.evaluate
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from .config import MODEL_PATH, PROCESSED_DATA_DIR, TARGET_COL


def main() -> None:
    """
    Load the trained model and evaluate on the test dataset.
    """
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found at {test_path}. Run train.py first."
        )

    print("📥 Loading test data...")
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )

    print("📦 Loading model...")
    model = joblib.load(MODEL_PATH)

    print("📊 Evaluating...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
