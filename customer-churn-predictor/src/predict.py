"""
Predict churn for a single customer example using the saved model.

Usage (example):

python -m src.predict \
    --tenure 12 \
    --MonthlyCharges 70 \
    --TotalCharges 840 \
    --gender Female \
    --SeniorCitizen 0 \
    --Partner Yes \
    --Dependents No \
    --PhoneService Yes \
    --MultipleLines No \
    --InternetService "Fiber optic" \
    --OnlineSecurity No \
    --OnlineBackup No \
    --DeviceProtection No \
    --TechSupport No \
    --StreamingTV Yes \
    --StreamingMovies Yes \
    --Contract "Month-to-month" \
    --PaperlessBilling Yes \
    --PaymentMethod "Electronic check"
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import joblib
import pandas as pd

from .config import MODEL_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def parse_args() -> Dict[str, Any]:
    """
    Parse CLI arguments into a dictionary of feature values.
    """
    parser = argparse.ArgumentParser(description="Churn prediction for one customer.")

    # Dynamically add arguments based on feature lists
    for col in NUMERIC_FEATURES:
        parser.add_argument(f"--{col}", type=float, required=True)

    for col in CATEGORICAL_FEATURES:
        parser.add_argument(f"--{col}", type=str, required=True)

    args = parser.parse_args()
    return vars(args)


def main() -> None:
    """
    Load model, read CLI inputs, and print churn probability & label.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )

    features = parse_args()
    df = pd.DataFrame([features])

    model = joblib.load(MODEL_PATH)

    proba = model.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)

    print(f"Churn probability: {proba:.4f}")
    print(f"Predicted label: {pred} (1 = churn, 0 = no churn)")


if __name__ == "__main__":
    main()
