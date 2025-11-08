"""
Streamlit app for Customer Churn Predictor.

Run with:
    streamlit run streamlit_app.py

This app:
- Loads the trained churn model pipeline.
- Provides a UI to input customer details.
- Returns churn probability and prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES


@st.cache_resource
def load_model(model_path: Path):
    """
    Load and cache the trained churn model pipeline.

    Parameters
    ----------
    model_path : Path
        Path to the saved joblib model.

    Returns
    -------
    Any
        Loaded model pipeline.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please run `python -m src.train` first."
        )
    return joblib.load(model_path)


def build_input_form() -> Dict[str, Any]:
    """
    Render the input widgets and collect feature values for a single customer.

    Returns
    -------
    Dict[str, Any]
        Dictionary of feature_name -> value suitable for model input.
    """
    st.sidebar.header("Customer Profile")

    # Numeric features
    tenure = st.sidebar.number_input(
        "Tenure (months)", min_value=0, max_value=1000, value=12, step=1
    )
    monthly_charges = st.sidebar.number_input(
        "Monthly Charges", min_value=0.0, max_value=10000.0, value=70.0, step=1.0
    )
    total_charges = st.sidebar.number_input(
        "Total Charges", min_value=0.0, max_value=1000000.0, value=840.0, step=10.0
    )

    # Categorical features - aligned with Telco churn dataset values
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["0", "1"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"],
    )
    internet_service = st.sidebar.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
    )
    online_security = st.sidebar.selectbox(
        "Online Security",
        ["No internet service", "No", "Yes"],
    )
    online_backup = st.sidebar.selectbox(
        "Online Backup",
        ["No internet service", "No", "Yes"],
    )
    device_protection = st.sidebar.selectbox(
        "Device Protection",
        ["No internet service", "No", "Yes"],
    )
    tech_support = st.sidebar.selectbox(
        "Tech Support",
        ["No internet service", "No", "Yes"],
    )
    streaming_tv = st.sidebar.selectbox(
        "Streaming TV",
        ["No internet service", "No", "Yes"],
    )
    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies",
        ["No internet service", "No", "Yes"],
    )
    contract = st.sidebar.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    features = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
    }

    # Keep only keys that are actually in our configured features
    filtered: Dict[str, Any] = {}
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col in features:
            filtered[col] = features[col]

    return filtered


def main() -> None:
    """Main entrypoint for the Streamlit churn prediction app."""
    st.set_page_config(
        page_title="Customer Churn Predictor",
        page_icon="📉",
        layout="wide",
    )

    st.title("📉 Customer Churn Predictor")
    st.markdown(
        """
        This interactive app uses a trained **Machine Learning model** to estimate the
        probability that a customer will churn.

        - Built with **Python, Pandas, Scikit-learn**
        - Uses a full **Pipeline** (preprocessing + model)
        - Backed by your trained model in `models/churn_model.joblib`
        """
    )

    # Load model
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run `python -m src.train` from the terminal and refresh this page.")
        return

    # Input form
    input_data = build_input_form()

    # Prediction area
    st.subheader("Prediction")
    if st.button("Predict Churn"):
        df = pd.DataFrame([input_data])
        proba = float(model.predict_proba(df)[0, 1])
        pred = int(proba >= 0.5)

        st.metric(
            label="Churn Probability",
            value=f"{proba * 100:.2f} %",
        )

        if pred == 1:
            st.error("⚠️ This customer is **likely to churn** (label = 1).")
        else:
            st.success("✅ This customer is **unlikely to churn** (label = 0).")

        st.caption("Threshold = 0.50 (configurable in production).")


if __name__ == "__main__":
    main()
