"""
Configuration module for the Customer Churn Predictor project.
Edit this file to match your dataset schema and paths.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Input dataset
RAW_DATA_FILE = RAW_DATA_DIR / "telecom_churn.csv"

# Target column name in the dataset
TARGET_COL = "Churn"  # e.g. "Churn" with values "Yes"/"No"

# List your feature columns (example for telecom-like dataset)
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Train/validation/test split ratios
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # portion of train set

RANDOM_STATE = 42

# Model output path
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
