# Customer Churn Predictor 📉🔍

End-to-end **Customer Churn Prediction** project built with **Python & Scikit-learn**.

This project is structured the way real client / production work is done:
- Clean module-based code
- Reproducible pipeline
- Train / validation / test split
- Saved model artifact
- CLI-based prediction for new customers

Perfect as:
- A **portfolio project** for AI/ML & Python freelancing
- A **template** for real-world churn use cases (Telco, SaaS, Banking, Subscription)

---

## 🔧 Tech Stack

- Python 3.10+
- Pandas, NumPy
- Scikit-learn (Pipelines + ColumnTransformer)
- RandomForestClassifier
- Joblib

---

## 📂 Project Structure

```text
customer-churn-predictor/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/
│  │  └─ telecom_churn.csv        # Telco churn dataset (Kaggle)
│  └─ processed/
│     ├─ train.csv
│     ├─ val.csv
│     └─ test.csv
├─ models/
│  └─ churn_model.joblib          # saved model pipeline
├─ src/
│  ├─ __init__.py
│  ├─ config.py                   # paths, feature lists, target column
│  ├─ data.py                     # load + clean + split
│  ├─ features.py                 # preprocessing pipelines
│  ├─ model.py                    # model pipeline definition
│  ├─ train.py                    # training + val metrics
│  ├─ evaluate.py                 # test set evaluation
│  └─ predict.py                  # single-customer prediction via CLI
└─ notebooks/
   └─ 01_eda_and_baseline.ipynb   # (optional) EDA

