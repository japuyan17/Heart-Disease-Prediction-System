# Core data handling
import numpy as np
import pandas as pd

# SQL (SQLite for local work; pandas.read_sql works with any SQLAlchemy engine)
import sqlite3
from sqlalchemy import create_engine
import pandasql as psql  # optional: run SQL directly on DataFrames

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and model selection
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
    precision_recall_curve
)

# Interpretability (optional but useful for a portfolio project)
import shap

import warnings
warnings.filterwarnings("ignore")

# Load CSV into pandas
df = pd.read_csv("heart_disease_data.csv")

print(df.shape)
print(df.head())
df.info()
print(df.isnull().sum())

# Load into SQLite and query with SQL
conn = sqlite3.connect("heart.db")
df.to_sql("heart", conn, if_exists="replace", index=False)

query = """
SELECT target, COUNT(*) AS n, ROUND(AVG(age), 1) AS avg_age, ROUND(AVG(chol), 1) AS avg_chol
FROM heart
GROUP BY target
"""
print(pd.read_sql(query, conn))

# Missing values and invalid category codes
print(df.isnull().sum())
for col in ["cp", "restecg", "slope", "thal", "ca"]:
    print(col, sorted(df[col].unique()))
df["thal"] = df["thal"].replace(0, np.nan)
df["ca"] = df["ca"].replace(4, np.nan)
print(df[["thal", "ca"]].isnull().sum())

# Remove duplicate rows (606 rows suggests the standard 303 are doubled)
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns="target")
y = df["target"]

numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical = ["cp", "restecg", "slope", "thal"]
passthrough = ["sex", "fbs", "exang", "ca"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical),
    ("pass", SimpleImputer(strategy="most_frequent"), passthrough),
])
preprocessor.set_output(transform="pandas")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

print(X_train_p.shape, X_test_p.shape)
print(X_train_p.head())

df = pd.read_csv("heart_disease_data.csv")
df["thal"] = df["thal"].replace(0, np.nan)
df["ca"] = df["ca"].replace(4, np.nan)
df = df.drop_duplicates().reset_index(drop=True)

conn = sqlite3.connect("heart.db")
df.to_sql("heart", conn, if_exists="replace", index=False)