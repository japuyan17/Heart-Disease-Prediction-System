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