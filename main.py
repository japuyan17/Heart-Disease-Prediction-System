# =============================================================================
# Heart Disease Prediction System
# Pandas + SQL + Logistic Regression (baseline) + Random Forest + XGBoost
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# =============================================================================
# 2. CONFIG
# =============================================================================
DATA_PATH = "heart_disease_data.csv"
DB_PATH = "heart.db"
FIG_DIR = Path("figures")
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_JOBS = 4             # parallel workers for the grid searches (lower if it gets killed)

# Raw target value that means "has heart disease" (confirmed by the heatmap)
DISEASE_LABEL = 0

# Minimum share of sick patients we want to catch when tuning the decision threshold
TARGET_RECALL = 0.90

# Feature groups
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]  # scaled
CATEGORICAL = ["cp", "restecg", "slope", "thal"]             # one-hot encoded
PASSTHROUGH = ["sex", "fbs", "exang", "ca"]                  # binary / ordinal

FIG_DIR.mkdir(exist_ok=True)


def save_fig(name: str) -> None:
    """Save the current figure to figures/<name>.png and close it."""
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# 3. LOAD AND CLEAN DATA
# =============================================================================
df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

# thal=0 and ca=4 are invalid codes that stand in for missing values
df["thal"] = df["thal"].replace(0, np.nan)
df["ca"] = df["ca"].replace(4, np.nan)

# The file contains every row twice; duplicates would leak across the split
df = df.drop_duplicates().reset_index(drop=True)
print(f"Clean shape: {df.shape}")

missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0])


# =============================================================================
# 4. SQL EXPLORATION (SQLite)
# =============================================================================
conn = sqlite3.connect(DB_PATH)
df.to_sql("heart", conn, if_exists="replace", index=False)


def run_sql(query: str) -> pd.DataFrame:
    """Run a SQL query against the heart table and return a DataFrame."""
    return pd.read_sql(query, conn)


# Note: raw target = 0 means disease in this dataset
print("\n--- Class balance ---")
print(run_sql("""
    SELECT CASE WHEN target = 0 THEN 'Disease' ELSE 'Healthy' END AS class,
           COUNT(*) AS n,
           ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM heart), 1) AS pct
    FROM heart
    GROUP BY target
"""))

print("\n--- Average vitals by class ---")
print(run_sql("""
    SELECT CASE WHEN target = 0 THEN 'Disease' ELSE 'Healthy' END AS class,
           ROUND(AVG(age), 1)      AS avg_age,
           ROUND(AVG(trestbps), 1) AS avg_resting_bp,
           ROUND(AVG(chol), 1)     AS avg_chol,
           ROUND(AVG(thalach), 1)  AS avg_max_hr,
           ROUND(AVG(oldpeak), 2)  AS avg_oldpeak
    FROM heart
    GROUP BY target
"""))

print("\n--- Disease rate by chest pain type ---")
print(run_sql("""
    SELECT cp, COUNT(*) AS n, ROUND(AVG(target = 0), 2) AS disease_rate
    FROM heart
    GROUP BY cp
    ORDER BY cp
"""))

print("\n--- Disease rate by age group ---")
print(run_sql("""
    SELECT CASE WHEN age < 40 THEN '1: <40'
                WHEN age < 50 THEN '2: 40-49'
                WHEN age < 60 THEN '3: 50-59'
                ELSE '4: 60+' END AS age_group,
           COUNT(*) AS n,
           ROUND(AVG(target = 0), 2) AS disease_rate
    FROM heart
    GROUP BY age_group
    ORDER BY age_group
"""))

conn.close()


# =============================================================================
# 5. TARGET RECODING AND CORRELATION HEATMAP
# =============================================================================
# Recode so that 1 = disease, 0 = healthy
y = (df["target"] == DISEASE_LABEL).astype(int)
X = df.drop(columns="target")
print(f"\nDisease prevalence: {y.mean():.1%}")

plt.figure(figsize=(10, 8))
sns.heatmap(X.assign(disease=y).corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlations")
plt.tight_layout()
save_fig("correlation_heatmap")


# =============================================================================
# 6. TRAIN / TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")


# =============================================================================
# 7. PREPROCESSING PIPELINE
# =============================================================================
# Fit inside each CV fold (via the pipeline) so no test information leaks in
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUMERIC),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAL),
    ("pass", SimpleImputer(strategy="most_frequent"), PASSTHROUGH),
])
preprocessor.set_output(transform="pandas")


# =============================================================================
# 8. MODELS AND HYPERPARAMETER GRIDS
# =============================================================================
# Models run single-threaded; parallelism happens once, in the grid search
pipes = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1)),
    ]),
}

# "model__" targets the model step inside each pipeline
grids = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10],
    },
    "Random Forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5, None],
        "model__min_samples_leaf": [1, 3],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.03, 0.1],
    },
}


# =============================================================================
# 9. CROSS-VALIDATION BASELINE (default hyperparameters)
# =============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["recall", "precision", "f1", "roc_auc", "accuracy"]

cv_rows = {}
for name, pipe in pipes.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=N_JOBS)
    cv_rows[name] = {m: scores[f"test_{m}"].mean() for m in scoring}

print("\n=== 5-fold CV baseline (defaults, training data) ===")
print(pd.DataFrame(cv_rows).T.round(3))


# =============================================================================
# 10. HYPERPARAMETER TUNING
# =============================================================================
# Tune on ROC-AUC (threshold-independent); the decision threshold is tuned
# separately in Section 11 to prioritize recall.
best_models = {}
for name, pipe in pipes.items():
    print(f"\nTuning {name}...")
    search = GridSearchCV(pipe, grids[name], cv=cv, scoring="roc_auc", n_jobs=N_JOBS)
    search.fit(X_train, y_train)
    best_models[name] = search.best_estimator_
    print(f"Best params: {search.best_params_}  (CV AUC {search.best_score_:.3f})")


# =============================================================================
# 11. RECALL-FOCUSED THRESHOLD TUNING
# =============================================================================
# Missing a sick patient is worse than a false alarm, so pick the highest
# threshold that still catches at least TARGET_RECALL of sick patients.
# It is chosen on out-of-fold TRAINING predictions, never on the test set.
thresholds = {}
for name, model in best_models.items():
    oof_prob = cross_val_predict(model, X_train, y_train, cv=cv,
                                 method="predict_proba", n_jobs=N_JOBS)[:, 1]
    _, recall_curve, thr = precision_recall_curve(y_train, oof_prob)
    ok = np.where(recall_curve[:-1] >= TARGET_RECALL)[0]
    thresholds[name] = float(thr[ok[-1]]) if len(ok) else 0.5

print(f"\n=== Recall-tuned thresholds (target recall {TARGET_RECALL:.0%}) ===")
for name, t in thresholds.items():
    print(f"{name:20s} {t:.3f}")


# =============================================================================
# 12. TEST SET EVALUATION
# =============================================================================
test_probs = {name: m.predict_proba(X_test)[:, 1] for name, m in best_models.items()}


def evaluate(prob: np.ndarray, threshold: float) -> tuple[np.ndarray, dict]:
    """Return predictions and a metrics dict for one model at one threshold."""
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return pred, {
        "Threshold": round(threshold, 3),
        "Recall": recall_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Accuracy": accuracy_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "Missed sick (FN)": fn,
        "False alarms (FP)": fp,
    }


rows, preds = [], {}
for name in best_models:
    for label, t in [("0.50", 0.5), ("recall-tuned", thresholds[name])]:
        pred, metrics = evaluate(test_probs[name], t)
        preds[(name, label)] = pred
        rows.append({"Model": name, "Threshold type": label, **metrics})

results = (pd.DataFrame(rows)
           .set_index(["Model", "Threshold type"])
           .sort_values(["Recall", "Precision"], ascending=False)
           .round(3))

print("\n=== Test set comparison (sorted by recall, then precision) ===")
print(results.to_string())


# =============================================================================
# 13. EVALUATION PLOTS
# =============================================================================
# Confusion matrices: top row = default 0.50 threshold, bottom = recall-tuned
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for row, label in enumerate(["0.50", "recall-tuned"]):
    for col, name in enumerate(best_models):
        ConfusionMatrixDisplay.from_predictions(
            y_test, preds[(name, label)], display_labels=["Healthy", "Disease"],
            cmap="Blues", colorbar=False, ax=axes[row, col])
        axes[row, col].set_title(f"{name}\n({label} threshold)")
plt.tight_layout()
save_fig("confusion_matrices")

# ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, model in best_models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set_title("ROC Curves")
ax.legend()
plt.tight_layout()
save_fig("roc_curves")

# Recall vs precision at the recall-tuned threshold
tuned = results.xs("recall-tuned", level="Threshold type")[["Recall", "Precision"]]
tuned.plot.bar(figsize=(7, 4), rot=15, ylim=(0, 1.05))
plt.title("Recall vs Precision (recall-tuned threshold)")
plt.tight_layout()
save_fig("recall_vs_precision")


# =============================================================================
# 14. FEATURE IMPORTANCE
# =============================================================================
feature_names = best_models["Random Forest"].named_steps["prep"].get_feature_names_out()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, best_models.items()):
    est = model.named_steps["model"]
    values = est.coef_[0] if name == "Logistic Regression" else est.feature_importances_
    label = "Coefficient" if name == "Logistic Regression" else "Importance"
    s = pd.Series(values, index=feature_names)
    s.reindex(s.abs().nlargest(10).index).sort_values().plot.barh(ax=ax)
    ax.set_title(f"{name}: Top 10 ({label})")
plt.tight_layout()
save_fig("feature_importance")


# =============================================================================
# 15. SHAP EXPLAINABILITY (XGBoost)
# =============================================================================
xgb_best = best_models["XGBoost"]
X_test_processed = xgb_best.named_steps["prep"].transform(X_test)
shap_values = shap.TreeExplainer(xgb_best.named_steps["model"])(X_test_processed)
shap.summary_plot(shap_values, X_test_processed, show=False)
save_fig("shap_summary")

print(f"\nDone. Figures saved to {FIG_DIR.resolve()}")

# =============================================================================
# 16. EXPORT FOR TABLEAU
# =============================================================================
from sklearn.metrics import roc_curve

TAB_DIR = Path("tableau")
TAB_DIR.mkdir(exist_ok=True)

# 1. Patient-level data with readable labels
patients = df.drop(columns="target").copy()
patients["disease"] = np.where(y == 1, "Disease", "Healthy")
patients["sex"] = patients["sex"].map({0: "Female", 1: "Male"})
patients["age_group"] = pd.cut(df["age"], [0, 39, 49, 59, 120],
                               labels=["<40", "40-49", "50-59", "60+"])
patients.to_csv(TAB_DIR / "patients.csv", index=False)

# 2. Model comparison table (all models x both thresholds)
results.reset_index().to_csv(TAB_DIR / "model_results.csv", index=False)

# 3. Confusion matrix cells in long format
cm_rows = []
for (name, label), pred in preds.items():
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    for actual, predicted, count, outcome in [
        ("Healthy", "Healthy", tn, "Correct (TN)"),
        ("Healthy", "Disease", fp, "False alarm (FP)"),
        ("Disease", "Healthy", fn, "Missed sick (FN)"),
        ("Disease", "Disease", tp, "Caught sick (TP)"),
    ]:
        cm_rows.append({"Model": name, "Threshold type": label, "Actual": actual,
                        "Predicted": predicted, "Count": int(count), "Outcome": outcome})
pd.DataFrame(cm_rows).to_csv(TAB_DIR / "confusion_matrix.csv", index=False)

# 4. ROC curve points
roc_rows = []
for name, prob in test_probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_rows += [{"Model": name, "FPR": a, "TPR": b} for a, b in zip(fpr, tpr)]
pd.DataFrame(roc_rows).to_csv(TAB_DIR / "roc_curves.csv", index=False)

# 5. Threshold sweep: how recall, precision, and errors change with the cutoff
sweep_rows = []
for name, prob in test_probs.items():
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        sweep_rows.append({
            "Model": name, "Threshold": round(t, 2),
            "Recall": recall_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Missed sick (FN)": int(fn), "False alarms (FP)": int(fp),
        })
pd.DataFrame(sweep_rows).to_csv(TAB_DIR / "threshold_sweep.csv", index=False)

# 6. Feature importance in long format
imp_rows = []
for name, model in best_models.items():
    est = model.named_steps["model"]
    vals = est.coef_[0] if name == "Logistic Regression" else est.feature_importances_
    imp_rows += [{"Model": name, "Feature": f, "Value": v, "Abs value": abs(v)}
                 for f, v in zip(feature_names, vals)]
pd.DataFrame(imp_rows).to_csv(TAB_DIR / "feature_importance.csv", index=False)

# 7. Test-set predictions per patient
test_out = X_test.copy()
test_out["actual"] = np.where(y_test == 1, "Disease", "Healthy")
for name, prob in test_probs.items():
    test_out[f"{name} risk"] = prob
test_out.to_csv(TAB_DIR / "test_predictions.csv", index=False)

print(f"Tableau files saved to {TAB_DIR.resolve()}")