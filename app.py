"""Streamlit app: enter patient information, get a heart disease risk estimate.

Run with:  streamlit run app.py
Requires models/heart_model.joblib, which main.py creates.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/heart_model.joblib")

st.set_page_config(page_title="Heart Disease Risk Estimator", page_icon="🫀", layout="centered")


@st.cache_resource
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


st.title("🫀 Heart Disease Risk Estimator")
st.caption("Educational machine learning project, not a medical device. Do not use it for health decisions.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python main.py` first to train and save it.")
    st.stop()

bundle = load_bundle()
model, threshold = bundle["model"], bundle["threshold"]

# ----------------------------------------------------------------------------
# Inputs (category codes are the dataset's own numeric codes)
# ----------------------------------------------------------------------------
st.subheader("Patient information")
left, right = st.columns(2)

with left:
    age = st.slider("Age", 20, 90, 55)
    sex = st.radio("Sex", [0, 1], format_func=lambda v: {0: "Female", 1: "Male"}[v], horizontal=True)
    cp = st.selectbox("Chest pain type (cp)", [0, 1, 2, 3], format_func=lambda v: f"cp = {v}")
    trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 245)
    fbs = st.radio("Fasting blood sugar > 120 mg/dl", [0, 1],
                   format_func=lambda v: {0: "No", 1: "Yes"}[v], horizontal=True)
    restecg = st.selectbox("Resting ECG result (restecg)", [0, 1, 2], format_func=lambda v: f"restecg = {v}")

with right:
    thalach = st.slider("Maximum heart rate achieved", 60, 220, 150)
    exang = st.radio("Exercise-induced angina", [0, 1],
                     format_func=lambda v: {0: "No", 1: "Yes"}[v], horizontal=True)
    oldpeak = st.slider("ST depression from exercise (oldpeak)", 0.0, 7.0, 1.0, 0.1)
    slope = st.selectbox("Slope of peak exercise ST segment (slope)", [0, 1, 2], index=1,
                         format_func=lambda v: f"slope = {v}")
    ca = st.selectbox("Major vessels colored by fluoroscopy (ca)", [0, 1, 2, 3, "Unknown"])
    thal = st.selectbox("Thalassemia test result (thal)", [1, 2, 3, "Unknown"], index=1)

patient = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
    "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
    "slope": slope,
    "ca": np.nan if ca == "Unknown" else ca,      # missing values are imputed by the pipeline
    "thal": np.nan if thal == "Unknown" else thal,
}])[bundle["features"]]

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if st.button("Estimate risk", type="primary"):
    risk = float(model.predict_proba(patient)[0, 1])

    st.subheader("Result")
    st.metric("Estimated disease probability", f"{risk:.0%}")
    st.progress(min(risk, 1.0))

    if risk >= threshold:
        st.error(f"**Flagged: higher risk.** The estimate is above the screening cutoff ({threshold:.0%}).")
    else:
        st.success(f"**Not flagged.** The estimate is below the screening cutoff ({threshold:.0%}).")

    st.caption(
        f"The cutoff is set low on purpose: it was tuned to catch about 90% of sick patients in training data, "
        f"because missing a sick patient is worse than a false alarm. Expect some false alarms."
    )

with st.expander("About the model"):
    st.write(
        f"**Model:** {bundle['name']}, trained on the Cleveland heart disease dataset (302 unique patients).\n\n"
        f"**Held-out test performance at the {threshold:.0%} cutoff:** "
        f"recall {bundle['test_recall']:.0%}, precision {bundle['test_precision']:.0%}, "
        f"ROC-AUC {bundle['test_roc_auc']:.2f}. The test set has only 61 patients, "
        f"so these numbers have wide uncertainty (see the README for bootstrap confidence intervals)."
    )