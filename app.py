import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# LOAD MODELS & FILES
# ===============================
model = joblib.load("random_forest_cicids2017.pkl")
scaler = joblib.load("scaler_cicids2017.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="CICIDS2017 IDS",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CICIDS2017 Intrusion Detection System")
st.markdown("Upload network traffic CSV for **BENIGN vs ATTACK** detection")

# ===============================
# FILE UPLOAD
# ===============================
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.write("### Original Data")
    st.dataframe(df.head())

    # ===============================
    # CLEAN DATA
    # ===============================
    df.columns = df.columns.str.strip()

    # DROP LABEL IF EXISTS
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # HANDLE INF & NAN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # ENCODE OBJECT COLUMNS
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).factorize()[0]

    # ===============================
    # ALIGN COLUMNS WITH TRAINING DATA
    # ===============================
    # ADD MISSING COLUMNS
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # KEEP SAME ORDER
    df = df[feature_columns]

    # ===============================
    # SCALE DATA
    # ===============================
    X = scaler.transform(df)

    # ===============================
    # PREDICTION
    # ===============================
    preds = model.predict(X)

    df["Prediction"] = np.where(preds == 0, "BENIGN", "ATTACK")

    # ===============================
    # OUTPUT
    # ===============================
    st.write("### Prediction Results")
    st.dataframe(df)

    st.success("Analysis Completed ✔")
