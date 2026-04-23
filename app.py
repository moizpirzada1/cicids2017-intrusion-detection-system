import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# LOAD MODEL + SCALER + COLUMNS
# ===============================
model = joblib.load("random_forest_cicids2017.pkl")
scaler = joblib.load("scaler_cicids2017.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ===============================
# APP UI
# ===============================
st.set_page_config(page_title="CICIDS2017 IDS", layout="wide")

st.title("🛡️ Intrusion Detection System (CICIDS2017)")

file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.write("### Raw Data")
    st.dataframe(df.head())

    # CLEAN COLUMN NAMES
    df.columns = df.columns.str.strip()

    # DROP LABEL IF EXISTS
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # HANDLE INF/NAN
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ENCODE OBJECT COLUMNS
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).factorize()[0]

    # ===============================
    # ALIGN FEATURES (VERY IMPORTANT FIX)
    # ===============================

    # add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # keep only training columns in correct order
    df = df[feature_columns]

    # ===============================
    # SCALE + PREDICT
    # ===============================
    X = scaler.transform(df)
    preds = model.predict(X)

    df["Prediction"] = np.where(preds == 0, "BENIGN", "ATTACK")

    st.write("### Results")
    st.dataframe(df)

    st.success("Prediction Completed ✔")
