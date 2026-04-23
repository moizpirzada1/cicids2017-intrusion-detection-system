import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# LOAD MODELS
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

# ===============================
# MODE SELECTION
# ===============================
mode = st.sidebar.radio("Select Mode", ["📁 CSV Upload", "✍️ Manual Test"])

# =========================================================
# 1. CSV MODE
# =========================================================
if mode == "📁 CSV Upload":

    file = st.file_uploader("Upload CICIDS2017 CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        st.write("### Raw Data")
        st.dataframe(df.head())

        # CLEAN
        df.columns = df.columns.str.strip()

        if "Label" in df.columns:
            df = df.drop(columns=["Label"])

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # ENCODE
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).factorize()[0]

        # ALIGN FEATURES
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        # SCALE + PREDICT
        X = scaler.transform(df)
        preds = model.predict(X)

        df["Prediction"] = np.where(preds == 0, "BENIGN", "ATTACK")

        st.write("### Results")
        st.dataframe(df)

        st.success("Prediction Completed ✔")

# =========================================================
# 2. MANUAL MODE
# =========================================================
else:

    st.write("### Enter Feature Values Manually")

    st.info("⚠ You must enter values similar to CICIDS2017 dataset scale")

    inputs = {}

    # Take first 8 features for demo (you can expand later)
    selected_features = feature_columns[:8]

    cols = st.columns(2)

    for i, col_name in enumerate(selected_features):

        with cols[i % 2]:
            inputs[col_name] = st.number_input(col_name, value=0.0)

    if st.button("Predict Attack / Normal"):

        # Convert to DataFrame
        input_df = pd.DataFrame([inputs])

        # Add missing columns
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        # Scale + Predict
        X = scaler.transform(input_df)
        pred = model.predict(X)[0]

        if pred == 0:
            st.success("✔ BENIGN (Normal Traffic)")
        else:
            st.error("🚨 ATTACK DETECTED")
