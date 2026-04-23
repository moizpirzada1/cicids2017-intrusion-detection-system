import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="CICIDS2017 IDS",
    page_icon="🛡️",
    layout="wide"
)

# ===============================
# LOAD MODEL & SCALER
# ===============================
model = joblib.load("random_forest_cicids2017.pkl")
scaler = joblib.load("scaler_cicids2017.pkl")

# ===============================
# TITLE
# ===============================
st.title("🛡️ Network Intrusion Detection System")
st.markdown("Machine Learning based IDS using CICIDS2017 + Random Forest")

# ===============================
# SIDEBAR
# ===============================
option = st.sidebar.radio("Select Mode:", ["Upload CSV", "Manual Test"])

# ===============================
# CSV MODE
# ===============================
if option == "Upload CSV":
    file = st.file_uploader("Upload your CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.write("### Input Data Preview")
        st.dataframe(df.head())

        # CLEAN DATA
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # ENCODE OBJECT COLUMNS
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).factorize()[0]

        # SCALE DATA
        X = scaler.transform(df)

        # PREDICTION
        preds = model.predict(X)

        df["Prediction"] = np.where(preds == 0, "BENIGN", "ATTACK")

        st.write("### Prediction Results")
        st.dataframe(df)

        st.success("Prediction Completed ✔")

# ===============================
# MANUAL TEST MODE
# ===============================
else:
    st.write("### Manual Input Test")

    col1, col2, col3 = st.columns(3)

    with col1:
        f1 = st.number_input("Feature 1", 0.0)
        f2 = st.number_input("Feature 2", 0.0)
        f3 = st.number_input("Feature 3", 0.0)

    with col2:
        f4 = st.number_input("Feature 4", 0.0)
        f5 = st.number_input("Feature 5", 0.0)
        f6 = st.number_input("Feature 6", 0.0)

    with col3:
        f7 = st.number_input("Feature 7", 0.0)
        f8 = st.number_input("Feature 8", 0.0)
        f9 = st.number_input("Feature 9", 0.0)

    if st.button("Predict"):
        sample = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9]])

        sample = scaler.transform(sample)
        result = model.predict(sample)[0]

        if result == 0:
            st.success("✔ BENIGN (Normal Traffic)")
        else:
            st.error("🚨 ATTACK DETECTED")
