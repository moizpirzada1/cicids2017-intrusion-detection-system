```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="CICIDS2017 Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# ===============================
# LOAD MODEL & SCALER
# ===============================
model = joblib.load("random_forest_cicids2017.pkl")
scaler = joblib.load("scaler_cicids2017.pkl")

# ===============================
# UI HEADER
# ===============================
st.title("🛡️ Network Intrusion Detection System")
st.markdown("Detect **Normal vs Attack** traffic using Machine Learning (Random Forest + CICIDS2017)")

st.sidebar.title("⚙️ Controls")
option = st.sidebar.radio("Choose Input Method:", ["Upload CSV", "Manual Input (Simple Test)"])

# ===============================
# CSV UPLOAD MODE
# ===============================
if option == "Upload CSV":
    file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.write("### Preview of Data")
        st.dataframe(df.head())

        # preprocessing
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # encode categorical
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).factorize()[0]

        # scaling
        X = scaler.transform(df)

        # prediction
        preds = model.predict(X)

        df["Prediction"] = np.where(preds == 0, "BENIGN", "ATTACK")

        st.write("### Results")
        st.dataframe(df)

        st.success("Prediction Completed ✔")

# ===============================
# SIMPLE MANUAL TEST MODE
# ===============================
else:
    st.write("### Enter Sample Values")

    col1, col2, col3 = st.columns(3)

    with col1:
        f1 = st.number_input("Feature 1", value=0.0)
        f2 = st.number_input("Feature 2", value=0.0)
        f3 = st.number_input("Feature 3", value=0.0)

    with col2:
        f4 = st.number_input("Feature 4", value=0.0)
        f5 = st.number_input("Feature 5", value=0.0)
        f6 = st.number_input("Feature 6", value=0.0)

    with col3:
        f7 = st.number_input("Feature 7", value=0.0)
        f8 = st.number_input("Feature 8", value=0.0)
        f9 = st.number_input("Feature 9", value=0.0)

    if st.button("Predict 🚀"):
        sample = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9]])

        sample_scaled = scaler.transform(sample)
        result = model.predict(sample_scaled)[0]

        if result == 0:
            st.success("✔ BENIGN (Normal Traffic)")
        else:
            st.error("🚨 ATTACK DETECTED")
```
