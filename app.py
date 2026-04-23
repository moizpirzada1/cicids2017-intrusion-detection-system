
---

# 🚀 FINAL CLEAN `app.py` (COPY EXACTLY THIS)

Use this **ONLY Python code**:

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# PAGE CONFIG
st.set_page_config(
    page_title="CICIDS2017 Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# LOAD MODEL
model = joblib.load("random_forest_cicids2017.pkl")
scaler = joblib.load("scaler_cicids2017.pkl")

# TITLE
st.title("🛡️ Network Intrusion Detection System")
st.markdown("Detect Normal vs Attack traffic using Machine Learning")

# SIDEBAR
option = st.sidebar.radio("Choose Input Method:", ["Upload CSV", "Manual Input"])

# CSV MODE
if option == "Upload CSV":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).factorize()[0]

        X = scaler.transform(df)
        preds = model.predict(X)

        df["Prediction"] = np.where(preds == 0, "BENIGN", "ATTACK")

        st.dataframe(df)
        st.success("Prediction Done ✔")

# MANUAL MODE
else:
    st.write("Manual Test")

    f1 = st.number_input("Feature 1", 0.0)
    f2 = st.number_input("Feature 2", 0.0)
    f3 = st.number_input("Feature 3", 0.0)

    if st.button("Predict"):
        sample = np.array([[f1, f2, f3]])
        sample = scaler.transform(sample)

        result = model.predict(sample)[0]

        if result == 0:
            st.success("BENIGN")
        else:
            st.error("ATTACK")
