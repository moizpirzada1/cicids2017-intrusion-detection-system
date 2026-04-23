# 🛡️ CICIDS2017 Intrusion Detection System (IDS)

## 🚀 Project Overview
This project is a **Machine Learning-based Intrusion Detection System (IDS)** developed using the **CICIDS2017 dataset**.  
It detects whether network traffic is **normal (BENIGN)** or **malicious (ATTACK)** using a trained **Random Forest Classifier**.

The system is deployed as an interactive **Streamlit web application** for real-time predictions and cybersecurity analysis.

---

## 🌐 Live Demo
👉 https://cicids2017-intrusion-detection-system-ctalhe52qzdvpdcw6uxzzo.streamlit.app/

---

## 🧠 Problem Statement
Modern networks face continuous cyber threats such as:
- DDoS attacks
- Brute force attacks
- Botnet activity
- Port scanning

This project aims to build an intelligent system that can automatically detect such attacks using machine learning.

---

## ⚙️ Solution Approach

### 🔹 Machine Learning Model
- Algorithm: Random Forest Classifier
- Type: Binary Classification
- Classes:
  - `0` → BENIGN (Normal Traffic)
  - `1` → ATTACK (Malicious Traffic)

### 🔹 Dataset
- CICIDS2017 (Canadian Institute for Cybersecurity Dataset)
- Contains real-world network traffic features

---

## 🧹 Data Preprocessing
- Removed missing and infinite values
- Encoded categorical features
- Standardized numerical features using StandardScaler
- Aligned input features with training schema

---

## 🚀 Features of the System

✔ Upload CICIDS2017 CSV files for bulk prediction  
✔ Manual testing mode for single input prediction  
✔ Automatic data cleaning and preprocessing  
✔ Feature alignment with trained model  
✔ Real-time attack detection  
✔ User-friendly Streamlit interface  

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib (optional for analysis)

---

## 📂 Project Structure
