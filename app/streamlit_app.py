import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.preprocessing import clean_and_basic_process, encode_categoricals
from src.features import create_features

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Intelligence Engine")
st.write("Upload a CSV file with customer records, and the model will predict churn probability.")

# --------------------------------------------------
# LOAD MODEL + ARTIFACTS
# --------------------------------------------------
MODEL_PATH = os.path.join(ROOT_DIR, "models", "xgb_model.joblib")
ENC_PATH = os.path.join(ROOT_DIR, "models", "encoders.joblib")
FEATURE_PATH = os.path.join(ROOT_DIR, "models", "feature_names.joblib")

model = joblib.load(MODEL_PATH)

try:
    encoders = joblib.load(ENC_PATH)
except:
    encoders = {}

try:
    model_features = joblib.load(FEATURE_PATH)
except:
    model_features = None
    st.warning("⚠ Model feature_names not found — aligning using numeric columns only.")

# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])


def align_features(df, model_features):
    """Ensure prediction dataframe matches EXACT order & columns expected by the model."""
    if model_features is None:
        # fallback: numeric only
        return df.select_dtypes(include=["number"])

    aligned = pd.DataFrame()

    for col in model_features:
        if col in df.columns:
            aligned[col] = df[col]
        else:
            aligned[col] = 0  # missing column → fill default

    return aligned[model_features]


# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded:
    # STEP 1: Read CSV safely
    try:
        df = pd.read_csv(uploaded, encoding="utf-8")
    except:
        df = pd.read_csv(uploaded, encoding="latin1")

    st.subheader("📁 Raw Input Data (first 10 rows)")
    st.dataframe(df.head(10))

    # Debug counts
    st.write("🔍 Rows BEFORE cleaning:", len(df))

    # STEP 2: Cleaning + normalization
    df_clean = clean_and_basic_process(df)
    st.write("🔍 Rows AFTER clean:", len(df_clean))

    # STEP 3: Encode categoricals
    df_encoded, _ = encode_categoricals(df_clean)
    st.write("🔍 Rows AFTER encoding:", len(df_encoded))

    # STEP 4: Feature engineering (safe)
    try:
        df_feats = create_features(df_encoded)
    except Exception:
        df_feats = df_encoded.copy()

    # Remove churn if exists
    if "Churn" in df_feats.columns:
        df_feats = df_feats.drop(columns=["Churn"])

    st.write("🔍 Rows BEFORE feature alignment:", len(df_feats))

    # STEP 5: Keep only numeric columns
    df_feats = df_feats.select_dtypes(include=["number"])

    # STEP 6: Align EXACTLY to model training features
    df_aligned = align_features(df_feats, model_features)

    st.write("🔍 Rows used for prediction:", len(df_aligned))
    st.write("🔍 Final feature count:", df_aligned.shape[1])

    # STEP 7: Prediction
    preds = model.predict_proba(df_aligned)[:, 1]

    df_output = df.copy()
    df_output["churn_probability"] = preds

    st.subheader("📈 Predictions")
    st.dataframe(df_output)

    # STEP 8: Downloadable results
    csv = df_output.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Predictions", data=csv, file_name="churn_predictions.csv")
