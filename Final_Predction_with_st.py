import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and preprocessors
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
rf_model = joblib.load("models/RandomForest.pkl")
xgb_model = joblib.load("models/XGBoost.pkl")

# Streamlit App
st.title("Spectral Data Prediction App")
st.write("Upload a CSV file with spectral data to get predictions.")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.write(df.head())
    
    # Select numeric features only
    features = df.select_dtypes(include=[np.number])
    
    # Ensure correct feature count
    expected_features = scaler.n_features_in_
    if features.shape[1] != expected_features:
        st.error(f"Feature mismatch! Uploaded data has {features.shape[1]} features, but model expects {expected_features}.")
    else:
        # Preprocess the data
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Make predictions
        rf_preds = rf_model.predict(features_pca)
        xgb_preds = xgb_model.predict(features_pca)
        
        df["RandomForest Prediction"] = rf_preds
        df["XGBoost Prediction"] = xgb_preds
        
        st.write("### Predictions:")
        st.write(df[["RandomForest Prediction", "XGBoost Prediction"]].head())
        
        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
