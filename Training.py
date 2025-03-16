import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(filename="model_training.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    logging.info("Dataset loaded successfully")
    return df

# Exploratory Data Analysis (EDA)
def explore_data(df):
    logging.info("Starting EDA")
    print("Dataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    # numeric_df.hist(bins=50, figsize=(20, 15))
    # plt.suptitle("Feature Distributions", fontsize=16)
    # plt.show()
    
    # # Boxplots to check for outliers
    # plt.figure(figsize=(20, 10))
    # sns.boxplot(data=numeric_df, orient='h')
    # plt.title("Feature Outliers", fontsize=16)
    # plt.show()
    
    # # Line plot for average reflectance over wavelengths
    # plt.figure(figsize=(12, 6))
    # mean_reflectance = numeric_df.mean()
    # plt.plot(mean_reflectance, marker='o', linestyle='-', color='b', label="Average Reflectance")
    # plt.xlabel("Wavelength Bands")
    # plt.ylabel("Reflectance")
    # plt.title("Average Reflectance Over Wavelengths", fontsize=16)
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    # # Heatmap for feature correlations
    # plt.figure(figsize=(15, 10))
    # sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
    # plt.title("Feature Correlation Heatmap", fontsize=16)
    # plt.show()

    return df

# Data Preprocessing with PCA
def preprocess_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    
    features = df.select_dtypes(include=[np.number]).iloc[:, :-1]  
    target = df.iloc[:, -1]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA to reduce dimensionality while keeping 95% variance
    pca = PCA(n_components=0.95)
    features_pca = pca.fit_transform(features_scaled)
    
    logging.info(f"Original Features: {features.shape[1]}, Reduced Features: {features_pca.shape[1]}")
    return features_pca, target, scaler, pca

def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    models = {}
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    grid_search = GridSearchCV(xgb, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='neg_mean_squared_error')
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    
    logging.info("Models trained successfully")
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        logging.info(f"{name} - MAE: {mae}, RMSE: {rmse}, R2: {r2}")
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_test, color='blue', label='Actual', alpha=0.6)
        plt.scatter(y_test, y_pred, color='red', label='Predicted', alpha=0.6)
        plt.xlabel("Actual DON Concentration")
        plt.ylabel("Predicted DON Concentration")
        plt.title(f"{name}: Actual vs. Predicted")
        plt.legend()
        plt.grid()
        plt.show()
    
    return results

# Model Interpretability with SHAP
def interpret_model(model, X_train):
    logging.info("Generating SHAP values for model interpretability")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)
    plt.show()

def save_models(models, scaler, pca, path="models"):
    os.makedirs(path, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(path, f"{name}.pkl"))
    joblib.dump(scaler, os.path.join(path, "scaler.pkl"))
    joblib.dump(pca, os.path.join(path, "pca.pkl"))
    logging.info("Models and preprocessors saved successfully")

def test_model_training():
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.rand(100)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    assert model.predict(X_dummy).shape[0] == 100, "Test Failed: Model output shape incorrect"
    logging.info("Unit test for model training passed")

def main():
    filepath = "MLE-Assignment.csv"
    df = load_data(filepath)
    df = explore_data(df)
    features, target, scaler, pca = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(features, target)
    models = train_models(X_train, y_train)
    
    results = evaluate_models(models, X_test, y_test)
    print("\nFinal Model Comparison:")
    print(pd.DataFrame(results))
    
    interpret_model(models['RandomForest'], X_train)
    interpret_model(models['XGBoost'], X_train)
    
    save_models(models, scaler, pca)
    
    test_model_training()

if __name__ == "__main__":
    main()
