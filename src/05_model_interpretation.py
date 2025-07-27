import pandas as pd
import numpy as np
import os
import joblib # For loading models
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb # Import for XGBoost specifically if needed for SHAP

# Define directories
processed_data_dir = 'data/processed'
models_dir = 'models'
reports_figures_dir = 'reports/figures'
os.makedirs(reports_figures_dir, exist_ok=True) # Ensure the figures directory exists

# Load the processed datasets (X_test for unbiased interpretation)
try:
    X_test = pd.read_csv(os.path.join(processed_data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv')).squeeze()
    print("Test data loaded successfully for model interpretation.")
except FileNotFoundError as e:
    print(f"Error loading test datasets: {e}. Please ensure '03_model_development_split_imbalance.py' was run and files exist.")
    exit()

# Load the best performing XGBoost model
try:
    best_xgb = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
    print("XGBoost model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading XGBoost model: {e}. Please ensure '04_model_training_evaluation.py' was run and the model was saved.")
    exit()

print("\n--- Starting Model Interpretation and Explainability ---")

# --- 1. Global Feature Importance (from XGBoost) ---
print("\nCalculating Global Feature Importance (XGBoost)...")
# XGBoost's built-in feature importance
feature_importances = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': best_xgb.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 10 Global Feature Importances:")
print(feature_importances.head(10))

# Plotting Global Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15), palette='viridis')
plt.title('Top 15 Global Feature Importances (XGBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(reports_figures_dir, 'global_feature_importance.png'))
print(f"Global feature importance plot saved to {os.path.join(reports_figures_dir, 'global_feature_importance.png')}")
plt.close() # Close the plot to free memory


# --- 2. SHAP Explanations (for overall feature importance and individual predictions) ---
print("\nCalculating SHAP values for Model Explainability...")

# Create a SHAP Explainer object for the XGBoost model
# For tree-based models, shap.TreeExplainer is efficient
explainer = shap.TreeExplainer(best_xgb)

# Calculate SHAP values for the test set
# This might take a moment depending on the number of samples and features
shap_values = explainer.shap_values(X_test)

print(f"SHAP values calculated. Shape: {shap_values.shape}")

# --- SHAP Summary Plot (Global Feature Importance) ---
# This plot summarizes the impact of features on the model output
print("Generating SHAP Summary Plot (Global)...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Global Feature Importance (Bar Plot)')
plt.tight_layout()
plt.savefig(os.path.join(reports_figures_dir, 'shap_global_feature_importance_bar.png'))
print(f"SHAP global feature importance (bar) plot saved to {os.path.join(reports_figures_dir, 'shap_global_feature_importance_bar.png')}")
plt.close()

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False) # Default is dot plot
plt.title('SHAP Global Feature Importance (Dot Plot)')
plt.tight_layout()
plt.savefig(os.path.join(reports_figures_dir, 'shap_global_feature_importance_dot.png'))
print(f"SHAP global feature importance (dot) plot saved to {os.path.join(reports_figures_dir, 'shap_global_feature_importance_dot.png')}")
plt.close()


# --- SHAP Force Plot (Individual Prediction Explanation) ---
# Explain a single prediction (e.g., the first churner in the test set)
print("\nGenerating SHAP Force Plot for an individual prediction...")

# Find an actual churner in the test set for a more interesting explanation
churn_indices = y_test[y_test == 1].index.tolist()
if churn_indices:
    sample_index = churn_indices[0] # Take the first churner
    print(f"Explaining prediction for User ID at index {sample_index} (a churner).")
    # Force plots are interactive in notebooks, saving them as static images is less informative.
    # We'll just print a message about how to view them.
    print("SHAP force plots are best viewed interactively in a Jupyter Notebook environment.")
    print("To view, you would typically run:")
    print(f"shap.initjs()")
    print(f"shap.force_plot(explainer.expected_value, shap_values[{sample_index},:], X_test.iloc[{sample_index},:])")
else:
    print("No churners found in the test set to generate a force plot for.")
    print("This should not happen with the updated data generation.")


print("\nModel interpretation complete. Feature importance plots saved to 'reports/figures' directory.")
