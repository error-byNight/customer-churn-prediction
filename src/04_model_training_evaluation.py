import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib # For saving models

# Define directories
processed_data_dir = 'data/processed'
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Load the processed datasets
try:
    X_train_resampled = pd.read_csv(os.path.join(processed_data_dir, 'X_train_resampled.csv'))
    y_train_resampled = pd.read_csv(os.path.join(processed_data_dir, 'y_train_resampled.csv')).squeeze() # .squeeze() to convert DataFrame to Series
    X_val = pd.read_csv(os.path.join(processed_data_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(processed_data_dir, 'y_val.csv')).squeeze()
    X_test = pd.read_csv(os.path.join(processed_data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv')).squeeze()
    print("Training, validation, and test sets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading processed datasets: {e}. Please ensure '03_model_development_split_imbalance.py' was run.")
    exit()

print(f"\nShapes of loaded data:")
print(f"X_train_resampled: {X_train_resampled.shape}, y_train_resampled: {y_train_resampled.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


# --- Step 3: Model Selection & Training ---
print("\n--- Starting Model Training and Evaluation ---")

# Define a function to evaluate models
def evaluate_model(model_name, y_true, y_pred, y_prob=None):
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}") # zero_division=0 to handle cases where no positive predictions are made
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    if y_prob is not None and len(np.unique(y_true)) > 1: # ROC AUC requires at least two classes
        print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    else:
        print("ROC-AUC: N/A (requires at least two classes in true labels)")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


# --- Logistic Regression ---
print("\nTraining Logistic Regression Model...")
log_reg = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', max_iter=1000)
# Using class_weight='balanced' is a good practice for imbalanced datasets,
# even though we used SMOTE, it can add robustness.
log_reg.fit(X_train_resampled, y_train_resampled)

# Predictions on Validation Set
y_val_pred_lr = log_reg.predict(X_val)
y_val_prob_lr = log_reg.predict_proba(X_val)[:, 1]
evaluate_model("Logistic Regression (Validation)", y_val, y_val_pred_lr, y_val_prob_lr)
joblib.dump(log_reg, os.path.join(models_dir, 'logistic_regression_model.pkl'))
print(f"Logistic Regression model saved to {os.path.join(models_dir, 'logistic_regression_model.pkl')}")


# --- Random Forest Classifier ---
print("\nTraining Random Forest Model...")
# Hyperparameter tuning for Random Forest (simplified for demonstration)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_leaf': [1, 5]
}
rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search_rf = GridSearchCV(estimator=rf_clf, param_grid=param_grid_rf, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train_resampled, y_train_resampled)

best_rf = grid_search_rf.best_estimator_
print(f"Best Random Forest parameters: {grid_search_rf.best_params_}")

# Predictions on Validation Set
y_val_pred_rf = best_rf.predict(X_val)
y_val_prob_rf = best_rf.predict_proba(X_val)[:, 1]
evaluate_model("Random Forest (Validation)", y_val, y_val_pred_rf, y_val_prob_rf)
joblib.dump(best_rf, os.path.join(models_dir, 'random_forest_model.pkl'))
print(f"Random Forest model saved to {os.path.join(models_dir, 'random_forest_model.pkl')}")


# --- XGBoost Classifier ---
print("\nTraining XGBoost Model...")
# Hyperparameter tuning for XGBoost (simplified)
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss',
                            scale_pos_weight=len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1]))
# scale_pos_weight is used for imbalanced datasets, even after SMOTE it can help.
# However, since SMOTE already balanced, scale_pos_weight might be 1.0 or not strictly necessary.
# Let's remove scale_pos_weight as SMOTE has already balanced it.
# If you didn't use SMOTE, this would be crucial: scale_pos_weight = (count_negative / count_positive)

xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss') # Removed scale_pos_weight as SMOTE balanced
grid_search_xgb = GridSearchCV(estimator=xgb_clf, param_grid=param_grid_xgb, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid_search_xgb.fit(X_train_resampled, y_train_resampled)

best_xgb = grid_search_xgb.best_estimator_
print(f"Best XGBoost parameters: {grid_search_xgb.best_params_}")

# Predictions on Validation Set
y_val_pred_xgb = best_xgb.predict(X_val)
y_val_prob_xgb = best_xgb.predict_proba(X_val)[:, 1]
evaluate_model("XGBoost (Validation)", y_val, y_val_pred_xgb, y_val_prob_xgb)
joblib.dump(best_xgb, os.path.join(models_dir, 'xgboost_model.pkl'))
print(f"XGBoost model saved to {os.path.join(models_dir, 'xgboost_model.pkl')}")


# --- Final Evaluation on Test Set (Important for unbiased evaluation) ---
print("\n--- Final Evaluation on Test Set ---")

# Logistic Regression Test Set Evaluation
y_test_pred_lr = log_reg.predict(X_test)
y_test_prob_lr = log_reg.predict_proba(X_test)[:, 1]
evaluate_model("Logistic Regression (Test)", y_test, y_test_pred_lr, y_test_prob_lr)

# Random Forest Test Set Evaluation
y_test_pred_rf = best_rf.predict(X_test)
y_test_prob_rf = best_rf.predict_proba(X_test)[:, 1]
evaluate_model("Random Forest (Test)", y_test, y_test_pred_rf, y_test_prob_rf)

# XGBoost Test Set Evaluation
y_test_pred_xgb = best_xgb.predict(X_test)
y_test_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
evaluate_model("XGBoost (Test)", y_test, y_test_pred_xgb, y_test_prob_xgb)

print("\nModel training and evaluation complete. Models saved to 'models' directory.")
