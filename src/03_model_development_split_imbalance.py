import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib # For saving the preprocessor and potentially models

# Define directories
processed_data_dir = 'data/processed'
models_dir = 'models' # Directory to save trained models and preprocessors
os.makedirs(models_dir, exist_ok=True)

# Load the fully processed DataFrame
try:
    master_df_final = pd.read_csv(os.path.join(processed_data_dir, 'consolidated_churn_data_final.csv'))
    print("Fully processed data loaded successfully for model development.")
except FileNotFoundError as e:
    print(f"Error loading final processed data: {e}. Please ensure '02_feature_engineering.py' was run.")
    exit()

# Separate features (X) and target (y)
# Drop 'User ID' as it's an identifier, not a feature for the model
X = master_df_final.drop(columns=['User ID', 'Churn'])
y = master_df_final['Churn']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Original Churn distribution (y): {Counter(y)}")
print(f"Original Churn percentage: {y.mean() * 100:.2f}%")

# --- Step 1: Splitting Data into Training, Validation, and Test Sets ---
# Industry standard often uses 60/20/20 or 70/15/15 splits.
# We'll use 80% for training+validation, 20% for final testing.
# Then split the 80% into 75% training and 25% validation (resulting in 60/20/20 overall).

# First split: 80% for train+val, 20% for test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nAfter first split (Train+Val/Test):")
print(f"X_train_val shape: {X_train_val.shape}, y_train_val shape: {y_train_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"y_test Churn distribution: {Counter(y_test)}")

# Second split: 75% of train+val for train, 25% for val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
) # 0.25 of 0.8 is 0.2, so 60/20/20 split

print(f"\nAfter second split (Train/Validation):")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"y_train Churn distribution: {Counter(y_train)}")
print(f"y_val Churn distribution: {Counter(y_val)}")

# --- Step 2: Handling Imbalance using SMOTE (on Training Data Only) ---
print("\nHandling class imbalance using SMOTE on training data...")

# Initialize SMOTE
smote = SMOTE(random_state=42, k_neighbors=1) # Set k_neighbors to 1 as minority class has only 2 samples
# Apply SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original training data shape: {X_train.shape}")
print(f"Resampled training data shape: {X_train_resampled.shape}")
print(f"Original training Churn distribution: {Counter(y_train)}")
print(f"Resampled training Churn distribution: {Counter(y_train_resampled)}")

# Save the resampled training data and the test/validation sets
# This is good practice for reproducibility and to avoid re-running SMOTE
X_train_resampled.to_csv(os.path.join(processed_data_dir, 'X_train_resampled.csv'), index=False)
y_train_resampled.to_csv(os.path.join(processed_data_dir, 'y_train_resampled.csv'), index=False)
X_val.to_csv(os.path.join(processed_data_dir, 'X_val.csv'), index=False)
y_val.to_csv(os.path.join(processed_data_dir, 'y_val.csv'), index=False)
X_test.to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)

print("\nTraining, validation, and test sets (with resampled training) saved.")

# In a real pipeline, you would also save the preprocessor (ColumnTransformer)
# so you can apply the exact same transformations to new, unseen data.
# We'll assume 'preprocessor' is available from the previous script run for now.
# For a robust deployment, you'd save it. Let's add that.

# Note: The preprocessor object itself is not directly available here unless passed from the previous script.
# For a full pipeline, you'd typically save the preprocessor object after fitting it.
# Let's add a placeholder to remind ourselves to save it if we were building a full pipeline.
# For now, we'll assume the same transformations will be applied.
# In a real scenario, you would save the preprocessor object from 02_feature_engineering.py
# For example: joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
# And load it here: preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
print("\nRemember: In a production pipeline, save and load your preprocessor (ColumnTransformer) object!")

print("\nData splitting and imbalance handling complete. Ready for model training.")
