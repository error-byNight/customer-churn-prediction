import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # Import joblib for saving objects

# Define directories
data_dir = 'data/raw'
processed_data_dir = 'data/processed'
models_dir = 'models' # Ensure this directory exists to save models and preprocessor
os.makedirs(processed_data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- Configuration (matching previous scripts) ---
END_DATE = datetime(2024, 7, 31)
CHURN_WINDOW_DAYS = 60 # Define churn as no activity for the last 60 days of the data period

# Load the initial consolidated DataFrame with churn label
try:
    master_df = pd.read_csv(os.path.join(processed_data_dir, 'consolidated_churn_data_initial.csv'), parse_dates=[
        'Overall Last Activity Date',
        'Last Web Activity Date',
        'Last Transaction Date',
        'Last Support Activity Date',
        'Last Social Activity Date'
    ])
    print("Consolidated initial data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading consolidated data: {e}. Please ensure '01_data_consolidation_churn.py' was run.")
    exit()

# Load raw event data again for window-based feature engineering
try:
    web_logs_df = pd.read_csv(os.path.join(data_dir, 'web_logs.csv'), parse_dates=['Timestamp'])
    transaction_df = pd.read_csv(os.path.join(data_dir, 'transactions.csv'), parse_dates=['Transaction Date'])
    support_df = pd.read_csv(os.path.join(data_dir, 'support_interactions.csv'), parse_dates=['Support Ticket Date'])
    social_df = pd.read_csv(os.path.join(data_dir, 'social_media_engagement.csv'), parse_dates=['Engagement Date'])
    print("Raw event data loaded for feature engineering.")
except FileNotFoundError as e:
    print(f"Error loading raw event data: {e}. Please ensure 'generate_and_load_data.py' was run.")
    exit()


# --- Step 3: Feature Engineering ---
print("\nStarting Feature Engineering...")

# Define time windows for recency and frequency features
time_windows = [7, 30, 90] # Days

# Function to calculate features for a given DataFrame and date column
def calculate_event_features(df, date_col, prefix):
    features = pd.DataFrame({'User ID': master_df['User ID'].unique()}) # Ensure all users are present
    df_filtered = df[df[date_col] <= END_DATE].copy() # Filter data up to END_DATE

    for window in time_windows:
        window_start = END_DATE - timedelta(days=window)
        df_window = df_filtered[df_filtered[date_col] >= window_start]

        # Count of events in window
        event_counts = df_window.groupby('User ID').size().reset_index(name=f'{prefix}_count_last_{window}d')
        features = pd.merge(features, event_counts, on='User ID', how='left')

        # Additional features based on data type
        if prefix == 'web_log':
            avg_page_views = df_window.groupby('User ID')['Page Views'].mean().reset_index(name=f'{prefix}_avg_page_views_last_{window}d')
            avg_session_duration = df_window.groupby('User ID')['Session Duration'].mean().reset_index(name=f'{prefix}_avg_session_duration_last_{window}d')
            features = pd.merge(features, avg_page_views, on='User ID', how='left')
            features = pd.merge(features, avg_session_duration, on='User ID', how='left')
        elif prefix == 'transaction':
            total_amount = df_window.groupby('User ID')['Transaction Amount'].sum().reset_index(name=f'{prefix}_total_amount_last_{window}d')
            avg_amount = df_window.groupby('User ID')['Transaction Amount'].mean().reset_index(name=f'{prefix}_avg_amount_last_{window}d')
            features = pd.merge(features, total_amount, on='User ID', how='left')
            features = pd.merge(features, avg_amount, on='User ID', how='left')
        elif prefix == 'support':
            avg_resolution_time = df_window.groupby('User ID')['Resolution Time'].mean().reset_index(name=f'{prefix}_avg_resolution_time_last_{window}d')
            avg_satisfaction_score = df_window.groupby('User ID')['Customer Satisfaction Score'].mean().reset_index(name=f'{prefix}_avg_satisfaction_score_last_{window}d')
            features = pd.merge(features, avg_resolution_time, on='User ID', how='left')
            features = pd.merge(features, avg_satisfaction_score, on='User ID', how='left')
        elif prefix == 'social':
            total_engagement_freq = df_window.groupby('User ID')['Engagement Frequency'].sum().reset_index(name=f'{prefix}_total_engagement_freq_last_{window}d')
            features = pd.merge(features, total_engagement_freq, on='User ID', how='left')

    return features

# Calculate features for each event type
web_features = calculate_event_features(web_logs_df, 'Timestamp', 'web_log')
transaction_features = calculate_event_features(transaction_df, 'Transaction Date', 'transaction')
support_features = calculate_event_features(support_df, 'Support Ticket Date', 'support')
social_features = calculate_event_features(social_df, 'Engagement Date', 'social')

# Merge all new features into the master_df
master_df = pd.merge(master_df, web_features, on='User ID', how='left')
master_df = pd.merge(master_df, transaction_features, on='User ID', how='left')
master_df = pd.merge(master_df, support_features, on='User ID', how='left')
master_df = pd.merge(master_df, social_features, on='User ID', how='left')

print(f"Master DataFrame shape after adding engineered features: {master_df.shape}")

# --- Handle Missing Values for newly created features ---
# For count/sum features, NaN means no activity in that window, so fill with 0
# For average features, NaN means no activity, so fill with 0.
for col in master_df.columns:
    if '_count_' in col or '_total_' in col or '_avg_' in col:
        master_df[col] = master_df[col].fillna(0)

print("\nMissing values in engineered features filled with 0.")

# --- Step 4: Categorical Encoding and Numerical Scaling ---
print("\nStarting Categorical Encoding and Numerical Scaling...")

# Separate features (X) and target (y) before preprocessing
# Drop User ID and all original date columns as they are not features for the model
columns_to_drop_from_X = [
    'User ID',
    'Churn', # This is our target variable
    'Overall Last Activity Date',
    'Last Web Activity Date',
    'Last Transaction Date',
    'Last Support Activity Date',
    'Last Social Activity Date'
]
X = master_df.drop(columns=columns_to_drop_from_X)
y = master_df['Churn']

print(f"Shape of X before preprocessing: {X.shape}")
print(f"Columns in X before preprocessing: {X.columns.tolist()}")


# Identify categorical and numerical columns in X
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

# Define which numerical columns should be scaled and which should be passed through
# 'Account Age' and 'Days Since Last Activity' are numerical but are durations/ages
# and are often kept as is or transformed differently. Let's pass them through for now.
# All other numerical columns (Age and engineered features) will be scaled.

cols_to_scale = [col for col in numerical_cols if col not in ['Account Age', 'Days Since Last Activity']]
cols_to_passthrough_numerical = [col for col in numerical_cols if col in ['Account Age', 'Days Since Last Activity']]


print(f"Categorical columns for OneHotEncoder: {categorical_cols}")
print(f"Numerical columns for StandardScaler: {cols_to_scale}")
print(f"Numerical columns to passthrough: {cols_to_passthrough_numerical}")

# --- Debugging X and ColumnTransformer inputs ---
print("\n--- Debugging X and ColumnTransformer inputs ---")
print(f"X.head():\n{X.head()}")
print(f"X.dtypes:\n{X.dtypes}")
print(f"Are all cols_to_scale in X? {all(col in X.columns for col in cols_to_scale)}")
print(f"Are all categorical_cols in X? {all(col in X.columns for col in categorical_cols)}")
print(f"Are all cols_to_passthrough_numerical in X? {all(col in X.columns for col in cols_to_passthrough_numerical)}")
print("--- End Debugging ---")


# Create a preprocessing pipeline for numerical and categorical features
# We will explicitly define the passthrough for clarity and robustness.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), cols_to_scale),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('passthrough_num', 'passthrough', cols_to_passthrough_numerical) # Explicitly pass through these numerical columns
    ]
)

# Fit and transform the features
X_processed = preprocessor.fit_transform(X)

# Convert to dense array if it's a sparse matrix (OneHotEncoder often outputs sparse)
if hasattr(X_processed, 'toarray'):
    X_processed = X_processed.toarray()

print(f"Shape of X_processed after preprocessing (and toarray if sparse): {X_processed.shape}")
print(f"Type of X_processed: {type(X_processed)}")
# Print a slice to see content, but only if it's not too wide
if X_processed.shape[1] < 50: # Avoid printing extremely wide arrays
    print(f"First 5 rows of X_processed:\n{X_processed[:5, :]}")


# Get feature names after one-hot encoding and scaling using preprocessor.get_feature_names_out()
all_feature_names = preprocessor.get_feature_names_out()
print(f"Number of feature names generated: {len(all_feature_names)}")
print(f"First few feature names: {all_feature_names[:5]}")
if len(all_feature_names) > 5:
    print(f"Last few feature names: {all_feature_names[-5:]}")


# Create a DataFrame from the processed features
X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

# Add User ID and Churn back to the processed DataFrame
# User ID was dropped from X, so we need to get it from the original master_df
master_df_final = X_processed_df.copy()
master_df_final['User ID'] = master_df['User ID']
master_df_final['Churn'] = master_df['Churn']


print(f"Final master_df_final shape after encoding and scaling: {master_df_final.shape}")
print("Final master_df_final info:")
master_df_final.info()
print("\nFinal master_df_final head:")
print(master_df_final.head())

# Save the fully processed DataFrame
output_filepath_final = os.path.join(processed_data_dir, 'consolidated_churn_data_final.csv')
master_df_final.to_csv(output_filepath_final, index=False)
print(f"\nFully processed DataFrame saved to {output_filepath_final}")

# Save the fitted preprocessor
joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
print(f"Fitted preprocessor saved to {os.path.join(models_dir, 'preprocessor.pkl')}")

# Save the list of original feature names that the preprocessor expects as input
original_feature_names_for_preprocessor = X.columns.tolist()
joblib.dump(original_feature_names_for_preprocessor, os.path.join(models_dir, 'original_feature_names.pkl'))
print(f"Original feature names for preprocessor saved to {os.path.join(models_dir, 'original_feature_names.pkl')}")
