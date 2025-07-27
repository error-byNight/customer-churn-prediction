import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Define the base directory for raw data
data_dir = 'data/raw'
processed_data_dir = 'data/processed'
os.makedirs(processed_data_dir, exist_ok=True)

# --- Configuration (matching generate_and_load_data.py) ---
END_DATE = datetime(2024, 7, 31)
CHURN_WINDOW_DAYS = 60 # Define churn as no activity for the last 60 days of the data period

# --- Load DataFrames (assuming they were generated and saved) ---
try:
    web_logs_df = pd.read_csv(os.path.join(data_dir, 'web_logs.csv'), parse_dates=['Timestamp'])
    transaction_df = pd.read_csv(os.path.join(data_dir, 'transactions.csv'), parse_dates=['Transaction Date'])
    support_df = pd.read_csv(os.path.join(data_dir, 'support_interactions.csv'), parse_dates=['Support Ticket Date'])
    social_df = pd.read_csv(os.path.join(data_dir, 'social_media_engagement.csv'), parse_dates=['Engagement Date'])
    demographics_df = pd.read_csv(os.path.join(data_dir, 'customer_demographics.csv'))
    print("All raw data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Please ensure 'generate_and_load_data.py' was run successfully and data/raw files exist.")
    exit() # Exit if raw data isn't found

# --- Step 1: Data Consolidation and Feature Aggregation ---

# Initialize a master DataFrame with demographics as the base
master_df = demographics_df.copy()
print(f"Initial master_df shape: {master_df.shape}")

# --- Calculate Last Activity Date for each user from event data ---
# This is crucial for defining churn based on recency

# Web Logs: Last activity timestamp
last_web_activity = web_logs_df.groupby('User ID')['Timestamp'].max().reset_index()
last_web_activity.rename(columns={'Timestamp': 'Last Web Activity Date'}, inplace=True)
master_df = pd.merge(master_df, last_web_activity, on='User ID', how='left')

# Transactions: Last activity timestamp
last_transaction_activity = transaction_df.groupby('User ID')['Transaction Date'].max().reset_index()
last_transaction_activity.rename(columns={'Transaction Date': 'Last Transaction Date'}, inplace=True)
master_df = pd.merge(master_df, last_transaction_activity, on='User ID', how='left')

# Support Interactions: Last activity timestamp
last_support_activity = support_df.groupby('User ID')['Support Ticket Date'].max().reset_index()
last_support_activity.rename(columns={'Support Ticket Date': 'Last Support Activity Date'}, inplace=True)
master_df = pd.merge(master_df, last_support_activity, on='User ID', how='left')

# Social Media Engagement: Last activity timestamp
last_social_activity = social_df.groupby('User ID')['Engagement Date'].max().reset_index()
last_social_activity.rename(columns={'Engagement Date': 'Last Social Activity Date'}, inplace=True)
master_df = pd.merge(master_df, last_social_activity, on='User ID', how='left')

# Combine all last activity dates to find the overall last activity date for each user
# Fill NaN with a very old date so that users with no activity in a category don't skew the max
# The NaT (Not a Time) values will correctly be ignored by .max() if other dates exist.
# If ALL activity dates for a user are NaT (meaning they had no activity in any category),
# then Overall Last Activity Date will be NaT.
master_df['Overall Last Activity Date'] = master_df[[
    'Last Web Activity Date',
    'Last Transaction Date',
    'Last Support Activity Date',
    'Last Social Activity Date'
]].max(axis=1)

# For users with absolutely no activity across any source (NaN in Overall Last Activity Date),
# fill these with a date far in the past to ensure they are marked as churned if they meet the criteria.
# This handles cases where a user might be in demographics but has no event data.
master_df['Overall Last Activity Date'].fillna(END_DATE - timedelta(days=365*10), inplace=True) # 10 years ago

print(f"Master DataFrame after merging last activity dates: {master_df.shape}")
print("Master DataFrame head with last activity dates:")
print(master_df[['User ID', 'Overall Last Activity Date', 'Last Web Activity Date', 'Last Transaction Date']].head())


# --- Step 2: Defining the Churn Label ---

# Calculate days since last activity relative to the END_DATE
master_df['Days Since Last Activity'] = (END_DATE - master_df['Overall Last Activity Date']).dt.days

# Define churn based on the CHURN_WINDOW_DAYS
master_df['Churn'] = (master_df['Days Since Last Activity'] > CHURN_WINDOW_DAYS).astype(int)

print(f"\nChurn label created. Churn distribution:")
print(master_df['Churn'].value_counts())
print(f"Percentage of churned customers: {master_df['Churn'].mean() * 100:.2f}%")

# Save the consolidated DataFrame with churn label for the next steps
output_filepath = os.path.join(processed_data_dir, 'consolidated_churn_data_initial.csv')
master_df.to_csv(output_filepath, index=False)
print(f"\nConsolidated DataFrame with churn label saved to {output_filepath}")

print("\nMaster DataFrame info after churn labeling:")
master_df.info()

print("\nMaster DataFrame head after churn labeling:")
print(master_df[['User ID', 'Account Age', 'Overall Last Activity Date', 'Days Since Last Activity', 'Churn']].head())
