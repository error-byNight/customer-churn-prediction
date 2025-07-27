import pandas as pd
import numpy as np
from faker import Faker
import os
from datetime import datetime, timedelta

# Initialize Faker for realistic data generation
fake = Faker()

# Define the base directory for data
data_dir = 'data/raw'
os.makedirs(data_dir, exist_ok=True) # Ensure the directory exists

# --- Configuration for Synthetic Data ---
NUM_USERS = 1000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 7, 31) # Data up to end of July 2024
CHURN_WINDOW_DAYS = 60 # Define churn as no activity for the last 60 days of the data period
CHURN_PERCENTAGE = 0.10 # Target 10% of users to be churned for better evaluation

# Generate User IDs
user_ids = [f'user_{i:04d}' for i in range(NUM_USERS)]
np.random.shuffle(user_ids) # Shuffle to randomly assign churn status

# Determine which users will be churners
num_churners = int(NUM_USERS * CHURN_PERCENTAGE)
churner_ids = set(user_ids[:num_churners])
active_ids = set(user_ids[num_churners:])

print(f"Generating data for {NUM_USERS} users, with {num_churners} ({CHURN_PERCENTAGE*100:.0f}%) targeted as churners.")

# --- Helper function to generate activity for a user ---
def generate_user_activity(user_id, start_date_activity, end_date_activity, num_events, event_type):
    activity_data = []
    # Ensure at least one event if num_events is not 0
    if num_events > 0:
        for _ in range(num_events):
            timestamp = fake.date_time_between(start_date=start_date_activity, end_date=end_date_activity)
            if event_type == 'web_log':
                activity_data.append([user_id, timestamp, np.random.randint(1, 20), np.random.randint(30, 1200), np.random.choice(['google.com', 'facebook.com', 'bing.com', 'direct', 'internal_page', 'twitter.com'])])
            elif event_type == 'transaction':
                activity_data.append([user_id, fake.uuid4(), round(np.random.uniform(5, 5000), 2), timestamp, np.random.choice(['Savings', 'Checking', 'Loans', 'Investments', 'Credit Card', 'Insurance'])])
            elif event_type == 'support':
                activity_data.append([user_id, fake.uuid4(), np.random.choice(['Billing', 'Technical', 'Account Access', 'Product Inquiry', 'Complaint', 'Feature Request']), np.random.randint(1, 72), np.random.randint(1, 6), timestamp])
            elif event_type == 'social':
                activity_data.append([user_id, np.random.choice(['like', 'share', 'comment', 'post', 'mention']), timestamp, np.random.randint(1, 10)])
    return activity_data

# --- 1. Web Logs Data ---
print("Generating Web Logs data...")
web_logs_data = []
for user_id in user_ids:
    if user_id in churner_ids:
        # For churners, ensure last activity is before the churn window
        last_activity_date = END_DATE - timedelta(days=CHURN_WINDOW_DAYS + np.random.randint(1, 90)) # Last activity 61-150 days ago
        activity_start = START_DATE
        activity_end = last_activity_date
    else:
        # For active users, ensure recent activity
        activity_start = END_DATE - timedelta(days=np.random.randint(1, 60)) # Recent activity in last 60 days
        activity_end = END_DATE

    web_logs_data.extend(generate_user_activity(user_id, activity_start, activity_end, np.random.randint(5, 15), 'web_log')) # Vary events per user

web_logs_df = pd.DataFrame(web_logs_data, columns=['User ID', 'Timestamp', 'Page Views', 'Session Duration', 'Referrer URL'])
web_logs_df.to_csv(os.path.join(data_dir, 'web_logs.csv'), index=False)
print(f"Web Logs data saved to {os.path.join(data_dir, 'web_logs.csv')}")


# --- 2. Transactional Systems Data ---
print("Generating Transactional Systems data...")
transaction_data = []
for user_id in user_ids:
    if user_id in churner_ids:
        last_activity_date = END_DATE - timedelta(days=CHURN_WINDOW_DAYS + np.random.randint(1, 90))
        activity_start = START_DATE
        activity_end = last_activity_date
    else:
        activity_start = END_DATE - timedelta(days=np.random.randint(1, 60))
        activity_end = END_DATE
    transaction_data.extend(generate_user_activity(user_id, activity_start, activity_end, np.random.randint(1, 5), 'transaction'))

transaction_df = pd.DataFrame(transaction_data, columns=['User ID', 'Transaction ID', 'Transaction Amount', 'Transaction Date', 'Product/Service Category'])
transaction_df.to_csv(os.path.join(data_dir, 'transactions.csv'), index=False)
print(f"Transactional data saved to {os.path.join(data_dir, 'transactions.csv')}")


# --- 3. Customer Support Interactions Data ---
print("Generating Customer Support Interactions data...")
support_data = []
for user_id in user_ids:
    if user_id in churner_ids:
        last_activity_date = END_DATE - timedelta(days=CHURN_WINDOW_DAYS + np.random.randint(1, 90))
        activity_start = START_DATE
        activity_end = last_activity_date
    else:
        activity_start = END_DATE - timedelta(days=np.random.randint(1, 60))
        activity_end = END_DATE
    support_data.extend(generate_user_activity(user_id, activity_start, activity_end, np.random.randint(0, 3), 'support')) # Some users might have no support interactions

support_df = pd.DataFrame(support_data, columns=['User ID', 'Support Ticket ID', 'Issue Type', 'Resolution Time', 'Customer Satisfaction Score', 'Support Ticket Date'])
support_df.to_csv(os.path.join(data_dir, 'support_interactions.csv'), index=False)
print(f"Customer Support data saved to {os.path.join(data_dir, 'support_interactions.csv')}")


# --- 4. Third-Party APIs (Social Media Engagement) Data ---
print("Generating Social Media Engagement data...")
social_data = []
for user_id in user_ids:
    if user_id in churner_ids:
        last_activity_date = END_DATE - timedelta(days=CHURN_WINDOW_DAYS + np.random.randint(1, 90))
        activity_start = START_DATE
        activity_end = last_activity_date
    else:
        activity_start = END_DATE - timedelta(days=np.random.randint(1, 60))
        activity_end = END_DATE
    social_data.extend(generate_user_activity(user_id, activity_start, activity_end, np.random.randint(1, 7), 'social'))

social_df = pd.DataFrame(social_data, columns=['User ID', 'Engagement Type', 'Engagement Date', 'Engagement Frequency'])
social_df.to_csv(os.path.join(data_dir, 'social_media_engagement.csv'), index=False)
print(f"Social Media Engagement data saved to {os.path.join(data_dir, 'social_media_engagement.csv')}")


# --- 5. Customer Demographics Data ---
print("Generating Customer Demographics data...")
demographics_data = []
for user_id in user_ids:
    age = np.random.randint(18, 70)
    gender = np.random.choice(['Male', 'Female', 'Non-binary'])
    # Use the limited set of locations for consistency with previous debugging
    location = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Miami', 'Atlanta', 'Boston', 'Raleigh', 'Denver'])
    account_age_days = (END_DATE - fake.date_time_between(start_date=START_DATE - timedelta(days=365*5), end_date=END_DATE)).days
    demographics_data.append([user_id, age, gender, location, account_age_days])

demographics_df = pd.DataFrame(demographics_data, columns=['User ID', 'Age', 'Gender', 'Location', 'Account Age'])
demographics_df.to_csv(os.path.join(data_dir, 'customer_demographics.csv'), index=False)
print(f"Customer Demographics data saved to {os.path.join(data_dir, 'customer_demographics.csv')}")

print("\n--- Data Generation Complete. Starting Data Loading and Inspection ---")

# --- Data Loading and Initial Inspection ---

# List of files and their date columns to parse
files_to_load = {
    'web_logs.csv': ['Timestamp'],
    'transactions.csv': ['Transaction Date'],
    'support_interactions.csv': ['Support Ticket Date'],
    'social_media_engagement.csv': ['Engagement Date'],
    'customer_demographics.csv': [] # No date columns to parse here
}

dataframes = {}

for filename, date_cols in files_to_load.items():
    filepath = os.path.join(data_dir, filename)
    try:
        if date_cols:
            df = pd.read_csv(filepath, parse_dates=date_cols)
        else:
            df = pd.read_csv(filepath)
        dataframes[filename.replace('.csv', '_df')] = df
        print(f"\n--- {filename} ---")
        print("df.head():")
        print(df.head())
        print("\ndf.info():")
        df.info()
        if date_cols:
            for col in date_cols:
                print(f"\n{col} Date Range: {df[col].min()} to {df[col].max()}")
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure the file is in the correct directory.")
    except Exception as e:
        print(f"An error occurred while processing {filename}: {e}")

# You can now access your DataFrames like:
# web_logs_df = dataframes['web_logs_df']
# transaction_df = dataframes['transactions_df']
# etc.
