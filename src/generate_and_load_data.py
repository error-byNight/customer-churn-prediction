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

# Generate User IDs
user_ids = [f'user_{i:04d}' for i in range(NUM_USERS)]

# --- 1. Web Logs Data ---
print("Generating Web Logs data...")
web_logs_data = []
for _ in range(NUM_USERS * 10): # More web logs per user
    user_id = np.random.choice(user_ids)
    timestamp = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)
    page_views = np.random.randint(1, 20)
    session_duration = np.random.randint(30, 1200) # seconds
    referrer_url = np.random.choice(['google.com', 'facebook.com', 'bing.com', 'direct', 'internal_page', 'twitter.com'])
    web_logs_data.append([user_id, timestamp, page_views, session_duration, referrer_url])

web_logs_df = pd.DataFrame(web_logs_data, columns=['User ID', 'Timestamp', 'Page Views', 'Session Duration', 'Referrer URL'])
web_logs_df.to_csv(os.path.join(data_dir, 'web_logs.csv'), index=False)
print(f"Web Logs data saved to {os.path.join(data_dir, 'web_logs.csv')}")


# --- 2. Transactional Systems Data ---
print("Generating Transactional Systems data...")
transaction_data = []
for _ in range(NUM_USERS * 5): # Fewer transactions than web logs
    user_id = np.random.choice(user_ids)
    transaction_date = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)
    transaction_amount = round(np.random.uniform(5, 5000), 2)
    product_category = np.random.choice(['Savings', 'Checking', 'Loans', 'Investments', 'Credit Card', 'Insurance'])
    transaction_data.append([user_id, fake.uuid4(), transaction_amount, transaction_date, product_category])

transaction_df = pd.DataFrame(transaction_data, columns=['User ID', 'Transaction ID', 'Transaction Amount', 'Transaction Date', 'Product/Service Category'])
transaction_df.to_csv(os.path.join(data_dir, 'transactions.csv'), index=False)
print(f"Transactional data saved to {os.path.join(data_dir, 'transactions.csv')}")


# --- 3. Customer Support Interactions Data ---
print("Generating Customer Support Interactions data...")
support_data = []
for _ in range(NUM_USERS * 2): # Even fewer support interactions
    user_id = np.random.choice(user_ids)
    support_ticket_date = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)
    issue_type = np.random.choice(['Billing', 'Technical', 'Account Access', 'Product Inquiry', 'Complaint', 'Feature Request'])
    resolution_time = np.random.randint(1, 72) # hours
    customer_satisfaction_score = np.random.randint(1, 6) # 1-5 scale
    support_data.append([user_id, fake.uuid4(), issue_type, resolution_time, customer_satisfaction_score, support_ticket_date])

support_df = pd.DataFrame(support_data, columns=['User ID', 'Support Ticket ID', 'Issue Type', 'Resolution Time', 'Customer Satisfaction Score', 'Support Ticket Date'])
support_df.to_csv(os.path.join(data_dir, 'support_interactions.csv'), index=False)
print(f"Customer Support data saved to {os.path.join(data_dir, 'support_interactions.csv')}")


# --- 4. Third-Party APIs (Social Media Engagement) Data ---
print("Generating Social Media Engagement data...")
social_data = []
for _ in range(NUM_USERS * 3): # Moderate social media engagement
    user_id = np.random.choice(user_ids)
    engagement_date = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)
    engagement_type = np.random.choice(['like', 'share', 'comment', 'post', 'mention'])
    engagement_frequency = np.random.randint(1, 10)
    social_data.append([user_id, engagement_type, engagement_date, engagement_frequency])

social_df = pd.DataFrame(social_data, columns=['User ID', 'Engagement Type', 'Engagement Date', 'Engagement Frequency'])
social_df.to_csv(os.path.join(data_dir, 'social_media_engagement.csv'), index=False)
print(f"Social Media Engagement data saved to {os.path.join(data_dir, 'social_media_engagement.csv')}")


# --- 5. Customer Demographics Data ---
print("Generating Customer Demographics data...")
demographics_data = []
for user_id in user_ids:
    age = np.random.randint(18, 70)
    gender = np.random.choice(['Male', 'Female', 'Non-binary'])
    location = fake.city()
    account_age_days = (END_DATE - fake.date_time_between(start_date=START_DATE - timedelta(days=365*5), end_date=END_DATE)).days # Account age can be older than data start
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
