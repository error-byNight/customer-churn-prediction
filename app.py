import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib # For loading models and preprocessor
from datetime import datetime, timedelta
# Removed direct imports for StandardScaler, OneHotEncoder, ColumnTransformer
# as they will be loaded from the saved preprocessor object

# --- Configuration (matching previous scripts) ---
# Ensure these match the values used during data generation and feature engineering
END_DATE = datetime(2024, 7, 31)
CHURN_WINDOW_DAYS = 60

# Define directories
processed_data_dir = 'data/processed'
models_dir = 'models'

# --- Load necessary components ---
# Load the *fitted* preprocessor (ColumnTransformer)
try:
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
    st.sidebar.success("Preprocessor Loaded Successfully!")
    
    # Load the original feature names that the preprocessor expects as input
    original_feature_names_for_preprocessor = joblib.load(os.path.join(models_dir, 'original_feature_names.pkl'))
    print(f"Loaded original feature names for preprocessor. Example: {original_feature_names_for_preprocessor[:5]}...")

except FileNotFoundError as e:
    st.error(f"Error loading preprocessor or original feature names: {e}. Ensure '02_feature_engineering.py' was run and saved these files.")
    st.stop()
except Exception as e:
    st.error(f"Error loading preprocessor: {e}")
    st.stop()


# Load the trained model (e.g., XGBoost)
try:
    model = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
    st.sidebar.success("XGBoost Model Loaded Successfully!")
except FileNotFoundError as e:
    st.sidebar.error(f"Error loading model: {e}. Ensure '04_model_training_evaluation.py' was run and the model was saved.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# --- Streamlit App UI ---
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction Dashboard")
st.markdown("Predict customer churn based on their demographic and activity data.")

# Sidebar for user input or selecting existing user
st.sidebar.header("Customer Data Input")
input_option = st.sidebar.radio("Choose input method:", ("Enter New Data", "Select Existing User"))

# Placeholder for existing users (from consolidated_churn_data_final.csv for demonstration)
try:
    existing_users_df = pd.read_csv(os.path.join(processed_data_dir, 'consolidated_churn_data_final.csv'))
    existing_user_ids = existing_users_df['User ID'].unique().tolist()
except FileNotFoundError:
    st.sidebar.warning("Existing user data not found. Please run the full pipeline first.")
    existing_user_ids = []


# Function to collect user input for a NEW customer
def get_new_user_input_features():
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age", 18, 70, 30)
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Non-binary'])
    location = st.sidebar.selectbox("Location", ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Miami', 'Atlanta', 'Boston', 'Raleigh', 'Denver'])
    account_age = st.sidebar.number_input("Account Age (days)", min_value=1, value=365)

    st.sidebar.subheader("Activity Metrics (Simulated for New Input)")
    st.sidebar.info("These values mimic engineered features. In a real application, they'd be derived from raw event data.")

    # These inputs directly correspond to the engineered features
    web_log_count_last_7d = st.sidebar.number_input("Web Log Count (last 7 days)", min_value=0, value=5)
    web_log_avg_page_views_last_7d = st.sidebar.number_input("Avg Page Views (last 7 days)", min_value=0.0, value=5.0)
    web_log_avg_session_duration_last_7d = st.sidebar.number_input("Avg Session Duration (last 7 days)", min_value=0.0, value=300.0)
    web_log_count_last_30d = st.sidebar.number_input("Web Log Count (last 30 days)", min_value=0, value=20)
    web_log_avg_page_views_last_30d = st.sidebar.number_input("Avg Page Views (last 30 days)", min_value=0.0, value=7.0)
    web_log_avg_session_duration_last_30d = st.sidebar.number_input("Avg Session Duration (last 30 days)", min_value=0.0, value=400.0)
    web_log_count_last_90d = st.sidebar.number_input("Web Log Count (last 90 days)", min_value=0, value=50)
    web_log_avg_page_views_last_90d = st.sidebar.number_input("Avg Page Views (last 90 days)", min_value=0.0, value=8.0)
    web_log_avg_session_duration_last_90d = st.sidebar.number_input("Avg Session Duration (last 90 days)", min_value=0.0, value=500.0)

    transaction_count_last_7d = st.sidebar.number_input("Transaction Count (last 7 days)", min_value=0, value=1)
    transaction_total_amount_last_7d = st.sidebar.number_input("Total Transaction Amount (last 7 days)", min_value=0.0, value=100.0)
    transaction_avg_amount_last_7d = st.sidebar.number_input("Avg Transaction Amount (last 7 days)", min_value=0.0, value=100.0)
    transaction_count_last_30d = st.sidebar.number_input("Transaction Count (last 30 days)", min_value=0, value=3)
    transaction_total_amount_last_30d = st.sidebar.number_input("Total Transaction Amount (last 30 days)", min_value=0.0, value=500.0)
    transaction_avg_amount_last_30d = st.sidebar.number_input("Avg Transaction Amount (last 30 days)", min_value=0.0, value=160.0)
    transaction_count_last_90d = st.sidebar.number_input("Transaction Count (last 90 days)", min_value=0, value=8)
    transaction_total_amount_last_90d = st.sidebar.number_input("Total Transaction Amount (last 90 days)", min_value=0.0, value=1500.0)
    transaction_avg_amount_last_90d = st.sidebar.number_input("Avg Transaction Amount (last 90 days)", min_value=0.0, value=180.0)

    support_count_last_7d = st.sidebar.number_input("Support Ticket Count (last 7 days)", min_value=0, value=0)
    support_avg_resolution_time_last_7d = st.sidebar.number_input("Avg Support Resolution Time (last 7 days)", min_value=0.0, value=0.0)
    support_avg_satisfaction_score_last_7d = st.sidebar.number_input("Avg Support Satisfaction (last 7 days)", min_value=0.0, value=0.0)
    support_count_last_30d = st.sidebar.number_input("Support Ticket Count (last 30 days)", min_value=0, value=1)
    support_avg_resolution_time_last_30d = st.sidebar.number_input("Avg Support Resolution Time (last 30 days)", min_value=0.0, value=24.0)
    support_avg_satisfaction_score_last_30d = st.sidebar.number_input("Avg Support Satisfaction (last 30 days)", min_value=0.0, value=4.0)
    support_count_last_90d = st.sidebar.number_input("Support Ticket Count (last 90 days)", min_value=0, value=2)
    support_avg_resolution_time_last_90d = st.sidebar.number_input("Avg Support Resolution Time (last 90 days)", min_value=0.0, value=30.0)
    support_avg_satisfaction_score_last_90d = st.sidebar.number_input("Avg Support Satisfaction (last 90 days)", min_value=0.0, value=3.5)

    social_count_last_7d = st.sidebar.number_input("Social Count (last 7 days)", min_value=0, value=1)
    social_total_engagement_freq_last_7d = st.sidebar.number_input("Total Social Engagement (last 7 days)", min_value=0.0, value=5.0)
    social_count_last_30d = st.sidebar.number_input("Social Count (last 30 days)", min_value=0, value=5)
    social_total_engagement_freq_last_30d = st.sidebar.number_input("Total Social Engagement (last 30 days)", min_value=0.0, value=20.0)
    social_count_last_90d = st.sidebar.number_input("Social Count (last 90 days)", min_value=0, value=15)
    social_total_engagement_freq_last_90d = st.sidebar.number_input("Total Social Engagement (last 90 days)", min_value=0.0, value=60.0)

    # Days Since Last Activity (crucial churn indicator)
    days_since_last_activity = st.sidebar.number_input("Days Since Last Activity (Overall)", min_value=0, value=30)


    # Create a dictionary with all input features.
    # The order of columns in this dictionary will be used to create the DataFrame.
    # It must match the order of `original_feature_names_for_preprocessor`.
    data = {
        'Age': age,
        'Gender': gender,
        'Location': location,
        'Account Age': account_age,
        'Days Since Last Activity': days_since_last_activity,
        'web_log_count_last_7d': web_log_count_last_7d,
        'web_log_avg_page_views_last_7d': web_log_avg_page_views_last_7d,
        'web_log_avg_session_duration_last_7d': web_log_avg_session_duration_last_7d,
        'web_log_count_last_30d': web_log_count_last_30d,
        'web_log_avg_page_views_last_30d': web_log_avg_page_views_last_30d,
        'web_log_avg_session_duration_last_30d': web_log_avg_session_duration_last_30d,
        'web_log_count_last_90d': web_log_count_last_90d,
        'web_log_avg_page_views_last_90d': web_log_avg_page_views_last_90d,
        'web_log_avg_session_duration_last_90d': web_log_avg_session_duration_last_90d,
        'transaction_count_last_7d': transaction_count_last_7d,
        'transaction_total_amount_last_7d': transaction_total_amount_last_7d,
        'transaction_avg_amount_last_7d': transaction_avg_amount_last_7d,
        'transaction_count_last_30d': transaction_count_last_30d,
        'transaction_total_amount_last_30d': transaction_total_amount_last_30d,
        'transaction_avg_amount_last_30d': transaction_avg_amount_last_30d,
        'transaction_count_last_90d': transaction_count_last_90d,
        'transaction_total_amount_last_90d': transaction_total_amount_last_90d,
        'transaction_avg_amount_last_90d': transaction_avg_amount_last_90d,
        'support_count_last_7d': support_count_last_7d,
        'support_avg_resolution_time_last_7d': support_avg_resolution_time_last_7d,
        'support_avg_satisfaction_score_last_7d': support_avg_satisfaction_score_last_7d,
        'support_count_last_30d': support_count_last_30d,
        'support_avg_resolution_time_last_30d': support_avg_resolution_time_last_30d,
        'support_avg_satisfaction_score_last_30d': support_avg_satisfaction_score_last_30d,
        'support_count_last_90d': support_count_last_90d,
        'support_avg_resolution_time_last_90d': support_avg_resolution_time_last_90d,
        'support_avg_satisfaction_score_last_90d': support_avg_satisfaction_score_last_90d,
        'social_count_last_7d': social_count_last_7d,
        'social_total_engagement_freq_last_7d': social_total_engagement_freq_last_7d,
        'social_count_last_30d': social_count_last_30d,
        'social_total_engagement_freq_last_30d': social_total_engagement_freq_last_30d,
        'social_count_last_90d': social_count_last_90d,
        'social_total_engagement_freq_last_90d': social_total_engagement_freq_last_90d
    }

    # Create DataFrame, ensuring column order matches original_feature_names_for_preprocessor
    # This is crucial for the preprocessor to work correctly.
    features = pd.DataFrame([data], columns=original_feature_names_for_preprocessor)
    return features


# Get input based on selection
if input_option == "Enter New Data":
    input_df = get_new_user_input_features()
    st.sidebar.markdown("---")
    st.sidebar.subheader("Raw Input Data Preview")
    st.sidebar.write(input_df)
    st.sidebar.markdown("---")
    if st.sidebar.button("Predict Churn"):
        # Preprocess the input data
        processed_input = preprocessor.transform(input_df)
        processed_input_df = pd.DataFrame(processed_input, columns=preprocessor.get_feature_names_out())

        # Make prediction
        prediction = model.predict(processed_input_df)[0]
        prediction_proba = model.predict_proba(processed_input_df)[:, 1][0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"**This customer is predicted to CHURN!**")
            st.write(f"Confidence (Probability of Churn): **{prediction_proba:.2f}**")
            st.warning("Immediate retention strategies are recommended.")
        else:
            st.success(f"**This customer is predicted to NOT CHURN.**")
            st.write(f"Confidence (Probability of Churn): **{prediction_proba:.2f}**")
            st.info("Continue monitoring customer engagement.")

        st.subheader("What led to this prediction?")
        st.info("For a detailed explanation of this specific prediction, you would use SHAP values (best viewed in a Jupyter Notebook).")
        st.write("Key factors often include recency of activity, transaction frequency, and support interactions.")

elif input_option == "Select Existing User":
    if existing_user_ids:
        selected_user_id = st.sidebar.selectbox("Select User ID", existing_user_ids)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Selected User's Data Preview")
        selected_user_data = existing_users_df[existing_users_df['User ID'] == selected_user_id].iloc[0]
        
        # Prepare data for prediction (drop User ID and Churn from features)
        # Ensure the columns match the original input features expected by the preprocessor
        user_features_raw = selected_user_data.drop(['User ID', 'Churn']).to_frame().T
        user_features = user_features_raw.reindex(columns=original_feature_names_for_preprocessor, fill_value=0)

        st.sidebar.write(user_features) # Show the raw features being passed to preprocessor
        st.sidebar.markdown("---")

        if st.sidebar.button("Predict Churn for Selected User"):
            # Preprocess the selected user's data using the fitted preprocessor
            processed_user_features = preprocessor.transform(user_features)
            processed_user_features_df = pd.DataFrame(processed_user_features, columns=preprocessor.get_feature_names_out())

            # Make prediction
            prediction = model.predict(processed_user_features_df)[0]
            prediction_proba = model.predict_proba(processed_user_features_df)[:, 1][0]

            st.subheader(f"Prediction Result for User: {selected_user_id}")
            if prediction == 1:
                st.error(f"**This customer is predicted to CHURN!**")
                st.write(f"Confidence (Probability of Churn): **{prediction_proba:.2f}**")
                st.warning("Immediate retention strategies are recommended.")
            else:
                st.success(f"**This customer is predicted to NOT CHURN.**")
                st.write(f"Confidence (Probability of Churn): **{prediction_proba:.2f}**")
                st.info("Continue monitoring customer engagement.")

            st.subheader("Actual Churn Status (for selected existing user):")
            actual_churn = selected_user_data['Churn']
            if actual_churn == 1:
                st.write(f"Actual Status: **CHURNED** (Label: 1)")
            else:
                st.write(f"Actual Status: **NOT CHURNED** (Label: 0)")

            st.subheader("What led to this prediction?")
            st.info("For a detailed explanation of this specific prediction, you would use SHAP values (best viewed in a Jupyter Notebook).")
            st.write("Key factors often include recency of activity, transaction frequency, and support interactions.")
    else:
        st.warning("No existing user data available. Please run the full pipeline to generate data.")


st.markdown("---")
st.subheader("About This Dashboard")
st.write("""
This dashboard demonstrates a customer churn prediction model.
- **Data Source:** Synthetic data mimicking web logs, transactional systems, customer support interactions, social media engagement, and customer demographics.
- **Pipeline:** Data is consolidated, features are engineered (recency, frequency, monetary, engagement), and preprocessed (scaling, one-hot encoding).
- **Model:** An XGBoost Classifier trained on the processed data with SMOTE for handling class imbalance.
- **Metrics:** The model aims to optimize for Precision, Recall, F1-Score, and ROC-AUC, especially for the minority churn class.
""")

st.markdown("---")
st.write("Developed as part of a predictive analytics pipeline project.")

