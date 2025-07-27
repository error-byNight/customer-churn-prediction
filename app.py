import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib # For loading models and preprocessor
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Configuration (matching previous scripts) ---
# Ensure these match the values used during data generation and feature engineering
END_DATE = datetime(2024, 7, 31)
CHURN_WINDOW_DAYS = 60

# Define directories
processed_data_dir = 'data/processed'
models_dir = 'models'

# --- Load necessary components ---
# IMPORTANT: In a real production pipeline, you would save the *fitted* preprocessor
# from 02_feature_engineering.py and load it here.
# For this demonstration, we'll re-initialize and fit a dummy preprocessor
# based on the structure of the final processed data.
# This assumes the column order and categories are consistent.

# Load a sample of the processed data to infer columns for the preprocessor
try:
    sample_df_for_cols = pd.read_csv(os.path.join(processed_data_dir, 'consolidated_churn_data_final.csv'))
    # Drop User ID and Churn from this sample to get feature columns for preprocessor
    X_sample = sample_df_for_cols.drop(columns=['User ID', 'Churn'])

    # Re-identify columns for preprocessor based on the sample data
    categorical_cols = X_sample.select_dtypes(include='object').columns.tolist()
    numerical_cols = X_sample.select_dtypes(include=np.number).columns.tolist()
    cols_to_scale = [col for col in numerical_cols if col not in ['Account Age', 'Days Since Last Activity']]
    cols_to_passthrough_numerical = [col for col in numerical_cols if col in ['Account Age', 'Days Since Last Activity']]

    # Re-create the ColumnTransformer structure
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cols_to_scale),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('passthrough_num', 'passthrough', cols_to_passthrough_numerical)
        ]
    )
    # Fit the preprocessor on the sample data.
    # This is a simplification for the app demo. In production, load the saved, fitted preprocessor.
    preprocessor.fit(X_sample)
    st.sidebar.success("Preprocessor initialized successfully (for demonstration).")

except FileNotFoundError as e:
    st.error(f"Error loading sample data for preprocessor setup: {e}. Ensure '02_feature_engineering.py' was run.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing preprocessor: {e}")
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


    # Create a dictionary with all 54 features, ensuring column names match X_sample.columns
    # Fill in placeholder values for features not explicitly exposed in UI
    data = {col: 0 for col in X_sample.columns} # Initialize with zeros
    data.update({
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
    })

    # Create DataFrame, ensuring column order matches X_sample.columns
    features = pd.DataFrame([data], columns=X_sample.columns)
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
        st.sidebar.write(selected_user_data.drop('Churn')) # Hide churn label from input preview
        st.sidebar.markdown("---")
        if st.sidebar.button("Predict Churn for Selected User"):
            # Prepare data for prediction (drop User ID and Churn from features)
            user_features = selected_user_data.drop(['User ID', 'Churn']).to_frame().T
            
            # Ensure the order of columns matches the training data by reindexing
            user_features = user_features.reindex(columns=X_sample.columns, fill_value=0)

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

# st.markdown("---")
# st.write("Developed as part of a predictive analytics pipeline project.")

