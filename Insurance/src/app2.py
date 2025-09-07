import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stButton>button:hover {background-color: #45a049;}
    .error {color: #ff4d4d; font-weight: bold;}
    .success {color: #4CAF50; font-weight: bold;}
    .sidebar .sidebar-content {background-color: #f0f2f6;}
    .stTextInput>input, .stNumberInput>input, .stSelectbox>select {
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stTitle {color: #2c3e50; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state for theme and form inputs
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'
if 'form_cleared' not in st.session_state:
    st.session_state.form_cleared = False

# Sidebar for navigation and settings
st.sidebar.title("Insurance Premium Predictor Settings")
st.sidebar.subheader("App Configuration")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=["Light", "Dark"].index(st.session_state.theme))

# Apply theme
if theme == "Dark":
    st.markdown("""
        <style>
        .main {background-color: #2c3e50; color: #ecf0f1;}
        .stButton>button {background-color: #3498db;}
        .stButton>button:hover {background-color: #2980b9;}
        .sidebar .sidebar-content {background-color: #34495e;}
        </style>
    """, unsafe_allow_html=True)
st.session_state.theme = theme

# Load model
model_path = r'd:\Project\Guvi\Project3\Insurance\models\best_model.pkl'
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please run main.py to train and save the model.")
    st.stop()

# Main app title
st.title("Insurance Premium Predictor")
st.markdown("Enter customer details to predict the insurance premium. All fields are required.")

# Input form
with st.form(key="prediction_form"):
    st.subheader("Customer and Policy Details")
    
    # Organize inputs in two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=50000, step=1000)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0)
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, value=0, step=1)
        education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"], index=0)
        occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"], index=0)
        health_score = st.number_input("Health Score", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    
    with col2:
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"], index=0)
        policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"], index=0)
        previous_claims = st.number_input("Previous Claims", min_value=0, max_value=10, value=0, step=1)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=20, value=5, step=1)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600, step=10)
        insurance_duration = st.number_input("Insurance Duration (years)", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
        smoking_status = st.selectbox("Smoking Status", ["Yes", "No"], index=1)
        exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"], index=2)
        property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"], index=0)
    
    # Form buttons
    col_submit, col_reset = st.columns([1, 1])
    with col_submit:
        submit_button = st.form_submit_button("Predict Premium")
    with col_reset:
        reset_button = st.form_submit_button("Reset Inputs")

# Reset form inputs
if reset_button or st.session_state.form_cleared:
    st.session_state.form_cleared = True
    st.experimental_rerun()

# Prediction logic
if submit_button:
    # Validate inputs
    if any([age is None, annual_income is None, health_score is None, previous_claims is None,
            vehicle_age is None, credit_score is None, insurance_duration is None]):
        st.markdown('<p class="error">All fields are required. Please fill in all inputs.</p>', unsafe_allow_html=True)
    else:
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Annual Income': [annual_income],
            'Marital Status': [marital_status],
            'Number of Dependents': [num_dependents],
            'Education Level': [education_level],
            'Occupation': [occupation],
            'Health Score': [health_score],
            'Location': [location],
            'Policy Type': [policy_type],
            'Previous Claims': [previous_claims],
            'Vehicle Age': [vehicle_age],
            'Credit Score': [credit_score],
            'Insurance Duration': [insurance_duration],
            'Smoking Status': [smoking_status],
            'Exercise Frequency': [exercise_frequency],
            'Property Type': [property_type]
        })
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            st.markdown(f'<p class="success">Predicted Premium Amount: ${prediction:.2f}</p>', unsafe_allow_html=True)
            
            # Optional: Show input data for debugging
            if st.sidebar.checkbox("Show Input Data"):
                st.subheader("Input Data")
                st.write(input_data)
            
            # Feature importance (if model supports it, e.g., Random Forest or XGBoost)
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                st.subheader("Feature Importance")
                feature_names = (
                    model.named_steps['preprocessor']
                    .transformers_[0][1]
                    .named_steps['scaler']
                    .get_feature_names_out(input_features=['Age', 'Annual Income', 'Number of Dependents', 
                                                          'Health Score', 'Previous Claims', 'Vehicle Age', 
                                                          'Credit Score', 'Insurance Duration'])
                    .tolist() +
                    model.named_steps['preprocessor']
                    .transformers_[1][1]
                    .named_steps['onehot']
                    .get_feature_names_out(input_features=['Gender', 'Marital Status', 'Education Level', 
                                                          'Occupation', 'Location', 'Policy Type', 
                                                          'Smoking Status', 'Exercise Frequency', 
                                                          'Property Type'])
                    .tolist()
                )
                importances = model.named_steps['model'].feature_importances_
                
                # Plot feature importance
                fig, ax = plt.subplots()
                sns.barplot(x=importances, y=feature_names, ax=ax)
                ax.set_title("Feature Importance")
                ax.set_xlabel("Importance")
                st.pyplot(fig)
                
        except Exception as e:
            st.markdown(f'<p class="error">Error during prediction: {str(e)}</p>', unsafe_allow_html=True)

# Additional options in sidebar
st.sidebar.subheader("Additional Options")
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.markdown('<p class="success">Cache cleared successfully!</p>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name] | Powered by Streamlit")