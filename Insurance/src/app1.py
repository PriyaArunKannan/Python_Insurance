import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/best_model.pkl')

st.title('Insurance Premium Predictor')

# Input fields
age = st.number_input('Age', 18, 80, 30)
gender = st.selectbox('Gender', ['Male', 'Female'])
annual_income = st.number_input('Annual Income', 10000, 1000000, 50000)
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
num_dependents = st.number_input('Number of Dependents', 0, 5, 0)
education_level = st.selectbox('Education Level', ['High School', "Bachelor's", "Master's", 'PhD'])
occupation = st.selectbox('Occupation', ['Employed', 'Self-Employed', 'Unemployed'])
health_score = st.number_input('Health Score', 0.0, 100.0, 50.0)
location = st.selectbox('Location', ['Urban', 'Suburban', 'Rural'])
policy_type = st.selectbox('Policy Type', ['Basic', 'Comprehensive', 'Premium'])
previous_claims = st.number_input('Previous Claims', 0, 10, 0)
vehicle_age = st.number_input('Vehicle Age', 0, 20, 5)
credit_score = st.number_input('Credit Score', 300, 850, 600)
insurance_duration = st.number_input('Insurance Duration (years)', 1.0, 10.0, 1.0)
smoking_status = st.selectbox('Smoking Status', ['Yes', 'No'])
exercise_frequency = st.selectbox('Exercise Frequency', ['Daily', 'Weekly', 'Monthly', 'Rarely'])
property_type = st.selectbox('Property Type', ['House', 'Apartment', 'Condo'])

if st.button('Predict Premium'):
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
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Premium Amount: ${prediction:.2f}')