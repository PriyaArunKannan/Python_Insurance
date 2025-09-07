import streamlit as st
import pandas as pd
import joblib
from preprocess import get_preprocessor  # For consistency, but model includes it
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('models/best_model.pkl')

# App Title
st.title("Insurance Premium Prediction Project")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Premium", "Project Details", "Dataset Overview", "Approach & Steps", "Results & Metrics"])

if page == "Home":
    st.header("Welcome to the Insurance Premium Predictor")
    st.write("""
    This app predicts insurance premiums based on customer characteristics and policy details.
    Use the 'Predict Premium' section for real-time quotes.
    Explore other sections for full project details.
    """)
    # st.image("https://via.placeholder.com/800x400?text=Insurance+Prediction+Dashboard", use_column_width=True)  # Replace with actual image if available

elif page == "Predict Premium":
    st.header("Real-Time Premium Prediction")
    st.write("Enter customer details below to get a predicted insurance premium.")
    
    # # Input fields based on dataset features
    # age = st.slider("Age", 18, 100, 30)
    # gender = st.selectbox("Gender", ["Male", "Female"])
    # annual_income = st.number_input("Annual Income", 10000, 1000000, 50000)
    # marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    # num_dependents = st.slider("Number of Dependents", 0, 10, 0)
    # education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    # occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
    # health_score = st.slider("Health Score", 0.0, 100.0, 50.0)
    # location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    # policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    # previous_claims = st.slider("Previous Claims", 0, 10, 0)
    # vehicle_age = st.slider("Vehicle Age", 0, 50, 5)
    # credit_score = st.slider("Credit Score", 300, 850, 700)
    # insurance_duration = st.slider("Insurance Duration (Years)", 1, 50, 1)
    # smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
    # exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    # property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])
    # policy_start_year = st.number_input("Policy Start Year", 2000, 2025, 2023)  # Derived feature
    
    # # Create input DataFrame
    # input_data = pd.DataFrame({
    #     'Age': [age],
    #     'Gender': [gender],
    #     'Annual Income': [annual_income],
    #     'Marital Status': [marital_status],
    #     'Number of Dependents': [num_dependents],
    #     'Education Level': [education_level],
    #     'Occupation': [occupation],
    #     'Health Score': [health_score],
    #     'Location': [location],
    #     'Policy Type': [policy_type],
    #     'Previous Claims': [previous_claims],
    #     'Vehicle Age': [vehicle_age],
    #     'Credit Score': [credit_score],
    #     'Insurance Duration': [insurance_duration],
    #     'Smoking Status': [smoking_status],
    #     'Exercise Frequency': [exercise_frequency],
    #     'Property Type': [property_type],
    #     'Policy Start Year': [policy_start_year]
    # })
    
    # if st.button("Predict Premium"):
    #     prediction = model.predict(input_data)[0]
    #     st.success(f"Predicted Insurance Premium: ${prediction:.2f}")

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

elif page == "Project Details":
    st.header("Problem Statement")
    st.write("""
    Insurance companies use various factors such as age, income, health status, and claim history to estimate premiums for customers. 
    The goal of this project is to build a machine learning model that accurately predicts insurance premiums based on customer characteristics and policy details.
    """)
    
    st.subheader("Business Use Cases")
    st.write("""
    💰 Insurance Companies: Optimize premium pricing based on risk factors.
    📊 Financial Institutions: Assess risk for loan approvals tied to insurance policies.
    🧑‍⚕️ Healthcare Providers: Estimate future healthcare costs for patients.
    🔍 Customer Service Optimization: Provide real-time insurance quotes based on data-driven predictions.
    """)

elif page == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("""
    This dataset contains 2Lk+ and 20 features with a mix of categorical, numerical, and text data. It includes missing values, incorrect data types, and skewed distributions.
    Target Variable: Premium Amount.
    """)
    
    st.subheader("Features")
    features = {
        "Age": "Numerical",
        "Gender": "Categorical: Male, Female",
        "Annual Income": "Numerical, skewed",
        "Marital Status": "Categorical: Single, Married, Divorced",
        "Number of Dependents": "Numerical, with missing values",
        "Education Level": "Categorical: High School, Bachelor's, Master's, PhD",
        "Occupation": "Categorical: Employed, Self-Employed, Unemployed",
        "Health Score": "Numerical, skewed",
        "Location": "Categorical: Urban, Suburban, Rural",
        "Policy Type": "Categorical: Basic, Comprehensive, Premium",
        "Previous Claims": "Numerical, with outliers",
        "Vehicle Age": "Numerical",
        "Credit Score": "Numerical, with missing values",
        "Insurance Duration": "Numerical, in years",
        "Premium Amount": "Target: Numerical, skewed",
        "Policy Start Date": "Text, improperly formatted",
        "Customer Feedback": "Text",
        "Smoking Status": "Categorical: Yes, No",
        "Exercise Frequency": "Categorical: Daily, Weekly, Monthly, Rarely",
        "Property Type": "Categorical: House, Apartment, Condo"
    }
    st.table(pd.DataFrame.from_dict(features, orient='index', columns=["Description"]))
    
    st.subheader("Data Characteristics")
    st.write("""
    - Missing Values: In features like Credit Score.
    - Incorrect Data Types: E.g., dates as text.
    - Skewed Distributions: E.g., Annual Income, Premium Amount.
    """)

elif page == "Approach & Steps":
    st.header("Approach & Steps")
    st.subheader("Step 1: Understanding the Data")
    st.write("Load dataset, perform EDA (distributions, correlations, visualizations).")
    
    st.subheader("Step 2: Data Preprocessing")
    st.write("Handle missing values, encode categoricals, scale features, split data (80/20).")
    
    st.subheader("Step 3: Model Development")
    st.write("Train regression models: Linear Regression, Decision Tree, Random Forest, XGBoost. Evaluate with RMSE, MAE, R².")
    
    st.subheader("Step 4: ML Pipeline & MLflow Integration")
    st.write("Automate workflow with pipelines. Track experiments with MLflow (log metrics, models).")
    
    st.subheader("Step 5: Model Deployment with Streamlit")
    st.write("Deploy as this web app for real-time predictions.")

elif page == "Results & Metrics":
    st.header("Results & Evaluation Metrics")
    st.write("""
    📊 Aim for low error rates in predictions.
    ✅ Deployed Streamlit app for real-time estimates.
    """)
    
    st.subheader("Metrics")
    st.write("""
    - Root Mean Squared Logarithmic Error (RMSLE)
    - Root Mean Squared Error (RMSE): Measures prediction accuracy.
    - R² Score: Explains variance in premiums.
    - Mean Absolute Error (MAE): Average prediction error.
    """)
    
    st.subheader("Technical Tags")
    st.write("🔹 Python, Pandas, NumPy, Scikit-Learn, XGBoost 🔹 MLflow, Streamlit, Git/GitHub 🔹 Data Preprocessing, Feature Engineering 🔹 Model Deployment")

# Footer
st.markdown("---")
st.write("Built with Streamlit. For source code and dataset, check the project repo.")