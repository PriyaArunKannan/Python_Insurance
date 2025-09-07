import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import random
from utils.preprocessing import create_preprocessor
from utils.models import train_and_evaluate_models
from utils.evaluation import save_model

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Annual Income': np.random.lognormal(mean=10, sigma=1, size=n_samples).astype(int),
        'Marital Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'Number of Dependents': np.random.choice([0, 1, 2, 3, 4, np.nan], n_samples, p=[0.4, 0.2, 0.2, 0.1, 0.05, 0.05]),
        'Education Level': np.random.choice(['High School', "Bachelor's", "Master's", 'PhD'], n_samples),
        'Occupation': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples),
        'Health Score': np.random.lognormal(mean=4, sigma=0.5, size=n_samples),
        'Location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        'Policy Type': np.random.choice(['Basic', 'Comprehensive', 'Premium'], n_samples),
        'Previous Claims': np.random.poisson(lam=1, size=n_samples) + np.random.choice([0, 0, 0, 10, 20], n_samples, p=[0.8, 0.05, 0.05, 0.05, 0.05]),
        'Vehicle Age': np.random.randint(0, 20, n_samples),
        'Credit Score': np.random.choice(np.arange(300, 850, 10), n_samples).astype(float),
        'Insurance Duration': np.random.uniform(1, 10, n_samples),
        'Policy Start Date': [datetime.date.today() - datetime.timedelta(days=random.randint(0, 365*5)) for _ in range(n_samples)],
        'Customer Feedback': np.random.choice(['Good service', 'Too expensive', 'Quick claim', 'Poor support', ''], n_samples),
        'Smoking Status': np.random.choice(['Yes', 'No'], n_samples),
        'Exercise Frequency': np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely'], n_samples),
        'Property Type': np.random.choice(['House', 'Apartment', 'Condo'], n_samples),
    }
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'Credit Score'] = np.nan
    df['Premium Amount'] = (
        100 + df['Age'] * 10 + df['Annual Income'] * 0.001 - df['Health Score'] * 5 + df['Previous Claims'] * 50 +
        df['Vehicle Age'] * 20 - df['Credit Score'].fillna(600) * 0.5 + df['Insurance Duration'] * 30 + np.random.normal(0, 100, n_samples)
    )
    df['Premium Amount'] = np.abs(df['Premium Amount'])
    df['Premium Amount'] = np.log1p(df['Premium Amount']) * 100
    df.to_csv('data/synthetic_insurance_data.csv', index=False)
    return df

def main():
    # Load or generate data
    # df = pd.read_csv('data/test.csv')  # Replace with actual path
    df = generate_synthetic_data()

    # Drop text features
    df = df.drop(['Policy Start Date', 'Customer Feedback'], axis=1)

    # Split data
    X = df.drop('Premium Amount', axis=1)
    y = df['Premium Amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create preprocessor
    numerical_features = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration']
    categorical_features = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location', 'Policy Type', 'Smoking Status', 'Exercise Frequency', 'Property Type']
    preprocessor = create_preprocessor(numerical_features, categorical_features)

    # Train and evaluate models
    best_model = train_and_evaluate_models(preprocessor, X_train, y_train, X_test, y_test)

    # Save the best model
    save_model(best_model, 'models/best_model.pkl')

if __name__ == '__main__':
    main()