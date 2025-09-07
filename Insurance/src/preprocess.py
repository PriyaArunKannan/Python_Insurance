import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(df):
    # Numerical: median impute
    num_cols = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 
                'Vehicle Age', 'Credit Score', 'Insurance Duration']
    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical: mode impute
    cat_cols = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location', 
                'Policy Type', 'Smoking Status', 'Exercise Frequency', 'Property Type']
    for col in cat_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def feature_engineering(df):
    # Parse dates
    if 'Policy Start Date' in df.columns:
        df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
        df['Policy Start Year'] = df['Policy Start Date'].dt.year
        df.drop('Policy Start Date', axis=1, inplace=True)
    
    # Drop irrelevant text if not used (e.g., Customer Feedback)
    if 'Customer Feedback' in df.columns:
        df.drop('Customer Feedback', axis=1, inplace=True)
    
    return df

def get_preprocessor():
    num_cols = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score', 'Previous Claims', 
                'Vehicle Age', 'Credit Score', 'Insurance Duration', 'Policy Start Year']
    
    cat_onehot_cols = ['Gender', 'Marital Status', 'Occupation', 'Location', 'Policy Type', 
                       'Smoking Status', 'Exercise Frequency', 'Property Type']
    
    cat_ordinal_cols = ['Education Level']  # Ordinal: High School < Bachelor's < Master's < PhD
    
    numerical_transformer = StandardScaler()
    categorical_onehot_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    categorical_ordinal_transformer = OrdinalEncoder(categories=[['High School', "Bachelor's", "Master's", 'PhD']])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat_onehot', categorical_onehot_transformer, cat_onehot_cols),
            ('cat_ordinal', categorical_ordinal_transformer, cat_ordinal_cols)
        ],
        remainder='passthrough'  # Keep other columns if any
    )
    
    return preprocessor