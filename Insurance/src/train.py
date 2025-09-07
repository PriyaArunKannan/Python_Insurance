import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib
from preprocess import load_data, handle_missing_values, feature_engineering, get_preprocessor

# Load and prepare data
df = load_data('data/train.csv')
df = handle_missing_values(df)
df = feature_engineering(df)

X = df.drop('Premium Amount', axis=1)
y = df['Premium Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = get_preprocessor()

# Models to train
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    # 'Random Forest': RandomForestRegressor(random_state=42),
    # 'XGBoost': xgb.XGBRegressor(random_state=42)
}

mlflow.set_experiment("Insurance_Premium_Prediction")

best_model = None
best_rmse = float('inf')

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))  # RMSLE metric
        
        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmsle", rmsle)
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}, RMSLE={rmsle:.2f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline

# Save best model
joblib.dump(best_model, 'models/best_model.pkl')
print("Training complete. Best model saved.")