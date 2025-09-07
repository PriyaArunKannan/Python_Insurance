# Insurance Premium Prediction

This project builds a machine learning model to predict insurance premiums based on customer characteristics (e.g., age, income, health status) and policy details (e.g., policy type, region). It includes exploratory data analysis (EDA), data preprocessing, model training, MLflow experiment tracking, and a Streamlit web app for real-time predictions. The project is designed for use cases in insurance pricing, financial risk assessment, and healthcare cost estimation.

## Project Structure

```
insurance-premium-prediction/
├── data/
│   └── train.csv  # Place downloaded dataset here
├── models/
│   └── best_model.pkl        # Trained model (generated after training)
├── src/
│   ├── preprocess.py         # Data preprocessing functions
│   ├── train.py              # Model training and MLflow logging
│   └── utils.py              # Utility functions (e.g., visualization)
├── app.py                    # Streamlit web app for predictions
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── .gitignore                # Ignore pycache, .env, etc.
```

## Dataset
- **Source**: Download from [provided link](#) (replace with actual dataset link).
- **Format**: CSV with 200,000+ rows and 20 features (numerical, categorical, text).
- **Features**: Age, Gender, Annual Income, Health Score, Policy Type, etc.
- **Target**: Premium Amount (numerical, skewed).
- **Characteristics**: Contains missing values, incorrect data types, and skewed distributions to mimic real-world challenges.

## Prerequisites
- Python 3.8+
- Git
- MLflow (for experiment tracking)
- Streamlit (for web app)

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PriyaArunKannan/Python_Insurance.git
   cd insurance
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   - Download the dataset from the provided link and extract `train.csv` to the `data/` folder.

4. **Run EDA and Training** (optional, for experimentation):
   - Alternatively, run:
     ```bash
     python src/train.py
     ```
     This trains models (Linear Regression, Decision Tree, Random Forest, XGBoost), logs results to MLflow, and saves the best model to `models/best_model.pkl`.

5. **Track Experiments with MLflow**:
   - Start the MLflow UI:
     ```bash
     mlflow ui
     ```
   - Access it at `http://localhost:5000` to view logged metrics (RMSE, MAE, R², RMSLE) and models.

6. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   - Access the web app at `http://localhost:8501`.
   - Use the "Predict Premium" tab to input customer details and get real-time predictions.
   - Other tabs display project details, dataset overview, approach, and results.

7. **Deploy the App** (optional):
   - Push the repo to GitHub.
   - Deploy to Streamlit Cloud:
     - Sign in to [Streamlit Cloud](https://share.streamlit.io/), connect your GitHub repo, and deploy `app.py`.
   - Alternatively, deploy on Heroku or AWS (refer to their documentation).

## Requirements
Install dependencies listed in `requirements.txt`:
```
pandas
numpy
scikit-learn
xgboost
mlflow
streamlit
joblib
matplotlib
seaborn
```

## Usage
1. **EDA and Model Training**:
    - Run `src/train.py` to train models and select the best one based on RMSE, MAE, R², and RMSLE.

2. **Real-Time Predictions**:
   - Launch the Streamlit app (`streamlit run app.py`).
   - Navigate to "Predict Premium" and enter customer details (e.g., Age, Income, Policy Type).
   - Click "Predict Premium" to see the predicted insurance premium.

3. **View Project Details**:
   - Use the app's tabs to explore the problem statement, dataset overview, approach, and evaluation metrics.

## Approach
1. **Understanding the Data**:
   - Load dataset and perform EDA (visualize distributions, correlations).
2. **Data Preprocessing**:
   - Handle missing values (median for numerical, mode for categorical).
   - Encode categorical features (one-hot for nominal, ordinal for education level).
   - Scale numerical features and parse dates.
3. **Model Development**:
   - Train regression models: Linear Regression, Decision Tree, Random Forest, XGBoost.
   - Evaluate using RMSE, MAE, R², and RMSLE.
4. **ML Pipeline & MLflow**:
   - Use Scikit-learn pipelines for preprocessing and modeling.
   - Log experiments with MLflow for tracking and comparison.
5. **Deployment**:
   - Deploy a Streamlit app for real-time predictions.

## Evaluation Metrics
- **Root Mean Squared Logarithmic Error (RMSLE)**: Measures error in log scale.
- **Root Mean Squared Error (RMSE)**: Quantifies prediction accuracy.
- **Mean Absolute Error (MAE)**: Average absolute prediction error.
- **R² Score**: Proportion of variance explained by the model.

## Results
- **Goal**: Achieve low error rates (RMSE, MAE, RMSLE) and high R².
- **Output**: A deployed Streamlit app providing real-time premium predictions.

## Technical Stack
- **Tools**: Python, Pandas, NumPy, Scikit-Learn, XGBoost, MLflow, Streamlit
- **Techniques**: Data Preprocessing, Feature Engineering, Model Deployment
- **Version Control**: Git/GitHub

## License
This project is licensed under the MIT License. The dataset is synthetic and provided for educational purposes.

## Contributing
Feel free to open issues or submit pull requests for improvements. For major changes, please discuss in an issue first.
