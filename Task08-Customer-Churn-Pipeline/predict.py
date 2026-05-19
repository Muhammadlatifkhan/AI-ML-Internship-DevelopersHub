"""
Load and use the trained pipeline for predictions
"""

import joblib
import pandas as pd
import numpy as np

def load_pipeline():
    """Load the trained pipeline"""
    return joblib.load('outputs/churn_pipeline.pkl')

def predict_churn(pipeline, customer_data):
    """
    Predict churn for a single customer or batch
    
    Args:
        pipeline: Loaded pipeline
        customer_data: Dict or DataFrame with customer information
    
    Returns:
        Prediction (0=No Churn, 1=Churn) and probability
    """
    # Convert dict to DataFrame if needed
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    
    # Make prediction
    prediction = pipeline.predict(customer_data)[0]
    probability = pipeline.predict_proba(customer_data)[0][1]
    
    return prediction, probability

def predict_batch(pipeline, data_file):
    """Predict for multiple customers from CSV"""
    df = pd.read_csv(data_file)
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]
    
    df['Churn_Prediction'] = predictions
    df['Churn_Probability'] = probabilities
    df['Churn_Risk'] = df['Churn_Probability'].apply(
        lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.3 else 'Low')
    )
    
    return df

if __name__ == "__main__":
    print("Loading pipeline...")
    pipeline = load_pipeline()
    print("✅ Pipeline loaded successfully!")
    
    # Example: Predict for a sample customer
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': '0',
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 24,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.5,
        'TotalCharges': 1500.0
    }
    
    prediction, probability = predict_churn(pipeline, sample_customer)
    
    print("\n📊 Sample Prediction:")
    print(f"   Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
    print(f"   Churn Probability: {probability:.2%}")