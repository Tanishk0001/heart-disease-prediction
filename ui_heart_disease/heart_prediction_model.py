import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load both the trained model and the scaler (ensure the scaler was saved)
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler

def predict_heart_disease(data):
    """
    Function to predict heart disease based on input features.
    :param data: A numpy array with input features (age, chest pain, thalach)
    :return: 1 (Heart Disease Detected) or 0 (No Heart Disease)
    """
    # Scale the input data using the same scaler that was used for training
    data_scaled = scaler.transform(data)
    
    # Make prediction
    prediction = model.predict(data_scaled)
    
    return prediction[0]
