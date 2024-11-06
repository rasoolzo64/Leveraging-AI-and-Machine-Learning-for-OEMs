# src/data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(data):
    # Separate features and target
    X = data[['temperature', 'pressure', 'vibration']]
    y = data['maintenance_required']
    # Split data into training and test sets
    return train_test_split(X, y, test_size=0.2, random_state=42)
