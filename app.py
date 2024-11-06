# app.py
from src.data_processing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model
import joblib

# Load and preprocess data
data = load_data('data/machine_data.csv')
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
accuracy = evaluate_model(model, X_test, y_test)
print(f'Model accuracy: {accuracy:.2f}')

# Save model
joblib.dump(model, 'src/model.pkl')
print('Model saved as src/model.pkl')
