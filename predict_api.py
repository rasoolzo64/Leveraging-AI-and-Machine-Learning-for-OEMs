# src/predict_api.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('src/model.pkl')  # Load pre-trained model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)[0]
    return jsonify({'maintenance_required': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
