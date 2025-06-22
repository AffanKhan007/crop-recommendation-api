from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load your model
model = joblib.load("crop_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[ 
        data['N'], data['P'], data['K'],
        data['temperature'], data['humidity'],
        data['ph'], data['rainfall']
    ]])
    prediction = model.predict(features)
    return jsonify({'recommended_crop': prediction[0]})

app.run(host='0.0.0.0', port=3000)
