
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
try:
    model = joblib.load('crop_model.pkl')
    print("✅ Pre-trained model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
except Exception as e:
    print(f"❌ Error loading pkl model: {e}")
    # Fallback to training new model
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv("Crop(Distric level).csv")
    df = df.drop('district', axis=1)
    X = df.drop("label", axis=1)
    y = df["label"]
    model = RandomForestClassifier()
    model.fit(X, y)
    print("✅ Fallback: New model trained")

# Start Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return {"message": "Crop Recommendation API is running. Use POST /predict to get recommendations."}

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

# Run on Replit
app.run(host='0.0.0.0', port=3000)
