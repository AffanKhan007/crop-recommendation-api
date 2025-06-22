from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load your actual dataset (rename if needed)
df = pd.read_csv("smart-farming-using-machine-learning.csv")

# Drop district column if not needed
df = df.drop('district', axis=1)

# Prepare training data
X = df.drop("label", axis=1)
y = df["label"]

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Start Flask app
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

# Run on Replit
app.run(host='0.0.0.0', port=3000)
