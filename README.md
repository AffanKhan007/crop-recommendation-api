🌾 Crop Recommendation API

A simple Flask API that recommends the best crop based on soil nutrients and climate conditions using a trained Random Forest machine learning model.

🚀 Features

Takes soil and climate values as input

Uses a machine learning model to predict the best crop

Supports 15+ crop types

Easy to run locally

Simple POST API endpoint

🛠️ Technologies Used

Python

Flask

scikit-learn

pandas

numpy

joblib

📁 Project Files
main.py              # Flask API code
crop_model.pkl       # Trained ML model
Crop(District).csv   # Training dataset
pyproject.toml       # Dependencies
README.md            # Documentation

🔧 How to Run the Project

Install dependencies

pip install flask joblib numpy pandas scikit-learn


Run the server

python main.py


API runs at:

http://localhost:5000

📡 API Usage
POST /predict

Example Request Body:

{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82,
  "ph": 6.5,
  "rainfall": 202
}


Example Response:

{
  "recommended_crop": "rice"
}

🤖 Model Details

Algorithm: Random Forest

Features used: N, P, K, temperature, humidity, pH, rainfall

Model file: crop_model.pkl

👤 Author

Affan Khan
