# Crop Recommendation API

## Overview
A machine learning-powered Flask API that recommends optimal crops based on soil and environmental parameters. The system uses a Random Forest Classifier trained on district-level agricultural data.

## Recent Changes
- **Dec 01, 2025**: Initial project setup with Flask API and ML model

## Project Architecture
- **main.py**: Flask web server exposing the crop recommendation API
- **crop_recommendation_training.py**: ML model training script (reference)
- **crop_model.pkl**: Pre-trained Random Forest model
- **Crop(Distric level).csv**: Training dataset with soil and environmental features

## API Endpoints
- `GET /`: Health check endpoint
- `POST /predict`: Accepts soil/environmental parameters and returns recommended crop

### Input Parameters
- N: Nitrogen content
- P: Phosphorus content
- K: Potassium content
- temperature: Temperature in Celsius
- humidity: Humidity percentage
- ph: Soil pH level
- rainfall: Rainfall in mm

## Tech Stack
- Flask 3.1.1
- scikit-learn 1.7.0
- pandas 2.3.0
- numpy 2.3.1
- joblib 1.5.1

## User Preferences
None documented yet
