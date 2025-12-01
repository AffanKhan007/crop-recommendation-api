
"""
Crop Recommendation Model Training Script
This file serves as a reference for the training process used to create the crop_model.pkl
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Crop(Distric level).csv')

# Display basic information
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset info:")
print(df.info())

print(f"\nCrop distribution:")
print(df['label'].value_counts())

# Drop district column
df = df.drop('district', axis=1)

# Prepare features and target
print("\nPreparing features and target...")
X = df.drop('label', axis=1)
y = df['label']

print(f"Features: {list(X.columns)}")
print(f"Number of unique crops: {y.nunique()}")

# Split the data into training and testing sets (80-20 split)
print("\nSplitting data into train and test sets (80-20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train the Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Training completed!")

# Make predictions
print("\nMaking predictions on test set...")
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# Confusion Matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
plt.figure(figsize=(10, 8))
disp.plot(cmap='viridis', xticks_rotation=90)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Feature Importance Plot
print("\nGenerating feature importance plot...")
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance['importance'], y=feature_importance['feature'])
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

# Save the trained model
print("\nSaving model to 'crop_model.pkl'...")
joblib.dump(rf, 'crop_model.pkl')
print("Model saved successfully!")

print("\n" + "="*50)
print("Training process completed!")
print("="*50)

# Example predictions
print("\n" + "="*50)
print("Example Predictions:")
print("="*50)

sample = [[90, 42, 43, 20.8, 82.0, 6.5, 202.9]]
predicted_crop = rf.predict(sample)
print(f"\n🌾 Sample 1 - Recommended Crop: {predicted_crop[0]}")

samples = [
    [70, 40, 35, 32.0, 45.0, 7.5, 90.0],
    [50, 30, 50, 18.0, 90.0, 5.5, 250.0],
    [120, 100, 90, 25.0, 70.0, 6.4, 180.0],
    [40, 20, 30, 30.0, 30.0, 7.2, 50.0]
]

for i, s in enumerate(samples, 2):
    sample = [s]
    prediction = rf.predict(sample)
    print(f"🌿 Sample {i} - Predicted Crop: {prediction[0]}")
