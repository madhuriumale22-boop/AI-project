"""
Quick Model Training Script - No Plotting
Fast training for immediate testing of the web application.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

def quick_train():
    """Quick training without plots."""
    print("Quick Training - AI Disease Prediction Model")
    print("=" * 50)
    
    # Load data
    data_path = os.path.join('data', 'disease_dataset.csv')
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} records")
    
    # Prepare data
    symptoms = [col for col in df.columns if col != 'disease']
    X = df[symptoms]
    y = df['disease']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Features: {len(symptoms)} symptoms")
    print(f"Classes: {len(label_encoder.classes_)} diseases")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save model
    print("Saving model files...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/disease_model.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(symptoms, 'models/symptoms.pkl')
    
    print("✅ Model training completed!")
    print("📁 Model files saved in 'models/' directory")
    print("🚀 Ready to run the web application!")
    
    # Test prediction
    print("\nTesting prediction...")
    test_symptoms = ['fever', 'cough', 'fatigue', 'body_ache']
    input_vector = np.zeros(len(symptoms))
    for i, symptom in enumerate(symptoms):
        if symptom in test_symptoms:
            input_vector[i] = 1
    
    prediction = model.predict([input_vector])[0]
    probability = model.predict_proba([input_vector])[0]
    disease = label_encoder.inverse_transform([prediction])[0]
    confidence = probability[prediction]
    
    print(f"Test symptoms: {test_symptoms}")
    print(f"Predicted disease: {disease}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    quick_train()