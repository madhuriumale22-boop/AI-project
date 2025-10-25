"""
Simple Streamlit App for Disease Prediction
A minimal version to test the system functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Use the appropriate caching decorator based on Streamlit version
try:
    # For Streamlit >= 1.18.0
    cache_decorator = st.cache_resource
except AttributeError:
    # For older Streamlit versions
    cache_decorator = st.cache

@cache_decorator
def load_model():
    """Load the trained model and related files."""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')
        
        model = joblib.load(os.path.join(models_dir, 'disease_model.pkl'))
        label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        symptoms = joblib.load(os.path.join(models_dir, 'symptoms.pkl'))
        return model, label_encoder, symptoms
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def main():
    st.title("🏥 AI Disease Prediction System")
    st.markdown("---")
    
    # Load model
    model, label_encoder, symptoms = load_model()
    
    if model is None:
        st.error("❌ Model not found. Please train the model first by running: `python quick_train.py`")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for symptoms selection
    st.sidebar.header("Select Your Symptoms")
    st.sidebar.markdown("Check the symptoms you are experiencing:")
    
    selected_symptoms = []
    for symptom in symptoms:
        if st.sidebar.checkbox(symptom.replace('_', ' ').title(), key=symptom):
            selected_symptoms.append(symptom)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Disease Prediction")
        
        if selected_symptoms:
            st.write("**Selected Symptoms:**")
            for symptom in selected_symptoms:
                st.write(f"• {symptom.replace('_', ' ').title()}")
            
            # Predict button
            if st.button("🔍 Predict Disease"):
                # Create input vector
                input_vector = np.zeros(len(symptoms))
                for i, symptom in enumerate(symptoms):
                    if symptom in selected_symptoms:
                        input_vector[i] = 1
                
                # Make prediction
                prediction = model.predict([input_vector])[0]
                probabilities = model.predict_proba([input_vector])[0]
                
                # Get top 3 predictions
                top_indices = np.argsort(probabilities)[::-1][:3]
                
                st.markdown("### 🎯 Prediction Results")
                
                for i, idx in enumerate(top_indices):
                    disease = label_encoder.inverse_transform([idx])[0]
                    confidence = probabilities[idx]
                    
                    if i == 0:
                        st.success(f"**Most Likely: {disease}** (Confidence: {confidence:.2%})")
                    else:
                        st.info(f"**Alternative {i}: {disease}** (Confidence: {confidence:.2%})")
                
                # Disclaimer
                st.warning("⚠️ **Disclaimer:** This is an AI prediction for educational purposes only. Please consult a healthcare professional for proper medical diagnosis.")
        
        else:
            st.info("👈 Please select symptoms from the sidebar to get a prediction.")
    
    with col2:
        st.header("System Info")
        st.write(f"**Available Symptoms:** {len(symptoms)}")
        st.write(f"**Diseases in Database:** {len(label_encoder.classes_)}")
        
        if selected_symptoms:
            st.write(f"**Selected Symptoms:** {len(selected_symptoms)}")
        
        st.markdown("### 📊 Available Diseases")
        for disease in label_encoder.classes_:
            st.write(f"• {disease}")

if __name__ == "__main__":
    main()