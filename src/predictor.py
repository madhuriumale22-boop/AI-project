"""
Disease Predictor Module
Contains the main prediction logic for the AI Disease Prediction System.
"""

import joblib
import numpy as np
import pandas as pd
import os

class DiseasePredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.symptoms = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained model and encoders."""
        try:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            
            # Load model
            model_path = os.path.join(models_dir, 'disease_model.pkl')
            self.model = joblib.load(model_path)
            
            # Load label encoder
            encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
            self.label_encoder = joblib.load(encoder_path)
            
            # Load symptoms list
            symptoms_path = os.path.join(models_dir, 'symptoms.pkl')
            self.symptoms = joblib.load(symptoms_path)
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_disease(self, selected_symptoms):
        """Predict disease based on selected symptoms."""
        if not self.model_loaded:
            if not self.load_model():
                return None, None, None
        
        # Create input vector
        input_vector = np.zeros(len(self.symptoms))
        for i, symptom in enumerate(self.symptoms):
            if symptom in selected_symptoms:
                input_vector[i] = 1
        
        # Make prediction
        prediction = self.model.predict([input_vector])[0]
        probabilities = self.model.predict_proba([input_vector])[0]
        
        # Get disease name and confidence
        disease = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        
        for idx in top_indices:
            disease_name = self.label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            top_predictions.append((disease_name, prob))
        
        return disease, confidence, top_predictions
    
    def get_symptoms_list(self):
        """Get the list of available symptoms."""
        if not self.model_loaded:
            if not self.load_model():
                return []
        return self.symptoms
    
    def get_disease_info(self, disease_name):
        """Get information about a specific disease."""
        disease_info = {
            'Common Cold': {
                'description': 'A viral infection of the upper respiratory tract.',
                'common_symptoms': ['runny_nose', 'sore_throat', 'cough', 'fatigue'],
                'recommendations': [
                    'Get plenty of rest',
                    'Stay hydrated',
                    'Use saline nasal drops',
                    'Consider over-the-counter pain relievers'
                ]
            },
            'Flu': {
                'description': 'A contagious respiratory illness caused by influenza viruses.',
                'common_symptoms': ['fever', 'body_ache', 'fatigue', 'headache'],
                'recommendations': [
                    'Rest and sleep',
                    'Drink plenty of fluids',
                    'Consider antiviral medications if caught early',
                    'Stay home to avoid spreading'
                ]
            },
            'Migraine': {
                'description': 'A type of headache characterized by severe throbbing pain.',
                'common_symptoms': ['headache', 'nausea', 'vomiting', 'dizziness'],
                'recommendations': [
                    'Rest in a quiet, dark room',
                    'Apply cold or warm compress',
                    'Stay hydrated',
                    'Consider prescribed migraine medications'
                ]
            },
            'Food Poisoning': {
                'description': 'Illness caused by eating contaminated food.',
                'common_symptoms': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain'],
                'recommendations': [
                    'Stay hydrated with clear fluids',
                    'Rest and avoid solid foods initially',
                    'Gradually return to bland foods',
                    'Seek medical attention if symptoms persist'
                ]
            },
            'Gastroenteritis': {
                'description': 'Inflammation of the stomach and intestines.',
                'common_symptoms': ['diarrhea', 'nausea', 'vomiting', 'abdominal_pain'],
                'recommendations': [
                    'Maintain fluid intake',
                    'Follow BRAT diet (Bananas, Rice, Applesauce, Toast)',
                    'Rest and avoid dairy products',
                    'Consult doctor if symptoms worsen'
                ]
            },
            'Pneumonia': {
                'description': 'Infection that inflames air sacs in one or both lungs.',
                'common_symptoms': ['cough', 'fever', 'shortness_of_breath', 'chest_pain'],
                'recommendations': [
                    'Seek immediate medical attention',
                    'Take prescribed antibiotics as directed',
                    'Get plenty of rest',
                    'Stay hydrated and use humidifier'
                ]
            },
            'Bronchitis': {
                'description': 'Inflammation of the lining of bronchial tubes.',
                'common_symptoms': ['cough', 'chest_pain', 'shortness_of_breath', 'fatigue'],
                'recommendations': [
                    'Rest and avoid irritants',
                    'Use humidifier or breathe steam',
                    'Stay hydrated',
                    'Consider cough suppressants'
                ]
            },
            'Allergic Reaction': {
                'description': 'Immune system response to a foreign substance.',
                'common_symptoms': ['skin_rash', 'runny_nose', 'cough', 'shortness_of_breath'],
                'recommendations': [
                    'Avoid known allergens',
                    'Take antihistamines',
                    'Use topical treatments for skin reactions',
                    'Seek emergency care for severe reactions'
                ]
            },
            'Muscle Strain': {
                'description': 'Injury to muscle or tendon from overuse or sudden movement.',
                'common_symptoms': ['body_ache', 'joint_pain', 'back_pain', 'fatigue'],
                'recommendations': [
                    'Rest and avoid activities that cause pain',
                    'Apply ice for first 24-48 hours',
                    'Use over-the-counter pain relievers',
                    'Gentle stretching and physical therapy'
                ]
            },
            'Hypertension': {
                'description': 'High blood pressure that can lead to serious health problems.',
                'common_symptoms': ['headache', 'dizziness', 'chest_pain', 'shortness_of_breath'],
                'recommendations': [
                    'Monitor blood pressure regularly',
                    'Follow prescribed medication regimen',
                    'Maintain healthy diet and exercise',
                    'Reduce sodium intake and manage stress'
                ]
            }
        }
        
        return disease_info.get(disease_name, {
            'description': 'Information not available for this condition.',
            'common_symptoms': [],
            'recommendations': ['Consult with a healthcare professional for proper diagnosis and treatment.']
        })